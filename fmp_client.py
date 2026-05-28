"""
fmp_client.py
=============
FMP (Financial Modeling Prep) API 클라이언트
- 새 stable endpoint 기반
- 호출 한도 추적
- 캐싱 지원
"""

import os
import pickle
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()
API_KEY = os.getenv("FMP_API_KEY", "")
BASE_URL = "https://financialmodelingprep.com/stable"

# 캐시 디렉토리 — FRIDAY 격리 (quant_tool과 분리)
CACHE_DIR = Path.home() / ".friday_cache" / "fmp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 호출 카운터 파일
CALL_COUNTER_FILE = CACHE_DIR / "call_counter.json"

# 분당 호출 제한 (Starter: 300/min, Free: 정확히 안나와있어 보수적으로 60/min)
RATE_LIMIT_PER_MIN = 250  # Starter 안전 마진 (300 - 50)

# 분당 호출 추적용
_recent_calls = []  # 최근 60초 호출 timestamps

# ============================================================
# 호출 카운터 (일일 한도 추적)
# ============================================================

def _load_counter() -> Dict:
    """오늘의 호출 카운터 로드"""
    import json
    today = datetime.now().strftime("%Y-%m-%d")
    if CALL_COUNTER_FILE.exists():
        try:
            with open(CALL_COUNTER_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "count": 0}


def _save_counter(counter: Dict) -> None:
    import json
    with open(CALL_COUNTER_FILE, "w") as f:
        json.dump(counter, f)


def get_call_count() -> int:
    """오늘 호출 횟수 반환"""
    return _load_counter()["count"]


def increment_counter() -> int:
    """호출 카운터 증가"""
    counter = _load_counter()
    counter["count"] += 1
    _save_counter(counter)
    return counter["count"]


# ============================================================
# 캐시
# ============================================================

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _is_cache_fresh(name: str, max_age_hours: int = 24) -> bool:
    path = _cache_path(name)
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def _save_cache(name: str, data: Any) -> None:
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def _load_cache(name: str) -> Any:
    with open(_cache_path(name), "rb") as f:
        return pickle.load(f)


# ============================================================
# 공통 호출 함수
# ============================================================

def _call_api(
    endpoint: str,
    params: Optional[Dict] = None,
    use_cache: bool = True,
    cache_hours: int = 24,
) -> Optional[Any]:
    """
    FMP API 호출 (캐시 + 카운터 통합)

    Args:
        endpoint: e.g. "quote", "ratios-ttm"
        params: 쿼리 파라미터 (예: {"symbol": "AAPL"})
        use_cache: 캐시 사용 여부
        cache_hours: 캐시 유효 시간

    Returns:
        JSON 응답 (list 또는 dict) 또는 None
    """
    if not API_KEY:
        logger.error("FMP_API_KEY not set in .env")
        return None

    params = params or {}

    # 캐시 키 생성
    param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    cache_name = f"{endpoint.replace('/', '_')}__{param_str}"

    # 캐시 체크
    if use_cache and _is_cache_fresh(cache_name, max_age_hours=cache_hours):
        return _load_cache(cache_name)

# Rate limit 체크 (분당 300 → 안전하게 250)
    import time as _time
    global _recent_calls
    now = _time.time()
    _recent_calls = [t for t in _recent_calls if now - t < 60]  # 60초 이내만 유지
    if len(_recent_calls) >= RATE_LIMIT_PER_MIN:
        sleep_time = 60 - (now - _recent_calls[0]) + 0.5
        logger.info(f"  Rate limit reached, sleeping {sleep_time:.1f}s...")
        _time.sleep(sleep_time)
        _recent_calls = []
    _recent_calls.append(now)

    # API 호출
    url = f"{BASE_URL}/{endpoint}"
    full_params = {**params, "apikey": API_KEY}
    
    try:
        r = requests.get(url, params=full_params, timeout=15)
        increment_counter()
        
        if r.status_code == 200:
            data = r.json()
            # 빈 응답이나 에러 메시지는 캐시 안 함 (장애 시 캐시 오염 방지)
            if isinstance(data, list) and len(data) == 0:
                logger.debug(f"{endpoint} returned empty list for {params}")
                return data  # 캐시 안 함
            if isinstance(data, dict) and "Error Message" in data:
                logger.warning(f"{endpoint} returned error: {data.get('Error Message', '')[:150]}")
                return None  # 캐시 안 함
            _save_cache(cache_name, data)
            return data
        elif r.status_code == 402:
            logger.warning(f"API {endpoint} returned 402 (Premium). Skip caching.")
            return None  # 캐시 안 함 ← 핵심!
        elif r.status_code == 429:
            logger.error(f"Rate limit hit! Today's calls: {get_call_count()}")
            return None
        else:
            logger.warning(f"API {endpoint} returned {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"API call failed for {endpoint}: {e}")
        return None


# ============================================================
# 데이터 함수들 (각 endpoint별)
# ============================================================

def get_analyst_estimates(symbol: str) -> Optional[Dict]:
    """
    애널리스트 EPS/Revenue Estimates
    - 핵심: 최근 EPS 추정 변화로 EPS Revisions 계산
    """
    data = _call_api(
        "analyst-estimates",
        params={"symbol": symbol, "period": "annual", "page": 0, "limit": 5},
        cache_hours=24,
    )
    if not data or not isinstance(data, list) or len(data) < 2:
        return None

    # 최근 vs 직전 추정 비교 (EPS Revision proxy)
    latest = data[0]
    return {
        "symbol": symbol,
        "latest_date": latest.get("date"),
        "eps_avg": latest.get("epsAvg"),
        "eps_high": latest.get("epsHigh"),
        "eps_low": latest.get("epsLow"),
        "revenue_avg": latest.get("revenueAvg"),
        "num_analysts_eps": latest.get("numAnalystsEps"),
        "num_analysts_revenue": latest.get("numAnalystsRevenue"),
        # 모든 데이터 저장 (수익률 변화 계산용)
        "all_estimates": data[:5],
    }


def get_financial_scores(symbol: str) -> Optional[Dict]:
    """
    Piotroski Score (9점 만점) + Altman Z-Score
    """
    data = _call_api("financial-scores", params={"symbol": symbol})
    if not data or not isinstance(data, list) or not data:
        return None
    s = data[0]
    return {
        "symbol": symbol,
        "piotroski": s.get("piotroskiScore"),
        "altman_z": s.get("altmanZScore"),
        "working_capital": s.get("workingCapital"),
        "ebit": s.get("ebit"),
        "market_cap_fmp": s.get("marketCap"),
    }


def get_ratios_ttm(symbol: str) -> Optional[Dict]:
    """재무비율 TTM (60개 지표 중 핵심 추출)"""
    data = _call_api("ratios-ttm", params={"symbol": symbol})
    if not data or not isinstance(data, list) or not data:
        return None
    r = data[0]
    return {
        "symbol": symbol,
        "gross_margin_ttm": r.get("grossProfitMarginTTM"),
        "operating_margin_ttm": r.get("operatingProfitMarginTTM"),
        "net_margin_ttm": r.get("netProfitMarginTTM"),
        "ebitda_margin_ttm": r.get("ebitdaMarginTTM"),
        "current_ratio_ttm": r.get("currentRatioTTM"),
        "quick_ratio_ttm": r.get("quickRatioTTM"),
        "debt_to_equity_ttm": r.get("debtToEquityRatioTTM"),
        "debt_to_assets_ttm": r.get("debtToAssetsRatioTTM"),
        "interest_coverage_ttm": r.get("interestCoverageRatioTTM"),
        "dividend_yield_ttm": r.get("dividendYieldTTM"),
        "dividend_payout_ttm": r.get("dividendPayoutRatioTTM"),
        "pe_ttm": r.get("priceToEarningsRatioTTM"),
        "pb_ttm": r.get("priceToBookRatioTTM"),
        "ps_ttm": r.get("priceToSalesRatioTTM"),
        "p_fcf_ttm": r.get("priceToFreeCashFlowRatioTTM"),
        "p_ocf_ttm": r.get("priceToOperatingCashFlowRatioTTM"),
        "ev_multiple_ttm": r.get("enterpriseValueMultipleTTM"),
        "asset_turnover_ttm": r.get("assetTurnoverTTM"),
        "fcf_to_ocf_ttm": r.get("freeCashFlowOperatingCashFlowRatioTTM"),
        "fcf_per_share_ttm": r.get("freeCashFlowPerShareTTM"),
    }

def get_key_metrics_ttm(symbol: str) -> Optional[Dict]:
    """Key Metrics TTM — ROE, ROA, ROIC, FCF Yield 등 핵심"""
    data = _call_api("key-metrics-ttm", params={"symbol": symbol})
    if not data or not isinstance(data, list) or not data:
        return None
    m = data[0]
    return {
        "symbol": symbol,
        "market_cap_fmp": m.get("marketCap"),
        "ev_ttm": m.get("enterpriseValueTTM"),
        "ev_to_sales_ttm": m.get("evToSalesTTM"),
        "ev_to_ebitda_ttm": m.get("evToEBITDATTM"),
        "ev_to_fcf_ttm": m.get("evToFreeCashFlowTTM"),
        "ev_to_ocf_ttm": m.get("evToOperatingCashFlowTTM"),
        "net_debt_to_ebitda_ttm": m.get("netDebtToEBITDATTM"),
        "roa_ttm": m.get("returnOnAssetsTTM"),
        "roe_ttm": m.get("returnOnEquityTTM"),
        "roic_ttm": m.get("returnOnInvestedCapitalTTM"),
        "roce_ttm": m.get("returnOnCapitalEmployedTTM"),
        "earnings_yield_ttm": m.get("earningsYieldTTM"),
        "fcf_yield_ttm": m.get("freeCashFlowYieldTTM"),
        "income_quality_ttm": m.get("incomeQualityTTM"),
        "graham_number": m.get("grahamNumberTTM"),
        "working_capital_ttm": m.get("workingCapitalTTM"),
        "rd_to_revenue_ttm": m.get("researchAndDevelopementToRevenueTTM"),
        "fcf_to_equity_ttm": m.get("freeCashFlowToEquityTTM"),
    }

def get_analyst_consensus(symbol: str) -> Optional[Dict]:
    """
    애널리스트 매수/매도 의견 컨센서스
    """
    data = _call_api("grades-consensus", params={"symbol": symbol})
    if not data or not isinstance(data, list) or not data:
        return None
    g = data[0]
    strong_buy = g.get("strongBuy", 0) or 0
    buy = g.get("buy", 0) or 0
    hold = g.get("hold", 0) or 0
    sell = g.get("sell", 0) or 0
    strong_sell = g.get("strongSell", 0) or 0
    total = strong_buy + buy + hold + sell + strong_sell
    
    # 0~100 점수 (Strong Buy=100, Strong Sell=0)
    if total > 0:
        score = (strong_buy * 100 + buy * 75 + hold * 50 + sell * 25 + strong_sell * 0) / total
    else:
        score = 50
    
    return {
        "symbol": symbol,
        "strong_buy": strong_buy,
        "buy": buy,
        "hold": hold,
        "sell": sell,
        "strong_sell": strong_sell,
        "total_analysts": total,
        "consensus": g.get("consensus"),
        "analyst_score": score,  # 0~100
    }


def get_price_target(symbol: str) -> Optional[Dict]:
    """
    애널리스트 목표가 컨센서스
    """
    data = _call_api("price-target-consensus", params={"symbol": symbol})
    if not data or not isinstance(data, list) or not data:
        return None
    t = data[0]
    return {
        "symbol": symbol,
        "target_high": t.get("targetHigh"),
        "target_low": t.get("targetLow"),
        "target_median": t.get("targetMedian"),
        "target_consensus": t.get("targetConsensus"),
    }


# ============================================================
# 통합 함수: 한 종목의 모든 FMP 데이터 가져오기
# ============================================================

def enrich_symbol(symbol: str) -> Dict:
    """
    한 종목의 모든 FMP 데이터를 통합해 반환
    호출 수: 약 6 calls/symbol
    """
    result = {"symbol": symbol}

    # 1. 애널리스트 추정 (EPS Revisions 강화용)
    est = get_analyst_estimates(symbol)
    if est:
        result.update({
            "fmp_eps_avg": est.get("eps_avg"),
            "fmp_revenue_avg": est.get("revenue_avg"),
            "fmp_num_analysts_eps": est.get("num_analysts_eps"),
        })

    # 2. Financial Scores (Piotroski, Altman) ⭐
    scores = get_financial_scores(symbol)
    if scores:
        result.update({
            "piotroski": scores.get("piotroski"),
            "altman_z": scores.get("altman_z"),
        })

    # 3. Ratios TTM
    ratios = get_ratios_ttm(symbol)
    if ratios:
        result.update({
            "fmp_gross_margin": ratios.get("gross_margin_ttm"),
            "fmp_operating_margin": ratios.get("operating_margin_ttm"),
            "fmp_net_margin": ratios.get("net_margin_ttm"),
            "fmp_ebitda_margin": ratios.get("ebitda_margin_ttm"),
            "fmp_current_ratio": ratios.get("current_ratio_ttm"),
            "fmp_debt_to_equity": ratios.get("debt_to_equity_ttm"),
            "fmp_debt_to_assets": ratios.get("debt_to_assets_ttm"),
            "fmp_interest_coverage": ratios.get("interest_coverage_ttm"),
            "fmp_dividend_yield": ratios.get("dividend_yield_ttm"),
            "fmp_pe": ratios.get("pe_ttm"),
            "fmp_pb": ratios.get("pb_ttm"),
            "fmp_ps": ratios.get("ps_ttm"),
            "fmp_p_fcf": ratios.get("p_fcf_ttm"),
            "fmp_fcf_to_ocf": ratios.get("fcf_to_ocf_ttm"),
        })

    # 4. Key Metrics (ROE, ROA, ROIC, EV ratios)
    metrics = get_key_metrics_ttm(symbol)
    if metrics:
        result.update({
            "fmp_market_cap": metrics.get("market_cap_fmp"),
            "fmp_ev": metrics.get("ev_ttm"),
            "fmp_ev_ebitda": metrics.get("ev_to_ebitda_ttm"),
            "fmp_ev_sales": metrics.get("ev_to_sales_ttm"),
            "fmp_ev_fcf": metrics.get("ev_to_fcf_ttm"),
            "fmp_net_debt_ebitda": metrics.get("net_debt_to_ebitda_ttm"),
            "fmp_roa": metrics.get("roa_ttm"),
            "fmp_roe": metrics.get("roe_ttm"),
            "fmp_roic": metrics.get("roic_ttm"),
            "fmp_roce": metrics.get("roce_ttm"),
            "fmp_earnings_yield": metrics.get("earnings_yield_ttm"),
            "fmp_fcf_yield": metrics.get("fcf_yield_ttm"),
            "fmp_income_quality": metrics.get("income_quality_ttm"),
            "fmp_rd_revenue": metrics.get("rd_to_revenue_ttm"),
        })

    # 5. 애널리스트 컨센서스
    cons = get_analyst_consensus(symbol)
    if cons:
        total = max(cons.get("total_analysts", 1), 1)
        result.update({
            "analyst_consensus": cons.get("consensus"),
            "analyst_score": cons.get("analyst_score"),
            "num_analyst_ratings": cons.get("total_analysts"),
            "buy_pct": (cons.get("strong_buy", 0) + cons.get("buy", 0)) / total * 100,
        })

    # 6. 목표가
    target = get_price_target(symbol)
    if target and target.get("target_consensus"):
        result["target_consensus"] = target.get("target_consensus")
        result["target_high"] = target.get("target_high")
        result["target_low"] = target.get("target_low")

    return result


def enrich_batch(symbols: List[str], delay_sec: float = 0.05) -> pd.DataFrame:
    """
    여러 종목을 batch로 FMP 데이터 가져오기
    """
    rows = []
    total = len(symbols)
    start_count = get_call_count()
    
    logger.info(f"FMP enrichment starting: {total} symbols, current calls today: {start_count}")
    
    for i, sym in enumerate(symbols):
        if i % 10 == 0:
            current = get_call_count()
            logger.info(f"  Progress: {i}/{total} | Calls used: {current - start_count}")
        
        try:
            data = enrich_symbol(sym)
            rows.append(data)
        except Exception as e:
            logger.warning(f"  Failed for {sym}: {e}")
            rows.append({"symbol": sym, "error": str(e)})
        
        time.sleep(delay_sec)  # rate limit 방지
    
    end_count = get_call_count()
    logger.info(f"FMP enrichment complete: {total} symbols, used {end_count - start_count} calls")
    
    return pd.DataFrame(rows)


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    print(f"Today's API calls so far: {get_call_count()}")
    print("\nEnriching AAPL...")
    data = enrich_symbol("NEM")
    for k, v in data.items():
        print(f"  {k}: {v}")
    print(f"\nTotal calls today: {get_call_count()}")