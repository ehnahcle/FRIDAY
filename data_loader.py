"""
data_loader.py
==============
FRIDAY 데이터 수집 모듈 — R1000 live universe (FMP company-screener)
+ yfinance 가격/펀더멘털 + FRED 거시 (VIX/SPY MA200)

quant_tool/data_loader.py에서 fork (2026-05-28). 차이:
  - 캐시 디렉토리 ~/.friday_cache/ 격리
  - get_sp500_tickers / get_extended_universe 제거
  - get_r1000_live_universe 신규 (FMP company-screener top-1000 by mcap)
  - 나머지 (가격/펀더/거시/레짐) 그대로

R1000 universe 정의: "현재 시점 top-1000 US non-ETF/non-Fund by market cap".
Phase 24 백테스트의 synthetic R1000 PIT 정의와 일치 (단 백테스트는 분기 historical,
live는 현재 snapshot). 학계 공식 R1000과 micro-편차 가능 (외국 ADR, REIT 일부 포함).
"""

from __future__ import annotations

import os
import time
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"

CACHE_DIR = Path.home() / ".friday_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# 캐시 헬퍼
# ============================================================

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _is_cache_fresh(name: str, max_age_hours: int = 12) -> bool:
    path = _cache_path(name)
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "wb") as f:
        pickle.dump(data, f)


def _load_cache(name: str):
    with open(_cache_path(name), "rb") as f:
        return pickle.load(f)


# ============================================================
# R1000 Live Universe (FMP company-screener)
# ============================================================

def get_r1000_live_universe(
    top_n: int = 1000,
    mcap_min: float = 300_000_000,
    cache_hours: int = 24,
) -> List[str]:
    """현재 시점 top-N US non-ETF/non-Fund by market cap.

    FMP /stable/company-screener 사용. mcap desc 정렬 후 top_n 추출.
    24h 캐시 (분기 rebal 주기엔 충분, 일중 변동 무시).

    Phase 24 backtest 의 synthetic R1000 PIT 정의와 일치:
      - country: US
      - isEtf: false, isFund: false
      - marketCapMoreThan: $300M (micro-cap 컷)
      - top-1000 by mcap

    Returns
    -------
    List[str] — 티커 (e.g., ['NVDA', 'MSFT', 'AAPL', ...]). 빈 리스트 시 API 실패.
    """
    cache_name = f"r1000_live_top{top_n}"
    if _is_cache_fresh(cache_name, max_age_hours=cache_hours):
        cached = _load_cache(cache_name)
        logger.info(f"R1000 universe loaded from cache: {len(cached)} tickers")
        return cached

    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not set in .env")
        return []

    try:
        # cushion: top_n + 500 (시총 컷 근처 turnover 흡수)
        limit = top_n + 500
        r = requests.get(
            f"{FMP_BASE}/company-screener",
            params={
                "marketCapMoreThan": mcap_min,
                "isEtf": "false",
                "isFund": "false",
                "country": "US",
                "limit": limit,
                "apikey": FMP_API_KEY,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"FMP company-screener failed: {e}")
        return []

    if not isinstance(data, list) or len(data) == 0:
        logger.error(f"FMP company-screener returned empty/invalid: {data!r}")
        return []

    # mcap desc 정렬 (FMP가 이미 정렬해주지만 safety)
    df = pd.DataFrame(data)
    if "marketCap" not in df.columns:
        logger.error(f"FMP response missing marketCap: {df.columns.tolist()}")
        return []

    df = df.dropna(subset=["marketCap", "symbol"])
    df = df.sort_values("marketCap", ascending=False).reset_index(drop=True)

    # yfinance용 ticker 정규화 (. → -; e.g., BRK.B → BRK-B)
    tickers = (
        df["symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    )
    tickers = [t for t in tickers if t and t != "nan"][:top_n]

    _save_cache(cache_name, tickers)
    logger.info(
        f"R1000 live universe fetched: {len(tickers)} tickers "
        f"(mcap range ${df['marketCap'].iloc[0]/1e9:.0f}B ~ ${df['marketCap'].iloc[top_n-1]/1e9:.2f}B)"
    )
    return tickers


# ============================================================
# 가격 데이터
# ============================================================

def get_price_history(
    tickers: List[str],
    period: str = "2y",
    batch_size: int = 50,
) -> pd.DataFrame:
    """배치로 가격 데이터 다운로드 (yfinance)."""
    cache_name = f"prices_{period}"
    if _is_cache_fresh(cache_name, max_age_hours=12):
        cached = _load_cache(cache_name)
        if isinstance(cached.columns, pd.MultiIndex):
            cached_tickers = set(cached.columns.get_level_values(1))
        else:
            cached_tickers = set(cached.columns)
        if set(tickers).issubset(cached_tickers):
            return cached

    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        if i % (batch_size * 4) == 0:
            logger.info(f"Price fetch progress: {i}/{len(tickers)}")
        try:
            data = yf.download(
                batch,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=True,
            )
            all_data.append(data)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Batch {i} failed: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, axis=1)
    _save_cache(cache_name, combined)
    return combined


def get_spy_price() -> pd.DataFrame:
    """SPY 가격 (시장 벤치마크 + 200일선 계산)."""
    cache_name = "spy_price"
    if _is_cache_fresh(cache_name, max_age_hours=12):
        return _load_cache(cache_name)

    try:
        spy = yf.download("SPY", period="2y", progress=False, auto_adjust=True)
        _save_cache(cache_name, spy)
        return spy
    except Exception as e:
        logger.error(f"SPY load failed: {e}")
        return pd.DataFrame()


# ============================================================
# ADV20 헬퍼 ($5M gate 용)
# ============================================================

def compute_adv20(prices: pd.DataFrame) -> pd.Series:
    """직전 20거래일 평균 일거래대금 (Close × Volume).

    Returns
    -------
    pd.Series indexed by ticker — ADV20 ($).
    """
    if prices.empty:
        return pd.Series(dtype=float)
    if not isinstance(prices.columns, pd.MultiIndex):
        return pd.Series(dtype=float)

    try:
        close = prices["Close"]
        volume = prices["Volume"]
    except KeyError:
        return pd.Series(dtype=float)

    dollar_vol = (close * volume).tail(20).mean()
    return dollar_vol


# ============================================================
# 펀더멘털 데이터 (yfinance — quant_tool/data_loader.py 와 동일)
# ============================================================

def get_fundamentals(ticker: str) -> Dict:
    """개별 종목의 재무 지표 추출 (yfinance .info)."""
    try:
        stk = yf.Ticker(ticker)
        info = stk.info or {}

        out = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", np.nan),
            "shares_out": info.get("sharesOutstanding", np.nan),
            "price": info.get("currentPrice", info.get("regularMarketPrice", np.nan)),
            "pe": info.get("trailingPE", np.nan),
            "pe_forward": info.get("forwardPE", np.nan),
            "pb": info.get("priceToBook", np.nan),
            "ps": info.get("priceToSalesTrailing12Months", np.nan),
            "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
            "ev_revenue": info.get("enterpriseToRevenue", np.nan),
            "fcf": info.get("freeCashflow", np.nan),
            "ebitda": info.get("ebitda", np.nan),
            "roe": info.get("returnOnEquity", np.nan),
            "roa": info.get("returnOnAssets", np.nan),
            "gross_margins": info.get("grossMargins", np.nan),
            "operating_margins": info.get("operatingMargins", np.nan),
            "profit_margins": info.get("profitMargins", np.nan),
            "debt_to_equity": info.get("debtToEquity", np.nan),
            "current_ratio": info.get("currentRatio", np.nan),
            "ocf": info.get("operatingCashflow", np.nan),
            "net_income": info.get("netIncomeToCommon", np.nan),
            "total_revenue": info.get("totalRevenue", np.nan),
            "total_assets": info.get("totalAssets", np.nan),
            "total_debt": info.get("totalDebt", np.nan),
            "total_cash": info.get("totalCash", np.nan),
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "payout_ratio": info.get("payoutRatio", np.nan),
            "revenue_growth": info.get("revenueGrowth", np.nan),
            "earnings_growth": info.get("earningsGrowth", np.nan),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", np.nan),
            "trailing_eps": info.get("trailingEps", np.nan),
            "forward_eps": info.get("forwardEps", np.nan),
            "beta": info.get("beta", 1.0),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", np.nan),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", np.nan),
        }

        try:
            shares = stk.get_shares_full(start=datetime.now() - timedelta(days=400))
            if shares is not None and len(shares) > 0:
                shares_now = shares.iloc[-1]
                shares_year_ago = shares.iloc[0]
                out["shares_change_yoy"] = (shares_now / shares_year_ago) - 1
            else:
                out["shares_change_yoy"] = np.nan
        except Exception:
            out["shares_change_yoy"] = np.nan

        if out["fcf"] and out["market_cap"] and out["market_cap"] > 0:
            out["fcf_yield"] = out["fcf"] / out["market_cap"]
        else:
            out["fcf_yield"] = np.nan

        if out["fcf"] and out["fcf"] > 0 and out["market_cap"]:
            out["p_fcf"] = out["market_cap"] / out["fcf"]
        else:
            out["p_fcf"] = np.nan

        if out["total_revenue"] and out["gross_margins"] and out["total_assets"]:
            gross_profit = out["total_revenue"] * out["gross_margins"]
            if out["total_assets"] > 0:
                out["gp_to_assets"] = gross_profit / out["total_assets"]
            else:
                out["gp_to_assets"] = np.nan
        else:
            out["gp_to_assets"] = np.nan

        if out["total_debt"] and out["ebitda"] and out["ebitda"] > 0:
            out["debt_to_ebitda"] = out["total_debt"] / out["ebitda"]
        else:
            out["debt_to_ebitda"] = np.nan

        if out["ocf"] and out["net_income"] and out["net_income"] > 0:
            out["ocf_to_ni"] = out["ocf"] / out["net_income"]
        else:
            out["ocf_to_ni"] = np.nan

        if out["trailing_eps"] and out["forward_eps"] and out["trailing_eps"] != 0:
            out["eps_revision"] = (out["forward_eps"] - out["trailing_eps"]) / abs(
                out["trailing_eps"]
            )
        else:
            out["eps_revision"] = np.nan

        return out

    except Exception as e:
        logger.warning(f"Failed to load fundamentals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def get_fundamentals_batch(
    tickers: List[str], use_cache: bool = True
) -> pd.DataFrame:
    """배치로 모든 종목의 펀더멘털 가져오기."""
    cache_name = "fundamentals_all"
    if use_cache and _is_cache_fresh(cache_name, max_age_hours=12):
        df = _load_cache(cache_name)
        if set(tickers).issubset(set(df["ticker"].tolist())):
            return df[df["ticker"].isin(tickers)].copy()

    rows = []
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            logger.info(f"Fundamentals progress: {i}/{len(tickers)}")
        rows.append(get_fundamentals(ticker))
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    _save_cache(cache_name, df)
    return df


# ============================================================
# 거시경제 데이터
# ============================================================

def get_macro_indicators() -> Dict:
    """거시경제 레짐 판단용 지표 (VIX/Yield curve/SPY MA200)."""
    cache_name = "macro_indicators"
    if _is_cache_fresh(cache_name, max_age_hours=4):
        return _load_cache(cache_name)

    macro = {}

    def _safe_float(x):
        try:
            if hasattr(x, "iloc"):
                return float(x.iloc[0]) if len(x) == 1 else float(x.values[0])
            return float(x)
        except Exception:
            return np.nan

    try:
        vix = yf.download("^VIX", period="3mo", progress=False, auto_adjust=True)
        if not vix.empty:
            macro["vix"] = _safe_float(vix["Close"].iloc[-1])
            macro["vix_avg_20d"] = _safe_float(vix["Close"].tail(20).mean())
    except Exception as e:
        logger.warning(f"VIX load failed: {e}")
        macro["vix"] = np.nan

    try:
        ten = yf.download("^TNX", period="3mo", progress=False, auto_adjust=True)
        two = yf.download("^IRX", period="3mo", progress=False, auto_adjust=True)
        if not ten.empty:
            macro["yield_10y"] = _safe_float(ten["Close"].iloc[-1])
        if not two.empty:
            macro["yield_3m"] = _safe_float(two["Close"].iloc[-1])
        if "yield_10y" in macro and "yield_3m" in macro:
            macro["yield_curve_10y_3m"] = macro["yield_10y"] - macro["yield_3m"]
    except Exception as e:
        logger.warning(f"Yield load failed: {e}")

    try:
        spy = get_spy_price()
        if not spy.empty and len(spy) >= 200:
            close = spy["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            ma200 = close.rolling(200).mean()
            current = _safe_float(close.iloc[-1])
            ma200_now = _safe_float(ma200.iloc[-1])
            macro["spy_price"] = current
            macro["spy_ma200"] = ma200_now
            macro["spy_above_ma200"] = current > ma200_now
            macro["spy_distance_ma200"] = (current / ma200_now - 1) * 100
    except Exception as e:
        logger.warning(f"SPY MA load failed: {e}")

    _save_cache(cache_name, macro)
    return macro


def determine_regime(macro: dict) -> str:
    """시장 레짐 판정: BULL / NEUTRAL / BEAR (SPACE도 ROCKET과 동일 룰)."""
    vix = macro.get("vix", 20)
    spy_above = macro.get("spy_above_ma200", False)

    if vix is None or pd.isna(vix):
        vix = 20
    if spy_above is None:
        spy_above = False

    if vix > 25:
        return "BEAR"
    elif spy_above and vix < 18:
        return "BULL"
    else:
        return "NEUTRAL"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 60)
    print("FRIDAY data_loader.py smoke test")
    print("=" * 60)

    tickers = get_r1000_live_universe(top_n=1000)
    print(f"\nR1000 live: {len(tickers)} tickers")
    print(f"Top 10: {tickers[:10]}")
    print(f"Tail 5 (mcap floor): {tickers[-5:]}")

    macro = get_macro_indicators()
    print(f"\nMacro: {macro}")
    print(f"Regime: {determine_regime(macro)}")
