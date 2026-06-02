"""
surge_screener.py
=================
FRIDAY — SURGE screener (단기 satellite, 예산 증식용 speculative 트레이드 후보 발굴).

⚠️  이것은 JARVIS(안정 코어)도 SPACE(quant)도 아니다. 단기(일 단위) 트레이드 후보만
    뽑는 별도 위성 모듈이다. 보유 기간은 "일", 청산은 타이트하게 — 이게 핵심이다.

근거 (deep-research 2026-06-02, 113 agents / 20 confirmed claims):
  - DIRECTION TRAP: surge "발생"을 예측하는 신호(MAX·jackpot·low-float·high-short)는
    그 자체로는 month-hold 시 음(-)의 기대수익. 이유 = 업사이드가 이미 프리미엄으로
    가격에 반영(skewness-seeking 리테일이 과지불). Bali-Cakici-Whitelaw(JFE): top-MAX
    decile -1.18%/mo 4F-alpha (t=-4.71), value-weighted.
  - 따라서 본 스크리너는 surge-readiness(타이밍) × fuel(폭발력) × CATALYST(기대수익)로
    구성. catalyst 없는 naked lottery는 피한다. 단기 보유로 month-hold trap을 회피.
  - 52WH 근접 = 모멘텀 dominant(George-Hwang JFE) but breakout-continuation은 reject(0-3).
    → 근접은 타이밍 context, 돌파 후 추격 아님.
  - Pump-and-dump 오염 = small-cap/저유동성에 집중. → hard liquidity gate가 firewall.

v1 데이터(기존 wired만): yfinance OHLCV + yfinance .info short/float + FMP analyst.
  (earnings-calendar / insider Form-4 = v2 hook)

Universe: aggressive small-cap (mcap $150M~$10B), price>$3, ADV20>$2M.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import fmp_client as fmp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"

# 캐시 — SURGE 전용 namespace (SPACE prices_2y 캐시와 충돌 방지)
CACHE_DIR = Path.home() / ".friday_cache" / "surge"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 운영 파라미터 (조정 가능)
# ============================================================
MCAP_MIN = 150_000_000      # $150M (aggressive small-cap 하단)
MCAP_MAX = 10_000_000_000   # $10B (mid-cap 상단까지)
MIN_PRICE = 3.0             # 페니주 컷 (aggressive)
MIN_ADV20 = 2_000_000       # $2M 일거래대금 (aggressive gate — pump-and-dump firewall)
CHASE_RET_5D = 0.20         # 5일 +20% 이미 급등 = "추격 금지" (MAX trap 회피)
TOP_K_ENRICH = 200          # readiness 상위 K개만 FMP/short enrich (호출 절약)
TOP_N_DEFAULT = 25          # 최종 출력 개수
SECTOR_CAP = 8              # 섹터당 최대 종목 (biotech 편중 방지 — surge는 biotech에 몰림)

# SURGE_SCORE 가중치 (research 반영: catalyst가 기대수익, capacity/fuel은 폭발력)
W_READINESS = 0.30          # 타이밍 (coiled spring) — OHLCV
W_EXPLO = 0.20              # 폭발 역량 (이 종목이 급등 "가능"한가) — OHLCV
W_FUEL = 0.20               # 폭발 연료 (short/float) — occurrence-predictor, EV는 음 → modest
W_CATALYST = 0.30           # 기대수익 (analyst) — direction trap 회피의 핵심

# 폭발 역량 게이트 (bank-skew 제거 — 은행은 주간 30% 못 움직임)
# 둘 중 하나 충족해야 통과: 작년 최고 5일 수익률 >= 15% OR 연환산 변동성 >= 50%
MIN_MAX_5D_GAIN_1Y = 0.15
MIN_ANN_VOL = 0.50


# ============================================================
# 캐시 헬퍼
# ============================================================
def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _is_cache_fresh(name: str, max_age_hours: float = 12) -> bool:
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
# 1. Universe — aggressive small-cap (FMP company-screener)
# ============================================================
def get_smallcap_universe(
    mcap_min: float = MCAP_MIN,
    mcap_max: float = MCAP_MAX,
    min_price: float = MIN_PRICE,
    min_share_vol: int = 300_000,
    limit: int = 3000,
    cache_hours: float = 24,
) -> List[str]:
    """small/mid-cap US 보통주 universe (FMP company-screener).

    API 레벨에서 price/mcap/share-volume 사전 필터 → yfinance 호출 폭 축소.
    정밀 $-ADV gate 는 가격 다운로드 후 적용한다.
    """
    cache_name = f"universe_{int(mcap_min/1e6)}_{int(mcap_max/1e9)}b"
    if _is_cache_fresh(cache_name, cache_hours):
        cached = _load_cache(cache_name)
        logger.info(f"SURGE universe from cache: {len(cached)} tickers")
        return cached

    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not set")
        return []

    try:
        r = requests.get(
            f"{FMP_BASE}/company-screener",
            params={
                "marketCapMoreThan": int(mcap_min),
                "marketCapLowerThan": int(mcap_max),
                "priceMoreThan": min_price,
                "volumeMoreThan": min_share_vol,
                "isEtf": "false",
                "isFund": "false",
                "isActivelyTrading": "true",
                "country": "US",
                "limit": limit,
                "apikey": FMP_API_KEY,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"company-screener failed: {e}")
        return []

    if not isinstance(data, list) or not data:
        logger.error(f"company-screener empty/invalid: {data!r}")
        return []

    df = pd.DataFrame(data).dropna(subset=["symbol"])
    # 주요 거래소만 (OTC/핑크시트 = pump-and-dump 핵심 제거)
    if "exchangeShortName" in df.columns:
        df = df[df["exchangeShortName"].isin(["NASDAQ", "NYSE", "AMEX"])]
    tickers = (
        df["symbol"].astype(str).str.strip()
        .str.replace(".", "-", regex=False).tolist()
    )
    tickers = [t for t in tickers if t and t != "nan"]
    _save_cache(cache_name, tickers)
    logger.info(f"SURGE universe fetched: {len(tickers)} tickers (mcap ${mcap_min/1e6:.0f}M~${mcap_max/1e9:.0f}B)")
    return tickers


# ============================================================
# 2. 가격 데이터 (SURGE 전용 캐시)
# ============================================================
def get_prices(tickers: List[str], period: str = "1y", batch_size: int = 100) -> pd.DataFrame:
    """배치 OHLCV 다운로드 (yfinance). SURGE 전용 캐시 — SPACE 캐시와 격리."""
    cache_name = f"surge_prices_{period}"
    if _is_cache_fresh(cache_name, max_age_hours=12):
        cached = _load_cache(cache_name)
        cached_tk = (set(cached.columns.get_level_values(1))
                     if isinstance(cached.columns, pd.MultiIndex) else set(cached.columns))
        if set(tickers).issubset(cached_tk):
            logger.info(f"SURGE prices from cache ({len(tickers)} tickers)")
            return cached

    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        if i % (batch_size * 3) == 0:
            logger.info(f"  price fetch {i}/{len(tickers)}")
        try:
            d = yf.download(batch, period=period, progress=False, auto_adjust=True, threads=True)
            if not d.empty:
                all_data.append(d)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  batch {i} failed: {e}")
    if not all_data:
        return pd.DataFrame()
    combined = pd.concat(all_data, axis=1)
    _save_cache(cache_name, combined)
    return combined


def _field(prices: pd.DataFrame, field: str) -> Optional[pd.DataFrame]:
    if prices.empty or not isinstance(prices.columns, pd.MultiIndex):
        return None
    if field not in prices.columns.get_level_values(0):
        return None
    return prices[field]


# ============================================================
# 3. Readiness 피처 — OHLCV only (coiled spring 타이밍)
# ============================================================
def _pctile_of_last(series: pd.Series, window: int = 100) -> float:
    """series 마지막 값이 직전 window 분포에서 차지하는 percentile (0~1). 낮을수록 '수축'."""
    s = series.dropna().tail(window)
    if len(s) < 20:
        return np.nan
    last = s.iloc[-1]
    return float((s < last).mean())


def compute_readiness(prices: pd.DataFrame) -> pd.DataFrame:
    """티커별 surge-readiness 피처 + 0~100 readiness 점수 (OHLCV only)."""
    close = _field(prices, "Close")
    high = _field(prices, "High")
    low = _field(prices, "Low")
    vol = _field(prices, "Volume")
    if close is None or high is None or low is None or vol is None:
        logger.error("price fields missing — cannot compute readiness")
        return pd.DataFrame()

    rows = []
    tickers = [c for c in close.columns]
    for t in tickers:
        try:
            c = close[t].dropna()
            h = high[t].reindex(c.index)
            l = low[t].reindex(c.index)
            v = vol[t].reindex(c.index)
            feat = compute_features_for_series(c, h, l, v)
            if feat is None:
                continue
            feat["ticker"] = t
            rows.append(feat)
        except Exception as e:
            logger.debug(f"  readiness failed {t}: {e}")
            continue

    return pd.DataFrame(rows)


def compute_features_for_series(c: pd.Series, h: pd.Series, l: pd.Series, v: pd.Series) -> Optional[Dict]:
    """단일 종목 OHLCV 시리즈 → readiness/explosiveness 피처 dict (as-of 마지막 bar).

    live 스크리너(compute_readiness)와 백테(PIT, truncate 후 호출)가 공유 — 동일 수식 보장.
    None 반환 = 데이터 부족(<60 bars).
    """
    c = c.dropna()
    if len(c) < 60:
        return None
    h = h.reindex(c.index)
    l = l.reindex(c.index)
    v = v.reindex(c.index)
    price = float(c.iloc[-1])

    # 유동성: ADV20 ($)
    adv20 = float((c * v).tail(20).mean())

    # 추격 금지 게이트: 5일 수익률
    ret_5d = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else np.nan
    ret_1d = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) >= 2 else np.nan

    # --- 폭발 역량 (surge capacity): 이 종목이 급등 "가능"한가 ---
    daily_ret = c.pct_change()
    ann_vol = float(daily_ret.tail(252).std() * np.sqrt(252)) if len(daily_ret) >= 30 else np.nan
    roll5 = c.pct_change(5)
    max_5d_gain = float(roll5.tail(252).max()) if len(roll5) >= 10 else np.nan
    big_up_days = int((daily_ret.tail(252) > 0.10).sum())  # 작년 +10% 일수
    e_vol = np.clip((ann_vol - 0.30) / 0.90, 0, 1) * 100 if not np.isnan(ann_vol) else 0.0
    e_max5 = np.clip((max_5d_gain - 0.10) / 0.50, 0, 1) * 100 if not np.isnan(max_5d_gain) else 0.0
    e_big = np.clip(big_up_days / 8.0, 0, 1) * 100
    explosiveness = float(0.40 * e_vol + 0.40 * e_max5 + 0.20 * e_big)
    capacity_ok = (
        (not np.isnan(max_5d_gain) and max_5d_gain >= MIN_MAX_5D_GAIN_1Y) or
        (not np.isnan(ann_vol) and ann_vol >= MIN_ANN_VOL)
    )

    # --- 변동성 수축 (ATR) ---
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(20).mean() / c
    atr_pctile = _pctile_of_last(atr_pct, 100)
    score_atr = (1 - atr_pctile) * 100 if not np.isnan(atr_pctile) else 50.0
    atr_pct_now = float(atr_pct.iloc[-1]) if len(atr_pct) and not np.isnan(atr_pct.iloc[-1]) else np.nan

    # --- Bollinger bandwidth squeeze ---
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    bw = (4 * std) / mid
    bw_pctile = _pctile_of_last(bw, 100)
    score_bb = (1 - bw_pctile) * 100 if not np.isnan(bw_pctile) else 50.0

    # --- NR7 ---
    rng = (h - l)
    nr7 = bool(rng.iloc[-1] <= rng.tail(7).min() + 1e-12) if len(rng) >= 7 else False
    score_nr7 = 100.0 if nr7 else 0.0

    # --- 거래량 dry-up ---
    vol50 = float(v.tail(50).mean())
    vol10 = float(v.tail(10).mean())
    dryup = (vol10 / vol50) if vol50 > 0 else np.nan
    rvol = float(v.iloc[-1] / vol50) if vol50 > 0 else np.nan
    score_vol = 50.0 if np.isnan(dryup) else float(np.clip((1.2 - dryup) / 0.8, 0, 1) * 100)

    # --- 52주 고가 근접 ---
    high252 = float(h.tail(252).max())
    dist_52wh = price / high252 if high252 > 0 else np.nan
    if np.isnan(dist_52wh):
        score_52 = 50.0
    elif 0.85 <= dist_52wh <= 1.0:
        score_52 = 100.0
    elif dist_52wh > 1.0:
        score_52 = 80.0
    else:
        score_52 = float(np.clip((dist_52wh - 0.5) / 0.35, 0, 1) * 100)

    readiness = (0.30 * score_atr + 0.20 * score_bb + 0.25 * score_52 +
                 0.15 * score_vol + 0.10 * score_nr7)

    return {
        "price": price, "adv20": adv20, "ret_5d": ret_5d, "ret_1d": ret_1d,
        "atr_pctile": atr_pctile, "atr_pct": atr_pct_now, "bb_pctile": bw_pctile, "nr7": nr7,
        "dryup": dryup, "rvol": rvol, "dist_52wh": dist_52wh,
        "readiness": float(readiness), "ann_vol": ann_vol,
        "max_5d_gain_1y": max_5d_gain, "big_up_days": big_up_days,
        "explosiveness": explosiveness, "capacity_ok": capacity_ok,
    }


# ============================================================
# 4. Fuel — short/float (yfinance .info), occurrence 연료
# ============================================================
def fetch_fuel(tickers: List[str]) -> pd.DataFrame:
    """short float / days-to-cover / float — squeeze 연료. yfinance .info."""
    rows = []
    for i, t in enumerate(tickers):
        if i % 25 == 0:
            logger.info(f"  fuel fetch {i}/{len(tickers)}")
        try:
            info = yf.Ticker(t).info or {}
            float_shares = info.get("floatShares")
            sf = info.get("shortPercentOfFloat")          # 0~1
            dtc = info.get("shortRatio")                   # days-to-cover
            rows.append({
                "ticker": t,
                "float_shares": float_shares,
                "short_pct_float": sf,
                "days_to_cover": dtc,
                "name": info.get("shortName") or info.get("longName") or t,
                "sector": info.get("sector", "Unknown"),
            })
        except Exception as e:
            logger.debug(f"  fuel failed {t}: {e}")
            rows.append({"ticker": t})
        time.sleep(0.05)
    return pd.DataFrame(rows)


def score_fuel(row: pd.Series) -> float:
    """short float + DTC + low-float → 0~100 폭발 연료 점수."""
    sf = row.get("short_pct_float")
    dtc = row.get("days_to_cover")
    fl = row.get("float_shares")
    parts, weights = [], []
    if sf is not None and not pd.isna(sf):
        parts.append(float(np.clip(sf / 0.30, 0, 1)) * 100); weights.append(0.45)
    if dtc is not None and not pd.isna(dtc):
        parts.append(float(np.clip(dtc / 10.0, 0, 1)) * 100); weights.append(0.35)
    if fl is not None and not pd.isna(fl) and fl > 0:
        # float 20M↓ = 100, 200M↑ = 0 (low float = 증폭)
        lf = float(np.clip((200e6 - fl) / 180e6, 0, 1)) * 100
        parts.append(lf); weights.append(0.20)
    if not parts:
        return 0.0
    return float(np.average(parts, weights=weights))


# ============================================================
# 5. Catalyst — analyst (FMP, 기존 wired). direction-trap 회피의 핵심
# ============================================================
def score_catalyst(enr: Dict, price: float) -> Dict:
    """analyst_score + 목표가 업사이드 → 0~100 catalyst. 무커버리지는 mild penalty."""
    analyst_score = enr.get("analyst_score")
    target = enr.get("target_consensus")
    n_analysts = enr.get("num_analyst_ratings", 0) or 0

    upside = np.nan
    if target and price and price > 0:
        upside = target / price - 1

    # 목표가 업사이드 점수: +40%에서 saturate (이미 충분히 강함)
    # binary_target = 업사이드 >100% → 무수익 biotech의 binary 베팅 = 신뢰 낮음 → 0.5 haircut
    #   (이게 catalyst-inflation 의 핵심 수정: pre-revenue biotech가 +200% 타겟으로
    #    catalyst를 saturate 시켜 discrimination 을 죽이는 현상 제거)
    binary_target = bool(not pd.isna(upside) and upside > 1.0)
    if pd.isna(upside):
        upside_score = np.nan
    else:
        upside_score = float(np.clip(upside / 0.40, 0, 1) * 100)
        if binary_target:
            upside_score *= 0.5

    if analyst_score is None and pd.isna(upside_score):
        return {"catalyst": 40.0, "low_coverage": True, "target_upside": np.nan,
                "analyst_score": np.nan, "binary_target": binary_target}  # 무커버리지 = mild penalty

    a = analyst_score if analyst_score is not None else 50.0
    if pd.isna(upside_score):
        catalyst = a
    else:
        # consensus 품질(a)에 더 비중 — 포화된 target 보다 discriminate 잘 됨
        catalyst = 0.65 * a + 0.35 * upside_score
    return {
        "catalyst": float(catalyst),
        "low_coverage": n_analysts < 3,
        "target_upside": float(upside) if not pd.isna(upside) else np.nan,
        "analyst_score": a,
        "binary_target": binary_target,
    }


# ============================================================
# 6. 메인 파이프라인
# ============================================================
def run_surge_screen(
    top_n: int = TOP_N_DEFAULT,
    limit_universe: Optional[int] = None,
    include_extended: bool = False,
    sector_cap: int = SECTOR_CAP,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("FRIDAY SURGE screener — 단기 위성 (speculative)")
    logger.info("=" * 60)

    # 1. universe
    universe = get_smallcap_universe()
    if not universe:
        logger.error("universe 비어있음 — 중단")
        return pd.DataFrame()
    if limit_universe:
        universe = universe[:limit_universe]
        logger.info(f"universe capped to {len(universe)} (--limit)")

    # 2. prices + 3. readiness
    prices = get_prices(universe, period="1y")
    if prices.empty:
        logger.error("가격 데이터 없음 — 중단")
        return pd.DataFrame()
    feat = compute_readiness(prices)
    if feat.empty:
        logger.error("readiness 계산 실패 — 중단")
        return pd.DataFrame()
    logger.info(f"readiness computed: {len(feat)} tickers")

    # 유동성 게이트 (pump-and-dump firewall)
    n0 = len(feat)
    feat = feat[(feat["price"] >= MIN_PRICE) & (feat["adv20"] >= MIN_ADV20)].copy()
    logger.info(f"liquidity gate (price>=${MIN_PRICE}, ADV20>=${MIN_ADV20/1e6:.0f}M): {len(feat)} (from {n0})")

    # 폭발 역량 게이트 (bank-skew 제거): 급등 불가능한 저변동 종목 컷
    n_cap = len(feat)
    feat = feat[feat["capacity_ok"].fillna(False)].copy()
    logger.info(
        f"capacity gate (max5d>={MIN_MAX_5D_GAIN_1Y:.0%} OR annvol>={MIN_ANN_VOL:.0%}): "
        f"{len(feat)} (dropped {n_cap-len(feat)} low-capacity)"
    )

    # 추격 금지: 5일 이미 급등 → extended 플래그
    feat["extended"] = feat["ret_5d"] > CHASE_RET_5D
    if not include_extended:
        n1 = len(feat)
        feat = feat[~feat["extended"]].copy()
        logger.info(f"don't-chase gate (5d ret <= {CHASE_RET_5D:.0%}): {len(feat)} (dropped {n1-len(feat)} extended)")

    if feat.empty:
        logger.warning("게이트 통과 종목 없음")
        return pd.DataFrame()

    # 4. enrich 후보 선정: readiness(타이밍) + explosiveness(역량) 결합으로 상위 K
    feat["pre_score"] = 0.5 * feat["readiness"].fillna(0) + 0.5 * feat["explosiveness"].fillna(0)
    feat = feat.sort_values("pre_score", ascending=False).reset_index(drop=True)
    cand = feat.head(TOP_K_ENRICH).copy()
    logger.info(f"enriching top {len(cand)} by readiness+explosiveness (fuel + catalyst)...")

    # fuel (short/float)
    fuel_df = fetch_fuel(cand["ticker"].tolist())
    cand = cand.merge(fuel_df, on="ticker", how="left")
    cand["fuel"] = cand.apply(score_fuel, axis=1)

    # catalyst (FMP analyst)
    cat_rows = []
    for _, row in cand.iterrows():
        enr = fmp.enrich_symbol(row["ticker"]) or {}
        cs = score_catalyst(enr, row["price"])
        cs["ticker"] = row["ticker"]
        cat_rows.append(cs)
    cat_df = pd.DataFrame(cat_rows)
    cand = cand.merge(cat_df, on="ticker", how="left")

    # 5. SURGE_SCORE
    cand["surge_score"] = (
        W_READINESS * cand["readiness"].fillna(0) +
        W_EXPLO * cand["explosiveness"].fillna(0) +
        W_FUEL * cand["fuel"].fillna(0) +
        W_CATALYST * cand["catalyst"].fillna(40)
    )

    # setup_type 분류
    def _setup(r):
        if r.get("extended"):
            return "extended"
        sf = r.get("short_pct_float")
        dtc = r.get("days_to_cover")
        if (sf is not None and not pd.isna(sf) and sf > 0.20) and \
           (dtc is not None and not pd.isna(dtc) and dtc > 5):
            return "squeeze_fuel"
        return "fresh_coil"
    cand["setup_type"] = cand.apply(_setup, axis=1)

    cand = cand.sort_values("surge_score", ascending=False).reset_index(drop=True)

    # 섹터 캡 (greedy): biotech 편중 방지 — 점수순으로 섹터당 최대 SECTOR_CAP개
    sec_count: Dict[str, int] = {}
    keep_idx = []
    for idx, row in cand.iterrows():
        sec = row.get("sector") or "Unknown"
        if sec_count.get(sec, 0) < sector_cap:
            keep_idx.append(idx)
            sec_count[sec] = sec_count.get(sec, 0) + 1
    n_pre = len(cand)
    cand = cand.loc[keep_idx].reset_index(drop=True)
    logger.info(f"sector cap (<= {sector_cap}/sector): {len(cand)} (from {n_pre}); "
                f"sectors: {dict(sorted(sec_count.items(), key=lambda x: -x[1]))}")

    # 출력 컬럼 정리
    out_cols = [
        "ticker", "name", "sector", "price", "adv20", "surge_score",
        "readiness", "explosiveness", "fuel", "catalyst", "setup_type",
        "ann_vol", "max_5d_gain_1y", "big_up_days",
        "ret_5d", "rvol", "dist_52wh", "atr_pctile", "bb_pctile", "nr7",
        "short_pct_float", "days_to_cover", "float_shares",
        "analyst_score", "target_upside", "binary_target", "low_coverage", "extended",
    ]
    out_cols = [c for c in out_cols if c in cand.columns]
    result = cand[out_cols].head(top_n).copy()

    # 저장 (스크립트와 같은 디렉토리)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = Path(__file__).resolve().parent / "surge_results.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"saved {out_path.name} ({len(result)} rows, stamp {stamp})")
    return result


def _print_result(df: pd.DataFrame, top_n: int) -> None:
    if df.empty:
        print("\n결과 없음.")
        return
    print("\n" + "=" * 100)
    print(f"  FRIDAY SURGE 후보 — Top {min(top_n, len(df))}  (단기 위성 / speculative)")
    print("=" * 100)
    show = df.head(top_n)
    for i, r in show.iterrows():
        up = r.get("target_upside")
        up_s = f"{up:+.0%}" if pd.notna(up) else "  n/a"
        sf = r.get("short_pct_float")
        sf_s = f"{sf:.0%}" if pd.notna(sf) else " n/a"
        av = r.get("ann_vol")
        av_s = f"{av:.0%}" if pd.notna(av) else " n/a"
        print(
            f"{i+1:>2}. {r['ticker']:<6} {r.get('setup_type',''):<12} "
            f"score={r['surge_score']:5.1f}  R={r['readiness']:3.0f} E={r['explosiveness']:3.0f} F={r['fuel']:3.0f} C={r['catalyst']:3.0f}  "
            f"${r['price']:>7.2f}  5d={r.get('ret_5d',float('nan')):+.1%}  "
            f"vol={av_s}  52wh={r.get('dist_52wh',float('nan')):.2f}  sf={sf_s}  tgt={up_s}  "
            f"{str(r.get('name',''))[:22]}"
        )
    print("=" * 100)
    print("⚠️  EXIT DISCIPLINE: 보유=일 단위, 하드스톱 필수. 이 이름들은 month-hold 시")
    print("    음(-) 기대수익(MAX/squeeze trap). catalyst 없는 'low_coverage'는 더 위험.")
    print("    'extended'(이미 급등)는 추격 금지. surge_results.csv 저장됨.")
    print("=" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FRIDAY SURGE screener (단기 위성)")
    ap.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="출력 개수")
    ap.add_argument("--limit", type=int, default=None, help="universe 상한 (테스트용)")
    ap.add_argument("--include-extended", action="store_true", help="이미 급등한 종목도 포함")
    ap.add_argument("--sector-cap", type=int, default=SECTOR_CAP, help="섹터당 최대 종목")
    args = ap.parse_args()

    res = run_surge_screen(
        top_n=args.top,
        limit_universe=args.limit,
        include_extended=args.include_extended,
        sector_cap=args.sector_cap,
    )
    _print_result(res, args.top)
