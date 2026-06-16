"""
screener.py
===========
FRIDAY (SPACE picker) — R1000 universe 매주 스크리닝, 분기 Top 10/20 갈아끼우기.

Phase 24 검증 config:
  - Universe: Synthetic R1000 PIT (top-1000 by mcap) — live는 FMP company-screener
  - Picker: C80_E20 (Momentum 80% + EPS Revisions 20%, regime-flat)
  - Liquidity gate: ADV20 ≥ $5M
  - Sector cap: 6
  - Trend filter: MA200 OR α6m > 0 (BEAR 시 AND)
  - Vol-Managed 30% layer (운영 dashboard에서 별도 처리)
  - 백테 24.99% / Sh 0.703 / MDD -49.11% / Calmar 0.509 / Vol 34.7% (Phase 26b live-faithful;
    Phase 24 헤드라인 26.96%/0.757 은 sector-map 버그 artifact)

quant_tool/screener.py 와 분리된 코드 — share-cardiac risk 회피 (5-26 broken state 사례).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fmp_client as fmp
from data_loader import (
    compute_adv20,
    determine_regime,
    get_fundamentals_batch,
    get_macro_indicators,
    get_price_history,
    get_r1000_live_universe,
    get_spy_price,
)
from factors import (
    PROFILES,
    calc_alpha_momentum,
    calc_altman_score_industry_adjusted,
    calc_eps_revisions_score,
    calc_momentum_score,
    calc_quality_score,
    calc_shareholder_yield_score,
    calc_value_score,
    get_altman_status,
    set_simple_mode,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Phase 24 검증 운영 파라미터
PROFILE = "SPACE"
MIN_ADV20 = 5_000_000          # $5M — Phase 24 검증 gate
MIN_PRICE = 5.0                # 페니주 제외
MIN_MARKET_CAP = 300_000_000   # $300M — get_r1000_live_universe mcap_min과 일치
SECTOR_CAP = 6                 # Phase 16/22 검증
TOP_N_DEFAULT = 20
TOP_N_FMP_ENRICH = 100         # FMP enrich 상위 100 (메가캡 + portfolio 강제 포함)


# ============================================================
# 헬퍼
# ============================================================

def has_price_field(prices: pd.DataFrame, field: str) -> bool:
    if prices.empty:
        return False
    if isinstance(prices.columns, pd.MultiIndex):
        return field in prices.columns.get_level_values(0)
    return field in prices.columns


# ============================================================
# 유니버스 필터 — ADV20 $5M gate (Phase 24 검증)
# ============================================================

def apply_universe_filter(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    min_market_cap: float = MIN_MARKET_CAP,
    min_price: float = MIN_PRICE,
    min_adv20: float = MIN_ADV20,
) -> pd.DataFrame:
    initial = len(df)

    df = df[df["market_cap"] >= min_market_cap].copy()
    logger.info(f"After mcap filter (>=${min_market_cap/1e9:.2f}B): {len(df)} (from {initial})")

    df = df[df["price"] >= min_price].copy()
    logger.info(f"After price filter (>=${min_price}): {len(df)}")

    if has_price_field(prices, "Close") and has_price_field(prices, "Volume"):
        try:
            adv20 = compute_adv20(prices)
            df["adv20"] = df["ticker"].map(adv20.to_dict())
            df = df[df["adv20"].fillna(0) >= min_adv20].copy()
            logger.info(f"After ADV20 gate (>=${min_adv20/1e6:.1f}M): {len(df)}")
        except Exception as e:
            logger.warning(f"ADV20 filter skipped: {e}")

    return df


# ============================================================
# 기술적 지표
# ============================================================

def calc_technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    if not has_price_field(prices, "Close"):
        return pd.DataFrame()

    close = prices["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()

    results = []
    for ticker in close.columns:
        s = close[ticker].dropna()
        if len(s) < 200:
            continue

        ma200 = s.rolling(200).mean().iloc[-1]
        current = s.iloc[-1]

        s_1y = s.tail(252)
        if len(s_1y) > 50:
            cum_max = s_1y.cummax()
            mdd = ((s_1y - cum_max) / cum_max).min()
        else:
            mdd = np.nan

        rets = s.pct_change().dropna()
        vol_60d = rets.tail(60).std() * np.sqrt(252) if len(rets) >= 60 else np.nan

        results.append({
            "ticker": ticker,
            "above_ma200": current > ma200 if pd.notna(ma200) else False,
            "ma200_distance_pct": (current / ma200 - 1) * 100 if pd.notna(ma200) and ma200 > 0 else np.nan,
            "mdd_1y": mdd,
            "volatility_60d": vol_60d,
        })

    return pd.DataFrame(results)


# ============================================================
# Buy/Sell 룰
# ============================================================

def apply_buy_filters(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    df["buy_eligible"] = True
    df["buy_block_reasons"] = ""

    if regime == "BEAR":
        cond = df["above_ma200"].fillna(False) & (df["alpha_6m"].fillna(-1) > 0)
    else:
        cond = df["above_ma200"].fillna(False) | (df["alpha_6m"].fillna(-1) > 0)

    blocked = ~cond
    df.loc[blocked, "buy_eligible"] = False
    df.loc[blocked, "buy_block_reasons"] += "TREND_BROKEN;"

    return df


def apply_sell_now_signals(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """SELL_NOW 룰 — SPACE는 전체 비활성 (백테 환경 일치).

    Phase 24 backtest_phase24_r1000.py는 VOLATILITY_SPIKE/MASSIVE_DILUTION/
    DEEP_DRAWDOWN/TREND_BREAK 룰 전혀 적용 안 함. 백테 결과(live-faithful
    24.99%/Sh 0.703)는 pure picker + ADV20 gate + size-slip만으로 도달.

    SP500용 threshold (vol > 0.80, dilution > 0.15, MDD < -0.50)는 R1000
    small/mid-cap에 너무 엄격 — 운영 결과 Top 20 중 372/835 차단되어
    백테 환경과 mismatch. 사용자 결정 (2026-05-28): 백테 일치 우선.

    참고용 컬럼만 유지 (volatility_60d / mdd_1y / shares_change_yoy) —
    dashboard에서 정보 표시는 가능하나 자동 SELL 트리거 X. 분기 rebal까지
    무조건 hold (Cap20/Hold20 + Vol-Managed 30% layer가 risk 관리).
    """
    df["sell_now"] = False
    df["sell_reasons"] = ""
    return df


# ============================================================
# FMP Enrichment — Top N + portfolio + mega-caps
# ============================================================

def enrich_with_fmp(
    df: pd.DataFrame,
    top_n: int = TOP_N_FMP_ENRICH,
    weights: dict | None = None,
) -> pd.DataFrame:
    logger.info(f"FMP enrichment for top {top_n} + force-includes...")

    eligible = df.copy()
    if "total_score" in eligible.columns:
        eligible = eligible.sort_values("total_score", ascending=False)
    candidates = eligible.head(top_n).copy()

    must_include = []

    # 1. 메가캡 ($300B+)
    if "market_cap" in df.columns:
        mega = df[df["market_cap"] >= 300e9]["ticker"].tolist()
        must_include.extend(mega)
        logger.info(f"  Mega-caps (>=$300B): {len(mega)}")

    # 2. FRIDAY portfolio holdings (별도 파일 — quant_tool/portfolio.json 와 분리)
    try:
        pf_file = Path(__file__).parent / "portfolio.json"
        if pf_file.exists():
            with open(pf_file) as f:
                pf_data = json.load(f)
            holdings = list(pf_data.get("holdings", {}).keys())
            must_include.extend(holdings)
            if holdings:
                logger.info(f"  FRIDAY portfolio holdings: {len(holdings)}")
    except Exception as e:
        logger.warning(f"portfolio.json load failed: {e}")

    must_include = list(set(must_include))
    must_include_in_df = [t for t in must_include if t in df["ticker"].values]
    extra = [t for t in must_include_in_df if t not in candidates["ticker"].values]
    if extra:
        extra_df = df[df["ticker"].isin(extra)].copy()
        candidates = pd.concat([candidates, extra_df], ignore_index=True)
        logger.info(f"  Force-enrich added: {extra}")

    logger.info(f"  Total to enrich: {len(candidates)}")

    if candidates.empty:
        return df

    symbols = candidates["ticker"].tolist()

    calls_before = fmp.get_call_count()
    fmp_df = fmp.enrich_batch(symbols, delay_sec=0.05)
    calls_after = fmp.get_call_count()
    logger.info(f"  FMP calls used: {calls_after - calls_before}")

    if fmp_df.empty:
        logger.warning("FMP returned empty; using base scores.")
        df["has_fmp_data"] = False
        df["quality_score_enhanced"] = df.get("quality_score", pd.Series(50.0, index=df.index))
        df["total_score_enhanced"] = df.get("total_score", pd.Series(50.0, index=df.index))
        df["display_score"] = df["total_score_enhanced"]
        return df

    fmp_df = fmp_df.rename(columns={"symbol": "ticker"})
    df = df.merge(fmp_df, on="ticker", how="left", suffixes=("", "_fmp_dup"))
    dup_cols = [c for c in df.columns if c.endswith("_fmp_dup")]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    df = recalculate_with_fmp(df, weights=weights)
    return df


def recalculate_with_fmp(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    if weights is None:
        weights = PROFILES["SPACE"]["NEUTRAL"]

    if "piotroski" in df.columns:
        df["piotroski_score"] = (df["piotroski"].fillna(4.5) / 9 * 100).clip(0, 100)

    if "altman_z" in df.columns:
        df["altman_score"] = df.apply(
            lambda row: calc_altman_score_industry_adjusted(
                row.get("altman_z"), row.get("sector", "Other")
            ),
            axis=1,
        )
        df["altman_status"] = df.apply(
            lambda row: get_altman_status(row.get("altman_z"), row.get("sector", "Other")),
            axis=1,
        )

    if "analyst_score" in df.columns:
        df["analyst_sentiment_score"] = df["analyst_score"].fillna(50)

    if "target_consensus" in df.columns and "price" in df.columns:
        df["upside_pct"] = ((df["target_consensus"] - df["price"]) / df["price"] * 100).fillna(0)
        df["upside_score"] = (df["upside_pct"].clip(-30, 30) / 60 * 100 + 50).clip(0, 100)

    quality_components = ["quality_score"]
    if "piotroski_score" in df.columns:
        quality_components.append("piotroski_score")
    if "altman_score" in df.columns:
        quality_components.append("altman_score")

    if len(quality_components) > 1:
        has_fmp = df["piotroski"].notna() if "piotroski" in df.columns else pd.Series(False, index=df.index)
        df.loc[has_fmp, "quality_score_enhanced"] = df.loc[has_fmp, quality_components].mean(axis=1)
        df["quality_score_enhanced"] = df["quality_score_enhanced"].fillna(df["quality_score"])
    else:
        df["quality_score_enhanced"] = df["quality_score"]

    # SPACE C80_E20: V=0, Q=0, M=0.80, S=0, E=0.20 (regime-flat)
    df["total_score_enhanced"] = (
        df["value_score"].fillna(50) * weights.get("value", 0.0)
        + df["quality_score_enhanced"].fillna(50) * weights.get("quality", 0.0)
        + df["momentum_score"].fillna(50) * weights.get("momentum", 0.80)
        + df["shareholder_score"].fillna(50) * weights.get("shareholder", 0.0)
        + df["eps_revisions_score"].fillna(50) * weights.get("eps_rev", 0.20)
    )

    if "analyst_sentiment_score" in df.columns:
        sentiment_bonus = (df["analyst_sentiment_score"].fillna(50) - 50) * 0.10
        df["total_score_enhanced"] = df["total_score_enhanced"] + sentiment_bonus

    df["has_fmp_data"] = False
    if "piotroski" in df.columns:
        df["has_fmp_data"] = df["piotroski"].notna()

    df["display_score"] = df["total_score_enhanced"].fillna(df["total_score"])
    return df


# ============================================================
# 메인 스크리너
# ============================================================

def run_screener(
    top_n: int = TOP_N_DEFAULT,
    output_csv: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    print("=" * 80)
    print("🚀 FRIDAY — SPACE Screener (R1000 + ADV20 $5M gate)")
    print(f"   Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Macro / Regime
    logger.info("Step 1/7: Loading macro indicators...")
    macro = get_macro_indicators()
    regime = determine_regime(macro)

    weights = PROFILES[PROFILE].get(regime, PROFILES[PROFILE]["NEUTRAL"])
    set_simple_mode(True)
    logger.info(f"Scoring mode: SIMPLE (global z-score, profile={PROFILE})")

    print(f"\n📊 Market Regime: {regime}")
    if pd.notna(macro.get("vix")):
        print(f"   VIX: {macro.get('vix'):.2f}")
    print(f"   SPY vs MA200: "
          f"{'ABOVE' if macro.get('spy_above_ma200') else 'BELOW'} "
          f"({macro.get('spy_distance_ma200', 0):+.2f}%)")
    print(f"\n📐 Factor Weights ({PROFILE} / {regime}):")
    for k, v in weights.items():
        print(f"   {k:15s}: {v*100:5.1f}%")

    # 2. R1000 Universe
    logger.info("Step 2/7: Loading R1000 live universe...")
    tickers = get_r1000_live_universe(top_n=1000)
    if not tickers:
        logger.error("Failed to load R1000 universe!")
        return pd.DataFrame()
    print(f"\n📈 R1000 universe: {len(tickers)} tickers (top-1000 by mcap)")

    # 3. Prices
    logger.info("Step 3/7: Loading price history (this takes ~5-10min cold)...")
    prices = get_price_history(tickers, period="2y")
    spy_price = get_spy_price()
    if prices.empty:
        logger.error("Failed to load prices!")
        return pd.DataFrame()

    # 4. Fundamentals
    logger.info("Step 4/7: Loading fundamentals (yfinance)...")
    fund_df = get_fundamentals_batch(tickers, use_cache=use_cache)
    fund_df = fund_df.dropna(subset=["market_cap"])

    # 5. Technical indicators + alpha
    logger.info("Step 5/7: Calculating technical indicators...")
    tech_df = calc_technical_indicators(prices)
    if tech_df.empty:
        logger.warning("Technical indicators empty; using defaults.")
        fund_df["above_ma200"] = False
        fund_df["ma200_distance_pct"] = np.nan
        fund_df["mdd_1y"] = np.nan
        fund_df["volatility_60d"] = np.nan
    else:
        fund_df = fund_df.merge(tech_df, on="ticker", how="left")

    # Alpha momentum
    alpha_6m = pd.Series(dtype=float)
    alpha_12m = pd.Series(dtype=float)
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            close_prices = prices["Close"]
        else:
            close_prices = prices

        if has_price_field(spy_price, "Close"):
            spy_close = spy_price["Close"]
            if isinstance(spy_close, pd.DataFrame):
                spy_close = spy_close.iloc[:, 0]
        else:
            spy_close = pd.Series(dtype=float)

        if not spy_close.empty and not close_prices.empty:
            logger.info(f"  Alpha 6M for {len(close_prices.columns)} tickers...")
            alpha_6m = calc_alpha_momentum(close_prices, spy_close, period_months=6)
            logger.info(f"  Alpha 12M...")
            alpha_12m = calc_alpha_momentum(close_prices, spy_close, period_months=12)

            fund_df["alpha_6m"] = fund_df["ticker"].map(alpha_6m)
            fund_df["alpha_12m"] = fund_df["ticker"].map(alpha_12m)
    except Exception as e:
        logger.error(f"Alpha momentum failed: {e}")
        import traceback
        traceback.print_exc()

    if "alpha_6m" not in fund_df.columns:
        fund_df["alpha_6m"] = np.nan
    if "alpha_12m" not in fund_df.columns:
        fund_df["alpha_12m"] = np.nan

    # 6. Universe filter (ADV20 $5M gate)
    fund_df = apply_universe_filter(fund_df, prices)

    # 7. Factor scoring
    logger.info("Step 6/7: Calculating factor scores...")
    fund_df["value_score"] = calc_value_score(fund_df)
    fund_df["quality_score"] = calc_quality_score(fund_df)
    fund_df["shareholder_score"] = calc_shareholder_yield_score(fund_df)
    fund_df["eps_revisions_score"] = calc_eps_revisions_score(fund_df)

    if not tech_df.empty and "ticker" in tech_df.columns:
        above_ma200_map = dict(zip(tech_df["ticker"], tech_df["above_ma200"]))
    else:
        above_ma200_map = {}

    fund_df["momentum_score"] = calc_momentum_score(
        fund_df,
        alpha_6m=alpha_6m,
        alpha_12m=alpha_12m,
        above_ma200=pd.Series(above_ma200_map),
    )

    fund_df["total_score"] = (
        fund_df["value_score"].fillna(50) * weights.get("value", 0.0)
        + fund_df["quality_score"].fillna(50) * weights.get("quality", 0.0)
        + fund_df["momentum_score"].fillna(50) * weights.get("momentum", 0.80)
        + fund_df["shareholder_score"].fillna(50) * weights.get("shareholder", 0.0)
        + fund_df["eps_revisions_score"].fillna(50) * weights.get("eps_rev", 0.20)
    )
    logger.info(f"Weights: M={weights.get('momentum', 0):.0%} E={weights.get('eps_rev', 0):.0%}")

    # FMP enrich (Top 100 + force-includes), then filter
    fund_df = fund_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    fund_df = enrich_with_fmp(fund_df, top_n=TOP_N_FMP_ENRICH, weights=weights)

    fund_df = apply_buy_filters(fund_df, regime)
    fund_df = apply_sell_now_signals(fund_df, regime)

    sort_col = "display_score" if "display_score" in fund_df.columns else "total_score"
    fund_df = fund_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    fund_df["rank"] = fund_df.index + 1

    # space_rank: SSoT for actual SPACE picks
    #   - buy_eligible=True, sell_now=False, sector cap 6
    logger.info("Step 7/7: Computing space_rank (SSoT)...")
    eligible_for_space = fund_df[
        (fund_df["buy_eligible"] == True) & (fund_df["sell_now"] != True)
    ].copy()

    space_picks: list[str] = []
    space_sector_count: dict[str, int] = {}
    for _, row in eligible_for_space.iterrows():
        if len(space_picks) >= 50:
            break
        sec = row.get("sector", "Unknown")
        if pd.isna(sec):
            sec = "Unknown"
        if space_sector_count.get(sec, 0) < SECTOR_CAP:
            space_picks.append(row["ticker"])
            space_sector_count[sec] = space_sector_count.get(sec, 0) + 1

    space_rank_map = {t: i + 1 for i, t in enumerate(space_picks)}
    fund_df["space_rank"] = fund_df["ticker"].map(space_rank_map)
    logger.info(f"space_rank assigned: {len(space_picks)} stocks pass filters")

    # Display Top N (sector-capped)
    final_picks = []
    sector_count: dict[str, int] = {}
    for _, row in fund_df[fund_df["buy_eligible"]].iterrows():
        sec = row.get("sector", "Unknown")
        if pd.isna(sec):
            sec = "Unknown"
        if sector_count.get(sec, 0) < SECTOR_CAP:
            final_picks.append(row)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(final_picks) >= top_n:
            break

    final_df = pd.DataFrame(final_picks).reset_index(drop=True)
    print_results(final_df, fund_df, regime, top_n)

    if output_csv is None:
        output_csv = "friday_results.csv"

    output_cols = [
        "rank", "space_rank", "ticker", "name", "sector", "industry",
        "market_cap", "price", "adv20",
        "has_fmp_data",
        "total_score", "display_score", "total_score_enhanced",
        "value_score", "quality_score", "quality_score_enhanced",
        "momentum_score", "shareholder_score", "eps_revisions_score",
        "piotroski_score", "altman_score",
        "analyst_sentiment_score", "upside_score", "upside_pct",
        "ev_ebitda", "pe", "pb", "fcf_yield", "roa", "gp_to_assets",
        "fmp_roe", "fmp_roic", "fmp_roa", "fmp_fcf_yield",
        "fmp_ev_ebitda", "fmp_net_debt_ebitda", "fmp_income_quality",
        "fmp_pe", "fmp_pb", "fmp_ps", "fmp_p_fcf",
        "piotroski", "altman_z", "altman_status",
        "analyst_consensus", "num_analyst_ratings", "buy_pct",
        "target_consensus", "target_high", "target_low",
        "eps_revision", "earnings_quarterly_growth",
        "alpha_6m", "alpha_12m", "above_ma200", "mdd_1y", "volatility_60d",
        "shares_change_yoy", "dividend_yield",
        "buy_eligible", "buy_block_reasons", "sell_now", "sell_reasons",
    ]
    available = [c for c in output_cols if c in fund_df.columns]
    full_output = fund_df[available].head(200).copy()
    full_output.to_csv(output_csv, index=False, float_format="%.4f")

    # Macro JSON (dashboard용)
    macro_save = {
        "vix": float(macro.get("vix")) if pd.notna(macro.get("vix")) else None,
        "spy_above_ma200": bool(macro.get("spy_above_ma200")) if macro.get("spy_above_ma200") is not None else None,
        "spy_distance_ma200": float(macro.get("spy_distance_ma200", 0)),
        "regime": regime,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("macro_indicators.json", "w") as f:
        json.dump(macro_save, f, indent=2)
    logger.info("Macro saved to macro_indicators.json")
    print(f"\n💾 Full results (top 200) → {output_csv}")

    return final_df


def print_results(final_df: pd.DataFrame, full_df: pd.DataFrame, regime: str, top_n: int):
    print(f"\n{'='*100}")
    print(f"🏆 SPACE TOP {top_n} (Sector-capped {SECTOR_CAP}, R1000 + ADV20 $5M)")
    print(f"{'='*100}\n")

    if final_df.empty:
        print("❌ No candidates found.")
        return

    has_fmp = (final_df.get("has_fmp_data", pd.Series(False)).fillna(False).any()
               if "has_fmp_data" in final_df.columns else False)
    score_col = "display_score" if has_fmp else "total_score"

    base_cols = ["rank", "space_rank", "ticker", "sector", score_col, "total_score", "adv20"]
    fmp_cols = ["piotroski", "altman_z", "analyst_consensus", "buy_pct", "upside_pct"]
    display_cols = [c for c in base_cols + fmp_cols if c in final_df.columns]
    display = final_df[display_cols].head(top_n).copy()

    if "buy_pct" in display.columns:
        display["buy_pct"] = display["buy_pct"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
    if "upside_pct" in display.columns:
        display["upside_pct"] = display["upside_pct"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) and x != 0 else "-")
    if "altman_z" in display.columns:
        display["altman_z"] = display["altman_z"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    if "piotroski" in display.columns:
        display["piotroski"] = display["piotroski"].apply(lambda x: f"{int(x)}/9" if pd.notna(x) else "-")
    if "analyst_consensus" in display.columns:
        display["analyst_consensus"] = display["analyst_consensus"].fillna("-")
    if "adv20" in display.columns:
        display["adv20"] = display["adv20"].apply(lambda x: f"${x/1e6:.0f}M" if pd.notna(x) else "-")
    for col in [score_col, "total_score"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    if "space_rank" in display.columns:
        display["space_rank"] = display["space_rank"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

    display = display.rename(columns={
        "rank": "Raw",
        "space_rank": "SPACE",
        "ticker": "Ticker",
        "sector": "Sector",
        "display_score": "Score⭐" if has_fmp else "Score",
        "total_score": "Base",
        "adv20": "ADV20",
        "piotroski": "Piot",
        "altman_z": "AltZ",
        "analyst_consensus": "Rating",
        "buy_pct": "Buy%",
        "upside_pct": "Target",
    })

    print(display.to_string(index=False))

    print(f"\n📊 Sector Distribution:")
    sec_dist = final_df["sector"].value_counts()
    for sec, cnt in sec_dist.items():
        print(f"   {sec:30s}: {cnt}")

    sell_now_df = full_df[full_df.get("sell_now", False) == True].copy()
    if len(sell_now_df) > 0:
        print(f"\n{'='*100}")
        print(f"⚠️  SELL_NOW Warnings ({len(sell_now_df)} stocks):")
        print(f"{'='*100}")
        print(sell_now_df[["ticker", "sector", "sell_reasons"]].head(20).to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FRIDAY SPACE Screener (R1000 + ADV20 $5M)"
    )
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    try:
        run_screener(
            top_n=args.top,
            output_csv=args.output,
            use_cache=not args.no_cache,
        )
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
