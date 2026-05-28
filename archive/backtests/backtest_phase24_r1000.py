"""
backtest_phase24_r1000.py
=========================
Phase 24: Synthetic R1000 universe + ADV20 liquidity gate + size-dependent slippage.
Picker = Phase 16 (6m raw alpha no-skip, C80_E20, sector cap 6) 그대로 — universe 만 교체.

Universe:
  Synthetic R1000 PIT = top-1000 by mcap each quarterly rebal (FMP historical-mcap).
  CSV: archive/backtests/russell1000_synthetic_pit_constituents.csv  (44 dates × 1000)
  Union: 1707 ticker (ever in top-1000 over 11y).
  Effective ~892/quarter (prices+all-3-fund 모두 보유).

Liquidity gate:
  ADV20 = 20-day average daily dollar volume = mean(Close × Volume) over [as_of-19, as_of].
  Filter: ADV20 ≥ $5M.

Slippage 모델 (Almgren-Chriss linear form):
  slip_bps = base_bps + α · (order_value / ADV20) · 100
  base_bps = 10 (기존 flat 10bps)
  α = 10
  → 1% of ADV: +10bps (총 20)
  → 5% of ADV: +50bps (총 60) ← punishing for medium-cap
  → 10% of ADV: +100bps (총 110) ← brutal — but 10% of daily volume is enormous
  per-order 단위로 buy_ticker/sell_full 에 transaction_cost 인자로 주입.

Modes:
  --mode r1000      : Synthetic R1000 universe + ADV20 gate + size-slip  (Phase 24 본격)
  --mode sp500      : SP500 PIT (Phase 16과 동일 universe), ADV20 / slip OFF — sanity
  --mode r1000-noslip: R1000 + ADV20 gate, but base flat 10bps (slip decomp)
  --mode r1000-nogate: R1000 + size-slip, no ADV20 filter (gate decomp)

Sanity:
  --mode sp500 결과는 Phase 16 18.09% / Sh 0.594 일치해야 framework OK.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))
ARCHIVE = QUANT_TOOL / "archive" / "backtests"
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

import factors
import backtest_advanced_risk as arb
from backtest import INITIAL_CAPITAL, get_static_fundamentals, load_sp500_tickers, load_spy
from backtest_v2 import determine_regime, load_vix
from backtest_phase2_pit import load_pit_membership, sp500_at_date, load_combined_prices
from backtest_phase12_live_logic import prepare_fund_df_for_live, SECTOR_CAP_LIVE
from backtest_phase16_full_pit import _const
from backtest_screener_exact import calc_above_ma200_all
from backtest_volmanaged_rocket import metrics
from factors import (
    calc_alpha_momentum,
    calc_value_score,
    calc_quality_score,
    calc_shareholder_yield_score,
    calc_eps_revisions_score,
    calc_momentum_score,
)
from pit_fundamentals import get_pit_fundamentals_universe, _load_all_caches_v2
from validate_cap20_hold20 import buy_ticker, enforce_cap, sell_full, current_value
from backtest_volmanaged_rocket import (
    compute_realized_vol,
    is_month_end,
    rescale_to_exposure,
    target_exposure,
)

logging.basicConfig(level=logging.WARNING)

START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"
BUY_TOP    = 10
HOLD_TOP   = 20
ADV20_DAYS = 20
ADV20_MIN  = 5_000_000      # $5M
SLIP_BASE_BPS = 10           # flat baseline
SLIP_ALPHA    = 10           # linear coefficient

# C80_E20 가중치 — Phase 16 채택값
SCHEDULE_C80_E20 = _const(0, 0, 80, 0, 20)

R1000_PIT_CSV     = ARCHIVE / "russell1000_synthetic_pit_constituents.csv"
R1000_EXT_PRICES  = Path.home() / ".quant_tool_cache" / "backtest" / "prices_r1000_extension_2015-01-01_2025-12-31.pkl"
SP500_PIT_CSV     = ARCHIVE / "sp500_pit_constituents.csv"


# ============================================================
# Loaders — R1000 PIT membership + Combined prices
# ============================================================

def load_r1000_synthetic_pit_membership() -> pd.DataFrame:
    """Synthetic R1000 daily membership 데이터 (load_pit_membership 와 동일 포맷)."""
    df = pd.read_csv(R1000_PIT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def r1000_at_date(pit_df: pd.DataFrame, as_of: pd.Timestamp) -> set[str]:
    idx = pit_df["date"].searchsorted(as_of, side="right") - 1
    if idx < 0:
        idx = 0
    return set(t.strip() for t in pit_df.iloc[idx]["tickers"].split(",") if t.strip())


def load_combined_prices_r1000() -> pd.DataFrame:
    """SP500 + delisted + R1000 extension 3 캐시 union."""
    base = load_combined_prices()
    if not R1000_EXT_PRICES.exists():
        print("WARNING: R1000 extension cache 없음 — SP500 PIT prices 만 사용")
        return base
    with open(R1000_EXT_PRICES, "rb") as f:
        ext = pickle.load(f)
    # 같은 multi-index 형태로 결합
    if isinstance(ext.columns, pd.MultiIndex):
        close_ext = ext["Close"]
        valid = [t for t in close_ext.columns if not close_ext[t].isna().all()]
        keep_cols = [(f, t) for f in ext.columns.levels[0] for t in valid if (f, t) in ext.columns]
        ext = ext[keep_cols]
    combined = pd.concat([base, ext], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# ============================================================
# ADV20 + Slippage model
# ============================================================

def compute_adv20(prices: pd.DataFrame, as_of: pd.Timestamp,
                  tickers: list[str], window: int = ADV20_DAYS) -> dict[str, float]:
    """20-day average daily dollar volume per ticker.

    ADV20(t,ticker) = mean(Close × Volume) over (t-window+1, t).
    Returns dict ticker → ADV20 ($). NaN 면 dict 에서 빠짐.
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        return {}
    close_df = prices["Close"]
    if "Volume" not in prices.columns.get_level_values(0):
        return {}
    vol_df = prices["Volume"]

    # as_of 까지 잘라서 마지막 window 일 평균
    close_sub = close_df.loc[:as_of].tail(window)
    vol_sub = vol_df.loc[:as_of].tail(window)
    if len(close_sub) < window // 2:
        return {}

    dollar_vol = close_sub * vol_sub
    common_t = [t for t in tickers if t in dollar_vol.columns]
    if not common_t:
        return {}
    adv = dollar_vol[common_t].mean(axis=0)
    return {t: float(v) for t, v in adv.items() if pd.notna(v) and v > 0}


def slip_bps(order_value: float, adv20: float,
             base_bps: float | None = None, alpha: float | None = None) -> float:
    """Linear market-impact 슬리피지. 호출자가 order_value, ADV20 모두 알 때 사용.

    NOTE: base_bps / alpha 가 None 이면 module-level 상수를 동적 lookup → sensitivity
    grid 에서 monkey-patch 가능. (default 인자는 함수 정의 시점에 binding 되므로 사용 X)
    """
    if base_bps is None:
        base_bps = SLIP_BASE_BPS
    if alpha is None:
        alpha = SLIP_ALPHA
    if adv20 is None or adv20 <= 0:
        return base_bps + alpha * 100  # 최악 가정
    participation = order_value / adv20
    return base_bps + alpha * participation * 100


def slip_frac(order_value: float, adv20: float | None,
              size_dependent: bool = True) -> float:
    """전체 슬리피지 fraction (= bps / 10000) — buy_ticker/sell_full 의 transaction_cost 인자."""
    if not size_dependent:
        return SLIP_BASE_BPS / 10000.0
    if adv20 is None:
        return slip_bps(order_value, 0) / 10000.0
    return slip_bps(order_value, adv20) / 10000.0


# ============================================================
# Picker — Phase 16 그대로 + ADV20 gate (옵션)
# ============================================================

def select_picks_phase24(
    prices, spy, vix, as_of, top_n, pit_membership,
    weight_schedule, sector_map, bal_cache, inc_cache, cf_cache,
    *,
    universe_fn,           # callable(pit_df, as_of) -> set[str]
    apply_adv20_gate: bool,
    adv20_min: float | None = None,
) -> tuple[list[str], str, dict[str, float]]:
    """Phase 16 picker + universe injection + 옵션 ADV20 gate.

    Returns: (picks, regime, adv20_map)
    """
    regime = determine_regime(as_of, vix, spy)
    weights = weight_schedule.get(regime, weight_schedule.get("NEUTRAL"))
    factors.set_simple_mode(True)

    close = arb.get_close_prices(prices.loc[:as_of])
    spy_hist = spy.loc[:as_of].dropna()

    members = universe_fn(pit_membership, as_of)
    eligible = [t for t in close.columns if t in members]
    if not eligible:
        return [], regime, {}
    close = close[eligible]

    # === ADV20 — slippage cost 산정에 항상 필요. gate 적용은 옵션. ===
    adv20_map = compute_adv20(prices.loc[:as_of], as_of, eligible)
    if apply_adv20_gate:
        thresh = adv20_min if adv20_min is not None else ADV20_MIN
        liquid = [t for t in eligible if adv20_map.get(t, 0.0) >= thresh]
        if liquid:
            eligible = liquid
            close = close[eligible]
        else:
            return [], regime, adv20_map

    fund_pit = get_pit_fundamentals_universe(eligible, as_of, close, sector_map)
    if fund_pit.empty:
        return [], regime, adv20_map

    above_ma200 = calc_above_ma200_all(close, as_of)
    alpha_6m = calc_alpha_momentum(close, spy_hist, period_months=6)
    alpha_12m = calc_alpha_momentum(close, spy_hist, period_months=12)

    df = prepare_fund_df_for_live(fund_pit)
    df["value_score"]        = calc_value_score(df)
    df["quality_score"]      = calc_quality_score(df)
    df["shareholder_score"]  = calc_shareholder_yield_score(df)
    df["eps_revisions_score"]= calc_eps_revisions_score(df)
    df["momentum_score"] = calc_momentum_score(
        df, alpha_6m, alpha_12m, pd.Series(above_ma200),
    )

    df["total_score"] = (
        df["value_score"].fillna(50) * weights.get("value", 0.0)
        + df["quality_score"].fillna(50) * weights.get("quality", 0.0)
        + df["momentum_score"].fillna(50) * weights.get("momentum", 0.0)
        + df["shareholder_score"].fillna(50) * weights.get("shareholder", 0.0)
        + df["eps_revisions_score"].fillna(50) * weights.get("eps_rev", 0.0)
    )

    df["_above_ma200"] = df["ticker"].map(above_ma200).fillna(False)
    df["_alpha_6m"] = df["ticker"].map(alpha_6m)
    df["_alpha_pos"] = df["_alpha_6m"].fillna(-1) > 0
    if regime == "BEAR":
        trend_ok = df["_above_ma200"] & df["_alpha_pos"]
    else:
        trend_ok = df["_above_ma200"] | df["_alpha_pos"]
    df = df[trend_ok].copy()

    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    picks: list[str] = []
    sector_count: dict[str, int] = {}
    for _, row in df.iterrows():
        if len(picks) >= top_n:
            break
        sec = row.get("sector", "Unknown")
        if pd.isna(sec) or sec is None:
            sec = "Unknown"
        if sector_count.get(sec, 0) < SECTOR_CAP_LIVE:
            picks.append(row["ticker"])
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return picks, regime, adv20_map


# ============================================================
# Simulate — size-dependent slippage, per-order cost
# ============================================================

def simulate(
    prices, spy, vix, pit_membership,
    weight_schedule, sector_map, bal_cache, inc_cache, cf_cache,
    *,
    universe_fn,
    apply_adv20_gate: bool,
    size_dependent_slip: bool,
    target_vol: float | None = None,   # ← VM30 stack: 0.30 면 30% 연환산 vol 타깃
) -> tuple[pd.Series, dict]:
    factors.set_simple_mode(True)
    prices_w = prices.loc[START_DATE:END_DATE]
    spy_w    = spy.loc[START_DATE:END_DATE]
    close = arb.get_close_prices(prices_w)
    rebal_dates = arb.get_rebalance_dates(START_DATE, END_DATE)

    holdings: dict = {}
    cash = float(INITIAL_CAPITAL)
    history = []
    yearly_realized: dict[int, float] = {}
    daily_total_history: list[float] = []
    current_exposure_target = 1.0
    idx = 0

    diag = {"slip_bps_sum": 0.0, "n_trades": 0, "median_adv20": [], "gate_drops": 0}
    dates_list = list(close.index)

    for i, date in enumerate(dates_list):
        next_date = dates_list[i + 1] if i + 1 < len(dates_list) else None
        if idx < len(rebal_dates) and date >= rebal_dates[idx]:
            yearly_realized.setdefault(date.year, 0.0)
            try:
                hold_picks, _, adv20_h = select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    HOLD_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=universe_fn,
                    apply_adv20_gate=apply_adv20_gate,
                )
                buy_picks, _, adv20_b = select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    BUY_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=universe_fn,
                    apply_adv20_gate=apply_adv20_gate,
                )
                adv20_map = {**adv20_h, **adv20_b}
                if adv20_map:
                    diag["median_adv20"].append(float(np.median(list(adv20_map.values()))))
            except Exception as e:
                logging.warning(f"pick fail @ {date}: {e}")
                hold_picks, buy_picks, adv20_map = [], [], {}

            hold_set = set(hold_picks)
            # Sell those not in hold
            for ticker in [t for t in list(holdings) if t not in hold_set]:
                if ticker not in close.columns or pd.isna(close.at[date, ticker]):
                    holdings.pop(ticker, None)
                    continue
                # 매도 order_value = current position value
                shares = holdings[ticker]["shares"]
                price = close.at[date, ticker]
                order_val = shares * price
                cost = slip_frac(order_val, adv20_map.get(ticker), size_dependent_slip)
                proceeds, realized = sell_full(close, date, holdings, ticker, cost)
                cash += proceeds
                yearly_realized[date.year] += realized
                diag["slip_bps_sum"] += cost * 10000
                diag["n_trades"] += 1

            slots = BUY_TOP - len([t for t in holdings if t in hold_set])
            to_buy = [t for t in buy_picks if t not in holdings and t in close.columns][:slots]
            if slots > 0 and to_buy and cash > 0:
                # VM 활성 시 매수 예산 = exposure target 까지만 (Phase 17 패턴)
                if target_vol is not None:
                    total, current_equity = current_value(close, date, holdings, cash)
                    target_equity = total * current_exposure_target
                    buy_budget = max(0.0, min(cash, target_equity - current_equity))
                else:
                    buy_budget = cash
                if buy_budget > 0:
                    alloc = buy_budget / len(to_buy)
                    spent = 0.0
                    for ticker in to_buy:
                        cost = slip_frac(alloc, adv20_map.get(ticker), size_dependent_slip)
                        spent += buy_ticker(close, date, holdings, ticker, alloc, cost)
                        diag["slip_bps_sum"] += cost * 10000
                        diag["n_trades"] += 1
                    cash -= spent
                    cash = max(cash, 0.0)

            # enforce_cap 는 동적 cost — 평균 base + 보수치 사용 (모든 trim 에 정확한 adv20 모름)
            cash, _, _, _ = enforce_cap(close, date, holdings, cash,
                                        SLIP_BASE_BPS / 10000.0, yearly_realized)
            idx += 1

        # === 월말 vol-managed 스케일링 (Phase 17 / 22b 패턴) ===
        if target_vol is not None and is_month_end(date, next_date):
            realized_vol = compute_realized_vol(daily_total_history)
            if realized_vol is not None:
                new_target = target_exposure(realized_vol, target_vol)
                if abs(new_target - current_exposure_target) > 0.05:
                    current_exposure_target = new_target
                    cash = rescale_to_exposure(
                        close, date, holdings, cash, current_exposure_target,
                        SLIP_BASE_BPS / 10000.0, yearly_realized,
                    )

        stock_equity = sum(
            holdings[t]["shares"] * close.at[date, t]
            for t in holdings if t in close.columns and pd.notna(close.at[date, t])
        )
        total_value = stock_equity + cash
        history.append({"date": date, "value": total_value})
        daily_total_history.append(total_value)

    curve = pd.Series(
        [h["value"] for h in history],
        index=[h["date"] for h in history],
    )
    if diag["n_trades"] > 0:
        diag["mean_slip_bps"] = diag["slip_bps_sum"] / diag["n_trades"]
    else:
        diag["mean_slip_bps"] = 0.0
    diag["median_adv20_across_rebal"] = (
        float(np.median(diag["median_adv20"])) if diag["median_adv20"] else 0.0
    )
    return curve, diag


# ============================================================
# Main
# ============================================================

MODES = {
    "sp500":         {"universe": "sp500",  "adv20": False, "size_slip": False, "vm": None},
    "r1000-nogate":  {"universe": "r1000",  "adv20": False, "size_slip": True,  "vm": None},
    "r1000-noslip":  {"universe": "r1000",  "adv20": True,  "size_slip": False, "vm": None},
    "r1000":         {"universe": "r1000",  "adv20": True,  "size_slip": True,  "vm": None},
    "r1000-vm30":    {"universe": "r1000",  "adv20": True,  "size_slip": True,  "vm": 0.30},
    "r1000-vm25":    {"universe": "r1000",  "adv20": True,  "size_slip": True,  "vm": 0.25},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES.keys()) + ["all"], default="all",
                   help="Backtest mode (default: all 4 modes back-to-back)")
    args = p.parse_args()

    print("=" * 110)
    print("Phase 24: R1000 universe + ADV20 gate + size-dependent slippage")
    print("=" * 110)

    print("\n[1/3] 데이터 로드 ...")
    prices = load_combined_prices_r1000()
    if isinstance(prices.columns, pd.MultiIndex):
        n_tickers = len(prices["Close"].columns)
    else:
        n_tickers = len(prices.columns)
    print(f"  Combined prices: {prices.shape[0]} rows × {n_tickers} tickers (SP500+delisted+R1000ext)")
    spy = load_spy(START_DATE, END_DATE)
    vix = load_vix(START_DATE, END_DATE)
    fund_static = get_static_fundamentals(load_sp500_tickers())
    sector_map = dict(zip(fund_static["ticker"], fund_static["sector"]))
    bal, inc, cf = _load_all_caches_v2()
    print(f"  PIT 캐시: {len(bal)} bal | {len(inc)} inc | {len(cf)} cf")

    sp500_pit = load_pit_membership()
    r1000_pit = load_r1000_synthetic_pit_membership()
    print(f"  SP500 PIT: {len(sp500_pit)} rebal dates")
    print(f"  Synth R1000 PIT: {len(r1000_pit)} rebal dates")

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]
    print(f"\n[2/3] 백테스트 — {len(modes_to_run)} mode(s): {modes_to_run}")
    results: dict[str, dict] = {}

    for name in modes_to_run:
        cfg = MODES[name]
        if cfg["universe"] == "sp500":
            universe_fn = sp500_at_date
            pit = sp500_pit
        else:
            universe_fn = r1000_at_date
            pit = r1000_pit

        print(f"\n  Running: {name:<14s}  universe={cfg['universe']}  ADV20={cfg['adv20']}  size_slip={cfg['size_slip']}  vm={cfg['vm']}")
        curve, diag = simulate(
            prices, spy, vix, pit,
            SCHEDULE_C80_E20, sector_map, bal, inc, cf,
            universe_fn=universe_fn,
            apply_adv20_gate=cfg["adv20"],
            size_dependent_slip=cfg["size_slip"],
            target_vol=cfg["vm"],
        )
        m = metrics(curve)
        results[name] = {**cfg, **m, **diag}
        print(f"    연 {m['annual']*100:5.2f}%  Sharpe {m['sharpe']:.3f}  "
              f"MDD {m['mdd']*100:6.2f}%  Calmar {m['calmar']:.3f}  Vol {m['vol']*100:.1f}%")
        print(f"    trades={diag['n_trades']}  mean slip={diag['mean_slip_bps']:.1f}bps  "
              f"median ADV20=${diag['median_adv20_across_rebal']/1e6:.1f}M")

    # 요약
    print(f"\n[3/3] 비교 — Phase 16 baseline (SP500, no gate, flat 10bps): 18.09% / Sh 0.594 / MDD -40.09%")
    print("=" * 110)
    print(f"  {'mode':<16} {'universe':<7} {'gate':<5} {'slip':<6} | {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'slipbps':>8}")
    print("  " + "-" * 92)
    order = [m for m in MODES.keys() if m in results]
    for name in order:
        r = results[name]
        print(f"  {name:<16} {r['universe']:<7} "
              f"{'ON' if r['adv20'] else 'OFF':<5} "
              f"{'size' if r['size_slip'] else 'flat':<6} | "
              f"{r['annual']*100:>6.2f}% {r['sharpe']:>7.3f} "
              f"{r['mdd']*100:>7.2f}% {r['calmar']:>7.3f} "
              f"{r['mean_slip_bps']:>7.1f}")
    print()
    if "sp500" in results:
        r = results["sp500"]
        print(f"  Sanity — sp500 모드: 18.09% / Sh 0.594 와 비교 →")
        print(f"    diff: ΔCAGR {(r['annual']-0.1809)*100:+.2f}pp / ΔSharpe {r['sharpe']-0.594:+.3f}")
    if "r1000" in results and "sp500" in results:
        rR = results["r1000"]; rS = results["sp500"]
        print(f"\n  R1000 vs SP500 (둘 다 동일 picker, R1000 만 gate/slip on):")
        print(f"    ΔCAGR {(rR['annual']-rS['annual'])*100:+.2f}pp  ΔSharpe {rR['sharpe']-rS['sharpe']:+.3f}  "
              f"ΔMDD {(rR['mdd']-rS['mdd'])*100:+.2f}pp")

    # 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = QUANT_TOOL / "archive" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"phase24_r1000_{ts}.csv"
    rows = [{"mode": k, **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
