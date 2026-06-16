"""
backtest_phase26_r1000_dualmom.py
=================================
Phase 26 (FRIDAY): Phase 24 R1000 picker (universe + ADV20 $5M gate + size-slip + VM30)
                   + Phase 25 Defensive Rotation sleeve on the de-grossed fraction.

가설
----
ROCKET(SP500)에서 Phase 25 DUALMOM 이 cash de-gross 를 전 dim Pareto-dominate
(+0.66pp CAGR / +0.022 Sharpe / +0.54pp MDD / +0.025 Calmar). FRIDAY(R1000)는
VM30 de-gross 시간이 더 길어(R1000 idiosyncratic crash 가 VM 더 자주 trigger,
lift ~3×) 방어자산 주차 이득이 더 클 것으로 기대 → 확인.

설계
----
- Equity sleeve  : Phase 24 select_picks_phase24 (R1000 + ADV20 gate + size-slip) 그대로.
- Vol scaling    : 월말 realized_vol 기반 노출 e (TARGET_VOL=0.30) — Phase 24 그대로.
- de-grossed 분 (1-e)*total 을:
    * mode=None      → cash (= Phase 24 VM30, 현 FRIDAY 운영)
    * mode="BIL"     → 항상 BIL (공정 cash baseline, risk-free 벎)
    * mode="DUALMOM" → {IEF,TLT,GLD,BIL} 중 6m abs-momentum 최댓값 (BIL floor)
- realized_vol 은 total(=equity+cash+def) 일별로 계산 (Phase 24/25 동일).
- 방어 sleeve 는 매 event(분기 rebal / 월말 scaling)마다 목표가로 재조정,
  flat 10bps TC (ETF 는 유동적, size-slip 미적용).

모두 세전. VM30_cash 가 Phase 24 r1000-vm30 (26.96%/0.757/-42.41%) 재현해야 sanity OK.
"""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FRIDAY = Path(__file__).resolve().parent.parent.parent          # ~/Documents/friday
QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
ARCHIVE_FRI = FRIDAY / "archive" / "backtests"
ARCHIVE_QT = QUANT_TOOL / "archive" / "backtests"
for p in (ARCHIVE_FRI, QUANT_TOOL, ARCHIVE_QT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import factors
import backtest_advanced_risk as arb
from backtest import INITIAL_CAPITAL, get_static_fundamentals, load_sp500_tickers, load_spy
from backtest_v2 import load_vix
from backtest_phase2_pit import load_pit_membership
from pit_fundamentals import _load_all_caches_v2
from validate_cap20_hold20 import buy_ticker, current_value, enforce_cap, sell_full
from backtest_volmanaged_rocket import (
    metrics,
    compute_realized_vol,
    target_exposure,
    is_month_end,
    rescale_to_exposure,
)

# Phase 24 R1000 machinery
import backtest_phase24_r1000 as p24

logging.basicConfig(level=logging.WARNING)

START_DATE = p24.START_DATE
END_DATE = p24.END_DATE
BUY_TOP = p24.BUY_TOP
HOLD_TOP = p24.HOLD_TOP
TARGET_VOL = 0.30
SLIP_BASE_BPS = p24.SLIP_BASE_BPS          # flat 10bps for the defensive sleeve

DEF_MOM_LOOKBACK = 126                      # 6 months trading days
DEF_TC = SLIP_BASE_BPS / 10000.0           # 0.001

DEF_CACHE = Path.home() / ".quant_tool_cache" / "backtest" / "defensive_etfs.pkl"
BIL_CACHE = Path.home() / ".quant_tool_cache" / "backtest" / "bil_price.pkl"


# ============================================================
# Defensive sleeve
# ============================================================

def load_defensive_prices_with_bil() -> pd.DataFrame:
    """IEF/TLT/GLD/BIL adj-close, cached."""
    if DEF_CACHE.exists():
        with open(DEF_CACHE, "rb") as f:
            close = pickle.load(f)
    else:
        raw = yf.download(["IEF", "TLT", "GLD"], start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        close = raw["Close"].ffill().dropna(how="all")
        DEF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(DEF_CACHE, "wb") as f:
            pickle.dump(close, f)
    close = close.copy()
    if BIL_CACHE.exists():
        with open(BIL_CACHE, "rb") as f:
            bil = pickle.load(f)
    else:
        raw = yf.download("BIL", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        bil = raw["Close"]
        if isinstance(bil, pd.DataFrame):
            bil = bil.iloc[:, 0]
        bil = bil.ffill().dropna()
        with open(BIL_CACHE, "wb") as f:
            pickle.dump(bil, f)
    close["BIL"] = bil.reindex(close.index).ffill()
    return close


def _price_on_or_before(series: pd.Series, date) -> float | None:
    s = series.loc[:date].dropna()
    return float(s.iloc[-1]) if not s.empty else None


def choose_defensive(def_prices: pd.DataFrame, date, mode: str | None,
                     assets: list[str]) -> str | None:
    """Returns chosen defensive ticker or None (=cash)."""
    if mode is None:
        return None
    if mode in assets:
        return mode
    # DUALMOM: best 6m abs momentum among assets with positive return (BIL = floor)
    best_t, best_m = None, 0.0
    for t in assets:
        s = def_prices[t].loc[:date].dropna()
        if len(s) < DEF_MOM_LOOKBACK + 1:
            continue
        mom = s.iloc[-1] / s.iloc[-1 - DEF_MOM_LOOKBACK] - 1.0
        if mom > best_m:
            best_m, best_t = mom, t
    return best_t


# ============================================================
# Simulate — Phase 24 equity loop + defensive sleeve
# ============================================================

def simulate(
    prices, spy, vix, pit_membership, weight_schedule, sector_map,
    bal_cache, inc_cache, cf_cache, def_prices,
    *,
    target_vol: float | None,
    def_mode: str | None,
    def_assets: list[str],
) -> pd.Series:
    factors.set_simple_mode(True)
    prices_w = prices.loc[START_DATE:END_DATE]
    spy_w = spy.loc[START_DATE:END_DATE]
    close = arb.get_close_prices(prices_w)
    rebal_dates = arb.get_rebalance_dates(START_DATE, END_DATE)

    holdings: dict = {}
    cash = float(INITIAL_CAPITAL)
    def_ticker: str | None = None
    def_shares: float = 0.0

    history: list[dict] = []
    yearly_realized: dict[int, float] = {}
    daily_total_history: list[float] = []
    current_exposure_target = 1.0
    idx = 0

    def def_value(date) -> float:
        if def_ticker is None or def_shares <= 0:
            return 0.0
        p = _price_on_or_before(def_prices[def_ticker], date)
        return def_shares * p if p else 0.0

    def liquidate_def(date):
        nonlocal cash, def_ticker, def_shares
        v = def_value(date)
        if v > 0:
            cash += v * (1.0 - DEF_TC)
        def_ticker, def_shares = None, 0.0

    def set_def(date, target_value: float, chosen: str | None):
        nonlocal cash, def_ticker, def_shares
        if chosen is None or target_value <= 0:
            return
        p = _price_on_or_before(def_prices[chosen], date)
        if not p:
            return
        spend = min(cash, target_value)
        if spend <= 0:
            return
        def_shares = (spend * (1.0 - DEF_TC)) / p
        def_ticker = chosen
        cash -= spend

    dates_list = list(close.index)
    for i, date in enumerate(dates_list):
        next_date = dates_list[i + 1] if i + 1 < len(dates_list) else None

        # === 분기 리밸런싱 (Phase 24 R1000 picker) ===
        if idx < len(rebal_dates) and date >= rebal_dates[idx]:
            yearly_realized.setdefault(date.year, 0.0)
            liquidate_def(date)
            try:
                hold_picks, _, adv20_h = p24.select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    HOLD_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=p24.r1000_at_date,
                    apply_adv20_gate=True,
                )
                buy_picks, _, adv20_b = p24.select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    BUY_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=p24.r1000_at_date,
                    apply_adv20_gate=True,
                )
                adv20_map = {**adv20_h, **adv20_b}
            except Exception as e:
                logging.warning(f"pick fail @ {date}: {e}")
                hold_picks, buy_picks, adv20_map = [], [], {}

            hold_set = set(hold_picks)
            for ticker in [t for t in list(holdings) if t not in hold_set]:
                if ticker not in close.columns or pd.isna(close.at[date, ticker]):
                    holdings.pop(ticker, None)
                    continue
                shares = holdings[ticker]["shares"]
                price = close.at[date, ticker]
                order_val = shares * price
                cost = p24.slip_frac(order_val, adv20_map.get(ticker), True)
                proceeds, realized = sell_full(close, date, holdings, ticker, cost)
                cash += proceeds
                yearly_realized[date.year] += realized

            slots = BUY_TOP - len([t for t in holdings if t in hold_set])
            to_buy = [t for t in buy_picks if t not in holdings and t in close.columns][:slots]
            if slots > 0 and to_buy and cash > 0:
                total, current_equity = current_value(close, date, holdings, cash)
                target_equity = total * current_exposure_target
                buy_budget = max(0.0, min(cash, target_equity - current_equity))
                if buy_budget > 0:
                    alloc = buy_budget / len(to_buy)
                    spent = 0.0
                    for ticker in to_buy:
                        cost = p24.slip_frac(alloc, adv20_map.get(ticker), True)
                        spent += buy_ticker(close, date, holdings, ticker, alloc, cost)
                    cash -= spent
                    cash = max(cash, 0.0)

            cash, _, _, _ = enforce_cap(close, date, holdings, cash,
                                        SLIP_BASE_BPS / 10000.0, yearly_realized)

            # de-grossed 분을 방어자산으로 재배치
            total, equity = current_value(close, date, holdings, cash)
            target_def = max(0.0, total * (1.0 - current_exposure_target))
            chosen = choose_defensive(def_prices, date, def_mode, def_assets)
            set_def(date, target_def, chosen)
            idx += 1

        # === 월말 변동성 스케일링 ===
        if target_vol is not None and is_month_end(date, next_date):
            realized_vol = compute_realized_vol(daily_total_history)
            if realized_vol is not None:
                new_target = target_exposure(realized_vol, target_vol)
                if abs(new_target - current_exposure_target) > 0.05:
                    current_exposure_target = new_target
                    liquidate_def(date)
                    cash = rescale_to_exposure(
                        close, date, holdings, cash, current_exposure_target,
                        SLIP_BASE_BPS / 10000.0, yearly_realized,
                    )
                    total, equity = current_value(close, date, holdings, cash)
                    target_def = max(0.0, total * (1.0 - current_exposure_target))
                    chosen = choose_defensive(def_prices, date, def_mode, def_assets)
                    set_def(date, target_def, chosen)

        total, _ = current_value(close, date, holdings, cash)
        total += def_value(date)
        history.append({"date": date, "value": total})
        daily_total_history.append(total)

    return pd.Series(
        [h["value"] for h in history],
        index=[h["date"] for h in history],
    )


def main():
    print("=" * 118)
    print("Phase 26 (FRIDAY): R1000 + ADV20 $5M + size-slip + VM30 — Defensive Rotation 비교")
    print("=" * 118)
    print(f"기간 {START_DATE}~{END_DATE} | VM target {TARGET_VOL:.0%} | def-TC {DEF_TC:.3%}")
    print("참조 (Phase 24 r1000-vm30): 26.96% / Sh 0.757 / MDD -42.41% / Calmar 0.636 / Vol 34.0%")
    print()

    print("[1/3] 데이터 로드 ...")
    prices = p24.load_combined_prices_r1000()
    spy = load_spy(START_DATE, END_DATE)
    vix = load_vix(START_DATE, END_DATE)
    fund_static = get_static_fundamentals(load_sp500_tickers())
    sector_map = dict(zip(fund_static["ticker"], fund_static["sector"]))
    bal, inc, cf = _load_all_caches_v2()
    r1000_pit = p24.load_r1000_synthetic_pit_membership()
    def_prices = load_defensive_prices_with_bil()
    if isinstance(prices.columns, pd.MultiIndex):
        n_tickers = len(prices["Close"].columns)
    else:
        n_tickers = len(prices.columns)
    print(f"  prices {prices.shape[0]}r × {n_tickers}t | R1000 PIT {len(r1000_pit)} rebal | "
          f"def {list(def_prices.columns)}")

    print("\n[2/3] 백테스트 변종")
    scenarios = [
        ("VM30_cash",        TARGET_VOL, None,      ["IEF", "TLT", "GLD"]),
        ("VM30_BIL",         TARGET_VOL, "BIL",     ["IEF", "TLT", "GLD", "BIL"]),
        ("VM30_DUALMOM",     TARGET_VOL, "DUALMOM", ["IEF", "TLT", "GLD"]),
        ("VM30_DUALMOM_BIL", TARGET_VOL, "DUALMOM", ["IEF", "TLT", "GLD", "BIL"]),
    ]

    results = {}
    for name, tv, dm, assets in scenarios:
        print(f"\n  Running: {name:<18s} (def={dm}, assets={assets}) ...")
        curve = simulate(
            prices, spy, vix, r1000_pit, p24.SCHEDULE_C80_E20, sector_map,
            bal, inc, cf, def_prices,
            target_vol=tv, def_mode=dm, def_assets=assets,
        )
        m = metrics(curve)
        results[name] = m
        print(f"    연 {m['annual']*100:5.2f}%  Sharpe {m['sharpe']:.3f}  "
              f"MDD {m['mdd']*100:6.2f}%  Calmar {m['calmar']:.3f}  Vol {m['vol']*100:.1f}%")

    print(f"\n[3/3] 비교 (vs VM30_cash = 현 FRIDAY 운영)")
    print("=" * 118)
    print(f"{'변종':<18} | {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Vol':>6} | "
          f"{'ΔCAGR':>7} {'ΔSh':>7} {'ΔMDD':>7} {'ΔCalmar':>8}")
    print("-" * 118)
    ref = results["VM30_cash"]
    for name, _, _, _ in scenarios:
        m = results[name]
        print(f"{name:<18} | {m['annual']*100:>6.2f}% {m['sharpe']:>7.3f} {m['mdd']*100:>7.2f}% "
              f"{m['calmar']:>7.3f} {m['vol']*100:>5.1f}% | "
              f"{(m['annual']-ref['annual'])*100:>+5.2f}pp {m['sharpe']-ref['sharpe']:>+6.3f} "
              f"{(m['mdd']-ref['mdd'])*100:>+5.2f}pp {m['calmar']-ref['calmar']:>+7.3f}")

    print(f"\n=== 핵심: DUALMOM_BIL vs VM30_BIL (공정 cash baseline) ===")
    a, b = results["VM30_DUALMOM_BIL"], results["VM30_BIL"]
    print(f"  ΔCAGR {(a['annual']-b['annual'])*100:+.2f}pp | ΔSharpe {a['sharpe']-b['sharpe']:+.3f} | "
          f"ΔMDD {(a['mdd']-b['mdd'])*100:+.2f}pp | ΔCalmar {a['calmar']-b['calmar']:+.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FRIDAY / "archive" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"phase26_r1000_dualmom_{ts}.csv"
    rows = [{"variant": n, "def_mode": dm, "assets": "+".join(a), **results[n]}
            for n, _, dm, a in scenarios]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
