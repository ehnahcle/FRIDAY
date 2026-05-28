"""
backtest_phase24_sensitivity.py
===============================
Phase 24 sensitivity grid: slippage α × ADV20 threshold.
가설 — α=10, ADV20=$5M 단일 점 채택 위험 (Phase 19e fragile-spike 교훈).
인접 grid 매끄러우면 robust, 단일 셀만 spike 면 fragile.

Grid (9 cells):
  α ∈ {5, 10, 20}             — slip_bps = 10 + α · (order_value/ADV20) · 100
  ADV20 ≥ {$2M, $5M, $10M}    — liquidity gate floor

핵심 모드: r1000 (gate ON + size-slip ON), 가중치 C80_E20 (Phase 16).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
sys.path.insert(0, str(QUANT_TOOL))
sys.path.insert(0, str(QUANT_TOOL / "archive" / "backtests"))

import backtest_phase24_r1000 as bt24
from backtest import get_static_fundamentals, load_sp500_tickers, load_spy
from backtest_v2 import load_vix
from backtest_phase2_pit import load_pit_membership
from backtest_volmanaged_rocket import metrics
from pit_fundamentals import _load_all_caches_v2

logging.basicConfig(level=logging.WARNING)

SLIP_GRID  = [5, 10, 20]
ADV20_GRID = [2_000_000, 5_000_000, 10_000_000]


def run_cell(alpha: float, adv20_min: float, ctx: dict) -> dict:
    # 모듈 상수 override
    bt24.SLIP_ALPHA = alpha
    bt24.ADV20_MIN = adv20_min
    curve, diag = bt24.simulate(
        ctx["prices"], ctx["spy"], ctx["vix"], ctx["r1000_pit"],
        bt24.SCHEDULE_C80_E20, ctx["sector_map"], *ctx["pit_caches"],
        universe_fn=bt24.r1000_at_date,
        apply_adv20_gate=True,
        size_dependent_slip=True,
    )
    m = metrics(curve)
    return {"alpha": alpha, "adv20_M": adv20_min / 1e6, **m, **diag}


def main():
    print("=" * 110)
    print(f"Phase 24 sensitivity — α × ADV20 grid ({len(SLIP_GRID)} × {len(ADV20_GRID)} = {len(SLIP_GRID)*len(ADV20_GRID)} cells)")
    print("=" * 110)

    print("\n[1/3] 데이터 로드 ...")
    prices = bt24.load_combined_prices_r1000()
    spy = load_spy(bt24.START_DATE, bt24.END_DATE)
    vix = load_vix(bt24.START_DATE, bt24.END_DATE)
    fund_static = get_static_fundamentals(load_sp500_tickers())
    sector_map = dict(zip(fund_static["ticker"], fund_static["sector"]))
    bal, inc, cf = _load_all_caches_v2()
    r1000_pit = bt24.load_r1000_synthetic_pit_membership()

    ctx = {
        "prices": prices, "spy": spy, "vix": vix,
        "sector_map": sector_map,
        "pit_caches": (bal, inc, cf),
        "r1000_pit": r1000_pit,
    }

    print(f"\n[2/3] {len(SLIP_GRID) * len(ADV20_GRID)}-cell grid 실행 ...")
    results: list[dict] = []
    for alpha in SLIP_GRID:
        for adv in ADV20_GRID:
            name = f"α{alpha}_${int(adv/1e6)}M"
            print(f"\n  Running: {name}")
            r = run_cell(alpha, adv, ctx)
            r["name"] = name
            results.append(r)
            print(f"    연 {r['annual']*100:5.2f}%  Sharpe {r['sharpe']:.3f}  "
                  f"MDD {r['mdd']*100:6.2f}%  Calmar {r['calmar']:.3f}  "
                  f"Vol {r['vol']*100:.1f}%  | slip {r['mean_slip_bps']:.1f}bps  "
                  f"trades {r['n_trades']}")

    # 표
    print(f"\n[3/3] Sensitivity 표")
    print(f"  Phase 16 baseline (SP500): 18.09% / Sh 0.594 / MDD -40.09%")
    print(f"  Phase 24 base cell (α=10, $5M): 26.82% / Sh 0.707 / MDD -50.09%")
    print("=" * 110)

    print(f"\n  CAGR (%):")
    print(f"  {'α \\ ADV20':<12} " + " ".join(f"{adv/1e6:>6.0f}M" for adv in ADV20_GRID))
    for alpha in SLIP_GRID:
        row = []
        for adv in ADV20_GRID:
            r = next((x for x in results if x["alpha"] == alpha and abs(x["adv20_M"] - adv/1e6) < 0.01), None)
            row.append(f"{r['annual']*100:>6.2f}%" if r else "  N/A  ")
        print(f"  α={alpha:<10} " + " ".join(row))

    print(f"\n  Sharpe:")
    print(f"  {'α \\ ADV20':<12} " + " ".join(f"{adv/1e6:>6.0f}M" for adv in ADV20_GRID))
    for alpha in SLIP_GRID:
        row = []
        for adv in ADV20_GRID:
            r = next((x for x in results if x["alpha"] == alpha and abs(x["adv20_M"] - adv/1e6) < 0.01), None)
            row.append(f"{r['sharpe']:>7.3f}" if r else "  N/A  ")
        print(f"  α={alpha:<10} " + " ".join(row))

    print(f"\n  Calmar:")
    print(f"  {'α \\ ADV20':<12} " + " ".join(f"{adv/1e6:>6.0f}M" for adv in ADV20_GRID))
    for alpha in SLIP_GRID:
        row = []
        for adv in ADV20_GRID:
            r = next((x for x in results if x["alpha"] == alpha and abs(x["adv20_M"] - adv/1e6) < 0.01), None)
            row.append(f"{r['calmar']:>7.3f}" if r else "  N/A  ")
        print(f"  α={alpha:<10} " + " ".join(row))

    print(f"\n  Mean slippage (bps):")
    print(f"  {'α \\ ADV20':<12} " + " ".join(f"{adv/1e6:>6.0f}M" for adv in ADV20_GRID))
    for alpha in SLIP_GRID:
        row = []
        for adv in ADV20_GRID:
            r = next((x for x in results if x["alpha"] == alpha and abs(x["adv20_M"] - adv/1e6) < 0.01), None)
            row.append(f"{r['mean_slip_bps']:>6.1f} " if r else "  N/A  ")
        print(f"  α={alpha:<10} " + " ".join(row))

    print(f"\n해석 가이드:")
    print(f"  - α 인접 셀 사이 ΔCAGR ≤ 1pp / ΔSharpe ≤ 0.05 면 plateau (robust)")
    print(f"  - 단일 셀만 spike → fragile (Ph19e 교훈) — 채택 X")
    print(f"  - α 증가 → CAGR 단조 감소, Sharpe 함께 감소면 'slippage 가 알파 누적 capture'")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = QUANT_TOOL / "archive" / "results" / f"phase24_sensitivity_{ts}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
