"""
backtest_phase26b_realsectors.py
================================
Phase 26b (FRIDAY): Re-run r1000-vm30 with a FULL-universe sector_map so the
backtest's sector cap matches LIVE screener.py behavior.

문제 (Phase 26 review):
  Phase 24 backtest 의 sector_map = SP500 static (503 names) 만. 유니버스의
  ~50-55%/quarter (non-SP500 R1000 names) 가 전부 sector="Unknown" → 단일 버킷
  cap 6 으로 묶임. live screener.py 는 모든 종목에 real sector (yfinance) 부여 →
  cap-6-per-real-sector 적용. 즉 backtest 와 live 가 다른 제약 → 27% 가 live 를
  충실히 예측 못 함 (backtest-optimistic 의심: Unknown lumping 이 momentum
  concentration 허용).

수정:
  sector_map 을 FMP company-screener 캐시 (live 와 동일 taxonomy) 로 전 유니버스
  커버 (1661/1707), SP500 static fallback. 나머지 ~46 (폐지주) 만 Unknown.
  그 외 모든 것 Phase 24 와 동일 (universe / ADV20 $5M / size-slip / VM30).

비교:
  baseline_sp500only_secmap : Phase 24 그대로 → 26.96% / 0.757 재현 (sanity)
  fixed_full_secmap         : live-faithful 숫자
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

FRIDAY = Path(__file__).resolve().parent.parent.parent
QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
ARCHIVE_FRI = FRIDAY / "archive" / "backtests"
ARCHIVE_QT = QUANT_TOOL / "archive" / "backtests"
for p in (ARCHIVE_FRI, QUANT_TOOL, ARCHIVE_QT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest import get_static_fundamentals, load_sp500_tickers, load_spy
from backtest_v2 import load_vix
from pit_fundamentals import _load_all_caches_v2
from backtest_volmanaged_rocket import metrics

import backtest_phase24_r1000 as p24

CAND_CACHE = Path.home() / ".quant_tool_cache" / "r1000_candidates.pkl"


def build_full_sector_map() -> tuple[dict, dict]:
    """Returns (sp500_only_map, full_map).
    full_map = FMP screener sector for every name (live taxonomy) + SP500 static fallback.
    """
    fs = get_static_fundamentals(load_sp500_tickers())
    sp500_map = dict(zip(fs["ticker"], fs["sector"]))

    with open(CAND_CACHE, "rb") as f:
        cand = pickle.load(f)
    screener_map = {d["symbol"]: d.get("sector") for d in cand if d.get("sector")}

    # full: screener sector primary (matches live yfinance), SP500 static fallback
    full_map = dict(sp500_map)
    for t, sec in screener_map.items():
        if sec:
            full_map[t] = sec
    return sp500_map, full_map


def run(sector_map: dict, prices, spy, vix, r1000_pit, bal, inc, cf) -> dict:
    curve, diag = p24.simulate(
        prices, spy, vix, r1000_pit,
        p24.SCHEDULE_C80_E20, sector_map, bal, inc, cf,
        universe_fn=p24.r1000_at_date,
        apply_adv20_gate=True,
        size_dependent_slip=True,
        target_vol=0.30,
    )
    return {**metrics(curve), **diag}


def main():
    print("=" * 110)
    print("Phase 26b (FRIDAY): r1000-vm30 — SP500-only sector_map vs FULL-universe sector_map")
    print("=" * 110)
    print("참조 (Phase 24 r1000-vm30): 26.96% / Sh 0.757 / MDD -42.41% / Calmar 0.636 / Vol 34.0%")

    print("\n[1/3] 데이터 로드 ...")
    prices = p24.load_combined_prices_r1000()
    spy = load_spy(p24.START_DATE, p24.END_DATE)
    vix = load_vix(p24.START_DATE, p24.END_DATE)
    bal, inc, cf = _load_all_caches_v2()
    r1000_pit = p24.load_r1000_synthetic_pit_membership()
    sp500_map, full_map = build_full_sector_map()

    union = set()
    for r in r1000_pit["tickers"]:
        union.update(t.strip() for t in r.split(","))
    cov_sp = sum(1 for t in union if t in sp500_map and sp500_map[t] and str(sp500_map[t]) != "nan")
    cov_full = sum(1 for t in union if t in full_map and full_map[t] and str(full_map[t]) != "nan")
    print(f"  sector coverage on union ({len(union)}): SP500-only={cov_sp} ({cov_sp/len(union)*100:.0f}%) | "
          f"full={cov_full} ({cov_full/len(union)*100:.0f}%)")

    print("\n[2/3] 백테스트 — 2 sector_map ...")
    print("  Running: baseline (SP500-only sector_map = Phase 24) ...")
    base = run(sp500_map, prices, spy, vix, r1000_pit, bal, inc, cf)
    print(f"    연 {base['annual']*100:5.2f}%  Sharpe {base['sharpe']:.3f}  MDD {base['mdd']*100:6.2f}%  "
          f"Calmar {base['calmar']:.3f}  Vol {base['vol']*100:.1f}%")

    print("  Running: fixed (FULL-universe sector_map = live-faithful) ...")
    fixed = run(full_map, prices, spy, vix, r1000_pit, bal, inc, cf)
    print(f"    연 {fixed['annual']*100:5.2f}%  Sharpe {fixed['sharpe']:.3f}  MDD {fixed['mdd']*100:6.2f}%  "
          f"Calmar {fixed['calmar']:.3f}  Vol {fixed['vol']*100:.1f}%")

    print("\n[3/3] 비교")
    print("=" * 110)
    print(f"{'variant':<28} | {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Vol':>6}")
    print("-" * 110)
    for name, m in [("baseline_sp500only_secmap", base), ("fixed_full_secmap", fixed)]:
        print(f"{name:<28} | {m['annual']*100:>6.2f}% {m['sharpe']:>7.3f} {m['mdd']*100:>7.2f}% "
              f"{m['calmar']:>7.3f} {m['vol']*100:>5.1f}%")
    print("-" * 110)
    print(f"{'Δ (fixed - baseline)':<28} | {(fixed['annual']-base['annual'])*100:>+5.2f}pp "
          f"{fixed['sharpe']-base['sharpe']:>+6.3f} {(fixed['mdd']-base['mdd'])*100:>+6.2f}pp "
          f"{fixed['calmar']-base['calmar']:>+7.3f} {(fixed['vol']-base['vol'])*100:>+5.1f}pp")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FRIDAY / "archive" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"phase26b_realsectors_{ts}.csv"
    pd.DataFrame([
        {"variant": "baseline_sp500only_secmap", **base},
        {"variant": "fixed_full_secmap", **fixed},
    ]).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
