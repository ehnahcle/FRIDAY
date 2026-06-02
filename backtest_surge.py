"""
backtest_surge.py
=================
FRIDAY SURGE 스크리너 검증 — point-in-time 백테스트.

질문: high SURGE-score 종목이 향후 5~7거래일 내 +25~30% 급등을 base rate 보다
      더 자주 달성하는가? (research: "급등 예측 ≠ 양(+) 수익" — 반드시 검증)

⚠️  3가지 HARD 제약 (결과 해석 시 필수):
  1. SURVIVORSHIP BIAS — universe = 현재 FMP snapshot. 폐지/파산 종목(small-cap
     biotech에 흔함) 누락 → 결과는 optimistic UPPER BOUND.
  2. OHLCV 신호만 검증 가능 — fuel(short float)/catalyst(analyst)는 현재 snapshot
     이라 historical 복원 불가. 따라서 readiness+explosiveness(=pre_score)만 검증.
     full SURGE_score 검증은 historical short-interest/analyst 데이터 확보 후(v2).
  3. point-in-time 무결성 OK — 각 날짜 T 에서 prices.loc[:T] 로 truncate 후 동일한
     compute_readiness 호출 → feature look-ahead 없음. forward return 만 미래 참조.

방법: 주간(every 5거래일) 날짜 T 마다 universe 스코어링 → gate → top-N 의 forward
     7d high 가 +25%/+30% 도달했는지 → top-N hit rate vs base rate vs score 분위.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import surge_screener as ss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_backtest(
    n_sample: int = 500,
    period: str = "3y",
    test_every: int = 5,
    top_n: int = 10,
    horizon: int = 7,
    surge_thresholds=(0.25, 0.30),
) -> dict:
    # 1. universe 샘플 (deterministic stride — 재현성)
    universe = ss.get_smallcap_universe()
    if not universe:
        logger.error("universe 비어있음")
        return {}
    stride = max(1, len(universe) // n_sample)
    sample = universe[::stride][:n_sample]
    logger.info(f"universe {len(universe)} → sample {len(sample)} (stride {stride})")

    # 2. 장기 가격 (PIT feature 계산 + forward return 용)
    prices = ss.get_prices(sample, period=period)
    if prices.empty:
        logger.error("가격 데이터 없음")
        return {}
    close = ss._field(prices, "Close")
    high = ss._field(prices, "High")
    if close is None or high is None:
        logger.error("price fields 없음")
        return {}
    dates = close.index
    logger.info(f"price history: {len(dates)} bars, {close.shape[1]} tickers "
                f"({dates[0].date()} ~ {dates[-1].date()})")

    # 3. 테스트 날짜: 252d warmup 후 ~ (끝 - horizon)
    start_i, end_i = 252, len(dates) - horizon - 1
    if end_i <= start_i:
        logger.error(f"history 부족: {len(dates)} bars (252 warmup + {horizon} horizon 필요)")
        return {}
    test_idxs = list(range(start_i, end_i, test_every))
    logger.info(f"테스트 날짜 {len(test_idxs)}개 ({dates[start_i].date()} ~ {dates[end_i].date()}, every {test_every}d)")

    # 4. 각 T 에서 PIT 스코어링 + forward return
    records = []
    for n, ti in enumerate(test_idxs):
        T = dates[ti]
        if n % 20 == 0:
            logger.info(f"  backtest {n}/{len(test_idxs)} @ {T.date()}")
        prices_T = prices.loc[:T]                      # truncate — look-ahead 차단
        feat = ss.compute_readiness(prices_T)
        if feat.empty:
            continue
        # 동일 gate 적용
        feat = feat[(feat["price"] >= ss.MIN_PRICE) & (feat["adv20"] >= ss.MIN_ADV20)]
        feat = feat[feat["capacity_ok"].fillna(False)]
        feat = feat[feat["ret_5d"].fillna(0) <= ss.CHASE_RET_5D]
        if feat.empty:
            continue
        feat = feat.copy()
        feat["pre_score"] = 0.5 * feat["readiness"].fillna(0) + 0.5 * feat["explosiveness"].fillna(0)

        # forward return: T+1 ~ T+horizon
        fwd_close = close.iloc[ti + 1: ti + 1 + horizon]
        fwd_high = high.iloc[ti + 1: ti + 1 + horizon]
        for _, r in feat.iterrows():
            tk = r["ticker"]
            c0 = r["price"]
            if c0 is None or c0 <= 0 or tk not in fwd_high.columns:
                continue
            fh = fwd_high[tk].dropna()
            fc = fwd_close[tk].dropna()
            if len(fh) < horizon - 1 or len(fc) < 1:   # forward 데이터 부족
                continue
            max_high_gain = float(fh.max() / c0 - 1)         # 급등 "발생" (intraday touch)
            max_close_gain = float(fc.max() / c0 - 1)        # 종가 기준 최고
            ret_end = float(fc.iloc[-1] / c0 - 1)            # T+horizon 종가 수익
            records.append({
                "date": T, "ticker": tk, "pre_score": r["pre_score"],
                "readiness": r["readiness"], "explosiveness": r["explosiveness"],
                "max_high_gain": max_high_gain, "max_close_gain": max_close_gain,
                "ret_end": ret_end,
            })

    if not records:
        logger.error("백테 레코드 0개")
        return {}
    df = pd.DataFrame(records)
    logger.info(f"백테 레코드 {len(df)}개 ({df['date'].nunique()} dates × ~{len(df)//max(df['date'].nunique(),1)} names/date)")

    # 5. 분석
    out = {"n_records": len(df), "n_dates": int(df["date"].nunique()),
           "period": period, "horizon": horizon, "top_n": top_n,
           "thresholds": {}, "quintiles": {}}

    for thr in surge_thresholds:
        df[f"hit_{int(thr*100)}"] = (df["max_high_gain"] >= thr).astype(int)
        col = f"hit_{int(thr*100)}"
        base = float(df[col].mean())
        # top-N per date by pre_score
        top = df.groupby("date", group_keys=False).apply(lambda g: g.nlargest(top_n, "pre_score"))
        top_rate = float(top[col].mean())
        out["thresholds"][f"{int(thr*100)}%"] = {
            "base_rate": base, "topN_rate": top_rate,
            "lift": (top_rate / base) if base > 0 else float("nan"),
        }

    # score 5분위 monotonicity (가장 중요한 테스트)
    df["q"] = pd.qcut(df["pre_score"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"], duplicates="drop")
    for thr in surge_thresholds:
        col = f"hit_{int(thr*100)}"
        qstat = df.groupby("q", observed=True)[col].mean()
        out["quintiles"][f"{int(thr*100)}%"] = {str(k): float(v) for k, v in qstat.items()}
    # forward 수익 분위별
    out["fwd_return_by_q"] = {str(k): float(v) for k, v in
                              df.groupby("q", observed=True)["ret_end"].mean().items()}
    out["max_high_gain_by_q"] = {str(k): float(v) for k, v in
                                 df.groupby("q", observed=True)["max_high_gain"].mean().items()}

    # 저장
    out_csv = Path(__file__).resolve().parent / "surge_backtest_records.csv"
    df.to_csv(out_csv, index=False)
    out["records_csv"] = str(out_csv.name)
    return out


def _print(out: dict) -> None:
    if not out:
        print("\n백테 실패 — 결과 없음.")
        return
    print("\n" + "=" * 78)
    print("  FRIDAY SURGE 백테스트 — pre_score (readiness+explosiveness, OHLCV only)")
    print("=" * 78)
    print(f"  records={out['n_records']}  dates={out['n_dates']}  period={out['period']}  "
          f"horizon={out['horizon']}d  top_n={out['top_n']}")
    print("-" * 78)
    print("  [급등 HIT RATE: forward 7d high 가 임계 도달]")
    for thr, s in out["thresholds"].items():
        print(f"    +{thr:<4}  base={s['base_rate']:6.2%}   top{out['top_n']}={s['topN_rate']:6.2%}   "
              f"lift={s['lift']:.2f}x")
    print("-" * 78)
    print("  [pre_score 5분위별 급등 hit rate — monotonic 이어야 신호가 real]")
    for thr, q in out["quintiles"].items():
        line = "  ".join(f"{k}={v:.1%}" for k, v in q.items())
        print(f"    +{thr}: {line}")
    print("-" * 78)
    print("  [5분위별 forward 종가수익 / 평균 max-high-gain]")
    fr = out["fwd_return_by_q"]; mg = out["max_high_gain_by_q"]
    print("    ret_end : " + "  ".join(f"{k}={v:+.1%}" for k, v in fr.items()))
    print("    maxhigh : " + "  ".join(f"{k}={v:+.1%}" for k, v in mg.items()))
    print("=" * 78)
    print("⚠️  CAVEATS: (1) SURVIVORSHIP BIAS — 현 universe만 → optimistic UPPER BOUND.")
    print("    (2) OHLCV 신호(readiness+explosiveness)만 검증 — fuel/catalyst 미포함.")
    print("    (3) base rate 자체가 capacity-gated universe 기준(이미 high-vol 종목들).")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FRIDAY SURGE 백테스트")
    ap.add_argument("--sample", type=int, default=500, help="universe 샘플 크기")
    ap.add_argument("--period", default="3y", help="가격 history (yfinance period)")
    ap.add_argument("--every", type=int, default=5, help="테스트 간격 (거래일)")
    ap.add_argument("--top", type=int, default=10, help="top-N hit rate")
    ap.add_argument("--horizon", type=int, default=7, help="forward 윈도우 (거래일)")
    args = ap.parse_args()

    out = run_backtest(
        n_sample=args.sample, period=args.period, test_every=args.every,
        top_n=args.top, horizon=args.horizon,
    )
    _print(out)
