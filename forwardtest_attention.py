"""
forwardtest_attention.py
========================
"AI attention" 엣지 정량 검증 — attention-필터 top-N 이 raw top-N 을 이기는가?

⚠️  METHODOLOGY (2단 제약):
  1. Claude(LLM) 자체는 forward-test 불가 — 2023 헤드라인 주고 "관심 빌딩중?" 물으면
     LLM이 이후 결과를 training data로 "알" 수 있음(hindsight 오염).
  2. news_spike(뉴스 기반)도 forward-test 불가 — 현 FMP 플랜: 과거 날짜범위 news(from/to)
     는 402 premium-gated, 최근 news 도 ~1-2개월만(forward window 없음). 확인됨(2026-06-02).
  → 따라서 **look-ahead-safe 하고 historical 재구성 가능한 attention proxy = RVOL**
    (relative volume = 오늘 거래량/50일평균). Da-Engelberg-Gao "In Search of Attention"
    이 abnormal volume 을 attention proxy 로 사용 — 학술적으로 정당. 순수 OHLCV 라 PIT 완전.
  → 이게 엣지 있으면 attention "개념"(관심↑→급등)이 forward 검증된 것. LIVE 모듈은
    그 위에 news+Claude narrative 를 더하지만 그 레이어 자체는 backtest 불가(LIVE 전용).

방법 (backtest_surge_hardened 머신 재사용, survivorship-inclusive OHLC):
  각 날짜 T → gated universe pre_score 랭킹 → pool = top_N(스크리너가 보여주는 후보).
  pool 내에서 PIT news_spike(attention) 계산.
  비교 (1) RAW top_K (pre_score) vs ATTN top_K (attention/blend) 의 forward hit25 + net.
       (2) pool 내 attention 분위별 forward 성과 (monotonic 이면 attention real).
exit: 고정 +30%/-20% (task1 best 근방, raw/attn 동일 적용이라 상대비교엔 무관).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import surge_screener as ss
from backtest_surge_hardened import (
    CACHE, FMP_API_KEY, FMP_BASE, fetch_ohlc, get_delisted, simulate_trade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 뉴스 (window 전체 1회 fetch → 날짜별 PIT 필터)
# ============================================================
def fetch_news_window(symbol: str, from_date: str, cache_days: float = 14) -> Optional[pd.Series]:
    """심볼의 window 내 모든 뉴스 → 일자별 count Series (date-indexed). PIT 재구성용."""
    p = CACHE / f"newshist_{symbol}_{from_date}.pkl"
    if p.exists() and (datetime.now().timestamp() - p.stat().st_mtime) < cache_days * 86400:
        with open(p, "rb") as f:
            return pickle.load(f)
    try:
        r = requests.get(f"{FMP_BASE}/news/stock",
                         params={"symbols": symbol, "from": from_date, "limit": 1000,
                                 "apikey": FMP_API_KEY}, timeout=25)
        d = r.json() if r.status_code == 200 else []
    except Exception:
        d = []
    if not isinstance(d, list) or not d:
        ser = pd.Series(dtype=int)
    else:
        dts = []
        for n in d:
            pd_ = (n.get("publishedDate") or "")[:10]
            try:
                dts.append(pd.to_datetime(pd_))
            except Exception:
                pass
        ser = pd.Series(1, index=pd.DatetimeIndex(dts)).groupby(level=0).sum().sort_index() \
            if dts else pd.Series(dtype=int)
    with open(p, "wb") as f:
        pickle.dump(ser, f)
    return ser


def news_spike_asof(counts: pd.Series, T: pd.Timestamp) -> float:
    """T 기준 최근 7일 뉴스건수 / (8~30일 baseline ×7). look-ahead 없음(<=T만)."""
    if counts is None or len(counts) == 0:
        return 0.0
    recent = counts[(counts.index > T - timedelta(days=7)) & (counts.index <= T)].sum()
    prior = counts[(counts.index > T - timedelta(days=30)) & (counts.index <= T - timedelta(days=7))].sum()
    base7 = (prior / 23 * 7) if prior > 0 else 0.3
    return float(recent / base7) if base7 > 0 else float(recent / 0.3)


# ============================================================
# 메인
# ============================================================
def run(live_sample=400, delisted_cap=300, period_from="2022-06-01",
        test_every=5, pool_n=20, top_k=10, horizon=7,
        target=0.30, stop=0.20, slip_bps=100, w_blend=0.5):
    universe = ss.get_smallcap_universe()
    stride = max(1, len(universe) // live_sample)
    live = universe[::stride][:live_sample]
    del_syms = [r["symbol"] for r in get_delisted(period_from, delisted_cap)]
    all_syms = list(dict.fromkeys(live + del_syms))
    logger.info(f"universe: {len(all_syms)} symbols (live {len(live)} + delisted {len(del_syms)})")

    # OHLC (hardened 캐시 공유 → 대부분 hit). 뉴스는 historical 불가 → RVOL proxy 사용.
    ohlc = {}
    for i, sym in enumerate(all_syms):
        if i % 50 == 0:
            logger.info(f"  OHLC {i}/{len(all_syms)}")
        df = fetch_ohlc(sym, period_from)
        if df is not None and len(df) >= 80:
            ohlc[sym] = df
    logger.info(f"OHLC 확보 {len(ohlc)} (attention proxy = RVOL, PIT)")
    if len(ohlc) < 20:
        logger.error("데이터 부족"); return {}

    ref = max(ohlc.values(), key=len)
    cal = ref.index
    start_i, end_i = 252, len(cal) - horizon - 1
    test_dates = [cal[i] for i in range(start_i, end_i, test_every)]
    logger.info(f"테스트 날짜 {len(test_dates)}개")

    # 날짜별: gate+score → pool(top_N by pre_score) → attention → forward outcome
    pool_rows = []
    for n, T in enumerate(test_dates):
        if n % 20 == 0:
            logger.info(f"  fwdtest {n}/{len(test_dates)} @ {T.date()}")
        cands = []
        for sym, df in ohlc.items():
            hist = df.loc[:T]
            if len(hist) < 60:
                continue
            feat = ss.compute_features_for_series(*[hist.tail(300)[c] for c in ["Close", "High", "Low", "Volume"]])
            if feat is None or not (feat["price"] >= ss.MIN_PRICE and feat["adv20"] >= ss.MIN_ADV20):
                continue
            if not feat["capacity_ok"]:
                continue
            if not (pd.isna(feat["ret_5d"]) or feat["ret_5d"] <= ss.CHASE_RET_5D):
                continue
            cands.append((sym, 0.5 * feat["readiness"] + 0.5 * feat["explosiveness"], feat))
        if len(cands) < pool_n:
            continue
        cands.sort(key=lambda x: -x[1])
        pool = cands[:pool_n]                       # 스크리너가 보여주는 후보 pool

        for sym, pre, feat in pool:
            atn = float(feat["rvol"]) if not pd.isna(feat.get("rvol")) else 0.0  # RVOL = attention proxy (PIT)
            fwd = ohlc[sym].loc[T:].iloc[1:1 + horizon]
            if fwd.empty:
                continue
            c0 = feat["price"]
            mhg = float(fwd["High"].max() / c0 - 1)
            tr = simulate_trade(fwd, target, stop, slip_bps)
            if tr is None:
                continue
            pool_rows.append({"date": T, "ticker": sym, "pre_score": pre, "attention": atn,
                              "hit25": int(mhg >= 0.25), "net_ret": tr["net_ret"]})

    if not pool_rows:
        logger.error("pool 레코드 0"); return {}
    df = pd.DataFrame(pool_rows)
    df.to_csv(Path(__file__).resolve().parent / "surge_attention_fwdtest.csv", index=False)

    # blend score: pre_score 와 attention 각각 날짜내 rank-normalize 후 blend
    def _znorm(g, col):
        r = g[col].rank(pct=True)
        return r
    df["pre_rk"] = df.groupby("date")["pre_score"].rank(pct=True)
    df["atn_rk"] = df.groupby("date")["attention"].rank(pct=True)
    df["blend"] = (1 - w_blend) * df["pre_rk"] + w_blend * df["atn_rk"]

    def pick_top(score_col):
        t = df.groupby("date", group_keys=False).apply(lambda g: g.nlargest(top_k, score_col))
        return float(t["hit25"].mean()), float(t["net_ret"].mean()), float((t["net_ret"] > 0).mean())

    raw = pick_top("pre_score")       # 기존 스크리너 방식
    attn = pick_top("attention")      # attention 만으로 재정렬
    blend = pick_top("blend")         # pre_score + attention blend

    # pool 내 attention 분위별 (attention 자체가 forward 신호인가)
    df["aq"] = pd.qcut(df["attention"].rank(method="first"), 5,
                       labels=["A1(low)", "A2", "A3", "A4", "A5(high)"])
    aq_hit = {str(k): float(v) for k, v in df.groupby("aq", observed=True)["hit25"].mean().items()}
    aq_net = {str(k): float(v) for k, v in df.groupby("aq", observed=True)["net_ret"].mean().items()}

    return {
        "n_pool_obs": len(df), "n_dates": int(df["date"].nunique()), "pool_n": pool_n, "top_k": top_k,
        "exit": f"+{target:.0%}/-{stop:.0%}/{horizon}d, {slip_bps}bps",
        "raw":   {"hit25": raw[0],   "net": raw[1],   "win": raw[2]},
        "attn":  {"hit25": attn[0],  "net": attn[1],  "win": attn[2]},
        "blend": {"hit25": blend[0], "net": blend[1], "win": blend[2], "w": w_blend},
        "attn_quintile_hit25": aq_hit, "attn_quintile_net": aq_net,
        "pool_attn_median": float(df["attention"].median()),
    }


def _print(o: dict):
    if not o:
        print("\n실패."); return
    print("\n" + "=" * 78)
    print("  FRIDAY SURGE — ATTENTION 엣지 forward-test (proxy=news_spike, look-ahead-safe)")
    print("=" * 78)
    print(f"  pool_obs={o['n_pool_obs']}  dates={o['n_dates']}  pool=top{o['pool_n']}  pick=top{o['top_k']}  exit={o['exit']}")
    print(f"  attention proxy = RVOL (PIT, median in pool={o['pool_attn_median']:.2f})")
    print("-" * 78)
    print("  [top_K 선택 방식별 forward 성과 — attention(RVOL)이 raw를 이기면 엣지 있음]")
    for lab, k in [("RAW (pre_score)", "raw"), ("RVOL-attn only", "attn"), (f"BLEND (w={o['blend']['w']})", "blend")]:
        s = o[k]
        print(f"    {lab:<22} hit25={s['hit25']:.2%}  net={s['net']:+.2%}  win={s['win']:.1%}")
    print("-" * 78)
    print("  [pool 내 RVOL(attention) 분위별 — monotonic이면 attention 자체가 forward 신호]")
    print("    hit25: " + "  ".join(f"{k}={v:.1%}" for k, v in o["attn_quintile_hit25"].items()))
    print("    net  : " + "  ".join(f"{k}={v:+.2%}" for k, v in o["attn_quintile_net"].items()))
    print("=" * 78)
    print("⚠️  proxy=RVOL만(look-ahead-safe, 학술적 attention measure). news_spike=FMP 플랜상 historical")
    print("    불가(402), Claude narrative=hindsight 오염 → 둘 다 forward-test 불가, LIVE 전용.")
    print("    이 결과는 attention '개념'(관심↑→급등)의 forward 검증.")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SURGE attention 엣지 forward-test")
    ap.add_argument("--live", type=int, default=400)
    ap.add_argument("--delisted", type=int, default=300)
    ap.add_argument("--from", dest="from_date", default="2022-06-01")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--w", type=float, default=0.5, help="blend 내 attention 비중")
    a = ap.parse_args()
    out = run(live_sample=a.live, delisted_cap=a.delisted, period_from=a.from_date,
              test_every=a.every, pool_n=a.pool, top_k=a.topk, horizon=a.horizon, w_blend=a.w)
    _print(out)
