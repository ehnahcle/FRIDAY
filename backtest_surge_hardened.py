"""
backtest_surge_hardened.py
==========================
SURGE 백테 강화판 — survivorship 제거 + slippage + 현실적 exit 모델.

기존 backtest_surge.py 대비 개선:
  1. SURVIVORSHIP 제거 — universe = 현 live-sample + **delisted 종목**(FMP delisted-companies).
     가격도 yfinance 대신 **FMP historical-price-eod/full** 통일 (delisted 종목 OHLC 확보).
     죽은 종목은 폐지 시점까지만 관측에 기여 → 폐지 직전 crash 가 신호를 정당하게 penalize.
  2. SLIPPAGE/비용 — round-trip 비용(bps) 차감. small-cap 현실 반영.
  3. 현실적 EXIT 모델 — entry = T+1 시가, exit = target(+X%) / stop(-Y%) / horizon timeout 중 먼저.
     max-high(체결 불가) 대신 실제 체결 가능 net return 산출. (같은 날 target&stop 동시 → stop 우선, 보수적)
  4. (옵션) CATALYST overlay — FMP grades-historical(날짜별 analyst rating)로 PIT catalyst 복원,
     OHLCV 신호에 catalyst 추가 시 개선되는지 검증 (--catalyst).

여전한 제약: (a) FMP delisted 가격 커버리지 불완전(확보된 것만), (b) fuel(short float) historical 미복원,
  (c) 단일 regime, (d) 시가 체결/target·stop 동일가 가정은 근사.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

import surge_screener as ss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# .env 직접 로드 (harness os.environ shadow 회피)
_ENV = dotenv_values(str(Path(__file__).resolve().parent / ".env"))
FMP_API_KEY = _ENV.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
CACHE = Path.home() / ".friday_cache" / "surge"
CACHE.mkdir(parents=True, exist_ok=True)


# ============================================================
# FMP 헬퍼 (OHLC + delisted + grades) — surge 캐시 격리
# ============================================================
def _get(ep: str, cache_name: str, cache_days: float = 7, **params):
    p = CACHE / f"{cache_name}.pkl"
    if p.exists() and (datetime.now().timestamp() - p.stat().st_mtime) < cache_days * 86400:
        with open(p, "rb") as f:
            return pickle.load(f)
    try:
        r = requests.get(f"{FMP_BASE}/{ep}", params={**params, "apikey": FMP_API_KEY}, timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception as e:
        logger.debug(f"FMP {ep} failed: {e}")
        return None
    with open(p, "wb") as f:
        pickle.dump(d, f)
    return d


def fetch_ohlc(symbol: str, from_date: str) -> Optional[pd.DataFrame]:
    """FMP historical-price-eod/full → date-indexed OHLCV DataFrame (live+delisted 공통)."""
    d = _get("historical-price-eod/full", f"ohlc_{symbol}_{from_date}", cache_days=7,
             symbol=symbol, **{"from": from_date})
    if not d:
        return None
    rows = d.get("historical") if isinstance(d, dict) else d
    if not isinstance(rows, list) or not rows:
        return None
    df = pd.DataFrame(rows)
    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]].rename(columns=str.capitalize)


def get_delisted(start: str, max_n: int, cache_days: float = 30) -> List[Dict]:
    """delistedDate >= start, 주요 거래소, 미국 보통주. 최대 max_n."""
    out = []
    for page in range(0, 40):
        d = _get("delisted-companies", f"delisted_p{page}", cache_days=cache_days, page=page, limit=100)
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        if len(d) < 100:
            break
    keep = []
    for r in out:
        dd = r.get("delistedDate") or ""
        ex = (r.get("exchange") or "").upper()
        sym = r.get("symbol") or ""
        if dd >= start and any(x in ex for x in ("NASDAQ", "NYSE", "AMEX")) and sym and "." not in sym:
            keep.append(r)
    logger.info(f"delisted in window (>= {start}, major exch): {len(keep)} (cap {max_n})")
    return keep[:max_n]


def fetch_grades_asof(symbol: str) -> Optional[pd.DataFrame]:
    """grades-historical → 날짜별 analyst rating count. PIT catalyst 복원용."""
    d = _get("grades-historical", f"grades_{symbol}", cache_days=14, symbol=symbol, limit=200)
    if not isinstance(d, list) or not d:
        return None
    df = pd.DataFrame(d)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def _analyst_score_from_grades(row: pd.Series) -> float:
    sb = row.get("analystRatingsStrongBuy", 0) or 0
    b = row.get("analystRatingsBuy", 0) or 0
    h = row.get("analystRatingsHold", 0) or 0
    s = row.get("analystRatingsSell", 0) or 0
    ss_ = row.get("analystRatingsStrongSell", 0) or 0
    tot = sb + b + h + s + ss_
    return (sb * 100 + b * 75 + h * 50 + s * 25) / tot if tot > 0 else 50.0


# ============================================================
# Exit 시뮬레이션 (entry T+1 시가 → target/stop/timeout)
# ============================================================
def simulate_trade(fwd: pd.DataFrame, target: float, stop: float, slip_bps: float) -> Optional[Dict]:
    """fwd = entry일(T+1)부터 horizon개 OHLC. net return + 보유일 + 청산사유."""
    if fwd.empty:
        return None
    entry = float(fwd["Open"].iloc[0])
    if entry <= 0:
        return None
    tgt_px, stop_px = entry * (1 + target), entry * (1 - stop)
    exit_px, reason, hold = None, "timeout", len(fwd)
    for i in range(len(fwd)):
        hi, lo = float(fwd["High"].iloc[i]), float(fwd["Low"].iloc[i])
        hit_stop = lo <= stop_px
        hit_tgt = hi >= tgt_px
        if hit_stop and hit_tgt:        # 동일 날 → stop 우선(보수적)
            exit_px, reason, hold = stop_px, "stop", i + 1; break
        if hit_stop:
            exit_px, reason, hold = stop_px, "stop", i + 1; break
        if hit_tgt:
            exit_px, reason, hold = tgt_px, "target", i + 1; break
    if exit_px is None:
        exit_px = float(fwd["Close"].iloc[-1])
    gross = exit_px / entry - 1
    net = gross - slip_bps / 10000.0      # round-trip 비용
    return {"net_ret": net, "gross_ret": gross, "hold_days": hold, "exit_reason": reason}


# ============================================================
# 메인
# ============================================================
def run(live_sample=400, delisted_cap=300, period_from="2022-06-01",
        test_every=5, top_n=10, horizon=7, target=0.20, stop=0.10,
        slip_bps=100, use_catalyst=False, atr_target=0.0, atr_stop=0.0):
    # ATR 모드: target/stop = ATR%의 배수 (종목별 vol-adaptive). 둘 다 >0 이어야 활성.
    atr_mode = atr_target > 0 and atr_stop > 0
    # 1. universe = live-sample + delisted
    universe = ss.get_smallcap_universe()
    stride = max(1, len(universe) // live_sample)
    live = universe[::stride][:live_sample]
    delisted = get_delisted(period_from, delisted_cap)
    del_syms = [r["symbol"] for r in delisted]
    all_syms = list(dict.fromkeys(live + del_syms))
    logger.info(f"universe: live {len(live)} + delisted {len(del_syms)} = {len(all_syms)} symbols")

    # 2. FMP OHLC (통일 소스, survivorship-inclusive)
    ohlc: Dict[str, pd.DataFrame] = {}
    delisted_with_data = 0
    for i, sym in enumerate(all_syms):
        if i % 50 == 0:
            logger.info(f"  OHLC fetch {i}/{len(all_syms)} (cache hits fast)")
        df = fetch_ohlc(sym, period_from)
        if df is not None and len(df) >= 80:
            ohlc[sym] = df
            if sym in del_syms:
                delisted_with_data += 1
    logger.info(f"OHLC 확보: {len(ohlc)}/{len(all_syms)} (delisted with data: {delisted_with_data}/{len(del_syms)})")
    if len(ohlc) < 20:
        logger.error("OHLC 데이터 부족 — 중단")
        return {}

    # catalyst (옵션)
    grades: Dict[str, pd.DataFrame] = {}
    if use_catalyst:
        logger.info("catalyst overlay: grades-historical 수집...")
        for i, sym in enumerate(ohlc.keys()):
            if i % 50 == 0:
                logger.info(f"  grades {i}/{len(ohlc)}")
            g = fetch_grades_asof(sym)
            if g is not None:
                grades[sym] = g
        logger.info(f"grades 확보: {len(grades)}/{len(ohlc)}")

    # 3. 마스터 캘린더 (live 종목들 — 전체 window 커버) + 테스트 날짜
    ref = max(ohlc.values(), key=len)
    cal = ref.index
    start_i, end_i = 252, len(cal) - horizon - 1
    if end_i <= start_i:
        logger.error("history 부족"); return {}
    test_dates = [cal[i] for i in range(start_i, end_i, test_every)]
    logger.info(f"테스트 날짜 {len(test_dates)}개 ({test_dates[0].date()} ~ {test_dates[-1].date()})")

    # 4. PIT 스코어링 + exit 시뮬
    records = []
    for n, T in enumerate(test_dates):
        if n % 20 == 0:
            logger.info(f"  backtest {n}/{len(test_dates)} @ {T.date()}")
        for sym, df in ohlc.items():
            hist = df.loc[:T]
            if len(hist) < 60:
                continue
            win = hist.tail(300)   # 252 lookback + rolling 충분, 속도 bound
            feat = ss.compute_features_for_series(win["Close"], win["High"], win["Low"], win["Volume"])
            if feat is None:
                continue
            # gate (live 동일)
            if not (feat["price"] >= ss.MIN_PRICE and feat["adv20"] >= ss.MIN_ADV20):
                continue
            if not feat["capacity_ok"]:
                continue
            if not (pd.isna(feat["ret_5d"]) or feat["ret_5d"] <= ss.CHASE_RET_5D):
                continue
            pre_score = 0.5 * feat["readiness"] + 0.5 * feat["explosiveness"]

            # catalyst overlay
            cat = np.nan
            if use_catalyst and sym in grades:
                g = grades[sym].loc[:T]
                if len(g) > 0:
                    cat = _analyst_score_from_grades(g.iloc[-1])
            score_full = pre_score if pd.isna(cat) else 0.7 * pre_score + 0.3 * cat

            # forward: entry T+1 시가
            fwd = df.loc[T:].iloc[1:1 + horizon]
            if fwd.empty:
                continue
            c0 = feat["price"]
            max_high_gain = float(fwd["High"].max() / c0 - 1) if len(fwd) else np.nan

            # ATR-scaled exit: target/stop = ATR%의 배수 (vol-adaptive), sane bound clip
            if atr_mode and not pd.isna(feat.get("atr_pct")):
                ap = feat["atr_pct"]
                tgt_i = float(np.clip(atr_target * ap, 0.10, 1.0))
                stop_i = float(np.clip(atr_stop * ap, 0.05, 0.50))
            else:
                tgt_i, stop_i = target, stop

            trade = simulate_trade(fwd, tgt_i, stop_i, slip_bps)
            if trade is None:
                continue
            records.append({
                "date": T, "ticker": sym, "delisted": sym in del_syms,
                "pre_score": pre_score, "score_full": score_full, "catalyst": cat,
                "max_high_gain": max_high_gain, "tgt_used": tgt_i, "stop_used": stop_i,
                "atr_pct": feat.get("atr_pct"), **trade,
            })

    if not records:
        logger.error("레코드 0개"); return {}
    df = pd.DataFrame(records)
    df.to_csv(Path(__file__).resolve().parent / "surge_backtest_hardened.csv", index=False)

    # 5. 분석
    score_col = "score_full" if use_catalyst else "pre_score"
    df["hit25"] = (df["max_high_gain"] >= 0.25).astype(int)
    df["hit30"] = (df["max_high_gain"] >= 0.30).astype(int)
    df["win"] = (df["net_ret"] > 0).astype(int)

    def topN(g): return g.nlargest(top_n, score_col)
    top = df.groupby("date", group_keys=False).apply(topN)

    df["q"] = pd.qcut(df[score_col], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    out = {
        "n_records": len(df), "n_dates": int(df["date"].nunique()),
        "n_delisted_obs": int(df["delisted"].sum()), "score_col": score_col,
        "params": {"top_n": top_n, "horizon": horizon, "target": target,
                   "stop": stop, "slip_bps": slip_bps, "catalyst": use_catalyst,
                   "atr_mode": atr_mode, "atr_target": atr_target, "atr_stop": atr_stop,
                   "avg_tgt_used": float(df["tgt_used"].mean()) if "tgt_used" in df else None,
                   "avg_stop_used": float(df["stop_used"].mean()) if "stop_used" in df else None},
        "occurrence": {
            "hit25_base": float(df["hit25"].mean()), "hit25_top": float(top["hit25"].mean()),
            "hit30_base": float(df["hit30"].mean()), "hit30_top": float(top["hit30"].mean()),
        },
        "net_trade": {
            "all_mean": float(df["net_ret"].mean()), "top_mean": float(top["net_ret"].mean()),
            "all_win": float(df["win"].mean()), "top_win": float(top["win"].mean()),
            "top_median_hold": float(top["hold_days"].median()),
            "top_exit_mix": {k: int(v) for k, v in top["exit_reason"].value_counts().items()},
        },
        "net_ret_by_q": {str(k): float(v) for k, v in df.groupby("q", observed=True)["net_ret"].mean().items()},
        "win_by_q": {str(k): float(v) for k, v in df.groupby("q", observed=True)["win"].mean().items()},
        "hit25_by_q": {str(k): float(v) for k, v in df.groupby("q", observed=True)["hit25"].mean().items()},
        "delisted_net_mean": float(df[df["delisted"]]["net_ret"].mean()) if df["delisted"].any() else None,
        "live_net_mean": float(df[~df["delisted"]]["net_ret"].mean()),
    }
    return out


def _print(o: dict):
    if not o:
        print("\n백테 실패."); return
    p = o["params"]
    print("\n" + "=" * 80)
    print("  FRIDAY SURGE 백테 (HARDENED: survivorship+slippage+exit model)")
    print("=" * 80)
    print(f"  records={o['n_records']}  dates={o['n_dates']}  delisted_obs={o['n_delisted_obs']}  "
          f"score={o['score_col']}")
    if p.get("atr_mode"):
        print(f"  exit: ATR-scaled target={p['atr_target']:.1f}×ATR stop={p['atr_stop']:.1f}×ATR "
              f"(avg tgt=+{p['avg_tgt_used']:.0%}/stop=-{p['avg_stop_used']:.0%}) timeout={p['horizon']}d  "
              f"slip={p['slip_bps']}bps  top_n={p['top_n']}  catalyst={p['catalyst']}")
    else:
        print(f"  exit: target=+{p['target']:.0%} stop=-{p['stop']:.0%} timeout={p['horizon']}d  "
              f"slip={p['slip_bps']}bps  top_n={p['top_n']}  catalyst={p['catalyst']}")
    print("-" * 80)
    oc = o["occurrence"]
    print("  [급등 발생률 (forward high)]")
    print(f"    +25%: base={oc['hit25_base']:.2%}  top{p['top_n']}={oc['hit25_top']:.2%}   "
          f"+30%: base={oc['hit30_base']:.2%}  top{p['top_n']}={oc['hit30_top']:.2%}")
    nt = o["net_trade"]
    print("-" * 80)
    print("  [현실적 NET 거래수익 (entry T+1 시가, target/stop/timeout, slippage 차감)]")
    print(f"    all-gated:  mean={nt['all_mean']:+.2%}  win={nt['all_win']:.1%}")
    print(f"    top{p['top_n']}:      mean={nt['top_mean']:+.2%}  win={nt['top_win']:.1%}  "
          f"median_hold={nt['top_median_hold']:.0f}d  exits={nt['top_exit_mix']}")
    print("-" * 80)
    print("  [score 5분위별 NET 거래수익 / 승률 / +25% 발생률 — monotonic 이어야 real]")
    nq, wq, hq = o["net_ret_by_q"], o["win_by_q"], o["hit25_by_q"]
    print("    net : " + "  ".join(f"{k}={v:+.2%}" for k, v in nq.items()))
    print("    win : " + "  ".join(f"{k}={v:.0%}" for k, v in wq.items()))
    print("    h25 : " + "  ".join(f"{k}={v:.1%}" for k, v in hq.items()))
    print("-" * 80)
    print(f"  [survivorship 체크] live net mean={o['live_net_mean']:+.2%}  "
          f"delisted net mean={o['delisted_net_mean'] if o['delisted_net_mean'] is None else format(o['delisted_net_mean'],'+.2%')}")
    print("=" * 80)
    print("⚠️  남은 제약: FMP delisted 가격 커버리지 불완전 / fuel historical 미복원 / 단일 regime /")
    print("    시가체결·동일가 target·stop 근사. 그래도 survivorship+비용은 반영됨.")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SURGE 백테 강화판")
    ap.add_argument("--live", type=int, default=400)
    ap.add_argument("--delisted", type=int, default=300)
    ap.add_argument("--from", dest="from_date", default="2022-06-01")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--target", type=float, default=0.20)
    ap.add_argument("--stop", type=float, default=0.10)
    ap.add_argument("--slip-bps", type=float, default=100)
    ap.add_argument("--catalyst", action="store_true")
    ap.add_argument("--atr-target", type=float, default=0.0, help="target = N×ATR%% (>0 활성)")
    ap.add_argument("--atr-stop", type=float, default=0.0, help="stop = M×ATR%% (>0 활성)")
    a = ap.parse_args()
    out = run(live_sample=a.live, delisted_cap=a.delisted, period_from=a.from_date,
              test_every=a.every, top_n=a.top, horizon=a.horizon, target=a.target,
              stop=a.stop, slip_bps=a.slip_bps, use_catalyst=a.catalyst,
              atr_target=a.atr_target, atr_stop=a.atr_stop)
    _print(out)
