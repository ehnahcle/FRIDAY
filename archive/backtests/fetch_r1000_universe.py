"""
fetch_r1000_universe.py
=======================
Phase 24 prep step 1: Synthetic R1000 universe — top-1000 by mcap PIT.

3단계:
  1) FMP company-screener → 현재 US 비ETF/Fund top-3000 by mcap (cushion)
     + 기존 626 PIT 캐시 tickers 와 union
  2) 각 candidate 에 대해 FMP historical-market-capitalization 시계열 fetch
     캐시: ~/.quant_tool_cache/historical_mcap/{TICKER}.pkl
  3) 분기별 rebal date (2015-2025 Q-start, ~44 dates)마다 top-1000 → CSV
     archive/backtests/russell1000_synthetic_pit_constituents.csv

Why "synthetic" R1000:
  FMP / Wikipedia 공식 R1000 historical membership 미제공.
  R1000 ≈ top-1000 US stocks by mcap (소수 rule 제외). 본 backtest 는 후자 사용 →
  PIT 보장 (survivorship-free), 다만 "공식 R1000" 정의와 micro-편차 가능.
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
ARCHIVE = QUANT_TOOL / "archive" / "backtests"
load_dotenv(QUANT_TOOL / ".env")
API_KEY = os.getenv("FMP_API_KEY", "")

BASE_URL = "https://financialmodelingprep.com/stable"
CACHE_ROOT = Path.home() / ".quant_tool_cache"
MCAP_CACHE = CACHE_ROOT / "historical_mcap"
MCAP_CACHE.mkdir(parents=True, exist_ok=True)

CANDIDATE_LIST_CACHE = CACHE_ROOT / "r1000_candidates.pkl"
PIT_CSV_OUT = ARCHIVE / "russell1000_synthetic_pit_constituents.csv"

START_DATE = pd.Timestamp("2015-01-01")
END_DATE   = pd.Timestamp("2025-12-31")
TOP_N      = 1000

RATE_LIMIT_PER_MIN = 250
_calls: list[float] = []
_rate_lock = threading.Lock()


def _rate_limit():
    """Thread-safe sliding-window rate limiter."""
    while True:
        sleep_s = 0.0
        with _rate_lock:
            now = time.time()
            _calls[:] = [t for t in _calls if now - t < 60]
            if len(_calls) < RATE_LIMIT_PER_MIN:
                _calls.append(now)
                return
            sleep_s = 60 - (now - _calls[0]) + 0.5
        # 락 해제 후 sleep — 다른 스레드도 진행 가능
        time.sleep(sleep_s)


# ---------------------------------------------------------------
# Step 1: candidate ticker list
# ---------------------------------------------------------------

def fetch_screener_candidates(limit: int = 3000, mcap_min: int = 300_000_000) -> list[dict]:
    if CANDIDATE_LIST_CACHE.exists():
        with open(CANDIDATE_LIST_CACHE, "rb") as f:
            return pickle.load(f)

    _rate_limit()
    r = requests.get(f"{BASE_URL}/company-screener", params={
        "marketCapMoreThan": mcap_min,
        "isEtf": "false",
        "isFund": "false",
        "country": "US",
        "limit": limit,
        "apikey": API_KEY,
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(CANDIDATE_LIST_CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def load_existing_pit_universe() -> list[str]:
    """기존 626 PIT 캐시 ticker (SP500 + 폐지/탈락) — 결합 union 용."""
    cur_path = CACHE_ROOT / "backtest" / "sp500_tickers.pkl"
    with open(cur_path, "rb") as f:
        cur = pickle.load(f)
    universe = set(cur)
    del_path = CACHE_ROOT / "backtest" / "prices_delisted_2015-01-01_2025-12-31.pkl"
    if del_path.exists():
        with open(del_path, "rb") as f:
            prices_del = pickle.load(f)
        if isinstance(prices_del.columns, pd.MultiIndex):
            close_del = prices_del["Close"]
            valid = [t for t in close_del.columns if not close_del[t].isna().all()]
        else:
            valid = list(prices_del.columns)
        universe.update(valid)
    return sorted(universe)


# ---------------------------------------------------------------
# Step 2: historical market cap fetch per ticker
# ---------------------------------------------------------------

def fetch_historical_mcap(symbol: str, limit: int = 5000) -> pd.Series | None:
    """Daily mcap 시리즈. 캐시 hit 면 즉시 반환.

    NOTE: FMP /stable/historical-market-capitalization 은 limit 만 주면 최근 ~3개월(61 rows) 만
    반환. 11년치를 받으려면 from/to 를 명시해야 함.
    """
    cache_file = MCAP_CACHE / f"{symbol}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    _rate_limit()
    try:
        r = requests.get(f"{BASE_URL}/historical-market-capitalization", params={
            "symbol": symbol,
            "from": START_DATE.strftime("%Y-%m-%d"),
            "to":   END_DATE.strftime("%Y-%m-%d"),
            "limit": limit,
            "apikey": API_KEY,
        }, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        # 404, 402 등 — 캐시 안 함 (재시도 가능)
        return None
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        # 빈 리스트도 캐시 안 함 (re-fetch 가능)
        return None

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    s = df["marketCap"].astype(float)
    s.name = symbol
    with open(cache_file, "wb") as f:
        pickle.dump(s, f)
    return s


# ---------------------------------------------------------------
# Step 3: build PIT top-1000 membership
# ---------------------------------------------------------------

def get_quarterly_rebal_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """1/4/7/10 first business day in range."""
    out = []
    for y in range(start.year, end.year + 1):
        for m in (1, 4, 7, 10):
            d = pd.Timestamp(year=y, month=m, day=1)
            # first business day
            while d.weekday() >= 5:
                d += pd.Timedelta(days=1)
            if start <= d <= end:
                out.append(d)
    return out


def build_pit_membership(mcap_panel: pd.DataFrame, rebal_dates: list[pd.Timestamp],
                         top_n: int = TOP_N) -> pd.DataFrame:
    """rebal_dates 각각에 top-N by mcap → DataFrame(date, tickers)."""
    mcap_panel = mcap_panel.sort_index()
    rows = []
    for d in rebal_dates:
        sub = mcap_panel.loc[:d]
        if len(sub) == 0:
            # rebal date 가 mcap panel 시작 이전 (예: 2015-01-01 휴일 + 데이터 2015-01-02 시작)
            # → 가장 이른 사용 가능 행으로 fallback
            snap = mcap_panel.iloc[0]
        else:
            snap = sub.ffill().iloc[-1]
        valid = snap.dropna()
        if len(valid) < top_n:
            top = valid.sort_values(ascending=False).index.tolist()
        else:
            top = valid.sort_values(ascending=False).head(top_n).index.tolist()
        rows.append({"date": d.strftime("%Y-%m-%d"), "tickers": ",".join(top)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)

    print("=" * 90)
    print("Phase 24 Step 1 — Synthetic R1000 universe build")
    print("=" * 90)

    # Step 1: candidate union
    print("\n[1/3] Candidate ticker union ...")
    screener = fetch_screener_candidates()
    screener_tickers = [d["symbol"] for d in screener]
    print(f"  Screener (top-3000 by current mcap, US, non-ETF/Fund): {len(screener_tickers)}")
    existing = load_existing_pit_universe()
    print(f"  Existing PIT cache: {len(existing)}")
    candidates = sorted(set(screener_tickers) | set(existing))
    print(f"  Union candidates:   {len(candidates)}")

    # 빠른 sanity — 어디서 새로 추가됐는지
    new_from_screener = sorted(set(screener_tickers) - set(existing))
    print(f"  new (screener \\ existing): {len(new_from_screener)}")
    only_in_existing  = sorted(set(existing) - set(screener_tickers))
    print(f"  delisted-style (existing \\ screener): {len(only_in_existing)}")

    # Step 2: historical mcap fetch
    print(f"\n[2/3] Historical mcap fetch — {len(candidates)} tickers ...")
    print(f"  Cache dir: {MCAP_CACHE}")

    start = time.time()
    mcap_series_list: list[pd.Series] = []
    failed: list[str] = []
    cached_n = sum(1 for t in candidates if (MCAP_CACHE / f"{t}.pkl").exists())
    print(f"  Already cached: {cached_n} / {len(candidates)}")
    print(f"  To fetch: ~{len(candidates) - cached_n}, rate-limited ~{(len(candidates) - cached_n)/250:.1f} min")

    # 병렬 fetch — 8 workers (rate limit 은 lock 으로 보장)
    done_count = [0]
    done_lock = threading.Lock()

    def _worker(sym: str):
        s = fetch_historical_mcap(sym)
        with done_lock:
            done_count[0] += 1
            i = done_count[0]
            if i % 100 == 0 or i == len(candidates):
                elapsed = time.time() - start
                print(f"    [{i}/{len(candidates)}] {sym} (elapsed {elapsed:.0f}s, "
                      f"ok={len(mcap_series_list)}, fail={len(failed)})", flush=True)
        return sym, s

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_worker, sym) for sym in candidates]
        for fut in as_completed(futures):
            sym, s = fut.result()
            if s is None or len(s) == 0:
                failed.append(sym)
            else:
                mcap_series_list.append(s)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.0f}s. ok={len(mcap_series_list)}, fail={len(failed)}")
    if failed[:20]:
        print(f"  Failed examples: {failed[:20]}")

    # Step 3: PIT membership
    print(f"\n[3/3] Building PIT top-{TOP_N} membership ...")
    if not mcap_series_list:
        print("ERROR: no mcap data collected.")
        sys.exit(2)

    # NOTE: 일부 ticker는 SP500 PIT 캐시 안에도 있지만 FMP historical-mcap 에서
    # 못 받았을 수 있음 — 그건 어쩔 수 없이 universe 밖.
    mcap_panel = pd.concat(mcap_series_list, axis=1, sort=True).sort_index()
    mcap_panel = mcap_panel.loc[
        (mcap_panel.index >= START_DATE - pd.Timedelta(days=400)) &
        (mcap_panel.index <= END_DATE + pd.Timedelta(days=10))
    ]
    print(f"  Mcap panel: {mcap_panel.shape[0]} rows × {mcap_panel.shape[1]} tickers")
    print(f"  Range: {mcap_panel.index.min().date()} ~ {mcap_panel.index.max().date()}")

    rebal_dates = get_quarterly_rebal_dates(START_DATE, END_DATE)
    print(f"  Rebal dates: {len(rebal_dates)} (Q-start)")

    pit_df = build_pit_membership(mcap_panel, rebal_dates, TOP_N)

    # turnover stat
    sizes = [len(r.split(",")) for r in pit_df["tickers"]]
    print(f"  Membership size: min={min(sizes)} median={int(pd.Series(sizes).median())} max={max(sizes)}")

    # quarterly turnover
    prev = set(pit_df.iloc[0]["tickers"].split(","))
    turnovers = []
    for _, row in pit_df.iloc[1:].iterrows():
        cur = set(row["tickers"].split(","))
        churn = len(cur ^ prev) / 2 / max(len(cur), len(prev))
        turnovers.append(churn)
        prev = cur
    print(f"  Quarterly turnover: median {pd.Series(turnovers).median()*100:.2f}%  max {pd.Series(turnovers).max()*100:.2f}%")

    pit_df.to_csv(PIT_CSV_OUT, index=False)
    print(f"\n저장: {PIT_CSV_OUT}")

    # union universe over all rebal dates
    union = set()
    for r in pit_df["tickers"]:
        union.update(r.split(","))
    print(f"  Union universe (ever in top-{TOP_N}): {len(union)}")

    # save union for next steps (price/PIT fundamentals fetch)
    union_path = ARCHIVE / "r1000_synthetic_union_tickers.pkl"
    with open(union_path, "wb") as f:
        pickle.dump(sorted(union), f)
    print(f"  Union list saved: {union_path}")


if __name__ == "__main__":
    main()
