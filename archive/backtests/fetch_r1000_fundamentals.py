"""
fetch_r1000_fundamentals.py
===========================
Phase 24 prep step 1 (continued): R1000 union universe 의 PIT fundamentals
(balance / income / cashflow) FMP fetch. 병렬화 (8 workers).

Input:
  archive/backtests/r1000_synthetic_union_tickers.pkl

기존:
  ~/.quant_tool_cache/pit_fundamentals/{balance-sheet|income|cash-flow-statement}_{TICKER}.pkl
  현재 626 ticker. (cash flow 617개 — 9개 누락 ticker 도 함께 재시도)

새 ticker 만 fetch (캐시 hit 면 skip). 3 endpoint × N 미캐시 ticker.
Rate limit: FMP 250/min (thread-safe sliding window).
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
ARCHIVE = QUANT_TOOL / "archive" / "backtests"
load_dotenv(QUANT_TOOL / ".env")
API_KEY = os.getenv("FMP_API_KEY", "")

BASE_URL = "https://financialmodelingprep.com/stable"
CACHE_DIR = Path.home() / ".quant_tool_cache" / "pit_fundamentals"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNION_PKL = ARCHIVE / "r1000_synthetic_union_tickers.pkl"
ENDPOINTS = ("balance-sheet-statement", "income-statement", "cash-flow-statement")

RATE_LIMIT_PER_MIN = 250
_calls: list[float] = []
_rate_lock = threading.Lock()


def _rate_limit():
    while True:
        sleep_s = 0.0
        with _rate_lock:
            now = time.time()
            _calls[:] = [t for t in _calls if now - t < 60]
            if len(_calls) < RATE_LIMIT_PER_MIN:
                _calls.append(now)
                return
            sleep_s = 60 - (now - _calls[0]) + 0.5
        time.sleep(sleep_s)


def _is_cached(symbol: str, endpoint: str) -> bool:
    path = CACHE_DIR / f"{endpoint}_{symbol}.pkl"
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


def _fetch_one(endpoint: str, symbol: str) -> bool:
    """단일 endpoint × symbol fetch. ok=True/False 반환."""
    cache_file = CACHE_DIR / f"{endpoint}_{symbol}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list) and len(data) > 0:
                return True
        except Exception:
            pass

    _rate_limit()
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params={
            "symbol": symbol, "period": "quarter", "limit": 50, "apikey": API_KEY,
        }, timeout=20)
    except Exception:
        return False
    if r.status_code != 200:
        return False
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        return False
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    return True


def main():
    if not API_KEY:
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)
    if not UNION_PKL.exists():
        print(f"ERROR: union pkl not found: {UNION_PKL}")
        sys.exit(1)

    with open(UNION_PKL, "rb") as f:
        union = pickle.load(f)
    print(f"Union universe: {len(union)} tickers")
    print(f"PIT 캐시: {CACHE_DIR}")

    cov = {ep: sum(_is_cached(t, ep) for t in union) for ep in ENDPOINTS}
    print(f"\n초기 coverage (union 기준):")
    for ep in ENDPOINTS:
        print(f"  {ep:<30s}  {cov[ep]:4d}/{len(union)}")

    to_fetch: list[tuple[str, str]] = []
    for sym in union:
        for ep in ENDPOINTS:
            if not _is_cached(sym, ep):
                to_fetch.append((ep, sym))
    print(f"\n신규 fetch: {len(to_fetch)} call (parallel 8 workers, ~{len(to_fetch)/250:.1f} min)")

    if not to_fetch:
        print("Nothing to fetch.")
        return

    start = time.time()
    done = [0]
    ok = [0]
    fail = [0]
    done_lock = threading.Lock()

    def _worker(arg):
        ep, sym = arg
        result = _fetch_one(ep, sym)
        with done_lock:
            done[0] += 1
            if result:
                ok[0] += 1
            else:
                fail[0] += 1
            i = done[0]
            if i % 200 == 0 or i == len(to_fetch):
                elapsed = time.time() - start
                print(f"  [{i}/{len(to_fetch)}] {ep[:14]:<14s} {sym:<8s} elapsed {elapsed:.0f}s ok={ok[0]} fail={fail[0]}",
                      flush=True)
        return result

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_worker, args) for args in to_fetch]
        for fut in as_completed(futures):
            fut.result()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s  ok={ok[0]} fail={fail[0]}")

    cov2 = {ep: sum(_is_cached(t, ep) for t in union) for ep in ENDPOINTS}
    print(f"\n최종 coverage (union 기준):")
    for ep in ENDPOINTS:
        print(f"  {ep:<30s}  {cov2[ep]:4d}/{len(union)}  (+{cov2[ep] - cov[ep]})")


if __name__ == "__main__":
    main()
