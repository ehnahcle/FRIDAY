"""
fetch_r1000_prices.py
=====================
Phase 24 prep step 1 (continued): R1000 union universe 의 새 ticker 가격 fetch.

Input:
  archive/backtests/r1000_synthetic_union_tickers.pkl
  (fetch_r1000_universe.py 가 생성)

기존 캐시:
  ~/.quant_tool_cache/backtest/prices_2015-01-01_2025-12-31.pkl  (SP500 503)
  ~/.quant_tool_cache/backtest/prices_delisted_2015-01-01_2025-12-31.pkl  (delisted 123)

신규 캐시:
  ~/.quant_tool_cache/backtest/prices_r1000_extension_2015-01-01_2025-12-31.pkl
  (Union − 기존 두 캐시 = 새 ~400-900 ticker)

yfinance MultiIndex (Open/High/Low/Close/Volume × ticker) 형식 그대로 저장.
load_combined_prices_r1000() 으로 세 캐시 concat → backtest 사용.

배치: 50 ticker씩 yfinance.download (threads=True). Rate limit 약함.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
if str(QUANT_TOOL) not in sys.path:
    sys.path.insert(0, str(QUANT_TOOL))
ARCHIVE = QUANT_TOOL / "archive" / "backtests"

CACHE_DIR = Path.home() / ".quant_tool_cache" / "backtest"
SP500_PRICES = CACHE_DIR / "prices_2015-01-01_2025-12-31.pkl"
DELISTED_PRICES = CACHE_DIR / "prices_delisted_2015-01-01_2025-12-31.pkl"
R1000_EXT_PRICES = CACHE_DIR / "prices_r1000_extension_2015-01-01_2025-12-31.pkl"

UNION_PKL = ARCHIVE / "r1000_synthetic_union_tickers.pkl"

START = "2015-01-01"
END   = "2025-12-31"
BATCH = 50

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _existing_tickers() -> set[str]:
    """기존 두 캐시에 valid 데이터가 있는 ticker set."""
    have = set()
    for path in (SP500_PRICES, DELISTED_PRICES):
        if not path.exists():
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
            valid = [t for t in close.columns if not close[t].isna().all()]
        else:
            valid = list(df.columns)
        have.update(valid)
    return have


def main():
    if not UNION_PKL.exists():
        print(f"ERROR: union pkl not found: {UNION_PKL}. fetch_r1000_universe.py 먼저 실행 필요.")
        sys.exit(1)

    with open(UNION_PKL, "rb") as f:
        union = pickle.load(f)
    print(f"Union universe size: {len(union)}")

    have = _existing_tickers()
    print(f"Existing price cache (SP500+delisted): {len(have)} tickers")

    to_fetch = sorted(set(union) - have)
    print(f"To fetch (new):     {len(to_fetch)}")

    # 이미 r1000-extension 캐시가 있으면 그 안의 tickers 도 skip
    if R1000_EXT_PRICES.exists():
        with open(R1000_EXT_PRICES, "rb") as f:
            existing_ext = pickle.load(f)
        if isinstance(existing_ext.columns, pd.MultiIndex):
            close_ext = existing_ext["Close"]
            already_ext = [t for t in close_ext.columns if not close_ext[t].isna().all()]
        else:
            already_ext = list(existing_ext.columns)
        print(f"R1000 ext cache existing: {len(already_ext)} → skipping those")
        to_fetch = sorted(set(to_fetch) - set(already_ext))
        print(f"To fetch (after ext skip): {len(to_fetch)}")
    else:
        existing_ext = None

    if not to_fetch:
        print("Nothing to fetch.")
        return

    # yf.download 배치
    print(f"\nFetching {len(to_fetch)} tickers in batches of {BATCH} ...")
    parts = []
    start_t = time.time()
    for i in range(0, len(to_fetch), BATCH):
        batch = to_fetch[i:i + BATCH]
        elapsed = time.time() - start_t
        print(f"  [{i}/{len(to_fetch)}] batch {len(batch)} (elapsed {elapsed:.0f}s)")
        try:
            df = yf.download(batch, start=START, end=END, auto_adjust=True,
                             progress=False, threads=True, group_by="column")
        except Exception as e:
            logger.warning(f"  batch failed: {e}")
            continue
        if df is None or df.empty:
            continue
        # 단일 ticker 일 경우 columns 이 flat 일 수 있음 — multiindex 로 통일
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([df.columns, batch])
        parts.append(df)

    if not parts:
        print("ERROR: no data fetched.")
        sys.exit(2)

    combined = pd.concat(parts, axis=1)
    # 중복 컬럼 제거
    combined = combined.loc[:, ~combined.columns.duplicated()]
    # 모두 NaN ticker 제거
    if isinstance(combined.columns, pd.MultiIndex):
        close = combined["Close"]
        valid_t = [t for t in close.columns if not close[t].isna().all()]
        keep_cols = [(f, t) for f in combined.columns.levels[0] for t in valid_t if (f, t) in combined.columns]
        combined = combined[keep_cols]

    # 기존 ext 캐시와 병합
    if existing_ext is not None:
        merged = pd.concat([existing_ext, combined], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
    else:
        merged = combined

    with open(R1000_EXT_PRICES, "wb") as f:
        pickle.dump(merged, f)

    elapsed = time.time() - start_t
    if isinstance(merged.columns, pd.MultiIndex):
        nticker = len(merged["Close"].columns)
    else:
        nticker = len(merged.columns)
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Extension cache: {nticker} tickers")
    print(f"  Saved: {R1000_EXT_PRICES}")

    # 결산
    new_have = _existing_tickers() | set(
        [t for t in merged["Close"].columns
         if not merged["Close"][t].isna().all()]
        if isinstance(merged.columns, pd.MultiIndex)
        else list(merged.columns)
    )
    missing = sorted(set(union) - new_have)
    print(f"  Total coverage: {len(new_have & set(union))}/{len(union)} union tickers")
    if missing:
        print(f"  Still missing ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
