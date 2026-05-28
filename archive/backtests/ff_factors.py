"""
ff_factors.py
=============
Kenneth French Data Library — daily Fama-French 3-factor loader.

Source ZIP:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip

Cache layout:
  archive/data_cache/F-F_Research_Data_Factors_daily_CSV.zip   (raw zip)
  archive/data_cache/F-F_Research_Data_Factors_daily.csv       (extracted CSV)
  archive/data_cache/ff3_daily.pkl                             (parsed DataFrame)

Returned DataFrame columns are decimal returns (not percent):
  index : DatetimeIndex (daily)
  MKT   : market excess return (Mkt - RF)
  SMB   : size factor
  HML   : value factor
  RF    : risk-free rate (1M T-Bill, daily compounded equivalent)
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
CACHE_DIR = QUANT_TOOL / "archive" / "data_cache"
ZIP_PATH = CACHE_DIR / "F-F_Research_Data_Factors_daily_CSV.zip"
CSV_PATH = CACHE_DIR / "F-F_Research_Data_Factors_daily.csv"
PKL_PATH = CACHE_DIR / "ff3_daily.pkl"

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

logger = logging.getLogger(__name__)


def _download_if_missing(force: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        return
    logger.info("Downloading FF3 daily from Ken French Data Library ...")
    req = urllib.request.Request(FF3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    ZIP_PATH.write_bytes(data)


def _extract_csv() -> Path:
    if CSV_PATH.exists():
        return CSV_PATH
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # ZIP에는 단일 csv가 들어있음
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV inside {ZIP_PATH}")
        zf.extract(names[0], CACHE_DIR)
        return CACHE_DIR / names[0]


def _parse_csv(path: Path) -> pd.DataFrame:
    """헤더/꼬리 잘라내고 daily block 만 파싱."""
    text = path.read_text()
    # 헤더 라인 찾기: ',Mkt-RF,SMB,HML,RF'
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(",Mkt-RF"):
            start = i
            break
    if start is None:
        raise RuntimeError("FF3 CSV header line not found")

    # 데이터: start+1 부터, 마지막 'Copyright' 등 텍스트 라인 직전까지
    body_lines = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s:
            continue
        # 데이터 라인은 YYYYMMDD,...로 시작
        first = s.split(",")[0].strip()
        if first.isdigit() and len(first) == 8:
            body_lines.append(line)
        else:
            # 텍스트 (Copyright 등) — 종료
            break

    buf = io.StringIO("\n".join([lines[start]] + body_lines))
    df = pd.read_csv(buf, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    # 첫 컬럼 (날짜)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"

    # percent → decimal
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.rename(columns={"Mkt-RF": "MKT"})
    return df[["MKT", "SMB", "HML", "RF"]].dropna()


def get_ff3_daily(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """FF3 daily factor DataFrame. 캐시 우선, 없으면 다운로드+파싱.

    Parameters
    ----------
    start, end : 잘라낼 날짜 범위 (양 끝 포함). None이면 전체.
    force_refresh : True면 zip부터 재다운로드.
    """
    if force_refresh and PKL_PATH.exists():
        PKL_PATH.unlink()

    if PKL_PATH.exists() and not force_refresh:
        df = pd.read_pickle(PKL_PATH)
    else:
        _download_if_missing(force=force_refresh)
        csv = _extract_csv()
        df = _parse_csv(csv)
        df.to_pickle(PKL_PATH)

    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = get_ff3_daily()
    print(f"FF3 daily rows: {len(df)}  range: {df.index.min().date()} ~ {df.index.max().date()}")
    print(df.tail(5))
    print(f"\nMean (annualized):")
    print((df.mean() * 252 * 100).round(2).astype(str) + "%")
