"""
vol_manage.py
=============
Moreira-Muir (2017) Volatility-Managed Portfolio — FRIDAY 실시간 운영용

SPACE Cap20/Hold20 (R1000 universe) 위에 변동성 스케일링 layer 추가.
매월 말 자동으로 권장 노출률 계산 → 대시보드에 표시.

honest PIT 검증 결과 (Phase 26b live-faithful, 2015-2025, SPACE C80_E20 + ADV20 $5M gate + α=10 slip,
real-sector secmap = live screener 동작 일치):
  베이스 (no vol-mgmt):  연 24.19% / Sharpe 0.644 / MDD -59.39% / Calmar 0.407 / Vol 39.9%
  Vol-Managed 30% 적용:  연 24.99% / Sharpe 0.703 / MDD -49.11% / Calmar 0.509 / Vol 34.7%
  → lift: ΔCAGR +0.80pp / ΔSharpe +0.059 / ΔMDD +10.3pp shallower / ΔCalmar +0.102 (Pareto+ on every dim)

NOTE: Phase 24 헤드라인 26.96%/0.757 은 sector-map 버그 artifact (유니버스 ~50%가 'Unknown'
단일 버킷 → 숨은 large-cap 틸트). 위는 real-sector 재실행 보정값. VM30 lift 는 honest run 에서
오히려 더 큼 — R1000 idiosyncratic crash 가 VM 신호를 더 잘 trigger.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ============================================================
# 핵심 파라미터 (백테스트 검증값)
# ============================================================

TARGET_VOL = 0.30        # 연환산 목표변동성 30%
VOL_LOOKBACK = 22        # 1개월 거래일
MIN_EXPOSURE = 0.30      # 최소 노출 30% (완전 현금화 방지)
MAX_EXPOSURE = 1.00      # 최대 노출 100% (무차입)


# ============================================================
# 유틸: 가격 데이터 → 변동성
# ============================================================

def compute_realized_vol(prices: pd.DataFrame, weights: pd.Series | None = None) -> float:
    """
    포트폴리오의 직전 22거래일 실현변동성 (연환산).

    prices: 종목별 일간 종가 DataFrame (columns=ticker, index=date)
    weights: 종목별 비중 Series (None → 균등)
    """
    if prices is None or prices.empty:
        return float("nan")

    # 일간 수익률
    rets = prices.pct_change().dropna(how="all")
    if len(rets) < VOL_LOOKBACK:
        logger.warning(f"가격 데이터 부족: {len(rets)}일 < {VOL_LOOKBACK}일")
        return float("nan")

    rets = rets.tail(VOL_LOOKBACK)

    if weights is None:
        weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)

    # 포트 일간 수익률
    port_rets = (rets * weights).sum(axis=1)
    daily_std = port_rets.std()
    return float(daily_std * np.sqrt(252))


def recommended_exposure(realized_vol: float, target_vol: float = TARGET_VOL) -> float:
    """노출률 = target_vol / realized_vol, [MIN, MAX] 클램프"""
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        return 1.0
    raw = target_vol / realized_vol
    return float(np.clip(raw, MIN_EXPOSURE, MAX_EXPOSURE))


# ============================================================
# SPACE 자동 노출률 계산 (현재 Top 10 picks 기준)
# ============================================================

def fetch_recent_prices(tickers: list[str], days: int = 30) -> pd.DataFrame:
    """yfinance에서 직전 N거래일 종가 가져오기"""
    if not tickers:
        return pd.DataFrame()

    end = datetime.now()
    start = end - timedelta(days=int(days * 1.6))  # 휴장일 고려 여유

    try:
        data = yf.download(
            tickers, start=start, end=end, progress=False,
            auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        logger.error(f"yfinance 다운로드 실패: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # 단일 vs 멀티 ticker 구조 통일
    if len(tickers) == 1:
        if "Close" in data.columns:
            return data[["Close"]].rename(columns={"Close": tickers[0]})
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_cols = {}
        for tk in tickers:
            try:
                close_cols[tk] = data[tk]["Close"]
            except Exception:
                continue
        return pd.DataFrame(close_cols)

    return data


def compute_space_exposure(space_csv: Path | str = "friday_results.csv",
                           top_n: int = 10) -> dict:
    """
    현재 SPACE Top 10 picks 기준 변동성 및 권장 노출률 계산.

    Returns
    -------
    {
        "tickers": [...],
        "realized_vol": float (연환산),
        "target_vol": float,
        "recommended_exposure": float,
        "current_cash_ratio": float (1 - 권장노출),
        "status": str ("FULL" | "REDUCED" | "INSUFFICIENT_DATA"),
        "as_of": str (datetime),
    }
    """
    csv_path = Path(space_csv)
    if not csv_path.exists():
        return {"status": "NO_DATA", "error": f"{space_csv} 없음"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

    if "ticker" not in df.columns:
        return {"status": "ERROR", "error": "ticker 컬럼 없음"}

    # 실제 매수 대상 Top N — space_rank 기준 (단일 진실의 원천)
    if "space_rank" in df.columns:
        eligible = (
            df[df["space_rank"].notna()]
            .sort_values("space_rank")
        )
    else:
        # 폴백 (옛 CSV)
        eligible = df.copy()
        if "rank" in eligible.columns:
            eligible = eligible.sort_values("rank")
        if "buy_eligible" in eligible.columns:
            eligible = eligible[eligible["buy_eligible"] == True]
        if "sell_now" in eligible.columns:
            eligible = eligible[eligible["sell_now"] != True]
    tickers = eligible["ticker"].head(top_n).tolist()

    if not tickers:
        return {"status": "INSUFFICIENT_DATA",
                "error": "space_rank 통과 종목 없음"}

    prices = fetch_recent_prices(tickers, days=35)
    if prices.empty or len(prices) < VOL_LOOKBACK:
        return {
            "status": "INSUFFICIENT_DATA",
            "tickers": tickers,
            "error": f"가격 데이터 부족 ({len(prices) if not prices.empty else 0}일)",
        }

    vol = compute_realized_vol(prices)
    exp = recommended_exposure(vol)

    return {
        "tickers": tickers,
        "realized_vol": vol,
        "target_vol": TARGET_VOL,
        "recommended_exposure": exp,
        "current_cash_ratio": 1.0 - exp,
        "status": "FULL" if exp >= 0.95 else ("REDUCED" if exp >= 0.6 else "DEFENSIVE"),
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lookback_days": VOL_LOOKBACK,
        "n_tickers": len(tickers),
    }


# ============================================================
# 월말 스냅샷 상태 관리 (실제 운영용)
# ============================================================

STATE_FILE = Path(__file__).parent / "volmanage_state.json"


def load_state() -> dict:
    """월말 스냅샷 상태 파일 읽기. 없으면 빈 dict."""
    import json
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"State file 읽기 실패: {e}")
        return {}


def save_snapshot(snapshot: dict) -> None:
    """월말 스냅샷 저장 (운영의 진실 — 이 값이 다음 달 동안 유지)"""
    import json
    snapshot_to_save = {
        "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
        "applied_exposure": snapshot["recommended_exposure"],
        "applied_cash_ratio": snapshot["current_cash_ratio"],
        "realized_vol": snapshot["realized_vol"],
        "target_vol": snapshot["target_vol"],
        "status": snapshot["status"],
        "tickers": snapshot["tickers"],
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot_to_save, f, indent=2, ensure_ascii=False)
    logger.info(f"Vol-Managed 스냅샷 저장: 노출 {snapshot['recommended_exposure']*100:.1f}%")


def is_snapshot_due() -> bool:
    """이번 달에 아직 스냅샷 안 했으면 True"""
    state = load_state()
    if not state.get("snapshot_date"):
        return True
    try:
        last = datetime.strptime(state["snapshot_date"], "%Y-%m-%d")
    except Exception:
        return True
    today = datetime.now()
    # 다른 (년, 월)이면 새 스냅샷 필요
    return (today.year, today.month) != (last.year, last.month)


def days_to_next_month_end() -> int:
    """이번 달 마지막 거래일까지 며칠? (주말 무시한 어림치)"""
    today = datetime.now()
    # 이번 달 말일 = 다음 달 1일 - 1일
    if today.month == 12:
        first_next = datetime(today.year + 1, 1, 1)
    else:
        first_next = datetime(today.year, today.month + 1, 1)
    last_of_month = first_next - timedelta(days=1)
    # 마지막 거래일: 월말이 토일이면 직전 금요일로
    while last_of_month.weekday() >= 5:
        last_of_month -= timedelta(days=1)
    delta = (last_of_month.date() - today.date()).days
    return delta


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--snapshot":
        # 월말 스냅샷 저장 (수동 강제 — is_snapshot_due 체크 없이 무조건 갱신)
        result = compute_space_exposure()
        if result.get("status") in ("FULL", "REDUCED", "DEFENSIVE"):
            save_snapshot(result)
            print(f"✅ 스냅샷 저장: 노출 {result['recommended_exposure']*100:.1f}%")
            print(f"   적용 기간: 이번 달 ~ 다음 월말 측정 전")
        else:
            print(f"❌ 스냅샷 실패: {result.get('status')} — {result.get('error', '')}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--snapshot-if-due":
        # GHA 자동화용: 이번 달에 아직 스냅샷 안 했으면만 저장 (idempotent daily 호출 안전)
        if is_snapshot_due():
            result = compute_space_exposure()
            if result.get("status") in ("FULL", "REDUCED", "DEFENSIVE"):
                save_snapshot(result)
                print(f"✅ 신규 월 스냅샷 저장: 노출 {result['recommended_exposure']*100:.1f}%")
            else:
                print(f"❌ 스냅샷 실패: {result.get('status')} — {result.get('error', '')}")
                sys.exit(1)
        else:
            state = load_state()
            print(f"⏭️  이번 달 스냅샷 이미 존재 ({state.get('snapshot_date')}), skip")
    else:
        # 현재 vol 출력 + 스냅샷 상태
        current = compute_space_exposure()
        state = load_state()
        print("=== 현재 측정값 (참고) ===")
        print(json.dumps(current, indent=2, ensure_ascii=False, default=str))
        print()
        print("=== 적용 중인 스냅샷 ===")
        if state:
            print(json.dumps(state, indent=2, ensure_ascii=False))
        else:
            print("(아직 스냅샷 없음 — 첫 월말에 `python vol_manage.py --snapshot` 실행)")
        print()
        print(f"이번 달 스냅샷 필요? {is_snapshot_due()}")
        print(f"다음 월말까지 약 {days_to_next_month_end()}일")
