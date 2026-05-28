"""
factors.py
==========
FRIDAY 팩터 계산 모듈 — SPACE picker 전용 (R1000 universe)

quant_tool/factors.py에서 fork (2026-05-28). 핵심 함수 (sector_zscore / calc_*_score /
Altman 산업 조정) 그대로 유지. PROFILES dict는 SPACE 단일 — C80_E20 가중치.

ROCKET/ROCKET_LEGACY는 제거 — FRIDAY는 R1000 SPACE 단일 시스템이며,
ROCKET 운영은 quant_tool/에서 분리 운영됨.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sector_zscore(df: pd.DataFrame, col: str, sector_col: str = "sector") -> pd.Series:
    """섹터 내 z-score 계산"""
    s = df[col].copy()
    s = s.replace([np.inf, -np.inf], np.nan)

    grouped = df.groupby(sector_col)[col]
    median_filled = s.fillna(grouped.transform("median"))
    median_filled = median_filled.fillna(s.median())

    mean = median_filled.groupby(df[sector_col]).transform("mean")
    std = median_filled.groupby(df[sector_col]).transform("std")
    z = (median_filled - mean) / std.replace(0, 1)
    z = z.clip(-3, 3)
    return z


def global_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """전체 시장 z-score (섹터 중립 미적용). ROCKET 운영 모드 (Phase 16/17 PIT 18.1% / Sh 0.59)."""
    s = df[col].copy()
    s = s.replace([np.inf, -np.inf], np.nan)
    median = s.median()
    s = s.fillna(median)
    mean = s.mean()
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=df.index)
    z = (s - mean) / std
    return z.clip(-3, 3)


# ROCKET 스코어링 모드: 전체 z-score + Alpha 6M 단일 (Phase 16 PIT 18.1% 검증)
USE_GLOBAL_ZSCORE = True


def set_simple_mode(simple: bool) -> None:
    """스코어링 모드 전환 (백테스트 호환용 — 실제 운영은 항상 True)"""
    global USE_GLOBAL_ZSCORE
    USE_GLOBAL_ZSCORE = bool(simple)


def _zscore(df: pd.DataFrame, col: str, sector_col: str = "sector") -> pd.Series:
    """현재 설정에 따라 sector_zscore 또는 global_zscore 사용"""
    if USE_GLOBAL_ZSCORE:
        return global_zscore(df, col)
    return sector_zscore(df, col, sector_col)


def zscore_to_score(z: pd.Series, invert: bool = False) -> pd.Series:
    """z-score → 0~100 점수"""
    if invert:
        z = -z
    score = ((z + 3) / 6) * 100
    return score.clip(0, 100)


# ============================================================
# Value Score
# ============================================================

def calc_value_score(df: pd.DataFrame) -> pd.Series:
    """Value 팩터 점수"""
    weights = {
        "ev_ebitda": 0.30,
        "ev_revenue": 0.25,
        "p_fcf": 0.20,
        "pe": 0.15,
        "pb": 0.10,
    }
    total_score = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)

    for col, w in weights.items():
        if col not in df.columns:
            continue
        valid = df[col] > 0
        z = _zscore(df[valid], col).reindex(df.index)
        z = z.fillna(3.0)
        z[~valid] = 3.0
        score = zscore_to_score(z, invert=True)

        mask = score.notna()
        total_score = total_score.add(score.fillna(0) * w * mask.astype(float), fill_value=0)
        total_weight = total_weight.add(w * mask.astype(float), fill_value=0)

    final = total_score / total_weight.replace(0, 1)
    return final.fillna(50)


# ============================================================
# Quality Score
# ============================================================

def calc_quality_score(df: pd.DataFrame) -> pd.Series:
    """Quality 팩터 점수"""
    weights_higher_better = {
        "gp_to_assets": 0.25,
        "roa": 0.15,
        "operating_margins": 0.15,
        "gross_margins": 0.10,
        "ocf_to_ni": 0.15,
    }
    weights_lower_better = {
        "debt_to_ebitda": 0.20,
    }

    total_score = pd.Series(0.0, index=df.index)
    total_weight = pd.Series(0.0, index=df.index)

    for col, w in weights_higher_better.items():
        if col not in df.columns:
            continue
        z = _zscore(df, col)
        score = zscore_to_score(z, invert=False)
        mask = score.notna()
        total_score = total_score.add(score.fillna(0) * w * mask.astype(float), fill_value=0)
        total_weight = total_weight.add(w * mask.astype(float), fill_value=0)

    for col, w in weights_lower_better.items():
        if col not in df.columns:
            continue
        valid = df[col] > 0
        z = _zscore(df[valid], col).reindex(df.index)
        z = z.fillna(3.0)
        z[~valid] = 3.0
        score = zscore_to_score(z, invert=True)
        mask = score.notna()
        total_score = total_score.add(score.fillna(0) * w * mask.astype(float), fill_value=0)
        total_weight = total_weight.add(w * mask.astype(float), fill_value=0)

    final = total_score / total_weight.replace(0, 1)
    return final.fillna(50)


# ============================================================
# Momentum Score (Alpha Momentum)
# ============================================================

def calc_alpha_momentum(
    prices: pd.DataFrame, spy_prices: pd.Series, period_months: int = 6
) -> pd.Series:
    """Alpha Momentum: 단순 누적 수익률 차이 (stock_ret − spy_ret)."""
    days = period_months * 21
    if len(prices) < days:
        return pd.Series(np.nan, index=prices.columns)

    # === SIMPLE 모드: 베타 조정 없음, 단순 누적 수익률 차이 ===
    if USE_GLOBAL_ZSCORE:
        end_date = prices.index[-1]
        start_date = end_date - pd.Timedelta(days=int(period_months * 30.5))
        period_data = prices.loc[start_date:end_date]
        if len(period_data) < days * 0.7:
            return pd.Series(np.nan, index=prices.columns)

        stock_ret = period_data.iloc[-1] / period_data.iloc[0] - 1
        spy_period = spy_prices.loc[start_date:end_date]
        if len(spy_period) < days * 0.7:
            spy_ret = 0.0
        else:
            spy_ret = float(spy_period.iloc[-1] / spy_period.iloc[0] - 1)
        return stock_ret - spy_ret

    # === SOPHISTICATED 모드: Beta 조정 잔차 알파 ===
    rets = prices.pct_change().dropna(how="all")
    spy_ret = spy_prices.pct_change().dropna()

    common_idx = rets.index.intersection(spy_ret.index)
    rets = rets.loc[common_idx].tail(days)
    spy_ret = spy_ret.loc[common_idx].tail(days)

    alphas = {}
    for ticker in rets.columns:
        try:
            stock_ret = rets[ticker].dropna()
            if len(stock_ret) < days * 0.7:
                alphas[ticker] = np.nan
                continue
            common = stock_ret.index.intersection(spy_ret.index)
            sr = stock_ret.loc[common]
            mr = spy_ret.loc[common]
            cov = np.cov(sr, mr)[0, 1]
            var = np.var(mr)
            beta = cov / var if var > 0 else 1.0
            beta = np.clip(beta, 0.3, 2.5)
            alpha_daily = sr - beta * mr
            cumulative_alpha = (1 + alpha_daily).prod() - 1
            alphas[ticker] = cumulative_alpha
        except Exception:
            alphas[ticker] = np.nan

    return pd.Series(alphas)


def calc_price_momentum(prices: pd.DataFrame, period_months: int = 6) -> pd.Series:
    """단순 가격 모멘텀"""
    days = period_months * 21
    if len(prices) < days:
        return pd.Series(np.nan, index=prices.columns)
    recent = prices.iloc[-1]
    past = prices.iloc[-days]
    return (recent / past) - 1


def calc_momentum_score(
    df: pd.DataFrame,
    alpha_6m: pd.Series,
    alpha_12m: pd.Series,
    above_ma200: pd.Series,
) -> pd.Series:
    """Momentum Score: 전체 z-score on Alpha 6M (Phase 16 PIT 검증 운영 모드)."""
    df_temp = df.copy()
    df_temp["alpha_6m"] = df_temp["ticker"].map(alpha_6m)

    if USE_GLOBAL_ZSCORE:
        z6 = global_zscore(df_temp, "alpha_6m")
        return zscore_to_score(z6, invert=False).fillna(50)

    df_temp["alpha_12m"] = df_temp["ticker"].map(alpha_12m)
    df_temp["above_ma200"] = df_temp["ticker"].map(above_ma200).fillna(False)

    z6 = sector_zscore(df_temp, "alpha_6m")
    score6 = zscore_to_score(z6, invert=False)

    z12 = sector_zscore(df_temp, "alpha_12m")
    score12 = zscore_to_score(z12, invert=False)

    score_ma = df_temp["above_ma200"].astype(float) * 100

    final = score6 * 0.5 + score12 * 0.3 + score_ma * 0.2
    return final.fillna(50)
def calc_momentum_score_v2(stock_prices, spy_prices):
    """
    V5 전략용: 3M + 6M + 12M 알파 모멘텀 + MA200 추세
    
    가중치:
    - 3M Alpha: 25% (단기 추세 잡기)
    - 6M Alpha: 30% (중기 추세)
    - 12M Alpha: 30% (장기 추세)
    - MA200 보너스: 15%
    """
    import pandas as pd
    
    if len(stock_prices) < 252:
        return None
    
    alpha_3m = calc_alpha_momentum(stock_prices, spy_prices, 63)
    alpha_6m = calc_alpha_momentum(stock_prices, spy_prices, 126)
    alpha_12m = calc_alpha_momentum(stock_prices, spy_prices, 252)
    
    # MA200 보너스
    if len(stock_prices) >= 200:
        above_ma200 = stock_prices.iloc[-1] > stock_prices.tail(200).mean()
    else:
        above_ma200 = False
    
    # NaN 처리
    if pd.isna(alpha_3m) or pd.isna(alpha_6m) or pd.isna(alpha_12m):
        return None
    
    return {
        "alpha_3m": alpha_3m,
        "alpha_6m": alpha_6m,
        "alpha_12m": alpha_12m,
        "above_ma200": above_ma200
    }


def momentum_score_v2_from_components(alpha_3m, alpha_6m, alpha_12m, above_ma200):
    """V5: 3M + 6M + 12M 결합 모멘텀 점수 (0-100)"""
    # z-score 가정 (이미 정규화된 값 받기)
    score = (
        alpha_3m * 0.25 +
        alpha_6m * 0.30 +
        alpha_12m * 0.30 +
        (100 if above_ma200 else 0) * 0.15
    )
    return max(0, min(100, score))

# ============================================================
# Shareholder Yield Score
# ============================================================

def calc_shareholder_yield_score(df: pd.DataFrame) -> pd.Series:
    """Shareholder Yield Score"""
    if "shares_change_yoy" in df.columns:
        z_buyback = _zscore(df, "shares_change_yoy")
        score_buyback = zscore_to_score(z_buyback, invert=True)
    else:
        score_buyback = pd.Series(50.0, index=df.index)

    if "dividend_yield" in df.columns:
        z_div = _zscore(df, "dividend_yield")
        score_div = zscore_to_score(z_div, invert=False)
    else:
        score_div = pd.Series(50.0, index=df.index)

    if "fcf_yield" in df.columns:
        valid = df["fcf_yield"] > 0
        z_fcf = _zscore(df[valid], "fcf_yield").reindex(df.index)
        z_fcf = z_fcf.fillna(-3)
        z_fcf[~valid] = -3
        score_fcf = zscore_to_score(z_fcf, invert=False)
    else:
        score_fcf = pd.Series(50.0, index=df.index)

    final = score_buyback * 0.50 + score_div * 0.25 + score_fcf * 0.25
    return final.fillna(50)


# ============================================================
# EPS Revisions Score
# ============================================================

def calc_eps_revisions_score(df: pd.DataFrame) -> pd.Series:
    """EPS Revisions Score.

    구성 (있는 것만, 가중치 정규화):
      - eps_revision: 분석가 forward EPS 수정 (live 환경에서만)    가중치 0.60
      - earnings_quarterly_growth: 분기 이익 YoY                     가중치 0.40

    NaN-safe: 결측 종목은 그 컴포넌트에서 중립 50으로 처리.
    Phase 19에서 SUE 추가 시도 → PIT 백테스트에서 noise 확인되어 폐기 (factors.py 60/40 복귀).
    """
    parts = []
    weights = []

    def _component(col: str, weight: float, invert: bool = False) -> None:
        if col not in df.columns:
            return
        nan_mask = df[col].isna()
        valid_df = df[~nan_mask].copy()
        if not valid_df.empty:
            z = _zscore(valid_df, col)
            score = zscore_to_score(z, invert=invert).reindex(df.index)
            score[nan_mask] = 50
        else:
            score = pd.Series(50.0, index=df.index)
        parts.append(score)
        weights.append(weight)

    _component("eps_revision", 0.60)
    _component("earnings_quarterly_growth", 0.40)

    if not parts:
        return pd.Series(50.0, index=df.index)

    weights = np.array(weights) / sum(weights)
    final = sum(p * w for p, w in zip(parts, weights))
    return final.fillna(50)


# ============================================================
# 프로파일 가중치 (Phase 16 PIT 검증)
# ============================================================

PROFILES = {
    "SPACE": {
        # Phase 24 (2026-05-28) — Synthetic R1000 universe 백테스트 확정 config. FRIDAY 운영 picker.
        # 가중치는 Phase 16 picker 그대로 C80_E20 (quant_tool ROCKET과 동일). 차이는 universe + liquidity:
        #   - Universe: Synthetic R1000 PIT (top-1000 by mcap, 분기별 reconstitution).
        #               백테스트는 archive/backtests/russell1000_synthetic_pit_constituents.csv 사용.
        #               Live screener는 FMP company-screener로 현재 시점 top-1000 by mcap fetch.
        #   - Liquidity gate: ADV20 ≥ $5M (quant_tool ROCKET 기본 $20M보다 완화 — universe 확장).
        #   - Slippage 모델 (백테스트 전용): Almgren-Chriss linear, α=10 (base 10bps + 10·participation·100).
        #   - Vol-Managed 30% layer 유지 (R1000 위에서 Pareto+: +0.14pp CAGR / +0.050 Sharpe / +7.7pp MDD).
        # 백테스트 결과 (2015-2025, Phase 24 Step 2):
        #   r1000+gate+slip α=10:        26.82% / 0.707 / -50.09% / Calmar 0.535 / Vol 38.6%
        #   r1000+gate+slip+VM30 ⭐:     26.96% / 0.757 / -42.41% / Calmar 0.636 / Vol 34.0%  ← 채택 config
        #   vs ROCKET (Ph17 SP500+VM30): 17.89% / 0.608 / -37.03% / Calmar 0.483 / Vol 26.3%
        #   Δ: +9.07pp CAGR / +0.149 Sharpe / +0.153 Calmar / Vol +7.7pp / MDD -5.4pp (Pareto 아님, high-CAGR cousin).
        # 9-cell sensitivity (α∈{5,10,20}×ADV20∈{$2M,$5M,$10M}): 모두 SP500 baseline 위, robust plateau.
        # 검증: archive/backtests/backtest_phase24_r1000.py + backtest_phase24_sensitivity.py
        "BULL":    {"value": 0.0, "quality": 0.0, "momentum": 0.80, "shareholder": 0.0, "eps_rev": 0.20},
        "NEUTRAL": {"value": 0.0, "quality": 0.0, "momentum": 0.80, "shareholder": 0.0, "eps_rev": 0.20},
        "BEAR":    {"value": 0.0, "quality": 0.0, "momentum": 0.80, "shareholder": 0.0, "eps_rev": 0.20},
    },
}

# Top N: Cap 20 / Hold 20 전략 (Top 10 매수, Top 20 보유 유지)
PROFILE_TOP_N = {
    "SPACE": 20,
}

DEFAULT_PROFILE = "SPACE"
REGIME_WEIGHTS = PROFILES[DEFAULT_PROFILE]


def get_weights(profile: str = "SPACE", regime: str = "NEUTRAL") -> dict:
    """프로필 + 레짐 조합으로 가중치 반환"""
    profile = profile.upper() if profile.upper() in PROFILES else "SPACE"
    regime = regime.upper() if regime.upper() in PROFILES[profile] else "NEUTRAL"
    return PROFILES[profile][regime]


def calc_total_score(scores: Dict[str, pd.Series], regime: str) -> pd.Series:
    """레짐별 가중치로 최종 점수 계산"""
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["NEUTRAL"])
    total = pd.Series(0.0, index=scores["value"].index)
    for factor, weight in weights.items():
        if factor in scores:
            total += scores[factor].fillna(50) * weight
    return total

# ============================================================
# Altman Z-Score 산업별 조정 (Hybrid 방식)
# ============================================================
# 산업별 자본구조 차이를 반영하면서도 절대 강점을 보존
# - 위험 영역: 산업별 임계값 (Financial 1.0 vs Tech 5.0)
# - 안전 영역: 산업 기준 통과 = 80점 + 절대값 클수록 보너스

ALTMAN_THRESHOLDS = {
    # 자본 부담 적음 (현금 많음)
    "Technology": {"safe": 5.0, "danger": 2.0},
    
    # 일반 산업
    "Healthcare": {"safe": 3.0, "danger": 1.5},
    "Industrials": {"safe": 3.0, "danger": 1.5},
    "Consumer Cyclical": {"safe": 2.5, "danger": 1.2},
    "Consumer Defensive": {"safe": 2.5, "danger": 1.2},
    "Communication Services": {"safe": 2.5, "danger": 1.0},
    "Energy": {"safe": 2.5, "danger": 1.0},
    "Basic Materials": {"safe": 2.5, "danger": 1.0},
    
    # 자본 부담 많음 (구조적 부채)
    "Real Estate": {"safe": 1.8, "danger": 0.8},
    "Utilities": {"safe": 1.5, "danger": 0.7},
    "Financial Services": {"safe": 1.0, "danger": 0.3},
    
    # 기본값
    "Other": {"safe": 1.8, "danger": 0.8},
}


def calc_altman_score_industry_adjusted(altman_z, sector):
    """
    Hybrid Altman 점수: 산업별 임계값 + 절대 강점 보너스
    
    - 위험 영역: 산업별 임계값 (Financial 0.3, Tech 2.0)
    - 안전 영역: 산업 기준 통과 = 80점, 절대값 클수록 보너스
    - Tech 성장주의 진짜 강점 (Z=63 같은) 도 인정
    
    Returns:
        0-100 점수
    """
    import pandas as pd
    
    if pd.isna(altman_z):
        return 50  # 데이터 없으면 중립
    
    # 산업별 임계값
    thresholds = ALTMAN_THRESHOLDS.get(sector, ALTMAN_THRESHOLDS["Other"])
    safe = thresholds["safe"]
    danger = thresholds["danger"]
    
    if altman_z >= safe:
        # 산업 기준 안전 → 80점 시작 + 절대값 보너스
        excess = altman_z - safe
        bonus = min(20, excess / safe * 10)
        return min(100, 80 + bonus)
    
    elif altman_z <= danger:
        # 위험 영역 → 0~30점
        if altman_z >= 0:
            return max(0, altman_z / danger * 30)
        else:
            return 0  # 음수면 매우 위험
    
    else:
        # 회색지대 → 30~80점 (선형 보간)
        ratio = (altman_z - danger) / (safe - danger)
        return 30 + ratio * 50


def get_altman_status(altman_z, sector):
    """
    산업 조정된 Altman 상태
    
    Returns:
        "safe" / "gray" / "danger" / "unknown"
    """
    import pandas as pd
    
    if pd.isna(altman_z):
        return "unknown"
    
    thresholds = ALTMAN_THRESHOLDS.get(sector, ALTMAN_THRESHOLDS["Other"])
    
    if altman_z >= thresholds["safe"]:
        return "safe"
    elif altman_z <= thresholds["danger"]:
        return "danger"
    else:
        return "gray"    
