"""
dashboard.py
============
FRIDAY (SPACE picker) — Streamlit 대시보드
실행: streamlit run dashboard.py

quant_tool/dashboard.py 의 ROCKET 인프라와 분리. MVP 4 탭:
  🏆 Top Picks | 💼 Portfolio | ⚠️ Alerts(참고) | ℹ️ About
Phase 2.5에서 Charts/Stock Detail/FMP Insights 확장 예정.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="🚀 FRIDAY",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 데이터 로딩
# ============================================================
@st.cache_data(ttl=300)
def load_results():
    """friday_results.csv 로드"""
    csv = Path("friday_results.csv")
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    ts = datetime.fromtimestamp(csv.stat().st_mtime)
    return df, ts


def load_macro() -> dict:
    """macro_indicators.json 로드 (클라우드+로컬 공통)"""
    p = Path("macro_indicators.json")
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_portfolio() -> dict:
    """friday/portfolio.json (cold start 시 빈 holdings)"""
    p = Path("portfolio.json")
    if not p.exists():
        return {"holdings": {}, "cash": 0}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"holdings": {}, "cash": 0}


@st.cache_data(ttl=300)
def load_surge():
    """surge_results.csv (단타 SURGE 후보) + 갱신시각"""
    csv = Path("surge_results.csv")
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    ts = datetime.fromtimestamp(csv.stat().st_mtime)
    return df, ts


@st.cache_data(ttl=300)
def load_surge_attention():
    """surge_attention.csv (AI attention re-rank) — 있으면"""
    csv = Path("surge_attention.csv")
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv)
    ts = datetime.fromtimestamp(csv.stat().st_mtime)
    return df, ts


def load_surge_backtest() -> dict:
    """surge_backtest_summary.json (검증 결과 스냅샷)"""
    p = Path("surge_backtest_summary.json")
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ============================================================
# 사이드바 — Macro + Vol-Managed
# ============================================================
st.sidebar.header("🚀 SPACE — Cap20 / Hold20")
st.sidebar.caption("R1000 + ADV20 $5M + α=10 slip + VM30")

with st.sidebar.expander("📊 백테스트 결과 (Phase 26 live-faithful, 2015-2025)", expanded=False):
    st.markdown("""
    | 지표 | + VM30 ⭐ | no VM |
    |---|---|---|
    | 연 수익 | **24.99%** | 24.19% |
    | Sharpe | **0.703** | 0.644 |
    | MDD | **-49.1%** | -59.4% |
    | Calmar | **0.509** | 0.407 |
    | 변동성 | 34.7% | 39.9% |

    **vs ROCKET (Ph17, 17.9%/0.61/-37%/0.48)**: ΔCAGR +7.1pp / ΔSh +0.09.
    Calmar edge 미미, MDD 는 ROCKET 보다 깊음 (high-CAGR/high-vol cousin).

    ⚠️ Phase 24 헤드라인 27.0%/0.76 은 sector-map 버그 artifact (유니버스
    ~50%가 'Unknown' → 숨은 large-cap 틸트). 위 수치는 real-sector 재실행
    (Phase 26b) = live screener 와 동일 동작. survivorship 상 upper bound.
    """)

# Vol-Managed 패널
@st.cache_data(ttl=600)
def _calc_volmanage():
    try:
        from vol_manage import compute_space_exposure
        return compute_space_exposure("friday_results.csv")
    except Exception as exc:
        return {"status": "ERROR", "error": str(exc)}


if st.sidebar.button("🔄 Vol-Managed 새로고침", help="캐시 비우고 즉시 재계산"):
    _calc_volmanage.clear()
    st.rerun()

try:
    from vol_manage import load_state, save_snapshot, is_snapshot_due, days_to_next_month_end
    vm_state = load_state()
    vm_due = is_snapshot_due()
    vm_days_left = days_to_next_month_end()
except Exception:
    vm_state, vm_due, vm_days_left = {}, True, None

# 1) 이번 달 적용 노출률
with st.sidebar.expander("🛡️ 이번 달 적용 노출률", expanded=True):
    if vm_state.get("snapshot_date"):
        applied_exp = vm_state["applied_exposure"]
        applied_cash = vm_state["applied_cash_ratio"]
        status_emoji = {"FULL": "🟢", "REDUCED": "🟡", "DEFENSIVE": "🔴"}.get(vm_state.get("status", ""), "⚪")
        st.markdown(f"""
        **{status_emoji} {vm_state['status']}** — 스냅샷 {vm_state['snapshot_date']}

        | 항목 | 값 |
        |---|---|
        | 적용 주식 노출 | **{applied_exp*100:.1f}%** |
        | 적용 현금 비중 | {applied_cash*100:.1f}% |
        | 측정 시점 vol | {vm_state['realized_vol']*100:.1f}% |
        """)
        if vm_days_left is not None and vm_days_left >= 0:
            if vm_due:
                st.warning(f"🔔 이번 달 스냅샷 필요 (D{-vm_days_left:+d}일)")
            else:
                st.caption(f"⏳ 다음 측정까지 약 **{vm_days_left}일**")
    else:
        st.info("📸 첫 스냅샷 없음. 월말이 되면 저장하세요.")

# 2) 월말 액션
if vm_due:
    with st.sidebar.expander("📸 월말 스냅샷 (지금 행동)", expanded=True):
        vm_now = _calc_volmanage()
        if vm_now.get("status") in ("FULL", "REDUCED", "DEFENSIVE"):
            exp_now = vm_now["recommended_exposure"]
            st.markdown(f"""
            **현재 측정값**:
            - 실현 vol: {vm_now['realized_vol']*100:.1f}%
            - 권장 노출: **{exp_now*100:.1f}%**
            """)
            if st.button("📸 스냅샷 저장", type="primary", width="stretch"):
                try:
                    save_snapshot(vm_now)
                    st.cache_data.clear()
                    st.success(f"✅ {exp_now*100:.1f}% 노출 적용")
                    st.rerun()
                except Exception as e:
                    st.error(f"저장 실패: {e}")
        else:
            st.caption(f"측정 불가: {vm_now.get('error')}")

# 3) 실시간 (참고용)
with st.sidebar.expander("👀 실시간 vol (참고용)", expanded=False):
    vm_now = _calc_volmanage()
    if vm_now.get("status") in ("FULL", "REDUCED", "DEFENSIVE"):
        st.caption("⚠️ 참고용 — 월 중간 노이즈. 행동 X.")
        st.markdown(f"""
        - 실현 vol: {vm_now['realized_vol']*100:.1f}%
        - 지금 스냅샷하면: {vm_now['recommended_exposure']*100:.1f}%
        - 측정 시점: {vm_now.get('as_of', '-')}
        """)
    else:
        st.caption(f"측정 불가: {vm_now.get('error')}")

st.sidebar.divider()

# ============================================================
# 헤더
# ============================================================
st.title("🚀 FRIDAY")
st.caption("R1000 SPACE Picker • C80_E20 + ADV20 $5M + Vol-Managed 30%")

df, last_run = load_results()
if df is None:
    st.error("⚠️ `friday_results.csv` 없음 — `python screener.py` 먼저 실행")
    st.stop()

macro = load_macro()
portfolio = load_portfolio()

# 헤더 metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    regime = macro.get("regime", "?")
    regime_emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(regime, "⚪")
    st.metric("Market Regime", f"{regime_emoji} {regime}")
with col2:
    vix = macro.get("vix")
    st.metric("VIX", f"{vix:.1f}" if vix else "-")
with col3:
    spy_dist = macro.get("spy_distance_ma200", 0) or 0
    above = macro.get("spy_above_ma200", False)
    st.metric("SPY vs MA200", f"{spy_dist:+.1f}%", delta="ABOVE" if above else "BELOW")
with col4:
    if last_run:
        st.metric("Last Run", last_run.strftime("%m-%d %H:%M"))

st.divider()


# ============================================================
# Top N 슬라이더 — 사이드바
# ============================================================
SCORE_COL = "display_score" if "display_score" in df.columns else "total_score"

top_n = st.sidebar.slider("Hold Top N", 10, 20, 20, help="Cap20/Hold20 — 표시 종목 수")

# space_rank 통과 종목만 SPACE picks
space_picks = df[df["space_rank"].notna()].copy().sort_values("space_rank")


# ============================================================
# 메인 탭
# ============================================================
tab_picks, tab_surge, tab_pf, tab_alerts, tab_about = st.tabs(
    ["🏆 Top Picks", "🚀 SURGE (단타)", "💼 Portfolio", "⚠️ Alerts (참고)", "ℹ️ About"]
)

# ============================================================
# 탭 1: Top Picks — space_rank 기준
# ============================================================
with tab_picks:
    st.subheader(f"🏆 SPACE Top {top_n} (space_rank SSoT)")
    st.caption(
        f"R1000 universe → ADV20 $5M gate → sector cap 6 → space_rank "
        f"({len(space_picks)} 통과 / display_score 기준)"
    )

    if space_picks.empty:
        st.warning("space_rank 통과 종목 없음 — 모든 종목이 trend filter에 걸림")
    else:
        display = space_picks.head(top_n).copy()

        base_cols = [
            "space_rank", "rank", "ticker", "name", "sector",
            SCORE_COL, "momentum_score", "eps_revisions_score",
            "market_cap", "price", "adv20",
        ]
        fmp_cols = ["piotroski", "altman_z", "analyst_consensus", "buy_pct", "upside_pct"]
        avail = [c for c in base_cols + fmp_cols if c in display.columns]
        show = display[avail].copy()

        if "space_rank" in show.columns:
            show["space_rank"] = show["space_rank"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
        if "market_cap" in show.columns:
            show["market_cap"] = show["market_cap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "-")
        if "price" in show.columns:
            show["price"] = show["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
        if "adv20" in show.columns:
            show["adv20"] = show["adv20"].apply(lambda x: f"${x/1e6:.0f}M" if pd.notna(x) else "-")
        if "buy_pct" in show.columns:
            show["buy_pct"] = show["buy_pct"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
        if "upside_pct" in show.columns:
            show["upside_pct"] = show["upside_pct"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) else "-")
        if "altman_z" in show.columns:
            show["altman_z"] = show["altman_z"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        if "piotroski" in show.columns:
            show["piotroski"] = show["piotroski"].apply(lambda x: f"{int(x)}/9" if pd.notna(x) else "-")
        if "analyst_consensus" in show.columns:
            show["analyst_consensus"] = show["analyst_consensus"].fillna("-")

        show = show.rename(columns={
            "space_rank": "SPACE",
            "rank": "Raw",
            "ticker": "Ticker",
            "name": "Name",
            "sector": "Sector",
            SCORE_COL: "Score⭐",
            "momentum_score": "Mom",
            "eps_revisions_score": "EPS Rev",
            "market_cap": "MktCap",
            "price": "Price",
            "adv20": "ADV20",
            "piotroski": "Piot",
            "altman_z": "AltZ",
            "analyst_consensus": "Rating",
            "buy_pct": "Buy%",
            "upside_pct": "Target",
        })

        # 색상
        def color_score(v):
            try:
                x = float(v)
                if x >= 70: return "background-color: #2d5016; color: white"
                elif x >= 60: return "background-color: #3d6920; color: white"
                elif x >= 50: return "background-color: #4a4a4a; color: white"
                else: return "background-color: #5a2020; color: white"
            except Exception:
                return ""

        def color_rating(v):
            s = str(v)
            if s in ["Strong Buy", "Buy"]: return "background-color: #2d5016; color: white"
            elif s == "Hold": return "background-color: #6b5b00; color: white"
            elif s in ["Sell", "Strong Sell"]: return "background-color: #8b0000; color: white"
            return ""

        score_cols = [c for c in ["Score⭐", "Mom", "EPS Rev"] if c in show.columns]
        styled = show.style.format({c: "{:.1f}" for c in score_cols}, na_rep="-")
        if score_cols:
            styled = styled.map(color_score, subset=score_cols)
        if "Rating" in show.columns:
            styled = styled.map(color_rating, subset=["Rating"])

        st.dataframe(styled, width="stretch", height=600, hide_index=True)

        # 섹터 분포
        sec_dist = display["sector"].value_counts()
        st.markdown("**섹터 분포** (sector cap 6):")
        st.dataframe(
            pd.DataFrame({"Sector": sec_dist.index, "Count": sec_dist.values}),
            hide_index=True,
            width="stretch",
        )

        # 다운로드
        st.download_button(
            "📥 Download Top Picks CSV",
            display.to_csv(index=False),
            f"friday_top{top_n}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )

# ============================================================
# 탭 2: Portfolio
# ============================================================
with tab_pf:
    st.subheader("💼 FRIDAY Portfolio")

    holdings = portfolio.get("holdings", {})
    if not holdings:
        st.info("""
        🐣 **Cold start** — 아직 보유 종목 없음.

        분기 첫 영업일 (다음: 2026-07-01)에 SPACE Top 10 매수.

        매수 후 `portfolio.json`에 다음 형식으로 직접 추가:
        ```json
        {
          "holdings": {
            "BE": {"shares": 50, "avg_cost": 35.20, "buy_date": "2026-07-01", "notes": ""},
            "MU": {"shares": 12, "avg_cost": 145.80, "buy_date": "2026-07-01", "notes": ""}
          }
        }
        ```
        """)

        st.markdown("### 🆕 분기 매수 후보 — SPACE Top 10")
        if not space_picks.empty:
            cols = ["space_rank", "ticker", "name", "sector", SCORE_COL, "price", "adv20"]
            avail = [c for c in cols if c in space_picks.columns]
            top10 = space_picks[avail].head(10).copy()
            if "price" in top10.columns:
                top10["price"] = top10["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            if "adv20" in top10.columns:
                top10["adv20"] = top10["adv20"].apply(lambda x: f"${x/1e6:.0f}M" if pd.notna(x) else "-")
            if SCORE_COL in top10.columns:
                top10[SCORE_COL] = top10[SCORE_COL].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            if "space_rank" in top10.columns:
                top10["space_rank"] = top10["space_rank"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            top10 = top10.rename(columns={
                "space_rank": "SPACE", "ticker": "Ticker", "name": "Name",
                "sector": "Sector", SCORE_COL: "Score", "price": "Price", "adv20": "ADV20",
            })
            st.dataframe(top10, hide_index=True, width="stretch")

    else:
        # 보유 종목 분석
        rows = []
        for tk, h in holdings.items():
            row = df[df["ticker"] == tk]
            if row.empty:
                rows.append({
                    "ticker": tk, "shares": h.get("shares", 0),
                    "avg_cost": h.get("avg_cost", 0),
                    "current": np.nan, "value": np.nan, "pnl_pct": np.nan,
                    "space_rank": np.nan, "in_top20": False,
                })
                continue
            r = row.iloc[0]
            current = r.get("price", np.nan)
            shares = h.get("shares", 0)
            cost = h.get("avg_cost", 0)
            value = current * shares if pd.notna(current) else np.nan
            pnl = ((current / cost) - 1) * 100 if pd.notna(current) and cost > 0 else np.nan
            rank = r.get("space_rank", np.nan)
            rows.append({
                "ticker": tk, "shares": shares, "avg_cost": cost,
                "current": current, "value": value, "pnl_pct": pnl,
                "space_rank": rank,
                "in_top20": pd.notna(rank) and rank <= 20,
            })
        hold_df = pd.DataFrame(rows)

        total_value = hold_df["value"].sum()
        total_cost = (hold_df["shares"] * hold_df["avg_cost"]).sum()
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdings", f"{len(holdings)}")
        c2.metric("Total Value", f"${total_value:,.0f}")
        c3.metric("Cost Basis", f"${total_cost:,.0f}")
        c4.metric("P&L", f"${total_pnl:,.0f}", delta=f"{total_pnl_pct:+.2f}%")

        # 매도/매수 권고 (Top 20 밖 / 신규 매수 후보)
        out_of_top20 = hold_df[~hold_df["in_top20"]]
        if not out_of_top20.empty:
            st.warning(f"⚠️ **분기 매도 권고 {len(out_of_top20)}건** (SPACE Top 20 밖): "
                       f"{', '.join(out_of_top20['ticker'].tolist())}")

        held_tickers = set(holdings.keys())
        new_buys = space_picks.head(10)[~space_picks.head(10)["ticker"].isin(held_tickers)]
        if not new_buys.empty:
            st.success(f"🆕 **신규 매수 후보 {len(new_buys)}건** (Top 10 미보유): "
                       f"{', '.join(new_buys['ticker'].tolist())}")

        # 보유 표
        show = hold_df.copy()
        for c in ["avg_cost", "current"]:
            if c in show.columns:
                show[c] = show[c].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
        if "value" in show.columns:
            show["value"] = show["value"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
        if "pnl_pct" in show.columns:
            show["pnl_pct"] = show["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        if "space_rank" in show.columns:
            show["space_rank"] = show["space_rank"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "Out")

        show = show.rename(columns={
            "ticker": "Ticker", "shares": "Shares", "avg_cost": "Avg Cost",
            "current": "Current", "value": "Value", "pnl_pct": "P&L",
            "space_rank": "SPACE", "in_top20": "In Top20",
        })
        st.dataframe(show, hide_index=True, width="stretch")


# ============================================================
# 탭 3: Alerts (참고용 — SELL_NOW 비활성)
# ============================================================
with tab_alerts:
    st.subheader("⚠️ Alerts (참고용)")

    st.info("""
    💡 **SELL_NOW 룰 비활성 (2026-05-28 결정)** — Phase 24 백테 환경 일치 (`backtest_phase24_r1000.py`
    에서 VOLATILITY_SPIKE/MASSIVE_DILUTION/DEEP_DRAWDOWN/TREND_BREAK 코드 0건, 백테 결과는
    pure picker + ADV20 gate + size-slip만으로 도달). 보유 종목은 분기 rebal까지 무조건 hold,
    R1000 변동성 ~40% 감수, Vol-Managed 30% layer가 1차 risk 관리.

    아래 표는 **참고 정보 only** — 자동 매도 트리거 X. 데이터로서만 표시.
    """)

    # 보유 종목 중 borderline 표시
    holdings = portfolio.get("holdings", {})
    if holdings:
        st.markdown("### 보유 종목 — 정보성 borderline")
        rows = []
        for tk in holdings.keys():
            row = df[df["ticker"] == tk]
            if row.empty:
                continue
            r = row.iloc[0]
            flags = []
            vol = r.get("volatility_60d")
            if pd.notna(vol) and vol > 0.80:
                flags.append(f"vol60d {vol*100:.0f}%")
            mdd = r.get("mdd_1y")
            if pd.notna(mdd) and mdd <= -0.50:
                flags.append(f"MDD {mdd*100:.0f}%")
            dilu = r.get("shares_change_yoy")
            if pd.notna(dilu) and dilu > 0.15:
                flags.append(f"dilution {dilu*100:.0f}% YoY")
            above = r.get("above_ma200")
            if above is False or above == "False":
                flags.append("below MA200")
            if flags:
                rows.append({"ticker": tk, "flags": " / ".join(flags)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.success("✅ 보유 종목 borderline 신호 없음")
    else:
        st.caption("보유 종목 없음 (cold start)")

    st.divider()
    st.markdown("### Top Picks 중 정보성 — 변동성 상위 / MA200 borderline")
    if not space_picks.empty:
        info = space_picks.head(20).copy()
        cols = ["space_rank", "ticker", "sector", "volatility_60d", "mdd_1y", "shares_change_yoy", "above_ma200"]
        avail = [c for c in cols if c in info.columns]
        info = info[avail]

        # 정렬: vol 큰 순
        if "volatility_60d" in info.columns:
            info = info.sort_values("volatility_60d", ascending=False)

        if "volatility_60d" in info.columns:
            info["volatility_60d"] = info["volatility_60d"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "-")
        if "mdd_1y" in info.columns:
            info["mdd_1y"] = info["mdd_1y"].apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "-")
        if "shares_change_yoy" in info.columns:
            info["shares_change_yoy"] = info["shares_change_yoy"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "-")
        if "space_rank" in info.columns:
            info["space_rank"] = info["space_rank"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")

        info = info.rename(columns={
            "space_rank": "SPACE", "ticker": "Ticker", "sector": "Sector",
            "volatility_60d": "Vol60d", "mdd_1y": "MDD1y",
            "shares_change_yoy": "DilutionYoY", "above_ma200": "AboveMA200",
        })
        st.dataframe(info, hide_index=True, width="stretch")


# ============================================================
# 탭 4: About
# ============================================================
with tab_about:
    st.subheader("🚀 FRIDAY — SPACE Picker (R1000)")

    st.markdown("""
    FRIDAY는 quant_tool ROCKET (S&P 500)과 **별도 운영되는** R1000 universe 모멘텀 시스템입니다.
    EDITH(한국 단타) 분리 패턴 따라 격리 — ROCKET 운영 코드 0 변경.

    **두 시스템 병행 운영**:
    - 🌟 [ROCKET](https://github.com/ehnahcle/Quant-Tool) (quant_tool): SP500, C80_E20, 백테 17.9%/Sh 0.61
    - 🚀 **FRIDAY** (이 시스템): R1000, C80_E20 + ADV20 $5M, 백테 **25.0%/Sh 0.70** (Phase 26b live-faithful)
    """)

    st.divider()

    st.markdown("### 🎯 전략 동기 — SP500 plateau 돌파")
    st.markdown("""
    quant_tool Phase 16~22의 일관된 결론: S&P 500 universe + C80_E20 picker = **Sharpe 0.6 / CAGR 18% 천장**.
    weight tuning (Ph19/19e), vol scaling (Ph17/19d), leverage (Ph21), trend filter (Ph20),
    sector cap (Ph22/22b), 모멘텀 정의 자체 (Ph23 residual/skip-1) 모두 saturated.

    Phase 20 메타 진단: "더 짜내려면 **데이터 차원 (R1000/R3000 확장)** 또는 운영 차원 (세금)".
    Phase 24/26b가 그 데이터 차원 가설을 확정 — Russell 1000 확장 시 진짜 알파 **+6.1pp** 존재
    (live-faithful real-sector 재실행 기준; Phase 24 헤드라인 +8.7pp는 sector-map 버그로 과대).
    """)

    st.divider()

    st.markdown("### 📐 핵심 사양")
    st.markdown("""
    - **Universe**: Synthetic Russell 1000 PIT (분기별 top-1000 by mcap reconstitution)
      - 백테: FMP `historical-market-capitalization` 기반 합성, 44 quarterly rebal × 1000 ticker
      - Live: FMP `company-screener` (현재 시점 top-1000, 매주 갱신)
      - 분기 turnover median 3.8% / annualized 14.4% (학계 R1000 12-15%/yr와 일치)
    - **Picker**: C80_E20 (Momentum 80% + EPS Revisions 20%, regime-flat)
    - **Liquidity gate**: ADV20 ≥ **$5M** (ROCKET 운영 $20M보다 완화 — universe 확장에 맞춤)
    - **Slippage 모델 (백테 전용)**: Almgren-Chriss linear, **α=10** (base 10bps + 10·participation·100). 평균 실현 슬리피지 ~45bps
    - **Sector cap**: 6
    - **Trend filter**: MA200 OR α6m > 0 (BEAR 시 AND)
    - **Vol-Managed**: 30% annual target (월말 rebal)
    - **SELL_NOW 룰 비활성** — 백테 환경 일치 (사용자 결정 2026-05-28)
    """)

    st.divider()

    st.markdown("### 📊 백테스트 (2015-2025) — 4-mode 분해 (Phase 26b live-faithful, real-sector)")
    space_bt = {
        "모드": [
            "SP500 sanity (universe만 변경)",
            "R1000 no-gate (gate OFF, slip ON) ⚠️",
            "R1000 no-slip (gate ON, flat 10bps)",
            "🚀 R1000 + gate + slip α=10",
            "🛡️ R1000 + gate + slip + VM30 ⭐",
        ],
        "연 수익": ["18.11%", "3.13%", "25.77%", "24.19%", "**24.99%** ⭐"],
        "Sharpe": ["0.592", "0.297", "0.676", "0.644", "**0.703** ⭐"],
        "MDD": ["-40.1%", "-83.3%", "-58.1%", "-59.4%", "**-49.1%** ⭐"],
        "Calmar": ["0.452", "0.037", "0.444", "0.407", "**0.509** ⭐"],
        "변동성": ["28.2%", "—", "39.8%", "39.9%", "34.7%"],
        "비고": [
            "Phase 16 baseline 재현 ✓",
            "gate 없으면 슬리피지가 알파 다 먹음",
            "R1000 raw 알파 상한",
            "α=10 슬리피지 적용 (현실값)",
            "VM30 stack → Pareto+ on every dim",
        ],
    }
    st.dataframe(pd.DataFrame(space_bt), hide_index=True, width="stretch")

    st.markdown("""
    #### 📊 3-way 알파 분해 (vs SP500 baseline, live-faithful)
    | 효과 | ΔCAGR | ΔSharpe | 의미 |
    |---|---|---|---|
    | Universe expansion (R1000 raw) | **+7.66pp** | +0.084 | R1000 small/mid-cap 모멘텀 알파 |
    | Size-slip cost (α=10) | -1.58pp | -0.032 | 작은 종목 거래비용 |
    | Gate 보호 효과 ($5M ADV20) | **+21.06pp** | +0.347 | 슬리피지로부터 알파 보호, **결정적** |
    | **Net r1000 (gate+slip)** | **+6.08pp** | **+0.052** | gate가 없으면 R1000 알파 못 살림 |

    #### 🔬 9-cell sensitivity
    α ∈ {5, 10, 20} × ADV20 ∈ {$2M, $5M, $10M} grid (Phase 24, **old sector_map**):
    - ADV20 inverted-U peak at **$5M** sweet spot, α 단조 감소 — plateau 모양은 유지.
    - ⚠️ 절대 수준은 sector-map 버그로 ~2pp 과대. real-sector 재실행 시 전 셀 약 -2pp shift
      (peak ≈ 25%대), $5M sweet spot 결론 불변. 정밀 재검증 pending.
    """)

    st.divider()

    st.markdown("### 🆚 ROCKET vs SPACE 직접 비교 (둘 다 VM30)")
    compare = {
        "지표": ["연 수익", "Sharpe", "MDD", "Calmar", "변동성", "Universe", "운영"],
        "🌟 ROCKET (Ph17)": ["17.89%", "0.608", "-37.03%", "0.483", "26.3%", "SP500 ~500", "실전"],
        "🚀 SPACE (Ph26b)": ["**24.99%**", "**0.703**", "-49.11%", "0.509", "34.7%", "R1000 ~1000", "FRIDAY"],
        "Δ (SPACE − ROCKET)": ["**+7.10pp**", "**+0.095**", "-12.1pp ⚠️", "+0.026", "+8.4pp ⚠️", "≈2× 확장", "—"],
    }
    st.dataframe(pd.DataFrame(compare), hide_index=True, width="stretch")

    st.warning("""
    ⚠️ **Pareto 아님 — high-CAGR / high-vol cousin**:
    SPACE는 ROCKET 대비 CAGR(+7.1pp)/Sharpe(+0.10) 우월하나 **MDD -49% (ROCKET -37%보다 훨씬 깊고),
    Vol +8.4pp, Calmar edge 미미(+0.03)**. 단순 "더 좋다"가 아닌 명확한 trade-off.
    ROCKET을 SPACE로 *교체*하지 말고 **자금 일부만** (e.g., 30%) 배분 권장.
    """)

    st.divider()

    st.markdown("### 📅 운영 절차 — Cap 20 / Hold 20 분기 투자")
    st.success("""
    **기본 파라미터**
    | 항목 | 값 | 의미 |
    |---|---|---|
    | 매수 기준 | **Top 10** | space_rank 상위 10개 |
    | 보유 유지 | **Top 20** | 보유 중이면 20위까지 버팀 |
    | Position Cap | **20%** | 단일 종목 20% 초과 시 자동 트리밍 |
    | 리밸런싱 주기 | **분기** | 1/1, 4/1, 7/1, 10/1 첫 영업일 |
    | 가중치 | **C80_E20** | Momentum 80% + EPS Revisions 20% |

    ---

    #### 📋 분기 리밸런싱 Step-by-Step

    **STEP 1 — 시스템 실행 (분기 첫 영업일)**
    1. 터미널: `cd ~/Documents/friday && ./run_and_upload.sh`
    2. 완료 후 대시보드 새로고침 → Top Picks 탭

    **STEP 2 — 매도 결정**
    - 보유 종목 중 **현재 SPACE Top 20 밖**으로 밀린 종목 → 전량 매도
    - Portfolio 탭 "분기 매도 권고" 표시 종목

    **STEP 3 — 매수 결정**
    - **현재 SPACE Top 10** 중 미보유 종목 → 신규 매수
    - Portfolio 탭 "신규 매수 후보" 표시 종목
    - 매수 자금 = STEP 2 매도 대금 + 유휴 현금
    - 신규 매수 종목들에 **균등 배분**

    **STEP 4 — Position Cap (20% 룰)**
    - 매수 후 개별 종목 비중 계산
    - 20% 초과 시 초과분 매도 → 다른 종목 재배분

    **STEP 5 — portfolio.json 업데이트**
    - 실제 매수/매도 반영하여 `~/Documents/friday/portfolio.json` 직접 편집
    - git add + commit 권장 (변동 이력 추적)

    ---

    #### 🛡️ STEP 6 — Vol-Managed 30% (매월 1회)
    - 매월 말 사이드바 "📸 월말 스냅샷" 버튼 1번 클릭
    - 노출률 변화 5%p 이상 시 보유 종목 비례 매도/매수
    - 그 사이엔 실시간 vol 변동 무시

    | 상태 | 권장 행동 |
    |---|---|
    | 🟢 FULL (≥95%) | 풀 노출 유지 |
    | 🟡 REDUCED (60~94%) | 비례 매도 → 현금 확보 |
    | 🔴 DEFENSIVE (30~59%) | 절반 이상 현금화 |

    현금 위치: MMF, SHV/BIL 단기 채권 ETF
    """)

    st.divider()

    st.markdown("### ⚠️ 한계 & 면책")
    st.error("""
    **백테스트 한계 (현실 보정 필수)**:
    - **헤드라인 27.0%는 sector-map 버그 artifact** — Phase 24 백테가 비-SP500 유니버스(~50%)를
      'Unknown' 단일 섹터로 묶어 숨은 large-cap 틸트 발생. real-sector 재실행(Phase 26b) =
      **24.99%/0.703/-49%/0.509** = live screener 와 동일 동작. 위 모든 표는 이 보정값 기준.
    - **Synthetic R1000 ≠ 공식 R1000** — top-1000 by mcap 합성, 외국 ADR/일부 REIT 포함
    - **Survivorship 잔존** — candidate pool 이 current survivors + SP500 폐지 261개만 (union 1707 중
      captured-dead 150개). non-SP500 mid-cap 폐지 누락 → 24.99% 도 **upper bound**.
    - **Live mcap reconstitution 미검증** — 매 분기 fresh fetch 시 mcap snapshot timing 노이즈 가능
    - **현실 기대치**: 백테 25.0% → 실전 **~20-24% / Sharpe 0.55~0.65**

    **투자 추천 아님**:
    - 📊 정보 제공 목적: 통계 + 학술 모델 기반 분석
    - 💡 의사결정은 본인 책임
    - 🎯 백테스트 ≠ 미래
    - 💰 손실 위험: 주식 투자 원금 손실 가능 (특히 R1000 small-cap 변동성)
    - 📈 분산: 한 시스템에 자산 전부 X — ROCKET/SPY와 분산 권장
    """)

    st.divider()

    st.markdown("### 📚 학술 근거 (Phase 24 추가)")
    st.markdown("""
    | 항목 | 학술 논문 | 의미 |
    |---|---|---|
    | **Size Effect** | Banz (1981), Fama-French (1992) | R1000 small/mid-cap 모멘텀 알파 +7.7pp raw |
    | **Liquidity Premium** | Amihud (2002), Pástor-Stambaugh (2003) | ADV20 $5M gate 결정적 |
    | **Linear Slippage** | Almgren-Chriss (2001), Frazzini-Israel-Moskowitz (2018) | α≈10 학술 중간값 |
    | **R1000 Reconstitution** | Madhavan (2003) | 분기 turnover 14.4%/yr = 학계 R1000과 일치 |
    | **모멘텀 기반** | Jegadeesh-Titman (1993), Carhart (1997) | Phase 16 picker |
    | **Vol-Managed** | Moreira-Muir (2017) *JF* | VM30 layer |
    """)

    st.divider()
    st.caption("🚀 FRIDAY v1.0 — R1000 SPACE 백테 (Phase 26b live-faithful) + live screener (2026-05-28)")
    st.caption("🔗 github.com/ehnahcle/FRIDAY")


# ============================================================
# 탭: SURGE (단타 위성)
# ============================================================
with tab_surge:
    st.subheader("🚀 SURGE — 단기 위성 (speculative day-hold)")
    st.warning(
        "⚠️ JARVIS(안정 코어)/SPACE(분기)와 **별개 시스템**. 단기 트레이드 후보 발굴기 — "
        "예산 증식용 위성 자금만. 보유 = **일 단위**, **하드스톱 필수**. "
        "month-hold 시 음(-) 기대수익 (MAX/squeeze trap)."
    )
    s_today, s_manual, s_bt = st.tabs(["🎯 Today's Picks", "🛠 Manual Run", "📊 Backtest Evidence"])

    # ---- Today's Picks ----
    with s_today:
        sdf, sts = load_surge()
        if sdf is None or sdf.empty:
            st.info("`surge_results.csv` 없음 — **🛠 Manual Run** 탭의 명령으로 먼저 생성하세요.")
        else:
            adf, ats = load_surge_attention()
            has_ai = adf is not None and not adf.empty and "surge_ai_score" in adf.columns
            show = sdf.copy()
            if has_ai:
                mcols = [c for c in ["ticker", "surge_ai_score", "attention_score",
                                     "narrative", "catalyst_type", "thesis"] if c in adf.columns]
                show = show.merge(adf[mcols], on="ticker", how="left")
            show = show.sort_values("surge_score", ascending=False)  # 검증된 ranker로 정렬

            c1, c2, c3 = st.columns(3)
            c1.metric("후보 수", len(sdf))
            c2.metric("Screener 갱신", sts.strftime("%m-%d %H:%M") if sts else "—")
            c3.metric("AI attention", "✅ 적용" if has_ai else "❌ 미적용")

            disp = ["ticker", "name", "setup_type", "surge_score",
                    "readiness", "explosiveness", "fuel", "catalyst"]
            if has_ai:
                disp += ["narrative", "attention_score"]
            disp += ["price", "ann_vol", "dist_52wh", "short_pct_float", "target_upside", "sector"]
            disp = [c for c in disp if c in show.columns]
            v = show[disp].copy()
            for c in ["surge_score", "readiness", "explosiveness", "fuel", "catalyst", "attention_score"]:
                if c in v:
                    v[c] = v[c].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
            if "price" in v:
                v["price"] = v["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
            if "ann_vol" in v:
                v["ann_vol"] = v["ann_vol"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
            if "dist_52wh" in v:
                v["dist_52wh"] = v["dist_52wh"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            if "short_pct_float" in v:
                v["short_pct_float"] = v["short_pct_float"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
            if "target_upside" in v:
                v["target_upside"] = v["target_upside"].apply(lambda x: f"{x:+.0%}" if pd.notna(x) else "-")
            st.dataframe(v, hide_index=True, width="stretch")

            if has_ai and "thesis" in show.columns:
                with st.expander("🧠 AI narrative thesis (rising/fading 판정 근거)"):
                    for _, r in show.head(15).iterrows():
                        if pd.notna(r.get("thesis")):
                            st.markdown(f"**{r['ticker']}** "
                                        f"({r.get('narrative','?')} / {r.get('catalyst_type','?')}): "
                                        f"{r['thesis']}")
            st.download_button("⬇️ surge_results.csv", sdf.to_csv(index=False),
                               "surge_results.csv", "text/csv")
        st.caption(
            "⚠️ EXIT DISCIPLINE: 보유=일 단위, 하드스톱. **AI narrative는 fade-filter로 사용** — "
            "'fading'/'extended' 회피용이지 chasing용 아님 (백테: high-attention일수록 net↓, coiled 우선)."
        )

    # ---- Manual Run ----
    with s_manual:
        st.markdown("#### 🛠 수동 실행 — SURGE는 자동 스케줄 없음 (단타라 원할 때 직접)")
        sdf2, sts2 = load_surge()
        adf2, ats2 = load_surge_attention()
        c1, c2 = st.columns(2)
        c1.metric("surge_results.csv", sts2.strftime("%Y-%m-%d %H:%M") if sts2 else "없음")
        c2.metric("surge_attention.csv", ats2.strftime("%Y-%m-%d %H:%M") if ats2 else "없음")
        st.markdown("**1) 스크리너 실행 → today's pick 생성:**")
        st.code("cd ~/Documents/friday && source ~/Documents/quant_tool/venv/bin/activate\n"
                "python surge_screener.py --top 30", language="bash")
        st.markdown("**2) (선택) AI attention re-rank — 뉴스 narrative 판정:**")
        st.code("python ai_attention.py --top 20 --model haiku", language="bash")
        st.caption("실행 후 좌측 🔄 새로고침 → Today's Picks 갱신.")
        st.divider()
        st.markdown("#### 🔎 수동 티커 체크 — 오늘 후보에 있는지 + 점수")
        q = st.text_input("티커 입력 (쉼표 구분, 예: IBRX, CYTK, KOD)", key="surge_manual_q")
        if q:
            if sdf2 is None or sdf2.empty:
                st.info("surge_results.csv 먼저 생성하세요.")
            else:
                wl = [t.strip().upper() for t in q.split(",") if t.strip()]
                up = sdf2["ticker"].astype(str).str.upper()
                hit = sdf2[up.isin(wl)]
                if not hit.empty:
                    cols = [c for c in ["ticker", "setup_type", "surge_score", "readiness",
                                        "explosiveness", "fuel", "catalyst", "price", "ann_vol",
                                        "short_pct_float", "target_upside"] if c in hit.columns]
                    st.dataframe(hit[cols], hide_index=True, width="stretch")
                miss = [t for t in wl if t not in set(up)]
                if miss:
                    st.caption(f"오늘 후보에 없음 (게이트 탈락 or 점수권 밖): {', '.join(miss)}")

    # ---- Backtest Evidence ----
    with s_bt:
        bt = load_surge_backtest()
        if not bt:
            st.info("`surge_backtest_summary.json` 없음.")
        else:
            st.caption(f"검증 범위: {bt.get('scope', '')}")
            st.success(bt.get("headline", ""))
            oc = bt.get("occurrence", {})
            st.markdown("#### 📈 급등 발생률 (occurrence) — 신호 검증 ✓")
            c1, c2, c3 = st.columns(3)
            c1.metric("+25% base rate", f"{oc.get('hit25_base', 0):.1%}")
            c2.metric("+25% top-10", f"{oc.get('hit25_top10', 0):.1%}", f"{oc.get('hit25_lift', 0):.1f}x lift")
            c3.metric("+30% top-10", f"{oc.get('hit30_top10', 0):.1%}")
            ql = oc.get("quintile_hit25", {})
            if ql:
                st.caption("score 5분위별 +25% 발생률 (monotonic = real signal):")
                st.dataframe(pd.DataFrame([{k: f"{val:.1%}" for k, val in ql.items()}]),
                             hide_index=True, width="stretch")
            st.markdown("#### 💵 현실적 NET 거래수익 (survivorship + slippage + exit)")
            nt = bt.get("net_trade_by_exit", {})
            rows = nt.get("rows", [])
            if rows:
                st.dataframe(pd.DataFrame([
                    {"Exit rule": r["exit"], "top10 net/trade": f"{r['top10_net']:+.2%}",
                     "win": f"{r['win']:.0%}", "note": r["note"]} for r in rows
                ]), hide_index=True, width="stretch")
            sc = bt.get("survivorship_check", {})
            st.warning(
                f"all-gated net (ATR): {nt.get('all_gated_net_atr', 0):+.2%} — "
                f"**top decile만 양(+); 나머지는 음**.  "
                f"survivorship: live {sc.get('live_net', 0):+.2%} vs delisted {sc.get('delisted_net', 0):+.2%} "
                f"(폐지종목이 수익 끌어내림 → naive 백테는 optimistic)"
            )
            st.markdown("#### 🧠 AI-attention 엣지 (forward-test, RVOL proxy)")
            at = bt.get("attention_forwardtest", {})
            tp = at.get("top10_pick", {})
            if tp:
                st.dataframe(pd.DataFrame([{
                    "RAW pre_score": f"{tp.get('RAW_pre_score', 0):+.2%}",
                    "RVOL-attn only": f"{tp.get('RVOL_attention_only', 0):+.2%}",
                    "Blend 50/50": f"{tp.get('BLEND_50_50', 0):+.2%}",
                }]), hide_index=True, width="stretch")
            if at.get("finding"):
                st.info(at["finding"])
            st.markdown("#### ⚠️ Caveats")
            for cav in bt.get("caveats", []):
                st.markdown(f"- {cav}")
            st.caption(f"스냅샷: {bt.get('as_of', '')} · 상세 CSV: surge_backtest_hardened.csv / surge_attention_fwdtest.csv")


st.divider()
st.caption("💡 새 데이터: `cd ~/Documents/friday && ./run_and_upload.sh` 후 새로고침")
st.caption("⚠️ 백테 25.0% (Phase 26b live-faithful)는 11년 honest 측정값 — survivorship 상 upper bound, 미래 알파 보장 X")
