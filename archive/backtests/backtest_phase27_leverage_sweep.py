"""
backtest_phase27_leverage_sweep.py
==================================
Phase 27 (FRIDAY): Bull-market leverage sweep on the JARVIS / SPACE core.

질문 (user): "bull 시장일때 레버리지 들어가면 좀더 수익률을 올릴 수 있지 않을까?"

핵심 통찰: JARVIS 의 vol-targeting(Moreira-Muir) 은 이미 조용한 상승장에서
raw = target_vol / realized_vol > 1.0 을 원한다. 단지 MAX_EXP=1.0 캡에서 잘릴 뿐.
=> "불장 레버리지" = MAX_EXP 캡 상향. 별도 레짐 감지기 불필요 (vol 이 튀면 자동 디레버).

이 스크립트는 Phase 26b live-faithful 셋업(full real-sector map, ADV20 $5M gate,
size-dependent slip, VM30, R1000 universe) 그대로 두고 MAX_EXP 만 sweep 한다.
레버리지를 충실히 모델링하기 위해:
  (1) exposure>1 일 때 음(-)의 현금(차입) 허용
  (2) 차입에 일일 조달이자 부과 — 펀딩 베이스 = ^IRX(13주 T-bill, 시변) + 스프레드
  (3) 양(+)의 현금(디그로스분)은 0% (기존 베이스라인과 apples-to-apples)

라이브 모듈 (vol_manage.py / backtest_volmanaged_rocket.py) 는 손대지 않음.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

FRIDAY = Path(__file__).resolve().parent.parent.parent
QUANT_TOOL = Path("/Users/chanhui/Documents/quant_tool")
ARCHIVE_FRI = FRIDAY / "archive" / "backtests"
ARCHIVE_QT = QUANT_TOOL / "archive" / "backtests"
for p in (ARCHIVE_FRI, QUANT_TOOL, ARCHIVE_QT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import backtest_phase24_r1000 as p24
from backtest_volmanaged_rocket import (
    compute_realized_vol,
    is_month_end,
    metrics,
    sell_partial,
)
import backtest_phase26b_realsectors as p26b

# p24 가 module attribute 로 들고 있는 헬퍼들 (validate_cap20_hold20 / arb / 상수)
arb = p24.arb
buy_ticker = p24.buy_ticker
enforce_cap = p24.enforce_cap
sell_full = p24.sell_full
current_value = p24.current_value
slip_frac = p24.slip_frac
select_picks_phase24 = p24.select_picks_phase24

START_DATE = p24.START_DATE
END_DATE = p24.END_DATE
BUY_TOP = p24.BUY_TOP
HOLD_TOP = p24.HOLD_TOP
SLIP_BASE_BPS = p24.SLIP_BASE_BPS
INITIAL_CAPITAL = p24.INITIAL_CAPITAL

MIN_EXP = 0.30          # 디펜시브 floor 불변
TARGET_VOL = 0.30
FIN_SPREAD = 0.015      # 차입 스프레드 150bps (IBKR Pro 수준)


# ============================================================
# 시변 조달비용: ^IRX (13주 T-bill, 연 %) + 스프레드
# ============================================================

def build_financing_daily(index: pd.DatetimeIndex, spread: float = FIN_SPREAD) -> pd.Series:
    """close index 에 정렬된 '일일' 차입금리 Series (연율). margin = IRX% + spread."""
    try:
        irx = yf.download("^IRX", start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=False)
        col = irx["Close"] if "Close" in irx.columns else irx.iloc[:, 0]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        rate = col / 100.0  # percent -> fraction
        rate.index = pd.to_datetime(rate.index)
        rate = rate.reindex(index).ffill().bfill()
        annual = (rate + spread).clip(lower=0.0)
        print(f"  ^IRX 펀딩베이스: {rate.min()*100:.2f}%~{rate.max()*100:.2f}% "
              f"(중앙값 {rate.median()*100:.2f}%), +{spread*10000:.0f}bps 스프레드 "
              f"→ 차입금리 중앙값 {(rate.median()+spread)*100:.2f}%")
        return annual
    except Exception as e:
        print(f"  WARNING: ^IRX 로드 실패 ({e}) — 상수 2.0% + 스프레드 사용")
        return pd.Series(0.02 + spread, index=index)


# ============================================================
# 레버리지 인식 target_exposure / rescale
# ============================================================

def target_exposure_lev(realized_vol, target_vol, max_exp):
    if realized_vol is None or realized_vol <= 0:
        return 1.0
    raw = target_vol / realized_vol
    return float(np.clip(raw, MIN_EXP, max_exp))


def rescale_to_exposure_lev(close, date, holdings, cash, target_exp,
                            transaction_cost, yearly_realized):
    """노출을 target_exp 로 조정. exposure>1 (target_equity>total) 이면 cash 음수 허용."""
    total, equity = current_value(close, date, holdings, cash)
    if total <= 0:
        return cash
    current_exp = equity / total
    if abs(current_exp - target_exp) < 0.02:
        return cash
    target_equity = total * target_exp

    if equity > target_equity:
        # 노출 축소 → 비례 매도
        excess = equity - target_equity
        for ticker in list(holdings):
            if ticker not in close.columns:
                continue
            price = close.at[date, ticker]
            if pd.isna(price) or price <= 0:
                continue
            value = holdings[ticker]["shares"] * price
            sell_value = value * (excess / equity)
            shares_to_sell = sell_value / price
            proceeds, realized = sell_partial(
                close, date, holdings[ticker], shares_to_sell, transaction_cost)
            cash += proceeds
            yearly_realized[date.year] = yearly_realized.get(date.year, 0.0) + realized
            if holdings[ticker]["shares"] < 0.001:
                del holdings[ticker]
    else:
        # 노출 확대 → 보유종목에 분배. cash 부족분은 차입(음수 cash 허용).
        shortfall = target_equity - equity
        if shortfall <= 0 or not holdings:
            return cash
        per_ticker = shortfall / len(holdings)
        spent = 0.0
        for ticker in list(holdings):
            if ticker not in close.columns:
                continue
            spent += buy_ticker(close, date, holdings, ticker, per_ticker, transaction_cost)
        cash -= spent      # ← max(cash,0) 제거: 차입 허용
    return cash


# ============================================================
# 레버리지 인식 simulate (p24.simulate fork)
# ============================================================

def simulate_lev(prices, spy, vix, pit_membership, weight_schedule, sector_map,
                 bal_cache, inc_cache, cf_cache, *, max_exp, fin_daily):
    p24.factors.set_simple_mode(True)
    prices_w = prices.loc[START_DATE:END_DATE]
    spy_w = spy.loc[START_DATE:END_DATE]
    close = arb.get_close_prices(prices_w)
    rebal_dates = arb.get_rebalance_dates(START_DATE, END_DATE)

    holdings: dict = {}
    cash = float(INITIAL_CAPITAL)
    history = []
    yearly_realized: dict[int, float] = {}
    daily_total_history: list[float] = []
    current_exposure_target = 1.0
    idx = 0
    fin_paid = 0.0
    dates_list = list(close.index)

    for i, date in enumerate(dates_list):
        next_date = dates_list[i + 1] if i + 1 < len(dates_list) else None

        # === 일일 조달이자 (차입 = 음의 현금) ===
        if cash < 0:
            rate = float(fin_daily.get(date, fin_daily.iloc[-1])) if len(fin_daily) else 0.0
            interest = (-cash) * rate / 252.0
            cash -= interest
            fin_paid += interest

        # === 분기 리밸런싱 ===
        if idx < len(rebal_dates) and date >= rebal_dates[idx]:
            yearly_realized.setdefault(date.year, 0.0)
            try:
                hold_picks, _, adv20_h = select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    HOLD_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=p24.r1000_at_date, apply_adv20_gate=True)
                buy_picks, _, adv20_b = select_picks_phase24(
                    prices_w.loc[:date], spy_w.loc[:date], vix, date,
                    BUY_TOP, pit_membership, weight_schedule, sector_map,
                    bal_cache, inc_cache, cf_cache,
                    universe_fn=p24.r1000_at_date, apply_adv20_gate=True)
                adv20_map = {**adv20_h, **adv20_b}
            except Exception as e:
                hold_picks, buy_picks, adv20_map = [], [], {}

            hold_set = set(hold_picks)
            for ticker in [t for t in list(holdings) if t not in hold_set]:
                if ticker not in close.columns or pd.isna(close.at[date, ticker]):
                    holdings.pop(ticker, None)
                    continue
                shares = holdings[ticker]["shares"]
                price = close.at[date, ticker]
                cost = slip_frac(shares * price, adv20_map.get(ticker), True)
                proceeds, realized = sell_full(close, date, holdings, ticker, cost)
                cash += proceeds
                yearly_realized[date.year] += realized

            slots = BUY_TOP - len([t for t in holdings if t in hold_set])
            to_buy = [t for t in buy_picks if t not in holdings and t in close.columns][:slots]
            # 매수 예산 = exposure target 까지 (cash 부족분은 차입 허용)
            total, current_equity = current_value(close, date, holdings, cash)
            target_equity = total * current_exposure_target
            buy_budget = max(0.0, target_equity - current_equity)
            if slots > 0 and to_buy and buy_budget > 0:
                alloc = buy_budget / len(to_buy)
                spent = 0.0
                for ticker in to_buy:
                    cost = slip_frac(alloc, adv20_map.get(ticker), True)
                    spent += buy_ticker(close, date, holdings, ticker, alloc, cost)
                cash -= spent      # ← 차입 허용 (max(cash,0) 없음)

            cash, _, _, _ = enforce_cap(close, date, holdings, cash,
                                        SLIP_BASE_BPS / 10000.0, yearly_realized)
            idx += 1

        # === 월말 vol-managed 스케일링 (레버리지 캡 = max_exp) ===
        if is_month_end(date, next_date):
            realized_vol = compute_realized_vol(daily_total_history)
            if realized_vol is not None:
                new_target = target_exposure_lev(realized_vol, TARGET_VOL, max_exp)
                if abs(new_target - current_exposure_target) > 0.05:
                    current_exposure_target = new_target
                    cash = rescale_to_exposure_lev(
                        close, date, holdings, cash, current_exposure_target,
                        SLIP_BASE_BPS / 10000.0, yearly_realized)

        stock_equity = sum(
            holdings[t]["shares"] * close.at[date, t]
            for t in holdings if t in close.columns and pd.notna(close.at[date, t]))
        total_value = stock_equity + cash
        history.append({"date": date, "value": total_value})
        daily_total_history.append(total_value)

    curve = pd.Series([h["value"] for h in history],
                      index=[h["date"] for h in history])
    return curve, {"fin_paid": fin_paid}


# ============================================================
# main
# ============================================================

def main():
    print("=" * 116)
    print("Phase 27 (FRIDAY): JARVIS/SPACE 불장 레버리지 sweep — 2015-2025 live-faithful")
    print("=" * 116)

    print("\n[1/3] 데이터 로드 ...")
    prices = p24.load_combined_prices_r1000()
    spy = p26b.load_spy(START_DATE, END_DATE)
    vix = p26b.load_vix(START_DATE, END_DATE)
    bal, inc, cf = p26b._load_all_caches_v2()
    r1000_pit = p24.load_r1000_synthetic_pit_membership()
    _, full_map = p26b.build_full_sector_map()

    close_idx = arb.get_close_prices(prices.loc[START_DATE:END_DATE]).index
    print("  조달비용 시계열 구성 ...")
    fin_daily = build_financing_daily(close_idx, FIN_SPREAD)

    SWEEP = [1.00, 1.20, 1.35, 1.50, 2.00]
    rows = []

    print("\n[2/3] sweep (각 변종 = 풀 2015-2025 R1000 백테스트) ...")
    for mx in SWEEP:
        label = f"MAX_EXP={mx:.2f}"
        print(f"  running {label} (financing=IRX+{FIN_SPREAD*10000:.0f}bps) ...")
        curve, diag = simulate_lev(prices, spy, vix, r1000_pit,
                                   p24.SCHEDULE_C80_E20, full_map, bal, inc, cf,
                                   max_exp=mx, fin_daily=fin_daily)
        m = metrics(curve)
        m["variant"] = label
        m["max_exp"] = mx
        m["fin_paid"] = diag["fin_paid"]
        rows.append(m)
        print(f"    연 {m['annual']*100:5.2f}%  Sharpe {m['sharpe']:.3f}  "
              f"MDD {m['mdd']*100:6.2f}%  Calmar {m['calmar']:.3f}  Vol {m['vol']*100:.1f}%  "
              f"조달누적 ${m['fin_paid']:,.0f}")

    # 비교용: MAX_EXP=1.5 무료 조달 (조달비용 상한 영향)
    print(f"  running MAX_EXP=1.50 (financing=0, 상한 비교) ...")
    zero_fin = pd.Series(0.0, index=close_idx)
    curve0, diag0 = simulate_lev(prices, spy, vix, r1000_pit,
                                 p24.SCHEDULE_C80_E20, full_map, bal, inc, cf,
                                 max_exp=1.50, fin_daily=zero_fin)
    m0 = metrics(curve0)
    m0.update({"variant": "MAX_EXP=1.50 (free fin)", "max_exp": 1.50, "fin_paid": 0.0})
    rows.append(m0)
    print(f"    연 {m0['annual']*100:5.2f}%  Sharpe {m0['sharpe']:.3f}  "
          f"MDD {m0['mdd']*100:6.2f}%  Calmar {m0['calmar']:.3f}  Vol {m0['vol']*100:.1f}%")

    print("\n[3/3] 비교")
    print("=" * 116)
    print(f"{'variant':<26} | {'CAGR':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7} {'Vol':>6} | {'조달누적':>12}")
    print("-" * 116)
    base = rows[0]
    for m in rows:
        print(f"{m['variant']:<26} | {m['annual']*100:>6.2f}% {m['sharpe']:>7.3f} "
              f"{m['mdd']*100:>7.2f}% {m['calmar']:>7.3f} {m['vol']*100:>5.1f}% | "
              f"${m['fin_paid']:>11,.0f}")
    print("-" * 116)
    print("Δ vs MAX_EXP=1.00 (무차입 베이스):")
    for m in rows[1:]:
        print(f"  {m['variant']:<24} ΔCAGR {(m['annual']-base['annual'])*100:>+5.2f}pp  "
              f"ΔSharpe {m['sharpe']-base['sharpe']:>+6.3f}  "
              f"ΔMDD {(m['mdd']-base['mdd'])*100:>+6.2f}pp  "
              f"ΔCalmar {m['calmar']-base['calmar']:>+7.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = FRIDAY / "archive" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"phase27_leverage_sweep_{ts}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
