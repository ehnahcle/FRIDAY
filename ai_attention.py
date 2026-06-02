"""
ai_attention.py
===============
FRIDAY SURGE — AI 강화 "attention/narrative" 모듈.

아이디어(사용자): "사람들이 관심 가지면 주가 오른다 → AI가 trend 를 스캔."
research(angle 6): Da-Engelberg-Gao "In Search of Attention" — 관심 spike 가 단기 상승 선행.

이 모듈이 하는 일 (스크리너 top-N candidate 에 대해):
  1. 최근 뉴스 수집 (FMP news/stock) — headline 텍스트 + 빈도.
  2. 정량 attention proxy: news_count spike(최근 7d vs 30d) + RVOL(가격에서).
  3. **Claude 가 narrative 스캔** → attention_score(0~100) + narrative_momentum(rising/flat/fading)
     + catalyst_type + 한 줄 thesis. (정성 — 빌딩 중인 스토리/관심 읽기)
  4. blend → attention_score, surge_score 의 re-ranker 로 사용.

⚠️  설계상 LIVE 전용 re-ranker. 백테 미포함(29k history row LLM 스코어링 비현실 + 뉴스 PIT 보장 어려움).
    ANTHROPIC key/SDK 없으면 proxy-only(news+rvol)로 graceful degrade.

CLI: python ai_attention.py [--top N] [--model haiku|sonnet]  → surge_results.csv 상위 N 스캔.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ENV = dotenv_values(str(Path(__file__).resolve().parent / ".env"))
FMP_API_KEY = _ENV.get("FMP_API_KEY") or os.getenv("FMP_API_KEY", "")
ANTHROPIC_API_KEY = _ENV.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
CACHE = Path.home() / ".friday_cache" / "surge"
CACHE.mkdir(parents=True, exist_ok=True)

MODELS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6"}

# attention_score blend: AI narrative 60% + news-spike 25% + rvol 15%
W_AI, W_NEWS, W_RVOL = 0.60, 0.25, 0.15

SYS_PROMPT = """You are a short-term equity surge analyst. For each ticker you receive recent news headlines plus two quantitative attention signals (news_spike = recent vs baseline news volume, rvol = today's volume vs 50-day average).

Your job: judge whether RETAIL/MARKET ATTENTION and a tradeable NARRATIVE are BUILDING (which precedes short-term surges) vs already-exhausted/fading.

For each ticker return a JSON object with:
- "ticker": the symbol
- "attention_score": 0-100. High = a fresh, building, catalyst-driven narrative drawing rising interest. Low = stale, no story, or post-spike exhaustion.
- "narrative": one of "rising" | "flat" | "fading"
- "catalyst_type": short tag, e.g. "FDA/trial", "earnings", "M&A", "product/AI", "short-squeeze", "macro/sector", "none"
- "thesis": one concise sentence (<=20 words) on why attention is/ isn't building.

Be skeptical: pump-and-dump and already-spiked names should score LOW (attention exhausted). No headlines + low signals = low score. Reward genuine, recent, forward-looking catalysts.

Return ONLY a JSON array of these objects, one per ticker, no prose."""


# ============================================================
# 캐시 + 뉴스
# ============================================================
def _cached(name: str, hours: float):
    p = CACHE / f"{name}.pkl"
    if p.exists() and (datetime.now().timestamp() - p.stat().st_mtime) < hours * 3600:
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save(name: str, data):
    with open(CACHE / f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)


def fetch_news(symbol: str, limit: int = 30, cache_hours: float = 6) -> List[Dict]:
    c = _cached(f"news_{symbol}", cache_hours)
    if c is not None:
        return c
    try:
        r = requests.get(f"{FMP_BASE}/news/stock",
                         params={"symbols": symbol, "limit": limit, "apikey": FMP_API_KEY}, timeout=20)
        d = r.json() if r.status_code == 200 else []
    except Exception:
        d = []
    d = d if isinstance(d, list) else []
    _save(f"news_{symbol}", d)
    return d


def news_features(news: List[Dict]) -> Dict:
    """news_spike = 최근 7d 건수 / (이전 8~30d 일평균 ×7). titles = 최근 헤드라인."""
    now = datetime.now()
    recent7, prior = 0, 0
    titles = []
    for n in news:
        pd_ = n.get("publishedDate") or ""
        try:
            dt = datetime.fromisoformat(pd_.replace("Z", "")[:19])
        except Exception:
            continue
        age = (now - dt).days
        if age <= 7:
            recent7 += 1
            if len(titles) < 8:
                titles.append(n.get("title", "")[:140])
        elif age <= 30:
            prior += 1
    base_per7 = (prior / 23 * 7) if prior > 0 else 0.3   # 8~30d=23일
    spike = recent7 / base_per7 if base_per7 > 0 else (recent7 / 0.3)
    return {"news_7d": recent7, "news_spike": round(spike, 2), "titles": titles}


# ============================================================
# Claude narrative 스캔 (batch)
# ============================================================
def _claude_batch(client, model: str, items: List[Dict]) -> List[Dict]:
    payload = [{
        "ticker": it["ticker"], "news_spike": it["news_spike"],
        "rvol": it.get("rvol"), "headlines": it["titles"] or ["(no recent news)"],
    } for it in items]
    user = "Score these tickers:\n" + json.dumps(payload, ensure_ascii=False)
    try:
        resp = client.messages.create(
            model=model, max_tokens=3000,
            system=[{"type": "text", "text": SYS_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        txt = resp.content[0].text.strip()
        s, e = txt.find("["), txt.rfind("]")
        if s >= 0 and e > s:
            return json.loads(txt[s:e + 1])
    except Exception as ex:
        logger.warning(f"Claude batch failed: {ex}")
    return []


def score_attention(
    tickers: List[str], rvol_map: Optional[Dict[str, float]] = None,
    model_key: str = "haiku", batch_size: int = 8,
) -> pd.DataFrame:
    rvol_map = rvol_map or {}

    # 1. 뉴스 + proxy
    items = []
    for i, t in enumerate(tickers):
        if i % 10 == 0:
            logger.info(f"  news fetch {i}/{len(tickers)}")
        nf = news_features(fetch_news(t))
        items.append({"ticker": t, "rvol": rvol_map.get(t), **nf})
        time.sleep(0.03)

    # 2. Claude narrative (가능 시)
    ai_rows: Dict[str, Dict] = {}
    have_ai = False
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            model = MODELS.get(model_key, MODELS["haiku"])
            for b in range(0, len(items), batch_size):
                batch = items[b:b + batch_size]
                logger.info(f"  Claude scan {b}/{len(items)} ({model_key})")
                for r in _claude_batch(client, model, batch):
                    if isinstance(r, dict) and r.get("ticker"):
                        ai_rows[str(r["ticker"]).upper()] = r
            have_ai = len(ai_rows) > 0
        except Exception as ex:
            logger.warning(f"anthropic 사용 불가 — proxy-only: {ex}")
    else:
        logger.warning("ANTHROPIC_API_KEY 없음 — proxy-only attention")

    # 3. blend
    rows = []
    for it in items:
        t = it["ticker"]
        ai = ai_rows.get(t.upper(), {})
        ai_score = ai.get("attention_score")
        news_spike_score = float(np.clip(it["news_spike"] / 3.0, 0, 1) * 100)  # spike 3x→100
        rvol = it.get("rvol")
        rvol_score = float(np.clip(((rvol or 1.0) - 0.8) / 1.7, 0, 1) * 100)   # rvol 0.8→0, 2.5→100
        if have_ai and ai_score is not None:
            attention = W_AI * float(ai_score) + W_NEWS * news_spike_score + W_RVOL * rvol_score
        else:  # proxy-only (AI 없을 때 재정규화)
            attention = (W_NEWS * news_spike_score + W_RVOL * rvol_score) / (W_NEWS + W_RVOL)
        rows.append({
            "ticker": t, "attention_score": round(attention, 1),
            "ai_attention": ai_score, "narrative": ai.get("narrative"),
            "catalyst_type": ai.get("catalyst_type"), "thesis": ai.get("thesis"),
            "news_7d": it["news_7d"], "news_spike": it["news_spike"], "rvol": rvol,
        })
    return pd.DataFrame(rows).sort_values("attention_score", ascending=False).reset_index(drop=True)


def blend_into_surge(surge_df: pd.DataFrame, att_df: pd.DataFrame, w_attention: float = 0.25) -> pd.DataFrame:
    """surge_score 와 attention_score 를 blend → surge_ai_score 로 re-rank."""
    m = surge_df.merge(att_df, on="ticker", how="left")
    m["attention_score"] = m["attention_score"].fillna(40.0)  # 무뉴스 = mild
    m["surge_ai_score"] = (1 - w_attention) * m["surge_score"] + w_attention * m["attention_score"]
    return m.sort_values("surge_ai_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FRIDAY SURGE AI attention re-ranker")
    ap.add_argument("--top", type=int, default=20, help="surge_results.csv 상위 N 스캔")
    ap.add_argument("--model", default="haiku", choices=list(MODELS.keys()))
    ap.add_argument("--w", type=float, default=0.25, help="attention blend weight")
    args = ap.parse_args()

    csv = Path(__file__).resolve().parent / "surge_results.csv"
    if not csv.exists():
        print("surge_results.csv 없음 — 먼저 surge_screener.py 실행."); raise SystemExit(1)
    sdf = pd.read_csv(csv).head(args.top)
    rvol_map = dict(zip(sdf["ticker"], sdf.get("rvol", pd.Series([None] * len(sdf)))))

    att = score_attention(sdf["ticker"].tolist(), rvol_map=rvol_map, model_key=args.model)
    blended = blend_into_surge(sdf, att, w_attention=args.w)
    out = Path(__file__).resolve().parent / "surge_attention.csv"
    keep = ["ticker", "name", "surge_ai_score", "surge_score", "attention_score",
            "ai_attention", "narrative", "catalyst_type", "news_7d", "news_spike", "rvol", "thesis"]
    keep = [c for c in keep if c in blended.columns]
    blended[keep].to_csv(out, index=False)

    print("\n" + "=" * 100)
    print(f"  FRIDAY SURGE + AI ATTENTION re-rank (model={args.model}, w={args.w})")
    print("=" * 100)
    for i, r in blended.head(args.top).iterrows():
        nar = (r.get("narrative") or "?")[:6]
        cat = (r.get("catalyst_type") or "?")[:12]
        print(f"{i+1:>2}. {r['ticker']:<6} ai_surge={r['surge_ai_score']:5.1f} "
              f"(surge={r['surge_score']:4.0f} att={r.get('attention_score',float('nan')):4.0f}) "
              f"{nar:<6} {cat:<12} news7d={int(r.get('news_7d') or 0):>2} "
              f"spike={r.get('news_spike',0):.1f}  {str(r.get('thesis') or '')[:50]}")
    print("=" * 100)
    print("⚠️  LIVE re-ranker only (백테 미검증). attention 은 surge 발생 선행 가설 — exit discipline 동일.")
    print("=" * 100)
