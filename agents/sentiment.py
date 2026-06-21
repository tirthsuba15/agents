"""
Sentiment Agent — Phase 2.
Model: Llama 3.3 70B Fast on Nebius Token Factory (OpenAI-compatible).
Fetches Finnhub company news, VIX, and event risk, then calls the model
to produce a structured SignalObject.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, UTC, timedelta
from typing import TypedDict

import finnhub
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_MODEL = "meta-llama/Llama-3.3-70B-Instruct-fast"
_MODEL_VERSION = "llama-3.3-70b-fast-v1"

# ── Lazy clients — initialized on first use so missing keys degrade gracefully ──
_nebius: OpenAI | None = None
_fh: finnhub.Client | None = None


def _get_nebius() -> OpenAI:
    global _nebius
    if _nebius is None:
        api_key = os.environ.get("NEBIUS_API_KEY") or "sk-placeholder"
        _nebius = OpenAI(
            base_url="https://api.tokenfactory.nebius.com",
            api_key=api_key,
        )
    return _nebius


def _get_finnhub() -> finnhub.Client:
    global _fh
    if _fh is None:
        _fh = finnhub.Client(api_key=os.environ.get("FINNHUB_API_KEY", ""))
    return _fh


# ────────────────────────────────────────────────────────────────
# SignalObject schema
# ────────────────────────────────────────────────────────────────

class SignalObject(TypedDict):
    agent_id: str
    ticker: str
    timestamp: str          # ISO-8601 string for JSON serializability
    direction: float        # -1.0 to +1.0
    conviction: float       # 0.0 to 1.0
    regime: str             # 'mean_revert' | 'momentum' | 'neutral'
    horizon_mins: int       # 30 = intraday, 10080 = weekly
    signals: dict
    model_version: str


def _fallback_signal(ticker: str, reason: str = "parse_error") -> SignalObject:
    return SignalObject(
        agent_id="sentiment",
        ticker=ticker,
        timestamp=datetime.now(UTC).isoformat(),
        direction=0.0,
        conviction=0.1,
        regime="neutral",
        horizon_mins=30,
        signals={"fallback_reason": reason},
        model_version=_MODEL_VERSION,
    )


# ────────────────────────────────────────────────────────────────
# Finnhub data fetchers
# ────────────────────────────────────────────────────────────────

def _fetch_news(ticker: str, lookback_hours: int = 24) -> list[dict]:
    """Fetch last N hours of company news from Finnhub."""
    now = datetime.now(UTC)
    from_dt = now - timedelta(hours=lookback_hours)
    try:
        articles = _get_finnhub().company_news(
            ticker,
            _from=from_dt.strftime("%Y-%m-%d"),
            to=now.strftime("%Y-%m-%d"),
        )
        return articles[:10] if articles else []
    except Exception as e:
        print(f"[sentiment] news fetch failed: {e}", file=sys.stderr)
        return []


def _fetch_vix() -> float | None:
    """Fetch current VIX from Finnhub quote endpoint."""
    try:
        q = _get_finnhub().quote("^VIX")
        return q.get("c")  # current price
    except Exception as e:
        print(f"[sentiment] VIX fetch failed: {e}", file=sys.stderr)
        return None


def _check_event_risk(ticker: str, horizon_days: int = 5) -> bool:
    """Return True if earnings OR macro event falls within horizon_days."""
    now = datetime.now(UTC)
    to_dt = now + timedelta(days=horizon_days)
    from_str = now.strftime("%Y-%m-%d")
    to_str = to_dt.strftime("%Y-%m-%d")

    try:
        earnings = _get_finnhub().earnings_calendar(_from=from_str, to=to_str, symbol=ticker)
        if earnings and earnings.get("earningsCalendar"):
            return True
    except Exception as e:
        print(f"[sentiment] earnings calendar failed: {e}", file=sys.stderr)

    try:
        econ = _get_finnhub().calendar_economic()
        events = econ.get("economicCalendar", [])
        for ev in events:
            ev_date_str = ev.get("time", "")
            if not ev_date_str:
                continue
            try:
                ev_date = datetime.fromisoformat(ev_date_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if now <= ev_date <= to_dt and ev.get("impact", "low") in ("high", "medium"):
                return True
    except Exception as e:
        # 403 = free tier; treat as no macro events detected
        print(f"[sentiment] economic calendar unavailable (plan limit): skipping", file=sys.stderr)

    return False


# ────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a quantitative sentiment analysis agent for equities trading.
You MUST respond with a single valid JSON object and nothing else — no markdown, no explanation.

The JSON must have exactly these fields:
{
  "direction": <float -1.0 to 1.0>,
  "conviction": <float 0.0 to 1.0>,
  "regime": <"mean_revert" | "momentum" | "neutral">,
  "horizon_mins": <integer — 30 for intraday signals, 10080 for weekly>,
  "headline_sentiment": <float -1.0 to 1.0>,
  "event_risk": <boolean>,
  "reasoning": <one sentence max>
}

Guidelines:
- direction > 0 = bullish, < 0 = bearish
- conviction is your confidence level (0 = random, 1 = near-certain)
- regime: "momentum" if news creates a trend, "mean_revert" if overreaction likely, "neutral" otherwise
- Higher VIX → reduce conviction; event risk within 5 days → flag event_risk=true and reduce horizon_mins"""


def _build_user_prompt(
    ticker: str,
    articles: list[dict],
    vix: float | None,
    event_risk: bool,
) -> str:
    lines = [f"Ticker: {ticker}", ""]

    lines.append("Recent news (last 24h):")
    if articles:
        for i, a in enumerate(articles[:10], 1):
            headline = a.get("headline", "N/A")
            summary = a.get("summary", "")[:200]
            lines.append(f"{i}. {headline}")
            if summary:
                lines.append(f"   {summary}")
    else:
        lines.append("No news available in the last 24 hours.")

    lines.append("")
    vix_str = f"{vix:.1f}" if vix is not None else "unavailable"
    lines.append(f"Current VIX: {vix_str}")
    lines.append(f"Earnings/macro event within 5 days: {event_risk}")
    lines.append("")
    lines.append("Respond with the JSON object only.")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Main scoring function
# ────────────────────────────────────────────────────────────────

def score_sentiment(ticker: str, news_items: list[dict] | None = None) -> SignalObject:
    """
    Fetch market context and call Llama 3.3 70B Fast to score sentiment.
    Returns a fully-typed SignalObject. Never raises — falls back on error.
    """
    articles = news_items if news_items is not None else _fetch_news(ticker)
    vix = _fetch_vix()
    event_risk = _check_event_risk(ticker)

    user_prompt = _build_user_prompt(ticker, articles, vix, event_risk)

    try:
        response = _get_nebius().chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps output despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        direction = float(parsed.get("direction", 0.0))
        conviction = float(parsed.get("conviction", 0.1))
        regime = parsed.get("regime", "neutral")
        horizon_mins = int(parsed.get("horizon_mins", 30))
        headline_sentiment = float(parsed.get("headline_sentiment", direction))

        # Clamp values to spec
        direction = max(-1.0, min(1.0, direction))
        conviction = max(0.0, min(1.0, conviction))
        if regime not in ("mean_revert", "momentum", "neutral"):
            regime = "neutral"

        return SignalObject(
            agent_id="sentiment",
            ticker=ticker,
            timestamp=datetime.now(UTC).isoformat(),
            direction=direction,
            conviction=conviction,
            regime=regime,
            horizon_mins=horizon_mins,
            signals={
                "headline_sentiment": headline_sentiment,
                "event_risk": bool(parsed.get("event_risk", event_risk)),
                "vix": vix,
                "n_articles": len(articles),
                "reasoning": parsed.get("reasoning", ""),
            },
            model_version=_MODEL_VERSION,
        )

    except json.JSONDecodeError as e:
        print(f"[sentiment] JSON parse failed: {e}", file=sys.stderr)
        return _fallback_signal(ticker, reason="json_parse_error")
    except Exception as e:
        err_str = str(e)
        # HTTP errors often return HTML bodies; detect and normalize
        reason = "http_error" if err_str.lstrip().startswith("<") else err_str[:120]
        print(f"[sentiment] model call failed: {reason}", file=sys.stderr)
        return _fallback_signal(ticker, reason=reason)


# ────────────────────────────────────────────────────────────────
# Acceptance test entrypoint
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"[sentiment] running for {ticker} ...", file=sys.stderr)

    result = score_sentiment(ticker)

    print(json.dumps(result, indent=2))
