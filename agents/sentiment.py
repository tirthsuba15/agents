"""
Sentiment Agent — Phase 1 stub.
Phase 2: wire Finnhub news client + real NLP scoring.
"""
from __future__ import annotations


def score_sentiment(ticker: str, news_items: list[dict] | None = None) -> float:
    """
    Return sentiment score in [-1.0, 1.0].
    Positive = bullish, negative = bearish.
    Stub returns a fixed mild-bullish score.
    """
    print(f"[Sentiment STUB] scoring {ticker}, {len(news_items or [])} news items")
    return 0.42
