"""
Person B's real embedder using all-MiniLM-L6-v2 (384-dim).
embed(text) -> list[float]  — JSON-safe, used by hydradb for trade setup storage.
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed(text: str) -> list[float]:
    if not isinstance(text, str):
        text = json.dumps(text) if isinstance(text, (dict, list)) else str(text)
    return _load_model().encode(text).tolist()


def embed_trade_setup(signals_json: dict, ticker: str, regime: str) -> np.ndarray:
    """Structured text embedding for trade setups (returns np.ndarray for similarity math)."""
    text = (
        f"Ticker: {ticker}. Regime: {regime}. "
        f"Sentiment direction: {signals_json.get('sentiment', {}).get('direction', 0.0):.2f}, "
        f"conviction: {signals_json.get('sentiment', {}).get('conviction', 0.0):.2f}. "
        f"Momentum direction: {signals_json.get('momentum', {}).get('direction', 0.0):.2f}. "
        f"Gamma direction: {signals_json.get('gamma', {}).get('direction', 0.0):.2f}."
    )
    return _load_model().encode(text)
