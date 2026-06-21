"""
Real sentence-transformers embedder using all-MiniLM-L6-v2 (384-dim).

Contract used by graph/stratos_graph.py:
    embed(text: str) -> list[float]   # JSON-serializable for hydradb storage
Trade-setup embedding for future use:
    embed_trade_setup(signals_json, ticker, regime) -> np.ndarray
"""
from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_trade_setup(
    signals_json: dict,
    ticker: str,
    regime: str,
) -> np.ndarray:
    signals = signals_json
    text = (
        f"Ticker: {ticker}. Regime: {regime}. "
        f"Sentiment direction: {signals.get('sentiment', {}).get('direction', 0.0):.2f}, "
        f"conviction: {signals.get('sentiment', {}).get('conviction', 0.0):.2f}. "
        f"Momentum direction: {signals.get('momentum', {}).get('direction', 0.0):.2f}. "
        f"Gamma direction: {signals.get('gamma', {}).get('direction', 0.0):.2f}. "
        f"GEX regime: {signals.get('gamma', {}).get('signals', {}).get('gex_regime', 'unknown')}. "
        f"OPEX week: {signals.get('momentum', {}).get('signals', {}).get('opex_flag', False)}"
    )
    model = load_model()
    return model.encode(text)  # np.ndarray(384,)


def embed(text: str) -> list[float]:
    if not isinstance(text, str):
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        else:
            text = str(text)
    model = load_model()
    return model.encode(text).tolist()  # list[float], JSON-safe


if __name__ == "__main__":
    signals = {
        "sentiment": {"direction": 0.6, "conviction": 0.8},
        "momentum": {"direction": 0.4, "signals": {"opex_flag": True}},
        "gamma": {"direction": -0.2, "signals": {"gex_regime": "positive"}},
    }
    v = embed_trade_setup(signals, "NVDA", "trend")
    assert v.shape == (384,), f"Expected (384,), got {v.shape}"
    assert v.dtype == np.float32 or v.dtype == np.float64, f"float dtype expected, got {v.dtype}"
    print(f"embed_trade_setup() shape={v.shape} dtype={v.dtype}")
    print(v)

    e = embed("hello world")
    assert len(e) == 384, f"Expected 384, got {len(e)}"
    assert isinstance(e, list), f"Expected list, got {type(e)}"
    assert isinstance(e[0], float), f"Expected float, got {type(e[0])}"
    print(f"embed() -> list[384] OK")
