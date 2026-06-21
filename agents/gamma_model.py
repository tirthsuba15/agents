"""
Person C's gamma model — stub for Phase 1.
Replace when Person C delivers gamma_model.py.
"""
from __future__ import annotations


def score_gamma(ticker: str, options_chain: dict | None = None) -> float:
    """
    Return gamma exposure signal in [-1.0, 1.0].
    Positive = dealers long gamma (dampening), negative = dealers short gamma (amplifying).
    Stub returns a fixed negative value (short gamma environment).
    """
    print(f"[Gamma STUB] scoring {ticker}")
    return -0.18
