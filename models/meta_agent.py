"""
Meta-Agent — signal aggregator for the StratOS trading system.

Combines three independent signal sources into one final trade decision:

  1. Momentum  (models.momentum_model.predict_momentum)
  2. Gamma     (models.gamma_model.get_composite_score)
  3. GEX regime(models.gex_engine.get_gex_regime)

Regime gates how much weight momentum vs. gamma receives and scales the final
conviction.  Position size is set with a fractional-Kelly rule (k=0.25, capped
at 20%).

Each signal source is wrapped in try/except so that one failing source degrades
gracefully rather than killing the whole signal.

CLI:
    python models/meta_agent.py
    python models/meta_agent.py AAPL MSFT NVDA
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# Make `models.*` / `data.*` imports resolve when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.momentum_model import predict_momentum, get_live_features  # noqa: E402

try:
    from models.gamma_model import get_composite_score  # noqa: E402
except ImportError:  # pragma: no cover - graceful until gamma_model exposes it
    def get_composite_score(symbol: str, r: float = None, vix_level: float = None) -> dict:
        """Fallback shim used if gamma_model.get_composite_score is unavailable.

        Treated as a failed gamma source by the aggregator (neutral signal).
        """
        raise RuntimeError("get_composite_score unavailable in models.gamma_model")

from models.gex_engine import (                                        # noqa: E402
    fetch_options_chain,
    fetch_spot_and_div,
    compute_gex,
    find_zero_gamma_flip,
    get_gex_regime,
)

try:
    from data.fred_client import get_risk_free_rate  # noqa: E402
except Exception:  # pragma: no cover - defensive: keep module importable
    def get_risk_free_rate() -> float:
        return 0.04


DEFAULT_SYMBOLS = ["SPY", "AAPL", "MSFT", "NVDA", "JPM"]

# Regime-conditioned weighting of the two directional sources.
REGIME_WEIGHTS = {
    "positive_gamma": {"momentum": 0.5, "gamma": 0.5},
    "negative_gamma": {"momentum": 0.7, "gamma": 0.3},
}

KELLY_FRACTION = 0.25      # fractional Kelly multiplier
MAX_POSITION_SIZE = 0.20   # hard cap on position size (20%)
CONVICTION_FLOOR = 0.25    # below this -> HOLD
DIRECTION_THRESHOLD = 0.10 # |direction| above which we act


# ---------------------------------------------------------------------------
# Individual signal fetchers (each isolated so one failure can't break others)
# ---------------------------------------------------------------------------

def _get_momentum_signal(momentum_features: dict | None) -> dict:
    """
    Return the full predict_momentum output, or a neutral fallback on failure.
    """
    fallback = {
        "model": "momentum_unavailable",
        "direction": 0.0,
        "conviction": 0.0,
        "error": None,
    }
    if not momentum_features:
        fallback["error"] = "no momentum_features supplied"
        return fallback
    try:
        return predict_momentum(momentum_features)
    except Exception as exc:  # noqa: BLE001
        fallback["error"] = str(exc)
        return fallback


def _get_gamma_signal(symbol: str, r: float | None) -> dict:
    """
    Return get_composite_score output, or a neutral fallback on failure.
    """
    try:
        return get_composite_score(symbol, r=r)
    except Exception as exc:  # noqa: BLE001
        return {
            "symbol": symbol,
            "score": 0.0,
            "conviction": 0.0,
            "regime": "unknown",
            "error": str(exc),
        }


def _get_gex_regime(symbol: str, r: float | None) -> str:
    """
    Compute the live GEX regime for `symbol`.

    This is expensive (options chain + spot + grid scan), so the entire block
    is wrapped in try/except.  On any failure we default to 'negative_gamma'
    (the momentum-friendly regime).
    """
    try:
        if r is None:
            r = get_risk_free_rate()
        chain = fetch_options_chain(symbol)
        if not chain:
            return "negative_gamma"
        spot, q = fetch_spot_and_div(symbol)
        net_gex = compute_gex(chain, spot, r, q)
        flip = find_zero_gamma_flip(chain, spot, r, q)
        return get_gex_regime(net_gex, spot, flip)
    except Exception:  # noqa: BLE001
        return "negative_gamma"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_signal(symbol: str, momentum_features: dict | None = None) -> dict:
    """
    Combine momentum, gamma, and GEX-regime signals into one trade decision.

    Returns a dict with action / direction / conviction / position_size plus
    the raw component signals for transparency.
    """
    # Shared risk-free rate (used by both gamma and GEX); fall back gracefully.
    try:
        r = get_risk_free_rate()
    except Exception:  # noqa: BLE001
        r = None

    gex_regime = _get_gex_regime(symbol, r)
    momentum_signal = _get_momentum_signal(momentum_features)
    gamma_signal = _get_gamma_signal(symbol, r)

    weights = REGIME_WEIGHTS.get(gex_regime, REGIME_WEIGHTS["negative_gamma"])
    momentum_weight = weights["momentum"]
    gamma_weight = weights["gamma"]

    momentum_direction = float(momentum_signal.get("direction", 0.0) or 0.0)
    momentum_conviction = float(momentum_signal.get("conviction", 0.0) or 0.0)
    gamma_score = float(gamma_signal.get("score", 0.0) or 0.0)
    gamma_conviction = float(gamma_signal.get("conviction", 0.0) or 0.0)

    # Composite direction.
    direction = momentum_weight * momentum_direction + gamma_weight * gamma_score

    # Conviction scaling — amplified in negative-gamma (momentum) regimes.
    base_conviction = (
        momentum_weight * momentum_conviction + gamma_weight * gamma_conviction
    )
    if gex_regime == "negative_gamma":
        final_conviction = base_conviction * 1.2
    else:
        final_conviction = base_conviction * 0.85
    final_conviction = min(1.0, final_conviction)

    # Final decision.
    if final_conviction < CONVICTION_FLOOR:
        action = "HOLD"
    elif direction > DIRECTION_THRESHOLD:
        action = "LONG"
    elif direction < -DIRECTION_THRESHOLD:
        action = "SHORT"
    else:
        action = "HOLD"

    # Fractional-Kelly position sizing.
    win_prob = (direction + 1) / 2   # map [-1, +1] -> [0, 1]
    odds = 1.0                        # 1:1 payout assumption
    kelly = (win_prob * (odds + 1) - 1) / odds
    position_size = max(0.0, kelly * KELLY_FRACTION * final_conviction)
    position_size = round(min(MAX_POSITION_SIZE, position_size), 4)

    return {
        "symbol": symbol,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "direction": round(direction, 4),
        "conviction": round(final_conviction, 4),
        "position_size": position_size,
        "gex_regime": gex_regime,
        "momentum_signal": momentum_signal,
        "gamma_signal": gamma_signal,
        "components": {
            "momentum_direction": round(momentum_direction, 4),
            "gamma_score": round(gamma_score, 4),
            "momentum_weight": momentum_weight,
            "gamma_weight": gamma_weight,
        },
    }


# ---------------------------------------------------------------------------
# Live batch runner
# ---------------------------------------------------------------------------

def run_live_signal(symbols: list[str] | None = None) -> list[dict]:
    """
    Build live momentum features for each symbol, aggregate signals, and return
    the results sorted by conviction (highest first).
    """
    if symbols is None:
        symbols = list(DEFAULT_SYMBOLS)

    results: list[dict] = []
    for symbol in symbols:
        try:
            features = get_live_features(symbol)
        except Exception:  # noqa: BLE001
            features = None
        results.append(aggregate_signal(symbol, momentum_features=features))

    results.sort(key=lambda r: r.get("conviction", 0.0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or None
    signals = run_live_signal(symbols)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"META-AGENT SIGNALS  {now}")
    print(
        f"{'Symbol':<8} {'Action':<6} {'Direction':>9}  {'Conviction':>10}  "
        f"{'Size':>6}  {'Regime'}"
    )
    print(
        f"{'------':<8} {'------':<6} {'---------':>9}  {'----------':>10}  "
        f"{'----':>6}  {'------'}"
    )
    for sig in signals:
        print(
            f"{sig['symbol']:<8} "
            f"{sig['action']:<6} "
            f"{sig['direction']:>+9.4f}  "
            f"{sig['conviction']:>10.4f}  "
            f"{sig['position_size'] * 100:>5.1f}%  "
            f"{sig['gex_regime']}"
        )


if __name__ == "__main__":
    main()
