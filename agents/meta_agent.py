"""
Meta-Agent — Phase 3.
Model: Qwen3 235B on Nebius Serverless AI (OpenAI-compatible).
Fuses sentiment + momentum + gamma signals, queries RAG context,
applies hard gates, then asks Qwen3 whether to trade.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, UTC, timedelta
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydradb
import embedder
from agents.sentiment import SignalObject, _check_event_risk, _fetch_vix
from execution.alpaca import submit_order

# ── Nebius Serverless AI client (Qwen3 235B) ────────────────────
_NEBIUS_ENDPOINT = os.environ.get(
    "NEBIUS_SERVERLESS_ENDPOINT",
    "https://api.studio.nebius.ai/v1",   # default Nebius Serverless endpoint
)
_MODEL = "Qwen/Qwen3-235B-A22B"
_MODEL_VERSION = "qwen3-235b-a22b-v1"

_qwen: OpenAI | None = None


def _get_qwen() -> OpenAI:
    global _qwen
    if _qwen is None:
        api_key = os.environ.get("NEBIUS_API_KEY") or "sk-placeholder"
        _qwen = OpenAI(base_url=_NEBIUS_ENDPOINT, api_key=api_key)
    return _qwen


# ────────────────────────────────────────────────────────────────
# Output types
# ────────────────────────────────────────────────────────────────

class MetaDecision(TypedDict):
    action: str            # BUY | SELL | PASS
    size_pct: float        # 0–10 % of portfolio
    rationale: str
    conviction: float      # pre-computed C
    hard_gate: str | None  # gate that triggered, or None
    model_version: str
    timestamp: str


# ────────────────────────────────────────────────────────────────
# Hard gates
# ────────────────────────────────────────────────────────────────

_VIX_CEILING = 35.0
_CONVICTION_MIN = 0.35

# OPEX = third Friday of each month
def _is_opex_week(ref: datetime) -> bool:
    """Return True if ref falls in OPEX week (Mon–Fri of third Friday)."""
    year, month = ref.year, ref.month
    fridays = [
        d for d in range(1, 32)
        if (d <= 28 or datetime(year, month, d).month == month)
        and datetime(year, month, d).weekday() == 4   # Friday
    ]
    if len(fridays) < 3:
        return False
    third_friday = datetime(year, month, fridays[2], tzinfo=UTC)
    # OPEX week: Mon through Fri that contains third Friday
    week_start = third_friday - timedelta(days=third_friday.weekday())
    week_end = week_start + timedelta(days=4)
    return week_start.date() <= ref.date() <= week_end.date()


def _has_fomc_cpi_in_opex(ticker: str) -> bool:
    """Check for FOMC or CPI event within OPEX week. Graceful on plan limit."""
    try:
        from agents.sentiment import _get_finnhub
        econ = _get_finnhub().calendar_economic()
        events = econ.get("economicCalendar", [])
        now = datetime.now(UTC)
        for ev in events:
            name = ev.get("event", "").upper()
            if "FOMC" not in name and "CPI" not in name:
                continue
            ev_date_str = ev.get("time", "")
            if not ev_date_str:
                continue
            try:
                ev_date = datetime.fromisoformat(ev_date_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ev_date >= now and _is_opex_week(ev_date):
                return True
    except Exception:
        pass  # free tier 403 → treat as no FOMC/CPI detected
    return False


def check_hard_gates(vix: float | None, ticker: str) -> str | None:
    """
    Return a gate name (str) if any hard gate fires, else None.
    Gates evaluated in priority order.
    """
    if vix is not None and vix > _VIX_CEILING:
        return f"vix_too_high ({vix:.1f} > {_VIX_CEILING})"

    if _check_event_risk(ticker, horizon_days=5):
        return "earnings_within_5_days"

    if _has_fomc_cpi_in_opex(ticker):
        return "fomc_cpi_in_opex_week"

    return None


# ────────────────────────────────────────────────────────────────
# Conviction computation
# ────────────────────────────────────────────────────────────────

def _extract_direction(sig: SignalObject | float | None) -> float:
    if sig is None:
        return 0.0
    if isinstance(sig, dict):
        return float(sig.get("direction", 0.0))
    return float(sig)


def compute_conviction(signals: dict, weights: dict) -> float:
    """Weighted sum of signal directions. Range: approximately -1 to +1."""
    d_s = _extract_direction(signals.get("sentiment"))
    d_m = _extract_direction(signals.get("momentum"))
    d_g = _extract_direction(signals.get("gamma"))

    w_s = weights.get("w_sentiment", 0.33)
    w_m = weights.get("w_momentum", 0.33)
    w_g = weights.get("w_gamma", 0.34)

    return (w_s * d_s) + (w_m * d_m) + (w_g * d_g)


# ────────────────────────────────────────────────────────────────
# VRP sizing guard
# ────────────────────────────────────────────────────────────────

def enforce_vrp_sizing(decision: dict, gamma_signal: SignalObject | float | None) -> dict:
    """
    If Gamma agent flagged a VRP (Variance Risk Premium) options trade:
    - Cap size_pct at 10%
    - Force defined_risk=True (never short naked options)
    """
    is_vrp = False
    if isinstance(gamma_signal, dict):
        is_vrp = gamma_signal.get("signals", {}).get("vrp_trade", False)

    if is_vrp:
        decision["size_pct"] = min(float(decision.get("size_pct", 0)), 10.0)
        decision["defined_risk"] = True
        decision["naked_options"] = False   # explicit safety flag
        print(f"[meta] VRP sizing enforced: size_pct={decision['size_pct']}%, defined-risk only")

    return decision


# ────────────────────────────────────────────────────────────────
# Prompt builder
# ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a quantitative trading meta-agent. You receive fused signals from three sub-agents,
historical context from a RAG retrieval, and a pre-computed conviction score.
You MUST respond with a single valid JSON object and nothing else.
Do NOT include <think> blocks, markdown, or any explanation outside the JSON.

Required JSON fields:
{
  "action": "BUY" | "SELL" | "PASS",
  "size_pct": <float 0.0–10.0 — percentage of portfolio to deploy>,
  "rationale": "<one concise sentence>"
}

Constraints:
- If conviction is weak (|C| < 0.35), output action=PASS with size_pct=0.
- size_pct must be 0 when action=PASS.
- Never recommend size_pct > 10.
- For options structures, only recommend defined-risk trades (spreads, not naked short)."""


def _format_signal(label: str, sig: SignalObject | float | None) -> str:
    if sig is None:
        return f"{label}: unavailable"
    if isinstance(sig, dict):
        return (
            f"{label}: direction={sig.get('direction', 0):.3f}, "
            f"conviction={sig.get('conviction', 0):.3f}, "
            f"regime={sig.get('regime', 'unknown')}, "
            f"horizon={sig.get('horizon_mins', '?')}min, "
            f"signals={json.dumps(sig.get('signals', {}), default=str)}"
        )
    return f"{label}: direction={float(sig):.3f}"


def build_prompt(
    ticker: str,
    signals: dict,
    weights: dict,
    conviction: float,
    rag_results: list[dict],
    vix: float | None,
) -> str:
    lines = [
        f"Ticker: {ticker}",
        f"Timestamp: {datetime.now(UTC).isoformat()}",
        "",
        "=== AGENT SIGNALS ===",
        _format_signal("Sentiment", signals.get("sentiment")),
        _format_signal("Momentum", signals.get("momentum")),
        _format_signal("Gamma", signals.get("gamma")),
        _format_signal("ZeroDTE", signals.get("zero_dte")),
        "",
        "=== WEIGHTS ===",
        f"w_sentiment={weights.get('w_sentiment', 0.33):.3f}, "
        f"w_momentum={weights.get('w_momentum', 0.33):.3f}, "
        f"w_gamma={weights.get('w_gamma', 0.34):.3f}",
        "",
        f"=== PRE-COMPUTED CONVICTION ===",
        f"C = {conviction:.4f}  (|C| {'>' if abs(conviction) > _CONVICTION_MIN else '<='} {_CONVICTION_MIN} threshold)",
        "",
        f"=== MARKET CONTEXT ===",
        f"VIX: {f'{vix:.1f}' if vix is not None else 'unavailable'}",
    ]

    if rag_results:
        lines += ["", "=== RAG: TOP SIMILAR PAST SETUPS ==="]
        for i, r in enumerate(rag_results[:5], 1):
            lines.append(f"{i}. {json.dumps(r, default=str)}")
    else:
        lines += ["", "=== RAG: No past similar setups found ==="]

    lines += [
        "",
        f"Given these signals, RAG context, and computed conviction C={conviction:.2f}, should we trade?",
        "Output JSON only: {action: BUY|SELL|PASS, size_pct: 0-10, rationale: str}",
    ]

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Qwen3 call
# ────────────────────────────────────────────────────────────────

def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks before JSON parsing."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(text: str) -> str:
    """Extract the last {...} block from model output."""
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if not matches:
        # Try relaxed: find outermost braces
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        raise ValueError("No JSON object found in model output")
    return matches[-1].group(0)


def call_qwen(prompt: str) -> dict:
    """Call Qwen3 235B. Returns parsed dict or raises."""
    response = _get_qwen().chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=512,
        extra_body={"enable_thinking": False},   # disable Qwen3 chain-of-thought for speed
    )
    raw = response.choices[0].message.content or ""
    cleaned = _strip_think_blocks(raw)
    json_str = _extract_json(cleaned)
    return json.loads(json_str)


# ────────────────────────────────────────────────────────────────
# Fallback decision
# ────────────────────────────────────────────────────────────────

def _pass_decision(conviction: float, reason: str) -> MetaDecision:
    return MetaDecision(
        action="PASS",
        size_pct=0.0,
        rationale=reason,
        conviction=conviction,
        hard_gate=reason if "gate" in reason.lower() or "vix" in reason.lower() else None,
        model_version=_MODEL_VERSION,
        timestamp=datetime.now(UTC).isoformat(),
    )


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────

def run_meta_agent(
    ticker: str,
    signals: dict,
    vix: float | None = None,
) -> MetaDecision:
    """
    Full meta-agent pipeline:
    1. Load weights from HydraDB
    2. Compute conviction C
    3. Apply hard gates
    4. Query RAG
    5. Call Qwen3 235B
    6. Enforce VRP sizing
    7. Log + execute if warranted
    """
    # Step 1: weights
    weights = hydradb.get_agent_weights()

    # Step 2: conviction
    conviction = compute_conviction(signals, weights)
    print(f"[meta] ticker={ticker} C={conviction:.4f} weights={weights}")

    # Step 3: hard gates (before model call — cheap check first)
    gate = check_hard_gates(vix, ticker)
    if gate:
        print(f"[meta] hard gate fired: {gate}")
        reason = f"hard gate: {gate}"
        hydradb.log_pass(reason, {"ticker": ticker, "conviction": conviction})
        return _pass_decision(conviction, reason)

    # Step 4: RAG — embed current signals, retrieve similar past setups
    signals_json = json.dumps(
        {
            k: (v if not isinstance(v, dict) else {
                "direction": v.get("direction"),
                "regime": v.get("regime"),
            })
            for k, v in signals.items()
            if not k.startswith("_")
        },
        default=str,
    )
    rag_results = hydradb.query_rag(signals_json, top_k=5)

    # Step 5: Qwen3 call
    prompt = build_prompt(ticker, signals, weights, conviction, rag_results, vix)
    try:
        parsed = call_qwen(prompt)
        action = str(parsed.get("action", "PASS")).upper()
        size_pct = float(parsed.get("size_pct", 0.0))
        rationale = str(parsed.get("rationale", ""))

        # Enforce model output constraints
        if action not in ("BUY", "SELL", "PASS"):
            action = "PASS"
        size_pct = max(0.0, min(10.0, size_pct))
        if action == "PASS":
            size_pct = 0.0

    except json.JSONDecodeError as e:
        print(f"[meta] Qwen3 JSON parse failed: {e}", file=sys.stderr)
        return _pass_decision(conviction, "model_parse_error")
    except Exception as e:
        err = str(e)[:120]
        print(f"[meta] Qwen3 call failed: {err}", file=sys.stderr)
        # Fallback: use conviction sign if |C| > threshold
        if abs(conviction) > _CONVICTION_MIN:
            action = "BUY" if conviction > 0 else "SELL"
            size_pct = min(5.0, round(abs(conviction) * 10, 1))
            rationale = f"conviction-only fallback (model unavailable): C={conviction:.3f}"
        else:
            return _pass_decision(conviction, f"model_unavailable: {err}")

    # Step 6: VRP sizing guard
    decision_dict = {"action": action, "size_pct": size_pct, "rationale": rationale}
    decision_dict = enforce_vrp_sizing(decision_dict, signals.get("gamma"))

    # Step 7: execute if warranted
    if action != "PASS" and abs(conviction) > _CONVICTION_MIN:
        trade_record = {
            "ticker": ticker,
            "action": action,
            "size_pct": decision_dict["size_pct"],
            "conviction": conviction,
            "rationale": decision_dict["rationale"],
            "defined_risk": decision_dict.get("defined_risk", False),
        }
        emb = embedder.embed(signals_json)
        trade_id = hydradb.log_trade(trade_record, embedding=emb)
        print(f"[meta] logged trade id={trade_id}")

        # Alpaca execution — convert size_pct to qty (stub: 1 share per % for now)
        alpaca_order = {
            "ticker": ticker,
            "action": action,
            "qty": max(1, int(decision_dict["size_pct"])),
            "entry_price": 0.0,   # Phase 4: pull real quote
        }
        fill = submit_order(alpaca_order)
        print(f"[meta] fill={fill}")
        decision_dict["fill"] = fill
    else:
        reason = (
            "action=PASS from model"
            if action == "PASS"
            else f"low conviction |C|={abs(conviction):.3f} < {_CONVICTION_MIN}"
        )
        hydradb.log_pass(reason, {"ticker": ticker, "conviction": conviction})

    return MetaDecision(
        action=decision_dict["action"],
        size_pct=decision_dict["size_pct"],
        rationale=decision_dict["rationale"],
        conviction=conviction,
        hard_gate=None,
        model_version=_MODEL_VERSION,
        timestamp=datetime.now(UTC).isoformat(),
    )


# ────────────────────────────────────────────────────────────────
# Acceptance test
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

    # Dummy signals — mimics what the graph populates in state["signals"]
    dummy_signals = {
        "sentiment": {
            "agent_id": "sentiment",
            "ticker": ticker,
            "timestamp": datetime.now(UTC).isoformat(),
            "direction": 0.62,
            "conviction": 0.75,
            "regime": "momentum",
            "horizon_mins": 30,
            "signals": {"headline_sentiment": 0.65, "event_risk": False, "vix": 18.4},
            "model_version": "llama-3.3-70b-fast-v1",
        },
        "momentum": 0.45,       # Person C stub — float until Phase 4
        "gamma": {
            "agent_id": "gamma",
            "ticker": ticker,
            "timestamp": datetime.now(UTC).isoformat(),
            "direction": 0.30,
            "conviction": 0.55,
            "regime": "momentum",
            "horizon_mins": 60,
            "signals": {"gex_sign": "positive", "vrp_trade": False},
            "model_version": "gamma-stub-v1",
        },
        "zero_dte": {
            "agent_id": "zero_dte",
            "ticker": ticker,
            "timestamp": datetime.now(UTC).isoformat(),
            "direction": 0.0,
            "conviction": 0.05,
            "regime": "neutral",
            "horizon_mins": 30,
            "signals": {"stub": True},
            "model_version": "zero-dte-stub-v1",
        },
    }

    dummy_vix = 18.4

    print(f"[meta] acceptance test — ticker={ticker} vix={dummy_vix}", file=sys.stderr)
    result = run_meta_agent(ticker, dummy_signals, vix=dummy_vix)

    print(json.dumps(result, indent=2))
