#!/usr/bin/env python3
"""
researcher/strategy_parser.py

Takes paper abstracts from arxiv_search, sends to Nemotron via Nebius Token Factory,
parses JSON strategy candidates, filters for Finnhub free-tier compatibility,
and stores results in HydraDB strategy_candidates table.

Usage:
    python researcher/strategy_parser.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from researcher.arxiv_search import fetch_papers
from researcher.nebius_client import chat_complete
from researcher.hydra_client import insert_many

SYSTEM_PROMPT = (
    "You are a quantitative finance expert specialising in systematic equity strategies. "
    "You output only valid JSON arrays. No prose, no markdown fences."
)

FINNHUB_FREE_DATA = [
    "daily OHLCV", "weekly OHLCV", "intraday hourly OHLCV",
    "basic financials", "earnings dates", "news sentiment",
    "basic options chain (strike, OI, volume, expiry)", "VIX index",
]

PROMPT_TEMPLATE = """
I am running a systematic equity trading system with these existing strategies:
1. OPEX momentum: buy basket of 21 large-cap stocks in OPEX weeks when SPY is in uptrend
2. Intraday momentum: trade based on first-hour return direction (r1) for same-day close
3. Weekly gamma signal: options-derived GEX regime detection for weekly directional bias

Given the following finance paper abstracts, identify trading strategies that are NOT already covered above.
For each new strategy, output a JSON array where each element has exactly these keys:
  name              - short strategy name
  signal_description - one sentence describing the core signal
  entry_rule        - specific entry condition
  exit_rule         - specific exit condition
  estimated_sharpe  - your estimate of out-of-sample Sharpe ratio (float)
  data_requirements - list of data needed

Filter OUT any strategy requiring: paid options data feeds, Level 2 order book, tick data, Bloomberg, proprietary datasets.
Only include strategies that can be implemented with: {finnhub_free}

Output ONLY a JSON array. If no new strategies found, output [].

Paper abstracts:
{abstracts}
""".strip()


def _extract_json(text: str) -> list[dict]:
    """Extract the first JSON array from LLM output, tolerating prose wrapping."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _is_finnhub_compatible(strategy: dict) -> bool:
    """Return True if data_requirements only reference Finnhub free-tier sources."""
    reqs = " ".join(strategy.get("data_requirements", [])).lower()
    blocklist = ["level 2", "tick data", "bloomberg", "refinitiv", "paid", "proprietary",
                 "intraday minute", "options greeks feed", "institutional flow"]
    return not any(b in reqs for b in blocklist)


def parse_and_store(papers: list[dict] = None) -> list[dict]:
    """
    Fetch papers (if not provided), call Nemotron, filter, store in HydraDB.
    Returns list of stored strategy records.
    """
    if papers is None:
        print("  Fetching papers from arXiv...")
        papers = fetch_papers(max_results=10)

    if not papers:
        print("  [parser] No papers to parse.")
        return []

    abstracts_text = "\n\n".join(
        f"[{p['arxiv_id']}] {p['title']}\n{p['abstract'][:400]}"
        for p in papers
    )

    prompt = PROMPT_TEMPLATE.format(
        finnhub_free=", ".join(FINNHUB_FREE_DATA),
        abstracts=abstracts_text,
    )

    print(f"  Calling Nemotron ({len(papers)} papers)...")
    try:
        raw = chat_complete(prompt, system=SYSTEM_PROMPT, max_tokens=2048, temperature=0.2)
    except Exception as exc:
        print(f"  [parser] Nemotron call failed: {exc}", file=sys.stderr)
        return []

    strategies = _extract_json(raw)
    print(f"  Nemotron returned {len(strategies)} strategy candidates")

    compatible = [s for s in strategies if _is_finnhub_compatible(s)]
    print(f"  {len(compatible)} passed Finnhub free-tier filter")

    if not compatible:
        return []

    for s in compatible:
        s["source"] = "arxiv_nemotron"
        s["status"] = "candidate"

    stored = insert_many(compatible)
    print(f"  {stored}/{len(compatible)} stored in HydraDB strategy_candidates")
    return compatible


def main() -> None:
    print("Strategy Parser — Nemotron via Nebius Token Factory")
    strategies = parse_and_store()
    if strategies:
        print(f"\n  Strategies found:")
        for s in strategies:
            print(f"    - {s.get('name')}: Sharpe~{s.get('estimated_sharpe', '?')}")
    else:
        print("  No new strategies identified.")


if __name__ == "__main__":
    main()
