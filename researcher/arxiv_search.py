#!/usr/bin/env python3
"""
researcher/arxiv_search.py

Queries the arXiv Atom API for equity momentum and options strategy papers.
Returns a list of {title, abstract, arxiv_id} dicts.

Usage:
    python researcher/arxiv_search.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
QUERIES = [
    "ti:equity AND ti:momentum",
    "ti:options AND ti:strategy",
    "ti:intraday AND ti:momentum",
    "ti:gamma AND ti:exposure AND ti:equity",
    "ti:cross-sectional AND ti:return",
]
MAX_PER_QUERY = 4
NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=NS)
        arxiv_id = arxiv_id_url.split("/abs/")[-1].strip()
        title    = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        if title and abstract:
            results.append({"title": title, "abstract": abstract, "arxiv_id": arxiv_id})
    return results


def fetch_papers(max_results: int = 10) -> list[dict]:
    """
    Fetch up to max_results unique papers across all query terms.
    Deduplicates by arxiv_id.
    """
    seen   = set()
    papers = []

    for query in QUERIES:
        if len(papers) >= max_results:
            break
        try:
            resp = requests.get(
                ARXIV_API,
                params={"search_query": query, "max_results": MAX_PER_QUERY, "sortBy": "relevance"},
                timeout=15,
            )
            resp.raise_for_status()
            for p in _parse_feed(resp.text):
                if p["arxiv_id"] not in seen:
                    seen.add(p["arxiv_id"])
                    papers.append(p)
                    if len(papers) >= max_results:
                        break
        except Exception as exc:
            print(f"  [arxiv] query '{query}' failed: {exc}", file=sys.stderr)

    return papers


def main() -> None:
    print("Fetching arXiv papers...")
    papers = fetch_papers(max_results=10)
    print(f"  Fetched {len(papers)} papers\n")
    for i, p in enumerate(papers, 1):
        print(f"  [{i}] {p['arxiv_id']}")
        print(f"       {p['title']}")
        print(f"       {p['abstract'][:120]}...")
        print()
    if len(papers) < 5:
        print("  [FAIL] Fewer than 5 papers returned — check network / arXiv availability")
        sys.exit(1)
    print(f"  [PASS] {len(papers)} papers fetched")


if __name__ == "__main__":
    main()
