#!/usr/bin/env python3
"""
scheduler.py

APScheduler nightly async agent for StratOS Phase 5.
Runs researcher pipeline at 02:00 UTC daily (markets closed).

Jobs:
  1. arxiv_search  → fetch latest papers
  2. strategy_parser → Nemotron via Nebius Token Factory → HydraDB
  3. mutator       → 5 parameter mutations → HydraDB

Not part of the real-time trading cycle — purely async research.

Usage:
    python scheduler.py
    python scheduler.py --once   (run immediately then exit, for testing)
"""

import argparse
import logging
import sys
import time

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def job_research_pipeline() -> None:
    log.info("=== Nightly Research Pipeline START ===")

    # Step 1 — arXiv
    log.info("Step 1/3: arXiv paper fetch")
    try:
        from researcher.arxiv_search import fetch_papers
        papers = fetch_papers(max_results=10)
        log.info(f"  Fetched {len(papers)} papers")
    except Exception as exc:
        log.error(f"  arxiv_search failed: {exc}")
        papers = []

    # Step 2 — strategy parser (Nemotron)
    log.info("Step 2/3: Strategy parser (Nemotron via Token Factory)")
    try:
        from researcher.strategy_parser import parse_and_store
        strategies = parse_and_store(papers=papers)
        log.info(f"  {len(strategies)} new strategies identified and stored")
    except Exception as exc:
        log.error(f"  strategy_parser failed: {exc}")

    # Step 3 — parameter mutator
    log.info("Step 3/3: Parameter mutator (5 mutations)")
    try:
        from researcher.mutator import run_mutations
        top = run_mutations()
        if top:
            log.info(f"  Top mutation: {top[0]['label']}  Sharpe={top[0]['sharpe']:.4f}")
    except Exception as exc:
        log.error(f"  mutator failed: {exc}")

    log.info("=== Nightly Research Pipeline DONE ===")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="StratOS research scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run pipeline once immediately and exit")
    args = parser.parse_args()

    if args.once:
        log.info("--once flag: running pipeline immediately")
        job_research_pipeline()
        return

    if not HAS_APSCHEDULER:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        log.info("Falling back to --once mode")
        job_research_pipeline()
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        job_research_pipeline,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_research",
        name="Nightly arXiv + Nemotron + Mutator",
        misfire_grace_time=3600,
        replace_existing=True,
    )

    log.info("Scheduler started — research pipeline runs at 02:00 UTC daily")
    log.info("Press Ctrl+C to stop")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


if __name__ == "__main__":
    main()
