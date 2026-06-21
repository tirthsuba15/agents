# StratOS — Agent-Driven Optimization Plan

Goal: push risk-adjusted return (CAGR / Sharpe, out-of-sample) materially above the
current `c`-branch gamma baseline, using a fleet of AI agents to optimize each
subsystem in parallel, with every change gated by a walk-forward backtest so we
never ship an in-sample mirage.

## Operating model (how agents are used)

Three-agent loop per work item (never the same agent writes and validates):
- **Claude (orchestrator)** — decomposes, dispatches, gates merges on the eval harness.
- **opencode (executor)** — implements/refactors, writes the Optuna studies and tests.
- **agy (reviewer)** — independent backtest re-run, leakage audit, stat-significance check.

Hard rules:
1. Every optimization is accepted ONLY if it beats baseline on **walk-forward OOS**, not the train window.
2. Writer ≠ validator. The agent that tunes a model is not the agent that confirms the lift.
3. All candidate params/results are persisted to HydraDB (`status: candidate → validated → promoted`) so the system learns across runs.
4. One metric of record per subsystem, fixed up front (below). No moving goalposts.

## Baseline & instrumentation (Phase 0 — do first)

- Lock a **frozen eval harness**: fixed ticker basket, fixed walk-forward splits (e.g. 6× expanding train / 1-month test), fixed cost model (slippage + commission). Owner: opencode; audited by agy.
- Record baseline per subsystem into HydraDB: gamma model OOS Sharpe/CAGR, momentum OOS, full-system paper-equity curve, hit-rate, max drawdown.
- Deliverable: `backtest/eval_harness.py` + a baseline row in HydraDB. **Nothing below merges without beating these numbers.**

## Phase 1 — Model hyperparameter optimization (parallel, highest ROI)

One agent per model, run concurrently:
- **gamma** (`models/train_gamma.py`): Optuna over XGBoost depth/eta/subsample/min_child_weight + the sector-split scheme; preserve c's high-CAGR feature set. Metric: OOS Sharpe per sector, then blended.
- **momentum** (`models/momentum_model.py`): walk-forward Optuna; tune lookbacks (mom4w/8w/13w) + RSI thresholds.
- **gex** (`models/gex_engine.py`): calibrate GEX-regime thresholds against realized vol.

Each agent: define search space → run study → agy re-runs the winner on a held-out split → promote only if OOS lift > noise (block on Phase 0 cost model). Deliverable: tuned `*.pkl` + the study artifact committed; HydraDB `validated` rows.

## Phase 2 — Feature expansion (researcher-driven)

Reactivate the `researcher/` pipeline as a feature factory:
- `arxiv_search → nebius_client (Nemotron) → strategy_parser → mutator` proposes new signals/features.
- Each proposed feature is added behind a flag, backtested in isolation, kept only on OOS lift + low correlation to existing features (de-redundancy check by agy).
- Reconcile the two gamma lineages: `a`'s #27/#29/#30/#31 work and `c-gamma`'s `rv_iv_ratio` real-IV feature were dropped in the c-CAGR build — re-test each as an additive feature on top of c's gamma rather than as a competing whole-file rewrite. **This is the "use all" recovery: bring the best ideas from a/c-gamma into c's winner.**

## Phase 3 — Meta-agent & self-reweighting upgrade

- Current `reweighter.py` adjusts agent weights from realized outcomes (linear). Upgrade to a **contextual bandit / regime-conditioned weighting**: weights become a function of market regime (VIX level, GEX regime, OPEX week) rather than global scalars.
- Train on the HydraDB trade history (RAG: `query_similar_setups`), validate on forward paper trades.
- Acceptance: higher blended Sharpe than static weights on OOS, and lower drawdown in regime transitions.

## Phase 4 — Continuous research → candidate → gate loop

- `scheduler.py` runs the researcher loop on a cadence: mine → mutate → fast 6-month backtest (`mutator.py`) → store top candidates.
- Add an **auto-promotion gate**: a candidate that clears the Phase 0 walk-forward harness (not just the fast backtest) is auto-promoted to `validated` and surfaced for human review before live weighting.

## Phase 5 — Risk & execution

- Position sizing as a function of model conviction + realized vol (vol-targeting), not fixed fraction.
- Execution agent models Alpaca slippage/fills; backtests must use the same cost model as Phase 0.
- Add portfolio-level guards: max gross, per-name cap, drawdown circuit-breaker.

## Phase 6 — Continuous evaluation & guardrails

- Nightly: full walk-forward re-run on latest data; alert on metric regression vs the promoted baseline.
- Leakage CI: a test that fails if any feature uses look-ahead data (the single most common silent killer of high backtest CAGR).
- All results streamed to HydraDB + the StratOS dashboard (`api.py` / frontend live view).

## Sequencing & parallelism

```
Phase 0  (blocking — harness first)
  └─► Phase 1 gamma | Phase 1 momentum | Phase 1 gex   (parallel)
        └─► Phase 2 features ──► Phase 3 meta/reweight
                                   └─► Phase 4 loop ──► Phase 5 risk ──► Phase 6 CI
```

## Definition of done (per item)

Beats the frozen baseline on **out-of-sample walk-forward** under the realistic
cost model, confirmed by a different agent than the one who built it, with the
result + params persisted to HydraDB. In-sample improvements are not "done."
