# StratOS Optimization Review
**Date:** 2026-06-21  |  **Branch base:** `c`  |  **Parallel run:** c-gamma / c-gex / c-system

This document records what went right, what went wrong, and the forward plan for each computer's optimization sprint. Written after all three branches were tested against live data.

---

## Scorecard

| Computer | Branch | Model | Before | After | Status |
|----------|--------|-------|--------|-------|--------|
| 1 | c-gamma | Gamma (global) | 50.80% | 50.95% | FORMULA PATH (below 51% gate) |
| 1 | c-gamma | Gamma (healthcare) | — | 53.54% | PASS — XGBoost active |
| 1 | c-gamma | Gamma (other) | — | 52.65% | PASS — XGBoost active |
| 1 | c-gamma | Gamma (financials) | — | 49.51% | FORMULA PATH |
| 1 | c-gamma | Gamma (tech) | — | 47.03% | FORMULA PATH |
| 2 | c-gex | Smart OPEX Sharpe | 2.25 | 0.41 | REGRESSION — do not merge filter |
| 2 | c-gex | GEX engine | working | working | PASS — vol-OI + cache + DEX |
| 3 | c-system | Momentum XGBoost | 58.33% | 60.77% | PASS |
| 3 | c-system | Momentum LightGBM | — | 60.15% | PASS |
| 3 | c-system | Momentum Ensemble | — | 61.20% | PASS — best result |

---

## Computer 1 — Gamma Model (c-gamma)

### What went well

**#27 — 4-week label:** Shifting the return horizon from 1-week to 4-week smooths the label distribution (18,367 / 19,936 vs prior imbalance). Signal is slightly easier to learn. Minor accuracy lift (+0.09pp on global model) but the real benefit appears in sector-specific models where smoother labels help healthcare and "other" cross the gate.

**#30 — GEX × VIX interaction:** `gex_x_vix` became the #2 feature by importance (0.1432), displacing `gex_regime_flag` (dropped to 0.0000). Confirms the interaction captures real signal that the raw flags miss. Kept in all future training runs.

**#29 — Sector models:** Two sectors crossed the 51% gate cleanly:
- Healthcare: 53.54% — defensive names with predictable options flows
- Other (industrials, utilities, consumer, energy, ETFs): 52.65% — heterogeneous but works in aggregate

These two sectors now run XGBoost at inference instead of formula. The sector split itself was the right structural decision.

### What went wrong

**Tech sector: 47.03%** — below random. Price proxies (HV spread, skewness, PCR) have negative predictive value for tech because tech returns are dominated by earnings surprises, Fed rate sensitivity, and narrative sentiment — none of which show up in historical vol ratios. Tech needs real-time options flow data or NLP signals to break 51%.

**Financials: 49.51%** — slightly below random. Financial stocks move with macro data releases (CPI, NFP, FOMC) that are not in the feature set. The hardcoded gate dates in the backtest exist precisely because of this — but the model doesn't use them as a feature yet.

**Global model ceiling (~51%):** With price-proxy features only, 113-ticker cross-sectional training cannot consistently exceed 51%. The ceiling is structural, not a tuning problem. Real options data (actual IV surfaces, actual OI by expiry/strike) is required to crack 52% globally.

### Forward plan — Computer 1

| Priority | Task | Expected gain |
|----------|------|---------------|
| HIGH | #26 — Real options features: use live IV surface from yfinance options (term structure slope, 25-delta skew, realized/implied vol ratio) instead of HV proxy | +1–3pp global |
| HIGH | Add macro gate flag as feature (1 if FOMC/CPI/NFP within 3 days, else 0) — fixes financials sector | +1pp financials |
| MEDIUM | #31 — XGBoost/formula blend: activate for healthcare + other sectors where XGBoost already passes gate | Smoother transitions |
| MEDIUM | Retrain tech sector with momentum features (r5d, RSI, MACD) instead of options proxies | May lift tech above 49% |
| LOW | Add `mom8w` and `mom13w` to feature set — test if medium-term momentum adds to 4-week label | Minor |

---

## Computer 2 — GEX Engine + Backtest (c-gex)

### What went well

**#10 — Volume-weighted OI:** `call_oi_eff = call_oi * (1 + call_vol / max(call_oi, 1))` correctly upweights freshly-traded strikes. Net GEX moved from $556M → $1.28B for NVDA, reflecting the higher effective gamma near active strikes. This is the more accurate picture of dealer positioning.

**#11 — 15-minute chain cache:** In-process dict cache with 900s TTL means repeated calls within a session hit memory instead of yfinance. Essential once the scheduler starts calling GEX on each signal cycle. No side effects.

**#12 — Delta exposure (DEX):** `compute_delta_exposure()` added and working. Net DEX for NVDA = $9.99B. Completes the Greek suite: GEX (gamma), DEX (delta), vanna, charm. Meta-agent can now use directional delta pressure as a separate signal layer.

**#9 — GEX by strike chart:** Chart generation confirmed working. Visual correctly shows green/red bars at each strike with spot price line. Key demo asset.

### What went wrong

**#3 — Adaptive trend filter (52-week high proximity) REGRESSION:**
- Old filter (10-week MA): 31 OPEX weeks, Smart OPEX +0.268%, Sharpe 2.25, p≈0.01
- New filter (52-week high proximity, within 8% of high): **29 OPEX weeks, +0.079%, Sharpe 0.41, p=0.762**
- Root cause: The 52-week high filter is too restrictive and uncorrelated with the OPEX gamma mechanism. The 10-week MA trend filter works because it captures sustained trend — gamma dealers are long-biased in uptrends, which amplifies OPEX pinning. Proximity to a 52-week high is a price level condition, not a trend condition, and it excludes valid OPEX weeks during consolidation phases while including weak ones near all-time highs.
- This task should be marked FAILED. The original filter is better.

### Forward plan — Computer 2

| Priority | Task | Expected gain |
|----------|------|---------------|
| IMMEDIATE | Revert adaptive filter — restore 10-week MA trend check in `build_spy_signals()` | Restores Sharpe 2.25, p<0.05 |
| HIGH | Test 3-week MA vs 10-week MA as alternative — shorter lookback may be more responsive without losing signal | Possible Sharpe lift |
| HIGH | #4 — Finnhub live FOMC/CPI/NFP calendar — replace 414 hardcoded dates with live API call | Removes maintenance burden |
| MEDIUM | Per-stock GEX regime signal — currently using SPY GEX for all stocks; stock-level GEX would be more accurate | +0.5–1pp per-stock Sharpe |
| MEDIUM | Zero-gamma flip distance as feature in backtest — stocks within 2% of flip level behave differently | Sharpe lift |
| LOW | Extend chain cache to Redis for multi-process scheduler support (Phase 5 need) | Infrastructure |

---

## Computer 3 — Momentum + Infrastructure (c-system)

### What went well

**#18 — Optuna hyperparameter tuning:** Best params found (n_estimators=433, max_depth=6, lr=0.1076, subsample=0.957, colsample=0.655) pushed XGBoost from 58.33% → 60.77%. The tuning found a deeper, more regularized tree than the default, which makes sense for cross-sectional data where each stock needs different feature weightings.

**#21 — XGBoost + LightGBM ensemble:** Ensemble at 61.20% beats both solo models (XGBoost 60.77%, LightGBM 60.15%). Probability averaging is the right approach here — LightGBM's leaf-wise growth finds different decision boundaries than XGBoost's level-wise, so the ensemble captures complementary patterns. Best result in the session.

**#24 — GitHub Actions CI:** Workflow correctly tests all three model imports on push to c or c-* branches. Catches broken imports before merge.

**#22 — .env support:** `python-dotenv` load_dotenv() added to gex_engine.py and opex_backtest.py. `.env.example` created. Removes hardcoded credential risk.

### What went wrong

**lightgbm missing from requirements.txt:** The ensemble crashes on a clean environment with `ModuleNotFoundError: No module named 'lightgbm'`. Computer 3's card explicitly says not to edit requirements.txt (it's "final") — but lightgbm is a hard runtime dependency, not optional. This needs to be added before merge or documented in the PR as a manual install step.

**sklearn feature name warning:** LGBMClassifier is fitted with a DataFrame (has feature names) but predict is called with a numpy array (no names). Cosmetic but should be fixed: pass `feature_names` to LGBMClassifier constructor or use DataFrame at predict time.

### Forward plan — Computer 3

| Priority | Task | Expected gain |
|----------|------|---------------|
| IMMEDIATE | Add `lightgbm>=4.0.0` to requirements.txt — hard dependency, not optional | Prevents clean-env crash |
| IMMEDIATE | Fix sklearn warning: pass `feature_name=FEATURE_NAMES` to LGBMClassifier or predict with DataFrame | Clean logs |
| HIGH | Add `min_child_weight` back to XGB params — currently stripped in `train()` but used in `tune()`. May improve regularization | Minor accuracy |
| HIGH | Walk-forward validation: retrain monthly on rolling 18-month window instead of static 70/30 split — prevents regime drift | Prevents accuracy decay in live use |
| MEDIUM | Per-stock momentum models using the same 21-stock basket split — some stocks may have stronger intraday momentum patterns | Possible lift for high-vol names |
| MEDIUM | Add `implied_move` feature from ATM straddle price — captures market's expectation of day's range | +0.5–1pp estimated |
| LOW | Calibrate ensemble probabilities with isotonic regression — improves conviction scores for Meta-agent sizing | Sharpe lift via position sizing |

---

## Framework-Level Observations

### Parallel development worked — with one structural flaw
File ownership boundaries (OPTIMIZATION_CARDS.md) prevented code conflicts. Zero merge conflicts expected. The one failure (c-gex adaptive filter) was a task-level decision error, not a process error — the experiment was valid to run.

### The 51% gate is doing its job
The gate prevented a 50.95% global gamma model from overriding the working formula. The formula at 0.8182 conviction for NVDA is more reliable than a model that's barely above chance. The gate should stay and the threshold should stay at 51%.

### Price proxies have a hard ceiling around 51%
Three separate training runs (cross-sectional 113-ticker, 8-feature gamma classifier) all converge to 50.8–51.0% with price-proxy features. This is not a tuning problem. The ceiling is informational — to exceed it, the feature set needs:
- Real options data: IV surface, actual OI by expiry, realized/implied spread
- Macro timing: FOMC/CPI distance as a numeric feature (not just a gate)
- Microstructure: bid/ask spread, options volume spikes, dark pool prints

### Ensemble is the right architecture
Every model that tested ensemble (momentum: 61.20% vs 60.77% solo) showed improvement. This pattern will hold for gamma too once the XGBoost model crosses the gate — the formula + XGBoost blend (#31) should be the default architecture for both models.

### Backtest signal is fragile to filter changes
The OPEX effect (+0.268%/week, Sharpe 2.25) is real but narrow — only 31 qualifying weeks over 10 years. Any filter that reduces this to <30 weeks loses statistical significance (p>0.05). Future filter experiments must use a minimum-n constraint: **reject any filter that produces fewer than 35 OPEX weeks in the backtest period.**

---

## Merge Readiness

| Branch | Ready to merge? | Blocker |
|--------|----------------|---------|
| c-gamma | YES (with note) | Tech sector will stay on formula — acceptable |
| c-gex | NO | Must revert adaptive filter (#3) to restore Sharpe 2.25 |
| c-system | YES (with fix) | Add lightgbm to requirements.txt + fix sklearn warning |

Merge order after fixes:
```
git checkout c
git merge c-gamma   --no-ff -m "merge: gamma sector models + gex_x_vix interaction"
git merge c-system  --no-ff -m "merge: momentum 61.20% ensemble + CI + .env"
git merge c-gex     --no-ff -m "merge: GEX vol-OI + cache + DEX (adaptive filter reverted)"
git push origin c
```
