# StratOS Optimization Backlog

Run this after all phases are complete. Each item has an estimated impact and effort rating.
Baseline metrics recorded at end of each phase — optimizations are measured against these.

---

## Phase 1 — OPEX Backtest (`backtest/opex_backtest.py`)

**Baseline (2015–2024, 20-stock basket)**
- Smart OPEX mean: +0.268% / week
- Naive OPEX mean: +0.097% / week
- Smart OPEX Sharpe: 2.25
- Demo target missed: non-OPEX mean 0.444% (need <0.20%)

**Optimizations**

| # | Change | Expected impact | Effort |
|---|--------|----------------|--------|
| 1 | Remove energy stocks (XOM, CVX) from basket — they follow commodity gamma, not equity options OPEX | Smart OPEX mean +0.05–0.10% | Low |
| 2 | Add sector ETFs (XLK, XLV, XLF) to basket instead of individual stocks — diversifies idiosyncratic risk | More stable Sharpe | Low |
| 3 | Replace fixed 10-week MA with adaptive trend filter (e.g. 52-week high proximity) | Better regime detection | Medium |
| 4 | Upgrade Finnhub plan to get live FOMC/CPI/NFP calendar — removes hardcoded date dependency | More accurate gating | Low |
| 5 | Extend backtest to 2010–2024 using longer yfinance history — larger sample, more OPEX weeks | More statistical significance | Low |
| 6 | Add t-stat and p-value to printed output — needed for academic credibility in demo | Cosmetic | Low |

---

## Phase 2 — GEX Engine (`models/gex_engine.py`)

**Baseline (NVDA live, 2026-06-21)**
- Net GEX: $556M
- Zero-gamma flip: $200.38
- Regime: positive_gamma
- Options source: yfinance (Finnhub options = paid)

**Optimizations**

| # | Change | Expected impact | Effort |
|---|--------|----------------|--------|
| 7 | Upgrade Finnhub key to get real options chain — IVs from Finnhub are more accurate than yfinance | More accurate GEX | Low |
| 8 | Filter to front-month + next-month expiries only (≤45 DTE) — captures the most gamma-dense contracts | More responsive to near-term positioning | Low |
| 9 | Add GEX by strike chart (matplotlib bar chart, x=strike, y=net GEX) — key demo visual | Visual output | Medium |
| 10 | Weight OI by volume (use `call_volume` / `put_volume` as recency proxy) — OI is stale, volume is fresh | More accurate dealer positioning | Medium |
| 11 | Cache options chain with 15-min TTL — avoid refetching on every predict cycle | Performance | Low |
| 12 | Add `compute_delta_exposure()` function — total market delta in dollars, completes the Greek suite | Completeness | Medium |

---

## Phase 3 — Momentum Model (`models/momentum_model.py`, `models/train.py`)

**Baseline**
- Test accuracy: 53.92% (target >52%) ✅
- Training samples: 722 (2 years SPY hourly)
- Features: 6 (r1, opex_flag, vix_level, volume_ratio, day_of_week, days_to_opex)
- Labels: synthetic (r1/r13 same-sign)

**Optimizations**

| # | Change | Expected accuracy gain | Effort |
|---|--------|----------------------|--------|
| 13 | Add `gex_regime` as feature (positive_gamma=1, negative_gamma=0) — Phase 2 output directly improves Phase 3 | +1–2% | Low |
| 14 | Add `overnight_return` feature — pre-market price vs prior close, one of the strongest intraday predictors | +1–2% | Low |
| 15 | Extend training to 10 years (2015–2024) SPY hourly — ~2,500 samples vs 722 | +1–2% | Low |
| 16 | Train on full 20-stock basket (cross-sectional) — multiplies training data 20× | +1–3% | Medium |
| 17 | Replace synthetic labels with real trade P&L labels once live — biggest single improvement | +2–4% | High (needs live data) |
| 18 | Hyperparameter tuning with Optuna + TimeSeriesSplit CV — honest walk-forward accuracy estimate | +0.5–1.5% | Medium |
| 19 | Add rolling features: 5-day momentum, 10-day realized vol, lagged r1 (prior day) | +0.5–1% | Low |
| 20 | Add interaction terms: `r1 * opex_flag`, `vix_level * volume_ratio` | +0.5% | Low |
| 21 | Ensemble XGBoost + LightGBM (average probas) — reduces variance | +0.5–1% | Medium |

**Realistic ceiling after optimizations: 56–58% accuracy**

---

## System-Wide

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 22 | Add `.env` file support (python-dotenv) — stop passing API keys as env vars manually | Dev experience | Low |
| 23 | Add `requirements.txt` at repo root covering all phases | Reproducibility | Low |
| 24 | Add GitHub Actions CI: run `pytest` on push to any branch | Code quality | Medium |
| 25 | Replace hardcoded FOMC/CPI/NFP dates with FRED calendar API — single source of truth | Accuracy | Medium |

---

*Last updated: Phase 3 complete. Append new entries as phases are added.*
