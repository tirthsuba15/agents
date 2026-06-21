# StratOS Optimization Cards
**Branch base:** `c`  |  **Repo:** github.com/tirthsuba15/agents  |  **Do not push to `main`**

Create your branch before starting:
```
git clone https://github.com/tirthsuba15/agents.git
git checkout c
git checkout -b <your-branch>   # see each card for branch name
```
Merge back to `c` when Phase 5 + 6 are done — not before.

---

## Computer 1 — Gamma Model
**Branch:** `c-gamma`  |  **Est. time:** 2–3 hrs

### You own these files
```
models/train_gamma.py
models/gamma_model.py
```

### Do NOT touch
```
models/train.py               ← Computer 3
models/momentum_model.py      ← Computer 3
models/gex_engine.py          ← Computer 2
backtest/opex_backtest.py     ← Computer 2
researcher/                   ← Phase 5
scheduler.py                  ← Phase 5
config.py                     ← Phase 5 reads this — do not edit
backtest/intraday_momentum.py ← Phase 6
```

### Tasks (do in order)

**#27 — 4-week label** *(biggest likely accuracy gain)*
In `train_gamma.py`, change `build_labels()` so the shift is `-4` instead of `-1`:
```python
stock_w = close_daily.resample("W-FRI").last().pct_change(4).shift(-4)
spy_w   = spy_close_daily.resample("W-FRI").last().pct_change(4).shift(-4)
```
Retrain. Target: classifier accuracy > 51% (4-week returns are smoother = easier to predict).

**#30 — GEX × VIX interaction term**
In `build_weekly_features()` add before the `df = pd.DataFrame(...)`:
```python
gex_x_vix_d = np.sign(close - sma20) * hv(close, 10)
gex_x_vix_w = gex_x_vix_d.resample("W-FRI").last()
```
Add `"gex_x_vix": gex_x_vix_w` to the DataFrame and append `"gex_x_vix"` to `FEATURE_NAMES`.
Retrain and compare accuracy to previous run.

**#29 — Sector-specific models**
Define 4 sector buckets at the top of `train_gamma.py`:
```python
SECTORS = {
    "tech":       ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","ADBE",
                   "ORCL","CRM","AMD","NFLX","CSCO","QCOM","AMAT","KLAC",
                   "LRCX","SNPS","CDNS","PANW","INTU","AVGO","XLK","SMH"],
    "healthcare": ["UNH","JNJ","LLY","ABT","TMO","SYK","ISRG","VRTX","REGN",
                   "GILD","AMGN","ZTS","BSX","BDX","DHR","CI","MRK","ABBV",
                   "PFE","XLV"],
    "financials": ["JPM","BAC","WFC","GS","MS","BLK","AXP","C","SPGI","MCO",
                   "CB","AON","CME","PGR","XLF","HYG"],
    "other":      [...remaining tickers...],
}
```
Train one `XGBClassifier` per sector, save as `gamma_model_tech.pkl`, `gamma_model_healthcare.pkl`, etc.
In `gamma_model.py`, add a `SECTOR_MAP` dict and load the matching pkl at inference.
Fall back to the full model if ticker not in any sector map.

**#31 — Blend XGBoost + formula**
In `predict_gamma()`, after getting `direction` from XGBoost, blend with the formula score:
```python
# formula_direction comes from compute_composite_scores()
blended = 0.5 * xgb_direction + 0.5 * formula_direction
direction = round(float(np.clip(blended, -1, 1)), 4)
method = "xgboost_blend_v1"
```
Only apply blend when XGBoost model is active (dir_acc > 0.51).

### How to test
```
python models/train_gamma.py        # must print accuracy + save pkl
python models/gamma_model.py NVDA   # must print valid SignalObject JSON
```

---

## Computer 2 — GEX Engine + Backtest
**Branch:** `c-gex`  |  **Est. time:** 2–3 hrs

### You own these files
```
models/gex_engine.py
backtest/opex_backtest.py
```

### Do NOT touch
```
models/train_gamma.py             ← Computer 1
models/gamma_model.py             ← Computer 1
models/train.py                   ← Computer 3
models/momentum_model.py          ← Computer 3
researcher/                       ← Phase 5
scheduler.py                      ← Phase 5
config.py                         ← Phase 5 reads this — do not edit
backtest/intraday_momentum.py     ← Phase 6 (new file, leave it alone)
```

### Tasks (do in order)

**#9 — GEX by strike chart** *(best demo visual)*
Add this function to `gex_engine.py` after `compute_delta_exposure()`:
```python
def plot_gex_by_strike(chain, spot, r, q, symbol, out_path=None):
    from pathlib import Path
    import matplotlib.pyplot as plt

    strikes, gex_vals = [], []
    for opt in chain:
        T = _tte(opt["expiry"])
        if T <= 0: continue
        K = opt["strike"]
        call_g = bs_gamma(spot, K, T, r, q, opt.get("call_iv", 0))
        put_g  = bs_gamma(spot, K, T, r, q, opt.get("put_iv",  0))
        net    = (call_g * opt.get("call_oi", 0) - put_g * opt.get("put_oi", 0)) * 100 * spot**2 * 0.01
        strikes.append(K)
        gex_vals.append(net / 1e6)   # in $M

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in gex_vals]
    ax.bar(strikes, gex_vals, color=colors, width=(max(strikes)-min(strikes))/len(strikes)*0.8)
    ax.axvline(spot, color="#2980b9", linewidth=1.5, linestyle="--", label=f"Spot ${spot:.0f}")
    ax.axhline(0,    color="#7f8c8d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Net GEX ($M)")
    ax.set_title(f"{symbol} — GEX by Strike")
    ax.legend()
    plt.tight_layout()
    path = out_path or Path(f"backtest/gex_by_strike_{symbol}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {path}")
```
Call it from `main()` in `gex_engine.py` after computing Greeks:
```python
plot_gex_by_strike(chain, spot, r, q, symbol)
```

**#10 — Volume-weight OI**
In `compute_gex()`, `compute_vanna_charm()`, and `compute_delta_exposure()`, replace raw OI with:
```python
# Inside each loop, before using call_oi / put_oi:
call_vol = opt.get("call_volume", 0)
put_vol  = opt.get("put_volume",  0)
call_oi_eff = call_oi * (1 + call_vol / max(call_oi, 1))
put_oi_eff  = put_oi  * (1 + put_vol  / max(put_oi,  1))
```
Use `call_oi_eff` / `put_oi_eff` in GEX calculations instead of raw OI.
Do this consistently in all three compute functions.

**#11 — 15-minute chain cache**
Add at the top of `gex_engine.py` after the imports:
```python
_chain_cache: dict = {}   # {symbol: (fetched_at, chain)}
_CACHE_TTL = 900          # 15 minutes in seconds
```
Wrap `fetch_options_chain()`:
```python
def fetch_options_chain(symbol: str) -> OptionsChain:
    import time
    cached = _chain_cache.get(symbol)
    if cached:
        fetched_at, chain = cached
        if time.time() - fetched_at < _CACHE_TTL:
            print(f"  Options source: cache ({len(chain)} strike rows)")
            return chain
    # ... existing fetch logic ...
    _chain_cache[symbol] = (time.time(), chain)
    return chain
```

**#3 — Adaptive trend filter for backtest**
In `opex_backtest.py`, replace the 10-week MA trend check with a 52-week high proximity filter.
In `build_spy_signals()`, add:
```python
rolling_high = close_s.rolling(52, min_periods=26).max().shift(1)
near_high    = (prior_close / rolling_high) > 0.92   # within 8% of 52wk high
```
Change `uptrend` to `near_high` in the return DataFrame.
Rerun the backtest and compare Smart OPEX Sharpe vs current baseline (Sharpe 2.25).

### How to test
```
python models/gex_engine.py NVDA          # must print GEX + DEX + save chart
python backtest/opex_backtest.py          # must print stats with t-stat + p-value
```

---

## Computer 3 — Momentum + Infrastructure
**Branch:** `c-system`  |  **Est. time:** 3–4 hrs

### You own these files
```
models/train.py
models/momentum_model.py
.github/workflows/        (create this directory)
.env.example              (create this file)
```

### Do NOT touch
```
models/train_gamma.py             ← Computer 1
models/gamma_model.py             ← Computer 1
models/gex_engine.py              ← Computer 2
backtest/opex_backtest.py         ← Computer 2
researcher/                       ← Phase 5
scheduler.py                      ← Phase 5
config.py                         ← Phase 5 reads this — do not edit
backtest/intraday_momentum.py     ← Phase 6 (new file, leave it alone)
requirements.txt                  ← already final, do not edit
```

### Tasks (do in order)

**#18 — Optuna hyperparameter tuning** *(could push momentum above 60%)*
Install: `pip install optuna lightgbm`

Add a `tune()` function to `train.py` (do not replace `train()` — add alongside it):
```python
def tune(n_trials: int = 50) -> None:
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    print(f"Building dataset for tuning...")
    df = build_dataset()
    X  = df[FEATURE_NAMES].values
    y  = df["label"].values

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        }
        accs = []
        for train_idx, val_idx in tscv.split(X):
            m = XGBClassifier(**params)
            m.fit(X[train_idx], y[train_idx])
            accs.append((m.predict(X[val_idx]) == y[val_idx]).mean())
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best CV accuracy : {study.best_value*100:.2f}%")
    print(f"  Best params      : {study.best_params}")
```
Run with `python models/train.py tune` — add a `if __name__ == "__main__": tune() if "tune" in sys.argv else train()` guard.
Then take the best params and hardcode them into `train()` for the final model. Retrain and report test accuracy.

**#21 — Ensemble XGBoost + LightGBM**
After Optuna finds best XGB params, train a LightGBM alongside it:
```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42, n_jobs=-1, verbose=-1,
)
lgbm.fit(X_train, y_train)

# Ensemble: average probabilities
xgb_proba  = model.predict_proba(X_test)[:, 1]
lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
ensemble_proba = (xgb_proba + lgbm_proba) / 2
ensemble_preds = (ensemble_proba >= 0.5).astype(int)
acc_ensemble = float((ensemble_preds == y_test).mean())
print(f"  Ensemble accuracy: {acc_ensemble*100:.2f}%")
```
Save both models in the pkl: `{"model": model, "lgbm": lgbm, "feature_names": FEATURE_NAMES}`.
Update `momentum_model.py` `_load_model()` to load both and average probas at inference.

**#24 — GitHub Actions CI**
Create `.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: ["c", "c-*"]
  pull_request:
    branches: ["c"]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python -c "from models.momentum_model import predict_momentum, FEATURE_NAMES; print('momentum import OK')"
      - run: python -c "from models.gex_engine import compute_gex, bs_gamma; print('gex import OK')"
      - run: python -c "from models.gamma_model import compute_composite_scores; print('gamma import OK')"
```

**#22 — .env support**
Install: `pip install python-dotenv`

Create `.env.example` at repo root:
```
FINNHUB_API_KEY=
NEBIUS_API_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Add to the top of `gex_engine.py` and `backtest/opex_backtest.py` (right after the imports, before any `os.environ.get` calls):
```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```

Add `python-dotenv>=1.0.0` to `requirements.txt` — actually, do NOT edit requirements.txt (it's final). Just note it in your PR description.

### How to test
```
python models/train.py              # baseline retrain, must pass 52%
python models/train.py tune         # Optuna run (takes ~10 min)
python models/momentum_model.py     # must print valid SignalObject JSON
```

---

## Merge checklist (do this when Phase 5 + 6 are done)

```
git checkout c
git merge c-gamma   --no-ff -m "merge: gamma model optimizations"
git merge c-gex     --no-ff -m "merge: GEX engine + backtest optimizations"
git merge c-system  --no-ff -m "merge: momentum ensemble + infrastructure"
git push origin c
```

If any conflict appears it will be in `OPTIMIZATIONS.md` (all three may update it).
Resolve by keeping all entries. No code conflicts expected.
