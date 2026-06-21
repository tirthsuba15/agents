
StratOS
Self-Tuning Trading Agent Suite
Full Architecture & Data Requirements Document

Version 1.0  ·  June 2026  ·  Hackathon Build
Built with: LangGraph · HydraDB · Nebius AI · Alpaca · Finnhub

1. Executive Summary
StratOS is a self-tuning, multi-agent algorithmic trading system targeting US large-cap equities and their options. Five specialized AI agents produce, fuse, and continuously improve trading signals using academically-backed strategies. The system paper-trades via Alpaca, logs every outcome to HydraDB with vector embeddings, uses those outcomes to reweight agent signals and fine-tune custom models, and deploys a dedicated Strategy Researcher agent to discover and backtest new strategies autonomously.

Core loop in one sentence
Sub-agents generate signals → Meta-agent fuses them and places a paper trade → outcome is embedded and stored → HydraDB answers 'which agent was most accurate?' → weights rebalance → Strategy Researcher proposes mutations → best mutations replace underperforming parameters → repeat.

Build target (4-hour hackathon sprint)
Person
Primary ownership
Deliverable
Person A
LangGraph graph + Sentiment & Meta agents
agents/ folder, working end-to-end cycle
Person B
HydraDB schema + reweight loop
memory/ folder, PnL attribution function
Person C
OPEX + intraday momentum models + backtest runner
researcher/ + backtest/ folders, demo chart



2. Strategy Catalogue & Signal Assignments
Six academically-backed strategy families drawn from peer-reviewed papers (JFE, RFS, JFQA, JBF) and SSRN working papers. Each is assigned to an agent, classified by execution type, and annotated with its key failure modes.

2.1  Stage 1 — Price/Volume Only (Build First)
These two strategies need only OHLCV bars and a calendar. No options data required. Build and validate backtest benchmarks before touching any options signals.

OPEX-Week Drift  (Momentum agent — custom model)
Academic basis: 
Stivers & Sun (2013), Journal of Banking & Finance 37(11):4226–4240. For S&P 100 large-caps with actively traded options, average weekly return during OPEX week was 0.45% vs 0.12% in other weeks (1996–2008). Annualized Sharpe ~1.29 for third-Friday week. Mechanism: reduction in call open interest → dealers unwind short-stock delta hedges → upward drift.
Signal construction:
Define OPEX week as Monday open through expiration Friday close (third Friday each month)
Universe: S&P 100 or liquid large-cap ETF basket (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA)
Entry: Monday open of OPEX week. Exit: Friday close. Hold cash all other weeks (~18% time in market)
Weight: equally-weighted or market-cap weighted basket
Benchmark to reproduce:
~0.45% mean return in OPEX weeks vs ~0.12% in non-OPEX weeks
Best months: April. Weak/negative: July and January — control for these
Decay warning: 
Retail replications show ~2% CAGR net of costs, 51–60% win rate. Treat as a long-only tilt, not a standalone book. Avoid OPEX weeks that overlap macro events (FOMC, CPI).

Market Intraday Momentum  (Momentum agent — custom model)
Academic basis: 
Gao, Han, Li & Zhou (2018), Journal of Financial Economics 129(2):394–414. On SPY 1993–2013: first 30-min return positively predicts last 30-min return, slope 6.94 (1% significance), R²=1.6%. Rising to ~2.6% combined with 12th half-hour. Stronger on high-volume, high-vol, macro-news, and recession days. Baltussen et al. (2021) confirm across 60+ global futures and link to gamma-hedging demand.
Signal construction:
Compute r₁ = return from prior close through first 30 minutes of session
If r₁ > threshold (e.g. +0.15%), go long SPY/QQQ at ~3:30pm, exit at close
If r₁ < -0.15%, go short or stand aside (long-only mode: stand aside)
Scale position size by |r₁| and by VIX level (stronger on high-vol days)
Benchmark to reproduce:
First-to-last half-hour slope ≈ 6.94, R² ≈ 1.6%
Effect strongest when VIX > 20 and volume > 20-day average
Decay warning: 
Concentrated on high-vol/news days; weak in calm regimes. Sensitive to execution quality at open and close. Slippage kills it in practice — model must account for bid-ask at entry and exit.

2.2  Stage 2 — Options-Implied Cross-Sectional Signals
These require an implied-volatility surface. Free-tier Finnhub provides IV data for large-caps. OptionMetrics IvyDB is the academic gold standard but costs ~$2K/month. Use Finnhub options chains as a proxy for the demo.

IV Spread  (Gamma agent — custom model)
Academic basis: 
Cremers & Weinbaum (2010), JFQA 45(2):335–367. IV spread = (OI-weighted) call IV minus put IV across matched strike/maturity pairs. Stocks with expensive calls outperform stocks with expensive puts by ~50 bp per week. Predictability strongest when option liquidity is high and stock liquidity is low.
Signal construction:
For each name: find matched call/put pairs (same strike, same expiry, nearest 30-day expiry)
IV spread_i = mean(IV_call_i,k - IV_put_i,k) across matched pairs, OI-weighted
Cross-sectionally rank the universe → long top decile, underweight bottom decile
Rebalance weekly
Decay warning: 
Cremers-Weinbaum explicitly document declining predictability over their sample. Post-2015, treat as a long-only tilt (overweight high-IV-spread names), not a long-short book.

Volatility Smirk  (Gamma agent — custom model)
Academic basis: 
Xing, Zhang & Zhao (2010), JFQA 45(3):641–662. Smirk = IV(OTM put) − IV(ATM call). Steepest-smirk stocks underperform flattest-smirk stocks by 10.9%/year risk-adjusted. A steep smirk signals informed pessimism/crash fear; effect persists up to ~6 months.
Signal construction:
For each name: OTM put = nearest strike with delta ≈ -0.20 to -0.30, ATM call = delta ≈ +0.50
Smirk_i = IV(OTM put_i) − IV(ATM call_i)
Rank cross-sectionally: avoid/underweight steep-smirk names, overweight flat-smirk names
Holding period: 1–4 weeks

Put-Call Ratio  (Gamma agent — custom model)
Academic basis: 
Pan & Poteshman (2006), RFS 19(3):871–908. Stocks with low put-call ratios outperform high-P/C stocks by >40 bp next day and >1% next week. Predictive power concentrated in non-public (signed) flow component, but unsigned P/C from public chains still works at weekly horizon.
Signal construction:
P/C_i = put volume / call volume for name i (from Finnhub options chain)
Signal_i = -rank(P/C_i) cross-sectionally (low P/C = bullish, high P/C = bearish)
Combine with IV spread and smirk: composite_score = 0.4 × PCR + 0.3 × IV_spread + 0.3 × smirk
Rebalance daily for 1-day horizon, weekly for swing

2.3  Stage 3 — GEX Regime Engine (Filter, Not Alpha)
GEX's independent vol-prediction power largely disappears after controlling for VIX/ATM IV. Use it as a regime switch to condition other strategies — not as a standalone directional signal.

Dealer Gamma Exposure (GEX)  (Gamma agent — deterministic computation)
Academic basis: 
Barbon & Buraschi (2021) SSRN 3725454; Baltussen et al. (2021) JFE 142(1); SqueezeMetrics GEX white paper (Dec 2017). Positive net GEX → dealers buy dips/sell rallies → mean-reversion regime. Negative net GEX → dealers amplify moves → momentum/breakout regime.
Signal construction (dollar-gamma-per-1%-move convention):
Per-strike: GEX_K = Γ_K × OI_K × 100 × S² × 0.01  (calls positive, puts negative)
Net GEX = Σ_K GEX_K across all strikes and expiries
Black-Scholes gamma: Γ = e^(−qT) × N'(d₁) / (S × σ × √T)
Zero-gamma flip: build spot grid ±20% (60 points), recompute GEX at each, find sign-change, linearly interpolate
Regime rules:
Spot > zero-gamma flip AND GEX > 0: positive gamma regime → favor mean-reversion and premium-selling
Spot < zero-gamma flip OR GEX < 0: negative gamma regime → favor momentum, avoid short-vol
Gate all GEX-conditioned sizing on VIX: inside VIX > 30 days, GEX adds little independent information

Vanna / Charm OPEX Pressure  (Gamma agent — deterministic computation)
Academic basis: 
Stivers & Sun (2013); Ni, Pearson & Poteshman (2005) JFE 78(1):49–87; Pearson, Poteshman & White (~2020). As OTM options decay via charm into OPEX, dealers unwind underlying hedges producing predictable drift. Vol-crush as expiry approaches drives vanna-related buying support.
Signal construction:
Vanna_K = −e^(−qT) × N'(d₁) × (d₂/σ) per strike
Charm_K (call) = qe^(−qT)N(d₁) − e^(−qT)N'(d₁) × [2(r−q)T − d₂σ√T] / (2Tσ√T)
Aggregate across OI-weighted strikes; compute net dealer vanna and charm exposure
Days 3–5 before OPEX: high positive aggregate vanna → expect IV crush and underlying support

0DTE Flow  (Sentiment agent — LLM call, experimental)
Literature status: UNSETTLED
Dim, Eraker & Vilkov (2024) find positive MM gamma dampens vol (stabilizing). Adams et al. (2024) confirm −60–90 annualized bp on 0DTE days. But Brogaard et al. (2024) find +9.1% vol increase per SD of 0DTE trading. Cboe's own data shows net MM hedging is de minimis. Treat as experimental — use LLM reasoning to interpret, not a fixed model.
0DTE now 59% of SPX volume (Cboe full-year 2025 report, 2.3M contracts/day ADV)
In positive-MM-gamma regimes: fade intraday extremes. In negative: expect momentum. Close before expiry.

2.4  Stage 4 — Variance Risk Premium (Defined Risk Only)
VRP Harvesting  (Meta-agent decision — LLM call)
Academic basis: 
Carr & Wu (2009), RFS 22(3):1311–1341; Bollerslev, Tauchen & Zhou (2009), RFS 22(11):4463–4492. Implied variance systematically exceeds realized variance. Selling that premium (short variance / short straddles) harvests the gap. VRP also predicts aggregate market returns at quarterly horizon.
Signal construction:
VRP = implied variance (model-free from option strip, or VIX²) − expected realized variance (5-min HF)
Sell when VRP > 2-month rolling 70th percentile AND VIX term structure is in contango
Stand aside when: VRP compresses, VIX backwardates, or VIX > 30
CRITICAL RISK CONTROLS — Volmageddon lesson (Feb 5, 2018):
NEVER short naked gamma or use leveraged short-vol vehicles (XIV lesson)
Use ONLY defined-risk structures: iron condors, vertical spreads, cash-secured puts
Size for a ≥4σ overnight gap at all times
Maximum allocation: 10% of paper-trade portfolio to VRP book
Volmageddon reference: Augustin, Cheng & Van den Bergen (2021), FAJ 77(3):35–51. VIX jumped 17.31→37.32 in one session; XIV fell >90%. The premium compensates real crash risk — it is not free money.

3. Agent Architecture
Five agents implemented in LangGraph. Three sub-agents produce typed signal objects. A Meta-agent fuses them and executes. A Strategy Researcher runs async, proposes mutations, and updates the RAG store.

3.1  Agent Roster
Agent
Type
Model
Strategies
Horizon
Sentiment
LLM call
Llama 3.3 70B Fast
Finnhub NLP, 0DTE interpretation
Intraday
Momentum
Custom model
XGBoost/MLP
OPEX drift, intraday momentum, reversal
30 min – 1 week
Gamma
Custom model + math
XGBoost + BS math
GEX regime, vanna/charm, IV spread, smirk, P/C ratio
Intraday – 1 month
Meta-agent
LLM call
Qwen3 235B
Signal fusion, VRP sizing, trade rationale
Per-cycle
Researcher
LLM call
Nemotron 3 Super
arXiv search, param mutation, backtest
Nightly async


Custom model = numerics in → scalar signal out. No language reasoning needed for IV spread decile rank or OPEX calendar flag. Train on HydraDB outcomes. LLM call = language reasoning genuinely required (NLP, conflicting literature, multi-signal narrative).

3.2  Signal Object Schema
Every sub-agent returns a typed SignalObject that the Meta-agent ingests:
SignalObject {
  agent_id:       str          # 'sentiment' | 'momentum' | 'gamma'
  ticker:         str          # e.g. 'NVDA'
  timestamp:      datetime
  direction:      float        # -1.0 to +1.0 (normalized)
  conviction:     float        # 0.0 to 1.0 (model confidence)
  regime:         str          # 'mean_revert' | 'momentum' | 'neutral'
  horizon_mins:   int          # expected signal duration in minutes
  signals:        dict         # raw sub-signals (opex_flag, gex_regime, etc.)
  model_version:  str          # for tracking weight changes
}

3.3  Meta-Agent Fusion Logic
The Meta-agent runs on Qwen3 235B via Nebius Serverless AI. Before each decision it:
Queries HydraDB RAG: retrieve top-5 most similar past setups (by embedding similarity)
Reads current agent weights W = [w_sentiment, w_momentum, w_gamma] from HydraDB
Computes conviction score: C = w₁ × direction_sentiment + w₂ × direction_momentum + w₃ × direction_gamma
If |C| > threshold (default 0.35) AND regime agreement across agents: submit trade to Alpaca
Writes trade rationale, signal values, and weights snapshot to HydraDB

3.4  Self-Reweighting Loop
After each trade resolves (default: 30-min holding for intraday, 1-week for swing):
Layer 1 — Win/loss per agent: which agent's direction matched outcome?
Layer 2 — Sharpe/PnL attribution: per-agent contribution to trade PnL (Brinson-Hood-Beebower attribution)
Layer 3 — LLM meta-reasoning: Meta-agent reads last 20 trades and writes a natural-language post-mortem
Weight update: exponential moving average of per-agent accuracy, clipped to [0.05, 0.80]
Custom model retraining: if Momentum or Gamma agent accuracy falls below 52% over 50 trades, trigger fine-tune on latest labeled data

4. Complete Data Requirements
Every data source, its access method, free-tier limits, and the strategies it enables. Organized by Stage.

4.1  Stage 1 Data — Price & Volume (Free)
Dataset
Source
API / access
Free tier
Used for
Daily OHLCV
Finnhub
REST: /stock/candle
60 calls/min, 1yr history
OPEX drift, reversal, backtests
Intraday 1-min bars
Alpaca Markets
REST: /v2/stocks/bars
Unlimited paper-mode
Intraday momentum signal
30-min bars
Alpaca Markets
REST: /v2/stocks/bars
Unlimited paper-mode
r₁ half-hour return
OPEX calendar
Computed
Python: third_friday(year, month)
No API needed
OPEX-week flag
Market hours / holidays
Alpaca Markets
REST: /v2/calendar
Free
Session validation
VIX index
Finnhub / CBOE
Symbol: VIX
Free
GEX gate, VRP sizing


4.2  Stage 2 Data — Options Surface
Dataset
Source
API / access
Free tier
Used for
Options chain (IV, OI, volume)
Finnhub
REST: /stock/option-chain
60 calls/min
IV spread, smirk, P/C ratio
ATM implied vol
Finnhub
REST: /stock/option-chain
Included above
VRP computation, GEX gate
Per-strike OI snapshot
Finnhub
REST: /stock/option-chain
Daily (official OI updates once/day)
GEX computation, vanna/charm
Historical IV surface
Polygon.io (backup)
REST: /v3/snapshot/options
Starter: 250 calls/min
Backtesting IV spread/smirk signals
Put / call signed volume
Finnhub
REST: /stock/option-chain
Unsigned only on free tier
Put-call ratio signal

OptionMetrics IvyDB is the academic gold standard for IV surfaces but costs ~$2K/month. For hackathon demo, use Finnhub option chains. For production, consider Polygon.io options at $200/month or CBOE LiveVol.

4.3  Stage 3 Data — GEX Engine
Dataset
Source
API / access
Free tier
Used for
Full option chain (all expiries)
Finnhub
/stock/option-chain
60 calls/min
GEX aggregation across all strikes
Per-strike Black-Scholes Greeks
Computed locally
scipy.stats.norm
No API
Γ, Δ, vanna, charm per strike
Risk-free rate
FRED API
series: DGS3MO
Free, unlimited
r in Black-Scholes
Dividend yield per name
Finnhub
/stock/dividend2
Free
q in Black-Scholes gamma
Spot price (real-time)
Alpaca
/v2/stocks/quotes/latest
Free paper-mode
S in GEX formula


4.4  Sentiment & News Data
Dataset
Source
API / access
Free tier
Used for
Company news feed
Finnhub
/company-news
60 calls/min
Sentiment agent NLP input
Market news
Finnhub
/news?category=general
Free
Macro context for Meta-agent
Earnings calendar
Finnhub
/calendar/earnings
Free
Avoid holding through earnings
SEC filings
EDGAR / EdgarTools
Python: edgar library
Free
Researcher agent strategy context
Economic calendar
Finnhub
/calendar/economic
Free
FOMC/CPI gate for OPEX week


4.5  Strategy Researcher Data Sources
Dataset
Source
Access
Cost
Used for
Academic papers
arXiv / SSRN
arXiv API (free), SSRN web fetch
Free
New strategy discovery
Historical OHLCV backtest
Alpaca / yfinance
Alpaca data API, yf.download()
Free
Mutation backtesting
Benchmark returns
Alpaca / Finnhub
SPY OHLCV
Free
Sharpe vs benchmark
Alternative signals (web search)
Tavily / SerpAPI
REST search API
Free tier: 1000 calls/month
Researcher web discovery


5. Technology Stack

5.1  Nebius Resource Allocation
Resource
Product
Budget
Models
Used for
Token Factory
api.tokenfactory.nebius.com
$50
Llama 3.3 70B Fast
Sentiment agent, real-time inference
Token Factory (batch)
api.tokenfactory.nebius.com
(same)
Llama 3.3 70B Base
Nightly batch: strategy research summaries
Serverless AI
Nebius AI Cloud (Aether 3.5)
$100
Qwen3 235B
Meta-agent fusion + conviction reasoning
Serverless AI (researcher)
Nebius AI Cloud (Aether 3.5)
(same)
Nemotron 3 Super 550B
Strategy Researcher (nightly, not per-trade)

Token Factory uses OpenAI-compatible SDK: set base_url='https://api.tokenfactory.nebius.com' and api_key=NEBIUS_API_KEY. Sub-second latency on Fast tier. Serverless AI: deploy vLLM endpoint via nebius CLI in ~5 minutes, OpenAI-compatible /v1/chat/completions.

5.2  Core Infrastructure
Component
Technology
Role
Notes
Agent orchestration
LangGraph
Defines agent graph, state transitions, conditional edges
StateGraph with typed AgentState
Memory / RAG store
HydraDB
Trade log + outcome embeddings + strategy vectors
REST memory/knowledge-graph API (api.hydradb.com) — semantic recall; NOT pgvector/SQL
Paper trading execution
Alpaca Markets
Submit orders, receive fills, track positions
Paper trading mode, free
Market data
Finnhub
News, OHLCV, options chains, dividends
60 calls/min free tier
Custom model runtime
scikit-learn / XGBoost
In-process inference for Momentum + Gamma signal models
Sub-millisecond, no API call
Embeddings
sentence-transformers
Encode trade setup for RAG similarity retrieval
all-MiniLM-L6-v2 (local, free)
Backtesting
vectorbt / pandas
Strategy mutation backtest in researcher agent
Vectorized, fast
Task scheduling
APScheduler
Nightly researcher run, weight update cycle
cron-style
Notifications
ntfy.sh
Push alerts on trades, weight updates, researcher findings
Free, no account needed


5.3  HydraDB Schema

> AS-BUILT NOTE (verified June 2026): HydraDB is a managed REST memory / knowledge-graph
> service (https://api.hydradb.com, Bearer auth, tenant `agents`, sub-tenant `day_trading`) —
> NOT a pgvector/SQL database. There are no SQL tables, no `CREATE TABLE`, and no server-side
> `VECTOR` columns. The three schemas below are the **logical record shapes**. Each record is
> stored as one HydraDB memory via `POST /memories/add_memory` with `infer:false` (verbatim
> JSON), keyed by a `source_id` of the form `<type>:<ISO8601>:<uuid>` (lexically sortable by
> time). Reads: `fetch/content` (exact by id), `list/data` (enumerate ids, paginated),
> `recall/recall_preferences` (semantic RAG retrieval). Ingestion is ASYNC (~5–15s). The
> `embedding` field is a 384-dim `list[float]` (all-MiniLM-L6-v2) stored INSIDE the record JSON;
> HydraDB also embeds the record text for recall. Client lives at root `hydradb.py`.

Three core record types (the SQL syntax below denotes the logical JSON record shape, not a real
relational schema). Embeddings generated locally with sentence-transformers.

Table: trades
trades (
  id              UUID PRIMARY KEY,
  ticker          TEXT,
  timestamp_entry TIMESTAMPTZ,
  timestamp_exit  TIMESTAMPTZ,
  direction       FLOAT,      -- +1 long / -1 short
  conviction      FLOAT,
  signals_json    JSONB,      -- raw signal values from all 3 agents
  weights_json    JSONB,      -- w1/w2/w3 at time of trade
  regime          TEXT,
  pnl_bps         FLOAT,      -- realized PnL in basis points
  outcome         BOOL,       -- TRUE if direction was correct
  embedding       VECTOR(384) -- sentence-transformer of setup
)

Table: agent_weights
agent_weights (
  id              SERIAL PRIMARY KEY,
  timestamp       TIMESTAMPTZ,
  w_sentiment     FLOAT,
  w_momentum      FLOAT,
  w_gamma         FLOAT,
  trigger         TEXT,       -- 'reweight' | 'manual' | 'init'
  accuracy_json   JSONB       -- per-agent accuracy over last N trades
)

Table: strategy_candidates
strategy_candidates (
  id              UUID PRIMARY KEY,
  timestamp       TIMESTAMPTZ,
  source          TEXT,       -- 'arxiv' | 'mutation' | 'manual'
  description     TEXT,
  params_json     JSONB,
  backtest_sharpe FLOAT,
  backtest_cagr   FLOAT,
  status          TEXT,       -- 'candidate' | 'deployed' | 'rejected'
  embedding       VECTOR(384)
)

6. Repository Structure
Designed to be handed to Claude Code. Each folder is a single person's ownership during the 4-hour sprint.

stratos/
├── agents/                        # Person A
│   ├── sentiment.py               # Finnhub news → NLP score → direction
│   ├── meta_agent.py              # Signal fusion, conviction, trade decision
│   └── zero_dte.py                # 0DTE flow stub (experimental)
│
├── models/                        # Person C (trains on backtest data)
│   ├── momentum_model.py          # OPEX flag + intraday momentum → XGBoost
│   ├── gamma_model.py             # IV spread + smirk + PCR → XGBoost
│   ├── gex_engine.py              # Black-Scholes GEX + zero-gamma flip
│   ├── vanna_charm.py             # Vanna + charm aggregation
│   └── train.py                   # Fine-tune on HydraDB labeled outcomes
│
├── hydradb.py                      # Person B (ROOT) — CRUD over HydraDB REST memory API; graph imports `hydradb`  [BUILT]
├── embedder.py                     # Person B (ROOT) — sentence-transformers all-MiniLM-L6-v2 (384-dim); graph imports `embedder`  [BUILT]
├── memory/                         # Person B (planned)
│   └── reweighter.py              # PnL attribution → w1/w2/w3 EMA update  [not yet built]
│
├── execution/
│   └── alpaca.py                  # Submit paper trades, get fills
│
├── researcher/                    # Person C (async nightly)
│   ├── arxiv_search.py            # Query arXiv API, fetch abstracts
│   ├── strategy_parser.py         # Nemotron 3 Super → structured strategy spec
│   └── mutator.py                 # Vary params ±20%, run backtest, rank
│
├── backtest/
│   ├── runner.py                  # Vectorized OHLCV backtest (vectorbt)
│   ├── opex_backtest.py           # Reproduce Stivers-Sun 0.45% vs 0.12%
│   └── intraday_momentum.py       # Reproduce Gao et al. slope ≈ 6.94
│
├── graph/
│   └── stratos_graph.py           # LangGraph StateGraph — all nodes wired
│
├── data/
│   ├── finnhub_client.py          # OHLCV, options chain, news, dividends
│   ├── alpaca_client.py           # Bars, quotes, calendar
│   └── fred_client.py             # Risk-free rate (DGS3MO)
│
├── config.py                      # API keys, thresholds, universe list
├── scheduler.py                   # APScheduler: trading cycle + nightly research
└── main.py                        # Entry point: run one full trading cycle

6.1  Trading Universe
Big-cap equities with actively traded options, maximum Finnhub free-tier coverage:

AAPL
MSFT
NVDA
AMZN
GOOGL
META
TSLA
JPM
SPY
Apple
Microsoft
Nvidia
Amazon
Alphabet
Meta
Tesla
JPMorgan
S&P 500 ETF

SPY is the benchmark and the primary vehicle for the intraday momentum strategy (Gao et al. was tested on SPY specifically). Single-name strategies run on the 8 equities above. QQQ can be added as a second ETF vehicle.

7. Risk Controls & Failure Mode Registry

7.1  Hard Rules (Never Override)
These are inviolable. They exist because the documented failure modes (Volmageddon, GEX edge collapse, reversal cost-decay) all violated one of these in some form.

NEVER short naked options or use leveraged short-vol instruments (XIV/SVIX)
ALL VRP harvesting via defined-risk structures only (iron condors, vertical spreads)
Maximum VRP allocation: 10% of paper portfolio
Gate every trade: if VIX > 35, no new positions. If VIX > 30, no VRP book expansion
No position held through earnings announcement — check Finnhub /calendar/earnings before entry
No OPEX-week position if macro event (FOMC, CPI) falls within that week — check Finnhub /calendar/economic
Zero-gamma flip rule: if spot crosses the flip zone during a position, halve size immediately

7.2  Strategy Failure Mode Registry
Strategy
Primary failure mode
Detection signal
Mitigation
OPEX drift
Macro event in OPEX week
Finnhub /calendar/economic
Skip entry if FOMC, CPI, NFP falls Mon–Fri that week
Intraday momentum
Low vol / low volume day
VIX < 15, volume < 80% of 20-day avg
Reduce size 50% or skip; effect concentrated on high-vol days
Short-term reversal
Transaction cost decay
Net PnL after slippage < 0 over 30 trades
Demote to long-only tilt; never run as long-short book
IV spread
Post-publication decay
Post-2015 long-short spread < transaction costs
Run as long-only tilt; track decay in HydraDB monthly
GEX regime
VIX explains most variance
Walk-forward accuracy < 54%
Use only as conditional filter; never standalone directional
VRP harvesting
Volatility spike / gap risk
VIX > 25, term structure backwardation
Stand down; defined-risk only; size for 4σ gap always
0DTE flow
Unsettled academic literature
Model accuracy < 52% over 30 trades
Experimental flag; weight capped at 5% of Meta-agent decision


7.3  Decay Monitoring
The Strategy Researcher agent runs a nightly decay check alongside new-strategy discovery:
Query HydraDB: per-strategy accuracy and net PnL over last 30 / 90 days
Compare against in-sample benchmark (pre-2020 backtest Sharpe)
If out-of-sample Sharpe < 0.5 for any strategy over 90 days: flag for demotion
If out-of-sample Sharpe < 0.0 over 30 days: auto-suspend and alert via ntfy.sh
Researcher writes decay summary to HydraDB strategy_candidates table with status='rejected'

8. Hackathon Demo Plan (4-Hour Sprint)

8.1  What Gets Built in 4 Hours
Hour
Owner
Build target
Acceptance criterion
0–1
A: LangGraph + SentimentB: HydraDB schemaC: OPEX backtest
Graph scaffold runs; DB tables created; OPEX backtest shows 0.45% vs 0.12%
Python scripts execute without error; backtest chart visible
1–2
A: Meta-agent on NebiusB: ReweighterC: GEX engine
Meta-agent calls Qwen3 235B, returns trade decision; reweight fn runs on dummy data; GEX computes zero-gamma flip
Meta-agent JSON output readable; weights update after simulated outcome; GEX regime flag outputs correct sign
2–3
A: Alpaca executionB: HydraDB RAG queryC: Momentum model
Paper trade submits to Alpaca; RAG returns top-5 similar setups; XGBoost trained on simulated data
Alpaca order confirmed; RAG query returns JSON; model accuracy > 50% on hold-out
3–4
All: integration + demo
Full end-to-end cycle: Finnhub → agents → Meta-agent → Alpaca → HydraDB → reweight
One live demo cycle executes in < 30 seconds; weights visibly change after 3 simulated trades


8.2  Demo Script (5 Minutes)
Minute 1: Show the architecture diagram. Explain the 5-agent system and self-reweight loop.
Minute 2: Run backtest_demo.py — chart showing OPEX-week 0.45% vs 0.12% for AAPL 2015–2025.
Minute 3: Trigger one live cycle: Finnhub pulls NVDA data, all 3 agents return signals, Meta-agent fuses and places a paper trade on Alpaca.
Minute 4: Simulate 5 past outcomes, run reweighter — show weights changing in real time on a terminal readout.
Minute 5: Show HydraDB RAG query: 'what happened last time GEX was negative AND intraday momentum was positive on NVDA?' — Meta-agent reads the retrieved context and explains its conviction.

8.3  Backtest Benchmarks to Reproduce
Strategy
Benchmark stat
Source paper
Your demo target
OPEX-week drift
0.45% vs 0.12%
Stivers-Sun (2013) JBF
>0.35% mean OPEX week on AAPL/SPY 2015–2024
Intraday momentum
Slope ≈ 6.94, R² ≈ 1.6%
Gao et al. (2018) JFE
Positive slope on SPY 30-min r₁ vs r₁₃, 2020–2024
Smirk signal
10.9%/yr risk-adj.
Xing et al. (2010) JFQA
Top-decile flat-smirk AAPL outperforms steep-smirk in backtest
GEX regime
ρ ≈ −0.36 vs realized vol
SqueezeMetrics white paper
Negative GEX days show higher next-day realized vol on SPX





StratOS · Full Architecture Document · v1.0
Aarnav · Tirth · Team · June 2026
