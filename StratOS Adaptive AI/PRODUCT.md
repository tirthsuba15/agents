# Vault — Product Context

## Register
**product** (primary). The dashboard at `/dashboard` is the core surface — design serves
the data. The marketing landing at `/` is a secondary **brand** surface; apply brand
sensibility there, product discipline everywhere else.

## What it is
A self-evolving, multi-agent algorithmic trading system. Three signal agents (sentiment,
momentum, gamma) feed a meta-agent that fuses them into a conviction score and sizes
positions; a reweighter rewrites the agents' decision weights from closed-trade outcomes.
The UI is wired to the live backend (FastAPI sidecar → HydraDB memory, Finnhub market
data, Alpaca paper broker).

## Users & purpose
Quant-literate operators and reviewers watching an autonomous trading loop: current
decision weights, recent agent decisions, open paper positions, account equity, and the
memory/feedback that drives self-evolution. Context: focused monitoring, often on a wide
screen, scanning for state and anomalies — not casual browsing.

## Brand personality
Precise, instrument-grade, quietly confident. Reads like a trading terminal crossed with
an ML observability console: monospace numerics, dense but legible, dark, calm under load.
Three words: **precise, live, engineered.**

## Anti-references
- Generic SaaS-cream / rounded-pastel marketing landing.
- Crypto-hype neon gradients, glow-on-everything, "🚀 to the moon" energy.
- Fake-precision dashboards where every number is decorative. Real data must read as real;
  illustrative data must be visibly marked (we use "sample data" badges).

## Accessibility
- Dark theme; body/numeric text must hold ≥4.5:1 (the muted-gray-on-dark-tint trap is the
  main risk here). Bull/bear color is never the *only* signal — pair with sign/arrow/label.
- Respect `prefers-reduced-motion` for the ticker, scan lines, and pulse/blink animations.

## Strategic design principles
1. **Truth in data.** Live vs. demo/sample is always visible. Never let fabricated numbers
   read as live.
2. **Numbers are the hero.** Tabular, monospace, aligned. Hierarchy by scale + weight, not
   chrome.
3. **Calm density.** Information-dense without noise; motion is feedback, not decoration.
