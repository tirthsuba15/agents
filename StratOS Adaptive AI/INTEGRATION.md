# Vault UI — System Integration

The dashboard is wired to the **real Vault backend** (`~/Documents/agents`) via a
read-only FastAPI sidecar. No more mock data on the live paths.

## Architecture

```
React UI (TanStack Start, :8080)
  └─ React Query hooks  src/hooks/use-stratos.ts   (client-only, SSR-gated)
       └─ typed client   src/lib/stratos-api.ts     (VITE_STRATOS_API)
            └─ FastAPI    ~/Documents/agents/api.py  (:8000, read-only, TTL-cached)
                 ├─ hydradb.py / reweighter.py  → weights, trades, accuracy, memory
                 ├─ Finnhub REST                → live quotes + company news
                 └─ Alpaca REST                 → paper account, positions, equity history
```

**Read-only by design.** No endpoint runs the LangGraph or submits orders (that would
place real paper trades + burn Nebius tokens). The UI reads *persisted* state only.

## Run it

1. **Backend API** (from `~/Documents/agents`, needs `.env` with keys):
   ```bash
   ./venv/bin/uvicorn api:app --port 8000
   ```
   Sanity check: `curl localhost:8000/api/health`

2. **Frontend** (from this folder):
   ```bash
   bun install
   bun run dev          # → http://localhost:8080
   ```
   `VITE_STRATOS_API` (in `.env`) points the client at the API; defaults to
   `http://localhost:8000`.

If the API is down or a key is missing, every panel falls back to representative
demo data and shows a "demo" badge instead of "live".

## What's wired to live data

| UI surface | Source |
|---|---|
| Nav connection badge | `/api/health` |
| Ticker tape, hero chart price | Finnhub `/api/quotes` |
| Hero stats (Sharpe / win rate / equity) | `/api/stats` (HydraDB trades + Alpaca) |
| Architecture graph node tags | `/api/weights`, `/api/stats`, `/api/portfolio` |
| Sentiment agent card | Finnhub `/api/news` |
| Agent weights (cards, evolution bars) | `/api/weights` (HydraDB) |
| Evolution log + counts | `/api/memory`, `/api/weights` |
| Execution: account, positions, trade log | `/api/portfolio`, `/api/trades` |
| Memory panel stats | `/api/memory` |
| Dashboard header / Overview / Positions | `/api/portfolio`, `/api/stats` |
| Dashboard equity curve | Alpaca `/api/equity_curve` |
| Dashboard settings data sources | `/api/health` |

## Notes

- HydraDB list+fetch is multi-second; the API TTL-caches everything (15s quotes → 60s memory).
- The `agents` paper account starts at $10,000. Its equity history is flat until trades close.
- Charts without a real intraday source (LiveChart candles, GEX-by-strike, backtest curve)
  remain illustrative; their headline numbers are wired to live values.
