/**
 * StratOS API client — typed fetch wrappers over the read-only FastAPI sidecar
 * (see ~/Documents/agents/api.py). Every response carries a `live` flag; the UI
 * uses it to show "live" vs "demo" and to fall back to representative data when
 * the backend is down or a key is missing.
 */

const BASE: string =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_STRATOS_API ??
  "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`StratOS API ${path} → ${res.status}`);
  return (await res.json()) as T;
}

// ── Response shapes ──────────────────────────────────────────────
export interface Health {
  status: string;
  live: boolean;
  services: { hydradb: boolean; finnhub: boolean; alpaca: boolean; nebius: boolean };
  time: string;
}

export interface Weights {
  live: boolean;
  w_sentiment: number;
  w_momentum: number;
  w_gamma: number;
  trigger: string | null;
  timestamp: string | null;
  accuracy_json: Record<string, unknown>;
}

export interface Accuracy {
  live: boolean;
  w_sentiment?: number | null;
  w_momentum?: number | null;
  w_gamma?: number | null;
  pnl_attribution?: Record<string, number>;
  retrain?: Record<string, boolean>;
  n_trades?: number;
  overall?: number;
}

export interface Trade {
  id: string | null;
  ticker: string;
  action: string;
  direction: number | null;
  conviction: number | null;
  size_pct: number | null;
  regime: string | null;
  pnl_bps: number | null;
  outcome: boolean | null;
  rationale: string | null;
  timestamp_entry: string | null;
  timestamp_exit: string | null;
  order_id: string | null;
  fill_price: number | null;
}

export interface Trades {
  live: boolean;
  count: number;
  trades: Trade[];
}

export interface Stats {
  live: boolean;
  n_trades: number;
  n_completed?: number;
  win_rate: number | null;
  avg_conviction: number | null;
  sharpe: number | null;
  total_pnl_bps?: number | null;
  equity?: number | null;
  buying_power?: number | null;
}

export interface Quote {
  symbol: string;
  price: number;
  change: number;
  change_pct: number;
  high: number;
  low: number;
  open: number;
  prev_close: number;
}

export interface Quotes {
  live: boolean;
  quotes: Quote[];
}

export interface NewsItem {
  source: string;
  headline: string;
  url: string | null;
  datetime: number | null;
  summary: string;
}

export interface News {
  live: boolean;
  ticker: string;
  news: NewsItem[];
}

export interface Position {
  ticker: string;
  qty: number;
  side: string;
  market_value: number;
  avg_entry_price: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  current_price: number;
}

export interface Portfolio {
  live: boolean;
  account: {
    equity: number;
    last_equity: number;
    buying_power: number;
    cash: number;
    status: string;
  } | null;
  positions: Position[];
}

export interface EquityCurve {
  live: boolean;
  base_value?: number | null;
  points: { t: number; v: number }[];
}

export interface BacktestTrade {
  week_start: string;
  ret_pct: number;
  outcome: boolean;
  equity: number;
}

export interface Backtest {
  live: boolean;
  strategy?: string;
  reference?: string;
  period?: { start: string; end: string };
  generated_at?: string;
  summary?: {
    cagr: number;
    total_return: number;
    sharpe: number;
    win_rate: number;
    max_drawdown: number;
    n_trades: number;
  };
  cohorts?: Record<string, { n: number; mean_pct: number; sharpe: number; win_rate: number }>;
  per_stock?: { symbol: string; smart_mean_pct: number }[];
  equity_curve?: { week_start: string; smart: number }[];
  benchmark_curve?: { week_start: string; bah: number }[];
  trades?: BacktestTrade[];
  reason?: string;
}

export interface MemorySummary {
  live: boolean;
  total: number;
  by_kind: Record<string, number>;
  recent: { kind: string; timestamp: string | null; id: string }[];
}

// ── Endpoints ────────────────────────────────────────────────────
export const api = {
  health: () => get<Health>("/api/health"),
  weights: () => get<Weights>("/api/weights"),
  accuracy: () => get<Accuracy>("/api/accuracy"),
  trades: (n = 25) => get<Trades>(`/api/trades?n=${n}`),
  stats: () => get<Stats>("/api/stats"),
  quotes: (symbols?: string) =>
    get<Quotes>(`/api/quotes${symbols ? `?symbols=${encodeURIComponent(symbols)}` : ""}`),
  news: (ticker = "NVDA") => get<News>(`/api/news?ticker=${encodeURIComponent(ticker)}`),
  portfolio: () => get<Portfolio>("/api/portfolio"),
  equityCurve: (period = "1M", timeframe = "1D") =>
    get<EquityCurve>(`/api/equity_curve?period=${period}&timeframe=${timeframe}`),
  backtest: () => get<Backtest>("/api/backtest"),
  memory: () => get<MemorySummary>("/api/memory"),
};
