import { useState } from "react";
import {
  LayoutDashboard, Layers, Radar, FlaskConical, Activity,
  Wallet, Settings, TrendingUp, TrendingDown, Circle, ArrowUpRight,
  ArrowDownRight, Play, Database, Zap,
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Cell,
} from "recharts";
import { TickerTape } from "./TickerTape";
import { useStats, usePortfolio, useWeights, useHealth, useEquityCurve, useBacktest } from "@/hooks/use-stratos";
import { fmt } from "./live";

const usdCompact = (v: number) =>
  v >= 1000 ? `$${(v / 1000).toFixed(0)}k` : `$${v.toFixed(0)}`;

// Deterministic mock data (no Math.random — SSR-safe)
const equityCurve = Array.from({ length: 60 }, (_, i) => ({
  d: i,
  v: 100000 + i * 420 + Math.sin(i / 4) * 2600 + (i > 40 ? (i - 40) * 380 : 0),
}));

const gexByStrike = [
  { k: 560, gex: -3.1 }, { k: 570, gex: -2.2 }, { k: 580, gex: -0.9 },
  { k: 590, gex: 0.6 }, { k: 595, gex: 2.4 }, { k: 600, gex: 5.8 },
  { k: 605, gex: 4.1 }, { k: 610, gex: 2.0 }, { k: 620, gex: 0.7 },
  { k: 630, gex: -0.4 },
];

const strategies = [
  { name: "Dealer Gamma Regime (GEX)", horizon: "Intraday / multi-day", type: "Regime filter", on: true, papers: 3, note: "Mean-revert above zero-gamma flip; trend below. Gate on VIX." },
  { name: "OPEX-Week Drift", horizon: "Swing (1 wk)", type: "Seasonal", on: true, papers: 3, note: "Long large-cap basket Mon to 3rd-Fri close. ~18% time in market." },
  { name: "Variance Risk Premium", horizon: "Swing (~1 mo)", type: "Premium harvest", on: false, papers: 3, note: "Defined-risk short vol only. Size for 4-sigma gap." },
  { name: "Intraday Momentum", horizon: "Intraday", type: "Momentum", on: true, papers: 2, note: "First 30-min return predicts last 30-min. High-vol days only." },
  { name: "Short-Term Reversal", horizon: "Swing (1 wk-1 mo)", type: "Mean reversion", on: false, papers: 2, note: "Long losers / short winners. Watch transaction costs." },
  { name: "Options-Implied Signals", horizon: "Swing (1 d-1 wk)", type: "Cross-sectional", on: true, papers: 3, note: "IV spread, smirk, put-call ratio decile long-short." },
];

const scannerRows = [
  { sym: "NVDA", ivSpread: 0.42, smirk: 1.1, pcr: 0.61, gexFlip: 142.5, regime: "pos", signal: "Long bias" },
  { sym: "AAPL", ivSpread: -0.18, smirk: 3.4, pcr: 1.32, gexFlip: 211.0, regime: "pos", signal: "Avoid" },
  { sym: "AMZN", ivSpread: 0.31, smirk: 1.6, pcr: 0.74, gexFlip: 198.2, regime: "neg", signal: "Long bias" },
  { sym: "JPM", ivSpread: 0.05, smirk: 2.2, pcr: 0.98, gexFlip: 268.4, regime: "pos", signal: "Neutral" },
  { sym: "AVGO", ivSpread: -0.27, smirk: 4.1, pcr: 1.45, gexFlip: 1620.0, regime: "neg", signal: "Short bias" },
  { sym: "META", ivSpread: 0.38, smirk: 1.3, pcr: 0.68, gexFlip: 588.0, regime: "pos", signal: "Long bias" },
];

const positions = [
  { sym: "NVDA", side: "Long", qty: 120, entry: 138.20, mark: 142.55, pnl: 522.0 },
  { sym: "META", side: "Long", qty: 40, entry: 575.10, mark: 588.30, pnl: 528.0 },
  { sym: "AVGO", side: "Short", qty: 25, entry: 1648.0, mark: 1620.0, pnl: 700.0 },
  { sym: "SPY condor", side: "Short vol", qty: 10, entry: 2.45, mark: 1.80, pnl: 650.0 },
];

const TABS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "strategies", label: "Strategies", icon: Layers },
  { id: "scanner", label: "Scanner", icon: Radar },
  { id: "backtest", label: "Backtest", icon: FlaskConical },
  { id: "gamma", label: "Gamma / Greeks", icon: Activity },
  { id: "positions", label: "Positions", icon: Wallet },
  { id: "settings", label: "Settings", icon: Settings },
] as const;

type TabId = (typeof TABS)[number]["id"];

function Stat({ label, value, delta, up }: { label: string; value: string; delta?: string; up?: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
      {delta && (
        <div className={`mt-1 flex items-center gap-1 text-xs ${up ? "text-bull" : "text-bear"}`}>
          {up ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
          {delta}
        </div>
      )}
    </div>
  );
}

// Marks panels whose data is illustrative (no live source — e.g. options/GEX
// data isn't on the Finnhub free tier). Keeps the demo honest now that the rest
// of the dashboard is wired to real feeds.
function SampleBadge() {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-warn/30 bg-warn/5 px-2 py-0.5 font-mono text-[10px] text-warn">
      <span className="h-1.5 w-1.5 rounded-full bg-warn" />
      sample data
    </span>
  );
}

function Panel({ title, subtitle, children, right }: { title: string; subtitle?: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-5 py-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-foreground">{title}</div>
          {subtitle && <div className="text-xs text-muted-foreground">{subtitle}</div>}
        </div>
        {right}
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

export function TradingFramework() {
  const [tab, setTab] = useState<TabId>("overview");
  const [strats, setStrats] = useState(strategies);
  const [universe, setUniverse] = useState("Large-cap optionable (S&P 100)");

  // Live data from the Vault API.
  const { data: stats } = useStats();
  const { data: portfolio } = usePortfolio();
  const { data: weights } = useWeights();
  const { data: health } = useHealth();

  const acct = portfolio?.account;
  const livePositions = portfolio?.positions ?? [];
  const equity = acct?.equity ?? null;
  const dayPnl = acct ? acct.equity - acct.last_equity : null;
  const dayPnlPct = acct && acct.last_equity ? (acct.equity - acct.last_equity) / acct.last_equity : null;
  const openRisk = livePositions.reduce((a, p) => a + Math.abs(p.market_value), 0);
  const openRiskPct = equity ? openRisk / equity : null;

  // Real equity curve from Alpaca portfolio history. A brand-new account
  // reports all-zero history, so flatten those to the real base value.
  const { data: bt } = useBacktest();
  const btSummary = bt?.summary;
  const btCurve = (bt?.equity_curve ?? []).map((p, i) => ({ d: i, v: p.smart, week: p.week_start }));
  const btTrades = bt?.trades ?? [];

  const { data: eqCurve } = useEquityCurve("1M", "1D");
  const eqData =
    eqCurve?.live && eqCurve.points.length
      ? (() => {
          const base = eqCurve.base_value ?? equity ?? 10000;
          // Drop leading days before the account was funded (Alpaca reports 0).
          const firstReal = eqCurve.points.findIndex((p) => p.v > 0);
          const live = firstReal >= 0 ? eqCurve.points.slice(firstReal) : [];
          if (live.length >= 2) return live.map((p, i) => ({ d: i, v: p.v }));
          // Too new to chart a curve — show a flat line at the real equity.
          return Array.from({ length: 12 }, (_, i) => ({ d: i, v: equity ?? base }));
        })()
      : equityCurve;
  const eqIsLive = !!eqCurve?.live && eqCurve.points.length > 0;

  const toggle = (i: number) =>
    setStrats((s) => s.map((x, idx) => (idx === i ? { ...x, on: !x.on } : x)));

  return (
    <div className="flex h-screen w-full flex-col bg-background font-sans text-foreground">
      <div className="flex flex-1 overflow-hidden">
      {/* Sidebar */}
      <aside className="hidden w-56 flex-col border-r border-border bg-card md:flex">
        <div className="flex items-center gap-2 px-5 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 text-primary">
            <Zap size={18} />
          </div>
          <div>
            <div className="text-sm font-bold text-foreground">Vault</div>
            <div className="text-[10px] text-muted-foreground">paper / research build</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 px-3 py-2">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  active
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                }`}
              >
                <Icon size={17} />
                {t.label}
              </button>
            );
          })}
        </nav>
        <div className="border-t border-border px-5 py-3 text-[11px] text-muted-foreground">
          Data:{" "}
          <span className={health?.live ? "text-primary" : "text-foreground/80"}>
            {health?.live ? "live · Vault API" : "offline"}
          </span>
          <div className="mt-0.5">
            {health?.services.alpaca ? "Alpaca · Finnhub · HydraDB" : "start uvicorn api:app :8000"}
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile brand bar */}
        <div className="flex items-center justify-between border-b border-border bg-card px-4 py-3 md:hidden">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/20 text-bull">
              <Zap size={15} />
            </div>
            <div className="text-sm font-bold text-foreground">Vault</div>
          </div>
          <div className="flex items-center gap-3 text-right">
            <div>
              <div className="text-[9px] uppercase text-muted-foreground">Equity</div>
              <div className="text-xs font-semibold text-foreground">$124,380</div>
            </div>
            <div>
              <div className="text-[9px] uppercase text-muted-foreground">Day</div>
              <div className="text-xs font-semibold text-bull">+$2,400</div>
            </div>
          </div>
        </div>

        {/* Mobile tab strip */}
        <nav className="flex gap-1 overflow-x-auto border-b border-border bg-card px-2 py-2 md:hidden">
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs transition ${
                  active ? "bg-primary/15 text-bull" : "text-muted-foreground hover:bg-secondary"
                }`}
              >
                <Icon size={14} />
                {t.label}
              </button>
            );
          })}
        </nav>

        <header className="hidden flex-wrap items-center justify-between gap-3 border-b border-border bg-card px-6 py-3 md:flex">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-semibold text-foreground">
              {TABS.find((t) => t.id === tab)?.label}
            </span>
            <span className="flex items-center gap-1.5 rounded-full bg-primary/15 px-2.5 py-1 text-xs text-bull">
              <Circle size={8} className="fill-bull text-bull" />
              Positive gamma regime
            </span>
            <span className="text-xs text-muted-foreground">
              SPX zero-gamma flip <span className="text-foreground/80">5,942</span> | VIX <span className="text-foreground/80">13.4</span>
            </span>
            <SampleBadge />
          </div>
          <div className="flex items-center gap-5 text-right">
            <div>
              <div className="text-[10px] uppercase text-muted-foreground">Equity</div>
              <div className="text-sm font-semibold text-foreground">{fmt.usd(equity)}</div>
            </div>
            <div>
              <div className="text-[10px] uppercase text-muted-foreground">Day P&amp;L</div>
              <div className={`text-sm font-semibold ${(dayPnl ?? 0) >= 0 ? "text-bull" : "text-bear"}`}>
                {dayPnl == null ? "—" : `${dayPnl >= 0 ? "+" : ""}${fmt.usd(dayPnl, 2)}`}
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-4 sm:p-6">
          {tab === "overview" && (
            <div className="space-y-4 sm:space-y-5">
              <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
                <Stat
                  label="Net Liq"
                  value={fmt.usd(equity)}
                  delta={dayPnlPct == null ? "paper · Alpaca" : `${fmt.signed(dayPnlPct * 100, 2)}% today`}
                  up={(dayPnl ?? 0) >= 0}
                />
                <Stat
                  label="Open Risk"
                  value={acct ? fmt.usd(openRisk) : "—"}
                  delta={openRiskPct == null ? "no positions" : `${fmt.pct(openRiskPct, 1)} of equity`}
                  up
                />
                <Stat
                  label="Sharpe"
                  value={stats?.sharpe != null ? stats.sharpe.toFixed(2) : "—"}
                  delta={stats?.n_completed ? `${stats.n_completed} closed` : "HydraDB"}
                  up
                />
                <Stat
                  label="Win Rate"
                  value={fmt.pct(stats?.win_rate, 1)}
                  delta={stats?.n_trades != null ? `${stats.n_trades} trades` : "meta-agent"}
                  up
                />
              </div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-3 lg:gap-5">
                <div className="lg:col-span-2">
                  <Panel
                    title="Equity Curve"
                    subtitle={eqIsLive ? "Live · Alpaca paper · daily" : "Paper account, last 60 sessions"}
                  >
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={eqData}>
                        <defs>
                          <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#34d399" stopOpacity={0.35} />
                            <stop offset="100%" stopColor="#34d399" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="#1e293b" vertical={false} />
                        <XAxis dataKey="d" stroke="#475569" fontSize={11} />
                        <YAxis stroke="#475569" fontSize={11} domain={["dataMin - 2000", "dataMax + 2000"]} tickFormatter={usdCompact} width={48} />
                        <Tooltip
                          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, color: "#e2e8f0" }}
                          formatter={(v: number) => [fmt.usd(v, 2), "equity"]}
                        />
                        <Area type="monotone" dataKey="v" stroke="#34d399" strokeWidth={2} fill="url(#eq)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>
                <Panel title="Active Strategies" subtitle={`${strats.filter((s) => s.on).length} live`}>
                  <div className="space-y-3">
                    {strats.filter((s) => s.on).map((s) => (
                      <div key={s.name} className="flex items-center justify-between">
                        <span className="text-sm text-foreground/80">{s.name}</span>
                        <span className="text-xs text-muted-foreground">{s.horizon}</span>
                      </div>
                    ))}
                  </div>
                </Panel>
              </div>
            </div>
          )}

          {tab === "strategies" && (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Strategy library mapped to the research. Toggle to enable, click a row to edit parameters.
              </p>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {strats.map((s, i) => (
                  <div key={s.name} className="rounded-xl border border-border bg-card p-4">
                    <div className="flex items-start justify-between">
                      <div className="min-w-0">
                        <div className="text-sm font-semibold text-foreground">{s.name}</div>
                        <div className="mt-0.5 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                          <span className="rounded bg-secondary px-1.5 py-0.5">{s.type}</span>
                          <span className="rounded bg-secondary px-1.5 py-0.5">{s.horizon}</span>
                          <span className="rounded bg-secondary px-1.5 py-0.5">{s.papers} papers</span>
                        </div>
                      </div>
                      <button
                        onClick={() => toggle(i)}
                        className={`relative h-5 w-9 shrink-0 rounded-full transition ${s.on ? "bg-primary" : "bg-muted"}`}
                      >
                        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${s.on ? "left-4" : "left-0.5"}`} />
                      </button>
                    </div>
                    <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{s.note}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === "scanner" && (
            <Panel title="Signal Scanner" subtitle="Cross-sectional options-implied signals across the universe"
              right={<div className="flex items-center gap-3"><span className="hidden text-xs text-muted-foreground sm:inline">{universe}</span><SampleBadge /></div>}>
              <div className="overflow-x-auto -mx-5">
              <table className="w-full min-w-[640px] px-5 text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase text-muted-foreground">
                    <th className="pb-2">Symbol</th>
                    <th className="pb-2">IV Spread</th>
                    <th className="pb-2">Smirk</th>
                    <th className="pb-2">Put/Call</th>
                    <th className="pb-2">Gamma Flip</th>
                    <th className="pb-2">Regime</th>
                    <th className="pb-2">Signal</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {scannerRows.map((r) => (
                    <tr key={r.sym} className="text-foreground/80">
                      <td className="py-2.5 font-semibold text-foreground">{r.sym}</td>
                      <td className={r.ivSpread >= 0 ? "text-bull" : "text-bear"}>{r.ivSpread.toFixed(2)}</td>
                      <td>{r.smirk.toFixed(1)}</td>
                      <td>{r.pcr.toFixed(2)}</td>
                      <td>{r.gexFlip.toFixed(1)}</td>
                      <td>
                        <span className={`rounded px-1.5 py-0.5 text-[11px] ${r.regime === "pos" ? "bg-primary/15 text-bull" : "bg-destructive/15 text-bear"}`}>
                          {r.regime === "pos" ? "Positive" : "Negative"}
                        </span>
                      </td>
                      <td>
                        <span className={`text-xs ${r.signal.includes("Long") ? "text-bull" : r.signal.includes("Short") || r.signal === "Avoid" ? "text-bear" : "text-muted-foreground"}`}>
                          {r.signal}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </Panel>
          )}

          {tab === "backtest" && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3 lg:gap-5">
              <Panel title="Configure Run" subtitle="OPEX replication · Stivers & Sun (2013)">
                <div className="space-y-4 text-sm">
                  <div>
                    <label className="text-xs text-muted-foreground">Strategy</label>
                    <select className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground">
                      {strats.map((s) => <option key={s.name}>{s.name}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-muted-foreground">Universe</label>
                    <input value={universe} onChange={(e) => setUniverse(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-muted-foreground">Start</label>
                      <input defaultValue="2015-01-01" className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground" />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">End</label>
                      <input defaultValue="2024-12-31" className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground" />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-muted-foreground">Cost (bps)</label>
                      <input defaultValue="5" className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground" />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">Slippage (bps)</label>
                      <input defaultValue="3" className="mt-1 w-full rounded-lg border border-border bg-secondary px-3 py-2 text-foreground" />
                    </div>
                  </div>
                  <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground hover:opacity-90">
                    <Play size={15} /> Run Backtest
                  </button>
                </div>
              </Panel>
              <div className="space-y-4 lg:col-span-2 lg:space-y-5">
                <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
                  <Stat label="CAGR" value={fmt.pct(btSummary?.cagr, 1)} delta={bt?.live ? "in-market wks" : "—"} up={(btSummary?.cagr ?? 0) >= 0} />
                  <Stat label="Sharpe" value={btSummary?.sharpe != null ? btSummary.sharpe.toFixed(2) : "—"} delta="annualized" up />
                  <Stat label="Max DD" value={fmt.pct(btSummary?.max_drawdown, 1)} delta="peak-to-trough" up={(btSummary?.max_drawdown ?? 0) >= -0.05} />
                  <Stat label="Win Rate" value={fmt.pct(btSummary?.win_rate, 0)} delta={btSummary?.n_trades ? `${btSummary.n_trades} trades` : "—"} up />
                </div>
                <Panel
                  title="Walk-Forward Equity"
                  subtitle={bt?.live ? `${bt.strategy} · ${bt.period?.start}–${bt.period?.end} · yfinance` : "Out-of-sample, net of costs"}
                  right={bt?.live ? undefined : <SampleBadge />}
                >
                  <ResponsiveContainer width="100%" height={210}>
                    <AreaChart data={btCurve.length ? btCurve : equityCurve}>
                      <defs>
                        <linearGradient id="bt" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.35} />
                          <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="d" stroke="#475569" fontSize={11} />
                      <YAxis stroke="#475569" fontSize={11} domain={btCurve.length ? ["dataMin", "dataMax"] : ["dataMin - 2000", "dataMax + 2000"]} tickFormatter={btCurve.length ? (v: number) => `${v.toFixed(2)}x` : undefined} width={btCurve.length ? 44 : undefined} />
                      <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} formatter={btCurve.length ? (v: number) => [`${v.toFixed(3)}x`, "equity"] : undefined} />
                      <Area type="monotone" dataKey="v" stroke="#60a5fa" strokeWidth={2} fill="url(#bt)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Panel>
                {btTrades.length > 0 && (
                  <Panel title="Backtested Trades" subtitle={`Smart-OPEX weeks the strategy was in the market · ${btTrades.length}`}>
                    <div className="max-h-72 overflow-y-auto -mx-5">
                      <table className="w-full min-w-[480px] px-5 text-sm">
                        <thead className="sticky top-0 bg-card">
                          <tr className="text-left text-xs uppercase text-muted-foreground">
                            <th className="pb-2">OPEX week</th>
                            <th className="pb-2 text-right">Return</th>
                            <th className="pb-2 text-right">Outcome</th>
                            <th className="pb-2 text-right">Equity</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {[...btTrades].reverse().map((t) => (
                            <tr key={t.week_start} className="text-foreground/80">
                              <td className="py-2 font-medium text-foreground tabular-nums">{t.week_start}</td>
                              <td className={`py-2 text-right tabular-nums ${t.ret_pct >= 0 ? "text-bull" : "text-bear"}`}>
                                {t.ret_pct >= 0 ? "+" : ""}{t.ret_pct.toFixed(2)}%
                              </td>
                              <td className={`py-2 text-right ${t.outcome ? "text-bull" : "text-bear"}`}>
                                {t.outcome ? "win" : "loss"}
                              </td>
                              <td className="py-2 text-right tabular-nums text-muted-foreground">{t.equity.toFixed(3)}x</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Panel>
                )}
              </div>
            </div>
          )}

          {tab === "gamma" && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3 lg:gap-5">
              <div className="lg:col-span-2">
                <Panel title="Net Dealer Gamma by Strike" subtitle="SPX | $B per 1% move | zero-gamma flip interpolated" right={<SampleBadge />}>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={gexByStrike}>
                      <CartesianGrid stroke="#1e293b" vertical={false} />
                      <XAxis dataKey="k" stroke="#475569" fontSize={11} />
                      <YAxis stroke="#475569" fontSize={11} />
                      <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                      <ReferenceLine y={0} stroke="#64748b" />
                      <Bar dataKey="gex" radius={[3, 3, 0, 0]}>
                        {gexByStrike.map((e, i) => (
                          <Cell key={i} fill={e.gex >= 0 ? "#34d399" : "#fb7185"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>
              </div>
              <div className="grid grid-cols-2 gap-3 lg:block lg:space-y-4">
                <Stat label="Net GEX" value="+$5.8B" delta="dealers long gamma" up />
                <Stat label="Zero-Gamma Flip" value="5,942" />
                <Stat label="Call Wall" value="6,000" />
                <Stat label="Put Wall" value="5,850" />
                <div className="col-span-2 rounded-xl border border-border bg-card p-4 text-xs leading-relaxed text-muted-foreground">
                  Greeks engine: delta, gamma, vanna, charm computed per strike from Black-Scholes. Flip solved numerically on a +/-20% spot grid (60 pts).
                </div>
              </div>
            </div>
          )}

          {tab === "positions" && (
            <Panel
              title="Open Positions"
              subtitle={portfolio?.live ? "Live · Alpaca paper account" : "Paper account"}
              right={
                acct ? (
                  <span className="text-xs text-muted-foreground">
                    equity <span className="text-foreground/80">{fmt.usd(acct.equity)}</span> · cash{" "}
                    <span className="text-foreground/80">{fmt.usd(acct.cash)}</span>
                  </span>
                ) : undefined
              }
            >
              <div className="overflow-x-auto -mx-5">
              <table className="w-full min-w-[560px] px-5 text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase text-muted-foreground">
                    <th className="pb-2">Symbol</th>
                    <th className="pb-2">Side</th>
                    <th className="pb-2">Qty</th>
                    <th className="pb-2">Avg Cost</th>
                    <th className="pb-2">Mark</th>
                    <th className="pb-2 text-right">Unrealized P&amp;L</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {livePositions.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-xs text-muted-foreground">
                        {portfolio?.live ? "Flat — no open positions" : "Connect the Vault API to stream positions"}
                      </td>
                    </tr>
                  ) : (
                    livePositions.map((p) => {
                      const isShort = p.side === "short";
                      return (
                        <tr key={p.ticker} className="text-foreground/80">
                          <td className="py-2.5 font-semibold text-foreground">{p.ticker}</td>
                          <td>
                            <span className="flex w-fit items-center gap-1 text-xs">
                              {isShort
                                ? <TrendingDown size={13} className="text-bear" />
                                : <TrendingUp size={13} className="text-bull" />}
                              {isShort ? "Short" : "Long"}
                            </span>
                          </td>
                          <td>{p.qty}</td>
                          <td>{p.avg_entry_price.toFixed(2)}</td>
                          <td>{p.current_price.toFixed(2)}</td>
                          <td className={`text-right font-medium ${p.unrealized_pl >= 0 ? "text-bull" : "text-bear"}`}>
                            {p.unrealized_pl >= 0 ? "+" : ""}${p.unrealized_pl.toFixed(0)} ({fmt.pct(p.unrealized_plpc, 2)})
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
              </div>
            </Panel>
          )}

          {tab === "settings" && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 md:gap-5">
              <Panel title="Data Sources" subtitle="Live service connections">
                <div className="space-y-3 text-sm">
                  {([
                    ["Finnhub — news & quotes", health?.services.finnhub, Activity],
                    ["Alpaca — paper broker", health?.services.alpaca, Wallet],
                    ["HydraDB — memory / RAG", health?.services.hydradb, Database],
                    ["Nebius — LLM agents", health?.services.nebius, Zap],
                  ] as const).map(([n, connected, Icon]) => (
                    <div key={n} className="flex items-center justify-between rounded-lg border border-border bg-secondary/40 px-3 py-2.5">
                      <span className="flex items-center gap-2 text-foreground/80">
                        <Icon size={15} className="text-muted-foreground" /> {n}
                      </span>
                      <span
                        className={`flex items-center gap-1.5 rounded px-2.5 py-1 text-xs ${
                          connected ? "bg-bull/15 text-bull" : "bg-muted text-muted-foreground"
                        }`}
                      >
                        <Circle size={7} className={connected ? "fill-bull text-bull" : "fill-muted-foreground text-muted-foreground"} />
                        {connected ? "Connected" : "Offline"}
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>
              <Panel title="Risk Limits" subtitle="Guardrails">
                <div className="space-y-4 text-sm">
                  {[
                    ["Max position size (% equity)", "8"],
                    ["Max portfolio gross exposure", "150"],
                    ["Short-vol cap (4-sigma gap)", "On"],
                    ["Daily loss limit (%)", "3"],
                  ].map(([n, v]) => (
                    <div key={n} className="flex items-center justify-between">
                      <span className="text-muted-foreground">{n}</span>
                      <input defaultValue={v} className="w-20 rounded border border-border bg-secondary px-2 py-1 text-right text-foreground" />
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          )}
        </main>
      </div>
      </div>
      <TickerTape />
    </div>
  );
}