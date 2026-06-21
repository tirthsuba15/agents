import { ArrowUpRight } from "lucide-react";
import { useTrades, usePortfolio } from "@/hooks/use-stratos";
import { LiveBadge, fmt } from "./live";
import type { Trade } from "@/lib/stratos-api";

export function ExecutionPanel() {
  const { data: tradesData } = useTrades(12);
  const { data: portfolio } = usePortfolio();

  const trades = tradesData?.trades ?? [];
  const account = portfolio?.account;
  const positions = portfolio?.positions ?? [];
  const totalUnrealized = positions.reduce((a, p) => a + p.unrealized_pl, 0);

  return (
    <section id="execution" className="py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">/ 04 · execution</div>
            <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              Conviction in. Position size out.
            </h2>
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            The meta-agent sizes positions from fused conviction, available risk budget, and current portfolio correlation. Orders route to Alpaca paper.
          </p>
        </div>

        {/* Live portfolio summary (Alpaca paper account) */}
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="equity" value={fmt.usd(account?.equity)} badge={<LiveBadge live={!!portfolio?.live} label={portfolio?.live ? "Alpaca" : "demo"} />} />
          <Stat label="cash" value={fmt.usd(account?.cash)} />
          <Stat label="buying power" value={fmt.usd(account?.buying_power)} />
          <Stat
            label="unrealized P&L"
            value={account ? fmt.usd(totalUnrealized, 2) : "—"}
            tone={totalUnrealized >= 0 ? "bull" : "bear"}
          />
        </div>

        {/* Open positions */}
        <div className="mt-8 overflow-hidden rounded-xl border border-border bg-card/70">
          <div className="flex items-center justify-between border-b border-border bg-background/60 px-6 py-3">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              open positions {positions.length ? `· ${positions.length}` : ""}
            </span>
            <LiveBadge live={!!portfolio?.live} label={portfolio?.live ? "live" : "demo"} />
          </div>
          {positions.length === 0 ? (
            <div className="px-6 py-8 text-center font-mono text-xs text-muted-foreground">
              {portfolio?.live ? "flat — no open positions" : "connect the Vault API to stream positions"}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-6 gap-4 border-b border-border bg-background/40 px-6 py-2.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                <div>symbol</div>
                <div>side</div>
                <div className="text-right">qty</div>
                <div className="text-right">avg cost</div>
                <div className="text-right">last</div>
                <div className="text-right">unreal. P&L</div>
              </div>
              {positions.map((p) => {
                const up = p.unrealized_pl >= 0;
                return (
                  <div key={p.ticker} className="grid grid-cols-6 items-center gap-4 border-b border-border/40 px-6 py-3 font-mono text-xs last:border-b-0 hover:bg-background/40">
                    <div className="font-semibold text-foreground">{p.ticker}</div>
                    <div className={p.side === "long" ? "text-bull" : "text-bear"}>
                      {p.side === "long" ? "▲ LONG" : "▼ SHORT"}
                    </div>
                    <div className="text-right tabular-nums text-foreground">{p.qty}</div>
                    <div className="text-right tabular-nums text-muted-foreground">{p.avg_entry_price.toFixed(2)}</div>
                    <div className="text-right tabular-nums text-foreground">{p.current_price.toFixed(2)}</div>
                    <div className={`text-right tabular-nums ${up ? "text-bull" : "text-bear"}`}>
                      {fmt.usd(p.unrealized_pl, 2)} ({fmt.pct(p.unrealized_plpc, 2)})
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>

        {/* Recent agent decisions (HydraDB trade log) */}
        <div className="mt-8 overflow-hidden rounded-xl border border-border bg-card/70">
          <div className="flex items-center justify-between border-b border-border bg-background/60 px-6 py-3">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              agent decision log {trades.length ? `· ${tradesData?.count}` : ""}
            </span>
            <LiveBadge live={!!tradesData?.live} label={tradesData?.live ? "HydraDB" : "demo"} />
          </div>
          <div className="grid grid-cols-6 gap-4 border-b border-border bg-background/40 px-6 py-2.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            <div>time</div>
            <div>symbol</div>
            <div>action</div>
            <div className="text-right">conv</div>
            <div className="text-right">regime</div>
            <div className="text-right">pnl (bps)</div>
          </div>
          {trades.length === 0 ? (
            <div className="px-6 py-8 text-center font-mono text-xs text-muted-foreground">
              {tradesData?.live ? "no trades logged yet" : "connect the Vault API to stream decisions"}
            </div>
          ) : (
            trades.map((tr: Trade, i) => {
              const isBuy = tr.action === "BUY";
              const isSell = tr.action === "SELL";
              const pnlTone = tr.pnl_bps == null ? "text-muted-foreground" : tr.pnl_bps >= 0 ? "text-bull" : "text-bear";
              return (
                <div key={tr.id ?? i} className="grid grid-cols-6 items-center gap-4 border-b border-border/40 px-6 py-3 font-mono text-xs last:border-b-0 hover:bg-background/40">
                  <div className="text-muted-foreground tabular-nums">{fmt.time(tr.timestamp_entry)}</div>
                  <div className="font-semibold text-foreground">{tr.ticker}</div>
                  <div className={isBuy ? "text-bull" : isSell ? "text-bear" : "text-muted-foreground"}>
                    {isBuy ? "▲ BUY" : isSell ? "▼ SELL" : tr.action}
                  </div>
                  <div className="text-right">
                    {tr.conviction == null ? (
                      <span className="tabular-nums text-muted-foreground">—</span>
                    ) : (
                      <div className="ml-auto flex w-20 items-center gap-2">
                        <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                          <div className="h-full bg-primary" style={{ width: `${Math.abs(tr.conviction) * 100}%` }} />
                        </div>
                        <span className="tabular-nums text-muted-foreground">{tr.conviction.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                  <div className="text-right tabular-nums text-muted-foreground">{tr.regime ?? "—"}</div>
                  <div className={`text-right tabular-nums ${pnlTone}`}>{fmt.signed(tr.pnl_bps, 0)}</div>
                </div>
              );
            })
          )}
        </div>

        {/* CTA */}
        <div className="relative mt-20 overflow-hidden rounded-2xl border border-border bg-card/70 p-10 sm:p-14">
          <div className="pointer-events-none absolute inset-0 grid-bg opacity-60" />
          <div className="pointer-events-none absolute -right-32 -top-32 h-80 w-80 rounded-full bg-primary/10 blur-3xl" />
          <div className="relative flex flex-col items-start justify-between gap-8 lg:flex-row lg:items-center">
            <div className="max-w-xl">
              <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">access</div>
              <h3 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">
                Deploy Vault against your own universe.
              </h3>
              <p className="mt-4 text-sm text-muted-foreground sm:text-base">
                Plug in a broker, define a watchlist, and let the agents run. Closed beta — limited seats per quarter.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <a href="#" className="group inline-flex items-center gap-2 rounded-md bg-primary px-6 py-3.5 font-mono text-sm font-semibold text-primary-foreground transition-transform hover:scale-[1.02]">
                Request access
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </a>
              <a href="#" className="inline-flex items-center justify-center rounded-md border border-border bg-background/40 px-6 py-3.5 font-mono text-sm text-foreground hover:border-primary/50 hover:text-primary">
                Read whitepaper
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  tone,
  badge,
}: {
  label: string;
  value: string;
  tone?: "bull" | "bear";
  badge?: React.ReactNode;
}) {
  const c = tone === "bull" ? "text-bull" : tone === "bear" ? "text-bear" : "text-foreground";
  return (
    <div className="rounded-xl border border-border bg-background/60 p-5">
      <div className="flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
        {badge}
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold tabular-nums ${c}`}>{value}</div>
    </div>
  );
}
