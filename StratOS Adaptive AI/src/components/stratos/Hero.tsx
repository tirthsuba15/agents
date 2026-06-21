import { ArrowUpRight, GitBranch } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { LiveChart } from "./LiveChart";
import { useStats } from "@/hooks/use-stratos";
import { fmt } from "./live";

export function Hero() {
  const { data: stats } = useStats();

  // Real metrics computed from HydraDB trades + Alpaca equity, with fallbacks.
  const metrics = [
    {
      k: "Sharpe",
      v: stats?.sharpe != null ? stats.sharpe.toFixed(2) : "—",
      d: stats?.n_completed ? `${stats.n_completed} closed trades` : "trailing",
    },
    {
      k: "Win rate",
      v: fmt.pct(stats?.win_rate, 1),
      d: stats?.n_trades != null ? `${stats.n_trades} trades logged` : "meta-agent",
    },
    {
      k: "Equity",
      v: fmt.usd(stats?.equity, 0),
      d: "paper · Alpaca",
    },
  ];

  return (
    <section className="relative">
      <div className="pointer-events-none absolute inset-0 grid-bg" />
      <div className="relative mx-auto grid max-w-7xl gap-12 px-6 pt-20 pb-24 lg:grid-cols-12 lg:gap-8 lg:pt-28">
        <div className="lg:col-span-7">
          <div className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/60 px-3 py-1 font-mono text-[11px] text-muted-foreground animate-rise">
            <span className="h-1.5 w-1.5 rounded-full bg-bull animate-blink" />
            multi-agent · LangGraph · self-evolving
          </div>

          <h1 className="mt-6 font-display text-5xl font-semibold leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl animate-rise" style={{ animationDelay: "60ms" }}>
            Signals in.<br />
            <span className="text-shimmer">Conviction out.</span>
          </h1>

          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg animate-rise" style={{ animationDelay: "140ms" }}>
            Vault decomposes market analysis into three specialized signal agents — sentiment, momentum, and gamma — fused by a meta-agent that continuously rewrites its own decision weights from live trade outcomes.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3 animate-rise" style={{ animationDelay: "220ms" }}>
            <Link to="/dashboard" className="group inline-flex items-center gap-2 rounded-md bg-primary px-5 py-3 font-mono text-sm font-semibold text-primary-foreground transition-transform hover:scale-[1.02]">
              Launch dashboard
              <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <a href="#evolution" className="inline-flex items-center gap-2 rounded-md border border-border bg-card/60 px-5 py-3 font-mono text-sm text-foreground hover:border-primary/50 hover:text-primary">
              <GitBranch className="h-4 w-4" />
              Live evolution log
            </a>
          </div>

          <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6 border-t border-border/60 pt-6 animate-rise" style={{ animationDelay: "300ms" }}>
            {metrics.map((s) => (
              <div key={s.k}>
                <dt className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{s.k}</dt>
                <dd className="mt-1 font-mono text-2xl font-semibold tabular-nums text-foreground">{s.v}</dd>
                <dd className="font-mono text-[11px] text-muted-foreground/80">{s.d}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="lg:col-span-5">
          <LiveChart />
        </div>
      </div>
    </section>
  );
}