import { Newspaper, TrendingUp, Sigma, Cpu, Zap } from "lucide-react";
import { useWeights, useStats, usePortfolio } from "@/hooks/use-stratos";
import { fmt } from "./live";

/**
 * Animated LangGraph-style architecture diagram.
 * Three signal agents feed a meta-agent which routes to execution.
 * Node tags reflect live decision weights, conviction, and open positions.
 */
export function AgentGraph() {
  const { data: weights } = useWeights();
  const { data: stats } = useStats();
  const { data: portfolio } = usePortfolio();
  const posCount = portfolio?.positions?.length ?? 0;

  return (
    <section id="architecture" className="relative border-y border-border/60 bg-card/30 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">/ 01 · architecture</div>
            <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              A LangGraph of specialists, not a monolith.
            </h2>
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            Each signal agent owns one domain end-to-end. The meta-agent fuses their outputs into a single conviction score and decides position size.
          </p>
        </div>

        <div className="relative mt-14 rounded-xl border border-border bg-background/60 p-6 sm:p-10">
          <svg viewBox="0 0 900 460" className="h-auto w-full">
            <defs>
              <linearGradient id="lineGrad" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0" stopColor="var(--bull)" stopOpacity="0.1" />
                <stop offset="0.5" stopColor="var(--bull)" stopOpacity="0.9" />
                <stop offset="1" stopColor="var(--bull)" stopOpacity="0.1" />
              </linearGradient>
              <linearGradient id="lineWarn" x1="0" x2="1" y1="0" y2="0">
                <stop offset="0" stopColor="var(--warn)" stopOpacity="0.1" />
                <stop offset="0.5" stopColor="var(--warn)" stopOpacity="0.9" />
                <stop offset="1" stopColor="var(--warn)" stopOpacity="0.1" />
              </linearGradient>
              <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
                <circle cx="1" cy="1" r="1" fill="oklch(0.28 0.018 240)" />
              </pattern>
            </defs>
            <rect width="900" height="460" fill="url(#dots)" opacity="0.4" />

            {/* Data source labels */}
            {[
              { x: 60, y: 80, label: "Finnhub · news" },
              { x: 60, y: 230, label: "OHLCV · candles" },
              { x: 60, y: 380, label: "Options flow · dark pool" },
            ].map((s) => (
              <g key={s.label}>
                <rect x={s.x - 20} y={s.y - 16} width="170" height="32" rx="4" fill="oklch(0.18 0.014 240)" stroke="oklch(0.28 0.018 240)" />
                <text x={s.x + 65} y={s.y + 5} textAnchor="middle" className="fill-muted-foreground font-mono" fontSize="11">
                  {s.label}
                </text>
              </g>
            ))}

            {/* Edges: source → agent */}
            {[
              { d: "M 230 80 C 290 80, 290 130, 360 130", color: "var(--bull)" },
              { d: "M 230 230 C 290 230, 290 230, 360 230", color: "var(--bull)" },
              { d: "M 230 380 C 290 380, 290 330, 360 330", color: "var(--warn)" },
            ].map((e, i) => (
              <g key={i}>
                <path d={e.d} stroke="oklch(0.28 0.018 240)" strokeWidth="1.5" fill="none" />
                <path d={e.d} stroke={e.color} strokeWidth="1.8" fill="none" strokeDasharray="6 14" className="animate-flow" opacity="0.9" />
              </g>
            ))}

            {/* Edges: agent → meta */}
            {[
              { d: "M 530 130 C 600 130, 600 200, 660 220", color: "var(--bull)" },
              { d: "M 530 230 L 660 230", color: "var(--bull)" },
              { d: "M 530 330 C 600 330, 600 260, 660 240", color: "var(--warn)" },
            ].map((e, i) => (
              <g key={i}>
                <path d={e.d} stroke="oklch(0.28 0.018 240)" strokeWidth="1.5" fill="none" />
                <path d={e.d} stroke={e.color} strokeWidth="1.8" fill="none" strokeDasharray="6 14" className="animate-flow" opacity="0.9" style={{ animationDelay: `${i * 0.3}s` }} />
              </g>
            ))}

            {/* Meta → execution */}
            <path d="M 780 230 L 860 230" stroke="oklch(0.28 0.018 240)" strokeWidth="1.5" fill="none" />
            <path d="M 780 230 L 860 230" stroke="var(--primary)" strokeWidth="2" fill="none" strokeDasharray="6 10" className="animate-flow" />

            {/* Feedback loop (execution → meta, dashed curve back) */}
            <path
              d="M 860 250 C 860 420, 400 440, 360 250"
              fill="none"
              stroke="var(--warn)"
              strokeWidth="1.2"
              strokeDasharray="3 6"
              opacity="0.5"
            />
            <text x="500" y="430" textAnchor="middle" className="fill-warn font-mono" fontSize="10">
              ← reward signal · weights update
            </text>
          </svg>

          {/* Overlay HTML nodes for richer styling */}
          <div className="pointer-events-none absolute inset-0 hidden sm:block">
            <Node x="40%" y="28%" icon={<Newspaper className="h-3.5 w-3.5" />} title="Sentiment" tag={`w · ${fmt.pct(weights?.w_sentiment, 0)}`} tone="bull" />
            <Node x="40%" y="50%" icon={<TrendingUp className="h-3.5 w-3.5" />} title="Momentum" tag={`w · ${fmt.pct(weights?.w_momentum, 0)}`} tone="bull" />
            <Node x="40%" y="72%" icon={<Sigma className="h-3.5 w-3.5" />} title="Gamma" tag={`w · ${fmt.pct(weights?.w_gamma, 0)}`} tone="warn" />
            <Node x="74%" y="50%" icon={<Cpu className="h-3.5 w-3.5" />} title="Meta-agent" tag={`conviction ${fmt.num(stats?.avg_conviction, 2)}`} tone="primary" wide />
            <Node x="94%" y="50%" icon={<Zap className="h-3.5 w-3.5" />} title="Exec" tag={posCount ? `${posCount} positions` : "flat"} tone="primary" small />
          </div>
        </div>
      </div>
    </section>
  );
}

function Node({
  x,
  y,
  icon,
  title,
  tag,
  tone,
  wide,
  small,
}: {
  x: string;
  y: string;
  icon: React.ReactNode;
  title: string;
  tag: string;
  tone: "bull" | "warn" | "primary";
  wide?: boolean;
  small?: boolean;
}) {
  const toneCls =
    tone === "bull" ? "border-bull/40 bg-bull/5 text-bull" : tone === "warn" ? "border-warn/40 bg-warn/5 text-warn" : "border-primary/50 bg-primary/10 text-primary";
  return (
    <div
      className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2"
      style={{ left: x, top: y }}
    >
      <div className={`flex ${wide ? "w-44" : small ? "w-28" : "w-40"} items-center gap-2 rounded-lg border ${toneCls} px-3 py-2 backdrop-blur-sm animate-pulse-node`}>
        <span className="grid h-6 w-6 place-items-center rounded-md bg-background/60">{icon}</span>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[11px] font-semibold text-foreground">{title}</div>
          <div className="truncate font-mono text-[10px] opacity-80">{tag}</div>
        </div>
      </div>
    </div>
  );
}