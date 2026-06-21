import { Database, KeyRound, Link2 } from "lucide-react";
import { useMemory } from "@/hooks/use-stratos";
import { LiveBadge } from "./live";

type Col = { name: string; type: string; pk?: boolean; fk?: string };
type Table = {
  id: string;
  name: string;
  tag: string;
  tone: "bull" | "warn" | "primary" | "muted";
  x: number; // % from left of canvas
  y: number; // % from top of canvas
  cols: Col[];
};

const TABLES: Table[] = [
  {
    id: "headlines",
    name: "headlines",
    tag: "sentiment · raw",
    tone: "bull",
    x: 2,
    y: 0,
    cols: [
      { name: "id", type: "uuid", pk: true },
      { name: "source", type: "text" },
      { name: "symbol", type: "text" },
      { name: "body", type: "text" },
      { name: "ts", type: "timestamptz" },
    ],
  },
  {
    id: "candles",
    name: "candles_1m",
    tag: "momentum · raw",
    tone: "bull",
    x: 2,
    y: 38,
    cols: [
      { name: "symbol", type: "text", pk: true },
      { name: "ts", type: "timestamptz", pk: true },
      { name: "o,h,l,c", type: "numeric" },
      { name: "volume", type: "bigint" },
    ],
  },
  {
    id: "gamma",
    name: "gamma_snapshots",
    tag: "gamma · raw",
    tone: "warn",
    x: 2,
    y: 73,
    cols: [
      { name: "id", type: "uuid", pk: true },
      { name: "symbol", type: "text" },
      { name: "gex_total", type: "numeric" },
      { name: "gamma_flip", type: "numeric" },
      { name: "dpi", type: "numeric" },
    ],
  },
  {
    id: "signals",
    name: "signals",
    tag: "agent output",
    tone: "primary",
    x: 38,
    y: 18,
    cols: [
      { name: "id", type: "uuid", pk: true },
      { name: "agent", type: "agent_kind" },
      { name: "symbol", type: "text" },
      { name: "direction", type: "smallint" },
      { name: "confidence", type: "numeric" },
      { name: "source_ref", type: "uuid", fk: "headlines.id" },
    ],
  },
  {
    id: "strategies",
    name: "strategies",
    tag: "candidate pool",
    tone: "muted",
    x: 38,
    y: 68,
    cols: [
      { name: "id", type: "uuid", pk: true },
      { name: "slug", type: "text" },
      { name: "status", type: "strategy_status" },
      { name: "score", type: "numeric" },
    ],
  },
  {
    id: "weights",
    name: "decision_weights",
    tag: "meta · state",
    tone: "primary",
    x: 70,
    y: 4,
    cols: [
      { name: "version", type: "bigint", pk: true },
      { name: "sentiment_w", type: "numeric" },
      { name: "momentum_w", type: "numeric" },
      { name: "gamma_w", type: "numeric" },
      { name: "applied_at", type: "timestamptz" },
    ],
  },
  {
    id: "trades",
    name: "trades",
    tag: "execution",
    tone: "bull",
    x: 70,
    y: 42,
    cols: [
      { name: "id", type: "uuid", pk: true },
      { name: "symbol", type: "text" },
      { name: "side", type: "side_kind" },
      { name: "qty", type: "int" },
      { name: "conviction", type: "numeric" },
      { name: "weights_v", type: "bigint", fk: "decision_weights.version" },
      { name: "signal_id", type: "uuid", fk: "signals.id" },
    ],
  },
  {
    id: "rewards",
    name: "rewards",
    tag: "feedback loop",
    tone: "warn",
    x: 70,
    y: 78,
    cols: [
      { name: "trade_id", type: "uuid", pk: true, fk: "trades.id" },
      { name: "pnl_r", type: "numeric" },
      { name: "label", type: "outcome_kind" },
      { name: "closed_at", type: "timestamptz" },
    ],
  },
];

// Edges: from column anchor → to column anchor
const EDGES: { from: string; fromCol: string; to: string; toCol: string; tone: "bull" | "warn" | "primary" }[] = [
  { from: "headlines", fromCol: "id", to: "signals", toCol: "source_ref", tone: "bull" },
  { from: "candles", fromCol: "symbol", to: "signals", toCol: "symbol", tone: "bull" },
  { from: "gamma", fromCol: "symbol", to: "signals", toCol: "symbol", tone: "warn" },
  { from: "signals", fromCol: "id", to: "trades", toCol: "signal_id", tone: "primary" },
  { from: "weights", fromCol: "version", to: "trades", toCol: "weights_v", tone: "primary" },
  { from: "trades", fromCol: "id", to: "rewards", toCol: "trade_id", tone: "warn" },
  { from: "rewards", fromCol: "pnl_r", to: "weights", toCol: "sentiment_w", tone: "warn" },
  { from: "rewards", fromCol: "pnl_r", to: "strategies", toCol: "score", tone: "warn" },
];

const CANVAS_W = 1200;
const CANVAS_H = 760;
const CARD_W = 280;
const HEADER_H = 56;
const ROW_H = 22;

function tableBox(t: Table) {
  const px = (t.x / 100) * CANVAS_W;
  const py = (t.y / 100) * CANVAS_H;
  return { x: px, y: py, w: CARD_W, h: HEADER_H + t.cols.length * ROW_H + 14 };
}

function colAnchor(t: Table, colName: string, side: "left" | "right") {
  const box = tableBox(t);
  const idx = t.cols.findIndex((c) => c.name === colName);
  const safeIdx = idx === -1 ? 0 : idx;
  const y = box.y + HEADER_H + safeIdx * ROW_H + ROW_H / 2 + 7;
  const x = side === "left" ? box.x : box.x + box.w;
  return { x, y };
}

export function MemorySchema() {
  const { data: memory } = useMemory();
  const byKind = memory?.by_kind ?? {};
  const kindOrder = ["trade", "weights", "candidate", "pass"];
  const kinds = Object.keys(byKind).sort(
    (a, b) => (kindOrder.indexOf(a) + 1 || 99) - (kindOrder.indexOf(b) + 1 || 99),
  );
  const kindTone: Record<string, string> = {
    trade: "text-bull",
    weights: "text-primary",
    candidate: "text-muted-foreground",
    pass: "text-warn",
  };

  return (
    <section id="memory" className="border-y border-border/60 bg-card/30 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">/ 05 · memory</div>
            <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              How everything ties together in the database.
            </h2>
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            Every raw tick, every agent signal, every fill, and every reward is persisted — and joined back to the weight version that produced it.
          </p>
        </div>

        {/* Live HydraDB memory stats */}
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-border bg-background/60 p-5">
            <div className="flex items-center justify-between">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">memory records</div>
              <LiveBadge live={!!memory?.live} label={memory?.live ? "HydraDB" : "demo"} />
            </div>
            <div className="mt-2 font-mono text-2xl font-semibold tabular-nums text-foreground">
              {memory?.total ?? "—"}
            </div>
          </div>
          {kinds.slice(0, 3).map((k) => (
            <div key={k} className="rounded-xl border border-border bg-background/60 p-5">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
              <div className={`mt-2 font-mono text-2xl font-semibold tabular-nums ${kindTone[k] ?? "text-foreground"}`}>
                {byKind[k]}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 overflow-hidden rounded-xl border border-border bg-background/60">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3 font-mono text-[11px]">
            <div className="flex items-center gap-2 text-foreground">
              <Database className="h-3.5 w-3.5 text-primary" />
              <span>stratos · HydraDB · memory schema</span>
            </div>
            <div className="flex items-center gap-4 text-muted-foreground">
              <span className="flex items-center gap-1.5"><KeyRound className="h-3 w-3 text-primary" />pk</span>
              <span className="flex items-center gap-1.5"><Link2 className="h-3 w-3 text-warn" />fk</span>
              <span className="flex items-center gap-1.5"><span className="h-1.5 w-3 rounded-full bg-warn animate-blink" />reward loop</span>
            </div>
          </div>

          <div className="relative overflow-x-auto">
            <svg viewBox={`0 0 ${CANVAS_W} ${CANVAS_H}`} className="block h-auto w-full min-w-[900px]">
              <defs>
                <pattern id="schemadots" width="22" height="22" patternUnits="userSpaceOnUse">
                  <circle cx="1" cy="1" r="1" fill="oklch(0.28 0.018 240)" />
                </pattern>
                <marker id="arrow-bull" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--bull)" />
                </marker>
                <marker id="arrow-warn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--warn)" />
                </marker>
                <marker id="arrow-primary" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--primary)" />
                </marker>
              </defs>
              <rect width={CANVAS_W} height={CANVAS_H} fill="url(#schemadots)" opacity="0.4" />

              {/* Edges drawn before tables so they sit beneath */}
              {EDGES.map((e, i) => {
                const fromTable = TABLES.find((t) => t.id === e.from)!;
                const toTable = TABLES.find((t) => t.id === e.to)!;
                const fromBox = tableBox(fromTable);
                const toBox = tableBox(toTable);
                const fromOnRight = fromBox.x + fromBox.w / 2 < toBox.x + toBox.w / 2;
                const a = colAnchor(fromTable, e.fromCol, fromOnRight ? "right" : "left");
                const b = colAnchor(toTable, e.toCol, fromOnRight ? "left" : "right");
                const dx = Math.abs(b.x - a.x);
                const cp = Math.max(60, dx * 0.5);
                const d = `M ${a.x} ${a.y} C ${a.x + (fromOnRight ? cp : -cp)} ${a.y}, ${b.x + (fromOnRight ? -cp : cp)} ${b.y}, ${b.x} ${b.y}`;
                const stroke =
                  e.tone === "bull" ? "var(--bull)" : e.tone === "warn" ? "var(--warn)" : "var(--primary)";
                const marker =
                  e.tone === "bull" ? "url(#arrow-bull)" : e.tone === "warn" ? "url(#arrow-warn)" : "url(#arrow-primary)";
                return (
                  <g key={i}>
                    <path d={d} fill="none" stroke="oklch(0.28 0.018 240)" strokeWidth="1.5" />
                    <path
                      d={d}
                      fill="none"
                      stroke={stroke}
                      strokeWidth="1.6"
                      strokeDasharray="5 12"
                      className="animate-flow"
                      style={{ animationDelay: `${i * 0.25}s` }}
                      markerEnd={marker}
                      opacity="0.95"
                    />
                  </g>
                );
              })}

              {/* Tables */}
              {TABLES.map((t) => {
                const box = tableBox(t);
                const accent =
                  t.tone === "bull"
                    ? "var(--bull)"
                    : t.tone === "warn"
                      ? "var(--warn)"
                      : t.tone === "primary"
                        ? "var(--primary)"
                        : "oklch(0.66 0.018 230)";
                return (
                  <g key={t.id}>
                    {/* card */}
                    <rect
                      x={box.x}
                      y={box.y}
                      width={box.w}
                      height={box.h}
                      rx="8"
                      fill="oklch(0.18 0.014 240)"
                      stroke="oklch(0.28 0.018 240)"
                      strokeWidth="1"
                    />
                    {/* accent stripe */}
                    <rect x={box.x} y={box.y} width="3" height={box.h} rx="2" fill={accent} />
                    {/* header */}
                    <text x={box.x + 16} y={box.y + 22} className="font-mono fill-foreground" fontSize="13" fontWeight="600">
                      {t.name}
                    </text>
                    <text x={box.x + 16} y={box.y + 40} className="font-mono fill-muted-foreground" fontSize="10">
                      {t.tag}
                    </text>
                    <line x1={box.x + 8} x2={box.x + box.w - 8} y1={box.y + HEADER_H - 6} y2={box.y + HEADER_H - 6} stroke="oklch(0.28 0.018 240)" />

                    {/* columns */}
                    {t.cols.map((c, idx) => {
                      const cy = box.y + HEADER_H + idx * ROW_H + 14;
                      const isPk = !!c.pk;
                      const isFk = !!c.fk;
                      return (
                        <g key={c.name}>
                          {isPk && (
                            <circle cx={box.x + 16} cy={cy - 4} r="3" fill="var(--primary)" />
                          )}
                          {!isPk && isFk && (
                            <circle cx={box.x + 16} cy={cy - 4} r="3" fill="none" stroke="var(--warn)" strokeWidth="1.2" />
                          )}
                          <text
                            x={box.x + 28}
                            y={cy}
                            className={`font-mono ${isPk ? "fill-primary" : "fill-foreground"}`}
                            fontSize="11"
                            fontWeight={isPk ? 600 : 400}
                          >
                            {c.name}
                          </text>
                          <text x={box.x + box.w - 14} y={cy} textAnchor="end" className="font-mono fill-muted-foreground" fontSize="10">
                            {c.type}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Legend / SQL hint footer */}
          <div className="grid gap-0 border-t border-border/60 font-mono text-[11px] sm:grid-cols-3">
            <div className="border-b border-border/60 p-4 sm:border-b-0 sm:border-r">
              <div className="text-muted-foreground">raw streams →</div>
              <div className="mt-1 text-foreground">
                <span className="text-bull">headlines</span>, <span className="text-bull">candles_1m</span>, <span className="text-warn">gamma_snapshots</span>
              </div>
            </div>
            <div className="border-b border-border/60 p-4 sm:border-b-0 sm:border-r">
              <div className="text-muted-foreground">agent state →</div>
              <div className="mt-1 text-foreground">
                <span className="text-primary">signals</span>, <span className="text-primary">decision_weights</span>, <span className="text-muted-foreground">strategies</span>
              </div>
            </div>
            <div className="p-4">
              <div className="text-muted-foreground">feedback loop →</div>
              <div className="mt-1 text-foreground">
                <span className="text-bull">trades</span> → <span className="text-warn">rewards</span> → <span className="text-primary">decision_weights++</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}