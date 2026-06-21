import { useWeights, useMemory, useAccuracy } from "@/hooks/use-stratos";
import { LiveBadge, fmt } from "./live";

const FALLBACK_WEIGHTS = [
  { agent: "sentiment", w: 0.4, tone: "bull" as const },
  { agent: "momentum", w: 0.35, tone: "bull" as const },
  { agent: "gamma", w: 0.25, tone: "warn" as const },
];

interface LogLine {
  t: string;
  msg: string;
}

export function EvolutionPanel() {
  const { data: weightsData } = useWeights();
  const { data: memory } = useMemory();
  const { data: accuracy } = useAccuracy();

  const live = !!weightsData?.live;

  const weights = weightsData
    ? [
        { agent: "sentiment", w: weightsData.w_sentiment, tone: "bull" as const },
        { agent: "momentum", w: weightsData.w_momentum, tone: "bull" as const },
        { agent: "gamma", w: weightsData.w_gamma, tone: "warn" as const },
      ]
    : FALLBACK_WEIGHTS;

  // Build the evolution log from real memory events (most recent first).
  const log: LogLine[] = [];
  if (weightsData?.timestamp) {
    log.push({
      t: fmt.time(weightsData.timestamp),
      msg: `reweight (${weightsData.trigger ?? "manual"}) → sent ${fmt.pct(
        weightsData.w_sentiment,
      )} · mom ${fmt.pct(weightsData.w_momentum)} · gam ${fmt.pct(weightsData.w_gamma)}`,
    });
  }
  if (accuracy?.n_trades) {
    log.push({
      t: fmt.time(weightsData?.timestamp),
      msg: `accuracy recompute over ${accuracy.n_trades} closed trades`,
    });
  }
  for (const ev of memory?.recent ?? []) {
    const label =
      ev.kind === "trade"
        ? `trade logged · ${ev.id}`
        : ev.kind === "weights"
          ? `decision_weights updated · ${ev.id}`
          : ev.kind === "pass"
            ? `signal evaluated → PASS · ${ev.id}`
            : ev.kind === "candidate"
              ? `candidate strategy admitted · ${ev.id}`
              : `${ev.kind} · ${ev.id}`;
    log.push({ t: fmt.time(ev.timestamp), msg: label });
  }
  const logLines = (log.length ? log : FALLBACK_LOG).slice(0, 8);

  const totalMem = memory?.total ?? null;
  const candidateCount = memory?.by_kind?.candidate ?? 0;
  const weightUpdates = memory?.by_kind?.weights ?? 0;

  return (
    <section id="evolution" className="border-y border-border/60 bg-card/30 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">/ 03 · self-evolution</div>
            <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              The system grades itself, then reweights.
            </h2>
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            Every closed position writes a reward signal back to the meta-agent. Weights drift. Candidate strategies enter the pool. Nothing is static.
          </p>
        </div>

        <div className="mt-12 grid gap-5 lg:grid-cols-5">
          {/* Weights */}
          <div className="rounded-xl border border-border bg-background/60 p-6 lg:col-span-2">
            <div className="flex items-center justify-between">
              <div className="font-mono text-xs font-semibold text-foreground">decision_weights.json</div>
              <LiveBadge live={live} label={live ? "HydraDB" : "demo"} />
            </div>
            <div className="mt-5 space-y-4">
              {weights.map((w) => {
                const pct = ((w.w ?? 0) * 100).toFixed(1);
                const tone = w.tone === "bull" ? "bg-bull" : "bg-warn";
                const text = w.tone === "bull" ? "text-bull" : "text-warn";
                return (
                  <div key={w.agent}>
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <span className="text-muted-foreground">{w.agent}_agent</span>
                      <span className={`tabular-nums ${text}`}>{pct}%</span>
                    </div>
                    <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
                      <div
                        className={`h-full ${tone} transition-all duration-700 ease-out`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-6 grid grid-cols-3 gap-3 border-t border-border/60 pt-4 font-mono text-[11px]">
              <div>
                <div className="text-muted-foreground">memory records</div>
                <div className="text-foreground tabular-nums text-lg font-semibold">{totalMem ?? "—"}</div>
              </div>
              <div>
                <div className="text-muted-foreground">candidates</div>
                <div className="text-foreground tabular-nums text-lg font-semibold">{candidateCount}</div>
              </div>
              <div>
                <div className="text-muted-foreground">reweights</div>
                <div className="text-bull tabular-nums text-lg font-semibold">{weightUpdates}</div>
              </div>
            </div>
          </div>

          {/* Log */}
          <div className="rounded-xl border border-border bg-background/60 p-6 lg:col-span-3">
            <div className="flex items-center justify-between">
              <div className="font-mono text-xs font-semibold text-foreground">~/stratos/evolution.log</div>
              <div className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-bear/70" />
                <span className="h-2 w-2 rounded-full bg-warn/80" />
                <span className="h-2 w-2 rounded-full bg-bull" />
              </div>
            </div>
            <div className="mt-5 space-y-1.5 font-mono text-[12px] leading-relaxed">
              {logLines.map((l, i) => (
                <div
                  key={`${l.t}-${i}`}
                  className="flex items-start gap-3 animate-rise"
                  style={{ opacity: 1 - i * 0.08 }}
                >
                  <span className="text-muted-foreground/70 tabular-nums">{l.t}</span>
                  <span className="text-primary">›</span>
                  <span className="text-foreground/90">{l.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const FALLBACK_LOG: LogLine[] = [
  { t: "—", msg: "waiting for Vault API · start uvicorn api:app on :8000" },
  { t: "—", msg: "decision_weights.json will stream from HydraDB once connected" },
];
