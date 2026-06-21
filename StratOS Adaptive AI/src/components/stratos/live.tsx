/**
 * Shared live-status primitives + formatters used across Vault panels.
 */
import { useHealth } from "@/hooks/use-stratos";

/** Small pill that reflects whether a given panel is showing live API data. */
export function LiveBadge({ live, label }: { live: boolean | undefined; label?: string }) {
  if (live) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/5 px-2 py-0.5 font-mono text-[10px] text-primary">
        <span className="h-1.5 w-1.5 rounded-full bg-primary animate-blink" />
        {label ?? "live"}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60" />
      {label ?? "demo"}
    </span>
  );
}

/** Global connection indicator (used in Nav) driven by /api/health. */
export function ConnectionBadge() {
  const { data, isError } = useHealth();
  const connected = !!data?.live && !isError;
  return <LiveBadge live={connected} label={connected ? "LIVE · API" : "DEMO"} />;
}

// ── formatters ───────────────────────────────────────────────────
export const fmt = {
  pct: (v: number | null | undefined, digits = 1) =>
    v == null ? "—" : `${(v * 100).toFixed(digits)}%`,
  num: (v: number | null | undefined, digits = 2) =>
    v == null ? "—" : v.toFixed(digits),
  usd: (v: number | null | undefined, digits = 0) =>
    v == null
      ? "—"
      : v.toLocaleString("en-US", {
          style: "currency",
          currency: "USD",
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        }),
  signed: (v: number | null | undefined, digits = 2) =>
    v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`,
  time: (iso: string | null | undefined) => {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleTimeString("en-GB", { hour12: false });
  },
};
