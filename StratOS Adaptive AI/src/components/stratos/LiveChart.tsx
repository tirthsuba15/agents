import { useEffect, useState } from "react";
import { useQuotes } from "@/hooks/use-stratos";

// Deterministic PRNG so SSR and first client render match (no hydration mismatch).
function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function genCandles(n: number, seed = 100, rngSeed = 42) {
  const rng = mulberry32(rngSeed);
  let p = seed;
  return Array.from({ length: n }, (_, i) => {
    const o = p;
    const c = o + (Math.sin(i * 0.6) + (rng() - 0.5)) * 2.4;
    const h = Math.max(o, c) + rng() * 1.4;
    const l = Math.min(o, c) - rng() * 1.4;
    p = c;
    return { o, c, h, l };
  });
}

export function LiveChart() {
  const [candles, setCandles] = useState(() => genCandles(40));
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = setInterval(() => {
      setCandles((prev) => {
        const last = prev[prev.length - 1];
        const o = last.c;
        const c = o + (Math.sin(tick * 0.6) + (Math.random() - 0.5)) * 2.4;
        const h = Math.max(o, c) + Math.random() * 1.4;
        const l = Math.min(o, c) - Math.random() * 1.4;
        return [...prev.slice(1), { o, c, h, l }];
      });
      setTick((x) => x + 1);
    }, 900);
    return () => clearInterval(t);
  }, [tick]);

  const w = 520;
  const h = 360;
  const padX = 16;
  const padY = 24;
  const allVals = candles.flatMap((c) => [c.h, c.l]);
  const min = Math.min(...allVals) - 1;
  const max = Math.max(...allVals) + 1;
  const cw = (w - padX * 2) / candles.length;
  const y = (v: number) => padY + ((max - v) / (max - min)) * (h - padY * 2);

  const last = candles[candles.length - 1];
  const first = candles[0];

  // Real NVDA quote for the header readout (intraday shape stays generative —
  // Finnhub's candle endpoint is premium-gated). Falls back to synthetic series.
  const { data: quotes } = useQuotes("NVDA");
  const nvda = quotes?.live ? quotes.quotes.find((q) => q.symbol === "NVDA") : undefined;
  const priceLabel = nvda ? nvda.price : last.c;
  const deltaAbs = nvda ? nvda.change : last.c - first.o;
  const deltaPct = nvda ? nvda.change_pct : ((last.c - first.o) / first.o) * 100;
  const up = deltaAbs >= 0;

  return (
    <div className="relative overflow-hidden rounded-lg border border-border bg-card/70 p-5 backdrop-blur animate-rise" style={{ animationDelay: "120ms" }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs font-semibold text-foreground">NVDA · 1m</span>
          <span className="flex items-center gap-1.5 rounded-full bg-bull/10 px-2 py-0.5 font-mono text-[10px] text-bull">
            <span className="h-1.5 w-1.5 rounded-full bg-bull animate-blink" /> LONG · 0.82
          </span>
        </div>
        <div className="text-right font-mono">
          <div className="text-lg font-semibold tabular-nums">{priceLabel.toFixed(2)}</div>
          <div className={`text-[11px] tabular-nums ${up ? "text-bull" : "text-bear"}`}>
            {up ? "+" : ""}
            {deltaAbs.toFixed(2)} ({deltaPct.toFixed(2)}%)
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${w} ${h}`} className="mt-4 h-72 w-full">
        <defs>
          <pattern id="chartgrid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="oklch(0.28 0.018 240)" strokeWidth="0.5" />
          </pattern>
          <linearGradient id="vwap" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="oklch(0.78 0.18 75)" stopOpacity="0" />
            <stop offset="0.5" stopColor="oklch(0.78 0.18 75)" />
            <stop offset="1" stopColor="oklch(0.78 0.18 75)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <rect width={w} height={h} fill="url(#chartgrid)" opacity="0.4" />

        {/* VWAP line */}
        <path
          d={candles
            .map((c, i) => {
              const cx = padX + i * cw + cw / 2;
              const vy = y((c.h + c.l + c.c) / 3);
              return `${i === 0 ? "M" : "L"} ${cx} ${vy}`;
            })
            .join(" ")}
          fill="none"
          stroke="url(#vwap)"
          strokeWidth="1.5"
          strokeDasharray="4 4"
        />

        {candles.map((c, i) => {
          const cx = padX + i * cw + cw / 2;
          const up = c.c >= c.o;
          const color = up ? "var(--bull)" : "var(--bear)";
          return (
            <g key={i}>
              <line x1={cx} x2={cx} y1={y(c.h)} y2={y(c.l)} stroke={color} strokeWidth="1" opacity="0.7" />
              <rect
                x={cx - cw * 0.35}
                y={y(Math.max(c.o, c.c))}
                width={cw * 0.7}
                height={Math.max(1.5, Math.abs(y(c.o) - y(c.c)))}
                fill={color}
                opacity={i === candles.length - 1 ? 1 : 0.85}
              />
            </g>
          );
        })}

        {/* last price marker */}
        <line x1={padX} x2={w - padX} y1={y(last.c)} y2={y(last.c)} stroke="var(--primary)" strokeWidth="0.5" strokeDasharray="2 4" opacity="0.6" />
      </svg>

      <div className="mt-2 grid grid-cols-4 gap-3 border-t border-border/60 pt-3 font-mono text-[10px]">
        {[
          { k: "RSI", v: "67.4", t: "bull" },
          { k: "MACD", v: "+1.82", t: "bull" },
          { k: "VWAP Δ", v: "+0.41%", t: "bull" },
          { k: "γ-exp", v: "$4.2B", t: "warn" },
        ].map((m) => (
          <div key={m.k}>
            <div className="text-muted-foreground">{m.k}</div>
            <div className={`mt-0.5 tabular-nums ${m.t === "bull" ? "text-bull" : m.t === "warn" ? "text-warn" : "text-bear"}`}>
              {m.v}
            </div>
          </div>
        ))}
      </div>

      {/* scanning line */}
      <div className="pointer-events-none absolute inset-x-5 top-16 h-px animate-scan bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
    </div>
  );
}