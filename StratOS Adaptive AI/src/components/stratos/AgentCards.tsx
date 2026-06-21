import { Newspaper, TrendingUp, Sigma } from "lucide-react";
import { useEffect, useState } from "react";
import { useNews, useWeights } from "@/hooks/use-stratos";
import { fmt } from "./live";

const FALLBACK_HEADLINES = [
  { src: "Reuters", txt: "Nvidia unveils next-gen Rubin GPU; analysts raise targets" },
  { src: "Bloomberg", txt: "Fed minutes hint at slower QT pace into Q1" },
  { src: "WSJ", txt: "Apple supplier orders soft, channel checks miss" },
  { src: "FT", txt: "Tesla deliveries beat in China; ASP holds" },
];

export function AgentCards() {
  return (
    <section id="agents" className="py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary">/ 02 · signal agents</div>
            <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              Three specialists. One conviction score.
            </h2>
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            Each agent emits a direction, magnitude, and confidence. The meta-agent owns disagreement resolution.
          </p>
        </div>

        <div className="mt-12 grid gap-5 lg:grid-cols-3">
          <SentimentCard />
          <MomentumCard />
          <GammaCard />
        </div>
      </div>
    </section>
  );
}

function Frame({
  tone,
  badge,
  icon,
  title,
  desc,
  children,
}: {
  tone: "bull" | "warn" | "primary";
  badge: string;
  icon: React.ReactNode;
  title: string;
  desc: string;
  children: React.ReactNode;
}) {
  const glow = tone === "bull" ? "glow-bull" : tone === "warn" ? "glow-warn" : "glow-bull";
  const ring = tone === "bull" ? "text-bull" : tone === "warn" ? "text-warn" : "text-primary";
  return (
    <div className={`group relative flex flex-col rounded-xl border border-border bg-card/70 p-6 backdrop-blur transition-all hover:${glow}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className={`grid h-9 w-9 place-items-center rounded-md bg-background/80 ${ring}`}>{icon}</span>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">agent</div>
            <div className="font-display text-lg font-semibold">{title}</div>
          </div>
        </div>
        <span className={`rounded-full border border-current/30 px-2 py-0.5 font-mono text-[10px] ${ring}`}>{badge}</span>
      </div>
      <p className="mt-3 text-sm text-muted-foreground">{desc}</p>
      <div className="mt-5 flex-1">{children}</div>
    </div>
  );
}

function WeightFooter({ label, weight, tone }: { label: string; weight: number | undefined; tone: "bull" | "warn" }) {
  const c = tone === "bull" ? "text-bull" : "text-warn";
  return (
    <div className="mt-4 flex items-center justify-between border-t border-border/60 pt-3 font-mono text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      <span className={c}>weight · {weight != null ? fmt.pct(weight, 1) : "—"}</span>
    </div>
  );
}

function SentimentCard() {
  const { data: news } = useNews("NVDA");
  const { data: weights } = useWeights();
  const [i, setI] = useState(0);

  const headlines = news?.live && news.news.length
    ? news.news.slice(0, 4).map((n) => ({ src: n.source, txt: n.headline }))
    : FALLBACK_HEADLINES;

  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % Math.max(1, headlines.length)), 2600);
    return () => clearInterval(t);
  }, [headlines.length]);

  return (
    <Frame
      tone="bull"
      badge={news?.live ? "LIVE · Finnhub" : "demo"}
      icon={<Newspaper className="h-4 w-4" />}
      title="Sentiment"
      desc="Streams real-time Finnhub headlines for NVDA, scored for direction & confidence by Llama-3.3-70B on Nebius."
    >
      <div className="space-y-2">
        {headlines.slice(0, 4).map((h, idx) => {
          const active = idx === i % headlines.length;
          return (
            <div
              key={idx}
              className={`flex items-center gap-3 rounded-md border px-3 py-2 transition-all ${
                active ? "border-bull/40 bg-bull/5" : "border-border bg-background/30"
              }`}
            >
              <span className="font-mono text-[10px] text-muted-foreground w-16 shrink-0 truncate">{h.src}</span>
              <span className="flex-1 truncate text-xs text-foreground">{h.txt}</span>
            </div>
          );
        })}
      </div>
      <WeightFooter label={`NVDA · ${news?.news?.length ?? 0} headlines`} weight={weights?.w_sentiment} tone="bull" />
    </Frame>
  );
}

function MomentumCard() {
  const { data: weights } = useWeights();
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((x) => x + 1), 700);
    return () => clearInterval(id);
  }, []);
  const pts = Array.from({ length: 60 }, (_, i) => {
    const x = i;
    const y = 50 + Math.sin((i + t) * 0.25) * 14 + Math.sin((i + t) * 0.07) * 8;
    return `${x * 4},${y}`;
  }).join(" ");
  return (
    <Frame
      tone="bull"
      badge="OHLCV · 1m"
      icon={<TrendingUp className="h-4 w-4" />}
      title="Momentum"
      desc="Ingests raw candles and computes RSI, MACD, and VWAP to identify trend strength and entry timing."
    >
      <div className="rounded-md border border-border bg-background/50 p-3">
        <svg viewBox="0 0 240 100" className="h-24 w-full">
          <defs>
            <linearGradient id="momentumFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="var(--bull)" stopOpacity="0.5" />
              <stop offset="1" stopColor="var(--bull)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <polyline points={pts} fill="none" stroke="var(--bull)" strokeWidth="1.6" />
          <polygon points={`0,100 ${pts} 240,100`} fill="url(#momentumFill)" />
        </svg>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px]">
        <Metric k="RSI" v="67.4" tone="bull" />
        <Metric k="MACD" v="+1.82" tone="bull" />
        <Metric k="VWAP" v="above" tone="bull" />
      </div>
      <WeightFooter label="trend model · XGBoost" weight={weights?.w_momentum} tone="bull" />
    </Frame>
  );
}

function GammaCard() {
  const { data: weights } = useWeights();
  return (
    <Frame
      tone="warn"
      badge="OPRA · dark pool"
      icon={<Sigma className="h-4 w-4" />}
      title="Gamma"
      desc="Maps dealer gamma exposure and dark pool prints to surface pinning zones, γ-flips, and unusual flow."
    >
      <div className="rounded-md border border-border bg-background/50 p-3">
        <svg viewBox="0 0 240 100" className="h-24 w-full">
          {/* GEX histogram */}
          {Array.from({ length: 22 }).map((_, i) => {
            const h = Math.sin(i * 0.55) * 28 + Math.cos(i * 0.3) * 14;
            const isPos = h >= 0;
            const bh = Math.abs(h);
            return (
              <rect
                key={i}
                x={i * 11 + 2}
                y={isPos ? 50 - bh : 50}
                width="8"
                height={bh}
                fill={isPos ? "var(--bull)" : "var(--bear)"}
                opacity={i === 11 ? 1 : 0.55}
              />
            );
          })}
          {/* spot marker */}
          <line x1="125" x2="125" y1="6" y2="94" stroke="var(--warn)" strokeWidth="1" strokeDasharray="3 3" />
          <text x="128" y="14" className="fill-warn font-mono" fontSize="9">
            spot 184.27
          </text>
        </svg>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[11px]">
        <Metric k="GEX" v="+$4.2B" tone="bull" />
        <Metric k="γ-flip" v="181.50" tone="warn" />
        <Metric k="DPI" v="62%" tone="warn" />
      </div>
      <WeightFooter label="gamma model · GEX" weight={weights?.w_gamma} tone="warn" />
    </Frame>
  );
}

function Metric({ k, v, tone }: { k: string; v: string; tone: "bull" | "warn" | "bear" }) {
  const c = tone === "bull" ? "text-bull" : tone === "warn" ? "text-warn" : "text-bear";
  return (
    <div className="rounded border border-border bg-background/40 px-2 py-1.5">
      <div className="text-[10px] text-muted-foreground">{k}</div>
      <div className={`tabular-nums ${c}`}>{v}</div>
    </div>
  );
}