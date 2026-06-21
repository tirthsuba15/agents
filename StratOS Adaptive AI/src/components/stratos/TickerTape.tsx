import { useQuotes } from "@/hooks/use-stratos";

// Fallback row shown during SSR / before live quotes load / if Finnhub is down.
const FALLBACK = [
  { sym: "SPY", px: 624.18, chg: 0.42 },
  { sym: "QQQ", px: 548.91, chg: 0.88 },
  { sym: "NVDA", px: 184.27, chg: 2.14 },
  { sym: "AAPL", px: 261.55, chg: -0.31 },
  { sym: "MSFT", px: 512.04, chg: 0.67 },
  { sym: "TSLA", px: 421.89, chg: -1.42 },
  { sym: "META", px: 718.36, chg: 1.05 },
  { sym: "AMZN", px: 248.71, chg: 0.28 },
  { sym: "GOOGL", px: 218.44, chg: -0.18 },
  { sym: "AMD", px: 198.62, chg: 3.21 },
];

export function TickerTape() {
  const { data } = useQuotes("SPY,QQQ,NVDA,AAPL,MSFT,TSLA,META,AMZN,GOOGL,AMD,AVGO,JPM");

  const tickers =
    data?.live && data.quotes.length
      ? data.quotes.map((q) => ({ sym: q.symbol, px: q.price, chg: q.change_pct }))
      : FALLBACK;

  const row = [...tickers, ...tickers];
  return (
    <div className="relative overflow-hidden border-b border-border/60 bg-card/40 py-2.5">
      <div className="flex w-max animate-ticker gap-10 font-mono text-xs">
        {row.map((t, i) => (
          <span key={i} className="flex items-center gap-2 whitespace-nowrap">
            <span className="font-semibold tracking-wide text-foreground">{t.sym}</span>
            <span className="text-muted-foreground">{t.px.toFixed(2)}</span>
            <span className={t.chg >= 0 ? "text-bull" : "text-bear"}>
              {t.chg >= 0 ? "▲" : "▼"} {Math.abs(t.chg).toFixed(2)}%
            </span>
          </span>
        ))}
      </div>
      <div className="pointer-events-none absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-background to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-24 bg-gradient-to-l from-background to-transparent" />
    </div>
  );
}
