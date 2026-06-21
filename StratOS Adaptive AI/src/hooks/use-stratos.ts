/**
 * React Query hooks over the StratOS API client.
 *
 * Two deliberate constraints (see api.py design contract + SSR):
 *  • Queries are CLIENT-ONLY (`enabled` flips true after mount). TanStack Start
 *    renders server-side; fetching localhost:8000 during SSR would fail and also
 *    cause a hydration mismatch. We render fallback first, then hydrate live.
 *  • staleTime is long because HydraDB-backed endpoints are slow and TTL-cached
 *    server-side already. Only quotes poll frequently.
 */
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/stratos-api";

function useMounted(): boolean {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted;
}

export function useHealth() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "health"],
    queryFn: api.health,
    enabled,
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useWeights() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "weights"],
    queryFn: api.weights,
    enabled,
    staleTime: 45_000,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useAccuracy() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "accuracy"],
    queryFn: api.accuracy,
    enabled,
    staleTime: 60_000,
    retry: 1,
  });
}

export function useTrades(n = 25) {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "trades", n],
    queryFn: () => api.trades(n),
    enabled,
    staleTime: 30_000,
    refetchInterval: 45_000,
    retry: 1,
  });
}

export function useStats() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "stats"],
    queryFn: api.stats,
    enabled,
    staleTime: 45_000,
    refetchInterval: 60_000,
    retry: 1,
  });
}

export function useQuotes(symbols?: string) {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "quotes", symbols ?? "default"],
    queryFn: () => api.quotes(symbols),
    enabled,
    staleTime: 15_000,
    refetchInterval: 20_000,
    retry: 1,
  });
}

export function useNews(ticker = "NVDA") {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "news", ticker],
    queryFn: () => api.news(ticker),
    enabled,
    staleTime: 120_000,
    refetchInterval: 180_000,
    retry: 1,
  });
}

export function usePortfolio() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "portfolio"],
    queryFn: api.portfolio,
    enabled,
    staleTime: 30_000,
    refetchInterval: 45_000,
    retry: 1,
  });
}

export function useEquityCurve(period = "1M", timeframe = "1D") {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "equity_curve", period, timeframe],
    queryFn: () => api.equityCurve(period, timeframe),
    enabled,
    staleTime: 60_000,
    refetchInterval: 90_000,
    retry: 1,
  });
}

export function useBacktest() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "backtest"],
    queryFn: api.backtest,
    enabled,
    staleTime: 300_000,
    retry: 1,
  });
}

export function useMemory() {
  const enabled = useMounted();
  return useQuery({
    queryKey: ["stratos", "memory"],
    queryFn: api.memory,
    enabled,
    staleTime: 60_000,
    refetchInterval: 90_000,
    retry: 1,
  });
}
