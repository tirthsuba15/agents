import { createFileRoute } from "@tanstack/react-router";
import { TradingFramework } from "@/components/stratos/TradingFramework";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Vault — Dashboard" },
      { name: "description", content: "Live Vault trading dashboard: strategies, scanner, gamma, positions." },
    ],
  }),
  component: TradingFramework,
});