import { createFileRoute } from "@tanstack/react-router";
import { StratosLanding } from "@/components/stratos/StratosLanding";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Vault — Self-Evolving Algorithmic Trading Agent Suite" },
      {
        name: "description",
        content:
          "A multi-agent LangGraph architecture that fuses sentiment, momentum, and gamma signals — and rewrites its own weights from live trade outcomes.",
      },
      { property: "og:title", content: "Vault — Self-Evolving Trading Agents" },
      {
        property: "og:description",
        content:
          "Three specialized signal agents. One meta-agent. Continuously evolving strategy weights.",
      },
    ],
  }),
  component: StratosLanding,
});
