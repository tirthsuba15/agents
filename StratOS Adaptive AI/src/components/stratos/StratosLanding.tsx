import { Nav } from "./Nav";
import { Hero } from "./Hero";
import { TickerTape } from "./TickerTape";
import { AgentGraph } from "./AgentGraph";
import { AgentCards } from "./AgentCards";
import { EvolutionPanel } from "./EvolutionPanel";
import { ExecutionPanel } from "./ExecutionPanel";
import { MemorySchema } from "./MemorySchema";
import { Footer } from "./Footer";

export function StratosLanding() {
  return (
    <main className="min-h-screen bg-background text-foreground overflow-x-hidden">
      <Nav />
      <TickerTape />
      <Hero />
      <AgentGraph />
      <AgentCards />
      <EvolutionPanel />
      <ExecutionPanel />
      <MemorySchema />
      <Footer />
    </main>
  );
}