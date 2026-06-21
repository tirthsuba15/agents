import { Activity } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { ConnectionBadge } from "./live";

export function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3.5">
        <a href="#" className="flex items-center gap-2.5 font-mono text-sm font-semibold tracking-tight">
          <span className="relative grid h-7 w-7 place-items-center rounded-md bg-primary/15 text-primary">
            <Activity className="h-3.5 w-3.5" />
            <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-primary animate-blink" />
          </span>
          <span>VA<span className="text-primary">ULT</span></span>
          <span className="ml-2 hidden sm:inline">
            <ConnectionBadge />
          </span>
        </a>
        <nav className="hidden gap-7 font-mono text-xs text-muted-foreground md:flex">
          <a href="#architecture" className="hover:text-foreground transition-colors">architecture</a>
          <a href="#agents" className="hover:text-foreground transition-colors">agents</a>
          <a href="#evolution" className="hover:text-foreground transition-colors">evolution</a>
          <a href="#execution" className="hover:text-foreground transition-colors">execution</a>
          <a href="#memory" className="hover:text-foreground transition-colors">memory</a>
        </nav>
        <div className="flex items-center gap-2">
          <a href="#" className="hidden font-mono text-xs text-muted-foreground hover:text-foreground sm:inline">docs ↗</a>
          <Link to="/dashboard" className="rounded-md bg-primary px-3.5 py-2 font-mono text-xs font-semibold text-primary-foreground transition-transform hover:scale-[1.02]">
            Launch app →
          </Link>
        </div>
      </div>
    </header>
  );
}