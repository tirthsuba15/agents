export function Footer() {
  return (
    <footer className="border-t border-border/60 bg-card/30">
      <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-6 px-6 py-10 sm:flex-row sm:items-center">
        <div className="font-mono text-xs text-muted-foreground">
          © 2026 VA<span className="text-primary">ULT</span> · not investment advice · trades with real risk
        </div>
        <div className="flex flex-wrap gap-6 font-mono text-xs text-muted-foreground">
          <a className="hover:text-foreground" href="#">docs</a>
          <a className="hover:text-foreground" href="#">whitepaper</a>
          <a className="hover:text-foreground" href="#">github ↗</a>
          <a className="hover:text-foreground" href="#">contact</a>
        </div>
      </div>
    </footer>
  );
}