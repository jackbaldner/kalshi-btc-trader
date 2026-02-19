"""Rich-based CLI dashboard for live monitoring."""

import time
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class Dashboard:
    """Terminal dashboard showing live trading status."""

    def __init__(self):
        self.console = Console()
        self._data = {
            "mode": "paper",
            "balance": 0.0,
            "btc_price": 0.0,
            "positions": [],
            "open_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "regime": "medium",
            "current_edge": 0.0,
            "next_expiry": "",
            "last_signal": "",
            "recent_trades": [],
            "daily_pnl": 0.0,
            "exposure": 0.0,
        }

    def update(self, **kwargs):
        self._data.update(kwargs)

    def _build_status_table(self) -> Table:
        table = Table(title="Trading Status", expand=True)
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="green")

        d = self._data
        table.add_row("Mode", d["mode"].upper())
        table.add_row("Balance", f"${d['balance']:.2f}")
        table.add_row("BTC Price", f"${d['btc_price']:,.2f}")
        table.add_row("Daily PnL", f"${d['daily_pnl']:.2f}")
        table.add_row("Exposure", f"${d['exposure']:.2f}")
        table.add_row("Vol Regime", d["regime"])
        table.add_row("Current Edge", f"{d['current_edge']:.3f}")
        table.add_row("Next Expiry", d["next_expiry"])
        table.add_row("Last Signal", d["last_signal"])

        return table

    def _build_performance_table(self) -> Table:
        table = Table(title="Performance", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Trades", str(self._data["total_trades"]))
        table.add_row("Win Rate", f"{self._data['win_rate']:.1%}")
        table.add_row("Open PnL", f"${self._data['open_pnl']:.2f}")

        return table

    def _build_trades_table(self) -> Table:
        table = Table(title="Recent Trades", expand=True)
        table.add_column("Time", style="dim")
        table.add_column("Side", style="cyan")
        table.add_column("Price", style="white")
        table.add_column("Size", style="white")
        table.add_column("Edge", style="green")
        table.add_column("PnL", style="green")

        for t in self._data["recent_trades"][-10:]:
            pnl_str = f"${t.get('pnl', 0):.2f}" if t.get("pnl") is not None else "pending"
            pnl_style = "green" if t.get("pnl", 0) >= 0 else "red"
            table.add_row(
                t.get("time", ""),
                t.get("side", ""),
                f"{t.get('price', 0)}c",
                str(t.get("size", 0)),
                f"{t.get('edge', 0):.3f}",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
            )

        return table

    def _build_positions_table(self) -> Table:
        table = Table(title="Open Positions", expand=True)
        table.add_column("Ticker", style="cyan")
        table.add_column("Yes", style="green")
        table.add_column("No", style="red")

        for p in self._data["positions"]:
            table.add_row(
                p.get("ticker", ""),
                str(p.get("yes_count", 0)),
                str(p.get("no_count", 0)),
            )

        if not self._data["positions"]:
            table.add_row("—", "0", "0")

        return table

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12),
        )

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        layout["header"].update(
            Panel(f"Kalshi BTC 15m Trader | {now}", style="bold white on blue")
        )

        layout["body"].split_row(
            Layout(self._build_status_table(), name="status"),
            Layout(self._build_performance_table(), name="perf"),
            Layout(self._build_positions_table(), name="positions"),
        )

        layout["footer"].update(self._build_trades_table())

        return layout

    def run_live(self, refresh_callback, interval: float = 5.0):
        """Run live dashboard with periodic refresh.

        Args:
            refresh_callback: Callable that updates dashboard data
            interval: Refresh interval in seconds
        """
        with Live(self.render(), console=self.console, refresh_per_second=1) as live:
            while True:
                try:
                    refresh_callback(self)
                    live.update(self.render())
                    time.sleep(interval)
                except KeyboardInterrupt:
                    break
