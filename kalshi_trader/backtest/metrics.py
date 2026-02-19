"""Backtest performance metrics."""

import numpy as np
import pandas as pd


def calculate_metrics(trades: list[dict]) -> dict:
    """Calculate performance metrics from a list of trade dicts.

    Each trade dict should have: pnl, edge, model_prob, implied_prob, strategy.
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
            "profit_factor": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    edges = [t.get("edge", 0) for t in trades]

    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Sharpe ratio (annualized, assuming 4 trades per hour, 24h)
    pnl_array = np.array(pnls)
    if pnl_array.std() > 0:
        trades_per_year = 4 * 24 * 365
        sharpe = (pnl_array.mean() / pnl_array.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Per-strategy breakdown
    strategy_pnl: dict[str, list[float]] = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        strategy_pnl.setdefault(s, []).append(t["pnl"])

    strategy_metrics = {}
    for s, spnls in strategy_pnl.items():
        sw = [p for p in spnls if p > 0]
        strategy_metrics[s] = {
            "trades": len(spnls),
            "win_rate": len(sw) / len(spnls) if spnls else 0.0,
            "total_pnl": sum(spnls),
            "avg_pnl": np.mean(spnls),
        }

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl": total_pnl,
        "avg_pnl": np.mean(pnls),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_edge": np.mean(edges),
        "profit_factor": profit_factor,
        "strategy_breakdown": strategy_metrics,
    }


def print_metrics(metrics: dict):
    """Pretty-print backtest metrics."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(metrics["total_trades"]))
    table.add_row("Win Rate", f"{metrics['win_rate']:.1%}")
    table.add_row("Total PnL", f"${metrics['total_pnl']:.2f}")
    table.add_row("Avg PnL/Trade", f"${metrics['avg_pnl']:.4f}")
    table.add_row("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    table.add_row("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
    table.add_row("Avg Edge", f"{metrics['avg_edge']:.3f}")
    table.add_row("Profit Factor", f"{metrics['profit_factor']:.2f}")

    console.print(table)

    # Strategy breakdown
    breakdown = metrics.get("strategy_breakdown", {})
    if breakdown:
        strat_table = Table(title="Strategy Breakdown")
        strat_table.add_column("Strategy", style="cyan")
        strat_table.add_column("Trades", style="white")
        strat_table.add_column("Win Rate", style="green")
        strat_table.add_column("Total PnL", style="green")

        for name, sm in breakdown.items():
            strat_table.add_row(
                name,
                str(sm["trades"]),
                f"{sm['win_rate']:.1%}",
                f"${sm['total_pnl']:.2f}",
            )
        console.print(strat_table)
