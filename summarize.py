#!/usr/bin/env python3
"""
Summarize tournament_out/tournament.json as a Rich CLI table.

Usage:
  python summarize.py --file tournament_out/tournament.json
  python summarize.py                      # uses default path
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"tournament summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_table(summary: Dict[str, Any]) -> Table:
    table = Table(title="Tournament Summary", show_lines=False)
    table.add_column("Agent", style="bold")
    table.add_column("mu", justify="right")
    table.add_column("sigma", justify="right")
    table.add_column("conservative", justify="right", style="cyan")
    table.add_column("Games", justify="right")
    table.add_column("W", justify="right", style="green")
    table.add_column("L", justify="right", style="red")
    table.add_column("D", justify="right", style="yellow")
    table.add_column("Win%", justify="right")
    table.add_column("Acc%", justify="right")
    table.add_column("ACPL", justify="right")

    agents = summary.get("agents", {})

    # Prepare rows sorted by conservative rating descending
    def conservative_of(agent_data: Dict[str, Any]) -> float:
        rating = agent_data.get("rating", {})
        return float(rating.get("conservative", 0.0))

    sorted_items = sorted(agents.items(), key=lambda kv: conservative_of(kv[1]), reverse=True)

    for agent_name, agent_data in sorted_items:
        rating = agent_data.get("rating", {})
        totals = agent_data.get("totals", {})
        engine = agent_data.get("engine_metrics_avg", {})

        mu = float(rating.get("mu", 0.0))
        sigma = float(rating.get("sigma", 0.0))
        conservative = float(rating.get("conservative", 0.0))
        games = int(totals.get("games", 0))
        wins = int(totals.get("wins", 0))
        losses = int(totals.get("losses", 0))
        draws = int(totals.get("draws", 0))
        win_pct = (wins + 0.5 * draws) / games * 100 if games else 0.0
        acc_pct = float(engine.get("accuracy_pct", 0.0))
        acpl = float(engine.get("acpl", 0.0))

        table.add_row(
            agent_name,
            f"{mu:.3f}",
            f"{sigma:.3f}",
            f"{conservative:.3f}",
            f"{games}",
            f"{wins}",
            f"{losses}",
            f"{draws}",
            f"{win_pct:.1f}",
            f"{acc_pct:.1f}",
            f"{acpl:.1f}",
        )

    return table


@click.command()
@click.option(
    "--file",
    "summary_file",
    default="tournament_out/tournament.json",
    help="Path to tournament JSON summary",
)
def main(summary_file: str):
    console = Console()
    path = Path(summary_file)

    try:
        summary = _load_summary(path)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", highlight=False)
        sys.exit(1)

    cfg = summary.get("tournament_config", {})
    header = Table(show_header=False, box=None)
    header.add_column("k", style="cyan")
    header.add_column("v", style="yellow")
    header.add_row("Games Played", str(cfg.get("games_played", cfg.get("num_games", "-"))))
    header.add_row("Agents", ", ".join(summary.get("tournament_config", {}).get("agents", [])))
    header.add_row("Scheduler", cfg.get("scheduler", "-"))
    header.add_row("Parallelism", str(cfg.get("parallelism", "-")))

    table = _build_table(summary)

    console.print(Panel(header, title="Config"))
    console.print(table)


if __name__ == "__main__":
    main()


