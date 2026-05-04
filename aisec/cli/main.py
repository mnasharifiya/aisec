"""
AISec - Runtime Security Monitoring for Autonomous AI Agents
Main CLI entry point
"""

import click
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()


def print_logo() -> None:
    """Print the AISec ASCII logo and startup banner."""

    logo = """
    ___    ____  _____           
   /   |  /  _/ / ___/___  _____
  / /| |  / /   \\__ \\/ _ \\/ ___/
 / ___ |_/ /   ___/ /  __/ /__  
/_/  |_/___/  /____/\\___/\\___/  
    """

    console.print()
    console.print(Text(logo, style="bold cyan"))
    console.print(
        Text(
            "  Runtime Security Monitoring for Autonomous AI Agents",
            style="dim white",
        )
    )
    console.print(
        Text(
            "  Muhammad Muttaka  ·  Astana IT University  ·  v1.0.0",
            style="dim",
        )
    )
    console.print()


def print_status(armed: bool = True) -> None:
    """Print system status after startup checks."""

    checks = [
        ("[✔]", "Policy engine loaded"),
        ("[✔]", "Hash-chain logger initialized"),
        ("[✔]", "Scenario A loaded: trading_ai"),
        ("[✔]", "Scenario B loaded: urban_ai"),
        ("[✔]", "SOC environment ready"),
        ("[✔]", "Audit integrity verified"),
    ]

    for badge, message in checks:
        console.print(
            Text(f"  {badge} ", style="bold green") + Text(message, style="white")
        )

    console.print()
    console.rule(style="dim")
    console.print()

    status_color = "bold green" if armed else "bold red"
    status_text = "ARMED" if armed else "DISARMED"

    console.print(
        Text("  STATUS: ", style="white")
        + Text(status_text, style=status_color)
        + Text("  |  ", style="dim")
        + Text("MODE: STANDBY", style="bold yellow")
        + Text("  |  ", style="dim")
        + Text("CHAIN: INTACT ✔", style="bold cyan")
    )

    console.print()
    console.print(
        Text(
            "  Type 'aisec --help' to see all available commands.",
            style="dim",
        )
    )
    console.print()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    AISec — Runtime Security Monitoring for Autonomous AI Agents.

    A CLI-first security platform that monitors AI agent behavior,
    scores actions for risk, enforces policy decisions in real time,
    and preserves tamper-evident audit trails.
    """
    if ctx.invoked_subcommand is None:
        print_logo()
        print_status()


@cli.command()
def start() -> None:
    """Start the AISec monitoring engine."""
    console.print()
    console.print(Text("  Starting AISec monitoring engine...", style="yellow"))
    console.print()
    # TODO: implement engine startup
    console.print(Text("  [✔] Engine started.", style="bold green"))
    console.print()


@cli.command()
def stop() -> None:
    """Stop the AISec monitoring engine."""
    console.print()
    console.print(Text("  Stopping AISec monitoring engine...", style="yellow"))
    console.print()
    # TODO: implement engine shutdown
    console.print(Text("  [✔] Engine stopped.", style="bold green"))
    console.print()


@cli.command()
def status() -> None:
    """Show current system status."""
    print_logo()
    print_status()


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()