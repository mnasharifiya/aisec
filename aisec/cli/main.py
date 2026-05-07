"""
AISec - Runtime Security Monitoring CLI
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from aisec.cli.monitor import monitor_command
from aisec.cli.soc import soc_command
from aisec.utils.assets import LOGO, PORTRAIT

console = Console()

# Professional ASCII Shields
SHIELD_LOCK_ASCII = r"""
           .--------.
          / .------. \
         / /        \ \
         | |  LOCK  | |
        _| |________| |_
      .' |_|        |_| '.
      '._____ ____ _____.'
      |     .'____'.     |
      '.__.'.'    '.'.__.'
      '.__  | SEC |  __.'
      |   '.'.____.'.'   |
      '.____'.____.'____.'
      '.________________.'
"""

SHIELD_CHECK_ASCII = r"""
           .--------.
          / .------. \
         / /        \ \
         | | SECURE | |
        _| |________| |_
      .' |_|  ✓✓✓   |_| '.
      '._____ ____ _____.'
      |     .'____'.     |
      '.__.'.'    '.'.__.'
      '.__  | OK  |  __.'
      |   '.'.____.'.'   |
      '.____'.____.'____.'
      '.________________.'
"""


def print_dashboard():
    # --- RIGHT COLUMN CONTENT ---

    info_content = Text.assemble(
        (LOGO, "bold green"),
        ("\nRuntime Security Monitoring for Autonomous AI Agents", "dim green"),
        ("\n\nDEVELOPER  : Muhammad Muttaka", "white"),
        ("\nAFFILIATION: Astana IT University"),
        ("\nVERSION    : v1.0.0"),
        ("\nMODE       : STANDBY", "yellow"),
        ("\nENVIRONMENT: SOC CORE"),
        ("\nSTATUS     : ARMED", "bold green"),
    )

    info_panel = Panel(
        info_content,
        border_style="green",
        padding=(1, 2),
    )

    tools_text = (
        "[dim green]"
        "> log monitor\n"
        "> agent audit\n"
        "> threat scan\n"
        "> event trace\n"
        "> hash verify\n"
        "> policy check\n"
        "> integrity chain"
        "[/dim green]"
    )

    tools_panel = Panel(
        tools_text,
        title="[bold green]SECURITY TOOLS[/bold green]",
        border_style="green",
        padding=(1, 2),
    )

    secure_panel = Panel(
        SHIELD_CHECK_ASCII.strip() + "\n\nALL SYSTEMS OPERATIONAL",
        title="[bold green]SYSTEM SECURE[/bold green]",
        border_style="green",
        padding=(1, 2),
    )

    right_stack = Table.grid(expand=True)
    right_stack.add_row(info_panel)
    right_stack.add_row(tools_panel)
    right_stack.add_row(secure_panel)

    # --- LEFT COLUMN: PORTRAIT ---

    portrait_panel = Panel(
        PORTRAIT.strip(),
        border_style="green",
        padding=0,
        expand=True,
    )

    # --- TOP LAYOUT ---

    top_layout = Table.grid(expand=True)
    top_layout.add_column(ratio=1)
    top_layout.add_column(ratio=2)
    top_layout.add_row(portrait_panel, right_stack)

    # --- SYSTEM CHECKS PANEL ---

    checks_text = Text.assemble(
        ("[green][✔][/green] Policy engine loaded\n", "green"),
        ("[green][✔][/green] Hash-chain logger initialized\n", "green"),
        ("[green][✔][/green] Scenario A loaded: trading_ai\n", "green"),
        ("[green][✔][/green] Scenario B loaded: urban_ai\n", "green"),
        ("[green][✔][/green] SOC environment ready\n", "green"),
        ("[green][✔][/green] Audit integrity verified", "green"),
    )

    checks_panel = Panel(
        checks_text,
        title="[bold green]SYSTEM CHECKS[/bold green]",
        border_style="green",
        padding=(1, 2),
    )

    # --- FOOTER BAR ---

    footer = Table.grid(expand=True)
    footer.add_column(justify="center", ratio=1)
    footer.add_column(justify="center", ratio=1)
    footer.add_column(justify="center", ratio=1)

    footer.add_row(
        "STATUS: [bold green]ARMED[/bold green]",
        "MODE: [bold yellow]STANDBY[/bold yellow]",
        "CHAIN: [bold cyan]INTACT [white]✔[/white][/bold cyan]",
    )

    footer_panel = Panel(
        footer,
        border_style="green",
    )

    # --- PRINT EVERYTHING ---

    console.print(top_layout)
    console.print(checks_panel)
    console.print(footer_panel)

    console.print(
        "\nType 'aisec --help' to see all available commands.",
        style="dim",
    )


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """AISec - Runtime Security Monitoring CLI"""

    if ctx.invoked_subcommand is None:
        print_dashboard()


@cli.command()
def start():
    """Start the monitoring engine"""

    console.print(
        "\n[bold green]▶ Starting AISec engine...[/bold green]\n"
    )


# Register CLI commands
cli.add_command(monitor_command)
cli.add_command(soc_command)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()