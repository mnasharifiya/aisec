"""
AISec serve CLI command — starts the REST API server.
"""

from __future__ import annotations

from pathlib import Path

import click

from aisec.storage.audit import DEFAULT_LOG_PATH


@click.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the server to. Use 0.0.0.0 for all interfaces.",
)
@click.option(
    "--port",
    type=click.IntRange(1024, 65535),
    default=8000,
    show_default=True,
    help="Port to listen on.",
)
@click.option(
    "--log-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOG_PATH,
    show_default=True,
    help="Path to the audit log file.",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development. Never use in production.",
)
def serve_command(
    host: str,
    port: int,
    log_path: Path,
    reload: bool,
) -> None:
    """
    Start the AISec REST API server.

    Exposes the analysis engine over HTTP so any language or
    platform can integrate with AISec without the Python SDK.

    \\b
    Examples:
        aisec serve
        aisec serve --host 0.0.0.0 --port 8000
        aisec serve --log-path /var/log/aisec/audit.jsonl

    API documentation available at:
        http://<host>:<port>/docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "uvicorn is required to run the API server.\n"
            "Install it with: pip install uvicorn",
            err=True,
        )
        raise SystemExit(1)

    from aisec.api.server import create_app

    if reload:
        click.echo(
            "  ⚠  Auto-reload enabled — development mode only.",
            err=True,
        )

    click.echo(f"\n  Starting AISec API server on http://{host}:{port}")
    click.echo(f"  API docs: http://{host}:{port}/docs")
    click.echo(f"  Audit log: {log_path}\n")

    app = create_app(log_path=log_path)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",  # AISec has its own structured logger
        access_log=False,  # We log requests ourselves
    )
