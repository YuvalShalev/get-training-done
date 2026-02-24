"""CLI for bbopt setup and configuration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import click


AGENT_SOURCE = Path(__file__).parent / "agents" / "ml-optimizer.md"
CLAUDE_AGENTS_DIR = Path.home() / ".claude" / "agents"
CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

MCP_SERVERS_CONFIG = {
    "bbopt-data": {
        "command": sys.executable,
        "args": ["-m", "bbopt.servers.data_server"],
        "env": {},
    },
    "bbopt-training": {
        "command": sys.executable,
        "args": ["-m", "bbopt.servers.training_server"],
        "env": {},
    },
    "bbopt-research": {
        "command": sys.executable,
        "args": ["-m", "bbopt.servers.research_server"],
        "env": {},
    },
}


@click.group()
def main() -> None:
    """Agent-driven black-box optimization for Claude Code."""


@main.command()
def setup() -> None:
    """Configure Claude Code with bbopt MCP servers and agent."""
    click.echo("Setting up bbopt for Claude Code...\n")

    _install_agent()
    _configure_mcp_servers()
    _check_dependencies()

    click.echo("\nSetup complete! You can now use the ML optimizer:")
    click.echo("  1. Open Claude Code: claude")
    click.echo("  2. Invoke the agent: /ml-optimizer")
    click.echo('  3. Or just ask: "Optimize my dataset at ./data.csv"')


@main.command()
def check() -> None:
    """Verify bbopt installation and dependencies."""
    click.echo("Checking bbopt installation...\n")

    all_ok = True

    # Check agent file
    agent_dest = CLAUDE_AGENTS_DIR / "ml-optimizer.md"
    if agent_dest.exists():
        click.echo("  [OK] Agent definition installed")
    else:
        click.echo("  [MISSING] Agent definition not found. Run: bbopt setup")
        all_ok = False

    # Check MCP servers in settings
    if CLAUDE_SETTINGS_PATH.exists():
        with open(CLAUDE_SETTINGS_PATH) as f:
            settings = json.load(f)
        mcp_servers = settings.get("mcpServers", {})
        for server_name in MCP_SERVERS_CONFIG:
            if server_name in mcp_servers:
                click.echo(f"  [OK] MCP server '{server_name}' configured")
            else:
                click.echo(f"  [MISSING] MCP server '{server_name}' not configured")
                all_ok = False
    else:
        click.echo("  [MISSING] Claude Code settings not found")
        all_ok = False

    # Check Python dependencies
    _check_dependencies()

    if all_ok:
        click.echo("\nAll checks passed!")
    else:
        click.echo("\nSome checks failed. Run: bbopt setup")


@main.command()
def uninstall() -> None:
    """Remove bbopt configuration from Claude Code."""
    click.echo("Removing bbopt configuration...\n")

    # Remove agent
    agent_dest = CLAUDE_AGENTS_DIR / "ml-optimizer.md"
    if agent_dest.exists():
        agent_dest.unlink()
        click.echo("  Removed agent definition")

    # Remove MCP servers from settings
    if CLAUDE_SETTINGS_PATH.exists():
        with open(CLAUDE_SETTINGS_PATH) as f:
            settings = json.load(f)

        mcp_servers = settings.get("mcpServers", {})
        updated = {k: v for k, v in mcp_servers.items() if k not in MCP_SERVERS_CONFIG}
        settings_updated = {**settings, "mcpServers": updated}

        with open(CLAUDE_SETTINGS_PATH, "w") as f:
            json.dump(settings_updated, f, indent=2)
        click.echo("  Removed MCP server configurations")

    click.echo("\nUninstall complete.")


def _install_agent() -> None:
    """Copy the agent definition to Claude Code agents directories."""
    if not AGENT_SOURCE.exists():
        click.echo(f"  [WARN] Agent source not found at {AGENT_SOURCE}")
        click.echo("         The agent file may need to be created manually.")
        return

    # Install to global ~/.claude/agents/
    CLAUDE_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    global_dest = CLAUDE_AGENTS_DIR / "ml-optimizer.md"
    shutil.copy2(str(AGENT_SOURCE), str(global_dest))
    click.echo(f"  [OK] Agent installed (global): {global_dest}")

    # Install to project-level .claude/agents/ (current working directory)
    project_agents_dir = Path.cwd() / ".claude" / "agents"
    project_agents_dir.mkdir(parents=True, exist_ok=True)
    project_dest = project_agents_dir / "ml-optimizer.md"
    shutil.copy2(str(AGENT_SOURCE), str(project_dest))
    click.echo(f"  [OK] Agent installed (project): {project_dest}")


def _configure_mcp_servers() -> None:
    """Add MCP server configurations to Claude Code settings."""
    CLAUDE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if CLAUDE_SETTINGS_PATH.exists():
        with open(CLAUDE_SETTINGS_PATH) as f:
            settings = json.load(f)

    mcp_servers = settings.get("mcpServers", {})
    updated_servers = {**mcp_servers, **MCP_SERVERS_CONFIG}
    updated_settings = {**settings, "mcpServers": updated_servers}

    with open(CLAUDE_SETTINGS_PATH, "w") as f:
        json.dump(updated_settings, f, indent=2)

    for name in MCP_SERVERS_CONFIG:
        click.echo(f"  [OK] MCP server '{name}' configured")


def _check_dependencies() -> None:
    """Verify required Python packages are installed."""
    required = [
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
        ("shap", "shap"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
    ]

    missing = []
    for display_name, import_name in required:
        try:
            __import__(import_name)
            click.echo(f"  [OK] {display_name}")
        except ImportError:
            click.echo(f"  [MISSING] {display_name}")
            missing.append(display_name)

    if missing:
        click.echo(f"\n  Install missing packages: pip install {' '.join(missing)}")
