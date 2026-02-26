#!/usr/bin/env bash
# Bootstraps Python venv and starts GTD MCP server
# Usage: run-server.sh <module_name>
#   e.g.: run-server.sh gtd.servers.data_server

set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PLUGIN_ROOT}/.venv"
MODULE="$1"

# Bootstrap venv on first run
if [ ! -f "${VENV_DIR}/bin/python" ]; then
  echo "GTD: Setting up Python environment (first run only)..." >&2
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --quiet -e "${PLUGIN_ROOT}"
fi

# Start the MCP server
exec "${VENV_DIR}/bin/python" -m "${MODULE}"
