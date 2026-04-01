#!/usr/bin/env bash
# Installs deep learning foundation model dependencies into the plugin venv.
# PyTorch (required by TabICL) needs Python <=3.12, so this script rebuilds
# the plugin venv with Python 3.12 if needed.
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PLUGIN_ROOT}/.venv"

# Check current Python version in venv
NEEDS_REBUILD=false
if [ -f "${VENV_DIR}/bin/python" ]; then
  PY_VERSION=$("${VENV_DIR}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  case "$PY_VERSION" in
    3.10|3.11|3.12) ;;
    *) NEEDS_REBUILD=true ;;
  esac
else
  NEEDS_REBUILD=true
fi

if [ "$NEEDS_REBUILD" = true ]; then
  echo "GTD: PyTorch requires Python 3.12. Setting up compatible environment..." >&2

  # Ensure Python 3.12 is available via uv
  if ! uv python list --only-installed 2>/dev/null | grep -q "cpython-3.12"; then
    echo "GTD: Installing Python 3.12 via uv..." >&2
    uv python install 3.12
  fi

  PY312=$(uv python find 3.12)
  echo "GTD: Rebuilding plugin venv with Python 3.12 ($PY312)..." >&2
  rm -rf "${VENV_DIR}"
  "$PY312" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --quiet -e "${PLUGIN_ROOT}"
fi

echo "GTD: Installing TabICL..." >&2
"${VENV_DIR}/bin/pip" install --quiet tabicl
# Touch .mcp.json to trigger Claude Code to auto-reload MCP servers
touch "${PLUGIN_ROOT}/.mcp.json"
echo "GTD: Done. MCP servers will auto-restart. Wait a few seconds before training." >&2
