#!/usr/bin/env bash
# In-place upgrade. Does not wipe models or recreate the venv.
#
#   cortex-deployer update
#   curl -fsSL https://cortex.shizuha.com/deployer/update.sh | bash
set -euo pipefail

PREFIX="${CORTEX_DEPLOYER_HOME:-${HOME}/.cortex-deployer}"
BIN_DIR="${CORTEX_DEPLOYER_BIN_DIR:-${HOME}/.local/bin}"

if [ -x "${BIN_DIR}/cortex-deployer" ]; then
  exec "${BIN_DIR}/cortex-deployer" update "$@"
fi
if [ -x "${PREFIX}/venv/bin/cortex-deployer" ]; then
  exec "${PREFIX}/venv/bin/cortex-deployer" update "$@"
fi
if command -v cortex-deployer >/dev/null 2>&1; then
  exec cortex-deployer update "$@"
fi

echo "cortex-deployer-update: not installed yet. First time:" >&2
echo "  curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash" >&2
exit 1
