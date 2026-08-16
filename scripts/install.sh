#!/usr/bin/env bash
# Cortex Deployer installer — same shape as https://shizuha.com/install.sh
#
#   curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
#   cortex-deployer server
#
# Never uses the OS package pip (PEP 668 / externally-managed-environment).
# Bootstraps an isolated uv + CPython under ~/.cortex-deployer and a
# wrapper on ~/.local/bin. No sudo. No apt. Never touches the OS Python.
#
# Optional:
#   CORTEX_DEPLOYER_HOME     install prefix (default: ~/.cortex-deployer)
#   CORTEX_DEPLOYER_BIN_DIR  wrapper dir (default: ~/.local/bin)
#   CORTEX_DEPLOYER_TARBALL  package URL (default: GitHub main archive)
#   CORTEX_DEPLOYER_START=1  start the UI after install
#   bash install.sh --server
set -euo pipefail

PREFIX="${CORTEX_DEPLOYER_HOME:-${HOME}/.cortex-deployer}"
BIN_DIR="${CORTEX_DEPLOYER_BIN_DIR:-${HOME}/.local/bin}"
TARBALL="${CORTEX_DEPLOYER_TARBALL:-https://github.com/shizuha-labs/cortex-deployer/archive/refs/heads/main.tar.gz}"
UV_VERSION="${CORTEX_DEPLOYER_UV_VERSION:-0.12.5}"
PY_VERSION="${CORTEX_DEPLOYER_PYTHON:-3.12}"

die() { echo "cortex-deployer-install: $*" >&2; exit 1; }
info() { echo "cortex-deployer-install: $*"; }

have() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1" dest="$2"
  info "fetch $url"
  if have curl; then
    curl -fL --retry 3 --retry-delay 1 --progress-bar -o "$dest" "$url"
  elif have wget; then
    wget -q --show-progress -O "$dest" "$url"
  else
    die "need curl or wget"
  fi
}

uv_asset() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    linux) os="unknown-linux-gnu" ;;
    darwin) os="apple-darwin" ;;
    mingw*|msys*|cygwin*)
      die "on Windows PowerShell run: irm https://cortex.shizuha.com/deployer/install.ps1 | iex"
      ;;
    *) die "unsupported OS: $os" ;;
  esac
  case "$arch" in
    x86_64|amd64) arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *) die "unsupported architecture: $arch" ;;
  esac
  echo "uv-${arch}-${os}.tar.gz"
}

ensure_uv() {
  if [ -x "${PREFIX}/bin/uv" ]; then
    echo "${PREFIX}/bin/uv"
    return
  fi
  if have uv; then
    command -v uv
    return
  fi
  mkdir -p "${PREFIX}/bin" "${PREFIX}/tmp"
  local asset tarball
  asset="$(uv_asset)"
  tarball="${PREFIX}/tmp/${asset}"
  download \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}" \
    "$tarball"
  tar -xzf "$tarball" -C "${PREFIX}/tmp"
  local found
  found="$(find "${PREFIX}/tmp" -type f -name uv | head -n 1)"
  [ -n "$found" ] || die "uv binary missing from ${asset}"
  cp "$found" "${PREFIX}/bin/uv"
  chmod +x "${PREFIX}/bin/uv"
  echo "${PREFIX}/bin/uv"
}

write_wrapper() {
  mkdir -p "$BIN_DIR"
  cat > "${BIN_DIR}/cortex-deployer" <<EOF
#!/usr/bin/env bash
exec "${PREFIX}/venv/bin/cortex-deployer" "\$@"
EOF
  chmod +x "${BIN_DIR}/cortex-deployer"
}

note_path() {
  case ":${PATH}:" in
    *":${BIN_DIR}:"*) return 0 ;;
  esac
  if [ -f "${HOME}/.profile" ] && ! grep -q 'CORTEX_DEPLOYER_BIN' "${HOME}/.profile" 2>/dev/null; then
    printf '\n# CORTEX_DEPLOYER_BIN\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "${HOME}/.profile"
    info "appended ${BIN_DIR} to ~/.profile (new login shells)"
  fi
  return 1
}

main() {
  local start=0
  for arg in "$@"; do
    case "$arg" in
      --server|--start) start=1 ;;
      -h|--help)
        cat <<'HELP'
Usage: curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
       bash install.sh [--server]
HELP
        exit 0
        ;;
    esac
  done
  if [ "${CORTEX_DEPLOYER_START:-}" = "1" ]; then
    start=1
  fi

  info "install prefix ${PREFIX}"
  mkdir -p "$PREFIX" "${PREFIX}/tmp"
  local uv
  uv="$(ensure_uv)"
  info "using uv $($uv --version 2>/dev/null || echo "$uv")"

  export UV_PYTHON_INSTALL_DIR="${PREFIX}/python"
  export UV_CACHE_DIR="${PREFIX}/cache"
  export UV_TOOL_DIR="${PREFIX}/tools"
  export UV_PYTHON_PREFERENCE=only-managed
  mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_TOOL_DIR"

  info "standalone CPython ${PY_VERSION} (not the OS interpreter)"
  "$uv" python install "$PY_VERSION"

  info "venv ${PREFIX}/venv"
  "$uv" venv "${PREFIX}/venv" --python "$PY_VERSION" --clear

  info "install cortex-deployer into the venv"
  "$uv" pip install --python "${PREFIX}/venv/bin/python" "$TARBALL"

  [ -x "${PREFIX}/venv/bin/cortex-deployer" ] \
    || die "venv is missing cortex-deployer after install"

  write_wrapper
  local path_ok=0
  if note_path; then
    path_ok=1
  fi

  info "ok  wrapper ${BIN_DIR}/cortex-deployer"
  "${BIN_DIR}/cortex-deployer" --version || true
  echo
  echo "Next:"
  if [ "$path_ok" -eq 0 ]; then
    echo "  export PATH=\"${BIN_DIR}:\$PATH\""
  fi
  echo "  cortex-deployer server"
  echo "  # http://127.0.0.1:7480  →  Choose a Qwen build"
  echo
  echo "NVIDIA driver is the only extra host package for GPU. The app"
  echo "fetches official llama.cpp builds and the GGUF itself."

  if [ "$start" -eq 1 ]; then
    exec "${BIN_DIR}/cortex-deployer" server
  fi
}

main "$@"
