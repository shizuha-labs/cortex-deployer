#!/usr/bin/env bash
# Cortex Deployer installer — same shape as https://shizuha.com/install.sh
#
#   curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
#   cortex-deployer server
#
# Host assumptions: curl (or wget) and tar. Nothing else.
# Does not use OS python, pip, venv, git, or a host `uv`.
# Logs go to stderr so command substitutions never swallow paths.
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
UV=""

die() { echo "cortex-deployer-install: $*" >&2; exit 1; }
info() { echo "cortex-deployer-install: $*" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

need_tar() {
  have tar || die "need tar (the uv/python bundles are .tar.gz). Install tar and re-run."
}

download() {
  local url="$1" dest="$2"
  info "fetch $url"
  mkdir -p "$(dirname "$dest")"
  if have curl; then
    curl -fL --retry 3 --retry-delay 1 --progress-bar -o "$dest" "$url"
  elif have wget; then
    wget -q --show-progress -O "$dest" "$url"
  else
    die "need curl or wget to download tools (no host Python/pip/git used)"
  fi
  [ -s "$dest" ] || die "empty download: $url"
}

uv_ok() {
  local bin="$1"
  [ -x "$bin" ] || return 1
  "$bin" --version >/dev/null 2>&1
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

install_own_uv() {
  need_tar
  mkdir -p "${PREFIX}/bin" "${PREFIX}/tmp"
  local asset tarball stem extracted
  asset="$(uv_asset)"
  stem="${asset%.tar.gz}"
  tarball="${PREFIX}/tmp/${asset}"
  download \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}" \
    "$tarball"
  rm -rf "${PREFIX}/tmp/${stem}"
  tar -xzf "$tarball" -C "${PREFIX}/tmp"
  extracted="${PREFIX}/tmp/${stem}/uv"
  [ -f "$extracted" ] || die "uv binary missing from ${asset}"
  cp "$extracted" "${PREFIX}/bin/uv"
  chmod 0755 "${PREFIX}/bin/uv"
  uv_ok "${PREFIX}/bin/uv" || die "bundled uv did not run after extract"
}

ensure_uv() {
  UV="${PREFIX}/bin/uv"
  if uv_ok "$UV"; then
    info "using bundled uv ($("$UV" --version))"
    return
  fi
  info "bootstrapping our own uv ${UV_VERSION} (ignoring any host uv/python)"
  install_own_uv
  UV="${PREFIX}/bin/uv"
  info "using bundled uv ($("$UV" --version))"
}

write_wrapper() {
  mkdir -p "$BIN_DIR"
  cat > "${BIN_DIR}/cortex-deployer" <<EOF
#!/usr/bin/env bash
exec "${PREFIX}/venv/bin/cortex-deployer" "\$@"
EOF
  chmod 0755 "${BIN_DIR}/cortex-deployer"
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
        cat <<'HELP' >&2
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

  have curl || have wget || die "need curl or wget"
  need_tar

  info "install prefix ${PREFIX}"
  mkdir -p "$PREFIX" "${PREFIX}/tmp" "${PREFIX}/bin"
  ensure_uv
  [ -n "$UV" ] && uv_ok "$UV" || die "internal: uv not ready"

  export UV_PYTHON_INSTALL_DIR="${PREFIX}/python"
  export UV_CACHE_DIR="${PREFIX}/cache"
  export UV_TOOL_DIR="${PREFIX}/tools"
  export UV_PYTHON_PREFERENCE=only-managed
  mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_TOOL_DIR"

  info "standalone CPython ${PY_VERSION} (not the OS interpreter)"
  "$UV" python install "$PY_VERSION"

  info "venv ${PREFIX}/venv"
  "$UV" venv "${PREFIX}/venv" --python "$PY_VERSION" --clear

  info "install cortex-deployer into the venv"
  "$UV" pip install --python "${PREFIX}/venv/bin/python" "$TARBALL"

  [ -x "${PREFIX}/venv/bin/cortex-deployer" ] \
    || die "venv is missing cortex-deployer after install"

  write_wrapper
  local path_ok=0
  if note_path; then
    path_ok=1
  fi

  info "ok  wrapper ${BIN_DIR}/cortex-deployer"
  "${BIN_DIR}/cortex-deployer" --version >&2 || true
  echo >&2
  echo "Next:" >&2
  if [ "$path_ok" -eq 0 ]; then
    echo "  export PATH=\"${BIN_DIR}:\$PATH\"" >&2
  fi
  echo "  cortex-deployer server" >&2
  echo "  # http://127.0.0.1:7480  →  Choose a Qwen build" >&2
  echo >&2
  echo "NVIDIA driver is the only extra host package for GPU. The app" >&2
  echo "fetches official llama.cpp builds and the GGUF itself." >&2

  if [ "$start" -eq 1 ]; then
    exec "${BIN_DIR}/cortex-deployer" server
  fi
}

main "$@"
