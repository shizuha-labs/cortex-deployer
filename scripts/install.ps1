# Cortex Deployer installer for Windows PowerShell.
#   irm https://cortex.shizuha.com/deployer/install.ps1 | iex
# Never uses the Microsoft Store / system Python pip. Isolates uv + CPython
# under %USERPROFILE%\.cortex-deployer and a wrapper on %USERPROFILE%\.local\bin.
$ErrorActionPreference = "Stop"

$Prefix = if ($env:CORTEX_DEPLOYER_HOME) { $env:CORTEX_DEPLOYER_HOME } else { Join-Path $env:USERPROFILE ".cortex-deployer" }
$BinDir = if ($env:CORTEX_DEPLOYER_BIN_DIR) { $env:CORTEX_DEPLOYER_BIN_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }
$Tarball = if ($env:CORTEX_DEPLOYER_TARBALL) { $env:CORTEX_DEPLOYER_TARBALL } else { "https://github.com/shizuha-labs/cortex-deployer/archive/refs/heads/main.tar.gz" }
$UvVersion = if ($env:CORTEX_DEPLOYER_UV_VERSION) { $env:CORTEX_DEPLOYER_UV_VERSION } else { "0.12.5" }
$PyVersion = if ($env:CORTEX_DEPLOYER_PYTHON) { $env:CORTEX_DEPLOYER_PYTHON } else { "3.12" }

function Info([string]$msg) { Write-Host "cortex-deployer-install: $msg" }

New-Item -ItemType Directory -Force -Path $Prefix, (Join-Path $Prefix "tmp"), (Join-Path $Prefix "bin"), $BinDir | Out-Null

$uv = Join-Path $Prefix "bin\uv.exe"
$uvOk = $false
if (Test-Path $uv) {
  try { & $uv --version | Out-Null; $uvOk = $true } catch { $uvOk = $false }
}
if (-not $uvOk) {
  $zip = Join-Path $Prefix "tmp\uv.zip"
  $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
  Info "bootstrapping our own uv $UvVersion (ignoring any host uv/python)"
  Info "fetch $url"
  Invoke-WebRequest -Uri $url -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath (Join-Path $Prefix "tmp\uv") -Force
  $found = Get-ChildItem -Path (Join-Path $Prefix "tmp\uv") -Recurse -Filter uv.exe | Select-Object -First 1
  if (-not $found) { throw "uv.exe missing from archive" }
  Copy-Item $found.FullName $uv -Force
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Prefix "python"
$env:UV_CACHE_DIR = Join-Path $Prefix "cache"
$env:UV_TOOL_DIR = Join-Path $Prefix "tools"
$env:UV_PYTHON_PREFERENCE = "only-managed"
New-Item -ItemType Directory -Force -Path $env:UV_PYTHON_INSTALL_DIR, $env:UV_CACHE_DIR, $env:UV_TOOL_DIR | Out-Null

Info "standalone CPython $PyVersion (not the OS interpreter)"
& $uv python install $PyVersion

$venv = Join-Path $Prefix "venv"
Info "venv $venv"
& $uv venv $venv --python $PyVersion --clear

Info "install cortex-deployer into the venv"
& $uv pip install --python (Join-Path $venv "Scripts\python.exe") $Tarball

$exe = Join-Path $venv "Scripts\cortex-deployer.exe"
if (-not (Test-Path $exe)) { throw "venv is missing cortex-deployer.exe after install" }

$cmd = Join-Path $BinDir "cortex-deployer.cmd"
@"
@echo off
"$exe" %*
"@ | Set-Content -Path $cmd -Encoding ASCII

$already = [Environment]::GetEnvironmentVariable("Path", "User")
if ($already -notlike "*$BinDir*") {
  [Environment]::SetEnvironmentVariable("Path", "$BinDir;$already", "User")
  $env:Path = "$BinDir;$env:Path"
  Info "added $BinDir to the user PATH"
}

Info "ok  wrapper $cmd"
& $exe --version
Write-Host ""
Write-Host "Next:"
Write-Host "  cortex-deployer server"
Write-Host "  # http://127.0.0.1:7480  ->  Choose a Qwen build"
Write-Host ""
Write-Host "NVIDIA Game Ready / Studio driver is the only extra host install."
Write-Host "The app fetches official llama.cpp CUDA 13 builds and the GGUF."
