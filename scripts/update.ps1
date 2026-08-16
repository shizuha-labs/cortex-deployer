# In-place upgrade. Does not wipe models or recreate the venv.
#   cortex-deployer update
#   irm https://cortex.shizuha.com/deployer/update.ps1 | iex
$ErrorActionPreference = "Stop"

$Prefix = if ($env:CORTEX_DEPLOYER_HOME) { $env:CORTEX_DEPLOYER_HOME } else { Join-Path $env:USERPROFILE ".cortex-deployer" }
$BinDir = if ($env:CORTEX_DEPLOYER_BIN_DIR) { $env:CORTEX_DEPLOYER_BIN_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }

$cmd = Join-Path $BinDir "cortex-deployer.cmd"
$exe = Join-Path $Prefix "venv\Scripts\cortex-deployer.exe"
if (Test-Path $cmd) {
  & $cmd update @args
  exit $LASTEXITCODE
}
if (Test-Path $exe) {
  & $exe update @args
  exit $LASTEXITCODE
}
if (Get-Command cortex-deployer -ErrorAction SilentlyContinue) {
  & cortex-deployer update @args
  exit $LASTEXITCODE
}

Write-Host "cortex-deployer-update: not installed yet. First time:"
Write-Host "  irm https://cortex.shizuha.com/deployer/install.ps1 | iex"
exit 1
