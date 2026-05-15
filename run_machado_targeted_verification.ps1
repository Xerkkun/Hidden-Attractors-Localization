$ErrorActionPreference = "Stop"

$repoRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -LiteralPath $repoRoot

$configPath = "configs/machado_targeted_verification_lm10.yaml"
$outputDir = "outputs/extended_search/machado_targeted_verification_lm10"
$logDir = Join-Path $outputDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "machado_targeted_lm10_$stamp.log"

$env:OMP_NUM_THREADS = "1"
$env:OMP_THREAD_LIMIT = "1"

$python = if (Get-Command python3 -ErrorAction SilentlyContinue) {
  "python3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  "python"
} else {
  throw "Python was not found in PATH."
}

& $python machado_targeted_verification.py `
  --candidate-id all `
  --config $configPath `
  --output-dir $outputDir `
  --resume `
  --max-trajectories 1000 *>&1 | Tee-Object -FilePath $logPath
