$ErrorActionPreference = "Stop"

Set-Location -LiteralPath "C:\Users\moren\Desktop\codigo_mac"

$outputDir = "outputs\extended_search\machado_targeted_verification"
$logDir = Join-Path $outputDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "machado_targeted_$stamp.log"

python machado_targeted_verification.py `
  --candidate-id all `
  --config configs\machado_targeted_verification.yaml `
  --output-dir outputs\extended_search\machado_targeted_verification `
  --resume `
  --max-trajectories 1000 *>&1 | Tee-Object -FilePath $logPath
