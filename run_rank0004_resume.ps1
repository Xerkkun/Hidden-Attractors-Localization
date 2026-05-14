param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$outDir = Join-Path $PSScriptRoot "outputs/lure_biased_multiparam_q09998"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$stdout = Join-Path $outDir "rank0004_resume_stdout.log"
$stderr = Join-Path $outDir "rank0004_resume_stderr.log"

$argsList = @(
    "lure_biased_multiparam_continuation.py",
    "--config", "configs/lure_biased_multiparam_q09998.yaml",
    "--post-continuation-only",
    "--resume",
    "--execute-early-filter",
    "--execute-robustness",
    "--survivor-id", "lure_biased_q_0p99980_rank_0004"
)

$p = Start-Process -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -WindowStyle Hidden

Write-Output "started_pid=$($p.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
