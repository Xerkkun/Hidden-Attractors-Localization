param(
    [ValidateRange(1, 16)]
    [int]$Workers = 4,
    [string]$Stamp = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = (Resolve-Path (Join-Path $ScriptRoot "..\..\..\.venv\Scripts\python.exe")).Path
$Search = (Resolve-Path (Join-Path $ScriptRoot "lure_biased_multiparam_search.py")).Path
$Config = (Resolve-Path (Join-Path $ScriptRoot "..\..\configs\machado_candidate_route.yaml")).Path
$OutputRoot = "outputs/machado_lure_route_$Stamp"
$RunId = "machado_lure_route_q09998_corrected_$Stamp"
$OutputPath = Join-Path $ScriptRoot $OutputRoot
$LogPath = Join-Path $OutputPath "parallel_logs"
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null

function Quote-Argument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Start-SearchWorker([int]$Index) {
    $arguments = @(
        (Quote-Argument $Search),
        "--config", (Quote-Argument $Config),
        "--output-root", (Quote-Argument $OutputRoot),
        "--run-id", (Quote-Argument $RunId),
        "--search-worker-count", "$Workers",
        "--search-worker-index", "$Index"
    )
    return Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogPath "search_worker_$Index.out.log") `
        -RedirectStandardError (Join-Path $LogPath "search_worker_$Index.err.log")
}

function Start-PeriodicityWorker([int]$Index) {
    $arguments = @(
        (Quote-Argument $Search),
        "--config", (Quote-Argument $Config),
        "--output-root", (Quote-Argument $OutputRoot),
        "--run-id", (Quote-Argument $RunId),
        "--periodicity-only",
        "--periodicity-worker-count", "$Workers",
        "--periodicity-worker-index", "$Index"
    )
    return Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ScriptRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogPath "periodicity_worker_$Index.out.log") `
        -RedirectStandardError (Join-Path $LogPath "periodicity_worker_$Index.err.log")
}

@{
    run_id = $RunId
    output_root = $OutputRoot
    workers = $Workers
    status = "raw_df_workers_started"
    generated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputPath "parallel_launcher_status.json")

$processes = 0..($Workers - 1) | ForEach-Object { Start-SearchWorker $_ }
$processes | Wait-Process
foreach ($process in $processes) {
    if ($process.ExitCode -ne 0) {
        throw "Raw DF/FDF worker $($process.Id) failed with exit code $($process.ExitCode)."
    }
}

& $Python $Search --config $Config --output-root $OutputRoot --run-id $RunId --search-worker-count $Workers --aggregate-search-workers --prepare-only `
    *> (Join-Path $LogPath "search_aggregate.log")
if ($LASTEXITCODE -ne 0) {
    throw "Global Machado DF/FDF aggregation/refinement failed with exit code $LASTEXITCODE."
}

@{
    run_id = $RunId
    output_root = $OutputRoot
    workers = $Workers
    status = "post_transient_workers_started"
    generated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputPath "parallel_launcher_status.json")

$processes = 0..($Workers - 1) | ForEach-Object { Start-PeriodicityWorker $_ }
$processes | Wait-Process
foreach ($process in $processes) {
    if ($process.ExitCode -ne 0) {
        throw "Post-transient periodicity worker $($process.Id) failed with exit code $($process.ExitCode)."
    }
}

& $Python $Search --config $Config --output-root $OutputRoot --run-id $RunId --periodicity-only --periodicity-worker-count $Workers --aggregate-periodicity-workers `
    *> (Join-Path $LogPath "periodicity_aggregate.log")
if ($LASTEXITCODE -ne 0) {
    throw "Machado periodicity aggregation failed with exit code $LASTEXITCODE."
}

@{
    run_id = $RunId
    output_root = $OutputRoot
    workers = $Workers
    status = "completed"
    generated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputPath "parallel_launcher_status.json")
