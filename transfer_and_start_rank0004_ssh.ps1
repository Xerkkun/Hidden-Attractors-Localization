param(
    [string]$User = "Fer",
    [string]$HostName = "192.168.14.38",
    [string]$ZipPath = "__transfer_rank0004_20260514_173005.zip",
    [string]$RemoteDir = "~/rank0004_transfer"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$resolvedZip = Resolve-Path -LiteralPath $ZipPath
$remote = "$User@$HostName"
$remoteZip = "~/rank0004_transfer.zip"

Write-Output "Copying $resolvedZip to ${remote}:$remoteZip"
& C:\WINDOWS\System32\OpenSSH\scp.exe $resolvedZip $remote`:$remoteZip

$remoteCommand = @"
set -e
mkdir -p $RemoteDir
unzip -o $remoteZip -d $RemoteDir
cd $RemoteDir
mkdir -p outputs/lure_biased_multiparam_q09998
nohup python3 lure_biased_multiparam_continuation.py \
  --config configs/lure_biased_multiparam_q09998.yaml \
  --post-continuation-only \
  --resume \
  --execute-early-filter \
  --execute-robustness \
  --survivor-id lure_biased_q_0p99980_rank_0004 \
  > outputs/lure_biased_multiparam_q09998/rank0004_remote_stdout.log \
  2> outputs/lure_biased_multiparam_q09998/rank0004_remote_stderr.log &
echo started_remote_pid=\$!
"@

Write-Output "Starting remote rank_0004 resume on $remote"
& C:\WINDOWS\System32\OpenSSH\ssh.exe $remote $remoteCommand
