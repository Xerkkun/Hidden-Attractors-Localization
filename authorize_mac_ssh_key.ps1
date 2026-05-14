param(
    [string]$User = "Fer",
    [string]$HostName = "192.168.14.38"
)

$ErrorActionPreference = "Stop"

$pubKeyPath = Join-Path $env:USERPROFILE ".ssh/id_ed25519.pub"
if (-not (Test-Path -LiteralPath $pubKeyPath)) {
    throw "No public key found at $pubKeyPath"
}

$pubKey = Get-Content -LiteralPath $pubKeyPath -Raw
$escaped = $pubKey.Replace("'", "'\''").Trim()

$remote = "$User@$HostName"
$remoteCommand = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '$escaped' ~/.ssh/authorized_keys || echo '$escaped' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

Write-Output "This will prompt for the SSH password for $remote."
& C:\WINDOWS\System32\OpenSSH\ssh.exe $remote $remoteCommand
