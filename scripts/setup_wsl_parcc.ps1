# Finish the WSL + Ubuntu setup for PARCC Betty. Run ONCE, AFTER the reboot that follows
# `wsl --install --no-distribution` (already done on 2026-09-13). No admin rights needed.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_parcc.ps1
#
# Steps: register Ubuntu 24.04 without the interactive first-run, run scripts/setup_ubuntu_parcc.sh
# as root inside it (packages, krb5.conf, user, ssh config), restart the distro, print a smoke test.
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$LinuxUser = "hyhuang"     # = PennKey, so `ssh betty` needs no username
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"

Write-Host "== WSL status =="
wsl.exe --status
if ($LASTEXITCODE -ne 0) { throw "WSL is not ready. Reboot first (VirtualMachinePlatform was enabled on 2026-09-13)." }

Write-Host "== register $Distro (no launch) =="
$installed = (wsl.exe --list --quiet) -join "`n"
if ($installed -notmatch [regex]::Escape($Distro)) {
    wsl.exe --install -d $Distro --no-launch
    if ($LASTEXITCODE -ne 0) { throw "distro install failed ($LASTEXITCODE)" }
}
wsl.exe --set-default $Distro | Out-Null

Write-Host "== run setup_ubuntu_parcc.sh as root =="
$sh = Join-Path $PSScriptRoot "setup_ubuntu_parcc.sh"
$content = (Get-Content $sh -Raw) -replace "`r`n", "`n"
$tmp = Join-Path $env:TEMP "setup_ubuntu_parcc.sh"
[IO.File]::WriteAllText($tmp, $content, (New-Object System.Text.UTF8Encoding($false)))
# copy the script into the distro through stdin (avoids path/quoting issues), then execute it
Get-Content $tmp -Raw | wsl.exe -d $Distro -u root -- bash -c "cat > /root/setup_ubuntu_parcc.sh"
wsl.exe -d $Distro -u root -- bash /root/setup_ubuntu_parcc.sh $LinuxUser
if ($LASTEXITCODE -ne 0) { throw "in-distro setup failed ($LASTEXITCODE)" }

Write-Host "== restart distro so /etc/wsl.conf applies =="
wsl.exe --terminate $Distro
Start-Sleep -Seconds 2

Write-Host "== smoke test (as $LinuxUser) =="
wsl.exe -d $Distro -- bash -lc 'echo "user: $(whoami)"; echo "realm: $(grep default_realm /etc/krb5.conf)"; kinit --version 2>&1 | head -1; ssh -V; nslookup -type=SRV _kerberos._udp.upenn.edu 2>/dev/null | grep -c kerberos | xargs echo "KDC SRV records:"'

Write-Host ""
Write-Host "Next (interactive, needs your PennKey password + Duo once):"
Write-Host "  wsl"
Write-Host "  kb                 # kinit $LinuxUser@UPENN.EDU"
Write-Host "  betty              # ssh login.betty.parcc.upenn.edu (Duo push)"
Write-Host "  ssh-copy-id betty  # register the WSL key; afterwards kinit + key = no Duo"
