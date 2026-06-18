# setup_rdp_gpu_keepalive.ps1
#
# Problem this solves:
#   On this self-hosted ADO agent the GPU (Intel Arc) is only usable by DirectML
#   while the user's RDP session is connected. When RDP disconnects, the session
#   loses the physical display adapter (falls back to "Microsoft Remote Display
#   Adapter", no D3D12) and every DirectML CI step fails. OpenVINO/CPU are
#   unaffected, so only the `dml_gpu` eval steps and `test_perf_e2e.py` break.
#
# Fix:
#   Register a SYSTEM, event-triggered scheduled task ("KeepSessionOnConsole")
#   that fires on RDP disconnect (TerminalServices-LocalSessionManager Event ID
#   24) and redirects the still-disconnected session back to the physical console
#   via `tscon`, so the GPU stays bound and DirectML keeps working headless.
#
# Usage (run once, elevated):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\agent_setup\setup_rdp_gpu_keepalive.ps1
#
# Idempotent: re-run any time to refresh the worker script and re-register the
# task. Editing only the worker script does NOT need elevation (C:\agent\tools
# grants Authenticated Users Modify); re-registering the task does.

$ErrorActionPreference = 'Stop'

# All artifacts live next to the agent so the script is machine-agnostic.
$ToolsDir   = 'C:\agent\tools'
$WorkerPath = Join-Path $ToolsDir 'keep_console.ps1'
$log        = Join-Path $ToolsDir 'setup_rdp_gpu_keepalive.log'
$TaskName   = 'KeepSessionOnConsole'

function L($m) { "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $m | Tee-Object -FilePath $log -Append }

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)

try {
    New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
    "=== install start (elevated=$elevated) ===" | Out-File $log -Encoding utf8

    if (-not $elevated) { L "ERROR: not elevated; re-run as administrator."; exit 1 }
    L "tools dir ready: $ToolsDir"

    # ---- keep_console.ps1 (the redirect worker) ----
    $worker = @'
param([string]$SessionId = '')

$ErrorActionPreference = 'Stop'
$log = 'C:\agent\tools\keep_console.log'
function Write-Log($m) { "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Append -Encoding utf8 }

# Parse `qwinsta` into a map of sessionId -> state. Anchors on the state keyword
# and the integer immediately before it, so an empty SESSIONNAME column is fine.
function Get-Sessions {
    $map = @{}
    foreach ($r in (qwinsta 2>$null) -split "`r?`n") {
        $t = ($r -split '\s+') | Where-Object { $_ -ne '' }
        for ($i = 0; $i -lt $t.Count; $i++) {
            if ($t[$i] -in @('Active','Disc','Conn','Listen','Down','Idle')) {
                if ($i -ge 1 -and $t[$i-1] -match '^\d+$') { $map[[int]$t[$i-1]] = $t[$i] }
                break
            }
        }
    }
    return $map
}

try {
    # Act ONLY on the session reported by the disconnect event. The broad
    # "find any Disc session" scan is intentionally gone: during a reconnect the
    # real session briefly flips Disc->Active and the scan would grab an unrelated
    # throwaway session (the source of the tscon 5023 noise).
    if ($SessionId -notmatch '^\d+$') {
        Write-Log "No SessionId supplied by event; nothing to do."
        exit 0
    }
    $target = [int]$SessionId
    if ($target -lt 2) {
        Write-Log "Session $target is a system/console session; skipping."
        exit 0
    }

    # Settle delay: a plain reconnect produces a transient disconnect. Waiting a
    # few seconds and re-checking lets a reconnecting session return to Active so
    # we DON'T yank it back to console mid-reconnect (which caused the
    # "connect twice" tug-of-war). Only a session that is STILL disconnected
    # after the delay is treated as "user really left".
    Start-Sleep -Seconds 10

    $sessions = Get-Sessions
    $state = $sessions[$target]
    if ($state -ne 'Disc') {
        Write-Log "Session $target is '$state' after settle (reconnected or gone); no redirect needed."
        exit 0
    }

    Write-Log "Session $target still Disc after settle; redirecting to console."
    $out = & tscon $target /dest:console 2>&1
    $code = $LASTEXITCODE
    if ($out) { $out | ForEach-Object { Write-Log "  tscon: $_" } }
    if ($code -eq 0) {
        Write-Log "Session $target redirected to console (GPU retained)."
    }
    else {
        # Non-fatal: the session state can change between the check and tscon.
        Write-Log "WARN: tscon $target /dest:console exited $code (session likely changed state); ignoring."
    }
    exit 0
}
catch {
    Write-Log "WARN: $($_.Exception.Message)"
    exit 0
}
'@
    Set-Content -Path $WorkerPath -Value $worker -Encoding utf8
    L "wrote $WorkerPath"

    # ---- scheduled task definition ----
    $xml = @'
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Redirect a disconnected RDP session back to the physical console so the GPU stays available for DirectML CI runs.</Description>
    <Author>winml-cli</Author>
  </RegistrationInfo>
  <Triggers>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="Microsoft-Windows-TerminalServices-LocalSessionManager/Operational"&gt;&lt;Select Path="Microsoft-Windows-TerminalServices-LocalSessionManager/Operational"&gt;*[System[(EventID=24)]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
      <ValueQueries>
        <Value name="SessionID">Event/UserData/EventXML/SessionID</Value>
      </ValueQueries>
    </EventTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>S-1-5-18</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <ExecutionTimeLimit>PT1M</ExecutionTimeLimit>
    <Priority>5</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "__WORKER__" -SessionId $(SessionID)</Arguments>
    </Exec>
  </Actions>
</Task>
'@
    # Single-quoted here-string keeps $(SessionID) literal for Task Scheduler;
    # inject the worker path via a placeholder so it stays parameterized.
    $xml = $xml.Replace('__WORKER__', $WorkerPath)

    Register-ScheduledTask -Xml $xml -TaskName $TaskName -Force | Out-Null
    L "registered task $TaskName"

    $t = Get-ScheduledTask -TaskName $TaskName
    L ("task state={0} principal={1} runlevel={2}" -f $t.State, $t.Principal.UserId, $t.Principal.RunLevel)

    # ---- safe smoke test: run worker once now (user is connected => no Disc => no redirect) ----
    $smoke = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $WorkerPath 2>&1
    L ("smoke worker exit={0}" -f $LASTEXITCODE)
    L ("smoke worker output: {0}" -f (($smoke | Out-String).Trim()))
    $tail = (Get-Content (Join-Path $ToolsDir 'keep_console.log') -Tail 3 -ErrorAction SilentlyContinue) -join ' | '
    L "smoke worker log: $tail"

    L "=== install OK ==="
}
catch {
    L "ERROR: $($_.Exception.Message)"
    L $_.ScriptStackTrace
    exit 1
}
