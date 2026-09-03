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
#   24). Genuine network disconnects are identified by the event's Address field
#   and redirected immediately via `tscon`, before Windows settles on a phantom
#   Remote Display Adapter topology. LOCAL events emitted by `tscon`/reconnects
#   are ignored so the worker cannot fight an in-progress reconnect. Stale PnP
#   display nodes are still removed as best-effort hygiene after the redirect.
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
  param(
    [string]$SessionId = '',
    [string]$SourceNetworkAddress = '',
    [switch]$CleanupOnly
  )

$ErrorActionPreference = 'Stop'
$log = 'C:\agent\tools\keep_console.log'
function Write-Log($m) { "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Append -Encoding utf8 }

# Parse `qwinsta` into a map of sessionId -> session facts. Anchors on the state
# keyword and the integer immediately before it, so an empty USERNAME column is
# fine. SESSIONNAME and USERNAME are only used for fail-closed safety checks.
function Get-Sessions {
    $map = @{}
    foreach ($r in (qwinsta 2>$null) -split "`r?`n") {
        $t = ($r -split '\s+') | Where-Object { $_ -ne '' }
        for ($i = 0; $i -lt $t.Count; $i++) {
            if ($t[$i] -in @('Active','Disc','Conn','Listen','Down','Idle')) {
                if ($i -ge 1 -and $t[$i-1] -match '^\d+$') {
                    $id = [int]$t[$i-1]
                    $name = if ($t.Count) { $t[0].TrimStart('>') } else { '' }
                    $username = if ($i -ge 3) { $t[$i-2] } else { '' }
                    $map[$id] = [pscustomobject]@{
                        Id = $id
                        Name = $name
                        Username = $username
                        State = $t[$i]
                    }
                }
                break
            }
        }
    }
    return $map
}

  # Backward-compatible fallback for an already-registered task whose action only
  # passes SessionId. Re-registering the task makes Address available directly,
  # but looking up the newest matching Event 24 lets a worker-only deployment take
  # effect immediately without elevation. Limit the lookup window so a stale
  # network disconnect can never be mistaken for the event being handled.
  function Resolve-SourceNetworkAddress([int]$Target, [string]$Provided) {
    $providedAddress = $Provided.Trim()
    if ($providedAddress) { return $providedAddress }

    try {
      $events = Get-WinEvent -FilterHashtable @{
        LogName = 'Microsoft-Windows-TerminalServices-LocalSessionManager/Operational'
        Id = 24
        StartTime = (Get-Date).AddMinutes(-1)
      } -ErrorAction Stop
      foreach ($event in $events) {
        [xml]$eventXml = $event.ToXml()
        $payload = $eventXml.Event.UserData.EventXML
        $eventSessionId = 0
        if (
          [int]::TryParse([string]$payload.SessionID, [ref]$eventSessionId) -and
          $eventSessionId -eq $Target
        ) {
          return [string]$payload.Address
        }
      }
    }
    catch {
      Write-Log "WARN: cannot resolve Event 24 source address: $($_.Exception.Message)"
    }
    return ''
  }

# Remove only stale/non-present RDP indirect-display device nodes. On affected
# machines a node can be CM_PROB_PHANTOM after `tscon`, yet its cloned GPU handle
# remains visible through DXGI/ORT and may sort before the real adapter. Removing
# that non-present node is safe for the physical GPU (which has a PCI instance ID)
# and a future RDP connection recreates its own Remote Display Adapter as needed.
function Remove-StaleRemoteDisplayAdapters {
    try {
        $stale = @(Get-PnpDevice -Class Display -ErrorAction Stop | Where-Object {
            $_.InstanceId -like 'SWD\REMOTEDISPLAYENUM\*' -and
            ($_.Problem -eq 'CM_PROB_PHANTOM' -or $_.ConfigManagerErrorCode -eq 45)
        })
    }
    catch {
        Write-Log "WARN: cannot enumerate display devices: $($_.Exception.Message)"
        return $false
    }

    if (-not $stale) {
        Write-Log "No stale Remote Display Adapter device nodes found."
        return $true
    }

    $allRemoved = $true
    $pnputil = Join-Path $env:SystemRoot 'System32\pnputil.exe'
    foreach ($device in $stale) {
        Write-Log "Removing stale Remote Display Adapter: $($device.InstanceId)"
        $out = & $pnputil /remove-device $device.InstanceId 2>&1
        $code = $LASTEXITCODE
        if ($out) { $out | ForEach-Object { Write-Log "  pnputil: $_" } }
        if ($code -ne 0) {
            $allRemoved = $false
            Write-Log "WARN: pnputil remove-device exited $code for $($device.InstanceId)."
        }
    }
    return $allRemoved
}

try {
    if ($CleanupOnly) {
        $sessions = Get-Sessions
        $console = @($sessions.Values | Where-Object {
            $_.Name -eq 'console' -and $_.State -in @('Active', 'Conn')
        })
        if (-not $console) {
            Write-Log "WARN: cleanup requested without a console session; refusing."
            exit 0
        }
        [void](Remove-StaleRemoteDisplayAdapters)
        exit 0
    }

    # Act ONLY on the session reported by the disconnect event. The broad
    # "find any Disc session" scan is intentionally gone: during a reconnect the
    # real session briefly flips Disc->Active and the scan would grab an unrelated
    # throwaway session (the source of the tscon 5023 noise).
    if ($SessionId -notmatch '^\d+$') {
        Write-Log "No SessionId supplied by event; nothing to do."
        exit 0
    }
    $target = [int]$SessionId
    if ($target -eq 0) {
      Write-Log "Session 0 is the services session; skipping."
        exit 0
    }

    # A network-backed Event 24 is the real RDP transport disconnect. LOCAL
    # events are generated by tscon and reconnect transitions; ignoring them
    # prevents the old "connect twice" tug-of-war without delaying the genuine
    # redirect long enough for Windows to settle on the phantom adapter.
    $sourceAddress = Resolve-SourceNetworkAddress $target $SourceNetworkAddress
    $parsedAddress = $null
    $isRemoteAddress = (
      [System.Net.IPAddress]::TryParse($sourceAddress, [ref]$parsedAddress) -and
      -not [System.Net.IPAddress]::IsLoopback($parsedAddress) -and
      -not $parsedAddress.Equals([System.Net.IPAddress]::Any) -and
      -not $parsedAddress.Equals([System.Net.IPAddress]::IPv6Any)
    )
    if (-not $isRemoteAddress) {
      Write-Log "Ignoring Event 24 for session $target with source '$sourceAddress'."
      exit 0
    }

    $sessions = Get-Sessions
    $session = $sessions[$target]
    if ($null -eq $session -or $session.State -ne 'Disc') {
      $state = if ($null -eq $session) { '' } else { $session.State }
      Write-Log "Session $target is '$state' at dispatch (reconnected or gone); no redirect needed."
        exit 0
    }

    # Never displace another user's active console session. A connected but
    # unowned console placeholder is safe; tscon will attach the target to it.
    $occupiedConsole = @($sessions.Values | Where-Object {
        $_.Id -ne $target -and $_.Name -eq 'console' -and
        $_.State -eq 'Active' -and $_.Username
    }) | Select-Object -First 1
    if ($occupiedConsole) {
        Write-Log "WARN: console is occupied by '$($occupiedConsole.Username)' (session $($occupiedConsole.Id)); refusing to redirect session $target."
        exit 0
    }

    Write-Log "Remote disconnect from $sourceAddress left session $target Disc; redirecting immediately."
    $redirected = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $out = & tscon $target /dest:console 2>&1
        $code = $LASTEXITCODE
        if ($out) { $out | ForEach-Object { Write-Log "  tscon: $_" } }

        Start-Sleep -Seconds 2
        $current = (Get-Sessions)[$target]
        if ($null -ne $current -and $current.Name -eq 'console' -and $current.State -in @('Active', 'Conn')) {
            $redirected = $true
            break
        }
        if ($null -eq $current -or $current.State -ne 'Disc') {
            Write-Log "WARN: session $target changed state after tscon; not retrying."
            break
        }
        Write-Log "WARN: tscon attempt ${attempt} exited $code without a verified console transition."
    }

    if (-not $redirected) {
        Write-Log "WARN: session $target was not verified on console; skipping display cleanup."
        exit 0
    }

    Write-Log "Session $target verified on console; cleaning stale RDP display nodes."
    [void](Remove-StaleRemoteDisplayAdapters)
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
    <Description>Redirect a disconnected RDP session to the physical console and remove stale RDP display nodes for DirectML CI runs.</Description>
    <Author>winml-cli</Author>
  </RegistrationInfo>
  <Triggers>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="Microsoft-Windows-TerminalServices-LocalSessionManager/Operational"&gt;&lt;Select Path="Microsoft-Windows-TerminalServices-LocalSessionManager/Operational"&gt;*[System[(EventID=24)]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
      <ValueQueries>
        <Value name="SessionID">Event/UserData/EventXML/SessionID</Value>
        <Value name="SourceNetworkAddress">Event/UserData/EventXML/Address</Value>
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
    <MultipleInstancesPolicy>Queue</MultipleInstancesPolicy>
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
      <Arguments>-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "__WORKER__" -SessionId "$(SessionID)" -SourceNetworkAddress "$(SourceNetworkAddress)"</Arguments>
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

    # ---- safe smoke/repair: clean only non-present RDP display nodes ----
    # The installer is elevated, so this also repairs a stale node left by a
    # disconnect that happened before the task was installed or refreshed.
    $smoke = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $WorkerPath -CleanupOnly 2>&1
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
