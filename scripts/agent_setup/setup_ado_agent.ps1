<#
.SYNOPSIS
    Registers a Windows Scheduled Task to start the ADO agent at user logon
    with a visible console window.

.DESCRIPTION
    Creates a scheduled task that runs C:\agent\run.cmd when the current
    user logs in. The agent runs interactively so its console window is
    visible on the desktop. The script will self-elevate to admin if needed
    (a UAC prompt will appear).

.PARAMETER Unregister
    Remove the scheduled task instead of creating it.
#>

param(
    [switch]$Unregister
)

# Self-elevate if not running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Not running as admin. Requesting elevation..." -ForegroundColor Yellow
    $argList = "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($Unregister) { $argList += " -Unregister" }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList -Wait
    exit
}

$TaskName = "ADO_Agent_AutoStart"
$AgentCmd = "C:\agent\run.cmd"

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Scheduled task '$TaskName' has been removed." -ForegroundColor Green
    }
    else {
        Write-Host "Scheduled task '$TaskName' does not exist." -ForegroundColor Yellow
    }
    return
}

# Validate that the agent command exists
if (-not (Test-Path $AgentCmd)) {
    Write-Error "Agent command not found at '$AgentCmd'. Please verify the path."
    exit 1
}

# Remove existing task if present to allow re-registration
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Task '$TaskName' already exists. Replacing..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$AgentCmd`"" -WorkingDirectory "C:\agent"
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Run interactively as the current user so the console window is visible
$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Starts the ADO agent (C:\agent\run.cmd) at user logon with visible console" | Out-Null

Write-Host "Scheduled task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "  Trigger  : At logon of $CurrentUser (console window visible)" -ForegroundColor Cyan
Write-Host "  User     : $CurrentUser" -ForegroundColor Cyan
Write-Host "  Action   : $AgentCmd" -ForegroundColor Cyan
Write-Host "  Run Level: Highest" -ForegroundColor Cyan
