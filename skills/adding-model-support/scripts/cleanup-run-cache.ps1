[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [double]$MinimumFreeGB = 25,
    [double]$TargetFreeGB = 40,
    [switch]$Execute,
    [string]$RecordPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$allowedKinds = @(
    "generated_model",
    "build_output",
    "private_environment",
    "run_cache",
    "eval_media",
    "python_bytecode",
    "git_worktree"
)

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root,
        [switch]$AllowEqual
    )

    $normalizedPath = Get-NormalizedPath $Path
    $normalizedRoot = Get-NormalizedPath $Root
    if ($AllowEqual -and $normalizedPath.Equals($normalizedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $prefix = $normalizedRoot + [System.IO.Path]::DirectorySeparatorChar
    return $normalizedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-ReparsePointInPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $current = Get-NormalizedPath $Path
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                return $true
            }
        }
        $parent = Split-Path -Parent $current
        if ($parent -eq $current) {
            break
        }
        $current = $parent
    }
    return $false
}

function Test-ReparsePointInTree {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-ReparsePointInPath $Path) {
        return $true
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push((Get-NormalizedPath $Path))
    while ($pending.Count -gt 0) {
        $directory = [System.IO.DirectoryInfo]::new($pending.Pop())
        foreach ($entry in $directory.EnumerateFileSystemInfos()) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                return $true
            }
            if (($entry.Attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {
                $pending.Push($entry.FullName)
            }
        }
    }
    return $false
}

function Get-PathBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return [int64]0
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) {
        return [int64]$item.Length
    }
    $measurement = Get-ChildItem -LiteralPath $Path -Force -Recurse -File |
        Measure-Object -Property Length -Sum
    if ($null -eq $measurement) {
        return [int64]0
    }
    $sumProperty = $measurement.PSObject.Properties["Sum"]
    if ($null -eq $sumProperty -or $null -eq $sumProperty.Value) {
        return [int64]0
    }
    return [int64]$sumProperty.Value
}

function Get-FreeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)

    $root = [System.IO.Path]::GetPathRoot((Get-NormalizedPath $Path))
    return [int64]([System.IO.DriveInfo]::new($root).AvailableFreeSpace)
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $output = & git -C $WorkingDirectory @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git -C '$WorkingDirectory' $($Arguments -join ' ') failed: $output"
    }
    return @($output)
}

function Test-CandidateShape {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$RunId
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) {
        if ($Kind -eq "generated_model" -and $item.Name -match "(?i)(\.onnx(?:\.data)?|\.data)$") {
            return "valid"
        }
        if ($Kind -eq "python_bytecode" -and $item.Name -match "(?i)\.(pyc|pyo)$") {
            return "valid"
        }
        if ($Kind -eq "eval_media" -and $item.Name -match "(?i)\.(bmp|flac|gif|jpe?g|mp3|mp4|ogg|png|wav|webp)$") {
            return "valid"
        }
        return "file type does not match the declared cleanup kind"
    }

    if ($Kind -eq "python_bytecode" -and $item.Name -eq "__pycache__") {
        return "valid"
    }
    if ($Kind -eq "git_worktree") {
        return "valid"
    }

    $markerPath = Join-Path $Path ".modelkit-cleanup-owned.json"
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf) -or (Test-ReparsePointInPath $markerPath)) {
        return "directory lacks a regular ownership marker"
    }
    try {
        $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    } catch {
        return "directory ownership marker is invalid JSON"
    }
    if ($marker.schema_version -ne 1 -or [string]$marker.run_id -ne $RunId -or [string]$marker.kind -ne $Kind) {
        return "directory ownership marker does not match run_id and kind"
    }
    return "valid"
}

if ($MinimumFreeGB -lt 25 -or $TargetFreeGB -lt 40 -or $TargetFreeGB -lt $MinimumFreeGB) {
    throw "Automatic cleanup requires MinimumFreeGB >= 25, TargetFreeGB >= 40, and TargetFreeGB >= MinimumFreeGB."
}

$manifestFullPath = Get-NormalizedPath $ManifestPath
if (-not (Test-Path -LiteralPath $manifestFullPath -PathType Leaf) -or (Test-ReparsePointInPath $manifestFullPath)) {
    throw "Cleanup manifest must be an existing non-reparse file."
}
$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1) {
    throw "Unsupported cleanup manifest schema_version '$($manifest.schema_version)'."
}
if ($manifest.terminal_state -notin @("APPROVE", "REJECT", "BLOCKED")) {
    throw "Cleanup requires a terminal run state."
}
if ($manifest.quiescent -ne $true -or $manifest.dependencies_complete -ne $true) {
    throw "Cleanup requires explicit quiescence and completed dependent checks."
}
if (-not $manifest.allowed_roots -or -not $manifest.candidates) {
    throw "Cleanup manifest must declare allowed_roots and candidates."
}

$allowedRoots = @($manifest.allowed_roots | ForEach-Object { Get-NormalizedPath ([string]$_) })
$repositoryRoots = @($manifest.repository_roots | ForEach-Object { Get-NormalizedPath ([string]$_) })
$protectedPaths = @(
    (Join-Path $HOME ".cache\huggingface"),
    (Join-Path $HOME ".cache\uv"),
    (Join-Path $HOME ".cache\pip"),
    (Join-Path $HOME ".cache\winml")
)
if ($env:LOCALAPPDATA) {
    $protectedPaths += Join-Path $env:LOCALAPPDATA "pip\Cache"
}
foreach ($variableName in @("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "UV_CACHE_DIR", "PIP_CACHE_DIR", "WINML_CACHE_DIR")) {
    $value = [Environment]::GetEnvironmentVariable($variableName)
    if ($value) {
        $protectedPaths += $value
    }
}
if ($manifest.PSObject.Properties["protected_paths"]) {
    $protectedPaths += @($manifest.protected_paths)
}
$protectedPaths = @($protectedPaths | ForEach-Object { Get-NormalizedPath $_ } | Select-Object -Unique)
$recordFullPath = $null
if ($RecordPath) {
    $recordFullPath = Get-NormalizedPath $RecordPath
    $recordDirectory = Split-Path -Parent $recordFullPath
    if ([System.IO.Path]::GetExtension($recordFullPath) -ne ".json") {
        throw "Cleanup record path must use the .json extension."
    }
    if (-not $recordDirectory -or -not (Test-Path -LiteralPath $recordDirectory -PathType Container) -or (Test-ReparsePointInPath $recordFullPath)) {
        throw "Cleanup record parent must be an existing non-reparse directory."
    }
    if (Test-Path -LiteralPath $recordFullPath -PathType Leaf) {
        try {
            $existingRecord = Get-Content -LiteralPath $recordFullPath -Raw | ConvertFrom-Json
        } catch {
            throw "Existing cleanup record is not valid JSON and will not be overwritten."
        }
        if ($existingRecord.schema_version -ne 1 -or [string]$existingRecord.run_id -ne [string]$manifest.run_id) {
            throw "Existing cleanup record belongs to another schema or run and will not be overwritten."
        }
    }
}

foreach ($allowedRoot in $allowedRoots) {
    if ($allowedRoot -eq [System.IO.Path]::GetPathRoot($allowedRoot)) {
        throw "A filesystem root cannot be declared as a run-owned root: '$allowedRoot'."
    }
    if (-not (Test-Path -LiteralPath $allowedRoot -PathType Container) -or (Test-ReparsePointInPath $allowedRoot)) {
        throw "Run-owned root must be an existing non-reparse directory: '$allowedRoot'."
    }
    if ($protectedPaths | Where-Object { Test-PathWithin $_ $allowedRoot -AllowEqual }) {
        throw "Run-owned root '$allowedRoot' contains a protected path. Declare a narrower root."
    }
    if ($repositoryRoots | Where-Object { Test-PathWithin $_ $allowedRoot -AllowEqual }) {
        throw "Run-owned root '$allowedRoot' contains a repository root. Declare a narrower root."
    }
    if (Test-PathWithin $manifestFullPath $allowedRoot -AllowEqual) {
        throw "The cleanup manifest must be outside disposable run-owned roots."
    }
    if ($recordFullPath -and (Test-PathWithin $recordFullPath $allowedRoot -AllowEqual)) {
        throw "The cleanup record must be outside disposable run-owned roots."
    }
}

$anchorPath = $allowedRoots[0]
$anchorVolume = [System.IO.Path]::GetPathRoot($anchorPath)
foreach ($allowedRoot in $allowedRoots) {
    if (-not ([System.IO.Path]::GetPathRoot($allowedRoot)).Equals($anchorVolume, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "All run-owned roots in one cleanup manifest must be on the same volume."
    }
}
$freeBefore = Get-FreeBytes $anchorPath
$minimumBytes = [int64]($MinimumFreeGB * 1GB)
$targetBytes = [int64]($TargetFreeGB * 1GB)
$lowSpace = $freeBefore -lt $minimumBytes
$mode = if ($Execute) { "execute" } else { "dry-run" }
$results = [System.Collections.Generic.List[object]]::new()
$plannedBytes = [int64]0
$removedBytes = [int64]0

foreach ($candidate in $manifest.candidates) {
    $candidatePath = Get-NormalizedPath ([string]$candidate.path)
    $kind = [string]$candidate.kind
    $status = "eligible"
    $reason = "run-owned regenerable candidate"

    if ($kind -notin $allowedKinds) {
        $status = "blocked"
        $reason = "kind is not in the cleanup allowlist"
    } elseif ($candidate.regenerable -ne $true -or $candidate.dependent_checks_complete -ne $true) {
        $status = "blocked"
        $reason = "candidate is not explicitly regenerable or still has dependent checks"
    } elseif ($kind -eq "eval_media" -and $candidate.provenance_preserved -ne $true) {
        $status = "blocked"
        $reason = "eval media provenance is not preserved"
    } elseif (-not ($allowedRoots | Where-Object { Test-PathWithin $candidatePath $_ })) {
        $status = "blocked"
        $reason = "path is outside or equal to a declared run-owned root"
    } elseif (Test-PathWithin $manifestFullPath $candidatePath -AllowEqual) {
        $status = "blocked"
        $reason = "candidate contains the cleanup manifest"
    } elseif ($RecordPath -and (Test-PathWithin (Get-NormalizedPath $RecordPath) $candidatePath -AllowEqual)) {
        $status = "blocked"
        $reason = "candidate contains the cleanup record"
    } elseif ($protectedPaths | Where-Object { Test-PathWithin $candidatePath $_ -AllowEqual }) {
        $status = "blocked"
        $reason = "path is in a protected shared cache"
    } elseif ($kind -ne "git_worktree" -and ($repositoryRoots | Where-Object { Test-PathWithin $candidatePath $_ -AllowEqual })) {
        $status = "blocked"
        $reason = "path is inside a protected repository root"
    } elseif ((Split-Path -Leaf $candidatePath) -eq ".git" -or $candidatePath -match "[\\/]\.git[\\/]") {
        $status = "blocked"
        $reason = "Git metadata is always protected"
    } elseif (Test-ReparsePointInTree $candidatePath) {
        $status = "blocked"
        $reason = "paths containing reparse points are not eligible for automatic deletion"
    }

    if ($status -eq "eligible") {
        $shape = Test-CandidateShape $candidatePath $kind ([string]$manifest.run_id)
        if ($shape -eq "missing") {
            $status = "skipped"
            $reason = "candidate no longer exists"
        } elseif ($shape -ne "valid") {
            $status = "blocked"
            $reason = $shape
        }
    }

    $bytes = [int64]0
    if ($status -eq "eligible") {
        $bytes = Get-PathBytes $candidatePath
        $plannedBytes += $bytes
        if (-not $lowSpace) {
            $status = "skipped"
            $reason = "free space is above the automatic cleanup threshold"
        } elseif ($Execute -and (Get-FreeBytes $anchorPath) -ge $targetBytes) {
            $status = "skipped"
            $reason = "target free space has already been reached"
        } elseif ($Execute -and (Test-Path -LiteralPath $candidatePath)) {
            if ($kind -eq "git_worktree") {
                if (-not $candidate.repository_root -or -not $candidate.recovery_ref) {
                    throw "Git worktree candidate '$candidatePath' requires repository_root and recovery_ref."
                }
                $repositoryRoot = Get-NormalizedPath ([string]$candidate.repository_root)
                if ($repositoryRoot -notin $repositoryRoots) {
                    throw "Git worktree candidate '$candidatePath' names an undeclared repository root."
                }
                if ((Invoke-Git $candidatePath @("status", "--porcelain"))) {
                    throw "Git worktree candidate '$candidatePath' is dirty."
                }
                $headOutput = @(Invoke-Git $candidatePath @("rev-parse", "HEAD"))
                $head = [string]$headOutput[0]
                $remoteNames = @(Invoke-Git $repositoryRoot @("remote"))
                $recoveryRef = [string]$candidate.recovery_ref
                if ($recoveryRef -match "^refs/remotes/([^/]+)/.+$" -and $Matches[1] -in $remoteNames) {
                    $remoteTrackingRef = $recoveryRef
                } elseif ($recoveryRef -match "^([^/]+)/.+$" -and $Matches[1] -in $remoteNames) {
                    $remoteTrackingRef = "refs/remotes/$recoveryRef"
                } else {
                    throw "Git worktree recovery_ref must name a configured remote-tracking ref."
                }
                Invoke-Git $repositoryRoot @("rev-parse", "--verify", "$remoteTrackingRef^{commit}") | Out-Null
                & git -C $repositoryRoot merge-base --is-ancestor $head $remoteTrackingRef
                if ($LASTEXITCODE -ne 0) {
                    throw "Git worktree HEAD '$head' is not recoverable from '$remoteTrackingRef'."
                }
                if (Test-ReparsePointInTree $candidatePath) {
                    throw "Git worktree '$candidatePath' gained a reparse point after inventory."
                }
                Invoke-Git $repositoryRoot @("worktree", "remove", $candidatePath) | Out-Null
            } else {
                if (Test-ReparsePointInTree $candidatePath) {
                    throw "Candidate '$candidatePath' gained a reparse point after inventory."
                }
                Remove-Item -LiteralPath $candidatePath -Force -Recurse
            }
            $removedBytes += $bytes
            $status = "removed"
            $reason = "removed after all safety gates passed"
        }
    }

    $results.Add([ordered]@{
        path = $candidatePath
        kind = $kind
        bytes = $bytes
        status = $status
        reason = $reason
    })
}

$freeAfter = Get-FreeBytes $anchorPath
$record = [ordered]@{
    schema_version = 1
    run_id = [string]$manifest.run_id
    terminal_state = [string]$manifest.terminal_state
    mode = $mode
    threshold_triggered = $lowSpace
    minimum_free_gb = $MinimumFreeGB
    target_free_gb = $TargetFreeGB
    free_bytes_before = $freeBefore
    free_bytes_after = $freeAfter
    planned_bytes = $plannedBytes
    removed_bytes = $removedBytes
    target_reached = $freeAfter -ge $targetBytes
    status = if ($results.status -contains "blocked" -or ($Execute -and $lowSpace -and $freeAfter -lt $targetBytes)) { "BLOCKED" } elseif ($Execute -and $lowSpace) { "SEALED" } else { "INVENTORY" }
    candidates = @($results)
}

$json = $record | ConvertTo-Json -Depth 8
if ($RecordPath) {
    if (Test-ReparsePointInPath $recordFullPath) {
        throw "Cleanup record path gained a reparse point after validation."
    }
    [System.IO.File]::WriteAllText($recordFullPath, $json + [Environment]::NewLine)
}
$json

if ($record.status -eq "BLOCKED") {
    exit 2
}
