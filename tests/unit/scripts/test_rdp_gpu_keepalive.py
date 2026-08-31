# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static contract tests for the generated RDP GPU keepalive worker."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[3]
INSTALLER = ROOT / "scripts" / "agent_setup" / "setup_rdp_gpu_keepalive.ps1"


def _installer_text() -> str:
    return INSTALLER.read_text(encoding="utf-8-sig")


def _worker_text() -> str:
    text = _installer_text()
    marker = "$worker = @'"
    start = text.index(marker) + len(marker)
    end = text.index("\n'@", start)
    return text[start:end].lstrip("\r\n")


def test_installer_and_generated_worker_parse_as_powershell(tmp_path: Path) -> None:
    worker = tmp_path / "keep_console.ps1"
    worker.write_text(_worker_text(), encoding="utf-8")
    powershell = shutil.which("powershell.exe")
    assert powershell is not None

    command = (
        "$errors=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{INSTALLER}',"
        "[ref]$null,[ref]$errors); "
        "if($errors){$errors | ForEach-Object {$_.ToString()}; exit 1}; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{worker}',"
        "[ref]$null,[ref]$errors); "
        "if($errors){$errors | ForEach-Object {$_.ToString()}; exit 1}"
    )
    proc = subprocess.run(  # noqa: S603 - executable and input are repository-controlled
        [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_task_queues_disconnect_events_and_passes_exact_session() -> None:
    text = _installer_text()

    assert "<MultipleInstancesPolicy>Queue</MultipleInstancesPolicy>" in text
    assert "*[System[(EventID=24)]]" in text
    assert "-SessionId $(SessionID)" in text


def test_worker_removes_only_non_present_remote_display_nodes() -> None:
    worker = _worker_text()

    assert "Get-PnpDevice -Class Display" in worker
    assert "SWD\\REMOTEDISPLAYENUM\\*" in worker
    assert "CM_PROB_PHANTOM" in worker
    assert "ConfigManagerErrorCode -eq 45" in worker
    assert "/remove-device $device.InstanceId" in worker
    assert "PCI\\" not in worker


def test_worker_verifies_console_before_display_cleanup() -> None:
    worker = _worker_text()
    verification = "Session $target verified on console; cleaning stale RDP display nodes."

    assert "$current.Name -eq 'console'" in worker
    assert worker.index(verification) < worker.rindex("Remove-StaleRemoteDisplayAdapters")
    assert "console is occupied" in worker


def test_worker_handles_session_one_but_never_services_session() -> None:
    worker = _worker_text()

    assert "$target -eq 0" in worker
    assert "$target -lt 2" not in worker


def test_worker_has_no_destructive_machine_or_session_recovery() -> None:
    worker = _worker_text().casefold()

    for forbidden in ("logoff ", "restart-computer", "stop-computer", "/disable-device"):
        assert forbidden not in worker
