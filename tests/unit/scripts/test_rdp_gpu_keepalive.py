# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static contract tests for the generated RDP GPU keepalive worker."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


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


def _run_worker(tmp_path: Path, source_address: str) -> tuple[str, bool]:
    log = tmp_path / "keep_console.log"
    marker = tmp_path / "tscon-called"
    worker = tmp_path / "keep_console.ps1"
    worker.write_text(
        _worker_text().replace(
            "$log = 'C:\\agent\\tools\\keep_console.log'",
            f"$log = '{log}'",
        ),
        encoding="utf-8",
    )
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        f"""
function global:qwinsta {{
    if ($env:KEEPALIVE_TEST_CONSOLE -eq '1') {{
        return '>console yuesu 2 Active'
    }}
    return @('>rdp-tcp#0 yuesu 2 Disc', ' console 8 Conn')
}}
function global:tscon {{
    Set-Content -Path '{marker}' -Value 'called'
    $env:KEEPALIVE_TEST_CONSOLE = '1'
}}
function global:Get-PnpDevice {{ return @() }}
& '{worker}' -SessionId 2 -SourceNetworkAddress '{source_address}'
""",
        encoding="utf-8",
    )
    powershell = shutil.which("powershell.exe")
    assert powershell is not None
    proc = subprocess.run(  # noqa: S603 - generated harness and executable are controlled
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(harness)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return log.read_text(encoding="utf-8-sig"), marker.exists()


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
    assert "Event/UserData/EventXML/Address" in text
    assert '-SessionId "$(SessionID)"' in text
    assert '-SourceNetworkAddress "$(SourceNetworkAddress)"' in text


def test_worker_redirects_network_disconnects_without_settle_delay() -> None:
    worker = _worker_text()

    assert "[System.Net.IPAddress]::TryParse" in worker
    assert "[System.Net.IPAddress]::IsLoopback" in worker
    assert "redirecting immediately" in worker
    assert "Start-Sleep -Seconds 10" not in worker


def test_worker_ignores_local_and_unparseable_disconnect_sources() -> None:
    worker = _worker_text()

    validation = worker.index("[System.Net.IPAddress]::TryParse")
    ignore = worker.index("Ignoring Event 24")
    redirect = worker.index("redirecting immediately")
    assert validation < ignore < redirect


@pytest.mark.parametrize("source_address", ["LOCAL", "本地", "", "not-an-address", "127.0.0.1"])
def test_worker_does_not_redirect_non_remote_events(
    tmp_path: Path,
    source_address: str,
) -> None:
    log, redirected = _run_worker(tmp_path, source_address)

    assert not redirected
    assert "Ignoring Event 24" in log


def test_worker_immediately_redirects_remote_event(tmp_path: Path) -> None:
    log, redirected = _run_worker(tmp_path, "192.0.2.10")

    assert redirected
    assert "Remote disconnect from 192.0.2.10" in log
    assert "Session 2 verified on console" in log


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
