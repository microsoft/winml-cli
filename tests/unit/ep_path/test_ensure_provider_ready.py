# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the EP-download progress helpers in ``ep_path``.

Covers ``_ensure_provider_ready`` (the tqdm-backed download UX driven by
``ensure_ready_async``), ``_parse_ep_metadata_from_path`` and
``_ep_download_timeout_default``.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest

from winml.modelkit import ep_path


def _install_fake_tqdm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``tqdm.tqdm``. Helper writes ``bar.n`` directly + refresh()."""
    fake_bar = MagicMock()
    fake_bar.n = 0

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = MagicMock(return_value=fake_bar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)
    return fake_bar


def test_ensure_provider_ready_drives_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The download goes through ensure_ready_async; on_progress drives a tqdm bar."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        # Simulate cumulative-fraction progress callbacks, then completion.
        for fraction in (0.0, 0.25, 0.5, 1.0):
            on_progress(fraction)
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    provider.ensure_ready_async.assert_called_once()
    op.get_status.assert_called_once_with()
    op.cancel.assert_not_called()
    op.close.assert_called_once_with()
    fake_bar.close.assert_called_once_with()
    # Success path forces bar.n to 100 even though the last fraction was 1.0.
    assert fake_bar.n == 100
    fake_bar.refresh.assert_called()


def test_ensure_provider_ready_ignores_stale_progress_after_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale on_progress fired after on_complete must NOT clobber bar.n.

    The native layer can fire a late progress callback from its own thread
    after on_complete; once `done` is set, the main thread owns bar.n and the
    callback should drop instead of writing an earlier fraction back over the
    forced 100 (or worse, writing to an already-closed bar)."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()
    saved_progress: list = []

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_progress(0.5)
        on_complete()
        saved_progress.append(on_progress)  # Fire a stale callback after.
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    # Fire the stale callback as if it arrived after _ensure_provider_ready
    # already set bar.n = 100 and closed the bar.
    assert fake_bar.n == 100
    saved_progress[0](0.62)
    # Stale callback must have been dropped — bar.n stays at 100.
    assert fake_bar.n == 100


def test_ensure_provider_ready_warns_before_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A notice is printed to the stderr Console before download
    so users know the wait is expected."""
    _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "[WinML] Installing Execution Provider" in err
    assert "FakeEP" in err


def test_ensure_provider_ready_suppresses_status_and_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Structured-output callers can hide all installation status."""
    fake_tqdm = MagicMock()
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = fake_tqdm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_progress(0.5)
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.version = "1.0.0"
    provider.package_family_name = "FakeEP_123"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    with ep_path._suppress_ep_install_status():
        ep_path._ensure_provider_ready(provider)

    assert capsys.readouterr().err == ""
    fake_tqdm.assert_not_called()
    op.get_status.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_forces_bar_to_100_on_success_without_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the async op completes successfully without ever firing on_progress,
    the success path forces the bar to 100 so the final render shows full."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()  # No progress firings, but completes immediately.
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    assert fake_bar.n == 100
    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_times_out_and_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When on_complete never fires within the timeout, cancel and raise TimeoutError."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    # Shrink the timeout so the test runs in milliseconds, not minutes.
    monkeypatch.setattr(ep_path, "EP_DOWNLOAD_TIMEOUT_SECONDS", 0.05)

    op = MagicMock()
    provider = MagicMock()
    provider.name = "SlowEP"
    # ensure_ready_async returns op but never calls on_complete -> times out.
    provider.ensure_ready_async.return_value = op

    with pytest.raises(TimeoutError, match="SlowEP"):
        ep_path._ensure_provider_ready(provider)

    op.cancel.assert_called_once_with()
    op.close.assert_called_once_with()
    fake_bar.close.assert_called_once_with()
    # Bar must NOT be force-filled on timeout — it should reflect where the
    # download stalled (here: 0 because no progress callbacks ever fired).
    assert fake_bar.n == 0


def test_ensure_provider_ready_surfaces_get_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure surfaced by get_status() propagates after cleanup."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()
    op.get_status.side_effect = OSError("native error")

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    with pytest.raises(OSError, match="native error"):
        ep_path._ensure_provider_ready(provider)

    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()
    # Native error must NOT force-fill the bar — it should reflect where
    # the download failed (here: 0 because no progress callbacks fired).
    assert fake_bar.n == 0


def test_ensure_provider_ready_prints_success_with_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a successful install, print '<EP> EP installed successfully.'
    followed by Version and Package Family Name lines."""
    _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "OpenVINOExecutionProvider"
    provider.version = "1.2.0"
    provider.package_family_name = "Microsoft.OpenVINOExecutionProvider_8wekyb3d8bbwe"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "OpenVINOExecutionProvider EP installed successfully." in err
    assert "- Version: 1.2.0" in err
    assert "- Package Family Name: Microsoft.OpenVINOExecutionProvider_8wekyb3d8bbwe" in err


def test_ensure_provider_ready_falls_back_to_path_metadata_when_native_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the native handle reports empty version/PFN, recover both from the
    MSIX install path."""
    _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "OpenVINOExecutionProvider"
    provider.version = ""
    provider.package_family_name = ""
    provider.library_path = (
        r"C:\Program Files\WindowsApps"
        r"\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.79.0_x64__8wekyb3d8bbwe"
        r"\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll"
    )
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "- Version: 1.8.79.0" in err
    assert (
        "- Package Family Name: MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_8wekyb3d8bbwe"
        in err
    )


def test_ensure_provider_ready_skips_metadata_lines_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If neither the native handle nor the path yields metadata, omit the
    Version / Package Family Name lines entirely (no blank fields)."""
    _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "MysteryEP"
    provider.version = ""
    provider.package_family_name = ""
    provider.library_path = r"C:\some\local\path\provider.dll"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "MysteryEP EP installed successfully." in err
    assert "- Version:" not in err
    assert "- Package Family Name:" not in err


def test_ensure_provider_ready_prints_failure_message_on_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A timed-out download prints the ❌ failure notice with retry hints,
    and does NOT emit the 'installed successfully.' line."""
    _install_fake_tqdm(monkeypatch)
    monkeypatch.setattr(ep_path, "EP_DOWNLOAD_TIMEOUT_SECONDS", 0.05)

    provider = MagicMock()
    provider.name = "SlowEP"
    provider.ensure_ready_async.return_value = MagicMock()

    with pytest.raises(TimeoutError):
        ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "installed successfully" not in err
    assert "Failed to download SlowEP EP" in err
    assert "Check your internet connection" in err
    assert "Troubleshoot:" in err
    assert "https://aka.ms/winmlcli/ep-errors" in err


def test_ensure_provider_ready_prints_failure_message_on_async_launch_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failure at the ensure_ready_async() launch also prints the ❌ block."""
    _install_fake_tqdm(monkeypatch)

    provider = MagicMock()
    provider.name = "BadEP"
    provider.ensure_ready_async.side_effect = OSError("native launch failed")

    with pytest.raises(OSError, match="native launch failed"):
        ep_path._ensure_provider_ready(provider)

    err = capsys.readouterr().err
    assert "Failed to download BadEP EP" in err
    assert "installed successfully" not in err


def test_ensure_provider_ready_closes_bar_when_ensure_ready_async_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ensure_ready_async itself raises, the bar must still be closed
    (op was never assigned, so op.close() is skipped via the None sentinel)."""
    fake_bar = _install_fake_tqdm(monkeypatch)

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = RuntimeError("native init failed")

    with pytest.raises(RuntimeError, match="native init failed"):
        ep_path._ensure_provider_ready(provider)

    fake_bar.close.assert_called_once_with()


def test_ensure_provider_ready_works_without_tqdm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tqdm (a dev-only optional dep) is missing, the download still
    completes via the _NoopBar fallback — no ImportError, no progress UI."""
    # Simulate tqdm being uninstalled: make `from tqdm import tqdm` raise.
    monkeypatch.setitem(sys.modules, "tqdm", None)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        # Drive a progress update too — the no-op bar must tolerate bar.n = ...
        on_progress(0.5)
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_path._ensure_provider_ready(provider)

    op.get_status.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_survives_progress_render_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tqdm rendering I/O error mid-download (e.g. stderr closed) must not
    abort the install: the bar degrades to a no-op and readiness still completes."""
    fake_bar = _install_fake_tqdm(monkeypatch)
    # Bar construction succeeds, but every render write fails from here on.
    fake_bar.refresh.side_effect = OSError("stderr closed")
    fake_bar.close.side_effect = OSError("stderr closed")

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_progress(0.5)  # triggers bar.refresh() -> raises -> degrades to no-op
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.version = ""
    provider.package_family_name = ""
    provider.library_path = r"C:\some\local\path\provider.dll"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    # Must NOT raise despite the render/close I/O failures.
    ep_path._ensure_provider_ready(provider)

    op.get_status.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_survives_console_sink_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing rich Console sink (e.g. stderr closed) must not abort the install.

    The install/success/metadata prints run inside ``WinMLCatalogSource.resolve``,
    where any raised exception drops the EP. Status rendering must degrade to a
    no-op so a provider that is actually Ready is still surfaced."""
    _install_fake_tqdm(monkeypatch)

    # Every console.print raises, as if stderr were closed/unwritable.
    failing_console = MagicMock()
    failing_console.print.side_effect = ValueError("I/O operation on closed file")
    monkeypatch.setattr("winml.modelkit.utils.console.get_console", lambda: failing_console)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_progress(0.5)
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.version = "1.2.3"
    provider.package_family_name = "Some.Pkg_hash"
    provider.library_path = r"C:\some\local\path\provider.dll"
    provider.ensure_ready_async.side_effect = fake_ensure_async

    # Must NOT raise despite every console.print failing.
    ep_path._ensure_provider_ready(provider)

    op.get_status.assert_called_once_with()
    op.close.assert_called_once_with()
    # The status prints were attempted (install notice, success, metadata) and
    # every failure was swallowed rather than propagated.
    assert failing_console.print.call_count >= 1


class TestParseEpMetadataFromPath:
    """`_parse_ep_metadata_from_path` recovers (version, PFN) from install paths."""

    def test_parses_openvino_windowsapps_path(self) -> None:
        path = (
            r"C:\Program Files\WindowsApps"
            r"\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.79.0_x64__8wekyb3d8bbwe"
            r"\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll"
        )
        version, pfn = ep_path._parse_ep_metadata_from_path(path)
        assert version == "1.8.79.0"
        assert pfn == "MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_8wekyb3d8bbwe"

    def test_empty_path_returns_empty(self) -> None:
        assert ep_path._parse_ep_metadata_from_path("") == ("", "")

    def test_non_windowsapps_path_returns_empty(self) -> None:
        assert ep_path._parse_ep_metadata_from_path(r"C:\local\ep\provider.dll") == ("", "")

    def test_non_numeric_version_segment_dropped_but_pfn_kept(self) -> None:
        """A folder that doesn't carry a dotted-numeric version still yields a
        PFN, but the version is left empty rather than guessed."""
        path = r"C:\Program Files\WindowsApps\Some.Package_notaversion_x64__pubhash\ep.dll"
        version, pfn = ep_path._parse_ep_metadata_from_path(path)
        assert version == ""
        assert pfn == "Some.Package_pubhash"


class TestEpDownloadTimeoutDefault:
    """`_ep_download_timeout_default` reads ``WINMLCLI_EP_DOWNLOAD_TIMEOUT``."""

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINMLCLI_EP_DOWNLOAD_TIMEOUT", raising=False)
        assert ep_path._ep_download_timeout_default() == 5 * 60

    def test_override_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINMLCLI_EP_DOWNLOAD_TIMEOUT", "1800")
        assert ep_path._ep_download_timeout_default() == 1800

    def test_falls_back_to_default_on_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("WINMLCLI_EP_DOWNLOAD_TIMEOUT", "not-a-number")
        with caplog.at_level(logging.WARNING, logger=ep_path.logger.name):
            assert ep_path._ep_download_timeout_default() == 5 * 60
        assert any("WINMLCLI_EP_DOWNLOAD_TIMEOUT" in r.getMessage() for r in caplog.records)

    def test_empty_string_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINMLCLI_EP_DOWNLOAD_TIMEOUT", "")
        assert ep_path._ep_download_timeout_default() == 5 * 60
