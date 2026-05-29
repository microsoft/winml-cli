# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module helpers."""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_windowsml(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Inject a fake ``windowsml`` module exposing only what the helper needs."""

    class _EpReadyState:
        Ready = 0
        NotReady = 1
        NotPresent = 2

    fake = types.ModuleType("windowsml")
    fake.EpReadyState = _EpReadyState  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windowsml", fake)
    return types.SimpleNamespace(EpReadyState=_EpReadyState)


def _install_fake_tqdm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``tqdm.tqdm`` whose ``n`` advances with ``update`` deltas."""

    fake_bar = MagicMock()
    fake_bar.n = 0
    fake_bar.update.side_effect = lambda d: setattr(fake_bar, "n", fake_bar.n + d)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = MagicMock(return_value=fake_bar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)
    return fake_bar


def test_ensure_provider_ready_skips_progress_when_already_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ready providers take the sync fast path and skip the async/progress flow."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
    provider = MagicMock()
    provider.ready_state = ns.EpReadyState.Ready

    ep_registry._ensure_provider_ready(provider)

    provider.ensure_ready.assert_called_once_with()
    provider.ensure_ready_async.assert_not_called()


def test_ensure_provider_ready_drives_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """NotReady providers go through ensure_ready_async; on_progress drives a tqdm bar."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
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
    provider.ready_state = ns.EpReadyState.NotPresent
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_registry._ensure_provider_ready(provider)

    provider.ensure_ready.assert_not_called()
    provider.ensure_ready_async.assert_called_once()
    op.get_status.assert_called_once_with()
    op.cancel.assert_not_called()
    op.close.assert_called_once_with()
    fake_bar.close.assert_called_once_with()
    assert fake_bar.n == 100


def test_ensure_provider_ready_warns_before_download(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A WARNING log is emitted before download so users know the wait is expected."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
    _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotPresent
    provider.ensure_ready_async.side_effect = fake_ensure_async

    with caplog.at_level(logging.WARNING, logger=ep_registry.logger.name):
        ep_registry._ensure_provider_ready(provider)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Downloading execution provider" in r.getMessage() for r in warnings)
    assert any("FakeEP" in r.getMessage() for r in warnings)


def test_ensure_provider_ready_finalizes_bar_when_no_progress_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the async op completes without firing on_progress, the bar is still
    advanced to 100 in the finally block so users see completion."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()  # No progress firings, but completes immediately.
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotReady
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_registry._ensure_provider_ready(provider)

    assert fake_bar.n == 100
    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_times_out_and_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When on_complete never fires within the timeout, cancel and raise TimeoutError."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
    fake_bar = _install_fake_tqdm(monkeypatch)

    # Shrink the timeout so the test runs in milliseconds, not minutes.
    monkeypatch.setattr(ep_registry, "EP_DOWNLOAD_TIMEOUT_SECONDS", 0.05)

    op = MagicMock()
    provider = MagicMock()
    provider.name = "SlowEP"
    provider.ready_state = ns.EpReadyState.NotPresent
    # ensure_ready_async returns op but never calls on_complete -> times out.
    provider.ensure_ready_async.return_value = op

    with pytest.raises(TimeoutError, match="SlowEP"):
        ep_registry._ensure_provider_ready(provider)

    op.cancel.assert_called_once_with()
    op.close.assert_called_once_with()
    fake_bar.close.assert_called_once_with()


def test_ensure_provider_ready_surfaces_get_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure surfaced by get_status() propagates after cleanup."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)
    fake_bar = _install_fake_tqdm(monkeypatch)

    op = MagicMock()
    op.get_status.side_effect = OSError("native error")

    def fake_ensure_async(on_complete=None, on_progress=None):
        on_complete()
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotPresent
    provider.ensure_ready_async.side_effect = fake_ensure_async

    with pytest.raises(OSError, match="native error"):
        ep_registry._ensure_provider_ready(provider)

    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()
