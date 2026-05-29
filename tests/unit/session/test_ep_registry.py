# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module helpers."""

from __future__ import annotations

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

    fake_bar = MagicMock()
    fake_bar.n = 0

    def fake_update(delta: float) -> None:
        fake_bar.n += delta

    fake_bar.update.side_effect = fake_update

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = MagicMock(return_value=fake_bar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    op = MagicMock()

    def fake_ensure_async(on_progress=None, **_kwargs):
        # Simulate cumulative-fraction progress callbacks (0.0..1.0).
        for fraction in (0.0, 0.25, 0.5, 1.0):
            on_progress(fraction)
        return op

    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotPresent
    provider.ensure_ready_async.side_effect = fake_ensure_async

    ep_registry._ensure_provider_ready(provider)

    provider.ensure_ready.assert_not_called()
    provider.ensure_ready_async.assert_called_once()
    op.wait.assert_called_once_with()
    op.close.assert_called_once_with()
    fake_bar.close.assert_called_once_with()
    # All deltas combined must reach the bar's total (100).
    assert fake_bar.n == 100


def test_ensure_provider_ready_finalizes_bar_when_no_progress_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the async op completes without firing on_progress, the bar is still
    advanced to 100 in the finally block so users see completion."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)

    fake_bar = MagicMock()
    fake_bar.n = 0

    def fake_update(delta: float) -> None:
        fake_bar.n += delta

    fake_bar.update.side_effect = fake_update

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = MagicMock(return_value=fake_bar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    op = MagicMock()
    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotReady
    provider.ensure_ready_async.return_value = op  # No on_progress firings.

    ep_registry._ensure_provider_ready(provider)

    assert fake_bar.n == 100
    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()


def test_ensure_provider_ready_closes_bar_on_wait_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait failures propagate but the bar and async op are still closed."""
    from winml.modelkit.session import ep_registry

    ns = _install_fake_windowsml(monkeypatch)

    fake_bar = MagicMock()
    fake_bar.n = 0
    fake_bar.update.side_effect = lambda d: setattr(fake_bar, "n", fake_bar.n + d)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = MagicMock(return_value=fake_bar)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    op = MagicMock()
    op.wait.side_effect = RuntimeError("network down")
    provider = MagicMock()
    provider.name = "FakeEP"
    provider.ready_state = ns.EpReadyState.NotPresent
    provider.ensure_ready_async.return_value = op

    with pytest.raises(RuntimeError, match="network down"):
        ep_registry._ensure_provider_ready(provider)

    fake_bar.close.assert_called_once_with()
    op.close.assert_called_once_with()
