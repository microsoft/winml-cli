# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``winml.add_ep_for_device``.

``add_ep_for_device`` is exact-match by design — the docstring on the
function explicitly says "no alias normalization layer; callers must
pass the spelling ORT registers under." These tests pin that contract:
canonical camelCase input binds; mismatched device type does not bind;
unknown spelling does not bind silently.

(An earlier draft attempted PascalCase ⇄ camelCase aliasing for NVIDIA's
public-docs spelling ``NvTensorRTRTXExecutionProvider``. That alias
layer was intentionally removed; callers normalize at their own layer
or use the canonical name directly.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import onnxruntime as ort
import pytest  # noqa: TC002 — used as runtime fixture-arg type via monkeypatch

from winml.modelkit.winml import add_ep_for_device


# ---------------------------------------------------------------------------
# Lightweight fakes for ORT EP-device discovery + SessionOptions sink.
# ---------------------------------------------------------------------------


@dataclass
class _FakeDevice:
    """Stand-in for ``OrtHardwareDevice`` exposing only ``type``."""

    type: Any


@dataclass
class _FakeEpDevice:
    """Stand-in for ``OrtEpDevice`` exposing ``ep_name`` and ``device``."""

    ep_name: str
    device: _FakeDevice


@dataclass
class _RecordingSessionOptions:
    """Captures ``add_provider_for_devices`` calls for assertion."""

    calls: list[tuple[list[_FakeEpDevice], dict]] = field(default_factory=list)

    def add_provider_for_devices(self, ep_devices: list[_FakeEpDevice], options: dict) -> None:
        self.calls.append((list(ep_devices), dict(options)))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestAddEpForDeviceExactMatch:
    """``add_ep_for_device`` is exact-match by canonical EP name (no aliasing)."""

    def test_canonical_input_binds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Callers pass the exact canonical spelling that ORT registers.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(ep_name="NvTensorRTRTXExecutionProvider", device=gpu)
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(opts, "NvTensorRTRTXExecutionProvider", ort.OrtHardwareDeviceType.GPU)

        assert len(opts.calls) == 1
        bound_devices, bound_options = opts.calls[0]
        assert bound_devices == [registered]
        assert bound_options == {}

    def test_device_type_mismatch_does_not_bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Device-type guard: a GPU EP asked for on the NPU does not bind
        # even when the EP-name string is canonical.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(ep_name="NvTensorRTRTXExecutionProvider", device=gpu)
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(opts, "NvTensorRTRTXExecutionProvider", ort.OrtHardwareDeviceType.NPU)

        assert opts.calls == []

    def test_ep_options_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ep_options dict reaches add_provider_for_devices unchanged.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(ep_name="NvTensorRTRTXExecutionProvider", device=gpu)
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()
        ep_options = {"opt_level": "all"}

        add_ep_for_device(
            opts,
            "NvTensorRTRTXExecutionProvider",
            ort.OrtHardwareDeviceType.GPU,
            ep_options=ep_options,
        )

        assert len(opts.calls) == 1
        _, bound_options = opts.calls[0]
        assert bound_options == ep_options

    def test_unknown_ep_name_does_not_bind_silently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Defensive default: an unknown EP name (typo, unsupported EP)
        # is NOT rewritten by the alias table — it falls through to ORT
        # and simply fails to match.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(ep_name="NvTensorRTRTXExecutionProvider", device=gpu)
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(opts, "TotallyMadeUpExecutionProvider", ort.OrtHardwareDeviceType.GPU)

        assert opts.calls == []
