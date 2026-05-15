# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``winml.add_ep_for_device``.

Covers EP-name alias canonicalization at the user-facing boundary: the
WinML EP Catalog registers NVIDIA's TensorRT-RTX EP under the camelCase
spelling ``NvTensorRtRtxExecutionProvider``, while NVIDIA's own public
docs use PascalCase ``NvTensorRTRTXExecutionProvider``. Both spellings
must bind successfully.
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

    def add_provider_for_devices(
        self, ep_devices: list[_FakeEpDevice], options: dict
    ) -> None:
        self.calls.append((list(ep_devices), dict(options)))


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class TestAddEpForDevicePascalCase:
    """The bug: PascalCase user input + camelCase ORT registration must bind."""

    def test_pascalcase_input_matches_camelcase_registration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ORT reports the EP under the camelCase spelling that the WinML
        # ``ExecutionProviderCatalog`` registered it with.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(
            ep_name="NvTensorRtRtxExecutionProvider", device=gpu
        )
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        # User passes the PascalCase spelling that NVIDIA's docs (and
        # several places in our own codebase) use.
        add_ep_for_device(
            opts, "NvTensorRTRTXExecutionProvider", ort.OrtHardwareDeviceType.GPU
        )

        # The fix routed the call to add_provider_for_devices despite
        # the case mismatch.
        assert len(opts.calls) == 1
        bound_devices, bound_options = opts.calls[0]
        assert bound_devices == [registered]
        assert bound_options == {}

    def test_camelcase_input_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Symmetric case: callers that already use the canonical
        # camelCase form must continue to work after the fix.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(
            ep_name="NvTensorRtRtxExecutionProvider", device=gpu
        )
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(
            opts, "NvTensorRtRtxExecutionProvider", ort.OrtHardwareDeviceType.GPU
        )

        assert len(opts.calls) == 1

    def test_device_type_mismatch_does_not_bind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The case-fix must not relax the device-type guard: a GPU EP
        # asked for on the NPU still does not bind.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(
            ep_name="NvTensorRtRtxExecutionProvider", device=gpu
        )
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(
            opts, "NvTensorRTRTXExecutionProvider", ort.OrtHardwareDeviceType.NPU
        )

        assert opts.calls == []

    def test_ep_options_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Sanity check that the fix did not drop the ep_options argument.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(
            ep_name="NvTensorRtRtxExecutionProvider", device=gpu
        )
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

    def test_unknown_ep_name_does_not_bind_silently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defensive default: an unknown EP name (typo, unsupported EP)
        # is NOT rewritten by the alias table — it falls through to ORT
        # and simply fails to match.
        gpu = _FakeDevice(type=ort.OrtHardwareDeviceType.GPU)
        registered = _FakeEpDevice(
            ep_name="NvTensorRtRtxExecutionProvider", device=gpu
        )
        monkeypatch.setattr(ort, "get_ep_devices", lambda: [registered])

        opts = _RecordingSessionOptions()

        add_ep_for_device(
            opts, "TotallyMadeUpExecutionProvider", ort.OrtHardwareDeviceType.GPU
        )

        assert opts.calls == []
