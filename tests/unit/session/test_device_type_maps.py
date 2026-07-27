# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""T-16 contract: ``DEVICE_TO_DEVICE_TYPE`` / ``DEVICE_TYPE_TO_DEVICE`` live in
``session.ep_device`` with **lowercase** keys/values, matching the rest of
the session taxonomy.

Previously the maps lived in ``utils.constants`` with uppercase keys
(``"CPU"``, ``"GPU"``, ``"NPU"``), creating a silent casing-mismatch
footgun whenever a lowercase device string from ``VALID_DEVICES`` flowed
to those lookups. T-16 unifies on lowercase across the entire taxonomy.
"""

from __future__ import annotations


def test_device_to_device_type_lives_in_ep_device_with_lowercase_keys() -> None:
    """``DEVICE_TO_DEVICE_TYPE`` is exported by ``session.ep_device`` with
    lowercase short-device keys.
    """
    import onnxruntime as ort

    from winml.modelkit.session import DEVICE_TO_DEVICE_TYPE

    assert set(DEVICE_TO_DEVICE_TYPE.keys()) == {"cpu", "gpu", "npu"}
    assert DEVICE_TO_DEVICE_TYPE["cpu"] is ort.OrtHardwareDeviceType.CPU
    assert DEVICE_TO_DEVICE_TYPE["gpu"] is ort.OrtHardwareDeviceType.GPU
    assert DEVICE_TO_DEVICE_TYPE["npu"] is ort.OrtHardwareDeviceType.NPU


def test_device_type_to_device_lives_in_ep_device_with_lowercase_values() -> None:
    """``DEVICE_TYPE_TO_DEVICE`` returns lowercase device short names."""
    import onnxruntime as ort

    from winml.modelkit.session import DEVICE_TYPE_TO_DEVICE

    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.CPU] == "cpu"
    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.GPU] == "gpu"
    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.NPU] == "npu"


def test_maps_are_inverses() -> None:
    """Round-trip: ``DEVICE_TYPE_TO_DEVICE[DEVICE_TO_DEVICE_TYPE[k]] == k``."""
    from winml.modelkit.session import (
        DEVICE_TO_DEVICE_TYPE,
        DEVICE_TYPE_TO_DEVICE,
    )

    for k in ("cpu", "gpu", "npu"):
        assert DEVICE_TYPE_TO_DEVICE[DEVICE_TO_DEVICE_TYPE[k]] == k


def test_device_type_maps_are_authoritative_in_ep_device() -> None:
    """The device-type maps' single owner is ``session.ep_device`` — not
    ``utils.constants``. Whether or not ``utils.constants`` exists as a
    transitional surface is intentionally NOT asserted here; the T-16 contract
    is that ``session.ep_device`` is the source of truth for these maps.

    Assert positively: the canonical import path works and returns the T-16
    lowercase-keyed maps.
    """
    from winml.modelkit.session import DEVICE_TO_DEVICE_TYPE, DEVICE_TYPE_TO_DEVICE

    assert set(DEVICE_TO_DEVICE_TYPE.keys()) == {"cpu", "gpu", "npu"}
    assert set(DEVICE_TYPE_TO_DEVICE.values()) == {"cpu", "gpu", "npu"}
