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

    from winml.modelkit.session.ep_device import DEVICE_TO_DEVICE_TYPE

    assert set(DEVICE_TO_DEVICE_TYPE.keys()) == {"cpu", "gpu", "npu"}
    assert DEVICE_TO_DEVICE_TYPE["cpu"] is ort.OrtHardwareDeviceType.CPU
    assert DEVICE_TO_DEVICE_TYPE["gpu"] is ort.OrtHardwareDeviceType.GPU
    assert DEVICE_TO_DEVICE_TYPE["npu"] is ort.OrtHardwareDeviceType.NPU


def test_device_type_to_device_lives_in_ep_device_with_lowercase_values() -> None:
    """``DEVICE_TYPE_TO_DEVICE`` returns lowercase device short names."""
    import onnxruntime as ort

    from winml.modelkit.session.ep_device import DEVICE_TYPE_TO_DEVICE

    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.CPU] == "cpu"
    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.GPU] == "gpu"
    assert DEVICE_TYPE_TO_DEVICE[ort.OrtHardwareDeviceType.NPU] == "npu"


def test_maps_are_inverses() -> None:
    """Round-trip: ``DEVICE_TYPE_TO_DEVICE[DEVICE_TO_DEVICE_TYPE[k]] == k``."""
    from winml.modelkit.session.ep_device import (
        DEVICE_TO_DEVICE_TYPE,
        DEVICE_TYPE_TO_DEVICE,
    )

    for k in ("cpu", "gpu", "npu"):
        assert DEVICE_TYPE_TO_DEVICE[DEVICE_TO_DEVICE_TYPE[k]] == k


def test_utils_constants_module_is_gone() -> None:
    """``utils.constants`` is deleted; its surviving symbols moved.

    ``normalize_ep_name`` / ``extract_ep_options`` migrated to
    ``utils.cli``; the device-type maps moved to ``session.ep_device``.
    Importing the old path is the canonical RED signal.
    """
    import pytest

    with pytest.raises(ImportError):
        import winml.modelkit.utils.constants  # noqa: F401
