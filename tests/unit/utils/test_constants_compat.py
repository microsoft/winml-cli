# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility coverage for constants moved to the session package."""

from __future__ import annotations

import warnings

from winml.modelkit.utils import constants


def _load_compat_constant(name: str):
    constants.__dict__.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        value = getattr(constants, name)
    deprecations = [warning for warning in caught if warning.category is DeprecationWarning]
    assert len(deprecations) == 1
    return value


def test_ep_name_to_alias_remains_available() -> None:
    mapping = _load_compat_constant("EP_NAME_TO_ALIAS")

    assert mapping["QNNExecutionProvider"] == "qnn"
    assert mapping["NvTensorRTRTXExecutionProvider"] == "nv_tensorrt_rtx"


def test_legacy_device_maps_keep_uppercase_string_contract() -> None:
    import onnxruntime as ort

    to_type = _load_compat_constant("DEVICE_TO_DEVICE_TYPE")
    from_type = _load_compat_constant("DEVICE_TYPE_TO_DEVICE")

    assert to_type == {
        "CPU": ort.OrtHardwareDeviceType.CPU,
        "GPU": ort.OrtHardwareDeviceType.GPU,
        "NPU": ort.OrtHardwareDeviceType.NPU,
    }
    assert from_type == {
        ort.OrtHardwareDeviceType.CPU: "CPU",
        ort.OrtHardwareDeviceType.GPU: "GPU",
        ort.OrtHardwareDeviceType.NPU: "NPU",
    }


def test_wildcard_import_includes_legacy_constants() -> None:
    names = {
        "EP_NAME_TO_ALIAS",
        "DEVICE_TO_DEVICE_TYPE",
        "DEVICE_TYPE_TO_DEVICE",
    }
    for name in names:
        constants.__dict__.pop(name, None)

    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec("from winml.modelkit.utils.constants import *", namespace)  # noqa: S102

    assert names <= namespace.keys()
