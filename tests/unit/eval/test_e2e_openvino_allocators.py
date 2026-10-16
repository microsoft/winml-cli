# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Verify that the E2E workaround is scoped and preserves session overrides."""

from types import SimpleNamespace
from unittest.mock import Mock

import onnxruntime as ort
import pytest

from scripts.e2e_eval.winml_cli import openvino_gpu_allocators
from winml.modelkit.session import session as session_module


@pytest.mark.parametrize(
    ("ep", "device", "expected"),
    [
        ("OpenVINOExecutionProvider", "GPU", "1"),
        ("OpenVINOExecutionProvider", "CPU", None),
        ("OpenVINOExecutionProvider", "NPU", None),
        ("DmlExecutionProvider", "GPU", None),
    ],
)
@pytest.mark.parametrize("explicit", [None, "0", "1"])
def test_scoped_allocator_default(monkeypatch, ep, device, expected, explicit):
    options = ort.SessionOptions()
    options.add_session_config_entry("test.other_option", "preserved")
    if explicit is not None:
        options.add_session_config_entry("session.use_env_allocators", explicit)
    original = Mock(return_value=options)
    monkeypatch.setattr(session_module, "_build_session_options", original)
    handle = SimpleNamespace(ep_name=ep, device=SimpleNamespace(type=SimpleNamespace(name=device)))
    selected = SimpleNamespace(device=SimpleNamespace(ort_handle=handle))
    with openvino_gpu_allocators():
        actual = session_module._build_session_options(selected, provider_options={"test": "value"})
        assert actual is options
        assert actual.get_session_config_entry("test.other_option") == "preserved"
        expected = explicit if explicit is not None else expected
        if expected is None:
            with pytest.raises(RuntimeError):
                actual.get_session_config_entry("session.use_env_allocators")
        else:
            assert actual.get_session_config_entry("session.use_env_allocators") == expected
    original.assert_called_once_with(selected, provider_options={"test": "value"})
    assert session_module._build_session_options is original


def test_restores_product_builder_after_failure(monkeypatch):
    original = Mock(side_effect=ValueError("original failure"))
    monkeypatch.setattr(session_module, "_build_session_options", original)
    with pytest.raises(ValueError, match="original failure"), openvino_gpu_allocators():
        session_module._build_session_options(None)
    assert session_module._build_session_options is original
