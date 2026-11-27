# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compile-only off-device session lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import onnx
import onnxruntime as ort
import pytest

from tests.unit.compiler.test_compiler_stages import create_epcontext_onnx
from winml.modelkit.compiler import EPConfig
from winml.modelkit.session import SessionState, WinMLEPDevice, WinMLOffDeviceSession


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def offdevice_session(
    simple_matmul_onnx: Path,
    qnn_npu_ep_device: WinMLEPDevice,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[WinMLOffDeviceSession, Mock]]:
    """Provide a bound compiler host and a code-generated compiled artifact."""
    native_device = qnn_npu_ep_device.device.ort_handle
    native_device.device.type = ort.OrtHardwareDeviceType.CPU
    native_device.ep_metadata = {"library_path": str(qnn_npu_ep_device.dll_path)}

    def write_context(
        model_path: Path,
        output_path: Path,
        _target: str,
        **_kwargs: object,
    ) -> None:
        create_epcontext_onnx(output_path, model_path.read_bytes().hex(), embed_mode=1)
        model = onnx.load(output_path)
        model.graph.node[0].attribute.append(
            onnx.helper.make_attribute("source", qnn_npu_ep_device.ep.source.ep_name)
        )
        onnx.save(model, output_path)

    backend = Mock(side_effect=write_context)
    monkeypatch.setattr(
        "winml.modelkit.session.offdevice_session._compile_bound_model", backend
    )
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        Mock(side_effect=AssertionError("Inference is forbidden")),
    )
    monkeypatch.setattr(
        ort,
        "get_ep_devices",
        Mock(side_effect=AssertionError("Device rediscovery is forbidden")),
    )
    session = WinMLOffDeviceSession(
        simple_matmul_onnx,
        qnn_npu_ep_device,
        EPConfig(
            provider="qnn",
            compiler="offdevice",
            embed_context=True,
            target_platform="60",
        ),
    )
    yield session, backend
    session.reset()


def test_compile_is_deferred_idempotent_and_resettable(
    offdevice_session: tuple[WinMLOffDeviceSession, Mock],
) -> None:
    session, backend = offdevice_session
    backend.assert_not_called()
    assert session.state == SessionState.INITIALIZED

    session.compile()
    session.compile()
    backend.assert_called_once()
    output_path = session.running_model_path
    assert output_path.is_file()
    assert session.is_compiled
    assert session._session is None
    with pytest.raises(RuntimeError, match="do not run inference"):
        session.run({})

    session.reset()
    assert not output_path.exists()
    assert not session.is_compiled
    assert session.state == SessionState.INITIALIZED
    session.compile()
    assert backend.call_count == 2


def test_compile_failure_records_error_and_removes_private_output(
    offdevice_session: tuple[WinMLOffDeviceSession, Mock],
) -> None:
    session, backend = offdevice_session
    backend.side_effect = RuntimeError("Compilation failed")
    with pytest.raises(RuntimeError, match="Compilation failed"):
        session.compile()

    assert session.state == SessionState.ERROR
    assert not session.is_compiled
    assert session._session is None
    assert not backend.call_args.args[1].parent.exists()


def test_qnn_session_rejects_a_physical_npu_binding(
    simple_matmul_onnx: Path,
    qnn_npu_ep_device: WinMLEPDevice,
) -> None:
    with pytest.raises(ValueError, match="CPU compilation host"):
        WinMLOffDeviceSession(
            simple_matmul_onnx,
            qnn_npu_ep_device,
            EPConfig(
                provider="qnn",
                compiler="offdevice",
                embed_context=True,
                target_platform="60",
            ),
        )
