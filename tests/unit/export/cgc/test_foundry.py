# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for FoundryToolbox discovery."""

from __future__ import annotations

import ctypes
from importlib import metadata
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from winml.modelkit.export.cgc.foundry import (
    FoundryCompileError,
    FoundryCompiler,
    FoundryToolboxUnavailableError,
    _FdyOverrideDynamicDimsByDimNamePassDescriptor,
    find_foundry_toolbox,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_find_foundry_toolbox_in_windowsml_wheel(tmp_path: Path) -> None:
    package_dir = tmp_path / "windowsml"
    dll_path = package_dir / "lib" / "FoundryToolbox.dll"
    dll_path.parent.mkdir(parents=True)
    dll_path.write_bytes(b"dll")
    distribution = Mock(version="2.6.10.dev0")
    distribution.locate_file.return_value = dll_path

    with patch(
        "winml.modelkit.export.cgc.foundry.metadata.distribution",
        return_value=distribution,
    ):
        assert find_foundry_toolbox() == dll_path.resolve()
    distribution.locate_file.assert_called_once_with(
        "windowsml/lib/FoundryToolbox.dll"
    )


def test_find_foundry_toolbox_requires_windowsml_distribution() -> None:
    with (
        patch(
            "winml.modelkit.export.cgc.foundry.metadata.distribution",
            side_effect=metadata.PackageNotFoundError,
        ),
        pytest.raises(
            FoundryToolboxUnavailableError,
            match="requires a windowsml wheel",
        ),
    ):
        find_foundry_toolbox()


def test_find_foundry_toolbox_requires_dll_in_wheel(tmp_path: Path) -> None:
    distribution = Mock(version="2.6.8.dev0")
    distribution.locate_file.return_value = (
        tmp_path / "windowsml" / "lib" / "FoundryToolbox.dll"
    )

    with (
        patch(
            "winml.modelkit.export.cgc.foundry.metadata.distribution",
            return_value=distribution,
        ),
        pytest.raises(
            FoundryToolboxUnavailableError,
            match=r"windowsml wheel \(2\.6\.8\.dev0\) does not contain",
        ),
    ):
        find_foundry_toolbox()


def _compiler_with_diagnostics(
    *,
    message: bytes,
    unsupported_op: bytes | None = None,
    missing_external_data: bytes | None = None,
) -> FoundryCompiler:
    compiler = object.__new__(FoundryCompiler)
    compiler._compiler = object()
    compiler._dll = Mock(
        FdyCompilerGetLastError=Mock(return_value=message),
        FdyCompilerGetLastUnsupportedOpName=Mock(return_value=unsupported_op),
        FdyCompilerGetLastMissingExternalDataFile=Mock(
            return_value=missing_external_data
        ),
    )
    return compiler


def test_foundry_error_classifies_result_and_preserves_native_message() -> None:
    compiler = _compiler_with_diagnostics(
        message=b"Could not infer output shape.",
    )

    error = compiler._last_error(3)

    assert isinstance(error, FoundryCompileError)
    assert error.result_code == 3
    assert error.result_name == "SHAPE_INFERENCE"
    assert error.native_message == "Could not infer output shape."
    assert str(error) == (
        "Foundry compiler failed [SHAPE_INFERENCE]: "
        "Could not infer output shape."
    )


def test_foundry_error_reports_first_unsupported_operator() -> None:
    compiler = _compiler_with_diagnostics(
        message=b"Operation is not registered.",
        unsupported_op=b"com.example.CustomOp",
    )

    error = compiler._last_error(9)

    assert error.unsupported_op == "com.example.CustomOp"
    assert error.result_name == "UNSUPPORTED_OP"
    assert "com.example.CustomOp" not in str(error)


def test_foundry_error_reports_missing_external_data_path() -> None:
    compiler = _compiler_with_diagnostics(
        message=b"External data file does not exist.",
        missing_external_data=b"weights/model.data",
    )

    error = compiler._last_error(11)

    assert error.missing_external_data == "weights/model.data"
    assert error.result_name == "MISSING_EXTERNAL_DATA"
    assert "weights/model.data" not in str(error)


def test_freeze_dims_populate_foundry_pass_descriptor(tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    payload = b"module { cgc.test }"

    class FakeDLL:
        @staticmethod
        def FdyCompilerCompile(  # noqa: N802
            _compiler, _source, options_pointer, module_pointer
        ):
            options = options_pointer._obj
            assert options.version == 4
            assert not options.safetensorsFiles
            assert options.safetensorsFileCount == 0
            assert options.passCount == 1
            descriptor = ctypes.cast(
                options.passes[0],
                ctypes.POINTER(_FdyOverrideDynamicDimsByDimNamePassDescriptor),
            ).contents
            observed["kind"] = descriptor.descriptor.kind
            assert descriptor.descriptor.stage == 0
            observed["names"] = tuple(
                descriptor.names[index].decode("utf-8")
                for index in range(descriptor.count)
            )
            observed["values"] = tuple(
                descriptor.values[index] for index in range(descriptor.count)
            )
            module_pointer._obj.value = 1
            return 0

        @staticmethod
        def FdyModuleSerialize(  # noqa: N802
            _module, _format, output, size_pointer
        ):
            size_pointer._obj.value = len(payload)
            if output.data:
                ctypes.memmove(output.data, payload, len(payload))
            return 0

        @staticmethod
        def FdyModuleDestroy(_module):  # noqa: N802
            return None

    compiler = object.__new__(FoundryCompiler)
    compiler._compiler = ctypes.c_void_p(1)
    compiler._dll = FakeDLL()

    result = compiler.compile_onnx(
        b"onnx",
        model_directory=tmp_path,
        update_opset=True,
        topo_sort_nodes=True,
        include_initializers=True,
        enable_lazy_external_data=False,
        freeze_dims={"batch": 1, "seq": 128},
    )

    assert result == payload
    assert observed == {
        "kind": 15,
        "names": ("batch", "seq"),
        "values": (1, 128),
    }
