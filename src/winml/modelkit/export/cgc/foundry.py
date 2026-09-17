# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Lazy ctypes binding for the FoundryToolbox compiler C API."""

from __future__ import annotations

import ctypes
import os
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


FDY_VERSION = 0x00000004
FDY_SOURCE_FORMAT_ONNX_PROTOBUF = 0
FDY_COMPILER_TARGET_DXCGC = 1
FDY_SERIALIZATION_FORMAT_TEXT = 1
FDY_COMPILER_RESULT_SUCCESS = 0
FDY_PASS_OVERRIDE_DYNAMIC_DIMS_BY_DIM_NAME = 15
FDY_PASS_STAGE_BEFORE_LOWERING = 0

_RESULT_NAMES = {
    1: "INVALID_ARGUMENT",
    2: "PARSE",
    3: "SHAPE_INFERENCE",
    4: "IMPORT",
    5: "VERIFICATION",
    6: "LOWERING",
    7: "SERIALIZATION",
    8: "BUFFER_TOO_SMALL",
    9: "UNSUPPORTED_OP",
    10: "MALFORMED",
    11: "MISSING_EXTERNAL_DATA",
}


class FoundryToolboxUnavailableError(RuntimeError):
    """Raised when a compatible FoundryToolbox DLL cannot be located or loaded."""


class FoundryCompileError(RuntimeError):
    """Native Foundry failure with copied diagnostic fields."""

    def __init__(
        self,
        result_code: int,
        *,
        native_message: str = "",
        unsupported_op: str | None = None,
        missing_external_data: str | None = None,
    ) -> None:
        self.result_code = result_code
        self.result_name = _RESULT_NAMES.get(result_code, "UNKNOWN")
        self.native_message = native_message
        self.unsupported_op = unsupported_op
        self.missing_external_data = missing_external_data
        suffix = f": {native_message}" if native_message else ""
        super().__init__(f"Foundry compiler failed [{self.result_name}]{suffix}")


class _FdyStringView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_size_t)]


class _FdySpan(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _FdyMutableSpan(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_size_t)]


class _FdyPassDescriptor(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_uint32), ("stage", ctypes.c_uint32)]


class _FdyOverrideDynamicDimsByDimNamePassDescriptor(ctypes.Structure):
    _fields_ = [
        ("descriptor", _FdyPassDescriptor),
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("values", ctypes.POINTER(ctypes.c_int64)),
        ("count", ctypes.c_size_t),
    ]


class _FdyCompilerOptions(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("sourceFormat", ctypes.c_uint32),
        ("target", ctypes.c_uint32),
        ("updateOpset", ctypes.c_bool),
        ("topoSortNodes", ctypes.c_bool),
        ("includeInitializers", ctypes.c_bool),
        ("outputDataFile", _FdyStringView),
        ("passes", ctypes.POINTER(ctypes.POINTER(_FdyPassDescriptor))),
        ("passCount", ctypes.c_uint32),
        ("modelDirectory", _FdyStringView),
        ("enableLazyExternalData", ctypes.c_bool),
        ("safetensorsFiles", ctypes.POINTER(_FdyStringView)),
        ("safetensorsFileCount", ctypes.c_uint32),
    ]


def find_foundry_toolbox() -> Path:
    """Resolve FoundryToolbox from the installed windowsml wheel."""
    try:
        distribution = metadata.distribution("windowsml")
    except metadata.PackageNotFoundError as e:
        raise FoundryToolboxUnavailableError(
            "CGC export requires a windowsml wheel containing FoundryToolbox.dll."
        ) from e

    candidate = Path(
        distribution.locate_file("windowsml/lib/FoundryToolbox.dll")
    )
    if not candidate.is_file():
        raise FoundryToolboxUnavailableError(
            "The installed windowsml wheel "
            f"({distribution.version}) does not contain FoundryToolbox.dll."
        )
    return candidate.resolve()


def _string_view(value: bytes) -> _FdyStringView:
    return _FdyStringView(value or None, len(value))


class FoundryCompiler:
    """Thin owner for one Foundry compiler context."""

    def __init__(self) -> None:
        path = find_foundry_toolbox()
        self._dll_directory = None
        try:
            if hasattr(os, "add_dll_directory"):
                self._dll_directory = os.add_dll_directory(str(path.parent))
            self._dll = ctypes.CDLL(str(path))
        except OSError as e:
            if self._dll_directory is not None:
                self._dll_directory.close()
            raise FoundryToolboxUnavailableError(
                f"Unable to load FoundryToolbox.dll from '{path}': {e}"
            ) from e

        self._compiler = ctypes.c_void_p()
        try:
            self._configure_signatures()
            result = self._dll.FdyCompilerCreate(ctypes.byref(self._compiler))
        except Exception:
            self.close()
            raise
        if result != FDY_COMPILER_RESULT_SUCCESS:
            self.close()
            raise FoundryCompileError(
                result,
                native_message="Foundry compiler context creation failed.",
            )

    def _configure_signatures(self) -> None:
        dll = self._dll
        dll.FdyCompilerCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        dll.FdyCompilerCreate.restype = ctypes.c_uint32
        dll.FdyCompilerDestroy.argtypes = [ctypes.c_void_p]
        dll.FdyCompilerDestroy.restype = None
        dll.FdyCompilerCompile.argtypes = [
            ctypes.c_void_p,
            _FdySpan,
            ctypes.POINTER(_FdyCompilerOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.FdyCompilerCompile.restype = ctypes.c_uint32
        dll.FdyModuleSerialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            _FdyMutableSpan,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        dll.FdyModuleSerialize.restype = ctypes.c_uint32
        dll.FdyModuleDestroy.argtypes = [ctypes.c_void_p]
        dll.FdyModuleDestroy.restype = None
        for name in (
            "FdyCompilerGetLastError",
            "FdyCompilerGetLastUnsupportedOpName",
            "FdyCompilerGetLastMissingExternalDataFile",
        ):
            function = getattr(dll, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_char_p

    def close(self) -> None:
        """Release the native compiler context."""
        if getattr(self, "_compiler", None):
            self._dll.FdyCompilerDestroy(self._compiler)
            self._compiler = ctypes.c_void_p()
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None

    def __enter__(self) -> FoundryCompiler:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _last_error(self, result: int) -> FoundryCompileError:
        # Foundry owns these strings only until the next call on this compiler.
        # Decode each value immediately so the exception owns durable copies.
        raw_message = self._dll.FdyCompilerGetLastError(self._compiler)
        message = raw_message.decode("utf-8", errors="replace") if raw_message else ""
        unsupported_op = None
        missing_external_data = None
        if result == 9:
            raw_detail = self._dll.FdyCompilerGetLastUnsupportedOpName(self._compiler)
            if raw_detail:
                unsupported_op = raw_detail.decode("utf-8", errors="replace")
        elif result == 11:
            raw_detail = self._dll.FdyCompilerGetLastMissingExternalDataFile(self._compiler)
            if raw_detail:
                missing_external_data = raw_detail.decode("utf-8", errors="replace")
        return FoundryCompileError(
            result,
            native_message=message,
            unsupported_op=unsupported_op,
            missing_external_data=missing_external_data,
        )

    def compile_onnx(
        self,
        source: bytes,
        *,
        model_directory: Path,
        update_opset: bool,
        topo_sort_nodes: bool,
        include_initializers: bool,
        enable_lazy_external_data: bool,
        output_data_file: Path | None = None,
        freeze_dims: Mapping[str, int] | None = None,
    ) -> bytes:
        """Compile ONNX protobuf bytes to textual DxCGC MLIR."""
        source_buffer = ctypes.create_string_buffer(source)
        model_directory_bytes = str(model_directory.resolve()).encode("utf-8")
        output_data_file_bytes = (
            str(output_data_file).encode("utf-8")
            if output_data_file is not None
            else b""
        )
        dim_names = tuple((freeze_dims or {}).keys())
        dim_name_bytes = tuple(name.encode("utf-8") for name in dim_names)
        dim_name_array = (
            (ctypes.c_char_p * len(dim_names))(*dim_name_bytes)
            if dim_names
            else None
        )
        dim_value_array = (
            (ctypes.c_int64 * len(dim_names))(
                *((freeze_dims or {})[name] for name in dim_names)
            )
            if dim_names
            else None
        )
        dim_descriptor = (
            _FdyOverrideDynamicDimsByDimNamePassDescriptor(
                descriptor=_FdyPassDescriptor(
                    kind=FDY_PASS_OVERRIDE_DYNAMIC_DIMS_BY_DIM_NAME,
                    stage=FDY_PASS_STAGE_BEFORE_LOWERING,
                ),
                names=dim_name_array,
                values=dim_value_array,
                count=len(dim_names),
            )
            if dim_names
            else None
        )
        passes = (
            (ctypes.POINTER(_FdyPassDescriptor) * 1)(
                ctypes.cast(
                    ctypes.pointer(dim_descriptor),
                    ctypes.POINTER(_FdyPassDescriptor),
                )
            )
            if dim_descriptor is not None
            else None
        )
        compiler_options = _FdyCompilerOptions(
            version=FDY_VERSION,
            sourceFormat=FDY_SOURCE_FORMAT_ONNX_PROTOBUF,
            target=FDY_COMPILER_TARGET_DXCGC,
            updateOpset=update_opset,
            topoSortNodes=topo_sort_nodes,
            includeInitializers=include_initializers,
            outputDataFile=_string_view(output_data_file_bytes),
            passes=passes,
            passCount=len(passes) if passes is not None else 0,
            modelDirectory=_string_view(model_directory_bytes),
            enableLazyExternalData=enable_lazy_external_data,
            safetensorsFiles=None,
            safetensorsFileCount=0,
        )
        module = ctypes.c_void_p()
        result = self._dll.FdyCompilerCompile(
            self._compiler,
            _FdySpan(ctypes.cast(source_buffer, ctypes.c_void_p), len(source)),
            ctypes.byref(compiler_options),
            ctypes.byref(module),
        )
        if result != FDY_COMPILER_RESULT_SUCCESS:
            raise self._last_error(result)

        try:
            size = ctypes.c_size_t()
            result = self._dll.FdyModuleSerialize(
                module,
                FDY_SERIALIZATION_FORMAT_TEXT,
                _FdyMutableSpan(None, 0),
                ctypes.byref(size),
            )
            if result != FDY_COMPILER_RESULT_SUCCESS:
                raise self._last_error(result)
            output = ctypes.create_string_buffer(size.value)
            result = self._dll.FdyModuleSerialize(
                module,
                FDY_SERIALIZATION_FORMAT_TEXT,
                _FdyMutableSpan(ctypes.cast(output, ctypes.c_void_p), len(output)),
                ctypes.byref(size),
            )
            if result != FDY_COMPILER_RESULT_SUCCESS:
                raise self._last_error(result)
            return output.raw[: size.value]
        finally:
            self._dll.FdyModuleDestroy(module)
