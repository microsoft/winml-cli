# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGC IR parsing and text serialization through the wheel's MLIR C API."""

from __future__ import annotations

import ctypes
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import click


class _MlirHandle(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)]


class _MlirStringRef(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


_StringCallback = ctypes.CFUNCTYPE(None, _MlirStringRef, ctypes.c_void_p)


def _cgc_ir_library() -> Any:
    try:
        distribution = metadata.distribution("windowsml")
    except metadata.PackageNotFoundError as e:
        raise click.ClickException(
            "Text CG Output IR requires an installed windowsml package."
        ) from e
    path = Path(str(distribution.locate_file(Path("windowsml") / "lib" / "cgc_ir.dll")))
    if not path.is_file():
        raise click.ClickException(
            f"The installed windowsml package ({distribution.version}) does not contain cgc_ir.dll."
        )
    with os.add_dll_directory(str(path.parent)):
        return ctypes.WinDLL(str(path))


def _bind(dll: Any, name: str, argtypes: list[Any], restype: Any) -> Any:
    try:
        function = getattr(dll, name)
    except AttributeError as e:
        raise click.ClickException(
            f"The installed cgc_ir.dll does not expose the required {name} API."
        ) from e
    function.argtypes = argtypes
    function.restype = restype
    return function


def read_mlir_text(path: Path) -> str:
    """Parse text or bytecode CGC IR and return its UTF-8 textual form."""
    dll = _cgc_ir_library()
    context_create = _bind(
        dll, "mlirContextCreateWithThreading", [ctypes.c_bool], _MlirHandle
    )
    context_destroy = _bind(dll, "mlirContextDestroy", [_MlirHandle], None)
    register_dialect = _bind(
        dll,
        "mlirDialectHandleRegisterDialect",
        [_MlirHandle, _MlirHandle],
        None,
    )
    module_parse = _bind(
        dll,
        "cgcModuleCreateParseFromFile",
        [_MlirHandle, _MlirStringRef],
        _MlirHandle,
    )
    module_destroy = _bind(dll, "cgcModuleDestroy", [_MlirHandle], None)
    module_operation = _bind(
        dll, "cgcModuleGetOperation", [_MlirHandle], _MlirHandle
    )
    operation_print = _bind(
        dll,
        "mlirOperationPrint",
        [_MlirHandle, _StringCallback, ctypes.c_void_p],
        None,
    )

    context = context_create(False)
    module = _MlirHandle()
    try:
        for name in (
            "mlirGetDialectHandle__cgc__",
            "mlirGetDialectHandle__cgc_op__",
            "mlirGetDialectHandle__cgc_experimental__",
        ):
            register_dialect(_bind(dll, name, [], _MlirHandle)(), context)

        encoded_path = os.fsencode(path)
        module = module_parse(
            context,
            _MlirStringRef(
                ctypes.cast(ctypes.c_char_p(encoded_path), ctypes.c_void_p),
                len(encoded_path),
            ),
        )
        if not module.ptr:
            raise click.ClickException(f"Cannot parse CG Output IR: {path}")

        chunks: list[bytes] = []

        def append_chunk(value: _MlirStringRef, _user_data: int) -> None:
            chunks.append(ctypes.string_at(value.data, value.length))

        callback = _StringCallback(append_chunk)
        operation_print(module_operation(module), callback, None)
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as e:
            raise click.ClickException("cgc_ir.dll returned non-UTF-8 MLIR text.") from e
    finally:
        if module.ptr:
            module_destroy(module)
        context_destroy(context)
