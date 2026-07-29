# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process-wide bookkeeping for native ORT EP registrations."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .session.ep_registry import WinMLEP


_NATIVE_REGISTRATION_LOCK = threading.RLock()
_NATIVE_REGISTERED_BY_PATH: dict[Path, WinMLEP] = {}
_NATIVE_REGISTERED_ARG0_BY_PATH: dict[Path, str] = {}
_NATIVE_REGISTRATION_COUNT: dict[str, int] = {}


def native_registration_key(ep_name: str, *, canonical_key_available: bool = True) -> str:
    """Return the ORT registration key for the next DLL of this EP name."""
    n = _NATIVE_REGISTRATION_COUNT.get(ep_name, 0)
    if n == 0 and not canonical_key_available:
        return f"{ep_name}_0"
    return ep_name if n == 0 else f"{ep_name}_{n}"


def record_native_registration(
    *,
    ep_name: str,
    dll_path: Path,
    arg0: str,
    winml_ep: WinMLEP | None = None,
) -> None:
    """Record a successful process-wide native registration."""
    dll_path = Path(dll_path)
    _NATIVE_REGISTERED_ARG0_BY_PATH[dll_path] = arg0
    if winml_ep is not None:
        _NATIVE_REGISTERED_BY_PATH[dll_path] = winml_ep
    _NATIVE_REGISTRATION_COUNT[ep_name] = max(
        _NATIVE_REGISTRATION_COUNT.get(ep_name, 0),
        _registration_count_after_key(ep_name, arg0),
    )


def forget_native_registration(dll_path: Path) -> None:
    """Drop cached metadata for a native registration path."""
    dll_path = Path(dll_path)
    _NATIVE_REGISTERED_BY_PATH.pop(dll_path, None)
    _NATIVE_REGISTERED_ARG0_BY_PATH.pop(dll_path, None)


def clear_native_registration_state() -> None:
    """Clear process-wide native registration bookkeeping for tests."""
    _NATIVE_REGISTERED_BY_PATH.clear()
    _NATIVE_REGISTERED_ARG0_BY_PATH.clear()
    _NATIVE_REGISTRATION_COUNT.clear()


def _registration_count_after_key(ep_name: str, arg0: str) -> int:
    if arg0 == ep_name:
        return 1
    prefix = f"{ep_name}_"
    if not arg0.startswith(prefix):
        return _NATIVE_REGISTRATION_COUNT.get(ep_name, 0)
    try:
        suffix = int(arg0.removeprefix(prefix))
    except ValueError:
        return _NATIVE_REGISTRATION_COUNT.get(ep_name, 0)
    return suffix + 1
