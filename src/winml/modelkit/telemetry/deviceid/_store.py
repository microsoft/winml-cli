# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

r"""Windows-registry-backed storage for telemetry's small key-value entries.

Location: ``HKCU\SOFTWARE\Microsoft\DeveloperTools\.modelkit``. Values are
``REG_SZ`` strings; non-string registry types are treated as absent on
read.
"""

from __future__ import annotations

import os
from pathlib import Path


_SUBPATH = Path("DeveloperTools") / ".modelkit"
_REGISTRY_KEY = r"SOFTWARE\Microsoft\DeveloperTools\.modelkit"


def get_telemetry_base_dir() -> Path:
    r"""Return the base directory for telemetry state (cache files etc.).

    Rooted at ``%LOCALAPPDATA%\Microsoft\DeveloperTools\.modelkit``. Plain
    function, **not** a ``@property`` — a descriptor is not callable and
    would silently break on import.
    """
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / _SUBPATH
    base.mkdir(parents=True, exist_ok=True)
    return base


def read_key(name: str) -> str | None:
    """Return the stored ``REG_SZ`` value, or ``None`` if absent or non-string."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REGISTRY_KEY) as key:
            value, value_type = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    if value_type != winreg.REG_SZ:
        return None
    return value  # already str for REG_SZ


def write_key(name: str, value: str) -> None:
    """Persist ``value`` as a ``REG_SZ`` under ``name``."""
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _REGISTRY_KEY) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def delete_key(name: str) -> None:
    """Remove the stored value for ``name``. Idempotent."""
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REGISTRY_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        # Value or key already absent — delete is idempotent.
        return
