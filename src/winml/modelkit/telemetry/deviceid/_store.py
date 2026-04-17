# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

r"""Platform-specific storage for telemetry's small key-value entries.

(device id, consent decision, etc.)

On Windows: HKCU\SOFTWARE\Microsoft\DeveloperTools\.modelkit
On other platforms (for parity/tests only): a JSON file under a per-user dir.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_SUBPATH = Path("DeveloperTools") / ".modelkit"
_REGISTRY_KEY = r"SOFTWARE\Microsoft\DeveloperTools\.modelkit"


def get_telemetry_base_dir() -> Path:
    """Return the base directory for telemetry state on the current platform.

    This helper is a plain function, **not** a module-level `@property`,
    because a property descriptor is not callable and silently breaks on
    any platform that imports it.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / _SUBPATH
    else:
        try:
            home = Path.home()
        except (RuntimeError, KeyError):
            home = Path("/var/tmp")  # noqa: S108
        base = home / _SUBPATH
    base.mkdir(parents=True, exist_ok=True)
    return base


def read_key(name: str) -> str | None:
    """Return the stored string value for `name`, or None if absent."""
    if sys.platform == "win32":
        return _read_registry(name)
    return _read_file(name)


def write_key(name: str, value: str) -> None:
    """Persist `value` under `name`."""
    if sys.platform == "win32":
        _write_registry(name, value)
    else:
        _write_file(name, value)


def delete_key(name: str) -> None:
    """Remove the stored value for `name`, if any."""
    if sys.platform == "win32":
        _delete_registry(name)
    else:
        _delete_file(name)


# --- Windows registry backend ---


def _read_registry(name: str) -> str | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REGISTRY_KEY) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _write_registry(name: str, value: str) -> None:
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _REGISTRY_KEY) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def _delete_registry(name: str) -> None:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _REGISTRY_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.DeleteValue(key, name)
    except FileNotFoundError:
        pass
    except OSError:
        pass


# --- File backend (Linux/macOS; used in parity tests) ---


def _state_file() -> Path:
    return get_telemetry_base_dir() / "state.json"


def _read_file(name: str) -> str | None:
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    value = data.get(name)
    return str(value) if value is not None else None


def _write_file(name: str, value: str) -> None:
    path = _state_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[name] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def _delete_file(name: str) -> None:
    path = _state_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    data.pop(name, None)
    path.write_text(json.dumps(data), encoding="utf-8")
