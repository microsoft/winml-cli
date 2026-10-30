# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Effective-configuration facts for GGUF evidence that the Task API does not report.

Model shard identity, installed package versions, Windows ML payload manifests,
backend-related environment variables, and the modules actually loaded in this
process. Everything here reads files, package metadata or the live process; the
Windows ML Runtime is never imported.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


PACKAGE_DISTRIBUTIONS: tuple[str, ...] = (
    "windowsml",
    "windowsml-llama-core",
    "windowsml-llama-cuda",
    "onnxruntime-windowsml",
    "winml-cli",
)

# gguf-split names every shard <prefix>-<index>-of-<count>.gguf (five digits each).
_SPLIT_SHARD = re.compile(
    r"^(?P<prefix>.+)-\d{5}(?P<separator>-of-)(?P<count>\d{5})(?P<suffix>\.gguf)$",
    re.IGNORECASE,
)
_ENVIRONMENT_PREFIXES = ("WINML_", "GGML_", "LLAMA_", "CUDA_")
# Read natively by the llama.cpp adapter; recorded even when unset.
_FIXED_ENVIRONMENT = ("WINML_LLAMA_USE_MMAP", "WINML_LLAMA_SWA_FULL")
# Evidence is published, so values of names like these (LLAMA_API_KEY, ...) are hidden.
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
REDACTED = "<redacted>"
_CORE_MANIFEST = "WinMLLlamaCoreManifest.json"
_BACKEND_MANIFEST = "WinMLLlamaBackendManifest.json"
_PAIRING_MANIFEST = "WinMLLlamaPythonPairing.json"
_LIST_MODULES_ALL = 0x03
_MAX_PATH_CHARS = 32768


# =============================================================================
# Model shards
# =============================================================================


def gguf_shard_paths(entry: Path) -> list[Path]:
    """Return every shard of a GGUF model, given its entry file.

    A gguf-split name (``<prefix>-00001-of-00003.gguf``) expands to all shards
    in the same folder; any other name is a single-file model.

    Raises:
        FileNotFoundError: A shard of a split model is missing.
    """
    match = _SPLIT_SHARD.match(entry.name)
    if match is None:
        return [entry]
    count = int(match["count"])
    shards = [
        entry.with_name(
            f"{match['prefix']}-{index:05d}{match['separator']}{match['count']}{match['suffix']}"
        )
        for index in range(1, count + 1)
    ]
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(
            f"GGUF split shard(s) not found next to {entry}: {', '.join(missing)}"
        )
    return shards


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in fixed-size chunks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_inventory(entry: Path) -> dict[str, Any]:
    """Return the evidence ``configuration.model`` object for a GGUF entry file.

    Shard paths are file names relative to the entry's folder, and the
    inventory digest hashes the path-sorted shards as compact, key-sorted JSON:
    the same definition the benchmark uses for its artifact identity.
    """
    shards: list[dict[str, Any]] = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(gguf_shard_paths(entry), key=lambda shard: shard.name)
    ]
    digest = hashlib.sha256(
        json.dumps(shards, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "entry_path": str(entry.resolve()),
        "shards": shards,
        "inventory_sha256": digest,
        "hashed_before_initialization": True,
    }


# =============================================================================
# Packages, manifests and environment
# =============================================================================


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    """Installed versions of the distributions on the GGUF path (``None`` when absent)."""
    return {name: _distribution_version(name) for name in PACKAGE_DISTRIBUTIONS}


def windowsml_package_dir() -> Path | None:
    """Locate the installed ``windowsml`` package without importing it."""
    spec = importlib.util.find_spec("windowsml")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _read_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    entry: dict[str, Any] = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fields": None,
    }
    try:
        entry["fields"] = json.loads(raw.decode("utf-8-sig"))
    except ValueError as exc:  # includes UnicodeDecodeError
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def read_manifests(package_dir: Path | None) -> list[dict[str, Any]]:
    """Read the llama.cpp payload manifests installed under ``windowsml/lib``.

    Returns the core manifest, each backend manifest (sorted by path) and the
    Python pairing record, for those that exist. Each entry carries the file's
    path and SHA-256 and its JSON ``fields`` (``None`` with an ``error`` when
    the file is not valid JSON).
    """
    if package_dir is None:
        return []
    lib = package_dir / "lib"
    candidates = [
        lib / _CORE_MANIFEST,
        *sorted((lib / "LlamaBackends").glob(f"*/{_BACKEND_MANIFEST}")),
        lib / _PAIRING_MANIFEST,
    ]
    return [_read_manifest(path) for path in candidates if path.is_file()]


def environment_snapshot(environ: Mapping[str, str]) -> dict[str, str | None]:
    """Environment variables that can change llama.cpp or Windows ML behavior.

    Keeps every variable starting with ``WINML_``, ``GGML_``, ``LLAMA_`` or
    ``CUDA_``, plus the adapter's own settings (``None`` when unset), sorted.
    Names that look like credentials (``KEY``, ``TOKEN``, ``SECRET``, ...) keep
    only the fact that they are set: their value is :data:`REDACTED`.
    """
    snapshot: dict[str, str | None] = dict.fromkeys(_FIXED_ENVIRONMENT)
    snapshot.update(
        {
            name: REDACTED if any(part in name.upper() for part in _SECRET_NAME_PARTS) else value
            for name, value in environ.items()
            if name.upper().startswith(_ENVIRONMENT_PREFIXES)
        }
    )
    return dict(sorted(snapshot.items()))


def _redact_url(url: str) -> str:
    """Hide the credentials a URL can carry: its user information and query values.

    The ``git`` user of an SSH VCS URL names the account type, not a secret, so it stays.
    """
    parts = urllib.parse.urlsplit(url)
    userinfo, at, host = parts.netloc.rpartition("@")
    netloc = f"{REDACTED}@{host}" if at and userinfo != "git" else parts.netloc
    query = "&".join(
        f"{key}={REDACTED}"
        for key, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    )
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc, query=query))


def _direct_url(name: str) -> Any:
    """Contents of a distribution's ``direct_url.json`` (PEP 610), or ``None``.

    Evidence is published, so credentials in the install URL are redacted.
    """
    try:
        text = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if text is None:
        return None
    try:
        document = json.loads(text)
    except ValueError:
        return None
    if isinstance(document, dict) and isinstance(document.get("url"), str):
        document["url"] = _redact_url(document["url"])
    return document


def producer_info() -> dict[str, Any]:
    """Identify this winml-cli installation as the evidence producer."""
    return {
        "name": "winml-cli",
        "version": _distribution_version("winml-cli"),
        "command": "perf",
        "install": {
            "location": str(Path(__file__).resolve().parents[1]),
            "direct_url": _direct_url("winml-cli"),
        },
    }


# =============================================================================
# Process, clock and loaded modules
# =============================================================================


def process_identity() -> dict[str, Any]:
    """PID and creation time, exactly as psutil reports them to any observer."""
    return {"pid": os.getpid(), "create_time": psutil.Process().create_time()}


def clock_info(started_qpc_ns: int) -> dict[str, Any]:
    """Describe the machine-wide QPC clock behind every evidence timestamp."""
    info = time.get_clock_info("perf_counter")
    return {
        "domain": "windows-qpc",
        "api": "time.perf_counter_ns",
        "implementation": info.implementation,
        "resolution": info.resolution,
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
        "started_qpc_ns": started_qpc_ns,
    }


@functools.cache
def _kernel32() -> Any:
    """A private kernel32 instance, so its prototypes never clash with other callers."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    kernel32.K32EnumProcessModulesEx.restype = wintypes.BOOL
    return kernel32


def _module_file_name(handle: Any) -> str | None:
    """Full path of a loaded module, or ``None`` when it is no longer loaded."""
    buffer = ctypes.create_unicode_buffer(_MAX_PATH_CHARS)
    length = _kernel32().GetModuleFileNameW(handle, buffer, len(buffer))
    return buffer.value if length else None


def loaded_module_handle(name: str) -> tuple[int, str] | None:
    """Handle and full path of a module already loaded in this process, or ``None``."""
    handle = _kernel32().GetModuleHandleW(name)
    if not handle:
        return None
    path = _module_file_name(handle)
    return None if path is None else (int(handle), path)


def list_loaded_modules() -> list[str]:
    """Full paths of every module loaded in this process, sorted and unique.

    Raises:
        OSError: The module list cannot be enumerated.
    """
    kernel32 = _kernel32()
    process = kernel32.GetCurrentProcess()
    handle_size = ctypes.sizeof(wintypes.HMODULE)
    capacity = 512
    while True:
        handles = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not kernel32.K32EnumProcessModulesEx(
            process, handles, ctypes.sizeof(handles), ctypes.byref(needed), _LIST_MODULES_ALL
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(handles):
            break
        capacity = needed.value // handle_size + 16
    names = {_module_file_name(handles[index]) for index in range(needed.value // handle_size)}
    return sorted(name for name in names if name)


def _normalized(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).absolute()))


def _declare(declared: dict[str, set[str]], path: Path, sha256: Any) -> None:
    if isinstance(sha256, str) and sha256:
        declared.setdefault(_normalized(path), set()).add(sha256.lower())


def _declared_files(manifest: Mapping[str, Any]) -> list[tuple[Path, Any]]:
    """``(file, declared sha256)`` pairs from one manifest, by manifest kind."""
    fields = manifest.get("fields")
    if not isinstance(fields, dict):
        return []
    location = Path(manifest["path"])
    folder = location.parent
    if location.name == _BACKEND_MANIFEST:
        return [(folder / str(fields.get("backend", "")), fields.get("sha256"))]
    key = {"WinMLLlamaCoreManifest.json": "path", _PAIRING_MANIFEST: "name"}.get(location.name)
    if key is None:
        return []
    return [
        (folder / str(item.get(key, "")), item.get("sha256"))
        for item in fields.get("files", [])
        if isinstance(item, dict)
    ]


def hash_modules(
    paths: Iterable[str],
    package_dir: Path | None,
    manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hash the loaded modules that belong to the ``windowsml`` package.

    ``matches_manifest`` is ``True`` when every manifest that declares the file
    gives its actual SHA-256, ``False`` when one gives another hash, and
    ``None`` when no manifest declares it.
    """
    if package_dir is None:
        return []
    root = _normalized(package_dir) + os.sep
    declared: dict[str, set[str]] = {}
    for manifest in manifests:
        for declared_path, sha256 in _declared_files(manifest):
            _declare(declared, declared_path, sha256)
    selected = sorted({module for module in paths if _normalized(module).startswith(root)})
    hashed: list[dict[str, Any]] = []
    for module in selected:
        actual = sha256_file(Path(module))
        expected = declared.get(_normalized(module))
        hashed.append(
            {
                "path": module,
                "size_bytes": Path(module).stat().st_size,
                "sha256": actual,
                "matches_manifest": None if not expected else expected == {actual},
            }
        )
    return hashed
