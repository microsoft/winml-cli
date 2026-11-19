# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Run provenance for benchmark reports that are compared across machines.

A report records what produced it so runs can be compared fairly: the host
(OS, CPU, memory, GPUs and their drivers), free-form run labels, the run index
in a repeated series, and SHA-256 hashes of files the user names (for example
the model and draft model a server loaded). Hardware queries are best-effort:
a failure is recorded in the report and never fails the run.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
from typing import TYPE_CHECKING, Any

import click


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

_LABEL_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def parse_key_values(values: Iterable[str], *, flag: str) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options into a dict.

    Raises:
        click.UsageError: A value has no ``=``, an empty or invalid key, or a
            repeated key.
    """
    parsed: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not _LABEL_KEY.match(key):
            raise click.UsageError(
                f"{flag} '{item}' must be KEY=VALUE with a key of letters, digits, '_', '.' or '-'."
            )
        if key in parsed:
            raise click.UsageError(f"{flag} repeats the key '{key}'.")
        parsed[key] = value
    return parsed


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Return ``{label: {"name", "size_bytes", "sha256"}}`` for each labelled file.

    Only the file name is recorded, not its folder, so reports do not carry
    local paths.

    Raises:
        click.UsageError: A file does not exist or cannot be read.
    """
    hashed: dict[str, dict[str, Any]] = {}
    for label, path in files.items():
        if not path.is_file():
            raise click.UsageError(f"--hash-file {label}={path}: no such file.")
        try:
            hashed[label] = {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        except OSError as exc:
            raise click.UsageError(f"--hash-file {label}={path}: {exc}") from exc
    return hashed


def _safe(query: Callable[[], Any], errors: list[str], label: str) -> Any:
    try:
        return query()
    except Exception as exc:
        logger.debug("Provenance query %s failed: %s", label, exc)
        errors.append(f"{label}: {exc}")
        return None


def collect_host_info() -> dict[str, Any]:
    """Describe the host: OS, CPU, memory and GPUs with driver versions.

    The host name is never recorded. Queries that fail are listed under
    ``errors`` and their fields are ``None``.
    """
    from ..sysinfo.hardware import CPU, GPU, RAM
    from ..sysinfo.software import OS

    errors: list[str] = []
    os_info = _safe(lambda: OS.get().to_dict(), errors, "os")
    cpus = _safe(lambda: [cpu.to_dict() for cpu in CPU.get_all()], errors, "cpu")
    gpus = _safe(lambda: [gpu.to_dict() for gpu in GPU.get_all()], errors, "gpu")
    ram = _safe(lambda: sum(module.capacity_mib for module in RAM.get_all()), errors, "ram")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "os": os_info,
        "cpus": cpus,
        "gpus": gpus,
        "ram_mib": ram,
        "errors": errors,
    }
