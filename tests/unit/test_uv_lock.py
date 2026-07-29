# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomllib


DISALLOWED_ACCELERATOR_PACKAGE_PREFIXES = ("cuda-", "nvidia-")


def _dependency_name(dependency: str | dict[str, Any]) -> str:
    if isinstance(dependency, str):
        return dependency
    return str(dependency["name"])


def test_uv_lock_does_not_include_cuda_accelerator_packages() -> None:
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))

    disallowed_refs: set[str] = set()
    for package in lock_data["package"]:
        package_name = str(package["name"])
        if package_name.startswith(DISALLOWED_ACCELERATOR_PACKAGE_PREFIXES):
            disallowed_refs.add(package_name)

        for dependency in package.get("dependencies", []):
            dependency_name = _dependency_name(dependency)
            if dependency_name.startswith(DISALLOWED_ACCELERATOR_PACKAGE_PREFIXES):
                disallowed_refs.add(f"{package_name} -> {dependency_name}")

    assert not disallowed_refs, "Unexpected CUDA/NVIDIA lock entries: " + ", ".join(
        sorted(disallowed_refs)
    )
