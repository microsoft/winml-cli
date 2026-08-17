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


def test_uv_lock_records_direct_project_dependencies() -> None:
    """The editable lock entry must retain every direct project dependency."""
    repo_root = Path(__file__).resolve().parents[2]
    project_data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock_data = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    direct_names = {
        dependency.split(";", 1)[0]
        .split("[", 1)[0]
        .split("=", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in project_data["project"]["dependencies"]
    }
    root_package = next(
        package
        for package in lock_data["package"]
        if package["name"] == "winml-cli" and package.get("source", {}).get("editable") == "."
    )
    locked_dependencies = {
        _dependency_name(dependency).lower().replace("_", "-")
        for dependency in root_package["dependencies"]
    }
    locked_requirements = {
        str(requirement["name"]).lower().replace("_", "-")
        for requirement in root_package["metadata"]["requires-dist"]
    }

    assert direct_names <= locked_dependencies
    assert direct_names <= locked_requirements
