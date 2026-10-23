# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Keep CLI metadata, automation, and runtime typing on the same Python floor."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from winml.modelkit.analyze.models import RuntimeTestResult


ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()
)


def test_python_runtime_policy_matches_project_tools() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert SpecifierSet(config["project"]["requires-python"]) == SpecifierSet(">=3.14")
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.14"
    assert config["tool"]["ruff"]["target-version"] == "py314"
    assert config["tool"]["mypy"]["python_version"] == "3.14"
    version_classifiers = [
        value
        for value in config["project"]["classifiers"]
        if value.startswith("Programming Language :: Python :: 3.")
    ]
    assert version_classifiers == ["Programming Language :: Python :: 3.14"]
    for raw in config["project"]["dependencies"]:
        requirement = Requirement(raw)
        assert "python_version" not in str(requirement.marker)
        assert "python_full_version" not in str(requirement.marker)


@pytest.mark.parametrize("filename", ["docs.yml", "lint.yml", "modelkit-ci.yml"])
def test_github_jobs_use_python314(filename: str) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
    )
    pins = []
    for job in workflow["jobs"].values():
        matrix = job.get("strategy", {}).get("matrix", {})
        assert "python-version" not in matrix
        for step in job.get("steps", []):
            if "python-version" in step.get("with", {}):
                pins.append(step["with"]["python-version"])
            for line in step.get("run", "").splitlines():
                if line.startswith("uv python install "):
                    pins.append(line.removeprefix("uv python install "))
                elif line.startswith("uv sync "):
                    assert "--python 3.14" in line
    assert pins and set(pins) == {"3.14"}


def test_pre_commit_uses_the_cli_python_runtime() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    assert config["default_language_version"]["python"] == "python3.14"


def test_ci_check_names_remain_stable_for_branch_protection() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "modelkit-ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["test"]
    assert job["name"] == "test (${{ matrix.group }})"
    assert set(job["strategy"]["matrix"]["group"]) == {
        "analyze",
        "models",
        "optim",
        "commands",
        "remaining",
    }


def test_vendor_compiler_formatting_preserves_its_isolated_runtime() -> None:
    script = ROOT / "src" / "winml" / "modelkit" / "session" / "qairt" / "compile_qairt_bin.py"
    ast.parse(script.read_text(encoding="utf-8"), feature_version=(3, 11))
    formatted = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(script), "-"],
        input="try:\n    pass\nexcept (ValueError, TypeError):\n    pass\n",
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    ast.parse(formatted.stdout, feature_version=(3, 11))


@pytest.mark.parametrize("filename", ["e2e-test-jobs.yml", "eval-report-jobs.yml"])
def test_self_hosted_jobs_pin_x64_python314(filename: str) -> None:
    template = (ROOT / ".pipelines" / "templates" / filename).read_text(encoding="utf-8")
    install_lines = [
        line.strip()
        for line in template.splitlines()
        if line.strip().startswith("uv python install ")
    ]
    assert install_lines == ["uv python install cpython-3.14-windows-x86_64-none"]
    environment_lines = [
        line.strip() for line in template.splitlines() if line.strip().startswith("uv venv ")
    ]
    assert environment_lines
    assert all("--python cpython-3.14-windows-x86_64-none" in line for line in environment_lines)


def test_runtime_debug_typeddict_validates_and_serializes_on_python314() -> None:
    result = RuntimeTestResult(
        compile=True,
        run=True,
        debug_details={"source": "unit-test", "case_indices": [1, "float32"]},
    )
    payload = json.loads(result.model_dump_json())
    assert payload["debug_details"] == {
        "source": "unit-test",
        "case_indices": [1, "float32"],
    }
    assert RuntimeTestResult.model_validate_json(result.model_dump_json()) == result
    schema = RuntimeTestResult.model_json_schema()["$defs"]["RuntimeDebugDetails"]
    assert schema.get("required", []) == []
    with pytest.raises(ValidationError):
        RuntimeTestResult.model_validate(
            {
                "compile": True,
                "run": True,
                "debug_details": {"total_rows": "not-an-integer"},
            }
        )
