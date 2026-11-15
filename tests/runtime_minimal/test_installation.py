# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Installation contract for the native Arm64 Runtime-only environment."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("machine", ["ARM64", "aarch64"])
def test_arm_dependencies_exclude_onnx_stack(machine):
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(value) for value in metadata["project"]["dependencies"]]
    selected = {
        req.name: req
        for req in requirements
        if req.marker is None
        or req.marker.evaluate({"sys_platform": "win32", "platform_machine": machine})
    }
    assert {"click", "windowsml", "windowsml-llama-core"} <= selected.keys()
    assert (
        not {
            "onnx",
            "onnxruntime-windowsml",
            "onnxruntime-genai-winml",
            "torch",
            "numpy",
            "optimum",
            "pyarrow",
        }
        & selected.keys()
    )
    assert str(selected["windowsml"].specifier) == str(selected["windowsml-llama-core"].specifier)


def test_x64_retains_onnx_dependencies():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    selected = set()
    for value in metadata["project"]["dependencies"]:
        req = Requirement(value)
        if req.marker is None or req.marker.evaluate(
            {"sys_platform": "win32", "platform_machine": "AMD64"}
        ):
            selected.add(req.name)
    assert {"onnx", "onnxruntime-windowsml", "torch", "numpy"} <= selected


@pytest.mark.parametrize("args", [["--help"], ["perf", "--help"]])
def test_help_without_onnx_imports(args):
    code = (
        "import sys; from click.testing import CliRunner; "
        "from winml.modelkit.cli import main; "
        f"result = CliRunner().invoke(main, {args!r}); "
        "assert result.exit_code == 0, result.output; "
        "assert not {'onnx', 'onnxruntime', 'torch', 'numpy'} & sys.modules.keys()"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test code; no shell
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_arm64_qwen_generation(tmp_path):
    model = os.environ.get("WINML_ARM64_TEST_MODEL")
    if model is None:
        pytest.skip("Set WINML_ARM64_TEST_MODEL to a local GGUF for native hardware validation")
    assert sys.platform == "win32"
    assert platform.machine().lower() in {"arm64", "aarch64"}
    report = Path(os.environ.get("WINML_ARM64_TEST_REPORT", str(tmp_path / "perf.json")))
    completed = subprocess.run(  # noqa: S603 - model path is a separate argument; no shell
        [
            sys.executable,
            "-m",
            "winml.modelkit",
            "perf",
            "-m",
            model,
            "--runtime",
            "winml-runtime",
            "--device",
            "cpu",
            "--prompt",
            "Name three primary colors. /no_think",
            "--max-new-tokens",
            "64",
            "--context-length",
            "512",
            "--warmup",
            "1",
            "--iterations",
            "3",
            "--no-memory",
            "--format",
            "json",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    info = data["benchmark_info"]
    assert info["python_architecture"].lower() in {"arm64", "aarch64"}
    assert info["effective_device"] == "cpu"
    assert info["api"] == "windowsml.tasks"
    assert info["backend"] == "llama.cpp"
    assert info["packages"]["windowsml"] == info["packages"]["windowsml-llama-core"]
    assert 0 < info["generated_tokens"] <= 64
    assert data["response_text"].strip()
    assert len(data["requests"]) == 4
    assert data["aggregate"]["timed_request_count"] == 3
    assert data["aggregate"]["task_ttft_duration_ms"]["mean"] > 0
    assert data["aggregate"]["steady_state_decode_tokens_per_second"]["mean"] > 0
