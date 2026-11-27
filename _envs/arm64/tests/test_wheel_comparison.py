# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compare installed wheels, not editable checkouts. See the ARM installation guide."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import onnx


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT.parents[1] / "tests" / "assets"
MODEL = str(ASSETS / "resnet_fp32" / "model.onnx")
QUANTIZED = str(ASSETS / "resnet_w8a8" / "model.onnx")

IDENTIFY = """
import hashlib, json, sys, sysconfig
from importlib.metadata import distribution, distributions
from pathlib import Path
from winml import modelkit
d = distribution("winml-cli")
origin = json.loads(d.read_text("direct_url.json"))
assert "archive_info" in origin, "Install the built wheel, not an editable project"
assert Path(modelkit.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
payload = sorted(f"{p}:{p.hash.value}" for p in d.files if p.hash and p.parts[0] == "winml")
packages = {p.metadata["Name"].lower(): p.version for p in distributions()}
print(json.dumps(dict(platform=sysconfig.get_platform(), python=sys.version.split()[0],
                      version=d.version, wheel=origin["url"].rsplit("/", 1)[-1],
                      payload_sha256=hashlib.sha256("\\n".join(payload).encode()).hexdigest(),
                      packages=packages)))
"""


def _run(python, args, cwd):
    return subprocess.run(  # noqa: S603 - the caller selects two local wheel interpreters
        [python, "-I", "-X", "utf8", *args],
        cwd=cwd,
        env={**os.environ, "CI": "1", "HF_HUB_OFFLINE": "1", "NO_COLOR": "1"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        check=False,
    )


def _probe(python, args, cwd, output=None):
    cwd.mkdir()
    try:
        result = _run(python, ["-m", "winml.modelkit.cli", *args], cwd)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": "Exceeded 240 seconds"}
    # Analyze returns 1 for a completed report that does not declare full model support.
    partial = args[0] == "analyze" and output and result.returncode == 1
    if result.returncode and not partial:
        lines = (result.stderr + "\n" + result.stdout).splitlines()
        detail = [s for s in lines if "Error:" in s or "No module named" in s]
        return {"status": "failed", "code": result.returncode, "detail": (detail or lines)[-5:]}
    if "--help" in args and "Usage:" not in result.stdout:
        return {"status": "invalid output", "detail": "Missing command usage"}
    if "--list-device" in args and not json.loads(result.stdout).get("devices"):
        return {"status": "invalid output", "detail": "No devices returned"}
    if output:
        path = cwd / output
        if not path.is_file():
            return {"status": "invalid output", "detail": f"Missing {output}"}
        if path.suffix == ".onnx":
            onnx.checker.check_model(str(path))
        elif path.suffix == ".json":
            data = json.loads(path.read_bytes())
            if not data:
                return {"status": "invalid output", "detail": f"Empty {output}"}
            if output == "analysis.json" and not data.get("results"):
                return {"status": "invalid output", "detail": "No EP analysis results"}
            if output == "analysis.json" and "--ep" in args:
                counts = {
                    level: len(ops) for level, ops in data["results"][0]["classification"].items()
                }
                if "Failed to read parquet rules" in result.stderr:
                    return {
                        "status": "degraded",
                        "detail": "Generated a report but could not read Parquet rules",
                        "classification": counts,
                    }
                return {"status": "passed", "code": result.returncode, "classification": counts}
            if output == "perf.json" and len(data["raw_samples_ms"]) != 1:
                return {"status": "invalid output", "detail": "No measured inference"}
            if output == "generation.json" and not any(
                r["generated_tokens"] for r in data["requests"]
            ):
                return {"status": "invalid output", "detail": "No generated tokens"}
    if partial:
        return {
            "status": "passed",
            "code": 1,
            "detail": "Report produced; model not fully supported",
        }
    return {"status": "passed"}


def test_installed_wheel_comparison(tmp_path):
    """Record unavailable workflows without treating a help listing as feature parity."""
    assert "WINML_X64_PYTHON" in os.environ, "Set WINML_X64_PYTHON to the x64 wheel interpreter"
    interpreters = {"arm64": sys.executable, "x64": os.environ["WINML_X64_PYTHON"]}
    environments, commands = {}, {}
    for arch, python in interpreters.items():
        probe = _run(python, ["-c", IDENTIFY], tmp_path)
        assert probe.returncode == 0, probe.stderr
        info = environments[arch] = json.loads(probe.stdout)
        platform = {"arm64": "win-arm64", "x64": "win-amd64"}[arch]
        assert info["platform"] == platform
        assert info["wheel"].endswith(platform.replace("-", "_") + ".whl")
        help_result = _run(python, ["-m", "winml.modelkit.cli", "--help"], tmp_path)
        assert help_result.returncode == 0, help_result.stderr
        commands[arch] = re.findall(
            r"^  ([\w-]+)  +", help_result.stdout.split("Commands:")[1], re.M
        )
        assert commands[arch]
    assert environments["x64"]["version"] == environments["arm64"]["version"]
    assert environments["x64"]["payload_sha256"] == environments["arm64"]["payload_sha256"]

    cases = {
        f"help-{cmd}": ([cmd, "--help"], None) for cmd in sorted(set().union(*commands.values()))
    }
    cases.update(
        {
            "catalog": (["catalog", "-o", "catalog.json"], "catalog.json"),
            "devices": (["sys", "--list-device", "--format", "json"], None),
            "cgc-adapters": (["cgc", "adapters"], None),
            "compilers": (["compile", "--list"], None),
            "inspect-tasks": (["inspect", "--list-tasks"], None),
            "config": (["config", "-m", MODEL, "-o", "config.json"], "config.json"),
            "analyze": (
                ["analyze", "-m", MODEL, "--device", "cpu", "-o", "analysis.json"],
                "analysis.json",
            ),
            "optimize": (["optimize", "-m", MODEL, "-o", "optimized.onnx"], "optimized.onnx"),
            "quantize-fp16": (
                ["quantize", "-m", MODEL, "-p", "fp16", "-o", "fp16.onnx"],
                "fp16.onnx",
            ),
            "eval-schema": (["eval", "--schema", "--task", "image-classification"], None),
            "eval-compare": (
                [
                    "eval",
                    "--mode",
                    "compare",
                    "-m",
                    MODEL,
                    "--reference",
                    MODEL,
                    "--device",
                    "cpu",
                    "--samples",
                    "1",
                    "-o",
                    "eval.json",
                ],
                "eval.json",
            ),
        }
    )
    for device, ep, model in (
        ("cpu", "cpu", MODEL),
        ("gpu", "dml", MODEL),
        ("npu", "qnn", QUANTIZED),
    ):
        cases[f"perf-{ep}"] = (
            [
                "perf",
                "-m",
                model,
                "--device",
                device,
                "--ep",
                ep,
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--no-memory",
                "--format",
                "json",
                "-o",
                "perf.json",
            ],
            "perf.json",
        )
    rules = os.environ.get("WINMLCLI_RULES_DIR")
    if rules and any((Path(rules) / "OpenVINOExecutionProvider_CPU").glob("*.parquet")):
        cases["analyze-parquet"] = (
            ["analyze", "-m", MODEL, "--ep", "openvino", "--device", "cpu", "-o", "analysis.json"],
            "analysis.json",
        )
    if checkpoint := os.environ.get("WINML_CACHED_HF_ID"):
        cases["export"] = (["export", "-m", checkpoint, "-o", "export.onnx"], "export.onnx")
        cases["build-cpu"] = (
            ["build", "-m", checkpoint, "-d", "cpu", "-p", "fp32", "-o", "build"],
            "build/model.onnx",
        )
    if gguf := os.environ.get("WINML_LOCAL_GGUF"):
        cases["gguf"] = (
            [
                "perf",
                "-m",
                str(Path(gguf).resolve(strict=True)),
                "--runtime",
                "winml-runtime",
                "--device",
                "cpu",
                "--prompt",
                "Hello",
                "--context-length",
                "128",
                "--max-new-tokens",
                "4",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--no-memory",
                "--format",
                "json",
                "-o",
                "generation.json",
            ],
            "generation.json",
        )
    results = {}
    for name, (args, output) in cases.items():
        results[name] = {
            arch: _probe(python, args, tmp_path / f"{name}-{arch}", output)
            for arch, python in interpreters.items()
        }
    report = PROJECT / "dist" / "wheel-comparison.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        json.dumps(
            {"environments": environments, "commands": commands, "cases": results}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Wheel comparison: {report}")
    assert sorted(commands["arm64"]) == sorted(commands["x64"])
    required = ["catalog", "devices", "perf-cpu", "analyze"]
    if checkpoint:
        required.extend(("export", "build-cpu"))
    for case in required:
        assert all(r["status"] == "passed" for r in results[case].values()), results[case]
