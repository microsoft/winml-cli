# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Final champion output-bundle tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from types import ModuleType


SKILL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_ROOT / "scripts" / "finalize_output.py"
VALID_REPRO_RUN_BODY = "$Root = $PSScriptRoot\n"


@pytest.fixture(scope="module")
def output_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_optimize_finalize_output", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report() -> dict[str, Any]:
    report = {
        "schema_version": 2,
        "title": "Output bundle test",
        "updated_at": "2026-08-12T00:00:00Z",
        "model": {
            "path": "source.onnx",
            "sha256": "a" * 64,
            "opset": 18,
            "node_count": 4,
            "inputs": [{"name": "x", "shape": [1, 4], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [1, 4], "dtype": "float32"}],
            "op_counts": {"Conv": 1},
            "components": [{"name": "body", "nodes": 4, "note": "main graph"}],
        },
        "target": {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "latency_target_ms": 25.0,
            "provider_options": {"mode": "burst"},
        },
        "baseline": {
            "p50_ms": 30.0,
            "p90_ms": 31.0,
            "p99_ms": 33.0,
            "throughput_ips": 33.3,
            "protocol": "paired",
            "hotspots": [{"name": "Conv", "hardware_time_us": 1000}],
            "trace": {"partitions": 1},
        },
        "evidence": {
            "diagnosis": "The body dominates device time.",
            "execution": {"partition_count": 1},
            "analyzer": {"runtime_support": "supported"},
            "detail_profile": {"status": "available"},
            "ranked_levers": [
                {
                    "rank": 1,
                    "lever": "fold constants",
                    "evidence": "profile",
                    "confidence": "high",
                }
            ],
            "gaps": [],
        },
        "leader": {
            "id": "c1",
            "status": "confirmed",
            "model_path": "candidate.onnx",
            "p50_ms": 20.0,
            "gain_pct": 33.3,
            "ci_low_pct": 30.0,
            "ci_high_pct": 36.0,
            "correctness": "pass",
            "quality": "pass",
        },
        "hypotheses": [
            {
                "mechanism": "fold constants",
                "change": "fold",
                "supporting_evidence": "profile",
                "expected_delta": "remove op",
                "status": "confirmed",
                "falsifier": "op remains",
            }
        ],
        "experiments": [
            {
                "id": "c1",
                "parent": "baseline",
                "status": "confirmed",
                "change": "fold",
                "correctness": "pass",
                "p50_ms": 20.0,
                "gain_pct": 33.3,
                "graph_delta": {"Add": -1},
                "trace_delta": {"transpose": 0, "partitions": 0},
            }
        ],
        "capability_closure": {
            "reviewer": "Graph Scout",
            "leader_sha256": "b" * 64,
            "registry_evidence": "winml optimize --list-capabilities --verbose",
            "analyzer_evidence": "winml optimize --check-optim",
            "review_summary": "All registered residual opportunities are closed.",
            "coverage_verdict": "NO_MATERIAL_OMISSION",
            "probe_limit": 3,
            "rows": [
                {
                    "capability": "constant folding",
                    "flag": "--enable-constant-folding",
                    "sources": ["registry", "prior delta"],
                    "residual_anchor": "constant subgraph",
                    "eligibility": "matched",
                    "safety": "established",
                    "analyzer": "reported",
                    "rationale": "leader exposed constants",
                    "probe": "winml optimize --enable-constant-folding ...",
                    "expected_delta": "remove constant work",
                    "status": "CLOSED_ALREADY_TESTED",
                    "closure_reason": "tested in c1",
                }
            ],
        },
        "feature_gaps": [],
        "conclusion": {
            "stop_reason": "target confirmed",
            "remaining_opportunities": [],
            "reproduce": ["winml build ..."],
        },
        "artifacts": {
            "champion_onnx": "pending",
            "companions": [],
            "winml_config": "pending",
            "manifest": "pending",
        },
    }

    sections = [
        (report["baseline"], ("p90_ms", "p99_ms", "throughput_ips")),
        (
            report["evidence"]["execution"],
            (
                "accelerator_pct",
                "host_overhead_pct",
                "partition_count",
                "fallback_nodes",
                "transfers",
            ),
        ),
        (report["evidence"]["analyzer"], ("coverage", "optimizations")),
        (
            report["evidence"]["detail_profile"],
            ("hardware_time_us", "memory_time_us", "ddr_read_bytes", "ddr_write_bytes"),
        ),
    ]
    sections += [(row, ("nodes",)) for row in report["model"]["components"]]
    sections += [
        (row, ("hardware_time_us", "memory_time_us", "dram_bytes"))
        for row in report["baseline"]["hotspots"]
    ]
    for row, fields in sections:
        row["missing_reasons"] = {
            key: "Not collected in this fixture" for key in fields if row.get(key) in (None, "", [])
        }
    return report


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    champion = source / "model.onnx"
    champion.write_bytes(b"onnx-champion")
    companion = source / "qnn_context.bin"
    companion.write_bytes(b"qnn-context")
    config = source / "winml_build_config.json"
    config.write_text(
        json.dumps({"ep": "qnn", "device": "npu", "optim": {"level": "all"}}),
        encoding="utf-8",
    )
    report = source / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    return report, champion, config, companion


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _refresh_manifest_entry(output: Path, relative: str) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    target = output / relative
    entry["size_bytes"] = target.stat().st_size
    entry["sha256"] = _sha256(target)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _valid_lock(assets: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "available",
        "source": {
            "kind": "huggingface",
            "id": "example/model",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "prepared_model_sha256": "a" * 64,
        },
        "toolchain": {
            "winml": {
                "kind": "git",
                "revision": "fedcba9876543210fedcba9876543210fedcba98",
            },
            "python": "3.11.14",
            "runtime": "onnxruntime",
            "provider": "QNNExecutionProvider",
            "sdk": "declared-sdk",
            "device": "declared-device",
            "driver": "declared-driver",
        },
        "provider_options": {"performance_mode": "burst"},
        "inputs": [
            {
                "path": asset.name,
                "sha256": _sha256(asset),
                "purpose": "performance" if asset.name == "perf_input.npz" else "correctness",
            }
            for asset in assets
        ],
        "expected": {
            "public_io": {"status": "preserved"},
            "correctness": {"status": "pass"},
            "topology": {"status": "pass"},
            "performance": {"acceptance": "bounded threshold recorded"},
        },
        "replay_validation": {"status": "pass", "clean_directory": True},
    }


def _repro_inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    source = tmp_path / "repro-source"
    source.mkdir()
    rebuild_config = source / "effective_rebuild_config.json"
    _write_json(
        rebuild_config,
        {
            "source": "example/model",
            "target_ep": "QNNExecutionProvider",
            "device": "NPU",
            "skip_optimize": False,
        },
    )
    repro_script = source / "run_repro.ps1"
    repro_script.write_text(
        VALID_REPRO_RUN_BODY + "pwsh -File (Join-Path $Root 'build.ps1')\n",
        encoding="utf-8",
    )
    assets = [
        source / "perf_input.npz",
        source / "eval_inputs.npz",
        source / "inputs_manifest.json",
    ]
    assets[0].write_bytes(b"perf-input")
    assets[1].write_bytes(b"eval-input")
    _write_json(assets[2], {"inputs": ["perf_input.npz", "eval_inputs.npz"]})
    repro_lock = source / "portable_repro.lock.json"
    _write_json(repro_lock, _valid_lock(assets))
    return rebuild_config, repro_script, repro_lock, assets


def test_finalize_output_publishes_complete_hash_bound_bundle(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"

    result = output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
    )

    assert result == output.resolve()
    assert {path.name for path in output.iterdir()} == {
        "champion.onnx",
        "qnn_context.bin",
        "winml_config.json",
        "report.json",
        "report.html",
        "manifest.json",
    }
    assert json.loads((output / "winml_config.json").read_text(encoding="utf-8"))["device"] == "npu"
    final_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert final_report["leader"]["model_path"] == "champion.onnx"
    assert final_report["artifacts"] == {
        "champion_onnx": "champion.onnx",
        "companions": [{"path": "qnn_context.bin", "role": "companion"}],
        "winml_config": "winml_config.json",
        "manifest": "manifest.json",
    }
    manifest = output_module.validate_output_bundle(output)
    assert manifest["schema_version"] == 1
    assert manifest["champion_dependencies"] == ["qnn_context.bin"]
    assert {entry["path"] for entry in manifest["files"]} == {
        "champion.onnx",
        "qnn_context.bin",
        "winml_config.json",
        "report.json",
        "report.html",
    }
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert 'href="champion.onnx"' in html
    assert 'href="qnn_context.bin"' in html


def test_finalize_output_publishes_portable_reproduction_assets(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=list(reversed(assets)),
    )

    expected_reproduction = {
        "script": "repro.ps1",
        "run_script": "repro-run.ps1",
        "rebuild_config": "rebuild_config.json",
        "lock": "repro.lock.json",
        "assets": ["eval_inputs.npz", "inputs_manifest.json", "perf_input.npz"],
    }
    assert {path.name for path in output.iterdir()} == {
        "champion.onnx",
        "qnn_context.bin",
        "winml_config.json",
        "rebuild_config.json",
        "repro.ps1",
        "repro-run.ps1",
        "repro.lock.json",
        "eval_inputs.npz",
        "inputs_manifest.json",
        "perf_input.npz",
        "report.json",
        "report.html",
        "manifest.json",
    }
    manifest = output_module.validate_output_bundle(output)
    assert manifest["champion_dependencies"] == ["qnn_context.bin"]
    assert manifest["reproduction"] == expected_reproduction
    assert (output / "repro-run.ps1").read_bytes() == repro_script.read_bytes()
    assert (output / "repro.ps1").read_bytes() != repro_script.read_bytes()
    roles = {entry["path"]: entry["role"] for entry in manifest["files"]}
    assert roles["repro.ps1"] == "reproduction_wrapper"
    assert roles["repro-run.ps1"] == "reproduction_script"
    assert roles["rebuild_config.json"] == "rebuild_config"
    assert roles["repro.lock.json"] == "reproduction_lock"
    for asset_name in expected_reproduction["assets"]:
        assert roles[asset_name] == "reproduction_asset"
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert entry["sha256"] == _sha256(path)

    final_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert final_report["artifacts"]["reproduction"] == expected_reproduction
    assert final_report["conclusion"]["reproduce"] == ["pwsh -File ./repro.ps1"]


def test_generated_repro_wrapper_is_deterministic_and_validates_before_replay(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"

    for output in (first_output, second_output):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            output,
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )

    first_wrapper = (first_output / "repro.ps1").read_text(encoding="utf-8")
    second_wrapper = (second_output / "repro.ps1").read_text(encoding="utf-8")
    assert first_wrapper == second_wrapper
    for required in (
        "param([switch]$ValidateOnly)",
        "manifest.json",
        "repro.lock.json",
        "Get-FileHash",
        "Get-Command winml",
        "size_bytes",
        "sha256",
        "repro-run.ps1",
    ):
        assert required in first_wrapper
    assert first_wrapper.index("if ($ValidateOnly)") < first_wrapper.index("repro-run.ps1")
    assert "& (Join-Path $PSScriptRoot 'repro-run.ps1')" in first_wrapper
    assert "exit $LASTEXITCODE" in first_wrapper
    assert "./repro-run.ps1" not in first_wrapper
    assert ".\\repro-run.ps1" not in first_wrapper


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_generated_repro_wrapper_propagates_replay_exit_code(
    output_module: ModuleType,
    tmp_path: Path,
    exit_code: int,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(
        VALID_REPRO_RUN_BODY + f"exit {exit_code}\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )

    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    (stub_dir / "winml.cmd").write_text(
        "@echo off\r\nexit /b 0\r\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")

    command = ["pwsh", "-NoProfile", "-File", str(output / "repro.ps1")]
    run_process = subprocess.run
    result = run_process(
        command,
        cwd=output,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == exit_code, result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh is required")
def test_wrapper_stops_before_later_native_success(output_module, tmp_path):
    wrapper = tmp_path / "repro.ps1"
    wrapper.write_text(output_module.REPRO_WRAPPER_TEXT, encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"files": []}', encoding="utf-8")
    (tmp_path / "repro.lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repro-run.ps1").write_text(
        chr(10).join(
            [
                "$Root = $PSScriptRoot",
                'pwsh -NoProfile -Command "exit 7"',
                'Set-Content (Join-Path $Root "unexpected") yes',
                'pwsh -NoProfile -Command "exit 0"',
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603 -- generated fixture and fixed PowerShell command
        [shutil.which("pwsh"), "-NoProfile", "-Command", "function winml {}; & ./repro.ps1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "unexpected").exists()


def test_invalid_performance_does_not_publish_bundle(output_module, tmp_path):
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["leader"]["ci_low_pct"] = float("nan")
    _write_json(report, facts)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="finite"):
        output_module.finalize_output(report, champion, config, [companion], output)
    assert not output.exists()


def test_legacy_bundle_has_no_reproduction_object_and_still_validates(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"

    output_module.finalize_output(report, champion, config, [companion], output)

    manifest = output_module.validate_output_bundle(output)
    final_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert "reproduction" not in manifest
    assert "reproduction" not in final_report["artifacts"]
    assert final_report["conclusion"]["reproduce"] == ["winml build ..."]


def test_missing_companion_does_not_publish_partial_output(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    companion.unlink()
    output = tmp_path / "output"

    with pytest.raises(output_module.OutputBundleError, match="companion"):
        output_module.finalize_output(report, champion, config, [companion], output)

    assert not output.exists()


def test_invalid_config_does_not_publish_partial_output(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    config.write_text("not-json", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(output_module.OutputBundleError, match="WinML config"):
        output_module.finalize_output(report, champion, config, [companion], output)

    assert not output.exists()


@pytest.mark.parametrize(
    "missing_name",
    ["rebuild_config", "repro_script", "repro_lock"],
)
def test_partial_reproduction_primary_trio_is_rejected_without_publishing(
    output_module: ModuleType,
    tmp_path: Path,
    missing_name: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    kwargs = {
        "rebuild_config": rebuild_config,
        "repro_script": repro_script,
        "repro_lock": repro_lock,
        "repro_assets": assets,
    }
    kwargs[missing_name] = None
    output = tmp_path / "output"

    with pytest.raises(output_module.OutputBundleError, match="reproduction"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            output,
            **kwargs,
        )

    assert not output.exists()


def test_reproduction_assets_without_primary_trio_are_rejected(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    _, _, _, assets = _repro_inputs(tmp_path)

    with pytest.raises(output_module.OutputBundleError, match="reproduction"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    ("rewrite", "match"),
    [
        (lambda path: path.unlink(), "rebuild config"),
        (lambda path: path.write_text("not-json", encoding="utf-8"), "rebuild config"),
        (lambda path: _write_json(path, {}), "rebuild config"),
        (lambda path: _write_json(path, {"skip_optimize": True}), "skip_optimize"),
    ],
)
def test_invalid_rebuild_config_is_rejected(
    output_module: ModuleType,
    tmp_path: Path,
    rewrite: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    rewrite(rebuild_config)

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    ("script_bytes", "match"),
    [
        (b"Write-Host repro\n", "PSScriptRoot"),
        ((VALID_REPRO_RUN_BODY + "C:\\temp\\model.onnx\n").encode(), "absolute"),
        (
            (VALID_REPRO_RUN_BODY + "\\\\server\\share\\model.onnx\n").encode(),
            "absolute",
        ),
        ((VALID_REPRO_RUN_BODY + "/" + "tmp/model.onnx\n").encode(), "absolute"),
        (
            (VALID_REPRO_RUN_BODY + "//server/share/model.onnx\n").encode(),
            "absolute",
        ),
        (
            (VALID_REPRO_RUN_BODY + "copy --share=//server/share/model.onnx\n").encode(),
            "absolute",
        ),
        ((VALID_REPRO_RUN_BODY + '"//tmp/path"\n').encode(), "absolute"),
        (b"$PSScriptRoot\n\xff", "UTF-8"),
    ],
)
def test_invalid_repro_script_is_rejected(
    output_module: ModuleType,
    tmp_path: Path,
    script_bytes: bytes,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_bytes(script_bytes)

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    "script_text",
    [
        "$Root = $PSScriptRoot\npwsh /NoProfile -File (Join-Path $Root 'build.ps1')\n",
        "$Root = $PSScriptRoot\npwsh.exe /NoProfile -File (Join-Path $Root 'build.ps1')\n",
        "$Root = $PSScriptRoot\npowershell /NoProfile -File (Join-Path $Root 'build.ps1')\n",
        "$Root = $PSScriptRoot\npowershell.exe '/NoProfile' -File (Join-Path $Root 'build.ps1')\n",
        "$Root = $PSScriptRoot\ncmd /c echo repro\n",
        "$Root = $PSScriptRoot\ncmd.exe '/c' echo repro\n",
    ],
)
def test_repro_script_allows_bare_shell_switches(
    output_module: ModuleType,
    tmp_path: Path,
    script_text: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(VALID_REPRO_RUN_BODY + script_text, encoding="utf-8")

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        tmp_path / "output",
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )


@pytest.mark.parametrize(
    "script_text",
    [
        (
            "# fake param([switch]$ValidateOnly)\n"
            "# fake if ($ValidateOnly) { exit 0 }\n"
            "param ( [switch] $ValidateOnly )\n"
            "if ( $ValidateOnly ) {\n"
            "    exit 0\n"
            "}\n"
            "$Root = $PSScriptRoot\n"
        ),
        (
            "'param([switch]$ValidateOnly)'\n"
            '"if ($ValidateOnly) { exit 0 }"\n'
            "@'\n"
            "pwsh -File (Join-Path $Root 'build.ps1')\n"
            "& $WinMLExe build\n"
            "'@\n"
            "param([switch]$ValidateOnly)\n"
            "if ($ValidateOnly) { exit 0 }\n"
            "$Root = $PSScriptRoot\n"
        ),
        (
            "param([switch]$ValidateOnly)\n"
            "if ($ValidateOnly) { exit 0 }\n"
            "$Root = $PSScriptRoot\n"
            "Write-Host 'replay'\n"
            "winml build --model model.onnx\n"
            "$WinMLExe = Join-Path $Root 'winml.exe'\n"
            "& $WinMLExe build\n"
            "pwsh -File (Join-Path $Root 'build.ps1')\n"
        ),
    ],
)
def test_repro_script_accepts_arbitrary_bundle_relative_body(
    output_module: ModuleType,
    tmp_path: Path,
    script_text: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(script_text, encoding="utf-8")

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        tmp_path / "output",
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )


@pytest.mark.parametrize(
    "script_text",
    [
        "$PSScriptRoot\nwinml build /tmp\n",
        "$PSScriptRoot\nwinml build '/opt'\n",
        "$PSScriptRoot\nwinml build /c\n",
        "$PSScriptRoot\nwinml build '/c'\n",
        "$PSScriptRoot\nwinml build /NoProfile\n",
        "$PSScriptRoot\nwinml build --model=C:\\temp\\model.onnx\n",
        '$PSScriptRoot\nwinml build "--model=C:\\temp\\model.onnx"\n',
        "$PSScriptRoot\nwinml build \\temp\\model.onnx\n",
        "$PSScriptRoot\nwinml build '\\temp\\model.onnx'\n",
        "$PSScriptRoot\nwinml build --model=\\temp\\model.onnx\n",
        '$PSScriptRoot\nwinml build "--model=\\temp\\model.onnx"\n',
        "$PSScriptRoot\nwinml build -Input=/tmp/model.onnx\n",
        "$PSScriptRoot\nwinml build '-Input=/tmp/model.onnx'\n",
        "$PSScriptRoot\nwinml build --root=/c\n",
        "$PSScriptRoot\nwinml build --root=/NoProfile\n",
        "$PSScriptRoot\nwinml build '--root=/c'\n",
        "$PSScriptRoot\nwinml build --root=/models\n",
        "$PSScriptRoot\ncopy --share=\\\\server\\share\\file\n",
        '$PSScriptRoot\ncopy "--share=\\\\server\\share\\file"\n',
    ],
)
def test_repro_script_rejects_assignment_form_absolute_paths(
    output_module: ModuleType,
    tmp_path: Path,
    script_text: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(VALID_REPRO_RUN_BODY + script_text, encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="absolute"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    "script_text",
    [
        "$PSScriptRoot\nwinml build --model='C:\\Private Dir\\model.onnx'\n",
        "$PSScriptRoot\nwinml build --model='/opt/Private Dir/model.onnx'\n",
        "$PSScriptRoot\n$Model='/opt/Private Dir/model.onnx'\n",
        "$PSScriptRoot\nwinml build --model='\\\\server\\Private Dir\\model.onnx'\n",
    ],
)
def test_repro_script_rejects_quoted_assignment_absolute_paths_with_spaces(
    output_module: ModuleType,
    tmp_path: Path,
    script_text: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(script_text, encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="absolute"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


def test_repro_script_preserves_quoted_assignment_relative_path_with_spaces(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    script_text = "$Root = $PSScriptRoot\nwinml build --model='models/Private Dir/model.onnx'\n"
    script_bytes = script_text.encode("utf-8")
    repro_script.write_bytes(script_bytes)
    output = tmp_path / "output"

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )

    assert (output / "repro-run.ps1").read_bytes() == script_bytes


@pytest.mark.parametrize(
    "script_argument",
    [
        "--model=`/opt/Private` Dir/model.onnx",
        r"--model=`\`\server`\Private` Dir`\model.onnx",
    ],
)
def test_repro_script_rejects_backtick_escaped_absolute_paths(
    output_module: ModuleType,
    tmp_path: Path,
    script_argument: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(
        VALID_REPRO_RUN_BODY + f"winml build {script_argument}\n",
        encoding="utf-8",
    )

    with pytest.raises(output_module.OutputBundleError, match="absolute"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


def test_repro_script_preserves_backtick_escaped_relative_path_bytes(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    script_text = VALID_REPRO_RUN_BODY + "winml build --model=models/Private` Dir/model.onnx\n"
    script_bytes = script_text.encode("utf-8")
    repro_script.write_bytes(script_bytes)
    output = tmp_path / "output"

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )

    assert (output / "repro-run.ps1").read_bytes() == script_bytes


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda lock: lock.pop("source"), "source"),
        (lambda lock: lock.update({"status": "pending"}), "status"),
        (lambda lock: lock.update({"status": "requires-unmerged-pr"}), "dependencies"),
        (
            lambda lock: lock.update(
                {
                    "status": "requires-unmerged-pr",
                    "dependencies": [{"url": "https://example.invalid/pr", "revision": "bad"}],
                }
            ),
            "revision",
        ),
        (
            lambda lock: lock["source"].update({"revision": "ABC" + "0" * 37}),
            "revision",
        ),
        (
            lambda lock: lock["source"].update({"prepared_model_sha256": "b" * 63}),
            "prepared",
        ),
        (lambda lock: lock["toolchain"]["winml"].update({"revision": "bad"}), "winml"),
        (lambda lock: lock.update({"provider_options": []}), "provider_options"),
        (lambda lock: lock["expected"].update({"topology": {}}), "topology"),
        (
            lambda lock: lock.update(
                {"replay_validation": {"status": "pass", "clean_directory": False}}
            ),
            "replay",
        ),
    ],
)
def test_invalid_repro_lock_schema_is_rejected(
    output_module: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    mutate(lock)
    _write_json(repro_lock, lock)

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


def test_repro_lock_accepts_release_winml_identity(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    lock["toolchain"]["winml"] = {
        "kind": "release",
        "version": "winml-cli 1.2.3+cpu.4",
    }
    _write_json(repro_lock, lock)

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        tmp_path / "output",
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda lock: lock["toolchain"]["winml"].update({"version": "1.2.3"}),
            "version",
        ),
        (
            lambda lock: lock["toolchain"].update({"winml": {"kind": "release"}}),
            "version",
        ),
        (
            lambda lock: lock["toolchain"].update({"winml": {"kind": "release", "version": ""}}),
            "version",
        ),
        (
            lambda lock: lock["toolchain"].update(
                {
                    "winml": {
                        "kind": "release",
                        "version": "1.2.3",
                        "revision": "f" * 40,
                    }
                }
            ),
            "revision",
        ),
        (
            lambda lock: lock["toolchain"].update({"winml": {"kind": "git", "version": "1.2.3"}}),
            "version",
        ),
        (
            lambda lock: lock["toolchain"].update(
                {"winml": {"kind": "archive", "version": "1.2.3"}}
            ),
            "kind",
        ),
        (lambda lock: lock["toolchain"].pop("winml"), "winml"),
    ],
)
def test_repro_lock_rejects_ambiguous_or_missing_winml_identity(
    output_module: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    mutate(lock)
    _write_json(repro_lock, lock)

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    "input_path",
    ["C:/temp/perf_input.npz", "../perf_input.npz", "nested/perf_input.npz"],
)
def test_repro_lock_rejects_unsafe_input_paths(
    output_module: ModuleType,
    tmp_path: Path,
    input_path: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    lock["inputs"][0]["path"] = input_path
    _write_json(repro_lock, lock)

    with pytest.raises(output_module.OutputBundleError, match="basename"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda lock, asset_list: lock["inputs"].append(dict(lock["inputs"][0])),
            "duplicate",
        ),
        (
            lambda lock, asset_list: lock["inputs"][0].update({"path": "missing.npz"}),
            "missing",
        ),
        (
            lambda lock, asset_list: lock["inputs"][0].update({"sha256": "0" * 64}),
            "hash",
        ),
        (lambda lock, asset_list: lock["inputs"][0].update({"purpose": ""}), "purpose"),
        (
            lambda lock, asset_list: asset_list.append(asset_list[0].with_name("undeclared.npz")),
            "undeclared",
        ),
    ],
)
def test_repro_lock_inputs_must_match_assets_exactly(
    output_module: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    undeclared = assets[0].with_name("undeclared.npz")
    undeclared.write_bytes(b"undeclared")
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    repro_assets = list(assets)
    mutate(lock, repro_assets)
    _write_json(repro_lock, lock)

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=repro_assets,
        )


@pytest.mark.parametrize(
    "collision_name",
    ["REPRO.PS1", "rebuild_CONFIG.json", "REPRO.LOCK.JSON", "QNN_CONTEXT.BIN"],
)
def test_reproduction_reserved_and_companion_name_collisions_are_rejected(
    output_module: ModuleType,
    tmp_path: Path,
    collision_name: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    collision = assets[0].with_name(collision_name)
    collision.write_bytes(b"collision")
    lock = json.loads(repro_lock.read_text(encoding="utf-8"))
    lock["inputs"].append(
        {"path": collision.name, "sha256": _sha256(collision), "purpose": "collision"}
    )
    _write_json(repro_lock, lock)

    with pytest.raises(output_module.OutputBundleError, match="collision"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            tmp_path / "output",
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=[*assets, collision],
        )


def test_reserved_or_duplicate_companion_names_are_rejected(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    collision = companion.parent / "report.json"
    collision.write_bytes(b"collision")

    with pytest.raises(output_module.OutputBundleError, match="reserved"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [collision],
            tmp_path / "output",
        )


@pytest.mark.parametrize("status", ["inconclusive", "unconfirmed"])
def test_unconfirmed_leader_is_not_deliverable(
    output_module: ModuleType,
    tmp_path: Path,
    status: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["leader"]["status"] = status
    report.write_text(json.dumps(facts), encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="confirmed"):
        output_module.finalize_output(report, champion, config, [companion], tmp_path / "output")


def test_nonpassing_correctness_is_not_deliverable(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["leader"]["correctness"] = "bypass"
    report.write_text(json.dumps(facts), encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="correctness"):
        output_module.finalize_output(report, champion, config, [companion], tmp_path / "output")


def test_disclosed_provisional_quality_is_deliverable(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["leader"]["status"] = "confirmed-performance-provisional-quality"
    facts["leader"]["quality"] = (
        "Task evaluator unavailable; tensor validation passed and evidence gap disclosed."
    )
    facts["leader"]["quality_gate"] = {
        "task_evaluator": "unavailable",
        "tensor_validation": "pass",
        "evidence_gap": "Task-level acceptance remains unavailable.",
    }
    report.write_text(json.dumps(facts), encoding="utf-8")

    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)

    output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    "quality_gate",
    [
        None,
        {},
        {
            "task_evaluator": "available",
            "tensor_validation": "pass",
            "evidence_gap": "Task-level acceptance remains unavailable.",
        },
        {
            "task_evaluator": "unavailable",
            "tensor_validation": "failed",
            "evidence_gap": "Task-level acceptance remains unavailable.",
        },
        {
            "task_evaluator": "unavailable",
            "tensor_validation": "pass",
            "evidence_gap": "",
        },
    ],
)
def test_provisional_quality_requires_structured_gate(
    output_module: ModuleType,
    tmp_path: Path,
    quality_gate: object,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["leader"]["status"] = "confirmed-performance-provisional-quality"
    facts["leader"]["quality"] = "provisional"
    if quality_gate is not None:
        facts["leader"]["quality_gate"] = quality_gate
    report.write_text(json.dumps(facts), encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="quality_gate"):
        output_module.finalize_output(report, champion, config, [companion], tmp_path / "output")


def test_existing_output_requires_overwrite(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="already exists"):
        output_module.finalize_output(report, champion, config, [companion], output)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_manifest_validation_detects_tampering(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)
    (output / "champion.onnx").write_bytes(b"tampered")

    with pytest.raises(output_module.OutputBundleError, match="hash mismatch"):
        output_module.validate_output_bundle(output)


def test_manifest_validation_rejects_untracked_files(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)
    (output / "stale_context.bin").write_bytes(b"untracked")

    with pytest.raises(output_module.OutputBundleError, match="untracked"):
        output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("champion", "wrong.onnx"),
        ("winml_config", "wrong-config.json"),
        ("report_json", "wrong-report.json"),
        ("report_html", "wrong-report.html"),
    ],
)
def test_manifest_validation_rejects_tampered_delivery_pointers(
    output_module: ModuleType,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match=field):
        output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda manifest: manifest["reproduction"].update({"script": "wrong.ps1"}),
            "reproduction",
        ),
        (
            lambda manifest: next(
                entry for entry in manifest["files"] if entry["path"] == "repro.ps1"
            ).update({"role": "companion"}),
            "reproduction_wrapper",
        ),
    ],
)
def test_manifest_validation_rejects_tampered_reproduction_pointers_or_roles(
    output_module: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.validate_output_bundle(output)


def test_manifest_validation_rejects_reproduction_file_hash_tampering(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )
    (output / "repro.lock.json").write_text("{}", encoding="utf-8")

    with pytest.raises(output_module.OutputBundleError, match="hash mismatch"):
        output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ("not-json", "rebuild config"),
        (json.dumps({}), "rebuild config"),
        (json.dumps({"skip_optimize": True}), "skip_optimize"),
    ],
)
def test_manifest_validation_rejects_semantically_invalid_rebuild_config_tampering(
    output_module: ModuleType,
    tmp_path: Path,
    replacement: str,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )
    (output / "rebuild_config.json").write_text(replacement, encoding="utf-8")
    _refresh_manifest_entry(output, "rebuild_config.json")

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (
            (
                "$Root = $PSScriptRoot\n"
                "if ($ValidateOnly) { pwsh -File (Join-Path $Root 'build.ps1') -ValidateOnly }\n"
                "pwsh -File (Join-Path $Root 'build.ps1')\n"
            ),
            "wrapper",
        ),
        (
            (
                "param([switch]$ValidateOnly)\n"
                "$Root = $PSScriptRoot\n"
                "pwsh -File (Join-Path $Root 'build.ps1')\n"
            ),
            "wrapper",
        ),
    ],
)
def test_manifest_validation_rejects_semantically_invalid_repro_script_tampering(
    output_module: ModuleType,
    tmp_path: Path,
    replacement: str,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )
    (output / "repro.ps1").write_text(replacement, encoding="utf-8")
    _refresh_manifest_entry(output, "repro.ps1")

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.validate_output_bundle(output)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda lock: lock["toolchain"]["winml"].update({"version": "1.2.3"}),
            "version",
        ),
        (
            lambda lock: lock["toolchain"].update(
                {"winml": {"kind": "release", "revision": "f" * 40}}
            ),
            "revision",
        ),
    ],
)
def test_manifest_validation_rejects_semantically_invalid_repro_lock_tampering(
    output_module: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        rebuild_config=rebuild_config,
        repro_script=repro_script,
        repro_lock=repro_lock,
        repro_assets=assets,
    )
    lock_path = output / "repro.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mutate(lock)
    _write_json(lock_path, lock)
    _refresh_manifest_entry(output, "repro.lock.json")

    with pytest.raises(output_module.OutputBundleError, match=match):
        output_module.validate_output_bundle(output)


def test_overwrite_replaces_bundle_only_after_new_bundle_validates(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)
    first_hash = hashlib.sha256((output / "champion.onnx").read_bytes()).hexdigest()
    champion.write_bytes(b"new-champion")

    output_module.finalize_output(
        report,
        champion,
        config,
        [companion],
        output,
        overwrite=True,
    )

    assert hashlib.sha256((output / "champion.onnx").read_bytes()).hexdigest() != first_hash
    output_module.validate_output_bundle(output)


def test_invalid_reproduction_overwrite_preserves_existing_bundle(
    output_module: ModuleType,
    tmp_path: Path,
) -> None:
    report, champion, config, companion = _inputs(tmp_path)
    output = tmp_path / "output"
    output_module.finalize_output(report, champion, config, [companion], output)
    original_champion = (output / "champion.onnx").read_bytes()
    original_manifest = (output / "manifest.json").read_bytes()
    champion.write_bytes(b"replacement-champion")
    rebuild_config, repro_script, repro_lock, assets = _repro_inputs(tmp_path)
    repro_script.write_text(
        VALID_REPRO_RUN_BODY + "winml build --model=C:\\temp\\model.onnx\n",
        encoding="utf-8",
    )

    with pytest.raises(output_module.OutputBundleError, match="absolute"):
        output_module.finalize_output(
            report,
            champion,
            config,
            [companion],
            output,
            overwrite=True,
            rebuild_config=rebuild_config,
            repro_script=repro_script,
            repro_lock=repro_lock,
            repro_assets=assets,
        )

    assert (output / "champion.onnx").read_bytes() == original_champion
    assert (output / "manifest.json").read_bytes() == original_manifest
    output_module.validate_output_bundle(output)


def test_finalizer_rejects_unfinished_closure(output_module, tmp_path):
    report, champion, config, companion = _inputs(tmp_path)
    facts = json.loads(report.read_text(encoding="utf-8"))
    facts["capability_closure"]["coverage_verdict"] = "INSUFFICIENT_EVIDENCE"
    report.write_text(json.dumps(facts), encoding="utf-8")
    with pytest.raises(ValueError, match="closure"):
        output_module.finalize_output(report, champion, config, [companion], tmp_path / "output")
    assert not (tmp_path / "output").exists()
