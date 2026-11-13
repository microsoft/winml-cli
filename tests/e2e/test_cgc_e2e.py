# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Real-model CGC command coverage on every Windows GPU vendor.

Run with ``uv run --no-sync pytest -m e2e tests/e2e/test_cgc_e2e.py``.
Requires preview windowsml/ORT wheels and a CGC-capable GPU driver; missing
packages and backend failures are errors, not skips.

Lazy module fixtures run HF -> ONNX and ONNX -> Input IR once. Export tests
assert those same command results; consumers request artifacts, never earlier
tests. Direct HF -> Input IR is a separate CLI route with an unavoidable
internal ONNX export, but reuses the downloaded HF checkpoint. Text and bytecode
compile coverage replaces test_compile_cgc_ir_e2e.py, including its overwrite
and invalid-input guards. Native compile validation checks reload/build/schema;
the separate perf and compare tests exercise actual Output IR inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import onnx
import pytest

from winml.modelkit.onnx import is_compiled_onnx


pytestmark = pytest.mark.e2e
_INPUT_ENCODINGS = ["text", "bytecode"]
if os.environ.get("WINML_TEST_CGC_INPUT_IR"):
    _INPUT_ENCODINGS.append("caller")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _io_names(model_path: Path) -> tuple[list[str], list[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    initializers.update(tensor.values.name for tensor in model.graph.sparse_initializer)
    return (
        [value.name for value in model.graph.input if value.name not in initializers],
        [value.name for value in model.graph.output],
    )


def _metadata(path: Path) -> dict:
    return json.loads(path.with_name(f"{path.stem}_metadata.json").read_text(encoding="utf-8"))


def _assert_input_ir(artifact, reference: Path) -> None:
    path, result = artifact
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CGC export complete" in result.stdout
    assert path.read_text(encoding="utf-8").strip()
    metadata = _metadata(path)
    assert metadata["format"] == "cgc-input-ir"
    for key, names in zip(("inputs", "outputs"), _io_names(reference), strict=True):
        assert metadata[key] == [
            {"name": name, "index": index} for index, name in enumerate(names)
        ]


def _assert_perf(
    report: Path, model: Path, reference_onnx: Path, *, runtime: str, ep: str | None = None,
) -> None:
    data = json.loads(report.read_text(encoding="utf-8"))
    info = data["benchmark_info"]
    assert Path(info["model_id"]).samefile(model)
    assert Path(info["running_model_path"]).samefile(model)
    assert info["runtime"] == runtime
    assert info["device"] == "gpu"
    assert info["warmup"] == 1
    assert info["iterations"] == 5
    if ep is not None:
        assert info["ep"] == ep
    else:
        assert info["backend"] == "cgc"
    assert len(data["raw_samples_ms"]) == 5
    assert all(math.isfinite(value) and value >= 0 for value in data["raw_samples_ms"])
    assert all(math.isfinite(value) and value >= 0 for value in data["latency_ms"].values())
    inputs, outputs = _io_names(reference_onnx)
    assert data["model_info"]["input_names"] == inputs
    assert data["model_info"]["output_names"] == outputs


def _perf(cgc_cli, model: Path, output: Path, *options: str) -> None:
    cgc_cli(
        "perf", "-m", str(model), "--device", "gpu", "--warmup", "1",
        "--iterations", "5", "--no-memory", "-o", str(output), *options,
    )


# Command: winml export -m <source.onnx> --target cgir -o <input.mlir>.
# The shared fixture executes it once; this test checks the resulting IR and I/O metadata.
def test_export_local_onnx_to_input_ir(cgc_input_ir, cgc_source_onnx: Path) -> None:
    _assert_input_ir(cgc_input_ir, cgc_source_onnx)


# Command: winml export -m <HF model ID> --target cgir -o <hf-input.mlir>.
# Check the direct HF route separately from exporting an existing ONNX file.
def test_export_hf_to_input_ir(cgc_hf_input_ir, cgc_source_onnx: Path) -> None:
    _assert_input_ir(cgc_hf_input_ir, cgc_source_onnx)


# Command: winml perf -m <input.mlir> --device gpu.
# Verify automatic Runtime CGC selection and a one-warmup, five-iteration report.
def test_perf_input_ir_auto_runtime(
    cgc_cli, cgc_input_ir, cgc_source_onnx: Path, tmp_path: Path,
) -> None:
    output = tmp_path / "input-perf.json"
    _perf(cgc_cli, cgc_input_ir[0], output)
    _assert_perf(output, cgc_input_ir[0], cgc_source_onnx, runtime="winml-runtime")


# Command: winml perf -m <source.onnx> --runtime winml-runtime --device gpu.
# Exercise online CGC compilation and execution of the exact source ONNX.
def test_perf_onnx_runtime(cgc_cli, cgc_source_onnx: Path, tmp_path: Path) -> None:
    output = tmp_path / "onnx-runtime-perf.json"
    _perf(cgc_cli, cgc_source_onnx, output, "--runtime", "winml-runtime")
    _assert_perf(output, cgc_source_onnx, cgc_source_onnx, runtime="winml-runtime")


# Command: winml eval -m <input/output.mlir> --runtime winml-runtime --mode compare
# --reference <source.onnx> --reference-device gpu --reference-ep dml --samples 5.
# Compare both IR forms against real ONNX outputs using each command's default GPU selection.
@pytest.mark.parametrize("ir_fixture", ["cgc_input_ir", "cgc_output_ir"], ids=["input", "output"])
def test_eval_ir_against_onnx_dml(
    request: pytest.FixtureRequest, ir_fixture: str, cgc_cli,
    cgc_source_onnx: Path, tmp_path: Path,
) -> None:
    artifact, _ = request.getfixturevalue(ir_fixture)
    output = tmp_path / "compare.json"
    cgc_cli(
        "eval", "-m", str(artifact), "--runtime", "winml-runtime",
        "--device", "gpu",
        "--mode", "compare", "--reference", str(cgc_source_onnx),
        "--reference-device", "gpu",
        "--reference-ep", "dml", "--samples", "5", "-o", str(output),
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert Path(data["model_path"]).samefile(artifact)
    assert Path(data["reference_path"]).samefile(cgc_source_onnx)
    assert data["dataset"]["samples"] == 5
    assert data["runtime"] == "winml-runtime"
    assert data["reference_device"] == "gpu"
    assert data["reference_ep"] == "dml"
    _, outputs = _io_names(cgc_source_onnx)
    metrics = data["metrics"]
    for metric in ("cosine_similarity", "mse", "max_abs_diff"):
        for stat in ("mean", "std", "min", "max"):
            values = metrics[f"{metric}_{stat}"]
            assert set(values) == set(outputs)
            assert all(math.isfinite(value) for value in values.values()), values
    # Compare every generated sample to actual DML outputs, not canned tensor values.
    for name in outputs:
        assert metrics["cosine_similarity_min"][name] >= 0.99, metrics
        # A 40 dB SQNR requires relative L2 error <= 1%; +inf means identical.
        assert metrics["sqnr_db_min"][name] >= 40, metrics
        assert metrics["mse_min"][name] >= 0, metrics
        assert metrics["max_abs_diff_min"][name] >= 0, metrics


# Command: winml perf -m <source.onnx> --ep winmlcg --device gpu.
# Exercise the ORT WinMLCG EP path, distinct from the Windows ML Runtime API.
def test_perf_onnx_winmlcg(cgc_cli, cgc_source_onnx: Path, tmp_path: Path) -> None:
    output = tmp_path / "winmlcg-perf.json"
    _perf(cgc_cli, cgc_source_onnx, output, "--ep", "winmlcg")
    _assert_perf(
        output, cgc_source_onnx, cgc_source_onnx,
        runtime="winml-ort", ep="WinMLCGExecutionProvider",
    )


# Commands: winml compile -m <source.onnx> --ep winmlcg -o <context.onnx>,
# then winml perf -m <context.onnx> --ep winmlcg --device gpu.
# Inspect the EPContext and reload that exact file in a fresh CLI process.
def test_compile_onnx_winmlcg_and_reload(
    cgc_cli, cgc_source_onnx: Path, tmp_path: Path,
) -> None:
    compiled = tmp_path / "context.onnx"
    cgc_cli(
        "compile", "-m", str(cgc_source_onnx), "--device", "gpu", "--ep", "winmlcg",
        "-o", str(compiled),
    )
    assert is_compiled_onnx(compiled)
    model = onnx.load(str(compiled), load_external_data=False)
    context_nodes = [node for node in model.graph.node if node.op_type == "EPContext"]
    assert context_nodes
    for node in context_nodes:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        assert attrs["ep_type"] == b"WinMLCGExecutionProvider"
        context = attrs["ep_cache_context"]
        assert context
        if not attrs.get("embed_mode", 0):
            assert (compiled.parent / context.decode("utf-8")).is_file()
    output = tmp_path / "reloaded-perf.json"
    _perf(cgc_cli, compiled, output, "--ep", "winmlcg")
    _assert_perf(
        output, compiled, cgc_source_onnx, runtime="winml-ort", ep="WinMLCGExecutionProvider",
    )


# Command: winml compile -m <input.mlir/input.mlirbc> --device gpu
# -o <output.mlir> (also accepts explicit caller input).
# Check shared compilation results, resource integrity and overwrite/invalid-input rejection.
@pytest.mark.parametrize("encoding", _INPUT_ENCODINGS)
def test_compile_input_ir_to_output_ir(
    request: pytest.FixtureRequest, encoding: str, cgc_cli,
    cgc_input_ir, cgc_source_onnx: Path, tmp_path: Path,
) -> None:
    if encoding == "caller":
        source = Path(os.environ["WINML_TEST_CGC_INPUT_IR"]).resolve(strict=True)
        assert source.is_file()
        output = tmp_path / "caller-output.mlir"
        original_hash = _sha256(source)
        result = cgc_cli(
            "compile", "-m", str(source), "--device", "gpu", "-o", str(output),
        )
    else:
        source = (
            cgc_input_ir[0] if encoding == "text" else request.getfixturevalue("cgc_bytecode_ir")
        )
        original_hash = _sha256(source)
        output, result = request.getfixturevalue(
            "cgc_output_ir" if encoding == "text" else "cgc_bytecode_output_ir"
        )
    assert "MLIR text" in result.stdout and "schema passed" in result.stdout
    assert output.read_text(encoding="utf-8").strip()
    assert not output.read_bytes().startswith(b"ML\xefR")
    receipt_path = output.with_name(f"{output.stem}_compile.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["input"]["sha256"] == original_hash
    assert receipt["validation"]["status"] == "passed"
    assert "no inference" in receipt["validation"]["scope"]
    assert receipt["compiled_form"] == "DEVICE_TARGETED"
    assert receipt["output_encoding"] == "mlir-text"
    assert receipt["artifacts"]
    for item in receipt["artifacts"]:
        path = output.parent / item["filename"]
        assert path.is_file()
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
    if encoding == "text":
        metadata = _metadata(output)
        assert metadata == {**_metadata(source), "format": "cg-output-ir"}
    assert _sha256(source) == original_hash
    args = [
        "compile", "-m", str(source), "--device", "gpu", "-o", str(output),
    ]
    retry = cgc_cli(*args, expect_success=False)
    assert retry.returncode != 0 and "--overwrite" in retry.stdout + retry.stderr
    if encoding == "text":
        invalid = tmp_path / "not-input-ir.mlir"
        invalid.write_bytes(cgc_source_onnx.read_bytes())
        invalid_output = tmp_path / "invalid-output.mlir"
        wrong = cgc_cli(
            "compile", "-m", str(invalid), "--device", "gpu",
            "-o", str(invalid_output), expect_success=False,
        )
        assert wrong.returncode != 0
        assert not invalid_output.exists()


# Command: winml perf -m <text-compiled/output.mlir> --device gpu.
# Run the persisted Output IR in a fresh process, beyond compile's schema-only validation.
def test_perf_output_ir(
    cgc_cli, cgc_output_ir, cgc_source_onnx: Path, tmp_path: Path,
) -> None:
    output = tmp_path / "output-perf.json"
    _perf(cgc_cli, cgc_output_ir[0], output)
    _assert_perf(output, cgc_output_ir[0], cgc_source_onnx, runtime="winml-runtime")
