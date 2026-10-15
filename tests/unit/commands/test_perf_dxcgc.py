"""Native DXCGC boundary validation, without GPU dependencies."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from winml.modelkit.commands._perf_dxcgc import _io_policy, _samples
from winml.modelkit.commands.perf import perf


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_reject_invalid_samples(tmp_path, value):
    path = tmp_path / "samples.csv"
    path.write_text(f"gpu_ms,host_ms\n{value},1\n")
    with pytest.raises(ValueError, match="invalid timings"):
        _samples(path, 1)


def test_reject_incomplete_samples(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("gpu_ms,host_ms\n1,2\n")
    with pytest.raises(ValueError, match="Expected 2"):
        _samples(path, 2)


@pytest.mark.parametrize("policy", [None, "resident", "per_call_cpu_ownership_and_binding"])
def test_native_io_policy_acknowledgement(tmp_path, policy):
    if policy is not None:
        (tmp_path / "io-policy.txt").write_text(policy)
    if policy == "per_call_cpu_ownership_and_binding":
        assert _io_policy(tmp_path) == policy
    else:
        with pytest.raises(ValueError, match="rebuild the native host"):
            _io_policy(tmp_path)


def test_requires_mlir(tmp_path):
    path = tmp_path / "model.onnx"
    path.touch()
    result = CliRunner().invoke(perf, ["-m", str(path), "--runtime", "dxcgc"])
    assert result.exit_code != 0
    assert "local .mlir" in result.output


def test_reject_runtime_options():
    result = CliRunner().invoke(perf, ["-m", "x.mlir", "--runtime", "dxcgc", "--monitor"])
    assert result.exit_code == 2
    assert "--monitor" in result.output


def test_options_require_dxcgc():
    result = CliRunner().invoke(perf, ["-m", "x.mlir", "--dxcgc-adapter", "0"])
    assert result.exit_code == 2
    assert "require --runtime dxcgc" in result.output


def test_disable_target_requires_dxcgc():
    result = CliRunner().invoke(perf, ["-m", "x.mlir", "--disable-d3d-mlir-programs"])
    assert result.exit_code == 2
    assert "--disable-d3d-mlir-programs requires --runtime dxcgc" in result.output


@pytest.mark.parametrize("disabled", [False, True])
def test_disable_target_forwarded(tmp_path, monkeypatch, disabled):
    from types import SimpleNamespace

    import winml.modelkit.commands._perf_dxcgc as native

    model = tmp_path / "input.mlir"
    model.touch()
    host = tmp_path / "host.exe"
    host.touch()
    (tmp_path / "dxcgc.dll").touch()
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, x=np.ones((1,), dtype=np.float32))
    captured = []

    def run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=1, stderr="test host error")

    monkeypatch.setattr(native.subprocess, "run", run)
    args = [
        "-m",
        str(model),
        "--runtime",
        "dxcgc",
        "--dxcgc-host",
        str(host),
        "--dxcgc-dir",
        str(tmp_path),
        "--dxcgc-sdk-path",
        str(tmp_path),
        "--input-data",
        str(inputs),
        "-o",
        str(tmp_path / "perf.json"),
    ]
    if disabled:
        args.append("--disable-d3d-mlir-programs")
    result = CliRunner().invoke(perf, args)
    assert "test host error" in result.output
    assert ("--disable-d3d-mlir-programs" in captured[0]) is disabled


@pytest.mark.skipif(not os.environ.get("WINML_DXCGC_MODEL"), reason="Requires DXCGC GPU fixtures")
@pytest.mark.parametrize("disabled", [False, True])
def test_resnet_native(tmp_path, disabled):
    args = [
        "-m",
        os.environ["WINML_DXCGC_MODEL"],
        "--runtime",
        "dxcgc",
        "--dxcgc-host",
        os.environ["WINML_DXCGC_HOST"],
        "--input-data",
        os.environ["WINML_DXCGC_INPUT"],
        "--iterations",
        "100",
        "--warmup",
        "10",
        "-o",
        str(tmp_path / "perf.json"),
    ]
    if disabled:
        args.append("--disable-d3d-mlir-programs")
    result = CliRunner().invoke(perf, args)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "perf.json").read_text())
    assert len(report["samples"]["gpu_ms"]) == 100
    assert report["io_mode"] == "cpu-roundtrip"
    assert report["io_binding_policy"] == "per_call_cpu_ownership_and_binding"
    assert report["call_driver"] == "python_ctypes_per_call"
    assert report["disable_d3d_mlir_programs"] is disabled
    assert any("windowsml" in path and path.endswith("dxcgc.dll") for path in report["provenance"])
    with (
        np.load(tmp_path / "perf.outputs.npz") as actual,
        np.load(Path(os.environ["WINML_DXCGC_REFERENCE"])) as expected,
    ):
        assert len(actual.files) == len(expected.files)
        for a, b in zip(actual.values(), expected.values(), strict=True):
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            relative = np.linalg.norm(a.astype(float) - b) / np.linalg.norm(b)
            assert relative < 0.01
