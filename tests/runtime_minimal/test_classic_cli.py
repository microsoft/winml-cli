# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Exercise the actual classic CLI in a process that forbids ORT and Torch."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["image", "text"])
def test_classic_cli_without_export_stack(kind, tmp_path):
    root = os.environ.get("WINML_CLIP_TEST_ROOT")
    if not root:
        pytest.skip("Set WINML_CLIP_TEST_ROOT for native Arm64 hardware validation")
    import platform
    from pathlib import Path

    import numpy as np

    assert platform.machine().lower() in {"arm64", "aarch64"}
    root = Path(root)
    if kind == "image":
        arrays = {
            "pixel_values": np.fromfile(root / "inputs/pixels.bin", np.float32).reshape(
                1, 3, 224, 224
            )
        }
    else:
        arrays = {
            name: np.fromfile(root / f"inputs/{name}.bin", np.int32).reshape(1, 77)
            for name in ("input_ids", "attention_mask")
        }
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, **arrays)
    report = tmp_path / "report.json"
    args = [
        "perf",
        "-m",
        str(root / f"models/{kind}/model.onnx"),
        "--runtime",
        "winml-runtime",
        "--backend",
        "cgc",
        "--device",
        "gpu",
        "--input-data",
        str(inputs),
        "--iterations",
        "3",
        "--warmup",
        "1",
        "--format",
        "json",
        "--output",
        str(report),
    ]
    code = """
import importlib.abc
import json
import sys
from click.testing import CliRunner
class BlockExportStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "onnxruntime", "transformers", "optimum"}:
            raise AssertionError("Forbidden dependency imported: " + fullname)
sys.meta_path.insert(0, BlockExportStack())
from winml.modelkit.cli import main
result = CliRunner().invoke(main, json.loads(sys.argv[1]))
print(result.output)
if result.exception:
    raise result.exception
assert result.exit_code == 0
"""
    completed = subprocess.run(  # noqa: S603 - fixed Python executable, no shell
        [sys.executable, "-c", code, json.dumps(args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["benchmark_info"]["runtime"] == "winml-runtime"
    assert data["benchmark_info"]["backend"] == "cgc"
    assert data["benchmark_info"]["python_architecture"] == "ARM64"
    assert len(data["raw_samples_ms"]) == 3
    assert data["latency_ms"]["mean"] > 0
    assert data["output_validation"]["all_finite"]
    assert data["gpu_execution"]["coverage"] == "FULL"
    assert all(share["device_kind"] == "GPU" for share in data["gpu_execution"]["work_shares"])
    assert not data["output_validation"]["accuracy_checked"]
    assert data["process_memory"]["rss_checkpoints_bytes"]["after_inference"] > 0
    destination = os.environ.get("WINML_CLIP_TEST_OUTPUT")
    if destination:
        out = Path(destination)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{kind}-cli.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_numpy_inputs_validate_shapes_and_integer_ranges():
    np = pytest.importorskip("numpy", reason="Requires the runtime-classic extra")
    from click import UsageError

    from winml.modelkit.commands._perf_runtime_classic import _inputs

    io = {
        "input_names": ["tokens"],
        "input_types": [np.dtype("int32")],
        "input_shapes": [[1, None]],
        "input_symbolic_shapes": [[1, "seq"]],
        "value_ranges": {},
    }
    with pytest.raises(UsageError, match="shape-config"):
        _inputs(io, None, {})
    with pytest.raises(UsageError, match="no value range"):
        _inputs(io, None, {"seq": 4})
    with pytest.raises(UsageError, match="rank"):
        _inputs(io, None, {"tokens": 4})
    io["value_ranges"]["tokens"] = (0, 100)
    arrays = _inputs(io, None, {"seq": 4})
    assert arrays["tokens"].shape == (1, 4)
    assert arrays["tokens"].dtype == np.int32
    assert np.all((arrays["tokens"] >= 0) & (arrays["tokens"] < 100))


@pytest.mark.parametrize(
    "options, message",
    [
        (["--monitor"], "does not support"),
        (["--device", "cpu"], "requires --device gpu"),
        (["--iterations", "0"], "Invalid value for '--iterations'"),
    ],
)
def test_unsupported_options_fail_before_native_load(monkeypatch, options, message):
    from click.testing import CliRunner

    from winml.modelkit.cli import main
    from winml.modelkit.commands import perf

    monkeypatch.setattr(perf.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(perf.sys, "platform", "win32")
    result = CliRunner().invoke(
        main,
        [
            "perf",
            "-m",
            "missing.onnx",
            "--runtime",
            "winml-runtime",
            "--backend",
            "cgc",
            *options,
        ],
    )
    assert result.exit_code == 2, result.output
    assert message in result.output
