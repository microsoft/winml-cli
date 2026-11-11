# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Final composed-config validation tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_config.py"


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("auto_optimize_validate_config", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"optim": {"required_fusion": True}}), encoding="utf-8")
    source = tmp_path / "source.onnx"
    champion = tmp_path / "champion.onnx"
    _model(source)
    shutil.copy2(source, champion)
    input_data = tmp_path / "inputs.npz"
    np.savez(input_data, x=np.ones((1, 1, 4), dtype=np.float32))
    return config, source, champion, input_data


def _runner(champion: Path, cosine: float):
    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command[1] == "build":
            output = Path(command[command.index("-o") + 1])
            output.mkdir(parents=True)
            shutil.copy2(champion, output / "model.onnx")
        elif command[1] == "eval":
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps({"metrics": {"cosine_similarity_mean": {"y": cosine}}}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    return run


def test_clean_rebuild_and_runtime_comparison_emit_bound_receipt(validator, tmp_path):
    config, source, champion, input_data = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"

    result = validator.validate_config(
        config_path=config,
        source=str(source),
        champion_path=champion,
        input_data=input_data,
        work_dir=tmp_path / "validation",
        receipt_path=receipt,
        device="npu",
        ep="qnn",
        min_cosine=0.999,
        runner=_runner(champion, 0.9999),
    )

    assert result["status"] == "pass"
    assert result["clean_directory"] is True
    assert result["build"]["public_io"] == "pass"
    assert result["runtime_load"]["status"] == "pass"
    assert result["correctness"]["observed"] == pytest.approx(0.9999)
    assert not (tmp_path / "validation" / "build").exists()
    validator.verify_receipt(receipt, config, champion, [input_data])


def test_final_composition_with_low_correctness_is_rejected(validator, tmp_path):
    config, source, champion, input_data = _inputs(tmp_path)

    with pytest.raises(validator.ConfigValidationError, match="below required"):
        validator.validate_config(
            config_path=config,
            source=str(source),
            champion_path=champion,
            input_data=input_data,
            work_dir=tmp_path / "validation",
            receipt_path=tmp_path / "receipt.json",
            device="npu",
            ep="qnn",
            min_cosine=0.999,
            runner=_runner(champion, 0.91),
        )

    assert not (tmp_path / "receipt.json").exists()


def test_receipt_is_invalidated_when_final_config_changes(validator, tmp_path):
    config, source, champion, input_data = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    validator.validate_config(
        config_path=config,
        source=str(source),
        champion_path=champion,
        input_data=input_data,
        work_dir=tmp_path / "validation",
        receipt_path=receipt,
        device="npu",
        ep="qnn",
        min_cosine=0.999,
        runner=_runner(champion, 1.0),
    )
    config.write_text(json.dumps({"optim": {}}), encoding="utf-8")

    with pytest.raises(validator.ConfigValidationError, match="config changed"):
        validator.verify_receipt(receipt, config, champion, [input_data])


def test_receipt_rejects_a_different_prepared_source_than_repro_lock(validator, tmp_path):
    config, source, champion, input_data = _inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    validator.validate_config(
        config_path=config,
        source=str(source),
        champion_path=champion,
        input_data=input_data,
        work_dir=tmp_path / "validation",
        receipt_path=receipt,
        device="npu",
        ep="qnn",
        min_cosine=0.999,
        runner=_runner(champion, 1.0),
    )
    lock = tmp_path / "repro.lock.json"
    lock.write_text(
        json.dumps({"source": {"prepared_model_sha256": "0" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(validator.ConfigValidationError, match="prepared source"):
        validator.verify_receipt(receipt, config, champion, [input_data], lock)