# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for native Hugging Face PyTorch evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from click.testing import CliRunner

from winml.modelkit.commands.eval import eval
from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.loader import NativeDevice, NativeHFModel


class TestPyTorchRuntimeCli:
    def test_help_shows_runtime_choices(self) -> None:
        result = CliRunner().invoke(eval, ["--help"])

        assert result.exit_code == 0
        assert "--runtime [winml|pytorch]" in result.output

    def test_pytorch_runtime_dispatches_pytorch(self, tmp_path) -> None:
        captured: dict[str, WinMLEvaluationConfig] = {}

        def fake_evaluate(config: WinMLEvaluationConfig) -> SimpleNamespace:
            captured["config"] = config
            return SimpleNamespace(config=config, metrics={}, to_dict=lambda: config.to_dict())

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=fake_evaluate),
            patch("winml.modelkit.commands.eval._write_and_display"),
        ):
            result = CliRunner().invoke(
                eval,
                [
                    "-m",
                    "fake/model",
                    "--task",
                    "image-classification",
                    "--dataset",
                    "fake/dataset",
                    "--runtime",
                    "pytorch",
                    "--device",
                    "cpu",
                    "-o",
                    str(tmp_path / "result.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.runtime == "pytorch"
        assert config.device == "cpu"
        assert config.model_id == "fake/model"
        assert config.model_path is None

    def test_default_path_still_exports(self) -> None:
        captured: dict[str, WinMLEvaluationConfig] = {}

        def fake_evaluate(config: WinMLEvaluationConfig) -> SimpleNamespace:
            captured["config"] = config
            return SimpleNamespace(config=config, metrics={}, to_dict=lambda: config.to_dict())

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device"),
            patch("winml.modelkit.commands.eval._write_and_display"),
        ):
            result = CliRunner().invoke(
                eval,
                [
                    "-m",
                    "fake/model",
                    "--task",
                    "image-classification",
                    "--dataset",
                    "fake/dataset",
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].runtime == "winml"

    def test_pytorch_runtime_loads_from_config_file(self, tmp_path) -> None:
        config_path = tmp_path / "eval.json"
        config_path.write_text(
            json.dumps(
                {
                    "eval": {
                        "runtime": "pytorch",
                        "model_id": "fake/model",
                        "task": "image-classification",
                        "device": "cpu",
                        "trust_remote_code": True,
                        "dataset": {"path": "fake/dataset"},
                    }
                }
            ),
            encoding="utf-8",
        )
        captured: dict[str, WinMLEvaluationConfig] = {}

        def fake_evaluate(config: WinMLEvaluationConfig) -> SimpleNamespace:
            captured["config"] = config
            return SimpleNamespace(config=config, metrics={}, to_dict=lambda: config.to_dict())

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=fake_evaluate),
            patch("winml.modelkit.commands.eval._write_and_display"),
            patch("winml.modelkit.commands.eval._run_dataset_script") as run_dataset_script,
        ):
            result = CliRunner().invoke(eval, ["--config", str(config_path)], obj={})

        assert result.exit_code == 0, result.output
        assert captured["config"].runtime == "pytorch"
        run_dataset_script.assert_called_once_with(captured["config"], True)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("ep", "cpu"),
            ("use_cache", True),
            ("skip_build", True),
            ("quant", True),
        ],
    )
    def test_native_config_file_rejects_onnx_only_fields(
        self,
        tmp_path,
        field: str,
        value: object,
    ) -> None:
        config_path = tmp_path / "eval.json"
        config_path.write_text(
            json.dumps(
                {
                    "eval": {
                        "runtime": "pytorch",
                        "model_id": "fake/model",
                        "task": "image-classification",
                        field: value,
                    }
                }
            ),
            encoding="utf-8",
        )

        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--config", str(config_path)],
            obj={},
        )

        assert result.exit_code == 2
        assert f"eval.{field}" in result.output

    @pytest.mark.parametrize(
        ("args", "expected_flag"),
        [
            (["--ep", "cpu"], "--ep"),
            (["--precision", "fp16"], "--precision"),
            (["--no-quant"], "--quant/--no-quant"),
            (["--no-optimize"], "--optimize/--no-optimize"),
            (["--no-analyze"], "--analyze/--no-analyze"),
            (["--max-optim-iterations", "2"], "--max-optim-iterations"),
            (["--allow-unsupported-nodes"], "--allow-unsupported-nodes"),
            (["--no-skip-build"], "--skip-build/--no-skip-build"),
            (["--no-use-cache"], "--use-cache/--no-use-cache"),
            (["--rebuild"], "--rebuild/--no-rebuild"),
            (["--mode", "compare"], "--mode"),
        ],
    )
    def test_rejects_onnx_only_options(
        self,
        args: list[str],
        expected_flag: str,
    ) -> None:
        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--runtime", "pytorch", *args],
            obj={},
        )

        assert result.exit_code == 2
        assert expected_flag in result.output

    def test_rejects_export_override(self, tmp_path) -> None:
        shape_config = tmp_path / "shape.json"
        shape_config.write_text(json.dumps({"height": 16}))

        result = CliRunner().invoke(
            eval,
            [
                "-m",
                "fake/model",
                "--runtime",
                "pytorch",
                "--shape-config",
                str(shape_config),
            ],
            obj={},
        )

        assert result.exit_code == 2
        assert "--shape-config" in result.output

    def test_rejects_onnx_input(self, tmp_path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"not used")

        result = CliRunner().invoke(
            eval,
            [
                "-m",
                str(model_path),
                "--model-id",
                "fake/model",
                "--runtime",
                "pytorch",
            ],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a Hugging Face model ID" in result.output

    def test_rejects_genai_bundle(self, tmp_path) -> None:
        (tmp_path / "genai_config.json").write_text("{}")

        result = CliRunner().invoke(
            eval,
            ["-m", str(tmp_path), "--runtime", "pytorch"],
            obj={},
        )

        assert result.exit_code == 2
        assert "GenAI bundles are not supported" in result.output

    def test_rejects_npu_device(self) -> None:
        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--runtime", "pytorch", "--device", "npu"],
            obj={},
        )

        assert result.exit_code == 2
        assert "use auto, cpu, or gpu" in result.output

    @pytest.mark.parametrize("task", ["text-generation", "mask-generation"])
    def test_evaluator_tasks_are_not_rejected_centrally(self, task: str) -> None:
        captured: dict[str, WinMLEvaluationConfig] = {}

        def fake_evaluate(config: WinMLEvaluationConfig) -> SimpleNamespace:
            captured["config"] = config
            return SimpleNamespace(config=config, metrics={}, to_dict=lambda: config.to_dict())

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=fake_evaluate),
            patch("winml.modelkit.commands.eval._write_and_display"),
        ):
            result = CliRunner().invoke(
                eval,
                ["-m", "fake/model", "--runtime", "pytorch", "--task", task],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].task == task

    def test_gpu_requires_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--runtime", "pytorch", "--device", "gpu"],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a CUDA-enabled PyTorch" in result.output


class TestNativeEvaluation:
    def test_pytorch_runtime_uses_pytorch_loader_kind(self) -> None:
        from winml.modelkit.eval.evaluate import _ModelLoaderKind, _select_model_loader

        config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            runtime="pytorch",
        )

        assert _select_model_loader(config) is _ModelLoaderKind.PYTORCH

    def test_public_evaluate_rejects_invalid_runtime(self) -> None:
        from typing import cast

        from winml.modelkit.eval import EvalRuntime, evaluate

        config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            runtime=cast("EvalRuntime", "invalid"),
        )

        with pytest.raises(ValueError, match="Invalid runtime"):
            evaluate(config)

    def test_public_evaluate_passes_supplied_model_to_evaluator(self) -> None:
        from winml.modelkit.eval import evaluate

        model = MagicMock()
        model.config = SimpleNamespace(_name_or_path="inferred/model")
        model.device = torch.device("cpu")
        captured: dict[str, object] = {}

        class FakeEvaluator:
            def __init__(self, config, evaluator_model) -> None:
                captured["config"] = config
                captured["model"] = evaluator_model

            def compute(self) -> dict[str, float]:
                return {"accuracy": 1.0}

        config = WinMLEvaluationConfig(
            task="image-classification",
            dataset=DatasetConfig(path="fake/dataset"),
        )

        with (
            patch(
                "winml.modelkit.eval.evaluate.get_evaluator_class",
                return_value=FakeEvaluator,
            ),
            patch(
                "winml.modelkit.eval.evaluate.load_model",
                return_value=model,
            ) as load_model,
        ):
            result = evaluate(config, pytorch_model=model)

        load_model.assert_called_once_with(result.config, model)
        assert captured["model"] is model
        assert captured["config"] is result.config
        assert result.config.runtime == "pytorch"
        assert result.config.model_id == "inferred/model"
        assert result.config.device == "cpu"
        assert config.runtime == "winml"
        assert config.model_id is None

    def test_public_evaluate_sets_real_supplied_module_to_eval_mode(self) -> None:
        from torch import nn

        from winml.modelkit.eval import evaluate

        model = nn.Linear(2, 2)
        model.config = SimpleNamespace(_name_or_path="inferred/model")  # type: ignore[attr-defined]
        model.train()

        class FakeEvaluator:
            def __init__(self, config, evaluator_model) -> None:
                assert evaluator_model is model
                assert not evaluator_model.training

            def compute(self) -> dict[str, float]:
                return {"accuracy": 1.0}

        config = WinMLEvaluationConfig(
            task="image-classification",
            dataset=DatasetConfig(path="fake/dataset"),
        )

        with patch(
            "winml.modelkit.eval.evaluate.get_evaluator_class",
            return_value=FakeEvaluator,
        ):
            evaluate(config, pytorch_model=model)

        assert not model.training

    def test_supplied_model_id_overrides_model_config(self) -> None:
        from winml.modelkit.eval.evaluate import _prepare_supplied_pytorch_model

        model = MagicMock()
        model.config = SimpleNamespace(_name_or_path="inferred/model")
        model.device = torch.device("cpu")
        config = WinMLEvaluationConfig(model_id="explicit/model")

        resolved = _prepare_supplied_pytorch_model(config, model)

        assert resolved.model_id == "explicit/model"

    def test_supplied_model_requires_processor_source(self) -> None:
        from winml.modelkit.eval import evaluate

        model = MagicMock()
        model.config = SimpleNamespace(_name_or_path="")
        model.device = torch.device("cpu")
        config = WinMLEvaluationConfig(
            task="image-classification",
            dataset=DatasetConfig(path="fake/dataset"),
        )

        with pytest.raises(ValueError, match=r"model\.config\._name_or_path"):
            evaluate(config, pytorch_model=model)

    def test_supplied_model_device_must_match_explicit_config(self) -> None:
        from winml.modelkit.eval.evaluate import _prepare_supplied_pytorch_model

        model = MagicMock()
        model.config = SimpleNamespace(_name_or_path="fake/model")
        model.device = torch.device("cuda")
        config = WinMLEvaluationConfig(device="cpu")

        with pytest.raises(ValueError, match="model is on gpu"):
            _prepare_supplied_pytorch_model(config, model)

    def test_supplied_model_preserves_cuda_ordinal_for_pipeline(self) -> None:
        from winml.modelkit.eval.evaluate import _prepare_supplied_pytorch_model

        model = MagicMock()
        model.config = SimpleNamespace(_name_or_path="fake/model")
        model.device = torch.device("cuda:1")

        resolved = _prepare_supplied_pytorch_model(WinMLEvaluationConfig(), model)

        assert resolved.device == "gpu"
        assert resolved.pipeline_device == "cuda:1"

    def test_public_evaluate_rejects_onnx_state(self) -> None:
        from winml.modelkit.eval import evaluate

        config = WinMLEvaluationConfig(
            model_id="fake/model",
            model_path="model.onnx",
            task="image-classification",
            runtime="pytorch",
        )

        with pytest.raises(ValueError, match="model_path"):
            evaluate(config)

    @pytest.mark.parametrize(
        ("config_override", "expected_field"),
        [
            ({"use_cache": False}, "use_cache"),
            ({"rebuild": True}, "rebuild"),
        ],
    )
    def test_public_evaluate_rejects_cache_state(
        self,
        config_override: dict[str, bool],
        expected_field: str,
    ) -> None:
        from winml.modelkit.eval import evaluate

        config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            runtime="pytorch",
            **config_override,
        )

        with pytest.raises(ValueError, match=expected_field):
            evaluate(config)

    def test_load_model_uses_shared_native_loader(self) -> None:
        from winml.modelkit.eval.evaluate import load_model

        model = MagicMock()
        loaded = NativeHFModel(
            model=model,
            device=NativeDevice(name="gpu", torch_device=torch.device("cuda")),
        )
        config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            device="gpu",
            runtime="pytorch",
            trust_remote_code=True,
        )

        with patch(
            "winml.modelkit.loader.load_native_hf_model",
            return_value=loaded,
        ) as load:
            assert load_model(config) is model

        load.assert_called_once_with(
            "fake/model",
            task="image-classification",
            device="gpu",
            trust_remote_code=True,
            torch_dtype="auto",
        )
        assert config.device == "gpu"

    def test_representative_evaluator_uses_native_pipeline_device(self) -> None:
        from winml.modelkit.eval.base_evaluator import WinMLEvaluator

        evaluator = WinMLEvaluator.__new__(WinMLEvaluator)
        evaluator.config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            device="gpu",
            runtime="pytorch",
            dataset=DatasetConfig(path="fake/dataset"),
        )
        evaluator.model = MagicMock()
        pipeline = MagicMock()

        with patch(
            "winml.modelkit.inference.pipeline.create_pipeline",
            return_value=pipeline,
        ) as create:
            assert evaluator.prepare_pipeline() is pipeline

        create.assert_called_once_with(
            "image-classification",
            evaluator.model,
            "fake/model",
            device="cuda",
            trust_remote_code=False,
        )

    def test_config_roundtrip_identifies_pytorch_runtime(self) -> None:
        config = WinMLEvaluationConfig(
            model_id="fake/model",
            device="cpu",
            runtime="pytorch",
        )

        serialized = config.to_dict()
        restored = WinMLEvaluationConfig.from_dict(serialized)

        assert serialized["runtime"] == "pytorch"
        assert "skip_build" not in serialized
        assert "use_cache" not in serialized
        assert "rebuild" not in serialized
        assert restored.runtime == "pytorch"
