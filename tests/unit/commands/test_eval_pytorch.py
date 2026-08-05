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


class TestNoExportCli:
    def test_help_shows_export_pair(self) -> None:
        result = CliRunner().invoke(eval, ["--help"])

        assert result.exit_code == 0
        assert "--export" in result.output
        assert "--no-export" in result.output

    def test_no_export_dispatches_pytorch_backend(self, tmp_path) -> None:
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
                    "--no-export",
                    "--device",
                    "cpu",
                    "-o",
                    str(tmp_path / "result.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert config.backend == "pytorch"
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
        assert captured["config"].backend == "onnx"
        assert captured["config"].export_model is True

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
            ["-m", "fake/model", "--no-export", *args],
            obj={},
        )

        assert result.exit_code == 2
        assert expected_flag in result.output

    def test_rejects_export_override(self, tmp_path) -> None:
        shape_config = tmp_path / "shape.json"
        shape_config.write_text(json.dumps({"height": 16}))

        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--no-export", "--shape-config", str(shape_config)],
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
                "--no-export",
            ],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a Hugging Face model ID" in result.output

    def test_rejects_genai_bundle(self, tmp_path) -> None:
        (tmp_path / "genai_config.json").write_text("{}")

        result = CliRunner().invoke(
            eval,
            ["-m", str(tmp_path), "--no-export"],
            obj={},
        )

        assert result.exit_code == 2
        assert "GenAI bundles are not supported" in result.output

    def test_rejects_npu_device(self) -> None:
        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--no-export", "--device", "npu"],
            obj={},
        )

        assert result.exit_code == 2
        assert "use auto, cpu, or gpu" in result.output

    @pytest.mark.parametrize(
        "task",
        [
            "fill-mask",
            "keypoint-detection",
            "mask-generation",
            "text-generation",
        ],
    )
    def test_rejects_non_pipeline_evaluators(self, task: str) -> None:
        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--no-export", "--task", task],
            obj={},
        )

        assert result.exit_code == 2
        assert "does not use the standard labeled Hugging Face pipeline" in result.output

    def test_gpu_requires_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = CliRunner().invoke(
            eval,
            ["-m", "fake/model", "--no-export", "--device", "gpu"],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a CUDA-enabled PyTorch" in result.output


class TestNativeEvaluation:
    def test_public_evaluate_rejects_onnx_state(self) -> None:
        from winml.modelkit.eval import evaluate

        config = WinMLEvaluationConfig(
            model_id="fake/model",
            model_path="model.onnx",
            task="image-classification",
            export_model=False,
        )

        with pytest.raises(ValueError, match="model_path"):
            evaluate(config)

    def test_load_model_uses_shared_native_loader(self) -> None:
        from winml.modelkit.eval.evaluate import _load_model

        model = MagicMock()
        loaded = NativeHFModel(
            model=model,
            config=MagicMock(),
            task="image-classification",
            device=NativeDevice(name="gpu", torch_device=torch.device("cuda")),
        )
        config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            device="gpu",
            export_model=False,
            trust_remote_code=True,
        )

        with patch(
            "winml.modelkit.loader.load_native_hf_model",
            return_value=loaded,
        ) as load:
            assert _load_model(config) is model

        load.assert_called_once_with(
            "fake/model",
            task="image-classification",
            device="gpu",
            trust_remote_code=True,
        )
        assert config.device == "gpu"

    def test_representative_evaluator_uses_native_pipeline_device(self) -> None:
        from winml.modelkit.eval.base_evaluator import WinMLEvaluator

        evaluator = WinMLEvaluator.__new__(WinMLEvaluator)
        evaluator.config = WinMLEvaluationConfig(
            model_id="fake/model",
            task="image-classification",
            device="gpu",
            export_model=False,
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
        )

    def test_config_roundtrip_identifies_pytorch_backend(self) -> None:
        config = WinMLEvaluationConfig(
            model_id="fake/model",
            device="cpu",
            export_model=False,
        )

        serialized = config.to_dict()
        restored = WinMLEvaluationConfig.from_dict(serialized)

        assert serialized["backend"] == "pytorch"
        assert restored.backend == "pytorch"
        assert restored.export_model is False
