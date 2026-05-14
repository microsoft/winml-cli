# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for eval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.datasets import DatasetConfig
from winml.modelkit.eval import EvalResult, WinMLEvaluationConfig


class TestEvaluationConfig:
    """Tests for config and result dataclasses."""

    def test_config_roundtrip(self):
        config = WinMLEvaluationConfig(
            model_id="test/model",
            model_path="model.onnx",
            task="image-classification",
            device="npu",
            dataset=DatasetConfig(
                path="imagenet-1k",
                split="test",
                samples=20,
                columns_mapping={"label_column": "lbl"},
            ),
        )
        restored = WinMLEvaluationConfig.from_dict(config.to_dict())
        assert restored.model_id == config.model_id
        assert restored.dataset.path == config.dataset.path
        assert restored.dataset.columns_mapping == config.dataset.columns_mapping

    def test_eval_result_to_dict(self):
        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="imagenet-1k"),
        )
        result = EvalResult(config=config, metrics={"accuracy": 0.9})
        d = result.to_dict()
        assert d["metrics"]["accuracy"] == 0.9
        assert d["dataset"]["path"] == "imagenet-1k"


class TestResolveTask:
    """Tests for _resolve_task."""

    def test_explicit_task(self):
        from winml.modelkit.eval.evaluate import _resolve_task

        config = WinMLEvaluationConfig(task="image-classification")
        assert _resolve_task(config) == "image-classification"

    def test_no_model_id_raises(self):
        from winml.modelkit.eval.evaluate import _resolve_task

        with pytest.raises(ValueError, match="Cannot infer task"):
            _resolve_task(WinMLEvaluationConfig())

    def test_infer_from_model_id(self):
        from winml.modelkit.eval.evaluate import _resolve_task

        fake_hf_config = MagicMock()
        config = WinMLEvaluationConfig(model_id="microsoft/resnet-50")
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=fake_hf_config,
            ),
            patch(
                "winml.modelkit.loader.task._detect_task_from_config",
                return_value="image-classification",
            ),
        ):
            assert _resolve_task(config) == "image-classification"


class TestEvaluate:
    """Tests for evaluate() entry point."""

    def test_no_dataset_no_default_raises(self):
        """Tasks without a default dataset raise ValueError."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        task_without_default = next(
            t
            for t in ["fill-mask", "summarization", "translation"]
            if t not in eval_mod._DEFAULT_DATASETS
        )

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task=task_without_default,
        )

        with (
            patch.object(eval_mod, "_load_model", return_value=MagicMock()),
            pytest.raises(ValueError, match="No dataset provided"),
        ):
            eval_mod.evaluate(config)

    def test_evaluate_does_not_mutate_caller_config(self):
        """evaluate() must not modify the caller's config object."""
        import importlib
        import sys
        from dataclasses import asdict

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task=None,
            dataset=DatasetConfig(path=None),
        )
        original = asdict(config)

        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = {"accuracy": 0.8}

        with (
            patch.object(eval_mod, "_resolve_task", return_value="text-classification"),
            patch.object(eval_mod, "_load_model", return_value=MagicMock()),
            patch.object(
                eval_mod,
                "_EVALUATOR_REGISTRY",
                {"text-classification": lambda *a: mock_evaluator},
            ),
        ):
            eval_mod.evaluate(config)

        assert asdict(config) == original, "evaluate() mutated the caller's config"


class TestWinMLEvaluator:
    """Tests for WinMLEvaluator base class."""

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_samples_capped_when_exceeds_dataset_size(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """When requested samples exceed dataset size, select uses actual size."""
        from winml.modelkit.eval import WinMLEvaluator

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 50
        mock_ds.shuffle.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="test-dataset", samples=100),
        )

        ev = WinMLEvaluator(config, model)
        # dataset.select should use actual dataset size (50), not requested (100)
        mock_ds.select.assert_called_once_with(range(50))
        # config.dataset.samples should NOT be mutated
        assert ev.config.dataset.samples == 100

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_compute_calls_hf_evaluator(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        from winml.modelkit.eval import WinMLEvaluator

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {"accuracy": 0.9}
        # Give the mock compute() a signature that includes label_mapping
        # so our inspect-based check finds and passes it
        import inspect

        def _fake_compute(
            *,
            model_or_pipeline=None,
            data=None,
            label_mapping=None,
            **kw,
        ):
            return {"accuracy": 0.9}

        mock_eval_inst.compute = MagicMock(
            side_effect=_fake_compute,
            __signature__=inspect.signature(_fake_compute),
        )
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = {"cat": 0, "dog": 1}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="test-dataset", samples=10),
        )

        ev = WinMLEvaluator(config, model)
        metrics = ev.compute()

        mock_hf_eval.assert_called_once_with("image-classification")
        call_kwargs = mock_eval_inst.compute.call_args[1]
        assert call_kwargs["label_mapping"] == {"cat": 0, "dog": 1}
        assert metrics["accuracy"] == 0.9

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_columns_mapping_passed(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        from winml.modelkit.eval import WinMLEvaluator

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()

        mock_eval_inst = MagicMock()
        # Give the mock compute() a **kwargs signature so inspect
        # doesn't strip our column overrides
        import inspect

        def _fake_compute(**kw):
            return {"accuracy": 0.5}

        mock_eval_inst.compute = MagicMock(
            side_effect=_fake_compute,
            __signature__=inspect.signature(_fake_compute),
        )
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="text-classification",
            dataset=DatasetConfig(
                path="glue",
                name="mrpc",
                columns_mapping={"input_column": "sentence1", "second_input_column": "sentence2"},
            ),
        )

        WinMLEvaluator(config, model).compute()

        call = mock_eval_inst.compute.call_args
        assert call[1]["input_column"] == "sentence1"
        assert call[1]["second_input_column"] == "sentence2"


class TestSequenceClassificationEvaluator:
    """Tests for text classification evaluator padding."""

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_sets_padding_for_text_model(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        from winml.modelkit.eval import (
            WinMLTextClassificationEvaluator,
        )

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {"accuracy": 0.9}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = {}
        model.io_config = {"input_shapes": [[1, 512], [1, 512], [1, 512]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="text-classification",
            dataset=DatasetConfig(path="glue", name="mrpc"),
        )

        WinMLTextClassificationEvaluator(config, model).compute()

        assert mock_pipe._preprocess_params["padding"] == "max_length"
        assert mock_pipe._preprocess_params["max_length"] == 512
        assert mock_pipe._preprocess_params["truncation"] is True

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_no_padding_without_tokenizer(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        from winml.modelkit.eval import (
            WinMLTextClassificationEvaluator,
        )

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = None
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 512]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="text-classification",
            dataset=DatasetConfig(path="glue"),
        )

        WinMLTextClassificationEvaluator(config, model).compute()

        assert "padding" not in mock_pipe._preprocess_params


class TestTokenClassificationEvaluator:
    """Tests for token classification evaluator padding."""

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_sets_tokenizer_params_nesting(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """Padding is set via tokenizer_params dict, not top-level."""
        from winml.modelkit.eval import (
            WinMLTokenClassificationEvaluator,
        )

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {"f1": 0.85}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = {"O": 0, "B-PER": 1}
        model.io_config = {"input_shapes": [[1, 128], [1, 128], [1, 128]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="token-classification",
            dataset=DatasetConfig(path="conll2003"),
        )

        WinMLTokenClassificationEvaluator(config, model).compute()

        tok_params = mock_pipe._preprocess_params["tokenizer_params"]
        assert tok_params["padding"] == "max_length"
        assert tok_params["max_length"] == 128
        assert mock_pipe._preprocess_params["truncation"] is True
        assert mock_pipe.tokenizer.model_max_length == 128

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_no_padding_without_tokenizer(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """No tokenizer → no padding config."""
        from winml.modelkit.eval import (
            WinMLTokenClassificationEvaluator,
        )

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = None
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 128]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="token-classification",
            dataset=DatasetConfig(path="conll2003"),
        )

        WinMLTokenClassificationEvaluator(config, model).compute()

        assert "tokenizer_params" not in mock_pipe._preprocess_params

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_no_padding_without_input_shapes(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """Missing input_shapes in io_config → no padding config."""
        from winml.modelkit.eval import (
            WinMLTokenClassificationEvaluator,
        )

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1000
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {}
        mock_hf_eval.return_value = mock_eval_inst

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="token-classification",
            dataset=DatasetConfig(path="conll2003"),
        )

        WinMLTokenClassificationEvaluator(config, model).compute()

        assert "tokenizer_params" not in mock_pipe._preprocess_params


class TestEvalCli:
    """Tests for CLI option mapping."""

    def test_cli_maps_options_to_config(self):
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("npu", ["npu", "cpu"])),
            patch("winml.modelkit.eval.evaluate") as mock_evaluate,
        ):
            mock_evaluate.return_value = EvalResult(
                config=WinMLEvaluationConfig(),
                metrics={},
            )
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    "test/model",
                    "--dataset",
                    "imagenet-1k",
                    "--task",
                    "image-classification",
                    "--samples",
                    "10",
                    "--split",
                    "test",
                    "--device",
                    "npu",
                    "--column",
                    "input_column=img",
                    "--column",
                    "label_column=lbl",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, result.output
            config = mock_evaluate.call_args[0][0]
            assert config.model_id == "test/model"
            assert config.dataset.path == "imagenet-1k"
            assert config.dataset.columns_mapping == {
                "input_column": "img",
                "label_column": "lbl",
            }

    def test_cli_onnx_model_path(self, tmp_path):
        from winml.modelkit.commands.eval import eval as eval_cmd

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("cpu", ["cpu"])),
            patch("winml.modelkit.eval.evaluate") as mock_evaluate,
        ):
            mock_evaluate.return_value = EvalResult(
                config=WinMLEvaluationConfig(),
                metrics={},
            )
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    str(onnx_file),
                    "--model-id",
                    "test/model",
                    "--dataset",
                    "imagenet-1k",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, result.output
            config = mock_evaluate.call_args[0][0]
            assert config.model_path == str(onnx_file)
            assert config.model_id == "test/model"

    def test_cli_missing_onnx_file_raises(self, tmp_path):
        """Passing a non-existent .onnx path must error, not silently fall back."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        missing = tmp_path / "nonexistent.onnx"

        runner = CliRunner()
        result = runner.invoke(
            eval_cmd,
            [
                "-m",
                str(missing),
                "--model-id",
                "test/model",
                "--dataset",
                "imagenet-1k",
            ],
        )

        assert result.exit_code != 0
        assert "ONNX file not found" in result.output

    def test_cli_no_model_raises(self):
        """Running without -m or --model-id must error early."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        result = runner.invoke(eval_cmd, ["--dataset", "imagenet-1k"])

        assert result.exit_code != 0
        assert "model is required" in result.output.lower()

    def test_cli_onnx_without_model_id_raises(self, tmp_path):
        """Using an ONNX file without --model-id must error early."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        runner = CliRunner()
        result = runner.invoke(
            eval_cmd,
            [
                "-m",
                str(onnx_file),
                "--dataset",
                "imagenet-1k",
            ],
        )

        assert result.exit_code != 0
        assert "--model-id is required" in result.output.lower()

    def test_cli_bad_column_format_raises(self):
        """--column without '=' must error."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        result = runner.invoke(
            eval_cmd,
            [
                "-m",
                "test/model",
                "--column",
                "bad_format",
            ],
        )

        assert result.exit_code != 0
        assert "key=value" in result.output.lower()

    def test_cli_evaluate_exception_shown_to_user(self):
        """Exceptions from evaluate() must surface to the user."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("cpu", ["cpu"])),
            patch("winml.modelkit.eval.evaluate", side_effect=RuntimeError("broken model")),
        ):
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    "test/model",
                    "--dataset",
                    "imagenet-1k",
                ],
            )

        assert result.exit_code != 0
        assert "broken model" in result.output

    def test_cli_ep_passed_through(self):
        """`--ep <name>` must propagate to WinMLEvaluationConfig.ep."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("npu", ["npu", "cpu"])),
            patch("winml.modelkit.eval.evaluate") as mock_evaluate,
        ):
            mock_evaluate.return_value = EvalResult(
                config=WinMLEvaluationConfig(),
                metrics={},
            )
            result = runner.invoke(
                eval_cmd,
                ["-m", "test/model", "--dataset", "imagenet-1k", "--ep", "qnn"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, result.output
            config = mock_evaluate.call_args[0][0]
            assert config.ep == "qnn"

    def test_cli_ep_invalid_value_rejected(self):
        """Unknown --ep value must be rejected by Click Choice validation."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        runner = CliRunner()
        result = runner.invoke(
            eval_cmd,
            ["-m", "test/model", "--dataset", "imagenet-1k", "--ep", "bogus_ep"],
        )
        assert result.exit_code != 0
        assert "bogus_ep" in result.output.lower() or "invalid" in result.output.lower()

    def test_cli_ep_from_build_config(self, tmp_path):
        """When --ep is omitted, ep is read from build_cfg.compile.ep_config.provider."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        config_file = tmp_path / "build.yaml"
        config_file.touch()

        fake_build_cfg = MagicMock()
        fake_build_cfg.loader = None
        fake_build_cfg.compile.ep_config.provider = "dml"
        fake_build_cfg.quant = None

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("gpu", ["gpu", "cpu"])),
            patch(
                "winml.modelkit.utils.cli.load_build_config",
                return_value=fake_build_cfg,
            ),
            patch("winml.modelkit.eval.evaluate") as mock_evaluate,
        ):
            mock_evaluate.return_value = EvalResult(
                config=WinMLEvaluationConfig(),
                metrics={},
            )
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    "test/model",
                    "--dataset",
                    "imagenet-1k",
                    "--config",
                    str(config_file),
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, result.output
            config = mock_evaluate.call_args[0][0]
            assert config.ep == "dml"

    def test_cli_ep_overrides_build_config(self, tmp_path):
        """Explicit --ep on the CLI must take precedence over build config value."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        config_file = tmp_path / "build.yaml"
        config_file.touch()

        fake_build_cfg = MagicMock()
        fake_build_cfg.loader = None
        fake_build_cfg.compile.ep_config.provider = "dml"
        fake_build_cfg.quant = None

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("npu", ["npu", "cpu"])),
            patch(
                "winml.modelkit.utils.cli.load_build_config",
                return_value=fake_build_cfg,
            ),
            patch("winml.modelkit.eval.evaluate") as mock_evaluate,
        ):
            mock_evaluate.return_value = EvalResult(
                config=WinMLEvaluationConfig(),
                metrics={},
            )
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    "test/model",
                    "--dataset",
                    "imagenet-1k",
                    "--config",
                    str(config_file),
                    "--ep",
                    "qnn",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0, result.output
            config = mock_evaluate.call_args[0][0]
            assert config.ep == "qnn"


class TestBuildEvalResultEpField:
    """Tests for build_eval_result handling of the optional `ep` field."""

    @staticmethod
    def _load_reporter():
        """Load scripts/e2e_eval/utils/reporter.py via importlib (not on sys.path)."""
        import importlib.util
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        utils_dir = repo_root / "scripts" / "e2e_eval" / "utils"

        # Pre-load the sibling module reporter.py imports relatively.
        if "_e2e_classifier" not in sys.modules:
            spec_c = importlib.util.spec_from_file_location(
                "_e2e_classifier", utils_dir / "classifier.py"
            )
            mod_c = importlib.util.module_from_spec(spec_c)
            sys.modules["_e2e_classifier"] = mod_c
            spec_c.loader.exec_module(mod_c)

        # Stub the relative import target so reporter.py's `from .classifier ...` works.
        pkg_name = "_e2e_reporter_pkg"
        if pkg_name not in sys.modules:
            pkg = type(sys)(pkg_name)
            pkg.__path__ = [str(utils_dir)]
            sys.modules[pkg_name] = pkg
            sys.modules[f"{pkg_name}.classifier"] = sys.modules["_e2e_classifier"]

        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.reporter", utils_dir / "reporter.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_entry(self):
        entry = MagicMock()
        entry.hf_id = "test/model"
        entry.task = "image-classification"
        entry.model_type = "resnet"
        entry.group = "Test"
        entry.priority = "P0"
        return entry

    def test_ep_omitted_when_none(self):
        reporter = self._load_reporter()

        result = reporter.build_eval_result(
            entry=self._make_entry(),
            perf_proc=None,
            device="cpu",
            eval_types_run=["accuracy"],
            accuracy_result=None,
            ep=None,
        )
        assert "ep" not in result

    def test_ep_present_when_provided(self):
        reporter = self._load_reporter()

        result = reporter.build_eval_result(
            entry=self._make_entry(),
            perf_proc=None,
            device="npu",
            eval_types_run=["accuracy"],
            accuracy_result=None,
            ep="qnn",
        )
        assert result["ep"] == "qnn"


class TestDefaultDatasetImmutability:
    """Tests that module-level _DEFAULT_DATASETS are not corrupted."""

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_default_dataset_not_mutated_after_evaluate(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """evaluate() must not corrupt _DEFAULT_DATASETS entries."""
        import importlib
        import sys
        from copy import deepcopy

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        # Snapshot the default datasets before evaluation
        defaults_before = deepcopy(eval_mod._DEFAULT_DATASETS)

        # Set up mocks: dataset with fewer samples than the default (100)
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 30
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {"accuracy": 0.7}
        mock_hf_eval.return_value = mock_eval_inst

        config = WinMLEvaluationConfig(
            model_id="test/model",
            dataset=DatasetConfig(path=None),
        )

        with (
            patch.object(eval_mod, "_load_model", return_value=MagicMock()),
            patch.object(eval_mod, "_resolve_task", return_value="image-classification"),
        ):
            eval_mod.evaluate(config)

        # Verify module-level defaults are unchanged (full dataclass state)
        from dataclasses import asdict

        for task, ds_cfg in eval_mod._DEFAULT_DATASETS.items():
            assert asdict(ds_cfg) == asdict(defaults_before[task]), (
                f"_DEFAULT_DATASETS['{task}'] was mutated"
            )

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_caller_dataset_not_mutated(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """evaluate() must not mutate the caller's DatasetConfig."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        # Dataset with fewer samples than requested
        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 30
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()

        mock_eval_inst = MagicMock()
        mock_eval_inst.compute.return_value = {"accuracy": 0.7}
        mock_hf_eval.return_value = mock_eval_inst

        caller_dataset = DatasetConfig(path="my-dataset", samples=100)
        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=caller_dataset,
        )

        with patch.object(eval_mod, "_load_model", return_value=MagicMock()):
            eval_mod.evaluate(config)

        # Caller's dataset must be untouched (full dataclass state)
        from dataclasses import asdict

        assert asdict(caller_dataset) == asdict(
            DatasetConfig(path="my-dataset", samples=100),
        ), "Caller's DatasetConfig was mutated"


class TestLoadModel:
    """Tests for _load_model."""

    def test_load_model_no_model_id_raises(self):
        """_load_model raises ValueError when model_id is None."""
        from winml.modelkit.eval.evaluate import _load_model

        config = WinMLEvaluationConfig(model_id=None)
        with pytest.raises(ValueError, match="model_id is required"):
            _load_model(config)

    def test_load_model_from_pretrained(self):
        """When no model_path, calls from_pretrained."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        mock_model = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            device="cpu",
        )

        with patch.dict(
            "sys.modules",
            {"winml.modelkit.models": MagicMock(WinMLAutoModel=mock_auto)},
        ):
            result = eval_mod._load_model(config)

        mock_auto.from_pretrained.assert_called_once_with(
            "test/model",
            task="image-classification",
            device="cpu",
            precision="auto",
            ep=None,
        )
        assert result is mock_model

    def test_load_model_from_onnx(self):
        """When model_path is set, calls from_onnx and attaches config."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        mock_model = MagicMock()
        mock_auto = MagicMock()
        mock_auto.from_onnx.return_value = mock_model
        mock_hf_config = MagicMock()

        config = WinMLEvaluationConfig(
            model_id="test/model",
            model_path="model.onnx",
            task="image-classification",
            device="cpu",
        )

        with (
            patch.dict(
                "sys.modules",
                {"winml.modelkit.models": MagicMock(WinMLAutoModel=mock_auto)},
            ),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=mock_hf_config,
            ),
        ):
            result = eval_mod._load_model(config)

        mock_auto.from_onnx.assert_called_once()
        assert result.config is mock_hf_config
