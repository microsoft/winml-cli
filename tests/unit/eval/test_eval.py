# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for eval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.eval import DatasetConfig, EvalResult, WinMLEvaluationConfig


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

    def test_config_roundtrip_preserves_revision(self):
        """DatasetConfig.revision survives to_dict/from_dict roundtrip."""
        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="depth-estimation",
            dataset=DatasetConfig(
                path="sayakpaul/nyu_depth_v2",
                revision="refs/convert/parquet",
            ),
        )
        restored = WinMLEvaluationConfig.from_dict(config.to_dict())
        assert restored.dataset.revision == "refs/convert/parquet"

    def test_dataset_config_revision_default_is_none(self):
        """Revision defaults to None when not specified."""
        ds = DatasetConfig(path="some-dataset")
        assert ds.revision is None
        assert "revision" not in ds.to_dict()

    def test_reference_path_default_is_none(self):
        """reference_path defaults to None and is omitted from to_dict."""
        config = WinMLEvaluationConfig(model_id="test/model")
        assert config.reference_path is None
        assert "reference_path" not in config.to_dict()

    def test_config_roundtrip_preserves_reference_path(self):
        """reference_path survives to_dict/from_dict roundtrip."""
        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
        )
        restored = WinMLEvaluationConfig.from_dict(config.to_dict())
        assert restored.reference_path == "ref.onnx"
        assert restored.mode == "compare"

    def test_input_data_default_is_none(self):
        """input_data defaults to None and is omitted from to_dict."""
        config = WinMLEvaluationConfig(model_id="test/model")
        assert config.input_data is None
        assert "input_data" not in config.to_dict()

    def test_config_roundtrip_preserves_input_data(self):
        """input_data survives to_dict/from_dict roundtrip."""
        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            input_data="inputs.npz",
            mode="compare",
        )
        restored = WinMLEvaluationConfig.from_dict(config.to_dict())
        assert restored.input_data == "inputs.npz"

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
        fake_resolution = MagicMock()
        fake_resolution.task = "image-classification"
        config = WinMLEvaluationConfig(model_id="microsoft/resnet-50")
        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=fake_hf_config,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=fake_resolution,
            ),
        ):
            assert _resolve_task(config) == "image-classification"

    def test_explicit_feature_extraction_preserved_verbatim(self):
        """Explicit --task is surfaced verbatim (explicit means explicit).

        The old reverse io_config upgrade (feature-extraction -> image-feature-extraction
        for vision models) is intentionally gone: per the canonical rule, a vision
        model's task is image-feature-extraction, so an explicit feature-extraction is
        out-of-domain and is not silently rewritten.
        """
        from winml.modelkit.eval.evaluate import _resolve_task

        config = WinMLEvaluationConfig(model_id="facebook/dinov2-base", task="feature-extraction")
        # feature-extraction is itself a registered (text) evaluator key, so resolution
        # returns it as-is; a vision model would then fail downstream at eval-run.
        assert _resolve_task(config) == "feature-extraction"

    def test_auto_detect_vision_feature_model_resolves_image_feature_extraction(self):
        """Auto-detect (no --task) for a vision embedding model resolves the
        modality-aware image-feature-extraction via resolve_task — the source-level
        fix for #778 that replaces the reverse io_config reconstruction."""
        from winml.modelkit.eval.evaluate import _resolve_task

        fake_resolution = MagicMock()
        fake_resolution.task = "image-feature-extraction"
        config = WinMLEvaluationConfig(model_id="facebook/dinov2-base")  # no explicit task
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=MagicMock()),
            patch(
                "winml.modelkit.loader.resolution.resolve_task",
                return_value=fake_resolution,
            ),
        ):
            assert _resolve_task(config) == "image-feature-extraction"


class TestGetEvaluatorClass:
    """Tests for get_evaluator_class registry lookup."""

    def test_registered_task_returns_class(self):
        from winml.modelkit.eval import WinMLEvaluationConfig, WinMLEvaluator, get_evaluator_class
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY

        # _EVALUATOR_REGISTRY stores "module_path:ClassName" strings so that
        # selecting one task does not eagerly import unrelated heavy
        # evaluators (e.g. fill-mask, zero-shot-classification, which pull
        # torch + transformers). Verify each entry resolves to a real
        # WinMLEvaluator subclass.
        for task, spec in _EVALUATOR_REGISTRY.items():
            assert isinstance(spec, str) and ":" in spec, (
                f"Registry value for {task!r} must be a 'module:Class' string."
            )
            cls = get_evaluator_class(WinMLEvaluationConfig(task=task))
            assert isinstance(cls, type)
            # Task evaluators inherit from WinMLEvaluator; "compare-tensor"
            # is a non-task entry (TensorSimilarityEvaluator) with its own
            # shape and is exempt from the base-class check.
            if task != "compare-tensor":
                assert issubclass(cls, WinMLEvaluator)
            # The resolved class must match the qualified name in the spec.
            module_path, class_name = spec.rsplit(":", 1)
            assert cls.__module__ == module_path
            assert cls.__name__ == class_name

    def test_unsupported_task_raises_value_error(self):
        from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class

        with pytest.raises(ValueError, match="not supported by `winml eval`"):
            get_evaluator_class(WinMLEvaluationConfig(task="made-up-task"))

    def test_evaluator_registry_matches_schema_tasks(self):
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY
        from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

        # "compare-tensor" is a non-task evaluator entry (no labeled-dataset
        # schema); exclude it from the task<->schema equivalence check.
        assert set(_EVALUATOR_REGISTRY) - {"compare-tensor"} == set(TASK_SCHEMAS)


class TestEvaluate:
    """Tests for evaluate() entry point."""

    def test_invalid_mode_raises(self):
        """evaluate() rejects unknown mode values with a clear error."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        config = WinMLEvaluationConfig(model_id="test/model", task="feature-extraction")
        config.mode = "hf"  # bypass dataclass type hint

        with pytest.raises(ValueError, match="Invalid mode"):
            eval_mod.evaluate(config)

    def test_none_mode_normalizes_to_onnx(self):
        """evaluate() treats mode=None as the default onnx mode."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="imagenet-1k"),
        )
        config.mode = None  # bypass dataclass type hint

        evaluator = MagicMock()
        evaluator.compute.return_value = {"accuracy": 1.0}
        with (
            patch.object(eval_mod, "_load_model", return_value=MagicMock()),
            patch.object(eval_mod, "get_evaluator_class", return_value=lambda *_a, **_k: evaluator),
        ):
            result = eval_mod.evaluate(config)
        assert result.config.mode == "onnx"

    def test_onnx_compare_skips_task_resolution_and_dataset(self):
        """Two-ONNX compare skips HF task resolution and default-dataset lookup."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
        )

        evaluator = MagicMock()
        evaluator.compute.return_value = {"cosine_mean": {"logits": 1.0}}
        with (
            patch.object(
                eval_mod,
                "_resolve_task",
                side_effect=AssertionError("task resolution must be skipped"),
            ),
            patch.object(eval_mod, "_load_model", return_value=None) as load_model,
            patch.object(eval_mod, "get_evaluator_class", return_value=lambda *_a, **_k: evaluator),
        ):
            result = eval_mod.evaluate(config)

        assert result.config.mode == "compare"
        assert result.config.task is None
        assert result.metrics == {"cosine_mean": {"logits": 1.0}}
        load_model.assert_called_once()

    def test_load_model_returns_none_for_onnx_compare(self):
        """_load_model short-circuits (no model_id needed) for two-ONNX compare."""
        from winml.modelkit.eval.evaluate import _load_model

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
        )
        assert _load_model(config) is None

    def test_no_dataset_no_default_raises(self):
        """Tasks without a default dataset raise ValueError."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        task_without_default = next(
            t
            for t in ["image-segmentation", "next-sentence-prediction", "image-to-text"]
            if t in eval_mod._EVALUATOR_REGISTRY and t not in eval_mod._DEFAULT_DATASETS
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
            dataset=DatasetConfig(path="some/dataset"),
        )
        original = asdict(config)

        mock_evaluator = MagicMock()
        mock_evaluator.compute.return_value = {"accuracy": 0.8}

        with (
            patch.object(eval_mod, "_resolve_task", return_value="text-classification"),
            patch.object(eval_mod, "_load_model", return_value=MagicMock()),
            # _EVALUATOR_REGISTRY now stores "module:Class" strings; patch the
            # public resolver instead of injecting a callable into the dict.
            patch.object(
                eval_mod,
                "get_evaluator_class",
                return_value=lambda *a: mock_evaluator,
            ),
        ):
            eval_mod.evaluate(config)

        assert asdict(config) == original, "evaluate() mutated the caller's config"

    def test_prints_config_before_model_load_failure(self):
        """Users should see the effective config even when model loading fails."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        calls = []
        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="test-dataset"),
        )

        def fake_print_config(_config):
            calls.append("print")

        def fake_load_model(_config):
            calls.append("load")
            raise RuntimeError("loader failed")

        with (
            patch.object(eval_mod, "print_config", side_effect=fake_print_config),
            patch.object(eval_mod, "_load_model", side_effect=fake_load_model),
            pytest.raises(ValueError) as exc_info,
        ):
            eval_mod.evaluate(config)

        assert calls == ["print", "load"]
        assert "Failed to load model 'test/model'" in str(exc_info.value)
        assert "expected model inputs" not in str(exc_info.value)

    def test_metric_runtime_error_propagates_without_schema_hint(self):
        """Internal evaluator failures should not be relabeled as schema issues."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        class FailingEvaluator:
            def __init__(self, _config, _model):
                pass

            def compute(self):
                raise RuntimeError("internal evaluator failure")

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="test-dataset"),
        )

        with (
            patch.object(eval_mod, "print_config", return_value=None),
            patch.object(eval_mod, "_load_model", return_value=object()),
            patch.object(eval_mod, "get_evaluator_class", return_value=FailingEvaluator),
            pytest.raises(RuntimeError, match="internal evaluator failure"),
        ):
            eval_mod.evaluate(config)

    def test_metric_data_shape_errors_keep_schema_hint(self):
        """Known data-shape exceptions still get a concise schema hint."""
        import importlib
        import sys

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        class FailingEvaluator:
            def __init__(self, _config, _model):
                pass

            def compute(self):
                raise KeyError("label")

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="test-dataset"),
        )

        with (
            patch.object(eval_mod, "print_config", return_value=None),
            patch.object(eval_mod, "_load_model", return_value=object()),
            patch.object(eval_mod, "get_evaluator_class", return_value=FailingEvaluator),
            pytest.raises(ValueError, match="expected schema") as exc_info,
        ):
            eval_mod.evaluate(config)

        assert isinstance(exc_info.value.__cause__, KeyError)


class TestWinMLEvaluator:
    """Tests for WinMLEvaluator base class."""

    @patch("datasets.load_dataset")
    def test_load_dataset_failure_wrapped_as_validation_error(self, mock_load_ds):
        """load_dataset failures surface as DatasetValidationError with dataset context."""
        from winml.modelkit.eval import WinMLEvaluator
        from winml.modelkit.utils.eval_utils import DatasetValidationError

        mock_load_ds.side_effect = ValueError(
            "Unknown split \"validation\". Should be one of ['train', 'val'].",
        )

        model = MagicMock()
        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="detection-datasets/fashionpedia", split="validation"),
        )

        with pytest.raises(DatasetValidationError) as exc_info:
            WinMLEvaluator(config, model)

        msg = str(exc_info.value)
        assert "Failed to load dataset 'detection-datasets/fashionpedia'" in msg
        assert "split='validation'" in msg
        assert "Unknown split" in msg
        assert isinstance(exc_info.value.__cause__, ValueError)

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
    def test_revision_passed_to_load_dataset(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """DatasetConfig.revision is forwarded to load_dataset()."""
        from winml.modelkit.eval import WinMLEvaluator

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 10
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()
        mock_hf_eval.return_value = MagicMock(compute=MagicMock(return_value={}))

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(
                path="some/dataset",
                samples=5,
                revision="refs/convert/parquet",
            ),
        )

        WinMLEvaluator(config, model)

        mock_load_ds.assert_called_once()
        assert mock_load_ds.call_args.kwargs["revision"] == "refs/convert/parquet"

    @patch("evaluate.evaluator")
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_revision_defaults_to_none(
        self,
        mock_load_ds,
        mock_pipeline,
        mock_hf_eval,
    ):
        """When revision is unset, load_dataset receives revision=None."""
        from winml.modelkit.eval import WinMLEvaluator

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 10
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds
        mock_pipeline.return_value = MagicMock()
        mock_hf_eval.return_value = MagicMock(compute=MagicMock(return_value={}))

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            dataset=DatasetConfig(path="some/dataset", samples=5),
        )

        WinMLEvaluator(config, model)

        mock_load_ds.assert_called_once()
        assert mock_load_ds.call_args.kwargs["revision"] is None

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
        """When --ep is omitted, ep is read from raw build-config JSON."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        config_file = tmp_path / "build.yaml"
        config_file.touch()

        raw_cfg = {"compile": {"execution_provider": "dml"}}

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("gpu", ["gpu", "cpu"])),
            patch(
                "winml.modelkit.utils.cli.load_build_config",
                return_value=(MagicMock(), raw_cfg),
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

        raw_cfg = {"compile": {"execution_provider": "dml"}}

        runner = CliRunner()
        with (
            patch("winml.modelkit.sysinfo.resolve_device", return_value=("npu", ["npu", "cpu"])),
            patch(
                "winml.modelkit.utils.cli.load_build_config",
                return_value=(MagicMock(), raw_cfg),
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

    def test_sanitize_fn_preserves_raw_perf_output(self):
        reporter = self._load_reporter()

        perf_proc = {
            "exit_code": 0,
            "stdout": "Latency (ms): 12.5\nThroughput: 80 samples/sec\nsome error line",
            "stderr": "warning: device busy",
            "elapsed": 5.0,
            "timeout": False,
            "command": "winml perf",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }

        def strip_perf(text: str) -> str:
            return "\n".join(
                line
                for line in text.splitlines()
                if "latency" not in line.lower() and "throughput" not in line.lower()
            )

        result = reporter.build_eval_result(
            entry=self._make_entry(),
            perf_proc=perf_proc,
            device="cpu",
            eval_types_run=["perf"],
            accuracy_result=None,
            ep=None,
            sanitize_fn=strip_perf,
        )

        perf = result["perf"]
        # sanitized output should not contain latency/throughput lines
        assert "Latency" not in perf["stdout_output"]
        assert "Throughput" not in perf["stdout_output"]
        # raw output preserves the original perf data
        assert "Latency (ms): 12.5" in perf["raw_stdout"]
        assert "Throughput: 80 samples/sec" in perf["raw_stdout"]
        assert perf["raw_stderr"] == "warning: device busy"


class TestDefaultDatasetImmutability:
    """Tests that module-level _DEFAULT_DATASETS are not corrupted."""

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

        # quant defaults to True -> no quant override (config=None); optimize/
        # analyze default True with max_optim_iterations=None -> no extra build
        # kwargs (build_pipeline_extra_kwargs returns {}).
        mock_auto.from_pretrained.assert_called_once_with(
            "test/model",
            task="image-classification",
            device="cpu",
            precision="auto",
            ep=None,
            allow_unsupported_nodes=False,
            config=None,
        )
        assert result is mock_model

    def test_load_model_forwards_build_flags(self):
        """--no-quant/--no-optimize/--max-optim-iterations reach from_pretrained."""
        import importlib
        import sys

        from winml.modelkit.config import WinMLBuildConfig

        eval_mod = sys.modules.get(
            "winml.modelkit.eval.evaluate",
        ) or importlib.import_module("winml.modelkit.eval.evaluate")

        mock_auto = MagicMock()
        mock_auto.from_pretrained.return_value = MagicMock()

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="image-classification",
            device="cpu",
            quant=False,
            optimize=False,
            max_optim_iterations=5,
        )

        with patch.dict(
            "sys.modules",
            {"winml.modelkit.models": MagicMock(WinMLAutoModel=mock_auto)},
        ):
            eval_mod._load_model(config)

        kwargs = mock_auto.from_pretrained.call_args.kwargs
        # --no-quant -> WinMLBuildConfig override with quant cleared.
        assert isinstance(kwargs["config"], WinMLBuildConfig)
        assert kwargs["config"].quant is None
        # --no-optimize -> skip_optimize; --max-optim-iterations 5 forwarded.
        assert kwargs["skip_optimize"] is True
        assert kwargs["hack_max_optim_iterations"] == 5

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
