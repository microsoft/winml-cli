# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for InferenceEngine internals.

Covers bug fixes and new features in the staged changes:
  - Task override via local variables (thread-safety)
  - _validate_inputs with explicit schema parameter
  - _prepare_pipeline_input with explicit schema/mapping
  - _normalize_pipeline_output with postprocess callbacks
  - _accepted_pipeline_kwargs filtering
  - _build_param_entry / _discover_pipeline_params_from_task
  - load_schema_only
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

from winml.modelkit.inference import (
    TASK_REGISTRY,
    InferenceEngine,
    InputField,
    PipelineMapping,
)

# Private helpers under test — internal imports are acceptable per CLAUDE.md
# for _-prefixed implementation details.
from winml.modelkit.inference.engine import (
    _build_param_entry,
    _discover_pipeline_params_from_task,
    _find_build_artifacts,
    _pick_sample_value,
    _sanitize_numpy,
)


# ---------------------------------------------------------------------------
# _build_param_entry
# ---------------------------------------------------------------------------


class TestBuildParamEntry:
    """Covers the refactored _build_param_entry helper."""

    def test_well_known_param_overrides_sentinel_type(self) -> None:
        """top_k with sentinel default '' should get type='integer' from _WELL_KNOWN_PARAMS."""
        param = inspect.Parameter("top_k", inspect.Parameter.KEYWORD_ONLY, default="")
        entry = _build_param_entry("top_k", param)
        assert entry is not None
        assert entry["type"] == "integer"

    def test_param_with_real_default(self) -> None:
        """Param with a concrete int default should keep type='integer' and store default."""
        param = inspect.Parameter("top_k", inspect.Parameter.KEYWORD_ONLY, default=5)
        entry = _build_param_entry("top_k", param)
        assert entry is not None
        assert entry["type"] == "integer"
        assert entry["default"] == 5

    def test_unknown_param_no_default_no_sample_returns_none(self) -> None:
        """Params with no default and no well-known entry should be filtered out."""
        param = inspect.Parameter(
            "offset_mapping", inspect.Parameter.KEYWORD_ONLY, default=inspect.Parameter.empty
        )
        entry = _build_param_entry("offset_mapping", param)
        assert entry is None

    def test_boolean_param_with_default(self) -> None:
        """Boolean params should get type='boolean'."""
        param = inspect.Parameter("do_sample", inspect.Parameter.KEYWORD_ONLY, default=False)
        entry = _build_param_entry("do_sample", param)
        assert entry is not None
        assert entry["type"] == "boolean"
        assert entry["default"] is False

    def test_var_positional_and_var_keyword_are_skipped_upstream(self) -> None:
        """_build_param_entry itself doesn't skip *args/**kwargs — that's the caller's job.
        Verify that a normal KEYWORD_ONLY param IS returned."""
        param = inspect.Parameter("threshold", inspect.Parameter.KEYWORD_ONLY, default=None)
        entry = _build_param_entry("threshold", param)
        assert entry is not None
        assert entry["name"] == "threshold"


# ---------------------------------------------------------------------------
# _pick_sample_value
# ---------------------------------------------------------------------------


class TestPickSampleValue:
    def test_actual_default_wins(self) -> None:
        assert _pick_sample_value("top_k", "integer", 10) == "10"

    def test_well_known_fallback(self) -> None:
        assert _pick_sample_value("top_k", "integer", None) == "5"

    def test_type_fallback(self) -> None:
        assert _pick_sample_value("unknown_int", "integer", None) == "5"

    def test_no_sample_for_unknown(self) -> None:
        assert _pick_sample_value("unknown_thing", "any", None) is None


# ---------------------------------------------------------------------------
# _discover_pipeline_params_from_task (lightweight, no model load)
# ---------------------------------------------------------------------------


class TestDiscoverPipelineParamsFromTask:
    def test_returns_list_for_known_task(self) -> None:
        params = _discover_pipeline_params_from_task("text-classification")
        assert isinstance(params, list)
        names = {p["name"] for p in params}
        assert "top_k" in names

    def test_returns_empty_for_unknown_task(self) -> None:
        assert _discover_pipeline_params_from_task("not-a-real-task") == []

    def test_returns_empty_for_none(self) -> None:
        assert _discover_pipeline_params_from_task(None) == []


# ---------------------------------------------------------------------------
# _accepted_pipeline_kwargs filtering
# ---------------------------------------------------------------------------


class TestAcceptedPipelineKwargs:
    def test_filters_unknown_kwargs(self) -> None:
        engine = InferenceEngine()
        engine._pipeline_params = [{"name": "top_k"}, {"name": "threshold"}]
        accepted = engine._accepted_pipeline_kwargs()
        assert accepted == frozenset({"top_k", "threshold"})

    def test_returns_none_when_no_params(self) -> None:
        engine = InferenceEngine()
        engine._pipeline_params = None
        assert engine._accepted_pipeline_kwargs() is None


# ---------------------------------------------------------------------------
# _validate_inputs with explicit schema
# ---------------------------------------------------------------------------


class TestValidateInputsWithSchema:
    def test_explicit_schema_overrides_self(self) -> None:
        """When schema= is passed, it should be used instead of self._user_input_schema."""
        engine = InferenceEngine()
        # self._user_input_schema expects "image"
        engine._user_input_schema = [
            InputField(name="image", type="image", required=True),
        ]
        # Override schema expects "text"
        override_schema = [
            InputField(name="text", type="text", required=True),
        ]
        result = engine._validate_inputs({"text": "hello"}, schema=override_schema)
        assert result == {"text": "hello"}

    def test_explicit_none_skips_validation(self) -> None:
        """schema=None should skip validation even when self has a schema."""
        engine = InferenceEngine()
        engine._user_input_schema = [
            InputField(name="image", type="image", required=True),
        ]
        result = engine._validate_inputs({"anything": 123}, schema=None)
        assert result == {"anything": 123}


# ---------------------------------------------------------------------------
# _prepare_pipeline_input with explicit schema/mapping
# ---------------------------------------------------------------------------


class TestPreparePipelineInputOverride:
    def test_explicit_mapping_overrides_self(self) -> None:
        engine = InferenceEngine()
        engine._user_input_schema = None
        engine._pipeline_mapping = None

        override_schema = [
            InputField(name="text_1", type="text", required=True),
            InputField(name="text_2", type="text", required=True),
        ]
        override_mapping = PipelineMapping(pipe_input=["text_1", "text_2"], pipe_input_as_list=True)
        inputs = {"text_1": "hello", "text_2": "world"}
        result = engine._prepare_pipeline_input(
            inputs,
            {},
            None,
            schema=override_schema,
            mapping=override_mapping,
        )
        assert result == ["hello", "world"]

    def test_none_mapping_returns_inputs_as_is(self) -> None:
        engine = InferenceEngine()
        engine._pipeline_mapping = PipelineMapping(pipe_input="x")
        inputs = {"x": 1}
        result = engine._prepare_pipeline_input(inputs, {}, None, mapping=None)
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# _normalize_pipeline_output with task override
# ---------------------------------------------------------------------------


class TestNormalizePipelineOutputTask:
    def test_task_parameter_selects_postprocess(self) -> None:
        """Task param should look up postprocess from TASK_REGISTRY."""
        engine = InferenceEngine()
        engine._task = "feature-extraction"
        # Use a mock pipeline without a tokenizer so that
        # _postprocess_sentence_similarity skips the masking path.
        engine._pipeline = MagicMock(spec=[])

        raw = [[1.0, 2.0], [3.0, 4.0]]
        result = engine._normalize_pipeline_output(
            raw,
            inputs={"text_1": "a", "text_2": "b"},
            task="sentence-similarity",
        )
        # sentence-similarity postprocess returns [Prediction(label="similarity", ...)]
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].label == "similarity"

    def test_default_task_from_self(self) -> None:
        """When task is not passed, self._task should be used."""
        engine = InferenceEngine()
        engine._task = None
        # Raw classification-like output
        raw = [{"label": "cat", "score": 0.95}]
        result = engine._normalize_pipeline_output(raw)
        assert isinstance(result, list)
        assert result[0].label == "cat"


# ---------------------------------------------------------------------------
# predict() with task override — thread-safety via local variables
# ---------------------------------------------------------------------------


class TestPredictTaskOverride:
    def _make_loaded_engine(self) -> InferenceEngine:
        engine = InferenceEngine()
        model = MagicMock()
        model._session._ep = "CPUExecutionProvider"
        engine._model = model
        pipe = MagicMock(return_value=[{"label": "cat", "score": 0.9}])
        # Remove auto-created tokenizer so postprocess callbacks that
        # probe getattr(pipeline, "tokenizer", None) get None.
        del pipe.tokenizer
        engine._pipeline = pipe
        engine._task = "image-classification"
        engine._user_input_schema = TASK_REGISTRY["image-classification"].user_inputs
        engine._pipeline_mapping = TASK_REGISTRY["image-classification"].mapping
        engine._pipeline_params = [{"name": "top_k"}]
        engine._device = "cpu"
        engine._ep = None
        engine._model_id = "test/model"
        return engine

    def test_task_override_does_not_mutate_self(self) -> None:
        """predict(task=X) must NOT mutate self._task, self._user_input_schema, etc."""
        engine = self._make_loaded_engine()
        original_task = engine._task
        original_schema = engine._user_input_schema
        original_mapping = engine._pipeline_mapping

        # Override: feature-extraction pipeline returns nested list
        engine._pipeline.return_value = [[[0.1, 0.2]], [[0.3, 0.4]]]
        result = engine.predict(
            inputs={"text_1": "hello", "text_2": "world"},
            task="sentence-similarity",
        )

        assert result.task == "sentence-similarity"
        # Self state must be unchanged
        assert engine._task == original_task
        assert engine._user_input_schema is original_schema
        assert engine._pipeline_mapping is original_mapping

    def test_predict_without_override_uses_self_task(self) -> None:
        engine = self._make_loaded_engine()
        # Switch to a text task to avoid needing a real image file
        engine._task = "text-classification"
        engine._user_input_schema = TASK_REGISTRY["text-classification"].user_inputs
        engine._pipeline_mapping = TASK_REGISTRY["text-classification"].mapping
        engine._pipeline.return_value = [{"label": "pos", "score": 0.9}]
        result = engine.predict(inputs={"text": "hello"})
        assert result.task == "text-classification"


# ---------------------------------------------------------------------------
# load_schema_only
# ---------------------------------------------------------------------------


class TestLoadForwardsAllowUnsupportedNodes:
    def test_load_from_hf_forwards_flag(self) -> None:
        """``allow_unsupported_nodes`` reaches WinMLAutoModel.from_pretrained."""
        engine = InferenceEngine()
        captured: dict[str, Any] = {}

        fake_model = MagicMock()
        fake_model.task = "text-classification"

        def _fake_from_pretrained(model_id: str, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return fake_model

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            side_effect=_fake_from_pretrained,
        ):
            engine._load_from_hf(
                "some/model",
                task="text-classification",
                device="cpu",
                ep=None,
                allow_unsupported_nodes=True,
            )

        assert captured.get("allow_unsupported_nodes") is True


class TestLoadSchemaOnly:
    def test_onnx_file_sets_model_id(self, tmp_path: Any) -> None:
        """load_schema_only for .onnx files should set _model_id."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        engine = InferenceEngine()
        engine.load_schema_only(onnx_path, task="image-classification")
        assert engine._model_id == str(onnx_path)
        assert engine._task == "image-classification"
        assert engine._user_input_schema is not None

    def test_build_dir_reads_manifest(self, tmp_path: Any) -> None:
        """load_schema_only for build dirs should read task from manifest."""
        import json

        manifest = {"model_id": "test/model", "task": "text-classification"}
        (tmp_path / "winml_manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "model.onnx").write_bytes(b"fake")
        engine = InferenceEngine()
        engine.load_schema_only(tmp_path)
        assert engine._task == "text-classification"
        assert engine._model_id == "test/model"
        assert engine._user_input_schema is not None

    def test_task_param_overrides_manifest(self, tmp_path: Any) -> None:
        """Explicit task= should override the manifest task."""
        import json

        manifest = {"model_id": "test/model", "task": "text-classification"}
        (tmp_path / "winml_manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "model.onnx").write_bytes(b"fake")
        engine = InferenceEngine()
        engine.load_schema_only(tmp_path, task="image-classification")
        assert engine._task == "image-classification"

    def test_hub_onnx_ref_is_resolved_before_routing(self, tmp_path: Any) -> None:
        """A Hub-style ONNX ref (``<org>/<repo>/<path>.onnx``) must be
        resolved to a local path BEFORE the .onnx-suffix-and-exists check,
        otherwise it falls through to the HF model-id branch and tries to
        load a Hub-ONNX path string as if it were a transformers config.

        Regression test for ``winml run`` and ``winml serve`` on Hub refs
        like ``onnx-community/sam3-tracker-ONNX/onnx/...``.
        """
        from unittest.mock import patch

        local = tmp_path / "vision_encoder_int8.onnx"
        local.write_bytes(b"fake-onnx")
        hub_ref = "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"

        engine = InferenceEngine()
        # ``WinMLSession.load_schema_only`` now routes Hub-ONNX resolution
        # through the unified ``resolve_model_input`` (in
        # ``winml.modelkit.utils.model_input``). Patch the underlying
        # downloader so the lazy ``from ..loader.onnx_hub import
        # resolve_hf_onnx_path`` picks up the mock at call time.
        with patch(
            "winml.modelkit.loader.onnx_hub.resolve_hf_onnx_path",
            return_value=local,
        ) as mock_resolve:
            engine.load_schema_only(hub_ref, task="mask-generation")
        mock_resolve.assert_called_once()
        # After resolution the engine should treat the input as a local
        # ONNX file (not as an HF model id), which means _model_id is the
        # resolved local path string, not the original Hub ref.
        assert engine._model_id == str(local)
        assert engine._task == "mask-generation"


# ---------------------------------------------------------------------------
# _sanitize_numpy
# ---------------------------------------------------------------------------


class TestSanitizeNumpy:
    """Ensure numpy scalars are converted to Python types for JSON serialization."""

    def test_float32_to_float(self) -> None:
        import numpy as np

        result = _sanitize_numpy({"score": np.float32(0.95)})
        assert isinstance(result["score"], float)
        assert abs(result["score"] - 0.95) < 0.001

    def test_int64_to_int(self) -> None:
        import numpy as np

        result = _sanitize_numpy({"start": np.int64(10)})
        assert isinstance(result["start"], int)
        assert result["start"] == 10

    def test_nested_dict(self) -> None:
        import numpy as np

        ner_output = {
            "entity_group": "PER",
            "score": np.float32(0.998),
            "word": "John",
            "start": np.int64(0),
            "end": np.int64(4),
        }
        result = _sanitize_numpy(ner_output)
        assert isinstance(result["score"], float)
        assert isinstance(result["start"], int)
        assert isinstance(result["end"], int)
        assert result["entity_group"] == "PER"

    def test_list_of_dicts(self) -> None:
        import numpy as np

        raw = [
            {"entity_group": "PER", "score": np.float32(0.99)},
            {"entity_group": "LOC", "score": np.float32(0.85)},
        ]
        result = _sanitize_numpy(raw)
        assert all(isinstance(d["score"], float) for d in result)

    def test_plain_types_unchanged(self) -> None:
        result = _sanitize_numpy({"label": "cat", "score": 0.95, "count": 5})
        assert result == {"label": "cat", "score": 0.95, "count": 5}

    def test_ndarray_to_list(self) -> None:
        import numpy as np

        result = _sanitize_numpy({"embedding": np.array([1.0, 2.0, 3.0])})
        assert result["embedding"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# _find_build_artifacts
# ---------------------------------------------------------------------------


class TestFindBuildArtifacts:
    def test_plain_layout(self, tmp_path: Any) -> None:
        import json

        (tmp_path / "model.onnx").write_bytes(b"fake")
        manifest = {"model_id": "test/model", "task": "text-classification"}
        (tmp_path / "winml_manifest.json").write_text(json.dumps(manifest))
        onnx_path, m = _find_build_artifacts(tmp_path)
        assert onnx_path.name == "model.onnx"
        assert m["task"] == "text-classification"

    def test_prefixed_layout(self, tmp_path: Any) -> None:
        import json

        (tmp_path / "txtcls_abc123_model.onnx").write_bytes(b"fake")
        manifest = {"model_id": "test/model", "task": "text-classification"}
        (tmp_path / "txtcls_abc123_winml_manifest.json").write_text(json.dumps(manifest))
        onnx_path, m = _find_build_artifacts(tmp_path)
        assert onnx_path.name == "txtcls_abc123_model.onnx"
        assert m["task"] == "text-classification"

    def test_no_onnx_raises(self, tmp_path: Any) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            _find_build_artifacts(tmp_path)

    def test_onnx_without_manifest(self, tmp_path: Any) -> None:
        (tmp_path / "model.onnx").write_bytes(b"fake")
        onnx_path, m = _find_build_artifacts(tmp_path)
        assert onnx_path.name == "model.onnx"
        assert m is None

    def test_task_filter_selects_matching(self, tmp_path: Any) -> None:
        """When task= is specified, only return artifacts whose manifest matches."""
        import json

        # Two variants in same directory
        (tmp_path / "feat_aaa_model.onnx").write_bytes(b"fake-feat")
        (tmp_path / "feat_aaa_winml_manifest.json").write_text(
            json.dumps({"model_id": "m", "task": "feature-extraction"})
        )
        (tmp_path / "txtcls_bbb_model.onnx").write_bytes(b"fake-txtcls")
        (tmp_path / "txtcls_bbb_winml_manifest.json").write_text(
            json.dumps({"model_id": "m", "task": "text-classification"})
        )

        onnx_path, m = _find_build_artifacts(tmp_path, task="text-classification")
        assert "txtcls" in onnx_path.name
        assert m["task"] == "text-classification"

    def test_task_filter_no_match_raises(self, tmp_path: Any) -> None:
        """When task= doesn't match any manifest, raise FileNotFoundError."""
        import json

        import pytest

        (tmp_path / "feat_aaa_model.onnx").write_bytes(b"fake")
        (tmp_path / "feat_aaa_winml_manifest.json").write_text(
            json.dumps({"model_id": "m", "task": "feature-extraction"})
        )

        with pytest.raises(FileNotFoundError):
            _find_build_artifacts(tmp_path, task="text-classification")

    def test_task_none_returns_first(self, tmp_path: Any) -> None:
        """Without task filter, return the first candidate."""
        import json

        (tmp_path / "feat_aaa_model.onnx").write_bytes(b"fake")
        (tmp_path / "feat_aaa_winml_manifest.json").write_text(
            json.dumps({"model_id": "m", "task": "feature-extraction"})
        )

        onnx_path, _manifest = _find_build_artifacts(tmp_path, task=None)
        assert onnx_path.exists()


# ---------------------------------------------------------------------------
# _normalize_pipeline_output sanitizes NER numpy types
# ---------------------------------------------------------------------------


class TestNormalizeNEROutput:
    def test_ner_output_numpy_sanitized(self) -> None:
        """NER pipeline output with numpy.float32 scores should be serializable."""
        import numpy as np

        engine = InferenceEngine()
        engine._task = "token-classification"
        # NER-like output: list of dicts with numpy scalars
        raw = [
            {
                "entity_group": "PER",
                "score": np.float32(0.998),
                "word": "John",
                "start": np.int64(0),
                "end": np.int64(4),
            },
        ]
        result = engine._normalize_pipeline_output(raw)
        # Should be serializable — all numpy types converted
        import json

        json.dumps(result)  # Must not raise
