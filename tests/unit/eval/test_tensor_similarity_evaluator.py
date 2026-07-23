# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for TensorSimilarityEvaluator.

Focuses on the ``_inference_model`` static helper and the composite-model
guard in ``__init__``. Per-sample metric math lives in
:mod:`TensorSimilarityMetric` (see ``test_tensor_similarity_metric.py``)
and end-to-end ``compute()`` is covered by ``tests/e2e/test_eval_e2e.py``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.tensor_similarity_evaluator import (
    TensorSimilarityEvaluator,
    _ONNXSessionModel,
)
from winml.modelkit.models.winml.composite_model import WinMLCompositeModel


# ---------------------------------------------------------------------------
# _inference_model
# ---------------------------------------------------------------------------


class _EchoModel:
    """Minimal stand-in that returns a BaseModelOutput from the inputs."""

    def __init__(self, output_dict):
        self._output = output_dict
        self.last_call_dtypes: dict[str, torch.dtype] = {}

    def __call__(self, **inputs):
        self.last_call_dtypes = {k: v.dtype for k, v in inputs.items()}
        return BaseModelOutput(last_hidden_state=self._output["last_hidden_state"])


class TestInferenceModel:
    def test_upcasts_narrow_int_to_int64(self):
        model = _EchoModel({"last_hidden_state": torch.zeros(1, 4)})
        sample = {
            "input_ids": torch.zeros(1, 8, dtype=torch.int32),
            "attention_mask": torch.ones(1, 8, dtype=torch.int8),
        }
        TensorSimilarityEvaluator._inference_model(model, sample)
        assert model.last_call_dtypes["input_ids"] == torch.int64
        assert model.last_call_dtypes["attention_mask"] == torch.int64

    def test_leaves_int64_and_float_untouched(self):
        model = _EchoModel({"last_hidden_state": torch.zeros(1, 4)})
        sample = {
            "input_ids": torch.zeros(1, 8, dtype=torch.int64),
            "pixel_values": torch.zeros(1, 3, 4, 4, dtype=torch.float32),
        }
        TensorSimilarityEvaluator._inference_model(model, sample)
        assert model.last_call_dtypes["input_ids"] == torch.int64
        assert model.last_call_dtypes["pixel_values"] == torch.float32

    def test_returns_numpy_dict_only_for_tensor_fields(self):
        model = _EchoModel({"last_hidden_state": torch.arange(6.0).reshape(1, 2, 3)})
        out = TensorSimilarityEvaluator._inference_model(
            model, {"input_ids": torch.zeros(1, 2, dtype=torch.int64)}
        )
        assert set(out) == {"last_hidden_state"}
        assert isinstance(out["last_hidden_state"], np.ndarray)
        assert out["last_hidden_state"].shape == (1, 2, 3)


# ---------------------------------------------------------------------------
# composite-model guard in __init__
# ---------------------------------------------------------------------------


class _FakeCompositeModel(WinMLCompositeModel):
    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "image-feature-extraction",
        "decoder": "text-generation",
    }


class TestCompositeGuard:
    def test_rejects_composite_with_helpful_message(self):
        composite = _FakeCompositeModel(sub_models={}, config=PretrainedConfig())
        config = WinMLEvaluationConfig(
            model_id="Salesforce/blip-image-captioning-base",
            task="image-to-text",
            mode="compare",
            dataset=DatasetConfig(),
        )
        with pytest.raises(TypeError) as exc:
            TensorSimilarityEvaluator(config, composite)
        msg = str(exc.value)
        assert "composite" in msg.lower()
        assert "image-feature-extraction" in msg
        assert "text-generation" in msg
        assert "Salesforce/blip-image-captioning-base" in msg


# ---------------------------------------------------------------------------
# Two-ONNX compare (reference_path set)
# ---------------------------------------------------------------------------


class _FakeSession:
    """Stand-in for WinMLSession that records construction and echoes outputs."""

    created: ClassVar[list[tuple[str, str, object]]] = []

    def __init__(self, onnx_path, device="auto", ep=None):
        _FakeSession.created.append((str(onnx_path), device, ep))
        self.io_config = {"input_names": ["input"], "input_types": ["float32"]}

    def run(self, inputs):
        return {"logits": np.arange(3.0, dtype=np.float32).reshape(1, 3)}


class _FakeRandomDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return 0


class TestONNXReferenceInit:
    def test_builds_two_raw_sessions_honoring_device(self, monkeypatch):
        import winml.modelkit.datasets.random_dataset as rd_mod
        import winml.modelkit.session.session as session_mod

        _FakeSession.created = []
        monkeypatch.setattr(session_mod, "WinMLSession", _FakeSession)
        monkeypatch.setattr(rd_mod, "RandomDataset", _FakeRandomDataset)

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            device="cpu",
            ep="dml",
            dataset=DatasetConfig(samples=5, seed=1),
        )

        # ``model`` is None in this path (evaluate._load_model returns None).
        evaluator = TensorSimilarityEvaluator(config, None)  # type: ignore[arg-type]

        assert isinstance(evaluator.model, _ONNXSessionModel)
        assert isinstance(evaluator.reference_model, _ONNXSessionModel)
        # Candidate first, reference second; both honor --device / --ep.
        assert _FakeSession.created[0][0].endswith("cand.onnx")
        assert _FakeSession.created[1][0].endswith("ref.onnx")
        assert [c[1] for c in _FakeSession.created] == ["cpu", "cpu"]
        assert [c[2] for c in _FakeSession.created] == ["dml", "dml"]
        # RandomDataset is built over the candidate ONNX I/O.
        assert evaluator.data.kwargs["model_path"].endswith("cand.onnx")
        assert evaluator.data.kwargs["max_samples"] == 5
        assert evaluator.data.kwargs["seed"] == 1


class TestONNXSessionModel:
    def test_call_returns_named_torch_tensors(self, monkeypatch):
        import winml.modelkit.session.session as session_mod

        _FakeSession.created = []
        monkeypatch.setattr(session_mod, "WinMLSession", _FakeSession)

        model = _ONNXSessionModel("x.onnx", device="cpu")
        out = model(input=torch.zeros(1, 3))

        assert set(out) == {"logits"}
        assert isinstance(out["logits"], torch.Tensor)
        assert out["logits"].shape == (1, 3)

    def test_io_config_delegates_to_session(self, monkeypatch):
        import winml.modelkit.session.session as session_mod

        _FakeSession.created = []
        monkeypatch.setattr(session_mod, "WinMLSession", _FakeSession)

        model = _ONNXSessionModel("x.onnx")
        assert model.io_config["input_names"] == ["input"]


# ---------------------------------------------------------------------------
# Real-input compare (input_data set)
# ---------------------------------------------------------------------------


class TestInputDataCompare:
    def test_prepare_data_uses_input_data_npz(self, monkeypatch, tmp_path):
        import winml.modelkit.session.session as session_mod
        from winml.modelkit.datasets.input_data import InputDataDataset

        _FakeSession.created = []
        monkeypatch.setattr(session_mod, "WinMLSession", _FakeSession)

        npz = tmp_path / "inputs.npz"
        np.savez(npz, input=np.ones((2, 3), dtype=np.float32))

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            input_data=str(npz),
        )

        # ``model`` is None in this path (evaluate._load_model returns None).
        evaluator = TensorSimilarityEvaluator(config, None)  # type: ignore[arg-type]

        assert isinstance(evaluator.data, InputDataDataset)
        # Leading axis is the sample axis: (2, 3) -> 2 samples of shape (1, 3).
        assert len(evaluator.data) == 2
        sample = evaluator.data[0]
        assert set(sample) == {"input"}
        assert isinstance(sample["input"], torch.Tensor)
        assert sample["input"].shape == (1, 3)
        # The effective config reflects the real sample count for the report/JSON.
        assert evaluator.config.dataset.samples == 2
