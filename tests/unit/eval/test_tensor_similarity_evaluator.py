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
from winml.modelkit.eval.tensor_similarity_evaluator import TensorSimilarityEvaluator
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
        composite = _FakeCompositeModel(
            sub_models={}, config=PretrainedConfig()
        )
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
