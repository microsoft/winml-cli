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

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

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
# Real-input compare (input_data set)
# ---------------------------------------------------------------------------


class _FakeCandidateModel:
    """Minimal candidate stand-in exposing the ONNX I/O config prepare_data reads."""

    def __init__(
        self,
        io_config: dict,
        *,
        path: str = "cand.onnx",
        output_name: str = "logits",
    ) -> None:
        self.io_config = io_config
        self.onnx_path = Path(path)
        self.output_name = output_name

    def __call__(self, **inputs):
        value = next(iter(inputs.values()))
        return {self.output_name: value.reshape(1, -1)[:, :3]}


class TestInputDataCompare:
    def test_prepare_data_uses_input_data_npz(self, monkeypatch, tmp_path):
        from winml.modelkit.datasets.input_data import InputDataDataset

        npz = tmp_path / "inputs.npz"
        np.savez(npz, input=np.ones((2, 3), dtype=np.float32))

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            model_id="test/model",
            mode="compare",
            input_data=str(npz),
        )
        model = _FakeCandidateModel({"input_names": ["input"], "input_types": ["float32"]})

        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=object(),
        ):
            evaluator = TensorSimilarityEvaluator(config, model)  # type: ignore[arg-type]

        assert isinstance(evaluator.data, InputDataDataset)
        # Leading axis is the sample axis: (2, 3) -> 2 samples of shape (1, 3).
        assert len(evaluator.data) == 2
        sample = evaluator.data[0]
        assert set(sample) == {"input"}
        assert isinstance(sample["input"], torch.Tensor)
        assert sample["input"].shape == (1, 3)
        # prepare_data must NOT mutate the config -- the real sample count is
        # surfaced via EvalResult.num_samples, not written back here.
        assert evaluator.config.dataset.samples == 100


# ---------------------------------------------------------------------------
# Injected candidate and reference models
# ---------------------------------------------------------------------------


class _FakeRandomDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return 0


class TestReferenceLoading:
    def test_loads_onnx_reference_with_independent_config(self, monkeypatch):
        import winml.modelkit.datasets.random_dataset as rd_mod

        monkeypatch.setattr(rd_mod, "RandomDataset", _FakeRandomDataset)

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            device="npu",
            ep="qnn",
            reference_device="gpu",
            reference_ep="dml",
            dataset=DatasetConfig(samples=5, seed=1),
        )
        candidate = _FakeCandidateModel({"input_names": ["input"]})
        reference = _FakeCandidateModel({"input_names": ["input"]}, path="ref.onnx")

        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=reference,
        ) as load_model:
            evaluator = TensorSimilarityEvaluator(
                config,
                candidate,  # type: ignore[arg-type]
            )

        assert evaluator.model is candidate
        assert evaluator.reference_model is reference
        reference_config = load_model.call_args.args[0]
        assert reference_config.model_path == "ref.onnx"
        assert reference_config.reference_path is None
        assert reference_config.device == "gpu"
        assert reference_config.ep == "dml"
        assert reference_config.task is None
        assert load_model.call_args.kwargs["torch_dtype"] is torch.float32
        # RandomDataset is built over the candidate ONNX I/O.
        assert evaluator.data.kwargs["model_path"].endswith("cand.onnx")
        assert evaluator.data.kwargs["max_samples"] == 5
        assert evaluator.data.kwargs["seed"] == 1

    def test_implicit_hf_reference_loads_on_cpu_fp32(self, monkeypatch):
        import winml.modelkit.datasets.random_dataset as rd_mod

        monkeypatch.setattr(rd_mod, "RandomDataset", _FakeRandomDataset)
        config = WinMLEvaluationConfig(
            model_id="test/model",
            model_path="cand.onnx",
            task="image-classification",
            mode="compare",
            dataset=DatasetConfig(samples=1),
        )
        candidate = _FakeCandidateModel({"input_names": ["input"]})

        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=object(),
        ) as load_model:
            TensorSimilarityEvaluator(config, candidate)  # type: ignore[arg-type]

        reference_config = load_model.call_args.args[0]
        assert reference_config.runtime == "pytorch"
        assert reference_config.device == "cpu"
        assert load_model.call_args.kwargs["torch_dtype"] is torch.float32


class TestONNXReferenceWithInputData:
    def test_prepare_data_uses_input_data_over_candidate_model(self, tmp_path):
        from winml.modelkit.datasets.input_data import InputDataDataset

        npz = tmp_path / "inputs.npz"
        np.savez(npz, input=np.ones((3, 4), dtype=np.float32))

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            input_data=str(npz),
        )
        candidate = _FakeCandidateModel(
            {"input_names": ["input"], "input_types": ["float32"]}
        )
        reference = _FakeCandidateModel(
            {"input_names": ["input"], "input_types": ["float32"]},
            path="ref.onnx",
        )

        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=reference,
        ):
            evaluator = TensorSimilarityEvaluator(
                config,
                candidate,  # type: ignore[arg-type]
            )

        assert isinstance(evaluator.data, InputDataDataset)
        assert len(evaluator.data) == 3
        sample = evaluator.data[0]
        assert set(sample) == {"input"}
        assert sample["input"].shape == (1, 4)

    def test_compute_runs_real_inputs_through_both_models(self, tmp_path):
        npz = tmp_path / "inputs.npz"
        np.savez(npz, input=np.ones((2, 3), dtype=np.float32))

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            input_data=str(npz),
        )
        io_config = {"input_names": ["input"], "input_types": ["float32"]}
        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=_FakeCandidateModel(io_config, path="ref.onnx"),
        ):
            evaluator = TensorSimilarityEvaluator(
                config,
                _FakeCandidateModel(io_config),  # type: ignore[arg-type]
            )

        result = evaluator.compute()

        assert result
        assert all("logits" in per_output for per_output in result.values())


class TestReferenceLabelInDiagnostics:
    def test_non_overlapping_outputs_error_names_reference_onnx(self, tmp_path):
        npz = tmp_path / "inputs.npz"
        np.savez(npz, input=np.ones((1, 3), dtype=np.float32))

        config = WinMLEvaluationConfig(
            model_path="cand.onnx",
            reference_path="ref.onnx",
            mode="compare",
            input_data=str(npz),
        )
        io_config = {"input_names": ["input"], "input_types": ["float32"]}
        with patch(
            "winml.modelkit.eval.evaluate.load_model",
            return_value=_FakeCandidateModel(
                io_config,
                path="ref.onnx",
                output_name="ref_out",
            ),
        ):
            evaluator = TensorSimilarityEvaluator(
                config,
                _FakeCandidateModel(  # type: ignore[arg-type]
                    io_config,
                    output_name="cand_out",
                ),
            )

        with pytest.raises(ValueError) as exc:
            evaluator.compute()

        # Two raw ONNX sides -> the diagnostic must say "reference ONNX", never
        # the misleading "HF reference".
        msg = str(exc.value)
        assert "reference ONNX" in msg
        assert "HF reference" not in msg
