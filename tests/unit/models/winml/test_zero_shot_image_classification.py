# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for the zero-shot-image-classification composite model.

``WinMLModelForZeroShotImageClassification`` is registered directly for both
``("clip", "zero-shot-image-classification")`` and
``("siglip", "zero-shot-image-classification")`` entries in
``COMPOSITE_MODEL_REGISTRY`` — no per-family subclass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from winml.modelkit.models.winml.composite_model import (
    COMPOSITE_MODEL_REGISTRY,
    WinMLCompositeModel,
)
from winml.modelkit.models.winml.zero_shot_image_classification import (
    WinMLModelForZeroShotImageClassification,
    ZeroShotImageClassifierOutput,
)


def _make_sub_model(
    output: dict[str, np.ndarray] | list[dict[str, np.ndarray]] | None = None,
    *,
    io_config: dict | None = None,
) -> MagicMock:
    """Make a stubbed sub-model with `_session.run` + `_format_inputs` + `io_config`."""
    m = MagicMock()
    m._session = MagicMock()
    if isinstance(output, list):
        m._session.run.side_effect = output
    elif output is not None:
        m._session.run.return_value = output
    m.io_config = io_config or {}
    # Forward torch→numpy like the real WinMLPreTrainedModel._format_inputs
    m._format_inputs = lambda **kw: {
        k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        for k, v in kw.items()
        if v is not None
    }
    return m


# ---------------------------------------------------------------------------
# Registry / _SUB_MODEL_CONFIG
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_clip_registered(self):
        assert COMPOSITE_MODEL_REGISTRY[("clip", "zero-shot-image-classification")] is \
            WinMLModelForZeroShotImageClassification

    def test_siglip_registered(self):
        assert COMPOSITE_MODEL_REGISTRY[("siglip", "zero-shot-image-classification")] is \
            WinMLModelForZeroShotImageClassification

    def test_sub_model_config(self):
        config = WinMLModelForZeroShotImageClassification._SUB_MODEL_CONFIG
        assert config["image-encoder"] == "image-feature-extraction"
        assert config["text-encoder"] == "feature-extraction"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_inherits_composite_base(self):
        assert issubclass(WinMLModelForZeroShotImageClassification, WinMLCompositeModel)

    def test_accepts_none_config(self):
        """Constructor works with ``config=None`` (composite model needs no config fields)."""
        model = WinMLModelForZeroShotImageClassification(
            sub_models={"image-encoder": MagicMock(), "text-encoder": MagicMock()},
            config=None,
        )
        assert isinstance(model, WinMLModelForZeroShotImageClassification)


# ---------------------------------------------------------------------------
# Forward / preprocess
# ---------------------------------------------------------------------------


class TestForward:
    def _make_model(
        self,
        *,
        text_seq_len: int = 8,
        image_output_key: str = "image_embeds",
        text_output_key: str = "text_embeds",
    ) -> tuple[WinMLModelForZeroShotImageClassification, MagicMock, MagicMock]:
        vision = _make_sub_model(
            output={image_output_key: np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)},
            io_config={"input_shapes": [(1, 3, 224, 224)]},
        )
        text = _make_sub_model(
            output=[
                {text_output_key: np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)},
                {text_output_key: np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)},
            ],
            io_config={"input_shapes": [(1, text_seq_len), (1, text_seq_len)]},
        )
        model = WinMLModelForZeroShotImageClassification(
            sub_models={"image-encoder": vision, "text-encoder": text},
            config=None,
        )
        return model, vision, text

    def test_forward_returns_zero_shot_output(self):
        model, _, _ = self._make_model()
        pixel_values = torch.zeros(1, 3, 224, 224)
        input_ids = torch.zeros(2, 8, dtype=torch.int64)
        attention_mask = torch.ones(2, 8, dtype=torch.int64)

        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        assert isinstance(out, ZeroShotImageClassifierOutput)
        assert out.logits_per_image.shape == (1, 2)
        assert out.logits_per_text.shape == (2, 1)
        assert out.image_embeds.shape == (1, 4)
        assert out.text_embeds.shape == (2, 4)

    def test_forward_cosine_matches(self):
        """First label is aligned with image_embeds, second is orthogonal."""
        model, _, _ = self._make_model()
        pixel_values = torch.zeros(1, 3, 224, 224)
        input_ids = torch.zeros(2, 8, dtype=torch.int64)
        attention_mask = torch.ones(2, 8, dtype=torch.int64)

        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Raw cosine similarity (logit_scale/bias not applied — rank-invariant).
        assert out.logits_per_image[0, 0].item() == pytest.approx(1.0, rel=1e-4)
        assert out.logits_per_image[0, 1].item() == pytest.approx(0.0, abs=1e-4)

    def test_forward_accepts_pooler_output_name(self):
        """SigLIP-style ONNX with pooler_output fallback is resolved by priority list."""
        model, _, _ = self._make_model(
            image_output_key="pooler_output",
            text_output_key="pooler_output",
        )
        pixel_values = torch.zeros(1, 3, 224, 224)
        input_ids = torch.zeros(2, 8, dtype=torch.int64)
        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=None)
        assert out.image_embeds.shape == (1, 4)
        assert out.text_embeds.shape == (2, 4)

    def test_text_seq_padding(self):
        """Pipeline-provided [N, L] (L < ONNX seq_len) is padded to ONNX seq_len."""
        model, _, text = self._make_model(text_seq_len=8)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        pixel_values = torch.zeros(1, 3, 224, 224)

        model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=None)

        call_inputs = text._session.run.call_args_list[0][0][0]
        assert call_inputs["input_ids"].shape == (1, 8)
        assert call_inputs["input_ids"][0, :3].tolist() == [1, 2, 3]
        assert call_inputs["input_ids"][0, 3:].tolist() == [0] * 5

    def test_text_seq_noop_when_matching(self):
        """No padding when pipeline already provides the expected seq_len."""
        model, _, text = self._make_model(text_seq_len=8)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
                                dtype=torch.int64)
        pixel_values = torch.zeros(1, 3, 224, 224)

        model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=None)

        first_call = text._session.run.call_args_list[0][0][0]
        assert first_call["input_ids"].shape == (1, 8)
        assert first_call["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Output-key resolution
# ---------------------------------------------------------------------------


class TestOutputKeyResolution:
    def test_unknown_output_keys_raise(self):
        """Clear error when none of the priority keys are in the ONNX output."""
        vision = _make_sub_model(
            output={"something_unexpected": np.zeros((1, 4), dtype=np.float32)},
            io_config={"input_shapes": [(1, 3, 224, 224)]},
        )
        text = _make_sub_model(
            output=[{"text_embeds": np.zeros((1, 4), dtype=np.float32)}],
            io_config={"input_shapes": [(1, 8), (1, 8)]},
        )
        model = WinMLModelForZeroShotImageClassification(
            sub_models={"image-encoder": vision, "text-encoder": text},
            config=None,
        )
        with pytest.raises(KeyError, match="None of"):
            model(
                pixel_values=torch.zeros(1, 3, 224, 224),
                input_ids=torch.zeros(1, 8, dtype=torch.int64),
                attention_mask=None,
            )


# ---------------------------------------------------------------------------
# _run_text batching
# ---------------------------------------------------------------------------


class TestRunTextBatching:
    """``_run_text`` must honor the ONNX text encoder's fixed batch size.

    Each call's inputs must be a contiguous slice of the original ``inputs``
    with leading dim equal to ``batch_size``. When ``N`` is not a multiple of
    ``batch_size``, the last chunk is zero-padded to ``batch_size`` and the
    padding rows must be stripped from the concatenated output so the final
    tensor has shape ``(N, D)`` in the original input order.
    """

    @staticmethod
    def _build_text_sub_model(
        batch_size: int | None,
        seq_len: int = 8,
        embed_dim: int = 4,
    ):
        """Text sub-model whose session returns embeddings encoding input identity.

        For an input chunk ``ids`` of shape ``(B, L)``, the session returns an
        embedding of shape ``(B, D)`` where row ``i`` is ``ids[i, 0]`` broadcast
        across ``D``.  This lets tests assert that chunking preserves both the
        count and order of samples.
        """
        text = MagicMock()
        calls: list[np.ndarray] = []

        def run(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            ids = inputs["input_ids"]
            calls.append(ids.copy())
            embeds = np.broadcast_to(
                ids[:, :1].astype(np.float32), (ids.shape[0], embed_dim)
            ).copy()
            return {"text_embeds": embeds}

        text._session = MagicMock()
        text._session.run.side_effect = run
        text.io_config = {"input_shapes": [(batch_size, seq_len), (batch_size, seq_len)]}
        text._format_inputs = lambda **kw: {
            k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            for k, v in kw.items()
            if v is not None
        }
        return text, calls

    @staticmethod
    def _make_model(text_sub_model):
        return WinMLModelForZeroShotImageClassification(
            sub_models={"image-encoder": MagicMock(), "text-encoder": text_sub_model},
            config=None,
        )

    def test_batch_size_1_one_call_per_sample(self):
        text, calls = self._build_text_sub_model(batch_size=1)
        model = self._make_model(text)
        ids = np.arange(1, 4, dtype=np.int64).reshape(3, 1) * np.ones((1, 8), dtype=np.int64)
        out = model._run_text({"input_ids": ids})

        assert len(calls) == 3
        for i, call in enumerate(calls):
            assert call.shape == (1, 8)
            assert call[0, 0] == i + 1
        # Output rows must match input order: each row's values == input first token.
        assert out.shape == (3, 4)
        assert out[:, 0].tolist() == [1.0, 2.0, 3.0]

    def test_batch_size_even_multiple(self):
        """N=4, B=2: two calls of size 2; order preserved."""
        text, calls = self._build_text_sub_model(batch_size=2)
        model = self._make_model(text)
        ids = np.arange(1, 5, dtype=np.int64).reshape(4, 1) * np.ones((1, 8), dtype=np.int64)
        out = model._run_text({"input_ids": ids})

        assert len(calls) == 2
        assert [c.shape for c in calls] == [(2, 8), (2, 8)]
        assert calls[0][:, 0].tolist() == [1, 2]
        assert calls[1][:, 0].tolist() == [3, 4]
        assert out.shape == (4, 4)
        assert out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_batch_size_with_remainder_pads_and_strips(self):
        """N=5, B=2: three calls; final chunk padded to (2, 8); padding stripped from output."""
        text, calls = self._build_text_sub_model(batch_size=2)
        model = self._make_model(text)
        ids = np.arange(1, 6, dtype=np.int64).reshape(5, 1) * np.ones((1, 8), dtype=np.int64)
        out = model._run_text({"input_ids": ids})

        assert len(calls) == 3
        assert calls[0][:, 0].tolist() == [1, 2]
        assert calls[1][:, 0].tolist() == [3, 4]
        # Last chunk: real sample 5 + one zero-padded row (so batch dim == 2).
        assert calls[2].shape == (2, 8)
        assert calls[2][0, 0] == 5
        assert calls[2][1].tolist() == [0] * 8
        # Padding row must NOT leak into the output.
        assert out.shape == (5, 4)
        assert out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_batch_size_larger_than_n(self):
        """N=2, B=4: single call with 2 padding rows; output still shape (2, D)."""
        text, calls = self._build_text_sub_model(batch_size=4)
        model = self._make_model(text)
        ids = np.arange(1, 3, dtype=np.int64).reshape(2, 1) * np.ones((1, 8), dtype=np.int64)
        out = model._run_text({"input_ids": ids})

        assert len(calls) == 1
        assert calls[0].shape == (4, 8)
        assert calls[0][:, 0].tolist() == [1, 2, 0, 0]
        assert out.shape == (2, 4)
        assert out[:, 0].tolist() == [1.0, 2.0]

    def test_dynamic_batch_dim_single_call(self):
        """Dynamic batch dim (non-positive / non-int) falls back to one full-input call."""
        text, calls = self._build_text_sub_model(batch_size=None)
        model = self._make_model(text)
        ids = np.arange(1, 4, dtype=np.int64).reshape(3, 1) * np.ones((1, 8), dtype=np.int64)
        out = model._run_text({"input_ids": ids})

        assert len(calls) == 1
        assert calls[0].shape == (3, 8)
        assert out.shape == (3, 4)
        assert out[:, 0].tolist() == [1.0, 2.0, 3.0]

    def test_multi_input_chunks_stay_aligned(self):
        """attention_mask and input_ids must be sliced consistently per chunk."""
        text, _calls = self._build_text_sub_model(batch_size=2)
        # Override side_effect to capture both inputs.
        captured: list[dict[str, np.ndarray]] = []

        def run(inputs):
            captured.append({k: v.copy() for k, v in inputs.items()})
            return {
                "text_embeds": np.broadcast_to(
                    inputs["input_ids"][:, :1].astype(np.float32),
                    (inputs["input_ids"].shape[0], 4),
                ).copy()
            }

        text._session.run.side_effect = run
        model = self._make_model(text)
        ids = np.arange(1, 4, dtype=np.int64).reshape(3, 1) * np.ones((1, 8), dtype=np.int64)
        mask = np.arange(10, 13, dtype=np.int64).reshape(3, 1) * np.ones((1, 8), dtype=np.int64)
        model._run_text({"input_ids": ids, "attention_mask": mask})

        assert len(captured) == 2
        # Chunk 0: rows [0, 1] of both inputs.
        assert captured[0]["input_ids"][:, 0].tolist() == [1, 2]
        assert captured[0]["attention_mask"][:, 0].tolist() == [10, 11]
        # Chunk 1: row [2] + one padded zero row for each input.
        assert captured[1]["input_ids"][:, 0].tolist() == [3, 0]
        assert captured[1]["attention_mask"][:, 0].tolist() == [12, 0]
