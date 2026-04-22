# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for the zero-shot-image-classification composite model.

``WinMLModelForZeroShotImageClassification`` is registered directly for both
``("clip", "zero-shot-image-classification")`` and
``("siglip", "zero-shot-image-classification")`` entries in
``PIPELINE_MODEL_REGISTRY`` — no per-family subclass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from winml.modelkit.models.winml.composite_model import (
    PIPELINE_MODEL_REGISTRY,
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
        assert PIPELINE_MODEL_REGISTRY[("clip", "zero-shot-image-classification")] is \
            WinMLModelForZeroShotImageClassification

    def test_siglip_registered(self):
        assert PIPELINE_MODEL_REGISTRY[("siglip", "zero-shot-image-classification")] is \
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
