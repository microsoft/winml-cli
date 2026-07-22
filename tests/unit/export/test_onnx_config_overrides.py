# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for WinML CLI's custom OnnxConfig overrides.

Verifies that WinML CLI's custom OnnxConfig registrations (BertIOConfig,
CLIPTextModelIOConfig, CLIPVisionModelIOConfig, RobertaIOConfig, etc.)
correctly override Optimum's defaults. These overrides are critical for:

- BERT: Uses max_position_embeddings (e.g., 512) instead of Optimum's hardcoded 16
- CLIP Text: Exposes attention_mask and uses max_position_embeddings (77)
- CLIP Vision: Custom outputs (image_embeds instead of pooler_output)
- Roberta/XLM-R/CamemBERT: Adjusts max_position_embeddings for position offset
  (e.g., 514 -> 512) to prevent embedding index OOB during export

If these registrations fail or get overwritten, export produces wrong shapes
and missing inputs -- a silent, critical bug.

See also: modelkit/models/hf/bert.py, modelkit/models/hf/clip.py, modelkit/models/hf/roberta.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from optimum.exporters.tasks import TasksManager

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.export.io import (  # Testing internal implementation
    _populate_image_size_from_preprocessor,
)
from winml.modelkit.models.hf.roberta import _adjust_position_embeddings
from winml.modelkit.onnx.io import InputTensorSpec


# =============================================================================
# Class 1: Registration verification
# =============================================================================


class TestOnnxConfigRegistration:
    """Verify WinML CLI's custom OnnxConfig classes are registered with TasksManager.

    If registration fails, Optimum's defaults are used, causing wrong shapes
    and missing inputs (e.g., BERT sequence_length=16 instead of 512).
    """

    @pytest.mark.parametrize(
        "model_type,task,expected_config_class",
        [
            ("bert", "fill-mask", "BertIOConfig"),
            ("clip_text_model", "feature-extraction", "CLIPTextModelIOConfig"),
            ("clip_vision_model", "feature-extraction", "CLIPVisionModelIOConfig"),
            ("roberta", "fill-mask", "RobertaIOConfig"),
            ("xlm-roberta", "fill-mask", "XLMRobertaIOConfig"),
            ("camembert", "fill-mask", "CamemBERTIOConfig"),
            ("mpnet", "fill-mask", "MPNetIOConfig"),
            ("layoutlm", "question-answering", "LayoutLMQAIOConfig"),
            ("zoedepth", "depth-estimation", "ZoeDepthIOConfig"),
        ],
        ids=[
            "bert",
            "clip-text",
            "clip-vision",
            "roberta",
            "xlm-roberta",
            "camembert",
            "mpnet",
            "layoutlm-qa",
            "zoedepth",
        ],
    )
    def test_custom_config_registered(
        self, model_type: str, task: str, expected_config_class: str
    ) -> None:
        """Our custom OnnxConfig must be returned, NOT Optimum's default."""
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type=model_type,
            task=task,
            library_name="transformers",
        )
        # The constructor is a functools.partial; its .func is the class
        actual_class_name = config_constructor.func.__name__
        assert actual_class_name == expected_config_class, (
            f"Expected {expected_config_class} for {model_type}/{task}, "
            f"got {actual_class_name}. Custom registration may have failed."
        )


# =============================================================================
# Class 2: BERT sequence length override
# =============================================================================


class TestBertSequenceLengthOverride:
    """BERT must use max_position_embeddings for sequence_length.

    Optimum's default BertOnnxConfig uses a hardcoded sequence_length=16.
    Our BertIOConfig overrides this to use max_position_embeddings from the
    model config (e.g., 512 for bert-base-uncased, 32 in test fixtures).
    """

    def test_bert_dummy_inputs_use_max_position_embeddings(
        self,
        bert_config,
    ) -> None:
        """Dummy inputs must have seq_len = max_position_embeddings, not 16."""
        inputs = generate_dummy_inputs("bert", "fill-mask", bert_config)

        seq_len = inputs["input_ids"].shape[1]
        # bert_config fixture has max_position_embeddings=32
        assert seq_len == bert_config.max_position_embeddings, (
            f"Expected seq_len={bert_config.max_position_embeddings}, got {seq_len}. "
            f"BertIOConfig override may not be active."
        )
        assert seq_len != 16, (
            "seq_len is 16 (Optimum's hardcoded default). BertIOConfig override is NOT active."
        )

    def test_bert_io_specs_shape_matches(self, bert_config) -> None:
        """I/O spec shapes must reflect max_position_embeddings."""
        specs = resolve_io_specs("bert", "fill-mask", bert_config)

        # All text inputs (input_ids, attention_mask, token_type_ids)
        # should have shape (batch, max_position_embeddings)
        for name, shape in zip(specs["input_names"], specs["input_shapes"], strict=True):
            assert shape[1] == bert_config.max_position_embeddings, (
                f"Input '{name}' has shape[1]={shape[1]}, "
                f"expected {bert_config.max_position_embeddings}"
            )


class TestLayoutLMQuestionAnsweringOverride:
    """LayoutLM QA export must include bbox and safe token_type_ids."""

    @pytest.mark.parametrize("precision", ["fp32", "fp16"])
    def test_layoutlm_recipe_bbox_generates_valid_boxes(self, precision: str) -> None:
        """Both explicit recipes must use the same meaningful bbox contract."""
        recipe_path = (
            Path(__file__).parents[3]
            / "examples"
            / "recipes"
            / "impira_layoutlm-invoices"
            / "cpu"
            / "cpu"
            / f"question-answering_{precision}_config.json"
        )
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        bbox_config = next(
            tensor for tensor in recipe["export"]["input_tensors"] if tensor["name"] == "bbox"
        )

        assert bbox_config["value_range"] == [0, 1001]
        bbox = InputTensorSpec.from_dict(bbox_config).to_tensor()
        assert bbox.shape == (1, 512, 4)
        assert bbox.min().item() >= 0
        assert bbox.max().item() < 1001
        assert (bbox[..., 0] < bbox[..., 2]).all().item()
        assert (bbox[..., 1] < bbox[..., 3]).all().item()

    def test_layoutlm_qa_dummy_inputs_include_valid_boxes_and_zero_token_types(self) -> None:
        """Dummy inputs must use meaningful boxes while forcing token types to zero."""
        from transformers import LayoutLMConfig

        layoutlm_config = LayoutLMConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            max_2d_position_embeddings=1024,
            type_vocab_size=1,
        )

        inputs = generate_dummy_inputs("layoutlm", "question-answering", layoutlm_config)

        assert set(inputs) == {"input_ids", "bbox", "attention_mask", "token_type_ids"}
        assert inputs["input_ids"].shape == (1, layoutlm_config.max_position_embeddings)
        assert inputs["bbox"].shape == (1, layoutlm_config.max_position_embeddings, 4)
        assert inputs["token_type_ids"].shape == (1, layoutlm_config.max_position_embeddings)
        assert inputs["token_type_ids"].max().item() == 0
        bbox = inputs["bbox"]
        assert bbox.min().item() >= 0
        assert bbox.max().item() < 1001
        assert (bbox[..., 0] < bbox[..., 2]).all().item()
        assert (bbox[..., 1] < bbox[..., 3]).all().item()

        specs = resolve_io_specs("layoutlm", "question-answering", layoutlm_config)
        assert specs["value_ranges"]["bbox"] == (0, 1001)

    def test_layoutlm_qa_uses_usable_length_for_padding_offset(self) -> None:
        """RoBERTa-style position offsets must not generate out-of-range positions."""
        from transformers import LayoutLMConfig

        layoutlm_config = LayoutLMConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=34,
            max_2d_position_embeddings=1024,
            type_vocab_size=1,
            pad_token_id=1,
        )

        inputs = generate_dummy_inputs("layoutlm", "question-answering", layoutlm_config)

        assert inputs["input_ids"].shape == (1, 32)
        assert inputs["bbox"].shape == (1, 32, 4)
        assert inputs["attention_mask"].shape == (1, 32)
        assert inputs["token_type_ids"].shape == (1, 32)

        specs = resolve_io_specs("layoutlm", "question-answering", layoutlm_config)
        assert specs["value_ranges"]["token_type_ids"] == (0, 1)

    def test_layoutlm_qa_bbox_respects_smaller_2d_embedding_table(self) -> None:
        """Coordinates and width/height indexes must fit the configured 2D table."""
        from transformers import LayoutLMConfig

        layoutlm_config = LayoutLMConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            max_2d_position_embeddings=16,
            type_vocab_size=1,
        )

        inputs = generate_dummy_inputs("layoutlm", "question-answering", layoutlm_config)
        bbox = inputs["bbox"]
        widths = bbox[..., 2] - bbox[..., 0]
        heights = bbox[..., 3] - bbox[..., 1]

        assert bbox.shape == (1, layoutlm_config.max_position_embeddings, 4)
        assert bbox.min().item() >= 0
        assert bbox.max().item() < layoutlm_config.max_2d_position_embeddings
        assert ((widths > 0) & (widths < layoutlm_config.max_2d_position_embeddings)).all().item()
        assert ((heights > 0) & (heights < layoutlm_config.max_2d_position_embeddings)).all().item()

        specs = resolve_io_specs("layoutlm", "question-answering", layoutlm_config)
        assert specs["value_ranges"]["bbox"] == (0, 16)

    def test_layoutlm_qa_bbox_handles_minimum_valid_2d_table(self) -> None:
        """The two-coordinate edge case must still produce positive-area boxes."""
        from transformers import LayoutLMConfig

        layoutlm_config = LayoutLMConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=8,
            max_2d_position_embeddings=2,
            type_vocab_size=1,
        )

        bbox = generate_dummy_inputs("layoutlm", "question-answering", layoutlm_config)["bbox"]

        assert bbox.shape == (1, 8, 4)
        assert set(bbox.unique().tolist()) == {0, 1}
        assert (bbox[..., 0] < bbox[..., 2]).all().item()
        assert (bbox[..., 1] < bbox[..., 3]).all().item()

    def test_layoutlm_qa_io_specs_include_span_outputs(self) -> None:
        """LayoutLM QA specs expose document bbox input and span logits outputs."""
        from transformers import LayoutLMConfig

        layoutlm_config = LayoutLMConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
            max_2d_position_embeddings=1024,
            type_vocab_size=1,
        )

        specs = resolve_io_specs("layoutlm", "question-answering", layoutlm_config)

        assert specs["input_names"] == [
            "input_ids",
            "bbox",
            "attention_mask",
            "token_type_ids",
        ]
        assert specs["input_shapes"] == [(1, 32), (1, 32, 4), (1, 32), (1, 32)]
        assert specs["output_names"] == ["start_logits", "end_logits"]


# =============================================================================
# Class 3: _adjust_position_embeddings unit tests
# =============================================================================


class TestAdjustPositionEmbeddings:
    """Unit tests for _adjust_position_embeddings edge cases."""

    def _make_config(self, max_pos=34, pad_token_id=1):
        """Create a minimal config-like object for testing."""
        from types import SimpleNamespace

        return SimpleNamespace(max_position_embeddings=max_pos, pad_token_id=pad_token_id)

    def test_adjusts_correctly(self) -> None:
        cfg = self._make_config(max_pos=34, pad_token_id=1)
        _adjust_position_embeddings(cfg)
        assert cfg.max_position_embeddings == 32

    def test_no_adjustment_when_pad_token_id_zero(self) -> None:
        cfg = self._make_config(max_pos=512, pad_token_id=0)
        _adjust_position_embeddings(cfg)
        assert cfg.max_position_embeddings == 512

    def test_no_adjustment_when_pad_token_id_none(self) -> None:
        cfg = self._make_config(max_pos=512, pad_token_id=None)
        _adjust_position_embeddings(cfg)
        assert cfg.max_position_embeddings == 512

    def test_double_call_is_idempotent(self) -> None:
        cfg = self._make_config(max_pos=34, pad_token_id=1)
        _adjust_position_embeddings(cfg)
        _adjust_position_embeddings(cfg)
        assert cfg.max_position_embeddings == 32

    def test_raises_on_non_positive_result(self) -> None:
        cfg = self._make_config(max_pos=2, pad_token_id=5)
        with pytest.raises(ValueError, match="non-positive"):
            _adjust_position_embeddings(cfg)

    def test_skips_when_no_max_position_embeddings(self) -> None:
        from types import SimpleNamespace

        cfg = SimpleNamespace(pad_token_id=1)
        _adjust_position_embeddings(cfg)  # should not raise
        assert not hasattr(cfg, "max_position_embeddings")

    def test_pad_token_id_greater_than_one(self) -> None:
        cfg = self._make_config(max_pos=100, pad_token_id=3)
        _adjust_position_embeddings(cfg)
        assert cfg.max_position_embeddings == 96  # 100 - 3 - 1


# =============================================================================
# Class 4: Roberta-family sequence length override
# =============================================================================


class TestRobertaSequenceLengthOverride:
    """Roberta-family must adjust max_position_embeddings for position offset.

    Roberta/XLM-R/CamemBERT set max_position_embeddings = usable + pad_token_id + 1
    (e.g., 514 = 512 + 1 + 1). Using the raw value causes position index OOB
    during ONNX export tracing.

    Test fixtures use max_position_embeddings=34, pad_token_id=1, so the
    expected usable sequence_length = 34 - 1 - 1 = 32.
    """

    _EXPECTED_SEQ_LEN = 32  # 34 - pad_token_id(1) - 1

    @pytest.mark.parametrize(
        "model_type,config_fixture",
        [
            ("roberta", "roberta_config"),
            ("xlm-roberta", "xlm_roberta_config"),
            ("camembert", "camembert_config"),
            ("mpnet", "mpnet_config"),
        ],
        ids=["roberta", "xlm-roberta", "camembert", "mpnet"],
    )
    def test_dummy_inputs_use_adjusted_sequence_length(
        self, model_type: str, config_fixture: str, request
    ) -> None:
        """Dummy inputs must use adjusted seq_len, not raw max_position_embeddings."""
        config = request.getfixturevalue(config_fixture)
        inputs = generate_dummy_inputs(model_type, "fill-mask", config)

        seq_len = inputs["input_ids"].shape[1]
        assert seq_len == self._EXPECTED_SEQ_LEN, (
            f"Expected seq_len={self._EXPECTED_SEQ_LEN} (adjusted), got {seq_len}. "
            f"Raw max_position_embeddings=34 should be reduced by pad_token_id + 1."
        )
        assert seq_len != 34, (
            "seq_len=34 (raw max_position_embeddings). "
            "Roberta position offset adjustment is NOT active."
        )

    @pytest.mark.parametrize(
        "model_type,config_fixture",
        [
            ("roberta", "roberta_config"),
            ("xlm-roberta", "xlm_roberta_config"),
            ("camembert", "camembert_config"),
            ("mpnet", "mpnet_config"),
        ],
        ids=["roberta", "xlm-roberta", "camembert", "mpnet"],
    )
    def test_io_specs_shape_uses_adjusted_length(
        self, model_type: str, config_fixture: str, request
    ) -> None:
        """I/O spec shapes must reflect adjusted sequence_length."""
        config = request.getfixturevalue(config_fixture)
        specs = resolve_io_specs(model_type, "fill-mask", config)

        for name, shape in zip(specs["input_names"], specs["input_shapes"], strict=True):
            assert shape[1] == self._EXPECTED_SEQ_LEN, (
                f"Input '{name}' has shape[1]={shape[1]}, "
                f"expected {self._EXPECTED_SEQ_LEN} (adjusted)"
            )

    def test_bert_not_affected_by_roberta_adjustment(self, bert_config) -> None:
        """BERT (pad_token_id=0) must NOT be affected by Roberta adjustment."""
        inputs = generate_dummy_inputs("bert", "fill-mask", bert_config)
        seq_len = inputs["input_ids"].shape[1]
        # bert_config has max_position_embeddings=32, no offset needed
        assert seq_len == bert_config.max_position_embeddings, (
            f"BERT seq_len={seq_len}, expected {bert_config.max_position_embeddings}. "
            f"Roberta adjustment may have incorrectly affected BERT."
        )


# =============================================================================
# Class 5: CLIP text override
# =============================================================================


class TestCLIPTextOverride:
    """CLIP text model must use max_position_embeddings and expose attention_mask.

    Optimum's default CLIPTextWithProjectionOnnxConfig:
    - Uses hardcoded sequence_length=16
    - May not expose attention_mask

    Our CLIPTextModelIOConfig overrides both behaviors.
    """

    def test_clip_text_uses_max_position_embeddings(
        self,
        clip_text_config,
    ) -> None:
        """CLIP text sequence_length = max_position_embeddings (32 in test fixture)."""
        inputs = generate_dummy_inputs("clip_text_model", "feature-extraction", clip_text_config)

        seq_len = inputs["input_ids"].shape[1]
        assert seq_len == clip_text_config.max_position_embeddings, (
            f"Expected seq_len={clip_text_config.max_position_embeddings}, "
            f"got {seq_len}. CLIPTextModelIOConfig override may not be active."
        )

    def test_clip_text_exposes_attention_mask(self, clip_text_config) -> None:
        """CLIP text must include attention_mask in input_names."""
        specs = resolve_io_specs("clip_text_model", "feature-extraction", clip_text_config)
        assert "attention_mask" in specs["input_names"], (
            f"attention_mask missing from input_names: {specs['input_names']}. "
            f"CLIPTextModelIOConfig override may not be active."
        )


# =============================================================================
# Class 6: Image size resolution
# =============================================================================


class TestImageSizeResolution:
    """Test _populate_image_size_from_preprocessor behavior.

    This function reads preprocessor_config.json from HuggingFace Hub to
    populate height/width in shape_kwargs. Tests verify edge cases without
    network access.
    """

    def test_existing_height_width_not_overridden(self) -> None:
        """If height/width already in shape_kwargs, they must not be overridden."""
        kwargs: dict = {"height": 384, "width": 384}
        _populate_image_size_from_preprocessor("microsoft/resnet-50", kwargs)
        assert kwargs["height"] == 384, "Existing height was overridden"
        assert kwargs["width"] == 384, "Existing width was overridden"

    def test_no_model_id_is_noop(self) -> None:
        """If model_id is None, shape_kwargs must remain unchanged."""
        kwargs: dict = {}
        _populate_image_size_from_preprocessor(None, kwargs)
        assert "height" not in kwargs
        assert "width" not in kwargs

    def test_invalid_model_id_is_noop(self) -> None:
        """If model_id is invalid, function must not crash (graceful no-op)."""
        kwargs: dict = {}
        # Should not raise - errors are caught internally
        _populate_image_size_from_preprocessor("nonexistent/model-xyz-999", kwargs)
        assert "height" not in kwargs
        assert "width" not in kwargs


# =============================================================================
# Class 7: Roberta-family WinMLBuildConfig registration
# =============================================================================


class TestRobertaFamilyBuildConfig:
    """Verify Roberta-family models have clamp_constant_values in build config.

    Roberta-family models (like BERT) produce extreme float constants
    (e.g., -FLT_MAX for attention masking) after ONNX optimization. Without
    clamp_constant_values=True, these extreme values widen the quantization
    range, causing precision loss in INT8/INT16 quantization.
    """

    @pytest.mark.parametrize(
        "model_type",
        ["roberta", "xlm-roberta", "camembert"],
        ids=["roberta", "xlm-roberta", "camembert"],
    )
    def test_build_config_registered(self, model_type: str) -> None:
        """Roberta-family models must be in MODEL_BUILD_CONFIGS."""
        from winml.modelkit.models.hf import MODEL_BUILD_CONFIGS

        assert model_type in MODEL_BUILD_CONFIGS, (
            f"'{model_type}' missing from MODEL_BUILD_CONFIGS. "
            f"Quantization may produce poor results without clamp_constant_values."
        )

    @pytest.mark.parametrize(
        "model_type",
        ["roberta", "xlm-roberta", "camembert"],
        ids=["roberta", "xlm-roberta", "camembert"],
    )
    def test_clamp_constant_values_enabled(self, model_type: str) -> None:
        """Roberta-family models must have clamp_constant_values=True."""
        from winml.modelkit.models.hf import MODEL_BUILD_CONFIGS

        config = MODEL_BUILD_CONFIGS[model_type]
        assert config.optim.get("clamp_constant_values") is True, (
            f"'{model_type}' WinMLBuildConfig missing clamp_constant_values=True. "
            f"Extreme constants (-FLT_MAX) will cause quantization precision loss."
        )

    def test_roberta_family_share_same_config(self) -> None:
        """All Roberta-family types should reference the same config object."""
        from winml.modelkit.models.hf import MODEL_BUILD_CONFIGS

        configs = [MODEL_BUILD_CONFIGS[t] for t in ("roberta", "xlm-roberta", "camembert")]
        assert configs[0] is configs[1] is configs[2], (
            "Roberta-family configs should be the same object (ROBERTA_FAMILY_CONFIG)"
        )
