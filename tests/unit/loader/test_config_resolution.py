# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for config-driven resolution in resolve_task.

Covers gaps NOT tested by existing loader tests:
- Auto-detect for new architecture categories (text_decoder, seq2seq, detection)
- User-task alias preservation (returning original task, not normalized)
- Double-lookup order (original then normalized) preventing CLIP task collapsing
- model_type normalization (underscore -> hyphen, case insensitive)

Uses mock configs (no network) with real resolve_task calls.
"""

from __future__ import annotations

import pytest

# Trigger ONNX config registrations and MODEL_CLASS_MAPPING population
import winml.modelkit.models  # noqa: F401
from winml.modelkit.loader.resolution import resolve_task


class TestResolveAutoDetectNewArchitectures:
    """Auto-detect task for architecture categories missing from existing tests.

    Existing tests cover: BERT (text_encoder), ResNet (vision), BLIP (multimodal).
    These tests extend coverage to: text_decoder, seq2seq, detection, additional vision.
    """

    @pytest.mark.parametrize(
        ("model_type", "arch_class_name", "expected_task"),
        [
            # text_decoder (NOT tested)
            pytest.param("gpt2", "GPT2LMHeadModel", "text-generation", id="gpt2"),
            pytest.param("llama", "LlamaForCausalLM", "text-generation", id="llama"),
            # seq2seq (NOT tested)
            pytest.param(
                "t5",
                "T5ForConditionalGeneration",
                "text2text-generation",
                id="t5",
            ),
            # detection (NOT tested)
            pytest.param(
                "detr",
                "DetrForObjectDetection",
                "object-detection",
                id="detr",
            ),
            # additional vision (NOT tested)
            pytest.param(
                "convnext",
                "ConvNextForImageClassification",
                "image-classification",
                id="convnext",
            ),
            pytest.param(
                "swin",
                "SwinForImageClassification",
                "image-classification",
                id="swin",
            ),
        ],
    )
    def test_auto_detect_new_architectures(
        self,
        model_type: str,
        arch_class_name: str,
        expected_task: str,
        make_mock_config,
    ) -> None:
        """Auto-detect task for architecture categories missing from existing tests."""
        config = make_mock_config(model_type, [arch_class_name])

        r = resolve_task(config)

        assert r.task == expected_task, (
            f"Expected task '{expected_task}' for {arch_class_name}, got '{r.task}'"
        )
        assert r.model_class is not None
        # model_class should be a real class (not None or MagicMock)
        assert hasattr(r.model_class, "__name__")


class TestResolveAmbiguousASRModelClass:
    """Resolve ASR's CTC and seq2seq families from architecture metadata."""

    @pytest.mark.parametrize("explicit_task", [False, True], ids=["auto-task", "explicit-task"])
    def test_ctc_architecture_uses_auto_model_for_ctc(
        self,
        explicit_task: bool,
        make_mock_config,
    ) -> None:
        config = make_mock_config("wav2vec2", ["Wav2Vec2ForCTC"])

        r = resolve_task(
            config,
            task="automatic-speech-recognition" if explicit_task else None,
        )

        assert r.task == "automatic-speech-recognition"
        assert r.model_class.__name__ == "AutoModelForCTC"

    def test_seq2seq_architecture_keeps_speech_seq2seq_default(self, make_mock_config) -> None:
        config = make_mock_config("whisper", ["WhisperForConditionalGeneration"])

        r = resolve_task(config)

        assert r.task == "automatic-speech-recognition"
        assert r.model_class.__name__ == "AutoModelForSpeechSeq2Seq"


class TestResolveTaskAliasPreservation:
    """User-task path: original task is returned, not the normalized form.

    resolve_task normalizes internally but MUST surface the original task so
    downstream consumers (dataset lookup, cache keys) see the user's original intent.
    """

    @pytest.mark.parametrize(
        ("original_task", "expected_returned_task"),
        [
            # Synonyms: user passes alias, gets alias BACK (not normalized)
            pytest.param(
                "image-feature-extraction",
                "image-feature-extraction",
                id="image-feature-extraction-alias",
            ),
            pytest.param(
                "masked-lm",
                "masked-lm",
                id="masked-lm-alias",
            ),
            # Canonical: stays unchanged
            pytest.param(
                "fill-mask",
                "fill-mask",
                id="fill-mask-canonical",
            ),
            pytest.param(
                "image-classification",
                "image-classification",
                id="image-classification-canonical",
            ),
        ],
    )
    def test_task_alias_preserved_in_return(
        self,
        original_task: str,
        expected_returned_task: str,
        make_mock_config,
    ) -> None:
        """User-task path: original task is returned, not the normalized form."""
        config = make_mock_config("bert", ["BertForMaskedLM"])

        r = resolve_task(config, task=original_task)

        assert r.task == expected_returned_task, (
            f"Expected returned task '{expected_returned_task}', got '{r.task}'. "
            f"resolve_task must surface the original task, not the normalized form."
        )


class TestResolveDoubleLookupOrder:
    """Verify original_task checked BEFORE normalized_task in specialization lookup.

    CLIP has two specializations:
      ("clip", "feature-extraction")       -> CLIPTextModelWithProjection
      ("clip", "image-feature-extraction") -> CLIPVisionModelWithProjection

    TasksManager normalizes "image-feature-extraction" -> "feature-extraction".
    Without double-lookup, both tasks would resolve to CLIPTextModelWithProjection.
    """

    def test_clip_image_feature_extraction_not_collapsed(
        self,
        make_mock_config,
    ) -> None:
        """'image-feature-extraction' must find CLIPVisionModelWithProjection.

        Must NOT collapse to 'feature-extraction' -> CLIPTextModelWithProjection.
        """
        config = make_mock_config("clip", ["CLIPModel"])

        r = resolve_task(config, task="image-feature-extraction")

        assert r.model_class.__name__ == "CLIPVisionModelWithProjection", (
            f"Expected CLIPVisionModelWithProjection, got {r.model_class.__name__}. "
            f"Double-lookup should check original_task 'image-feature-extraction' first."
        )
        assert r.task == "image-feature-extraction"

    def test_clip_feature_extraction_gives_text(
        self,
        make_mock_config,
    ) -> None:
        """'feature-extraction' gives CLIPTextModelWithProjection."""
        config = make_mock_config("clip", ["CLIPModel"])

        r = resolve_task(config, task="feature-extraction")

        assert r.model_class.__name__ == "CLIPTextModelWithProjection", (
            f"Expected CLIPTextModelWithProjection, got {r.model_class.__name__}"
        )
        assert r.task == "feature-extraction"


class TestResolveModelTypeNormalization:
    """Tests that model_type normalization (underscore -> hyphen, case) works.

    _get_custom_model_class normalizes model_type: lowercase + replace _ with -.
    This ensures 'sam2_video' finds the same specialization as 'sam2-video',
    and 'CLIP' finds the same as 'clip'.
    """

    @pytest.mark.parametrize(
        ("raw_model_type", "task", "arch_class_name", "expected_class_name"),
        [
            # sam2_video (underscore) should normalize to sam2-video and find specialization
            pytest.param(
                "sam2_video",
                "feature-extraction",
                "Sam2Model",
                "Sam2VisionEncoder",
                id="underscore-sam2-video",
            ),
            # CLIP (uppercase) should normalize to clip and find specialization
            pytest.param(
                "CLIP",
                "feature-extraction",
                "CLIPModel",
                "CLIPTextModelWithProjection",
                id="uppercase-clip",
            ),
        ],
    )
    def test_model_type_normalization(
        self,
        raw_model_type: str,
        task: str,
        arch_class_name: str,
        expected_class_name: str,
        make_mock_config,
    ) -> None:
        """model_type with underscores/uppercase is normalized before lookup."""
        config = make_mock_config(raw_model_type, [arch_class_name])

        r = resolve_task(config, task=task)

        assert r.model_class.__name__ == expected_class_name, (
            f"Expected {expected_class_name} for model_type='{raw_model_type}', "
            f"got {r.model_class.__name__}. model_type normalization may be broken."
        )
