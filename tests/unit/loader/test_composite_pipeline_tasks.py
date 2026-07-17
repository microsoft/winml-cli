# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``composite_pipeline_tasks`` — registry-driven, offline."""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader import composite_pipeline_tasks
from winml.modelkit.loader.resolution import resolve_composite_load_task


def test_bart_serves_summarization_and_table_qa_sorted():
    # Sorted -> deterministic & model-id-independent: TAPEX and a plain bart
    # summarizer are config-identical, so the order must not imply which pipeline
    # a given checkpoint is.
    assert composite_pipeline_tasks("bart") == ["summarization", "table-question-answering"]


def test_marian_serves_translation():
    assert composite_pipeline_tasks("marian") == ["translation"]


def test_qwen3_serves_text_generation():
    assert composite_pipeline_tasks("qwen3") == ["text-generation"]


def test_non_composite_model_types_return_empty():
    assert composite_pipeline_tasks("bert") == []
    assert composite_pipeline_tasks("resnet") == []


def test_registry_accessor_raises_loudly_when_empty(monkeypatch):
    # The registry is populated as an import side effect; if registrations ever
    # move/rename and it comes up empty, the shared accessor must fail loudly rather
    # than let every reader silently return []/None (composites disabled unnoticed).
    import winml.modelkit.models.hf  # noqa: F401 — ensure real registrations land first

    monkeypatch.setattr("winml.modelkit.models.winml.composite_model.COMPOSITE_MODEL_REGISTRY", {})
    from winml.modelkit.loader.resolution import _composite_registry

    with pytest.raises(RuntimeError, match="COMPOSITE_MODEL_REGISTRY is empty"):
        _composite_registry()


class TestResolveCompositeLoadTask:
    """``resolve_composite_load_task`` bridges detection to a loadable pipeline task.

    The fan-out commands use ``resolve_composite_components`` (which sub-models),
    but the model loaders need a concrete registry task to instantiate the
    pipeline. This helper maps a detected composite back to its sorted-first
    pipeline task so a bare ``winml perf -m <composite>`` builds the whole
    pipeline. Config loading is mocked to keep the test offline.
    """

    def test_none_model_returns_none_without_loading_config(self) -> None:
        # No model id -> nothing to resolve, and no config round-trip attempted.
        with patch("transformers.AutoConfig.from_pretrained") as mock_cfg:
            assert resolve_composite_load_task(None) is None
        mock_cfg.assert_not_called()

    def test_composite_model_maps_to_sorted_first_pipeline_task(self, make_mock_config) -> None:
        # A seq2seq composite (T5) resolves to its deterministic sorted-first
        # pipeline task -- sub-model-equivalent to the others it registers.
        config = make_mock_config("t5", ["T5ForConditionalGeneration"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=config):
            result = resolve_composite_load_task("some/t5-checkpoint")

        tasks = composite_pipeline_tasks("t5")
        assert tasks, "t5 should register composite pipeline tasks"
        assert result == tasks[0]

    def test_non_composite_model_returns_none(self, make_mock_config) -> None:
        # A plain classifier is not a composite -> no pipeline task to load.
        config = make_mock_config("resnet", ["ResNetForImageClassification"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=config):
            assert resolve_composite_load_task("some/resnet-checkpoint") is None

    def test_model_type_is_normalized_before_registry_lookup(self, make_mock_config) -> None:
        # Registry keys are lower/hyphenated; a raw model_type with underscores or
        # mixed case must still map to its pipeline task. resolve_task is stubbed to
        # report "composite" so this isolates the normalization step (without it,
        # composite_pipeline_tasks("Vision_Encoder_Decoder") is [] -> None).
        registered = "vision-encoder-decoder"
        expected = composite_pipeline_tasks(registered)
        assert expected, f"{registered!r} should be a registered composite"

        config = make_mock_config("Vision_Encoder_Decoder", ["VisionEncoderDecoderModel"])
        detected = MagicMock()
        detected.composite = {"encoder": "feature-extraction", "decoder": "image-to-text"}
        with (
            patch("winml.modelkit.loader.resolution.resolve_task", return_value=detected),
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
        ):
            result = resolve_composite_load_task("some/vision-encoder-decoder-checkpoint")

        assert result == expected[0]
