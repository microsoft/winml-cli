# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for normalize_task and get_task_abbrev utility functions."""

from winml.modelkit.loader.task import get_task_abbrev, normalize_task


class TestNormalizeTask:
    """Tests for normalize_task function."""

    def test_causal_lm_normalizes_to_text_generation(self):
        """causal-lm is a known alias for text-generation."""
        assert normalize_task("causal-lm") == "text-generation"

    def test_seq2seq_lm_normalizes_to_text2text_generation(self):
        """seq2seq-lm is a known alias for text2text-generation."""
        assert normalize_task("seq2seq-lm") == "text2text-generation"

    def test_masked_lm_normalizes_to_fill_mask(self):
        """masked-lm is a known alias for fill-mask."""
        assert normalize_task("masked-lm") == "fill-mask"

    def test_canonical_task_unchanged(self):
        """Canonical task names are returned unchanged."""
        assert normalize_task("image-classification") == "image-classification"
        assert normalize_task("text-generation") == "text-generation"
        assert normalize_task("fill-mask") == "fill-mask"

    def test_unknown_task_passthrough(self):
        """Unknown task names are returned unchanged (passthrough)."""
        assert normalize_task("my-custom-task") == "my-custom-task"
        assert normalize_task("nonexistent-task") == "nonexistent-task"


class TestGetTaskAbbrev:
    """Tests for get_task_abbrev function."""

    def test_known_vision_task(self):
        """Known vision tasks return correct abbreviations."""
        assert get_task_abbrev("image-classification") == "imgcls"
        assert get_task_abbrev("object-detection") == "objdet"
        assert get_task_abbrev("depth-estimation") == "depth"

    def test_known_nlp_task(self):
        """Known NLP tasks return correct abbreviations."""
        assert get_task_abbrev("text-classification") == "txtcls"
        assert get_task_abbrev("fill-mask") == "mask"
        assert get_task_abbrev("question-answering") == "qa"
        assert get_task_abbrev("text-generation") == "txtgen"

    def test_known_audio_task(self):
        """Known audio tasks return correct abbreviations."""
        assert get_task_abbrev("automatic-speech-recognition") == "asr"
        assert get_task_abbrev("audio-classification") == "audiocls"

    def test_unknown_task_truncated_to_8_chars(self):
        """Unknown tasks are truncated to first 8 characters."""
        assert get_task_abbrev("my-custom-task") == "my-custo"
        assert get_task_abbrev("abcdefghijklmnop") == "abcdefgh"

    def test_short_unknown_task(self):
        """Short unknown task names are returned as-is (< 8 chars)."""
        assert get_task_abbrev("short") == "short"

    def test_feature_extraction(self):
        """feature-extraction returns 'feat'."""
        assert get_task_abbrev("feature-extraction") == "feat"

    def test_image_feature_extraction(self):
        """image-feature-extraction returns 'imgfeat'."""
        assert get_task_abbrev("image-feature-extraction") == "imgfeat"

    def test_next_sentence_prediction(self):
        """next-sentence-prediction returns 'nsp'."""
        assert get_task_abbrev("next-sentence-prediction") == "nsp"
