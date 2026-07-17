# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.modelkit.utils.hub_utils.

Focused coverage of :func:`get_pipeline_tag` — the lightweight Hub helper used
by the Stage 1d ``pipeline_tag`` fallback in loader task resolution. These tests
exercise the helper directly (local-path short-circuit, API failure, tag
extraction) rather than through a mocked stand-in, so the real ``_is_local_path``
guard and ``except`` fallthrough are covered.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from winml.modelkit.utils import get_pipeline_tag


_HF_API = "huggingface_hub.HfApi"


def test_get_pipeline_tag_local_path_returns_none_without_network() -> None:
    """A local path is rejected up front — the Hub API is never constructed."""
    with patch(_HF_API) as mock_api:
        assert get_pipeline_tag("./local-model") is None
    mock_api.assert_not_called()


def test_get_pipeline_tag_returns_tag_for_hub_model() -> None:
    """A reachable Hub model returns its pipeline_tag."""
    info = MagicMock(pipeline_tag="audio-classification")
    api = MagicMock()
    api.model_info.return_value = info
    with patch(_HF_API, return_value=api):
        assert get_pipeline_tag("audeering/wav2vec2-large-robust-24-ft-age-gender") == (
            "audio-classification"
        )
    api.model_info.assert_called_once()


def test_get_pipeline_tag_none_when_model_has_no_tag() -> None:
    """A model without a pipeline_tag yields None."""
    api = MagicMock()
    api.model_info.return_value = MagicMock(pipeline_tag=None)
    with patch(_HF_API, return_value=api):
        assert get_pipeline_tag("someone/some-model") is None


def test_get_pipeline_tag_swallows_api_error_and_returns_none() -> None:
    """A network/Hub error is caught — the helper returns None instead of raising."""
    api = MagicMock()
    api.model_info.side_effect = ConnectionError("offline")
    with patch(_HF_API, return_value=api):
        assert get_pipeline_tag("someone/some-model") is None
