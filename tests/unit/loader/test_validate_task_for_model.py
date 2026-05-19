# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for validate_task_for_model.

Verifies the shared task/architecture compatibility validator used at
command entry by ``winml build`` (and reusable from other commands).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader import validate_task_for_model


class TestValidateTaskForModel:
    """Tests for validate_task_for_model."""

    def test_compatible_task_passes(self) -> None:
        """A task in the supported list passes silently (no exception)."""
        mock_cfg = MagicMock()
        mock_cfg.model_type = "resnet"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
        ):
            # Should not raise
            validate_task_for_model("image-classification", "microsoft/resnet-50")

    def test_incompatible_task_raises_with_actionable_message(self) -> None:
        """A task not supported by the model architecture raises a one-line ValueError.

        The message must mention the offending task, the model id, the
        resolved architecture, and the supported-task list.
        """
        mock_cfg = MagicMock()
        mock_cfg.model_type = "resnet"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
            patch(
                "winml.modelkit.loader.task.normalize_task",
                side_effect=lambda t: t,
            ),
            pytest.raises(ValueError) as excinfo,
        ):
            validate_task_for_model("text-generation", "microsoft/resnet-50")

        msg = str(excinfo.value)
        assert "text-generation" in msg
        assert "microsoft/resnet-50" in msg
        assert "resnet" in msg
        assert "image-classification" in msg
        assert "image-feature-extraction" in msg

    def test_synonym_task_passes(self) -> None:
        """A task alias resolved by normalize_task to a supported name passes."""
        mock_cfg = MagicMock()
        mock_cfg.model_type = "gpt2"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["text-generation", "feature-extraction"],
            ),
            patch(
                "winml.modelkit.loader.task.normalize_task",
                return_value="text-generation",
            ),
        ):
            # "causal-lm" is an alias of "text-generation"
            validate_task_for_model("causal-lm", "gpt2")

    def test_empty_task_skips(self) -> None:
        """An empty/None task short-circuits without loading config."""
        with patch("transformers.AutoConfig.from_pretrained") as mock_load:
            validate_task_for_model("", "microsoft/resnet-50")
            validate_task_for_model(None, "microsoft/resnet-50")  # type: ignore[arg-type]
            mock_load.assert_not_called()

    def test_empty_model_id_skips(self) -> None:
        """An empty/None model_id short-circuits without loading config."""
        with patch("transformers.AutoConfig.from_pretrained") as mock_load:
            validate_task_for_model("text-generation", "")
            validate_task_for_model("text-generation", None)  # type: ignore[arg-type]
            mock_load.assert_not_called()

    def test_autoconfig_failure_skips_silently(self) -> None:
        """If AutoConfig.from_pretrained fails, validation degrades gracefully."""
        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=OSError("network error"),
        ):
            # Should not raise; the build pipeline will surface the real error later.
            validate_task_for_model("text-generation", "microsoft/resnet-50")

    def test_unknown_model_type_skips(self) -> None:
        """If model_type is unavailable, validation skips."""
        mock_cfg = MagicMock(spec=[])  # no model_type attr
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks"
            ) as mock_supported,
        ):
            validate_task_for_model("text-generation", "some/model")
            mock_supported.assert_not_called()

    def test_empty_supported_list_skips(self) -> None:
        """If TasksManager has no supported tasks for the architecture, skip."""
        mock_cfg = MagicMock()
        mock_cfg.model_type = "brand-new-arch"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=[],
            ),
        ):
            # Should not raise
            validate_task_for_model("text-generation", "some/model")
