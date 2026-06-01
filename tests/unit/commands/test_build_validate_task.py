# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for `_validate_task_supported_for_model` preflight in build CLI.

This helper used to live at `loader/config.py::validate_task_supported_for_model`
but was demoted to a private helper of the build command because it is the only
caller. Tests live in a dedicated module so they bypass the autouse fixture in
`test_build.py` that mocks the helper out for CLI-plumbing tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.commands.build import _validate_task_supported_for_model


class TestValidateTaskSupportedForModel:
    """Tests for `_validate_task_supported_for_model` preflight helper."""

    def test_raises_for_task_model_mismatch(self) -> None:
        """Incompatible task/model combinations raise a clear ValueError."""
        mock_config = MagicMock()
        mock_config.model_type = "resnet"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
            patch("winml.modelkit.loader.task.normalize_task", side_effect=lambda t: t),
            pytest.raises(
                ValueError,
                match=r"config\.loader\.task='text-generation' is not supported",
            ),
        ):
            _validate_task_supported_for_model(
                model_id="microsoft/resnet-50",
                task="text-generation",
                task_field_name="config.loader.task",
            )

    def test_accepts_supported_task(self) -> None:
        """A supported task should pass without raising."""
        mock_config = MagicMock()
        mock_config.model_type = "resnet"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
            patch("winml.modelkit.loader.task.normalize_task", side_effect=lambda t: t),
        ):
            _validate_task_supported_for_model(
                model_id="microsoft/resnet-50",
                task="image-classification",
            )

    def test_ensure_hf_models_registered_called_before_lookup(self) -> None:
        """ensure_hf_models_registered() is called to populate the ONNX registry
        before get_supported_tasks, so models like resnet return the correct tasks."""
        mock_config = MagicMock()
        mock_config.model_type = "resnet"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered") as mock_ensure,
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction", "image-classification"],
            ),
            patch("winml.modelkit.loader.task.normalize_task", side_effect=lambda t: t),
            pytest.raises(ValueError, match=r"text-generation.*is not supported"),
        ):
            _validate_task_supported_for_model(
                model_id="microsoft/resnet-50",
                task="text-generation",
                task_field_name="config.loader.task",
            )
        mock_ensure.assert_called_once()

    def test_defers_when_registry_still_empty_after_registration(self) -> None:
        """When get_supported_tasks returns [] even after registry population,
        validation defers to the downstream loader without raising."""
        mock_config = MagicMock()
        mock_config.model_type = "custom-model"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch("winml.modelkit.loader.task.get_supported_tasks", return_value=[]),
        ):
            # Should NOT raise — defer to downstream loader
            _validate_task_supported_for_model(
                model_id="org/custom-model",
                task="text-generation",
            )

    def test_error_message_format(self) -> None:
        """Error message has task/model/architecture on line 1, Supported tasks on line 2."""
        mock_config = MagicMock()
        mock_config.model_type = "resnet"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification"],
            ),
            patch("winml.modelkit.loader.task.normalize_task", side_effect=lambda t: t),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            pytest.raises(ValueError) as exc_info,
        ):
            _validate_task_supported_for_model(
                model_id="microsoft/resnet-50",
                task="text-generation",
                task_field_name="config.loader.task",
            )

        msg = str(exc_info.value)
        lines = msg.splitlines()
        assert len(lines) == 2
        assert lines[0].endswith("(architecture: resnet).")
        assert lines[1].startswith("Supported tasks:")

    def test_accepts_next_sentence_prediction_for_bert(self) -> None:
        """``next-sentence-prediction`` is in ``TASK_SYNONYM_EXTENSIONS`` and must
        be accepted, even though Optimum's per-arch supported_tasks does not list
        it. Regression for pre-PR behavior, see review claim 2.
        """
        mock_config = MagicMock()
        mock_config.model_type = "bert"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction", "fill-mask", "text-classification"],
            ),
        ):
            # Should NOT raise — short-circuited via TASK_SYNONYM_EXTENSIONS.
            _validate_task_supported_for_model(
                model_id="bert-base-uncased",
                task="next-sentence-prediction",
            )

    def test_accepts_mask_generation_via_synonym_extensions(self) -> None:
        """``mask-generation`` is preserved in ``TASK_SYNONYM_EXTENSIONS`` for SAM2.

        Optimum's ``map_from_synonym`` would normalize it to ``feature-extraction``,
        which is wrong for the HF-pipeline-keyed downstream registries.
        """
        mock_config = MagicMock()
        mock_config.model_type = "sam"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction"],
            ),
        ):
            _validate_task_supported_for_model(
                model_id="facebook/sam-vit-base",
                task="mask-generation",
            )

    def test_accepts_optimum_synonym_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Optimum-known synonyms (e.g. ``masked-lm`` -> ``fill-mask``) are accepted
        but logged as a warning so users converge on the canonical spelling.
        """
        mock_config = MagicMock()
        mock_config.model_type = "bert"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction", "fill-mask"],
            ),
            patch(
                "winml.modelkit.loader.task.normalize_task",
                side_effect=lambda t: {"masked-lm": "fill-mask"}.get(t, t),
            ),
            caplog.at_level("WARNING", logger="winml.modelkit.commands.build"),
        ):
            _validate_task_supported_for_model(
                model_id="bert-base-uncased",
                task="masked-lm",
            )

        assert any(
            "synonym" in rec.message and "fill-mask" in rec.message for rec in caplog.records
        ), f"Expected canonical-name hint, got: {[r.message for r in caplog.records]}"

    def test_silently_accepts_cross_modality_feature_extraction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Documented limitation: Optimum collapses ``image-feature-extraction``
        and ``feature-extraction``. A text-only arch with ``--task
        image-feature-extraction`` is therefore accepted (with a warning) at this
        gate; cross-modality routing errors must surface downstream where the
        HF-pipeline-keyed registries see the un-collapsed ``loader.task``.

        See review claim 1 — this test documents the limitation rather than
        asserting a fix.
        """
        mock_config = MagicMock()
        mock_config.model_type = "bert"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction", "fill-mask"],
            ),
            patch(
                "winml.modelkit.loader.task.normalize_task",
                side_effect=lambda t: "feature-extraction"
                if t in {"image-feature-extraction", "feature-extraction"}
                else t,
            ),
            caplog.at_level("WARNING", logger="winml.modelkit.commands.build"),
        ):
            _validate_task_supported_for_model(
                model_id="bert-base-uncased",
                task="image-feature-extraction",
                task_field_name="config.loader.task",
            )

        # Accepted, but the warning must fire so the limitation is at least visible.
        assert any("synonym" in rec.message for rec in caplog.records)

    def test_rejects_unrelated_task_after_all_fallbacks(self) -> None:
        """A task that is not verbatim-supported, not in ``TASK_SYNONYM_EXTENSIONS``,
        and whose Optimum-normalized form is not in the arch's supported set is
        still rejected. Ensures the new branches did not turn the gate into a
        no-op.
        """
        mock_config = MagicMock()
        mock_config.model_type = "resnet"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
            patch("winml.modelkit.loader.task.normalize_task", side_effect=lambda t: t),
            pytest.raises(ValueError, match=r"text-generation.*is not supported"),
        ):
            _validate_task_supported_for_model(
                model_id="microsoft/resnet-50",
                task="text-generation",
            )

    def test_verbatim_match_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """When the task is the exact canonical name in the supported set, no
        synonym-warning should fire (verbatim branch short-circuits before
        normalization).
        """
        mock_config = MagicMock()
        mock_config.model_type = "vit"

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_config),
            patch("winml.modelkit.export.io.ensure_hf_models_registered"),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["feature-extraction", "image-classification"],
            ),
            caplog.at_level("WARNING", logger="winml.modelkit.commands.build"),
        ):
            _validate_task_supported_for_model(
                model_id="google/vit-base-patch16-224",
                task="image-classification",
            )

        assert not any("synonym" in rec.message for rec in caplog.records)
