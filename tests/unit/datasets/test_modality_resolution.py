# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modality-aware task resolution via canonical_task_to_known_task.

Optimum collapses several modality-distinct HF pipeline tasks under one
canonical name (e.g. ``image-feature-extraction`` -> ``feature-extraction``
for vision FE models like DINOv2). WinML registries (datasets, evaluators)
are keyed by HF-pipeline names, so the helper undoes the collapse by
inspecting the model's static OnnxConfig inputs.

These tests exercise the helper directly (with mocked HF/Optimum bridges)
and the wiring through DatasetCalibrationReader.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_disambiguation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str = "dinov2",
    onnx_inputs: list[str],
) -> dict[str, list]:
    """Patch AutoConfig and _get_onnx_config so the helper runs offline.

    Returns a dict that records call args for assertions.
    """
    calls: dict[str, list] = {"autoconfig": [], "get_onnx_config": []}

    def fake_from_pretrained(model_id, *args, **kwargs):
        calls["autoconfig"].append(model_id)
        cfg = MagicMock()
        cfg.model_type = model_type
        return cfg

    def fake_get_onnx_config(mt, task, hf_config):
        calls["get_onnx_config"].append((mt, task))
        onnx_cfg = MagicMock()
        onnx_cfg.inputs = {name: object() for name in onnx_inputs}
        return onnx_cfg

    import transformers

    import winml.modelkit.export.io as export_io

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(export_io, "_get_onnx_config", fake_get_onnx_config)
    return calls


# ---------------------------------------------------------------------------
# canonical_task_to_known_task
# ---------------------------------------------------------------------------


class TestCanonicalTaskToKnownTask:
    """Direct tests for the loader helper."""

    def test_unambiguous_task_passthrough_without_model_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tasks not in the ambiguity dict must not trigger any AutoConfig load."""
        from winml.modelkit.loader import canonical_task_to_known_task

        # If AutoConfig.from_pretrained is called, the test fails -- short-circuit
        # must happen before any HF/Optimum work.
        import transformers

        def explode(*args, **kwargs):  # pragma: no cover - exercised on regression
            raise AssertionError("AutoConfig.from_pretrained should not be called")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", explode)

        assert (
            canonical_task_to_known_task("image-classification", "facebook/dinov2-base")
            == "image-classification"
        )

    def test_missing_model_id_passthrough(self) -> None:
        """No model_id -> passthrough (caller decides how to surface the error)."""
        from winml.modelkit.loader import canonical_task_to_known_task

        assert canonical_task_to_known_task("feature-extraction", None) == "feature-extraction"

    def test_feature_extraction_with_pixel_values_resolves_to_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vision FE model -> image-feature-extraction."""
        from winml.modelkit.loader import canonical_task_to_known_task

        calls = _patch_disambiguation(
            monkeypatch, model_type="dinov2", onnx_inputs=["pixel_values"]
        )

        assert (
            canonical_task_to_known_task("feature-extraction", "facebook/dinov2-base")
            == "image-feature-extraction"
        )
        assert calls["autoconfig"] == ["facebook/dinov2-base"]
        assert calls["get_onnx_config"] == [("dinov2", "feature-extraction")]

    def test_feature_extraction_with_input_ids_resolves_to_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text FE model -> stays feature-extraction."""
        from winml.modelkit.loader import canonical_task_to_known_task

        _patch_disambiguation(
            monkeypatch,
            model_type="bert",
            onnx_inputs=["input_ids", "attention_mask"],
        )

        assert (
            canonical_task_to_known_task("feature-extraction", "bert-base-uncased")
            == "feature-extraction"
        )

    def test_unknown_inputs_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OnnxConfig with no recognised inputs -> passthrough to canonical."""
        from winml.modelkit.loader import canonical_task_to_known_task

        _patch_disambiguation(monkeypatch, model_type="custom", onnx_inputs=["mystery"])

        assert (
            canonical_task_to_known_task("feature-extraction", "some/model")
            == "feature-extraction"
        )

    def test_autoconfig_failure_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AutoConfig.from_pretrained failure -> passthrough, no raise."""
        from winml.modelkit.loader import canonical_task_to_known_task

        import transformers

        def boom(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)

        assert (
            canonical_task_to_known_task("feature-extraction", "facebook/dinov2-base")
            == "feature-extraction"
        )

    def test_get_onnx_config_failure_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OnnxConfig lookup failure -> passthrough, no raise."""
        from winml.modelkit.loader import canonical_task_to_known_task

        import transformers

        import winml.modelkit.export.io as export_io

        def fake_from_pretrained(*args, **kwargs):
            cfg = MagicMock()
            cfg.model_type = "weird"
            return cfg

        def fake_get_onnx_config(*args, **kwargs):
            raise KeyError("no OnnxConfig registered")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_from_pretrained)
        monkeypatch.setattr(export_io, "_get_onnx_config", fake_get_onnx_config)

        assert (
            canonical_task_to_known_task("feature-extraction", "some/model")
            == "feature-extraction"
        )


# ---------------------------------------------------------------------------
# DatasetCalibrationReader wiring
# ---------------------------------------------------------------------------


class TestDatasetCalibrationReaderModalityWiring:
    """Verify the calibration reader translates the task before dataset lookup."""

    def test_dinov2_feature_extraction_routes_to_image_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The original DINOv2 bug scenario: feature-extraction + vision model
        must be translated to image-feature-extraction so ImageDataset is picked
        (not the text TextDataset that calls AutoTokenizer).
        """
        _patch_disambiguation(
            monkeypatch, model_type="dinov2", onnx_inputs=["pixel_values"]
        )

        # Stub the dataset factory so we observe the resolved task without
        # downloading any real dataset or instantiating a real preprocessor.
        captured: dict[str, object] = {}

        def fake_universal_calib_dataset(*, model_name, task, **kwargs):
            captured["task"] = task
            captured["model_name"] = model_name
            stub = MagicMock()
            stub.__len__ = MagicMock(return_value=0)
            return stub

        import winml.modelkit.datasets as datasets_pkg

        monkeypatch.setattr(
            datasets_pkg, "universal_calib_dataset", fake_universal_calib_dataset
        )

        reader = datasets_pkg.DatasetCalibrationReader(
            model_name="facebook/dinov2-base",
            task="feature-extraction",
            max_samples=2,
        )

        assert captured["task"] == "image-feature-extraction"
        assert reader.task == "image-feature-extraction"

    def test_text_feature_extraction_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Text FE model keeps feature-extraction (matches TextDataset)."""
        _patch_disambiguation(
            monkeypatch,
            model_type="bert",
            onnx_inputs=["input_ids", "attention_mask"],
        )

        captured: dict[str, object] = {}

        def fake_universal_calib_dataset(*, model_name, task, **kwargs):
            captured["task"] = task
            stub = MagicMock()
            stub.__len__ = MagicMock(return_value=0)
            return stub

        import winml.modelkit.datasets as datasets_pkg

        monkeypatch.setattr(
            datasets_pkg, "universal_calib_dataset", fake_universal_calib_dataset
        )

        reader = datasets_pkg.DatasetCalibrationReader(
            model_name="bert-base-uncased",
            task="feature-extraction",
            max_samples=2,
        )

        assert captured["task"] == "feature-extraction"
        assert reader.task == "feature-extraction"

    def test_unambiguous_task_not_perturbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-ambiguous tasks (e.g. image-classification) must skip the
        AutoConfig load entirely -- zero extra cost."""
        import transformers

        def explode(*args, **kwargs):  # pragma: no cover - exercised on regression
            raise AssertionError("AutoConfig.from_pretrained should not be called")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", explode)

        captured: dict[str, object] = {}

        def fake_universal_calib_dataset(*, model_name, task, **kwargs):
            captured["task"] = task
            stub = MagicMock()
            stub.__len__ = MagicMock(return_value=0)
            return stub

        import winml.modelkit.datasets as datasets_pkg

        monkeypatch.setattr(
            datasets_pkg, "universal_calib_dataset", fake_universal_calib_dataset
        )

        reader = datasets_pkg.DatasetCalibrationReader(
            model_name="microsoft/resnet-50",
            task="image-classification",
            max_samples=2,
        )

        assert captured["task"] == "image-classification"
        assert reader.task == "image-classification"
