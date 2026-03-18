"""Tests for ObjectDetectionDataset and processor utilities."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


_hf_hub_available = os.environ.get("WINML_TEST_OFFLINE", "0") != "1"


class TestGetImageProcessorConfig:
    """Tests for get_image_processor_config utility function."""

    @pytest.mark.skipif(not _hf_hub_available, reason="HF Hub disabled in offline mode")
    def test_loads_config_from_huggingface(self) -> None:
        """Should load preprocessor config from HuggingFace model."""
        from winml.modelkit.datasets.processor_utils import get_image_processor_config

        # Use a real model to test loading
        config = get_image_processor_config("facebook/detr-resnet-50")

        # DETR models have these standard config keys
        assert isinstance(config, dict)
        assert "image_processor_type" in config or "do_rescale" in config

    @pytest.mark.skipif(not _hf_hub_available, reason="HF Hub disabled in offline mode")
    def test_merges_kwargs_with_loaded_config(self) -> None:
        """kwargs should override loaded config values."""
        from winml.modelkit.datasets.processor_utils import get_image_processor_config

        config = get_image_processor_config(
            "facebook/detr-resnet-50",
            do_pad=False,
            custom_key="custom_value",
        )

        assert config["do_pad"] is False
        assert config["custom_key"] == "custom_value"

    @pytest.mark.skipif(not _hf_hub_available, reason="HF Hub disabled in offline mode")
    def test_kwargs_take_precedence(self) -> None:
        """kwargs should take precedence over loaded config."""
        from winml.modelkit.datasets.processor_utils import get_image_processor_config

        # Override a value that exists in the original config
        config = get_image_processor_config(
            "facebook/detr-resnet-50",
            do_rescale=False,  # Override the default True
        )

        assert config["do_rescale"] is False

    def test_handles_invalid_model_gracefully(self) -> None:
        """Should return empty dict with kwargs on invalid model."""
        from winml.modelkit.datasets.processor_utils import get_image_processor_config

        config = get_image_processor_config(
            "nonexistent/model-that-does-not-exist-xyz",
            fallback_key="fallback_value",
        )

        # Should still have kwargs even if loading failed
        assert config["fallback_key"] == "fallback_value"


class TestObjectDetectionDatasetDeriveOverrides:
    """Tests for ObjectDetectionDataset._derive_overrides method."""

    @pytest.fixture
    def dataset_class(self) -> type:
        """Get ObjectDetectionDataset class without instantiation."""
        from winml.modelkit.datasets.object_detection import ObjectDetectionDataset
        return ObjectDetectionDataset

    def test_no_io_config_returns_empty_overrides(
        self, dataset_class: type
    ) -> None:
        """Should return empty dict when io_config is None."""
        # Create instance without full initialization to test method
        instance = object.__new__(dataset_class)
        overrides = instance._derive_overrides(None)

        assert overrides == {}

    def test_sets_do_pad_false_when_no_pixel_mask(
        self, dataset_class: type
    ) -> None:
        """Should set do_pad=False when pixel_mask is not in io_config."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [1, 3, 640, 640]},
            # No pixel_mask
        }

        overrides = instance._derive_overrides(io_config)

        assert overrides["do_pad"] is False

    def test_does_not_set_do_pad_when_pixel_mask_present(
        self, dataset_class: type
    ) -> None:
        """Should not set do_pad when pixel_mask is in io_config."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [1, 3, 800, 800]},
            "pixel_mask": {"shape": [1, 800, 800]},
        }

        overrides = instance._derive_overrides(io_config)

        assert "do_pad" not in overrides

    def test_extracts_size_from_pixel_values_shape(
        self, dataset_class: type
    ) -> None:
        """Should extract height/width from pixel_values shape."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [1, 3, 800, 1200]},  # H=800, W=1200
        }

        overrides = instance._derive_overrides(io_config)

        assert overrides["size"] == {"height": 800, "width": 1200}

    def test_handles_dynamic_dimensions(
        self, dataset_class: type
    ) -> None:
        """Should not set size when dimensions are dynamic (None)."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [None, 3, None, None]},  # Dynamic
        }

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides

    def test_handles_partial_dynamic_dimensions(
        self, dataset_class: type
    ) -> None:
        """Should not set size when any dimension is dynamic."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [1, 3, 640, None]},  # Width is dynamic
        }

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides

    def test_handles_missing_shape_key(
        self, dataset_class: type
    ) -> None:
        """Should handle pixel_values without shape key."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {},  # No shape
        }

        overrides = instance._derive_overrides(io_config)

        # Should still set do_pad=False (no pixel_mask)
        assert overrides["do_pad"] is False
        assert "size" not in overrides

    def test_handles_short_shape_list(
        self, dataset_class: type
    ) -> None:
        """Should handle shape with fewer than 4 dimensions."""
        instance = object.__new__(dataset_class)
        io_config = {
            "pixel_values": {"shape": [640, 640]},  # Missing batch and channels
        }

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides


class TestObjectDetectionDatasetIntegration:
    """Integration tests for ObjectDetectionDataset with real models."""

    @pytest.fixture
    def mock_dataset(self) -> MagicMock:
        """Create a mock HuggingFace dataset."""
        from datasets.features import Image

        mock_ds = MagicMock()
        mock_ds.features = {
            "image": Image(),
            "objects": {"category": [1, 2], "bbox": [[0, 0, 10, 10]]},
        }
        mock_ds.__len__ = MagicMock(return_value=10)
        return mock_ds

    @pytest.mark.skipif(not _hf_hub_available, reason="HF Hub disabled in offline mode")
    def test_with_detr_model(self) -> None:
        """Integration test with real DETR model using mini-imagenet.

        For calibration, we only need images - labels don't matter.
        mini-imagenet is small and fast to download.
        """
        from winml.modelkit.datasets import ObjectDetectionDataset

        dataset = ObjectDetectionDataset(
            model_name="facebook/detr-resnet-50",
            dataset_name="timm/mini-imagenet",
            max_samples=2,
            data_split="train",
        )

        assert len(dataset) == 2
        sample = dataset[0]
        assert "pixel_values" in sample
        # Verify shape matches expected (batch=1, channels=3, height=640, width=640)
        assert sample["pixel_values"].shape[1] == 3  # channels

    def test_default_size_is_640(self) -> None:
        """Default image size should be 640 for object detection."""
        from winml.modelkit.datasets.object_detection import DEFAULT_OBJECT_DETECTION_SIZE

        assert DEFAULT_OBJECT_DETECTION_SIZE == 640

    def test_task_mapping_uses_object_detection_dataset(self) -> None:
        """TASK_DATASET_MAPPING should map object-detection to ObjectDetectionDataset."""
        from winml.modelkit.datasets import TASK_DATASET_MAPPING, ObjectDetectionDataset

        assert TASK_DATASET_MAPPING["object-detection"] is ObjectDetectionDataset

    def test_io_config_passed_through_kwargs(self) -> None:
        """io_config should be accessible via self._config."""
        from winml.modelkit.datasets.object_detection import ObjectDetectionDataset

        # Create instance without calling _initialize
        instance = object.__new__(ObjectDetectionDataset)
        instance._config = {"io_config": {"pixel_values": {"shape": [1, 3, 640, 640]}}}

        io_config = instance._config.get("io_config")

        assert io_config is not None
        assert io_config["pixel_values"]["shape"] == [1, 3, 640, 640]


class TestObjectDetectionDatasetWithMockedDeps:
    """Tests with mocked dependencies for faster execution."""

    @patch("winml.modelkit.datasets.object_detection.load_dataset")
    @patch("winml.modelkit.datasets.object_detection.AutoImageProcessor")
    def test_applies_do_pad_false_override(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Should apply do_pad=False when no pixel_mask in io_config."""
        from datasets.features import Image

        from winml.modelkit.datasets import ObjectDetectionDataset

        # Setup mocks
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        # Create mock dataset with proper structure
        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "objects": MagicMock()}
        mock_ds.__len__ = MagicMock(return_value=2)

        # Mock the map and with_format chain
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=2)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped

        mock_load_dataset.return_value = mock_ds

        # Create dataset with io_config that has no pixel_mask
        io_config = {"pixel_values": {"shape": [1, 3, 640, 640]}}

        ObjectDetectionDataset(
            model_name="facebook/detr-resnet-50",
            dataset_name="mock-dataset",
            max_samples=2,
            data_split="train",
            io_config=io_config,
        )

        # Verify processor was created with do_pad=False
        call_kwargs = mock_processor_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("do_pad") is False

    @patch("winml.modelkit.datasets.object_detection.load_dataset")
    @patch("winml.modelkit.datasets.object_detection.AutoImageProcessor")
    def test_uses_default_size_when_no_io_config(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Should use default 640 size when io_config has no shape."""
        from datasets.features import Image

        from winml.modelkit.datasets import ObjectDetectionDataset

        # Setup mocks
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "objects": MagicMock()}
        mock_ds.__len__ = MagicMock(return_value=2)

        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=2)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped

        mock_load_dataset.return_value = mock_ds

        # Create dataset without io_config
        ObjectDetectionDataset(
            model_name="facebook/detr-resnet-50",
            dataset_name="mock-dataset",
            max_samples=2,
            data_split="train",
        )

        # Verify processor was created with default size
        call_kwargs = mock_processor_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("size") == {"height": 640, "width": 640}


class TestObjectDetectionDatasetExports:
    """Tests for module exports and public API."""

    def test_object_detection_dataset_in_all(self) -> None:
        """ObjectDetectionDataset should be in __all__."""
        from winml.modelkit import datasets

        assert "ObjectDetectionDataset" in datasets.__all__

    def test_get_image_processor_config_in_all(self) -> None:
        """get_image_processor_config should be in __all__."""
        from winml.modelkit import datasets

        assert "get_image_processor_config" in datasets.__all__

    def test_can_import_from_package(self) -> None:
        """Should be importable from winml.modelkit.datasets."""
        from winml.modelkit.datasets import ObjectDetectionDataset, get_image_processor_config

        assert ObjectDetectionDataset is not None
        assert get_image_processor_config is not None
