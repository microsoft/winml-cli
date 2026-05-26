# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for DepthEstimationDataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDepthEstimationDatasetDeriveOverrides:
    """Tests for DepthEstimationDataset._derive_overrides method."""

    @pytest.fixture
    def dataset_class(self) -> type:
        """Get DepthEstimationDataset class without instantiation."""
        from winml.modelkit.datasets import DepthEstimationDataset

        return DepthEstimationDataset

    def test_no_io_config_returns_static_overrides(self, dataset_class: type) -> None:
        """Even without io_config, static overrides are set."""
        instance = object.__new__(dataset_class)
        overrides = instance._derive_overrides(None)

        assert overrides == {"keep_aspect_ratio": False, "do_pad": False}

    def test_always_disables_keep_aspect_ratio(self, dataset_class: type) -> None:
        """keep_aspect_ratio=False is always set (Depth-Anything default is True)."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {"shape": [1, 3, 518, 518]}}

        overrides = instance._derive_overrides(io_config)

        assert overrides["keep_aspect_ratio"] is False

    def test_always_disables_do_pad(self, dataset_class: type) -> None:
        """do_pad=False is always set (depth processors may pad otherwise)."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {"shape": [1, 3, 518, 518]}}

        overrides = instance._derive_overrides(io_config)

        assert overrides["do_pad"] is False

    def test_extracts_size_from_pixel_values_shape(self, dataset_class: type) -> None:
        """Should extract height/width from pixel_values shape."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {"shape": [1, 3, 518, 518]}}

        overrides = instance._derive_overrides(io_config)

        assert overrides["size"] == {"height": 518, "width": 518}

    def test_handles_dynamic_dimensions(self, dataset_class: type) -> None:
        """Should not set size when dimensions are dynamic (None)."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {"shape": [None, 3, None, None]}}

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides
        # Static overrides still present
        assert overrides["keep_aspect_ratio"] is False
        assert overrides["do_pad"] is False

    def test_handles_missing_shape_key(self, dataset_class: type) -> None:
        """Should handle pixel_values without shape key."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {}}

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides
        assert overrides["keep_aspect_ratio"] is False
        assert overrides["do_pad"] is False

    def test_handles_short_shape_list(self, dataset_class: type) -> None:
        """Should handle shape with fewer than 4 dimensions."""
        instance = object.__new__(dataset_class)
        io_config = {"pixel_values": {"shape": [518, 518]}}

        overrides = instance._derive_overrides(io_config)

        assert "size" not in overrides


class TestDepthEstimationDatasetWithMockedDeps:
    """Tests with mocked dependencies for faster execution."""

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_processor_created_with_static_shape(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Processor receives size matching pixel_values shape + static overrides."""
        from datasets.features import Image

        from winml.modelkit.datasets import DepthEstimationDataset

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "depth_map": Image()}
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=2)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        io_config = {"pixel_values": {"shape": [1, 3, 518, 518]}}

        DepthEstimationDataset(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            dataset_name="mock-dataset",
            max_samples=2,
            data_split="validation",
            io_config=io_config,
        )

        call_kwargs = mock_processor_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("size") == {"height": 518, "width": 518}
        assert call_kwargs.get("keep_aspect_ratio") is False
        assert call_kwargs.get("do_pad") is False

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_uses_default_size_when_no_io_config(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Falls back to DEFAULT_DEPTH_ESTIMATION_SIZE when io_config is absent."""
        from datasets.features import Image

        from winml.modelkit.datasets import (
            DEFAULT_DEPTH_ESTIMATION_SIZE,
            DepthEstimationDataset,
        )

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "depth_map": Image()}
        mock_ds.__len__ = MagicMock(return_value=2)
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=2)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        DepthEstimationDataset(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            dataset_name="mock-dataset",
            max_samples=2,
            data_split="validation",
        )

        call_kwargs = mock_processor_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("size") == {
            "height": DEFAULT_DEPTH_ESTIMATION_SIZE,
            "width": DEFAULT_DEPTH_ESTIMATION_SIZE,
        }


class TestDepthEstimationDatasetColumnDetection:
    """Tests for column detection without ClassLabel."""

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_detects_image_and_depth_columns(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Detects Image column for input and depth_map column for ground truth."""
        from datasets.features import Image

        from winml.modelkit.datasets import DepthEstimationDataset

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "depth_map": Image()}
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=1)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        ds = DepthEstimationDataset(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            dataset_name="mock-dataset",
            max_samples=1,
            data_split="validation",
        )

        assert ds._image_col == "image"
        assert ds.label_col == "depth_map"

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_raises_when_no_image_column(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Raises ValueError when the dataset has no Image column."""
        from winml.modelkit.datasets import DepthEstimationDataset

        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"text": MagicMock()}
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        with pytest.raises(ValueError, match="No Image column"):
            DepthEstimationDataset(
                model_name="depth-anything/Depth-Anything-V2-Small-hf",
                dataset_name="mock-dataset",
                max_samples=1,
                data_split="validation",
            )


class TestDepthEstimationDatasetCalibrationDefaults:
    """Tests for the calibration path where dataset_name is not provided.

    Quantization's universal_calib_dataset constructs DepthEstimationDataset
    without dataset_name. The class must therefore apply task-specific
    defaults (NYU + parquet revision) inside _initialize().
    """

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_uses_nyu_with_parquet_revision_when_no_dataset_name(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """When dataset_name is None, load_dataset is called with NYU + parquet revision."""
        from datasets.features import Image

        from winml.modelkit.datasets import DepthEstimationDataset
        from winml.modelkit.datasets.depth_estimation import (
            DEFAULT_DEPTH_ESTIMATION_DATASET,
            DEFAULT_DEPTH_ESTIMATION_REVISION,
            DEFAULT_DEPTH_ESTIMATION_SPLIT,
        )

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "depth_map": Image()}
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=1)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        DepthEstimationDataset(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            max_samples=1,
        )

        # load_dataset must be called with NYU + parquet revision
        args, kwargs = mock_load_dataset.call_args
        assert args[0] == DEFAULT_DEPTH_ESTIMATION_DATASET
        assert kwargs.get("split") == DEFAULT_DEPTH_ESTIMATION_SPLIT
        assert kwargs.get("revision") == DEFAULT_DEPTH_ESTIMATION_REVISION

    @patch("winml.modelkit.datasets.image.load_dataset")
    @patch("winml.modelkit.datasets.depth_estimation.AutoImageProcessor")
    def test_no_revision_when_user_specifies_dataset(
        self,
        mock_processor_cls: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """When the user explicitly specifies a dataset, no revision is forced."""
        from datasets.features import Image

        from winml.modelkit.datasets import DepthEstimationDataset

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor_cls.from_pretrained.return_value = mock_processor

        mock_ds = MagicMock()
        mock_ds.features = {"image": Image(), "depth_map": Image()}
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_mapped = MagicMock()
        mock_mapped.__len__ = MagicMock(return_value=1)
        mock_mapped.with_format.return_value = mock_mapped
        mock_ds.map.return_value = mock_mapped
        mock_ds.select.return_value = mock_ds
        mock_load_dataset.return_value = mock_ds

        DepthEstimationDataset(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            dataset_name="custom/depth-dataset",
            data_split="validation",
            max_samples=1,
        )

        args, kwargs = mock_load_dataset.call_args
        assert args[0] == "custom/depth-dataset"
        assert kwargs.get("revision") is None


class TestDepthEstimationDatasetExports:
    """Tests for module exports and public API."""

    def test_depth_estimation_dataset_in_all(self) -> None:
        """DepthEstimationDataset should be in __all__."""
        from winml.modelkit import datasets

        assert "DepthEstimationDataset" in datasets.__all__

    def test_default_size_constant_in_all(self) -> None:
        """DEFAULT_DEPTH_ESTIMATION_SIZE should be in __all__."""
        from winml.modelkit import datasets

        assert "DEFAULT_DEPTH_ESTIMATION_SIZE" in datasets.__all__

    def test_task_mapping_uses_depth_estimation_dataset(self) -> None:
        """TASK_DATASET_MAPPING should map depth-estimation to DepthEstimationDataset."""
        from winml.modelkit.datasets import TASK_DATASET_MAPPING, DepthEstimationDataset

        assert TASK_DATASET_MAPPING["depth-estimation"] is DepthEstimationDataset
