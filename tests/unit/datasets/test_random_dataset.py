# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for RandomDataset and fallback behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def simple_onnx_model(tmp_path: Path) -> Path:
    """Create a simple ONNX model for testing RandomDataset.

    Graph: A @ B = C
    Where A is input (1, 4), B is constant (4, 4), C is output (1, 4)
    """
    # Input
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Output
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Constant weights
    B_values = np.random.randn(4, 4).astype(np.float32)  # noqa: N806
    B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], B_values.flatten())  # noqa: N806

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        [matmul_node],
        "test_matmul",
        [A],
        [C],
        [B_tensor],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Save
    output_path = tmp_path / "test_model.onnx"
    onnx.save(model, str(output_path))

    return output_path


class TestRandomDataset:
    """Tests for RandomDataset class."""

    def test_random_dataset_with_model_path(self, simple_onnx_model: Path) -> None:
        """RandomDataset should work when model_path is provided."""
        from winml.modelkit.datasets import RandomDataset

        dataset = RandomDataset(
            model_path=str(simple_onnx_model),
            max_samples=10,
            seed=42,
        )

        assert len(dataset) == 10
        sample = dataset[0]
        assert "A" in sample  # Input name from ONNX model
        assert sample["A"].shape == (1, 4)

    def test_random_dataset_generates_correct_dtype(self, simple_onnx_model: Path) -> None:
        """RandomDataset should generate data with correct dtype."""
        import torch

        from winml.modelkit.datasets import RandomDataset

        dataset = RandomDataset(
            model_path=str(simple_onnx_model),
            max_samples=5,
        )

        sample = dataset[0]
        # ONNX model has FLOAT input, should be float32
        assert sample["A"].dtype in (np.float32, torch.float32)

    def test_random_dataset_reproducible_with_seed(self, simple_onnx_model: Path) -> None:
        """RandomDataset should produce same data with same seed."""
        from winml.modelkit.datasets import RandomDataset

        dataset1 = RandomDataset(
            model_path=str(simple_onnx_model),
            max_samples=5,
            seed=123,
        )
        dataset2 = RandomDataset(
            model_path=str(simple_onnx_model),
            max_samples=5,
            seed=123,
        )

        # Same seed should produce same data
        sample1 = dataset1[0]["A"]
        sample2 = dataset2[0]["A"]

        # Convert to numpy if tensor
        if hasattr(sample1, "numpy"):
            sample1 = sample1.numpy()
        if hasattr(sample2, "numpy"):
            sample2 = sample2.numpy()

        np.testing.assert_array_equal(sample1, sample2)


class TestUniversalCalibDatasetFallback:
    """Tests for universal_calib_dataset fallback to random."""

    def test_unknown_task_falls_back_to_random(self, simple_onnx_model: Path) -> None:
        """Unknown task should fallback to RandomDataset."""
        from winml.modelkit.datasets import RandomDataset, universal_calib_dataset

        dataset = universal_calib_dataset(
            model_name="test-model",
            task="unknown-task-xyz",  # Not in TASK_DATASET_MAPPING
            model_path=str(simple_onnx_model),
            max_samples=5,
        )

        assert isinstance(dataset, RandomDataset)
        assert len(dataset) == 5

    def test_explicit_random_task_works(self, simple_onnx_model: Path) -> None:
        """Explicitly requesting 'random' task should work."""
        from winml.modelkit.datasets import RandomDataset, universal_calib_dataset

        dataset = universal_calib_dataset(
            model_name="any-model",
            task="random",
            model_path=str(simple_onnx_model),
            max_samples=10,
        )

        assert isinstance(dataset, RandomDataset)
        assert len(dataset) == 10


class TestTaskDatasetMapping:
    """Verify all supported tasks map to correct dataset classes."""

    def test_all_tasks_have_mappings(self) -> None:
        """Every task in TASK_DATASET_MAPPING maps to a callable dataset class."""
        from winml.modelkit.datasets import TASK_DATASET_MAPPING

        for task, cls in TASK_DATASET_MAPPING.items():
            assert callable(cls), f"Task {task!r} maps to non-callable {cls}"

    @pytest.mark.parametrize(
        ("task", "module_path", "class_name"),
        [
            ("image-classification", "winml.modelkit.datasets.image", "ImageDataset"),
            ("image-feature-extraction", "winml.modelkit.datasets.image", "ImageDataset"),
            ("text-classification", "winml.modelkit.datasets.text", "TextDataset"),
            ("text-feature-extraction", "winml.modelkit.datasets.text", "TextDataset"),
            ("next-sentence-prediction", "winml.modelkit.datasets.text", "TextDataset"),
            ("fill-mask", "winml.modelkit.datasets.text", "TextDataset"),
            (
                "object-detection",
                "winml.modelkit.datasets.object_detection",
                "ObjectDetectionDataset",
            ),
            (
                "image-segmentation",
                "winml.modelkit.datasets.image_segmentation",
                "ImageSegmentationDataset",
            ),
            ("random", "winml.modelkit.datasets.random_dataset", "RandomDataset"),
        ],
    )
    def test_task_maps_to_dataset_class(self, task: str, module_path: str, class_name: str) -> None:
        """Each task maps to its expected dataset class."""
        import importlib

        from winml.modelkit.datasets import TASK_DATASET_MAPPING

        module = importlib.import_module(module_path)
        expected_cls = getattr(module, class_name)
        assert TASK_DATASET_MAPPING[task] is expected_cls


class TestDatasetCalibrationReaderFallback:
    """Tests for DatasetCalibrationReader with unsupported tasks."""

    def test_calibration_reader_with_unknown_task(self, simple_onnx_model: Path) -> None:
        """DatasetCalibrationReader should handle unknown tasks via fallback."""
        from winml.modelkit.datasets import DatasetCalibrationReader

        reader = DatasetCalibrationReader(
            model_name="test-model",
            task="pretraining",  # Unknown task
            max_samples=5,
            model_path=str(simple_onnx_model),  # Passed through for RandomDataset
        )

        assert len(reader) == 5

        # Should be able to get samples
        sample = reader.get_next()
        assert sample is not None
        assert "A" in sample  # Input from ONNX model
        assert isinstance(sample["A"], np.ndarray)

    def test_calibration_reader_get_next_exhaustion(self, simple_onnx_model: Path) -> None:
        """DatasetCalibrationReader should return None when exhausted."""
        from winml.modelkit.datasets import DatasetCalibrationReader

        reader = DatasetCalibrationReader(
            model_name="test-model",
            task="random",
            max_samples=3,
            model_path=str(simple_onnx_model),
        )

        # Get all samples
        samples = []
        while True:
            sample = reader.get_next()
            if sample is None:
                break
            samples.append(sample)

        assert len(samples) == 3

        # Should return None now
        assert reader.get_next() is None

    def test_calibration_reader_rewind(self, simple_onnx_model: Path) -> None:
        """DatasetCalibrationReader should support rewind."""
        from winml.modelkit.datasets import DatasetCalibrationReader

        reader = DatasetCalibrationReader(
            model_name="test-model",
            task="random",
            max_samples=3,
            model_path=str(simple_onnx_model),
        )

        # Exhaust
        for _ in range(3):
            reader.get_next()

        assert reader.get_next() is None

        # Rewind and try again
        reader.rewind()
        sample = reader.get_next()
        assert sample is not None


# =============================================================================
# Class 5: RandomDataset auto-reading value ranges from ONNX metadata
# =============================================================================


def _create_onnx_with_metadata(
    tmp_path: Path,
    name: str,
    value_ranges: dict[str, list] | None = None,
) -> Path:
    """Create a BERT-like ONNX model, optionally with winml.io.inputs metadata."""
    import json

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 16])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 16])
    output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])

    cast_node = helper.make_node("Cast", ["input_ids"], ["cast_out"], to=1)
    reduce_node = helper.make_node("ReduceMean", ["cast_out"], ["logits"], axes=[1], keepdims=0)

    graph = helper.make_graph(
        [cast_node, reduce_node],
        "bert_like",
        [input_ids, attention_mask],
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    if value_ranges:
        io_inputs = []
        for input_name, vr in value_ranges.items():
            io_inputs.append(
                {"name": input_name, "dtype": "int32", "shape": [1, 16], "value_range": vr}
            )
        model.metadata_props.add(key="winml.io.inputs", value=json.dumps(io_inputs))

    model_path = tmp_path / name
    onnx.save(model, str(model_path))
    return model_path


class TestRandomDatasetValueRanges:
    """Test RandomDataset auto-reading value ranges from ONNX metadata."""

    def test_value_ranges_from_metadata(self, tmp_path: Path) -> None:
        """RandomDataset auto-reads winml.io.inputs and uses value ranges."""
        from winml.modelkit.datasets.random_dataset import RandomDataset

        model_path = _create_onnx_with_metadata(
            tmp_path,
            "bert_with_meta.onnx",
            value_ranges={"input_ids": [0, 100], "attention_mask": [0, 2]},
        )
        dataset = RandomDataset(str(model_path), max_samples=5)

        sample = dataset[0]
        assert sample["input_ids"].min().item() >= 0
        assert sample["input_ids"].max().item() < 100
        assert sample["attention_mask"].min().item() >= 0
        assert sample["attention_mask"].max().item() < 2

    def test_fallback_without_metadata(self, tmp_path: Path) -> None:
        """RandomDataset falls back to dtype-based ranges without metadata."""
        from winml.modelkit.datasets.random_dataset import RandomDataset

        model_path = _create_onnx_with_metadata(
            tmp_path,
            "bert_no_meta.onnx",
        )
        dataset = RandomDataset(str(model_path), max_samples=5)

        sample = dataset[0]
        # Default int range is [-1000, 1000)
        assert sample["input_ids"].min().item() >= -1000
        assert sample["input_ids"].max().item() < 1000

    def test_partial_metadata(self, tmp_path: Path) -> None:
        """Inputs without value_range in metadata fall back."""
        from winml.modelkit.datasets.random_dataset import RandomDataset

        model_path = _create_onnx_with_metadata(
            tmp_path,
            "bert_partial_meta.onnx",
            value_ranges={"input_ids": [0, 50]},  # no attention_mask range
        )
        dataset = RandomDataset(str(model_path), max_samples=5)

        sample = dataset[0]
        assert sample["input_ids"].min().item() >= 0
        assert sample["input_ids"].max().item() < 50
        assert "attention_mask" in sample


class TestOnnxIoMetadata:
    """Test reading winml.io.inputs from ONNX metadata via get_io_config."""

    def test_get_io_config_reads_value_ranges(self, tmp_path: Path) -> None:
        """get_io_config reads value ranges embedded in ONNX metadata_props."""
        import json

        from winml.modelkit.onnx import get_io_config

        # Create ONNX model with winml.io.inputs metadata
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph([node], "test", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        # Add metadata
        io_inputs = [
            {"name": "input_ids", "dtype": "int32", "shape": [1, 512], "value_range": [0, 30522]},
            {"name": "attention_mask", "dtype": "int32", "shape": [1, 512], "value_range": [0, 2]},
        ]
        model.metadata_props.add(key="winml.io.inputs", value=json.dumps(io_inputs))

        model_path = tmp_path / "model_with_metadata.onnx"
        onnx.save(model, str(model_path))

        # Read it back via get_io_config
        io_config = get_io_config(str(model_path))
        ranges = io_config["value_ranges"]
        assert ranges["input_ids"] == (0, 30522)
        assert ranges["attention_mask"] == (0, 2)

    def test_get_io_config_no_metadata(self, simple_onnx_model: Path) -> None:
        """Returns empty value_ranges when ONNX has no winml.io.inputs metadata."""
        from winml.modelkit.onnx import get_io_config

        io_config = get_io_config(str(simple_onnx_model))
        assert io_config["value_ranges"] == {}

    def test_get_io_config_nonexistent_file(self) -> None:
        """Raises FileNotFoundError for nonexistent file."""
        from winml.modelkit.onnx import get_io_config

        with pytest.raises(FileNotFoundError):
            get_io_config("/nonexistent/model.onnx")
