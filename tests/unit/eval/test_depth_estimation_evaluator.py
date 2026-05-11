# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLDepthEstimationEvaluator schema validation,
column-mapping handling, and pipeline-output extraction."""

import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from winml.modelkit.eval import WinMLDepthEstimationEvaluator
from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    def __init__(self):
        self.config = type("Cfg", (), {})()

    @property
    def io_config(self):
        return {"input_shapes": [[1, 3, 224, 224]]}


def make_evaluator(
    input_col: str = "image",
    depth_col: str = "depth_map",
    align: str = "median",
    min_depth: float = 1e-3,
    max_depth=10.0,
):
    """Create evaluator without triggering __init__ data loading."""
    ev = object.__new__(WinMLDepthEstimationEvaluator)
    ev.model = MockModel()
    ev._input_col = input_col
    ev._depth_col = depth_col
    ev._align = align
    ev._min_depth = float(min_depth)
    ev._max_depth = None if max_depth is None else float(max_depth)
    return ev


def create_rgb_image(width: int, height: int):
    return PILImage.new("RGB", (width, height), (128, 128, 128))


def create_depth_image(arr: np.ndarray):
    """Create an HF-friendly single-channel float depth image."""
    return PILImage.fromarray(arr.astype(np.float32), mode="F")


def make_depth_dataset(images, depth_maps, depth_col: str = "depth_map"):
    features = Features(
        {
            "image": Image(mode="RGB"),
            depth_col: Image(mode="F"),
        }
    )
    return Dataset.from_dict(
        {
            "image": images,
            depth_col: depth_maps,
        },
        features=features,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_depth_estimation_registered(self):
        assert "depth-estimation" in _EVALUATOR_REGISTRY
        assert _EVALUATOR_REGISTRY["depth-estimation"] is WinMLDepthEstimationEvaluator


# ---------------------------------------------------------------------------
# Schema info
# ---------------------------------------------------------------------------


class TestSchemaInfo:
    def test_schema_info_contains_image_and_depth(self):
        cols = WinMLDepthEstimationEvaluator.schema_info()
        names = [c.name for c in cols]
        assert "image" in names
        assert "depth_map" in names

    def test_schema_info_input_and_depth_marked_required(self):
        cols = {c.name: c for c in WinMLDepthEstimationEvaluator.schema_info()}
        # Required columns expose their --column override key.
        assert cols["image"].override == "input_column"
        assert cols["depth_map"].override == "depth_column"
        assert cols["image"].required is True
        assert cols["depth_map"].required is True


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema_passes(self):
        ev = make_evaluator()
        img = create_rgb_image(4, 3)
        depth = create_depth_image(np.ones((3, 4), dtype=np.float32))
        ds = make_depth_dataset([img], [depth])
        ev._validate_schema(ds)  # should not raise

    def test_missing_image_column_raises(self):
        ev = make_evaluator()
        ds = Dataset.from_dict({"text": ["hello"], "depth_map": ["a"]})
        with pytest.raises(ValueError, match="missing input column 'image'"):
            ev._validate_schema(ds)

    def test_missing_depth_column_raises(self):
        ev = make_evaluator()
        features = Features({"image": Image(mode="RGB"), "label": Value("int64")})
        img = create_rgb_image(4, 3)
        ds = Dataset.from_dict({"image": [img], "label": [0]}, features=features)
        with pytest.raises(ValueError, match="missing depth column 'depth_map'"):
            ev._validate_schema(ds)

    def test_custom_columns_mapping(self):
        ev = make_evaluator(input_col="rgb", depth_col="z")
        features = Features({"rgb": Image(mode="RGB"), "z": Image(mode="F")})
        img = create_rgb_image(4, 3)
        depth = create_depth_image(np.ones((3, 4), dtype=np.float32))
        ds = Dataset.from_dict({"rgb": [img], "z": [depth]}, features=features)
        ev._validate_schema(ds)  # should not raise


# ---------------------------------------------------------------------------
# Pipeline output extraction
# ---------------------------------------------------------------------------


class TestExtractPredictedDepth:
    def test_torch_tensor_output(self):
        out = {"predicted_depth": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        arr = WinMLDepthEstimationEvaluator._extract_predicted_depth(out)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32

    def test_singleton_dim_squeezed(self):
        out = {"predicted_depth": torch.zeros((1, 1, 4, 5))}
        arr = WinMLDepthEstimationEvaluator._extract_predicted_depth(out)
        assert arr.shape == (4, 5)

    def test_numpy_predicted_depth(self):
        out = {"predicted_depth": np.ones((3, 3), dtype=np.float64)}
        arr = WinMLDepthEstimationEvaluator._extract_predicted_depth(out)
        assert arr.shape == (3, 3)
        assert arr.dtype == np.float32

    def test_pil_depth_only_raises(self):
        # The pipeline's "depth" key is an 8-bit grayscale visualization,
        # not the numeric tensor. Don't silently use it as a metric input.
        depth_img = PILImage.new("L", (4, 3), 0)
        with pytest.raises(ValueError, match="missing"):
            WinMLDepthEstimationEvaluator._extract_predicted_depth({"depth": depth_img})

    def test_missing_keys_raise(self):
        with pytest.raises(ValueError, match="missing"):
            WinMLDepthEstimationEvaluator._extract_predicted_depth({"foo": 1})

    def test_non_dict_raises(self):
        with pytest.raises(TypeError, match="dict"):
            WinMLDepthEstimationEvaluator._extract_predicted_depth([1, 2, 3])


# ---------------------------------------------------------------------------
# prepare_pipeline — image processor alignment to ONNX input shape
# ---------------------------------------------------------------------------


class _FakeImageProcessor:
    """Stand-in for HF AutoImageProcessor with attribute-based knobs."""

    def __init__(
        self,
        size=None,
        keep_aspect_ratio: bool | None = None,
        do_pad: bool | None = None,
    ):
        self.size = size if size is not None else {"height": 0, "width": 0}
        if keep_aspect_ratio is not None:
            self.keep_aspect_ratio = keep_aspect_ratio
        if do_pad is not None:
            self.do_pad = do_pad


class _FakePreparedPipe:
    """Stand-in for the parent ``prepare_pipeline()`` return value."""

    def __init__(self, image_processor):
        self.image_processor = image_processor


class TestPreparePipeline:
    """Verify processor is forced to the static ONNX shape exactly."""

    @staticmethod
    def _patch_super_pipeline(monkeypatch, processor):
        """Make ``WinMLEvaluator.prepare_pipeline`` return a pipe with `processor`."""
        from winml.modelkit.eval import base_evaluator

        monkeypatch.setattr(
            base_evaluator.WinMLEvaluator,
            "prepare_pipeline",
            lambda self: _FakePreparedPipe(processor),
        )

    def test_sets_size_from_io_config(self, monkeypatch):
        """ONNX (h, w) is written into ``image_processor.size``."""
        proc = _FakeImageProcessor(size={"height": 0, "width": 0})
        self._patch_super_pipeline(monkeypatch, proc)

        ev = make_evaluator()
        ev.model = type(
            "M", (), {"io_config": {"input_shapes": [[1, 3, 518, 518]]}}
        )()

        pipe = ev.prepare_pipeline()
        assert pipe.image_processor.size == {"height": 518, "width": 518}

    def test_disables_keep_aspect_ratio_when_present(self, monkeypatch):
        """``keep_aspect_ratio`` is turned off so the resize hits the exact target."""
        proc = _FakeImageProcessor(
            size={"height": 0, "width": 0},
            keep_aspect_ratio=True,
        )
        self._patch_super_pipeline(monkeypatch, proc)

        ev = make_evaluator()
        ev.model = type(
            "M", (), {"io_config": {"input_shapes": [[1, 3, 518, 518]]}}
        )()

        pipe = ev.prepare_pipeline()
        assert pipe.image_processor.keep_aspect_ratio is False

    def test_disables_do_pad_when_present(self, monkeypatch):
        """``do_pad`` is turned off so the processor doesn't pad past target."""
        proc = _FakeImageProcessor(
            size={"height": 0, "width": 0},
            do_pad=True,
        )
        self._patch_super_pipeline(monkeypatch, proc)

        ev = make_evaluator()
        ev.model = type(
            "M", (), {"io_config": {"input_shapes": [[1, 3, 384, 384]]}}
        )()

        pipe = ev.prepare_pipeline()
        assert pipe.image_processor.do_pad is False

    def test_no_attrs_does_not_raise(self, monkeypatch):
        """Processors without keep_aspect_ratio / do_pad work unchanged."""
        proc = _FakeImageProcessor(size={"height": 0, "width": 0})
        # No keep_aspect_ratio, no do_pad attributes.
        assert not hasattr(proc, "keep_aspect_ratio")
        assert not hasattr(proc, "do_pad")
        self._patch_super_pipeline(monkeypatch, proc)

        ev = make_evaluator()
        ev.model = type(
            "M", (), {"io_config": {"input_shapes": [[1, 3, 224, 224]]}}
        )()

        pipe = ev.prepare_pipeline()
        assert pipe.image_processor.size == {"height": 224, "width": 224}
        assert not hasattr(pipe.image_processor, "keep_aspect_ratio")
        assert not hasattr(pipe.image_processor, "do_pad")

    def test_missing_io_config_is_noop(self, monkeypatch):
        """If model lacks io_config, processor is not modified."""
        proc = _FakeImageProcessor(
            size={"height": 0, "width": 0}, keep_aspect_ratio=True
        )
        self._patch_super_pipeline(monkeypatch, proc)

        ev = make_evaluator()
        ev.model = type("M", (), {})()  # no io_config

        pipe = ev.prepare_pipeline()
        # Untouched
        assert pipe.image_processor.size == {"height": 0, "width": 0}
        assert pipe.image_processor.keep_aspect_ratio is True


# ---------------------------------------------------------------------------
# align_labels
# ---------------------------------------------------------------------------


class TestAlignLabels:
    def test_align_labels_returns_dataset_unchanged(self):
        ev = make_evaluator()
        img = create_rgb_image(4, 3)
        depth = create_depth_image(np.ones((3, 4), dtype=np.float32))
        ds = make_depth_dataset([img], [depth])
        ds_config = type("Cfg", (), {"label_mapping": None})()
        result = ev.align_labels(ds, ds_config)
        # Same dataset, no remapping.
        assert result.column_names == ds.column_names
        assert len(result) == len(ds)

    def test_align_labels_invalid_schema_raises(self):
        ev = make_evaluator()
        bad = Dataset.from_dict({"foo": [1]})
        ds_config = type("Cfg", (), {"label_mapping": None})()
        with pytest.raises(ValueError, match="missing input column"):
            ev.align_labels(bad, ds_config)


# ---------------------------------------------------------------------------
# compute() integration with mocked pipeline
# ---------------------------------------------------------------------------


class _FakePipe:
    """Minimal pipe stand-in that returns a perfect prediction."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        # Use the image's size to produce a same-shape prediction.
        h = getattr(image, "height", 4)
        w = getattr(image, "width", 4)
        return {"predicted_depth": torch.full((h, w), 5.0)}


class TestCompute:
    def test_compute_perfect_prediction(self):
        ev = make_evaluator(align="none", min_depth=0.0, max_depth=None)
        ev.pipe = _FakePipe()

        img = create_rgb_image(4, 4)
        depth = np.full((4, 4), 5.0, dtype=np.float32)
        ev.data = [{"image": img, "depth_map": depth}]

        result = ev.compute()
        assert result["abs_rel"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["delta1"] == pytest.approx(1.0)
        assert result["num_images"] == 1

    def test_compute_skips_missing_samples(self):
        ev = make_evaluator(align="none", min_depth=0.0, max_depth=None)
        ev.pipe = _FakePipe()

        img = create_rgb_image(4, 4)
        depth = np.full((4, 4), 5.0, dtype=np.float32)
        ev.data = [
            {"image": None, "depth_map": depth},  # skipped
            {"image": img, "depth_map": None},  # skipped
            {"image": img, "depth_map": depth},  # valid
        ]
        result = ev.compute()
        assert result["num_images"] == 1

    def test_compute_raises_on_pred_gt_shape_mismatch(self):
        """Pred/GT shape mismatch propagates as a clear error from DepthMetric."""

        class MismatchPipe:
            def __call__(self, image):
                return {"predicted_depth": torch.full((8, 8), 5.0)}

        ev = make_evaluator(align="none", min_depth=0.0, max_depth=None)
        ev.pipe = MismatchPipe()
        img = create_rgb_image(4, 4)
        depth = np.full((4, 4), 5.0, dtype=np.float32)
        ev.data = [{"image": img, "depth_map": depth}]

        with pytest.raises(ValueError, match="share shape"):
            ev.compute()
