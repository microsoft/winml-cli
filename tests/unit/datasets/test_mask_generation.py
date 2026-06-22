# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for MaskGenerationDataset and its geometry helpers.

These tests mock ``datasets.load_dataset`` so they don't hit the network --
each test builds a small in-memory fake dataset that mimics the HF
``datasets.Dataset`` interface enough for ``MaskGenerationDataset`` to
operate on.
"""

from __future__ import annotations

from random import Random
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from datasets.features import Features, Value
from datasets.features import Image as HFImage
from PIL import Image as PILImage

from winml.modelkit.datasets.mask_generation import (
    MaskGenerationDataset,
    _bbox_from_mask,
    _foreground_point,
)


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestBboxFromMask:
    def test_centered_square(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 4:9] = True
        assert _bbox_from_mask(mask) == (4, 3, 8, 6)

    def test_single_pixel(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True
        assert _bbox_from_mask(mask) == (3, 2, 3, 2)

    def test_empty_mask_raises(self):
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.raises(ValueError, match="empty mask"):
            _bbox_from_mask(mask)

    def test_accepts_non_bool_dtype(self):
        """Helper should coerce uint8/int masks transparently."""
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 1
        assert _bbox_from_mask(mask) == (2, 1, 4, 3)


class TestForegroundPoint:
    def test_centroid_inside_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True  # convex square; centroid at (9, 9)
        rng = Random(0)
        x, y = _foreground_point(mask, rng)
        # Centroid of pixels 5..14 = (9.5 floored to 9)
        assert mask[y, x]  # point IS on mask
        assert (x, y) == (9, 9)

    def test_concave_mask_falls_back_to_random(self):
        """Two disjoint blobs -- centroid lands in background."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:5, 2:5] = True
        mask[15:18, 15:18] = True
        # Mean of all foreground ys/xs ~= (9.5, 9.5) which is background.
        rng = Random(42)
        x, y = _foreground_point(mask, rng)
        assert mask[y, x]  # picked an actual foreground pixel

    def test_empty_mask_raises(self):
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.raises(ValueError, match="empty mask"):
            _foreground_point(mask, Random(0))


# ---------------------------------------------------------------------------
# Fake HuggingFace dataset
# ---------------------------------------------------------------------------


def _make_pil(image_arr: np.ndarray) -> PILImage.Image:
    """Wrap a numpy array as a PIL image (uint8 RGB or single-channel)."""
    if image_arr.ndim == 2:
        return PILImage.fromarray(image_arr.astype(np.uint8), mode="L")
    return PILImage.fromarray(image_arr.astype(np.uint8), mode="RGB")


class _FakeHFDataset:
    """Minimal stand-in for ``datasets.Dataset`` -- enough for
    ``MaskGenerationDataset`` to call ``len()``, ``[]``, ``.select()``,
    ``.shuffle()``, ``.features``.
    """

    def __init__(self, rows: list[dict[str, Any]], features: Features):
        self._rows = rows
        self.features = features

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices], self.features)

    def shuffle(self, seed: int = 0):
        rng = Random(seed)
        rows = list(self._rows)
        rng.shuffle(rows)
        return _FakeHFDataset(rows, self.features)


def _build_fake_ds(n_rows: int = 5, with_text: bool = False) -> _FakeHFDataset:
    """Build a fake dataset of (image, mask) pairs.

    Image: 32x40 RGB (W=40, H=32 in PIL terms).
    Mask: same size, foreground = a centered 10x12 rectangle.
    """
    feats = {
        "image": HFImage(),
        "mask": HFImage(),
    }
    if with_text:
        feats["text"] = Value("string")
    features = Features(feats)

    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        img = np.full((32, 40, 3), 200, dtype=np.uint8)
        mask = np.zeros((32, 40), dtype=np.uint8)
        # Cover ~10*12/(32*40) = 9.4% (passes default 0.5%-95% filter).
        mask[10:22, 14:24] = 1
        row = {"image": _make_pil(img), "mask": _make_pil(mask)}
        if with_text:
            row["text"] = f"object_{i}"
        rows.append(row)
    return _FakeHFDataset(rows, features)


# ---------------------------------------------------------------------------
# MaskGenerationDataset
# ---------------------------------------------------------------------------


def _make_ds(monkeypatch, fake_ds, **kwargs) -> MaskGenerationDataset:
    """Construct a MaskGenerationDataset with load_dataset patched to fake_ds."""
    with patch(
        "winml.modelkit.datasets.mask_generation.load_dataset",
        return_value=fake_ds,
    ):
        return MaskGenerationDataset(
            model_name="dummy/model",
            dataset_name="dummy/dataset",
            **kwargs,
        )


class TestBboxPromptMode:
    def test_basic_bbox_sample(self, monkeypatch):
        fake = _build_fake_ds(n_rows=3)
        ds = _make_ds(monkeypatch, fake)
        assert ds.prompt_mode == "bbox"
        sample = ds[0]
        assert sample is not None
        assert isinstance(sample["image"], PILImage.Image)
        assert sample["image"].size == (40, 32)
        assert sample["gt_mask"].shape == (32, 40)
        assert sample["gt_mask"].dtype == np.bool_
        assert sample["prompt"] == {"bbox": [14, 10, 23, 21]}
        assert sample["sample_id"] == "sample_0000"

    def test_image_col_property(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1)
        ds = _make_ds(monkeypatch, fake)
        assert ds.image_col == "image"
        assert ds.mask_col == "mask"
        assert ds.label_col == ds.mask_col


class TestPointPromptMode:
    def test_point_inside_foreground(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1)
        ds = _make_ds(monkeypatch, fake, prompt_mode="point")
        sample = ds[0]
        assert sample is not None
        assert "point" in sample["prompt"]
        x, y = sample["prompt"]["point"]
        assert sample["gt_mask"][y, x]
        assert sample["prompt"]["label"] == 1


class TestTextPromptMode:
    def test_text_pulled_from_configured_col(self, monkeypatch):
        fake = _build_fake_ds(n_rows=2, with_text=True)
        ds = _make_ds(monkeypatch, fake, prompt_mode="text", text_col="text")
        sample = ds[1]
        assert sample is not None
        assert sample["prompt"] == {"text": "object_1"}

    def test_text_mode_requires_text_col(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1, with_text=False)
        with pytest.raises(ValueError, match="requires text_col"):
            _make_ds(monkeypatch, fake, prompt_mode="text")

    def test_text_col_must_exist(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1, with_text=False)
        with pytest.raises(ValueError, match="not found"):
            _make_ds(monkeypatch, fake, prompt_mode="text", text_col="missing")


class TestCoverageFilter:
    def test_below_min_returns_none(self, monkeypatch):
        # Build a dataset with a 1-pixel mask -> coverage ~0.08%
        feats = Features({"image": HFImage(), "mask": HFImage()})
        img = np.full((32, 40, 3), 0, dtype=np.uint8)
        mask = np.zeros((32, 40), dtype=np.uint8)
        mask[0, 0] = 1
        fake = _FakeHFDataset(
            [{"image": _make_pil(img), "mask": _make_pil(mask)}],
            feats,
        )
        ds = _make_ds(monkeypatch, fake, min_mask_coverage=0.01)
        assert ds[0] is None  # filtered out

    def test_above_max_returns_none(self, monkeypatch):
        feats = Features({"image": HFImage(), "mask": HFImage()})
        img = np.full((10, 10, 3), 0, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)  # 100% coverage
        fake = _FakeHFDataset(
            [{"image": _make_pil(img), "mask": _make_pil(mask)}],
            feats,
        )
        ds = _make_ds(monkeypatch, fake, max_mask_coverage=0.95)
        assert ds[0] is None

    def test_iter_valid_skips_filtered(self, monkeypatch):
        feats = Features({"image": HFImage(), "mask": HFImage()})
        rows = []
        # 3 good, 2 too-small interleaved
        for cov in [1.0, 0.0, 1.0, 0.0, 1.0]:
            img = np.zeros((32, 40, 3), dtype=np.uint8)
            mask = np.zeros((32, 40), dtype=np.uint8)
            if cov > 0:
                mask[5:15, 5:15] = 1
            rows.append({"image": _make_pil(img), "mask": _make_pil(mask)})
        fake = _FakeHFDataset(rows, feats)
        ds = _make_ds(monkeypatch, fake)
        valid = list(ds.iter_valid())
        assert len(valid) == 3
        assert [s["sample_id"] for s in valid] == [
            "sample_0000", "sample_0002", "sample_0004",
        ]

    def test_iter_valid_respects_cap(self, monkeypatch):
        fake = _build_fake_ds(n_rows=10)
        ds = _make_ds(monkeypatch, fake)
        assert len(list(ds.iter_valid(max_samples=4))) == 4


class TestValidation:
    def test_bad_prompt_mode(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1)
        with pytest.raises(ValueError, match="prompt_mode"):
            _make_ds(monkeypatch, fake, prompt_mode="silly")  # type: ignore[arg-type]

    def test_bad_coverage_bounds(self, monkeypatch):
        fake = _build_fake_ds(n_rows=1)
        with pytest.raises(ValueError, match="min_mask_coverage"):
            _make_ds(
                monkeypatch, fake,
                min_mask_coverage=0.9, max_mask_coverage=0.1,
            )

    def test_missing_columns_raises(self, monkeypatch):
        """Dataset with no Image features should error out clearly."""
        feats = Features({"label": Value("int32")})
        fake = _FakeHFDataset([{"label": 1}], feats)
        with pytest.raises(ValueError, match="auto-detect"):
            _make_ds(monkeypatch, fake)


class TestRegistry:
    def test_mask_generation_in_task_dataset_mapping(self):
        """Mask-generation task is wired into the dataset registry."""
        from winml.modelkit.datasets import TASK_DATASET_MAPPING

        assert TASK_DATASET_MAPPING["mask-generation"] is MaskGenerationDataset
