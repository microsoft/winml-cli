# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Mask-generation dataset for promptable segmentation models (SAM/SAM2/SAM3).

Unlike ``ImageSegmentationDataset`` (semantic segmentation, single model →
pixel-wise class map), promptable mask-generation requires
``(image, prompt, gt_mask)`` triples: the model receives an image *and* a
user prompt (bbox / point / text concept), and emits the corresponding
binary mask.

This dataset yields raw image + GT mask + an auto-derived prompt -- it
does **not** apply SAM-specific preprocessing (1008x1008 padding,
ImageNet-mean normalization), because that lives in the evaluator
alongside the encoder/decoder ONNX sessions.

Supported prompt modes:
- ``"bbox"`` -- tight axis-aligned bbox derived from the GT mask.
  Matches the SAM family's standard mIoU benchmark protocol.
- ``"point"`` — single foreground point sampled from the GT mask
  centroid (or a random foreground pixel if centroid is outside the
  mask).
- ``"text"`` — free-form text concept. Requires the dataset to expose a
  text column via ``text_col`` config (typically a class name / caption).
  SAM 3's flagship "concept-prompted segmentation" mode.

The yielded GT mask is collapsed to **binary foreground vs background**
(``mask > 0``) by default. Datasets that distinguish multiple instances
or classes can opt into instance-level GT via ``binarize=False``, but
multi-instance evaluation is left to the evaluator (this dataset still
emits one prompt per sample; instance-level AP requires repeating
samples once per instance, which the COCO evaluator handles).
"""

from __future__ import annotations

import logging
from random import Random
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from datasets import load_dataset
from datasets.features import Image as HFImage
from PIL import Image as PILImage

from .base import BaseTaskDataset


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)

PromptMode = Literal["bbox", "point", "text"]
VALID_PROMPT_MODES: tuple[PromptMode, ...] = ("bbox", "point", "text")

# Mask coverage filter defaults (fraction of pixels that are foreground).
# Excludes degenerate empty masks (no signal) and near-full masks (trivial).
DEFAULT_MIN_COVERAGE = 0.005  # 0.5%
DEFAULT_MAX_COVERAGE = 0.95   # 95%


class MaskGenerationDataset(BaseTaskDataset):
    """Dataset for promptable mask-generation tasks (SAM/SAM2/SAM3 family).

    Each sample is a ``dict`` with keys:

    - ``image``: ``PIL.Image.Image`` in RGB, at original resolution.
    - ``gt_mask``: ``np.ndarray`` of shape ``(H, W)`` and dtype ``bool``,
      same size as ``image``.
    - ``prompt``: ``dict`` whose key depends on ``prompt_mode``:
        - bbox: ``{"bbox": [x1, y1, x2, y2]}`` (xyxy, in original pixels).
        - point: ``{"point": [x, y], "label": 1}`` (foreground point).
        - text: ``{"text": "<concept-string>"}``.
    - ``sample_id``: ``str`` -- stable identifier for logging/visualization.
    """

    DEFAULT_DATASET = "mattmdjaga/human_parsing_dataset"
    DEFAULT_SPLIT = "train"

    def __init__(
        self,
        model_name: str,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        prompt_mode: PromptMode = "bbox",
        binarize: bool = True,
        min_mask_coverage: float = DEFAULT_MIN_COVERAGE,
        max_mask_coverage: float = DEFAULT_MAX_COVERAGE,
        text_col: str | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the mask-generation dataset.

        Args:
            model_name: HuggingFace model identifier (kept for API parity
                with other datasets; mask-generation does not consult the
                model's image processor since SAM does its own
                preprocessing in the evaluator).
            dataset_name: Source dataset (defaults to
                ``mattmdjaga/human_parsing_dataset``).
            max_samples: Cap the number of samples (None = all).
            data_split: HF dataset split (defaults to ``"train"``).
            prompt_mode: One of ``"bbox"``, ``"point"``, ``"text"``.
            binarize: Collapse multi-class masks to foreground-vs-background.
            min_mask_coverage: Drop samples whose foreground fraction is
                below this (filters empty/near-empty masks).
            max_mask_coverage: Drop samples whose foreground fraction is
                above this (filters trivial near-full masks).
            text_col: For ``prompt_mode="text"``, the dataset column holding
                the text prompt. Ignored otherwise.
            seed: RNG seed for point sampling + sample subselection.
            **kwargs: forwarded to ``BaseTaskDataset``.
        """
        if prompt_mode not in VALID_PROMPT_MODES:
            raise ValueError(
                f"prompt_mode={prompt_mode!r} is not one of {VALID_PROMPT_MODES}"
            )
        if not 0.0 <= min_mask_coverage <= max_mask_coverage <= 1.0:
            raise ValueError(
                "Require 0 <= min_mask_coverage <= max_mask_coverage <= 1; "
                f"got min={min_mask_coverage}, max={max_mask_coverage}"
            )

        self._prompt_mode = prompt_mode
        self._binarize = binarize
        self._min_coverage = min_mask_coverage
        self._max_coverage = max_mask_coverage
        self._text_col = text_col
        self._seed = seed
        self._rng = Random(seed)

        if data_split is None:
            data_split = self.DEFAULT_SPLIT

        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if self._dataset_name is None:
            self._dataset_name = self.DEFAULT_DATASET

        logger.info(
            "Loading mask-generation dataset %s (split=%s, prompt_mode=%s)",
            self._dataset_name, self._data_split, self._prompt_mode,
        )
        dataset = load_dataset(self._dataset_name, split=self._data_split)
        self._detect_columns(dataset)

        if self._prompt_mode == "text" and self._text_col is None:
            raise ValueError(
                "prompt_mode='text' requires text_col to be set explicitly "
                f"(dataset {self._dataset_name!r} columns: "
                f"{list(dataset.features) if hasattr(dataset, 'features') else 'unknown'})"
            )
        if self._text_col is not None and self._text_col not in dataset.features:
            raise ValueError(
                f"text_col={self._text_col!r} not found in dataset features "
                f"({list(dataset.features)})"
            )

        # Optional shuffle BEFORE truncation so different seeds give
        # different sample windows.
        shuffle = self._config.get("shuffle", False)
        if shuffle:
            dataset = dataset.shuffle(seed=self._seed)

        # Apply max_samples cap (or take all). We over-fetch slightly to
        # absorb the coverage filter so a hard cap of N still yields ~N
        # samples in most cases.
        if self._max_samples is not None:
            cap = self._max_samples
            over = min(max(2 * cap, cap + 20), len(dataset))
            dataset = dataset.select(range(over))
        # else: keep full dataset (caller manages cost)

        self._dataset = dataset
        logger.info(
            "Mask-generation dataset ready: %d candidate samples, image='%s', mask='%s'",
            len(self._dataset), self._image_col, self._mask_col,
        )

    def _detect_columns(self, dataset: Any) -> None:
        """Pick image + mask columns.

        Uses the same heuristics as ``ImageSegmentationDataset``: prefer
        name-matched HF Image features, else fall back to the
        only-two-Image-columns convention.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")
        features = dataset.features

        image_cands: list[str] = []
        mask_cands: list[str] = []
        for col, feat in features.items():
            if not isinstance(feat, HFImage):
                continue
            lc = col.lower()
            mask_keywords = ("annotation", "mask", "label", "segmentation", "target", "gt")
            if any(k in lc for k in mask_keywords):
                mask_cands.append(col)
            else:
                image_cands.append(col)

        # Fallback: if only two Image columns and we couldn't classify both,
        # assume order [image, mask].
        all_image_feats = [c for c, f in features.items() if isinstance(f, HFImage)]
        if len(all_image_feats) == 2 and (not image_cands or not mask_cands):
            image_cands = [all_image_feats[0]]
            mask_cands = [all_image_feats[1]]

        if not image_cands or not mask_cands:
            raise ValueError(
                f"Could not auto-detect image + mask columns in "
                f"{self._dataset_name!r}; available features: {list(features)}"
            )

        # Prefer canonical names if present.
        self._image_col = next((c for c in image_cands if c.lower() == "image"), image_cands[0])
        preferred_mask = ("mask", "annotation", "label", "segmentation")
        self._mask_col = next(
            (c for c in mask_cands if c.lower() in preferred_mask),
            mask_cands[0],
        )

    # ------------------------------------------------------------------
    # ABC overrides
    # ------------------------------------------------------------------

    @property
    def label_col(self) -> str:
        """Dataset column holding the per-sample mask (alias of ``mask_col``)."""
        return self._mask_col

    @property
    def mask_col(self) -> str:
        """Dataset column holding the per-sample mask."""
        return self._mask_col

    @property
    def image_col(self) -> str:
        """Dataset column holding the per-sample image."""
        return self._image_col

    @property
    def prompt_mode(self) -> PromptMode:
        """Configured prompt mode (``bbox`` / ``point`` / ``text``)."""
        return self._prompt_mode

    def __len__(self) -> int:
        # Note: len() reflects the candidate window; the coverage filter
        # is applied lazily in __getitem__, so iterators should check
        # the returned dict for None (skip) or use iter_valid() below.
        return len(self._dataset) if self._dataset is not None else 0

    def __getitem__(self, idx: int) -> dict[str, Any] | None:  # type: ignore[override]
        """Return one ``(image, gt_mask, prompt)`` triple, or ``None``.

        Returns ``None`` when the sample fails the coverage filter (caller
        should skip).
        """
        if self._dataset is None:
            raise IndexError("Dataset not initialized")
        row = self._dataset[idx]
        image = row[self._image_col]
        mask = row[self._mask_col]
        if not isinstance(image, PILImage.Image):
            raise TypeError(f"Expected PIL.Image at row {idx}, got {type(image).__name__}")
        image = image.convert("RGB")

        gt = self._to_binary_mask(mask, target_size=(image.height, image.width))
        coverage = float(gt.mean())
        if not self._min_coverage <= coverage <= self._max_coverage:
            return None

        prompt = self._derive_prompt(gt, row)
        return {
            "image": image,
            "gt_mask": gt,
            "prompt": prompt,
            "sample_id": f"sample_{idx:04d}",
            "coverage": coverage,
        }

    def iter_valid(self, max_samples: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield only samples that pass the coverage filter.

        Args:
            max_samples: stop after this many valid samples (None = all
                that pass).
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized")
        cap = max_samples if max_samples is not None else (self._max_samples or len(self._dataset))
        yielded = 0
        for i in range(len(self._dataset)):
            if yielded >= cap:
                break
            sample = self[i]
            if sample is None:
                continue
            yield sample
            yielded += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_binary_mask(self, mask: Any, target_size: tuple[int, int]) -> np.ndarray:
        """Convert raw mask to ``(H, W) bool`` foreground array.

        ``target_size`` is ``(H, W)`` of the paired image. Some datasets
        (notably LIP / human_parsing) store masks at transposed
        resolution; if shapes disagree we resize the mask to match the
        image.
        """
        if not isinstance(mask, PILImage.Image):
            raise TypeError(f"Expected PIL.Image for mask, got {type(mask).__name__}")
        if mask.size != (target_size[1], target_size[0]):
            mask = mask.resize((target_size[1], target_size[0]), PILImage.Resampling.NEAREST)
        arr = np.array(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if self._binarize:
            return arr > 0
        # Instance mode: caller will handle multi-class.
        return arr.astype(np.int32)

    def _derive_prompt(self, gt: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
        """Derive a prompt of the configured mode from the GT mask + row."""
        if self._prompt_mode == "bbox":
            x1, y1, x2, y2 = _bbox_from_mask(gt)
            return {"bbox": [int(x1), int(y1), int(x2), int(y2)]}
        if self._prompt_mode == "point":
            x, y = _foreground_point(gt, self._rng)
            return {"point": [int(x), int(y)], "label": 1}
        # text mode
        text = row[self._text_col]  # type: ignore[index]
        if not isinstance(text, str):
            text = str(text)
        return {"text": text}


# ----------------------------------------------------------------------
# Geometry helpers (small, pure, easy to unit-test)
# ----------------------------------------------------------------------


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight axis-aligned bbox ``(x1, y1, x2, y2)`` for ``mask``.

    Computed around the True pixels of ``mask`` (bool 2D).
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Cannot derive bbox from empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _foreground_point(mask: np.ndarray, rng: Random) -> tuple[int, int]:
    """Single foreground point.

    Returns the mask centroid when it lies inside the mask, else a
    random foreground pixel. Falling back to a random pixel handles
    concave / disconnected masks (e.g., people with bags) where the
    centroid lands in background.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("Cannot derive point from empty mask")
    cy, cx = int(ys.mean()), int(xs.mean())
    if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
        return cx, cy
    # Fall back to a random foreground pixel.
    i = rng.randrange(ys.size)
    return int(xs[i]), int(ys[i])
