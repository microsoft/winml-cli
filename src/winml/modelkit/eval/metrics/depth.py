# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Monocular depth estimation metrics: AbsRel, RMSE, delta1.

Follows the standard NYU/KITTI evaluation protocol (Eigen et al. 2014).
Metrics are computed only over pixels where ground truth is finite,
positive, and within an optional ``[min_depth, max_depth]`` range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class DepthMetric:
    """Per-pixel depth estimation metric (AbsRel, RMSE, delta1).

    Accumulates statistics across calls to :meth:`update` and returns a
    dict from :meth:`compute`. Predictions and ground truth must be the
    same 2D shape; resampling is the caller's responsibility.
    """

    _VALID_ALIGN = ("none", "median", "affine")
    _VALID_DEPTH_KIND = ("depth", "disparity")

    def __init__(
        self,
        align: str = "affine",
        depth_kind: str = "depth",
        min_depth: float = 1e-3,
        max_depth: float | None = 10.0,
        delta_threshold: float = 1.25,
    ) -> None:
        """Initialize depth metric.

        Args:
            align: Per-image alignment of predictions to ground truth.
                ``"affine"`` (default) fits ``s * pred + t`` via
                least-squares — standard for relative-depth models
                (MiDaS, Depth-Anything, Marigold). ``"median"`` rescales
                by ``median(gt) / median(pred)`` (scale only, no shift).
                ``"none"`` evaluates predictions as-is — for metric-depth
                models like ZoeDepth and DepthPro.
            depth_kind: Output space of ``prediction``. ``"depth"``
                (default) treats values as forward depth/distance.
                ``"disparity"`` first inverts the prediction to depth
                (``1 / pred``) — for DPT/MiDaS-style models whose
                output is inverse depth.
            min_depth: Lower bound (inclusive) for valid ground-truth
                pixels in the same units as ``gt``.
            max_depth: Upper bound (inclusive) for valid ground-truth
                pixels, or ``None`` to disable. Defaults to 10 m
                (NYU indoor convention).
            delta_threshold: Threshold for delta1 accuracy.
        """
        if align not in self._VALID_ALIGN:
            raise ValueError(
                f"align must be one of {self._VALID_ALIGN}, got {align!r}.",
            )
        if depth_kind not in self._VALID_DEPTH_KIND:
            raise ValueError(
                f"depth_kind must be one of {self._VALID_DEPTH_KIND}, got {depth_kind!r}.",
            )
        if delta_threshold <= 1.0:
            raise ValueError(f"delta_threshold must be > 1, got {delta_threshold}.")

        self._align = align
        self._depth_kind = depth_kind
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth) if max_depth is not None else None
        self._delta_threshold = float(delta_threshold)

        self._abs_rel_sum = 0.0
        self._sq_err_sum = 0.0
        self._delta_hits = 0
        self._pixel_count = 0
        self._image_count = 0

    def update(self, prediction: Any, reference: Any) -> None:
        """Add one image's prediction and ground-truth depth map.

        Args:
            prediction: ``(H, W)`` array-like of predicted depth (or
                disparity, when ``depth_kind="disparity"``).
                Negative or non-finite values are treated as invalid.
            reference: ``(H, W)`` array-like of ground-truth depth in
                the same units as the aligned prediction.
        """
        pred = self._to_numpy(prediction)
        gt = self._to_numpy(reference)
        if pred.shape != gt.shape:
            raise ValueError(
                f"prediction and reference must share shape; got {pred.shape} vs {gt.shape}.",
            )

        if self._depth_kind == "disparity":
            with np.errstate(divide="ignore", invalid="ignore"):
                pred = np.where(pred > 0, 1.0 / pred, np.nan)

        valid = self._valid_mask(pred, gt)
        if not valid.any():
            self._image_count += 1
            return

        pred_v = pred[valid].astype(np.float64)
        gt_v = gt[valid].astype(np.float64)

        if self._align == "median":
            scale = np.median(gt_v) / np.median(pred_v)
            pred_v = pred_v * scale
        elif self._align == "affine":
            # Least-squares fit of (s, t) such that s * pred + t ~ gt.
            # Standard scale-and-shift alignment for relative-depth models
            # (MiDaS, Depth-Anything, Marigold).
            ones = np.ones_like(pred_v)
            a = np.stack([pred_v, ones], axis=1)
            (scale, shift), *_ = np.linalg.lstsq(a, gt_v, rcond=None)
            pred_v = pred_v * scale + shift
            # Drop pixels that fall below min_depth after alignment;
            # leaving them in would dominate abs_rel via tiny gt-side ratios
            # only when gt is small (already gated by _valid_mask).
            pos = pred_v > self._min_depth
            if not pos.any():
                self._image_count += 1
                return
            pred_v = pred_v[pos]
            gt_v = gt_v[pos]

        diff = pred_v - gt_v
        ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)

        self._abs_rel_sum += float(np.sum(np.abs(diff) / gt_v))
        self._sq_err_sum += float(np.sum(diff * diff))
        self._delta_hits += int(np.sum(ratio < self._delta_threshold))
        self._pixel_count += int(pred_v.size)
        self._image_count += 1

    def compute(self) -> dict[str, Any]:
        """Return aggregated metrics over all updates."""
        if self._pixel_count == 0:
            raise ValueError(
                "DepthMetric.compute() called with no valid pixels; "
                "check ground-truth ranges and update calls.",
            )
        return {
            "abs_rel": self._abs_rel_sum / self._pixel_count,
            "rmse": float(np.sqrt(self._sq_err_sum / self._pixel_count)),
            "delta1": self._delta_hits / self._pixel_count,
            "num_images": self._image_count,
            "num_valid_pixels": self._pixel_count,
        }

    def reset(self) -> None:
        """Clear accumulated state for a fresh evaluation."""
        self._abs_rel_sum = 0.0
        self._sq_err_sum = 0.0
        self._delta_hits = 0
        self._pixel_count = 0
        self._image_count = 0

    def _valid_mask(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Pixels where both prediction and ground truth are usable."""
        mask = np.isfinite(gt) & (gt > self._min_depth)
        if self._max_depth is not None:
            mask &= gt <= self._max_depth
        mask &= np.isfinite(pred) & (pred > 0)
        return mask

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        """Convert torch.Tensor / PIL / numpy to a 2D float numpy array."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().squeeze()
        return np.asarray(arr).squeeze()
