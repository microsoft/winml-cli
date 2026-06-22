# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""COCO keypoint detection metric: OKS-based Average Precision.

Computes the official COCO keypoint score — Average Precision averaged over
Object Keypoint Similarity (OKS) thresholds 0.50:0.95 — via ``pycocotools``
``COCOeval(iouType="keypoints")``. This mirrors how the object-detection
evaluator reuses the COCO mAP protocol, but for pose keypoints.
"""

from __future__ import annotations

from typing import Any


# Standard COCO 17-keypoint OKS per-keypoint constants (pycocotools default).
# Exposed so non-COCO keypoint layouts can override them.
COCO_KEYPOINT_SIGMAS: tuple[float, ...] = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)

# COCO person keypoint names (order matters; index == keypoint id).
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class KeypointAPMetric:
    """COCO-standard keypoint AP (OKS) wrapping ``pycocotools`` ``COCOeval``.

    Accepts per-instance predictions and ground truth as plain Python dicts
    keyed by ``image_id`` and builds the COCO JSON structures internally. One
    instance is one person (top-down pose estimation produces one keypoint set
    per person box).
    """

    def compute(
        self,
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
        sigmas: tuple[float, ...] = COCO_KEYPOINT_SIGMAS,
        keypoint_names: tuple[str, ...] = COCO_KEYPOINT_NAMES,
    ) -> dict[str, float]:
        """Compute COCO keypoint AP/AR.

        Args:
            predictions: Per-person predictions. Each dict has:
                - ``image_id``: int grouping key
                - ``keypoints``: flat list ``[x1, y1, s1, ...]`` of length
                  ``3 * num_keypoints`` (``s`` is the per-keypoint score)
                - ``score``: overall person confidence (float)
            references: Per-person ground truth. Each dict has:
                - ``image_id``: int grouping key
                - ``keypoints``: flat list ``[x1, y1, v1, ...]`` (``v`` is the
                  COCO visibility flag 0/1/2)
                - ``bbox``: ``[x, y, w, h]`` person box
                - ``area``: person area used by the OKS normalization
                - ``num_keypoints``: number of labeled keypoints (optional;
                  derived from visibility flags when absent)
            sigmas: Per-keypoint OKS constants. Defaults to the COCO 17.
            keypoint_names: Keypoint names for the category definition.

        Returns:
            Dict with ``AP``, ``AP50``, ``AP75``, ``AP_medium``, ``AP_large``,
            ``AR``, ``AR50``, ``AR75``, plus ``num_predictions``,
            ``num_ground_truths`` and ``num_images``.
        """
        import contextlib
        import io

        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self._validate_keypoint_counts(predictions, references, len(sigmas))

        image_ids = sorted(
            {int(r["image_id"]) for r in references} | {int(p["image_id"]) for p in predictions}
        )

        gt_dict = {
            "images": [{"id": image_id} for image_id in image_ids],
            "annotations": self._build_gt_annotations(references),
            "categories": [
                {"id": 1, "name": "person", "keypoints": list(keypoint_names), "skeleton": []}
            ],
        }

        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        # pycocotools writes progress to stdout; keep eval output quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()

        detections = [
            {
                "image_id": int(p["image_id"]),
                "category_id": 1,
                "keypoints": [float(v) for v in p["keypoints"]],
                "score": float(p["score"]),
            }
            for p in predictions
        ]

        if not detections or not gt_dict["annotations"]:
            return self._empty_result(predictions, references, image_ids)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(detections)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
            coco_eval.params.kpt_oks_sigmas = np.array(sigmas, dtype=np.float64)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_medium": float(stats[3]),
            "AP_large": float(stats[4]),
            "AR": float(stats[5]),
            "AR50": float(stats[6]),
            "AR75": float(stats[7]),
            "num_predictions": len(detections),
            "num_ground_truths": len(gt_dict["annotations"]),
            "num_images": len(image_ids),
        }

    @staticmethod
    def _validate_keypoint_counts(
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
        num_sigmas: int,
    ) -> None:
        """Ensure predictions, references and sigmas describe the same layout.

        OKS is only defined when the model's keypoints match the ground-truth
        keypoint set. A model with a different layout (e.g. SynthPose's 52
        anatomical markers vs COCO's 17) cannot be scored against COCO ground
        truth, so fail early with an actionable message instead of a numpy
        broadcast error inside pycocotools.
        """
        for kind, items in (("prediction", predictions), ("reference", references)):
            for item in items:
                count = len(item["keypoints"]) // 3
                if count != num_sigmas:
                    raise ValueError(
                        f"Keypoint count mismatch: {kind} has {count} keypoints but the "
                        f"metric expects {num_sigmas} (from sigmas). The model's keypoint "
                        f"layout must match the dataset and sigmas. For a non-COCO layout "
                        f"(e.g. SynthPose's 52 markers), pass matching sigmas and "
                        f"keypoint_names and use a dataset with the same keypoint definition."
                    )

    @staticmethod
    def _build_gt_annotations(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert ground-truth instances to COCO annotation dicts."""
        annotations = []
        for i, ref in enumerate(references):
            keypoints = [float(v) for v in ref["keypoints"]]
            num_keypoints = ref.get("num_keypoints")
            if num_keypoints is None:
                # COCO visibility flag is every 3rd value; >0 means labeled.
                num_keypoints = sum(1 for v in keypoints[2::3] if v > 0)
            annotations.append(
                {
                    "id": i + 1,
                    "image_id": int(ref["image_id"]),
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": int(num_keypoints),
                    "bbox": [float(v) for v in ref["bbox"]],
                    "area": float(ref["area"]),
                    "iscrowd": 0,
                }
            )
        return annotations

    @staticmethod
    def _empty_result(
        predictions: list[dict[str, Any]],
        references: list[dict[str, Any]],
        image_ids: list[int],
    ) -> dict[str, float]:
        """Return zeroed metrics when there is nothing to score."""
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR": 0.0,
            "AR50": 0.0,
            "AR75": 0.0,
            "num_predictions": len(predictions),
            "num_ground_truths": len(references),
            "num_images": len(image_ids),
        }
