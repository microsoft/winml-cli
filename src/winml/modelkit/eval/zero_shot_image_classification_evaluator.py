# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Zero-shot image classification evaluator using top-k accuracy.

Evaluates CLIP/SigLIP models by classifying images into dataset-defined
categories using the HF ZeroShotImageClassificationPipeline.

The evaluator works with both HF PyTorch models and WinML ONNX models.
The WinML composite model class handles split-encoder orchestration
internally, so the evaluator just uses the HF pipeline uniformly.

Metric: top-1 and top-5 accuracy via TopKAccuracyMetric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

class WinMLZeroShotImageClassificationEvaluator(WinMLEvaluator):
    """Evaluator for zero-shot image classification using top-k accuracy."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for zero-shot image classification."""
        from .config import SchemaColumn

        return [
            SchemaColumn("image", "Image", "input_column", description="PIL Image"),
            SchemaColumn(
                "label", "ClassLabel", "label_column", description="integer class label"
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        ds = config.dataset
        self._image_col = ds.columns_mapping.get("input_column", "image")
        self._label_col = ds.columns_mapping.get("label_column", "label")
        super().__init__(config, model)
        self._candidate_labels = self._resolve_class_names()

    def _resolve_class_names(self) -> list[str]:
        """Extract class name strings from dataset ClassLabel feature.

        Underscores in raw dataset labels (e.g. CIFAR-100 ``aquarium_fish``)
        tokenize poorly under CLIP's BPE vocabulary and depress zero-shot
        accuracy by ~10pp; published CLIP evaluations replace them with spaces.
        """
        from datasets import ClassLabel

        features = self.data.features
        label_feature = features.get(self._label_col)
        if isinstance(label_feature, ClassLabel):
            return [name.replace("_", " ") for name in label_feature.names]

        raise ValueError(
            f"Dataset column '{self._label_col}' is not a ClassLabel feature. "
            f"Cannot resolve class names for zero-shot classification. "
            f"Feature type: {type(label_feature)}"
        )

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: zero-shot models have no fixed label vocabulary."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run zero-shot classification evaluation and return top-k accuracy."""
        from tqdm.auto import tqdm

        from .metrics.top_k_accuracy import TopKAccuracyMetric

        metric = TopKAccuracyMetric()

        for sample in tqdm(self.data, desc="Evaluating", unit="sample"):
            image = sample.get(self._image_col)
            label_idx = sample.get(self._label_col)
            if image is None or label_idx is None:
                continue

            results = self.pipe(image, candidate_labels=self._candidate_labels)
            pred_labels = [r["label"] for r in results]

            metric.update(pred_labels, self._candidate_labels[label_idx])

        return metric.compute()
