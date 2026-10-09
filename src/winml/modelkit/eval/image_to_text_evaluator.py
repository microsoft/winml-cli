# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Image-to-text evaluator for captioning + OCR models.

Covers the shared ``image-to-text`` pipeline surface used by:

- Captioners (BLIP, vit-gpt2): metric of interest is **CIDEr** against
  multi-reference ground-truth captions.
- OCR-style models (TrOCR, manga-ocr, donut, nougat): metric of interest
  is **CER** against the reference transcription.

Both metrics are reported on every run so the e2e harness can pick the
right one per model via ``winml_metric_key``.

Dataset schema: ``input_column`` for the image, ``label_column`` for
the reference text (str or list[str]).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig


logger = logging.getLogger(__name__)


class WinMLImageToTextEvaluator(WinMLEvaluator):
    """Image-to-text evaluator. Reports CER and CIDEr."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        cm = config.dataset.columns_mapping
        self._image_col = cm.get("input_column", get_default("image-to-text", "input_column"))
        self._label_col = cm.get("label_column", get_default("image-to-text", "label_column"))
        super().__init__(config, model)

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: free-text labels need no ClassLabel alignment."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run the pipeline over each sample and return CER + CIDEr."""
        from tqdm.auto import tqdm

        from .metrics.text_similarity import TextSimilarityMetric

        metric = TextSimilarityMetric()
        skipped = 0

        for sample in tqdm(self.data, desc="Evaluating", unit="sample"):
            image = sample.get(self._image_col)
            references = sample.get(self._label_col)
            if image is None or references is None:
                skipped += 1
                continue

            try:
                out = self.pipe(image)
            except Exception as e:
                logger.warning("Pipeline call failed (skipping): %s", e)
                skipped += 1
                continue

            # HF image-to-text pipeline returns either a list of dicts
            # ([{"generated_text": "..."}]) or a single dict.
            if isinstance(out, list) and out:
                pred = out[0].get("generated_text", "")
            elif isinstance(out, dict):
                pred = out.get("generated_text", "")
            else:
                pred = str(out)

            metric.update(pred.strip(), references)

        result = metric.compute()
        if skipped:
            result["skipped"] = skipped
        return result
