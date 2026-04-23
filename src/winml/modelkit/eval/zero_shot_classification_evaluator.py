# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Zero-shot classification evaluator for NLI checkpoints.

Computes accuracy and macro-F1 via ClassificationMetric.
HF evaluate library has no zero-shot-classification evaluator, so this
class runs the metric loop manually: each text is scored against every
candidate label as a (premise, hypothesis) NLI pair, and the label with
the top entailment score wins.

Candidate labels come from ``columns_mapping["candidate_labels"]`` if set
(comma-separated), otherwise from ``dataset.features[label_column].names``
when the column is a ``ClassLabel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers.pipelines.zero_shot_classification import ZeroShotClassificationPipeline

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig


class _FixedShapeZeroShotPipeline(ZeroShotClassificationPipeline):
    """Pad to ``tokenizer.model_max_length`` for fixed-shape ONNX exports."""

    def _parse_and_tokenize(self, sequence_pairs: Any, **kwargs: Any) -> Any:
        kwargs["padding"] = "max_length"
        kwargs.setdefault("truncation", True)
        return super()._parse_and_tokenize(sequence_pairs, **kwargs)


class WinMLZeroShotClassificationEvaluator(WinMLEvaluator):
    """Evaluator for zero-shot text classification using NLI models."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for zero-shot classification."""
        from .config import SchemaColumn

        return [
            SchemaColumn("text", "Value(string)", "input_column", description="input text"),
            SchemaColumn(
                "label",
                "ClassLabel",
                "label_column",
                description="gold label (ClassLabel or string)",
            ),
            SchemaColumn(
                "<candidate_labels>",
                "comma-separated str",
                "candidate_labels",
                required=False,
                description="override candidate labels (required for non-ClassLabel columns)",
            ),
            SchemaColumn(
                "<hypothesis_template>",
                "Value(string)",
                "hypothesis_template",
                required=False,
                description='NLI hypothesis template (default: "This example is {}.")',
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        mapping = config.dataset.columns_mapping
        self._input_col = mapping.get("input_column", "text")
        self._label_col = mapping.get("label_column", "label")
        self._candidate_labels_override = mapping.get("candidate_labels")
        self._hypothesis_template = mapping.get("hypothesis_template")
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline with fixed-length tokenization for ONNX."""
        from transformers import pipeline

        io_config = getattr(self.model, "io_config", None) or {}
        shapes = io_config.get("input_shapes", [[]])
        max_length: int | None = None
        if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
            max_length = shapes[0][1]

        pipe = pipeline(
            "zero-shot-classification",
            model=self.model,
            framework="pt",
            tokenizer=self.config.model_id,
            device="cpu",
            pipeline_class=_FixedShapeZeroShotPipeline,
        )

        if pipe.tokenizer is not None and max_length is not None:
            pipe.tokenizer.model_max_length = max_length

            # Drop tokenizer keys the ONNX graph does not accept
            # (e.g. DeBERTa-v3 MNLI exports omit token_type_ids).
            input_names = io_config.get("input_names", [])
            if input_names:
                filtered = [n for n in pipe.tokenizer.model_input_names if n in input_names]
                if filtered:
                    pipe.tokenizer.model_input_names = filtered

        return pipe

    def align_labels(
        self,
        dataset: Dataset,
        ds_config: DatasetConfig,
    ) -> Dataset:
        """Validate input and label columns.

        Base-class label alignment is bypassed: NLI ``label2id`` identifies
        entailment/neutral/contradiction classes, which are unrelated to the
        ground-truth labels used for accuracy.
        """
        col_names = set(dataset.column_names)
        for col in (self._input_col, self._label_col):
            if col not in col_names:
                raise ValueError(f"Column '{col}' not found in dataset: {sorted(col_names)}")
        return dataset

    def _resolve_candidate_labels(self, dataset: Dataset) -> list[str]:
        """Return candidate labels from user override or dataset ``ClassLabel``."""
        if self._candidate_labels_override:
            labels = [s.strip() for s in self._candidate_labels_override.split(",") if s.strip()]
            if not labels:
                raise ValueError("candidate_labels override must not be empty.")
            return labels

        names = getattr(dataset.features.get(self._label_col), "names", None)
        if names:
            return list(names)

        raise ValueError(
            f"Column '{self._label_col}' is not a ClassLabel; pass "
            f'--column "candidate_labels=a,b,...".',
        )

    def compute(self) -> dict[str, Any]:
        """Compute accuracy and macro-F1 over all samples."""
        from .metrics import ClassificationMetric

        candidate_labels = self._resolve_candidate_labels(self.data)
        class_names = getattr(self.data.features.get(self._label_col), "names", None)

        pipe_kwargs: dict[str, Any] = {"candidate_labels": candidate_labels}
        if self._hypothesis_template is not None:
            pipe_kwargs["hypothesis_template"] = self._hypothesis_template

        predictions: list[str] = []
        references: list[str] = []
        for sample in self.data:
            result = self.pipe(sample[self._input_col], **pipe_kwargs)
            predictions.append(result["labels"][0])
            raw = sample[self._label_col]
            references.append(class_names[int(raw)] if class_names else str(raw))

        return ClassificationMetric().compute(predictions, references, candidate_labels)
