# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Zero-shot classification evaluator for NLI checkpoints.

Computes accuracy and macro-F1. The HF evaluate library has no
zero-shot-classification evaluator, so this class runs the metric loop
manually: each text is scored against every candidate label as a
(premise, hypothesis) NLI pair, and the label with the top entailment
score wins.

Candidate labels come from ``columns_mapping["candidate_labels"]`` if set
(comma-separated), otherwise from ``dataset.features[label_column].names``
when the column is a ``ClassLabel``. An override on a ``ClassLabel`` column
must list one label per class, in order, and replaces the class names
positionally for both predictions and references.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


try:
    from transformers.pipelines.zero_shot_classification import ZeroShotClassificationPipeline
except Exception:  # pragma: no cover - import fallback for constrained environments
    class ZeroShotClassificationPipeline:  # type: ignore[no-redef]
        """Fallback zero-shot pipeline type when transformers is unavailable."""


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig


class _FixedShapeZeroShotPipeline(ZeroShotClassificationPipeline):
    """Resize tokenized pairs for fixed-shape ONNX exports.

    Delegates padding/truncation to the owning ``WinMLEvaluator`` (set as
    ``_winml_evaluator`` after construction).
    """

    _winml_evaluator: WinMLEvaluator | None = None

    def _parse_and_tokenize(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("padding", True)
        kwargs.setdefault("truncation", True)
        encoding = super()._parse_and_tokenize(*args, **kwargs)
        if self._winml_evaluator is None or self.tokenizer is None:
            return encoding
        return self._winml_evaluator._pad_or_truncate(encoding, self.tokenizer)


class WinMLZeroShotClassificationEvaluator(WinMLEvaluator):
    """Evaluator for zero-shot text classification using NLI models."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "zero-shot-classification"
        self._input_col = mapping.get("input_column", get_default(task, "input_column"))
        self._label_col = mapping.get("label_column", get_default(task, "label_column"))
        self._input_pair_col = mapping.get("input_pair_column")
        self._candidate_labels_override = mapping.get("candidate_labels")
        self._hypothesis_template = mapping.get("hypothesis_template")
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline with fixed-length tokenization for ONNX."""
        from transformers import pipeline

        if self._input_pair_col:
            return self._prepare_nli_pipeline()

        max_length = self._fixed_seq_length()

        # WinMLPreTrainedModel isn't in transformers' Pipeline model union;
        # the pipeline_class override is also outside the Literal overloads.
        pipeline_kwargs: dict[str, Any] = {}
        if self.config.trust_remote_code:
            pipeline_kwargs["trust_remote_code"] = True
        pipe = cast(
            "_FixedShapeZeroShotPipeline",
            pipeline(
                "zero-shot-classification",
                model=self.model,
                tokenizer=self.config.model_id,
                device=self.config.pipeline_device,
                pipeline_class=_FixedShapeZeroShotPipeline,
                **pipeline_kwargs,
            ),
        )
        pipe._winml_evaluator = self

        if pipe.tokenizer is not None and max_length is not None:
            pipe.tokenizer.model_max_length = max_length

            # Drop tokenizer keys the ONNX graph does not accept
            # (some NLI exports omit token_type_ids).
            io_config = getattr(self.model, "io_config", None) or {}
            input_names = io_config.get("input_names", [])
            if input_names:
                filtered = [n for n in pipe.tokenizer.model_input_names if n in input_names]
                if filtered:
                    pipe.tokenizer.model_input_names = filtered

        return cast("Pipeline", pipe)

    def _prepare_nli_pipeline(self) -> Pipeline:
        """Create sequence-classification pipeline for text-pair NLI eval."""
        from transformers import pipeline

        max_length = self._fixed_seq_length()

        pipeline_kwargs: dict[str, Any] = {}
        if self.config.trust_remote_code:
            pipeline_kwargs["trust_remote_code"] = True
        pipe = cast(
            "Pipeline",
            pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.config.model_id,
                device=self.config.pipeline_device,
                **pipeline_kwargs,
            ),
        )

        if pipe.tokenizer is not None and max_length is not None:
            pipe.tokenizer.model_max_length = max_length
            pipe._preprocess_params.setdefault("padding", "max_length")
            pipe._preprocess_params.setdefault("max_length", max_length)
            pipe._preprocess_params.setdefault("truncation", True)

            io_config = getattr(self.model, "io_config", None) or {}
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
        required_cols = [self._input_col, self._label_col]
        if self._input_pair_col:
            required_cols.append(self._input_pair_col)
        for col in required_cols:
            if col not in col_names:
                raise DatasetValidationError(
                    f"Column '{col}' not found in dataset: {sorted(col_names)}",
                )
        return dataset

    @staticmethod
    def _normalize_nli_label(label: str) -> str | None:
        """Map label text to one of entailment/contradiction/neutral."""
        token = re.sub(r"[^a-z]+", "_", label.strip().lower()).strip("_")
        if token in {"entailment", "contradiction", "neutral"}:
            return token
        return None

    def _resolve_reference_label(
        self,
        sample: dict[str, Any],
        class_names: list[str] | None,
    ) -> str:
        """Convert dataset label value to canonical NLI class name."""
        label_col = self._label_col
        if label_col is None:
            raise DatasetValidationError("label column is not configured for evaluation.")
        raw = sample[label_col]
        name = class_names[int(raw)] if class_names is not None else str(raw)
        normalized = self._normalize_nli_label(name)
        if normalized is None:
            raise DatasetValidationError(
                f"Reference label '{name}' is not one of entailment/contradiction/neutral.",
            )
        return normalized

    def _resolve_prediction_label(self, result: Any) -> str:
        """Pick top model label from pipeline output and canonicalize it."""
        candidates = result
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
            candidates = candidates[0]
        if isinstance(candidates, dict):
            candidates = [candidates]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Unexpected pipeline output for NLI zero-shot evaluation.")

        best = max(candidates, key=lambda item: float(item.get("score", 0.0)))
        raw_label = str(best.get("label", ""))
        normalized = self._normalize_nli_label(raw_label)
        if normalized is None:
            raise ValueError(
                "NLI zero-shot evaluation requires model labels to include "
                "entailment/contradiction/neutral.",
            )
        return normalized

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

        raise DatasetValidationError(
            f"Column '{self._label_col}' is not a ClassLabel; pass "
            f'--column "candidate_labels=a,b,...".',
        )

    def compute(self) -> dict[str, Any]:
        """Compute accuracy and macro-F1 over all samples."""
        from .metrics import ClassificationMetric

        if self._input_pair_col:
            class_names = getattr(self.data.features.get(self._label_col), "names", None)
            labels = ["entailment", "contradiction", "neutral"]
            pair_predictions: list[str] = []
            pair_references: list[str] = []
            for sample in tqdm(self.data, desc="Evaluating zero-shot (NLI pairs)"):
                result = self.pipe(
                    sample[self._input_col],
                    text_pair=sample[self._input_pair_col],
                    top_k=None,
                )
                pair_predictions.append(self._resolve_prediction_label(result))
                pair_references.append(self._resolve_reference_label(sample, class_names))
            return ClassificationMetric().compute(pair_predictions, pair_references, labels)

        candidate_labels = self._resolve_candidate_labels(self.data)
        class_names = getattr(self.data.features.get(self._label_col), "names", None)

        # An override replaces ClassLabel.names positionally, so references use
        # the same vocabulary as predictions.
        if self._candidate_labels_override and class_names is not None:
            if len(candidate_labels) != len(class_names):
                raise ValueError(
                    f"candidate_labels override has {len(candidate_labels)} entries "
                    f"but dataset ClassLabel has {len(class_names)}; provide one "
                    f"override label per class, in order.",
                )
            class_names = candidate_labels

        pipe_kwargs: dict[str, Any] = {"candidate_labels": candidate_labels}
        if self._hypothesis_template is not None:
            pipe_kwargs["hypothesis_template"] = self._hypothesis_template

        candidate_predictions: list[str] = []
        candidate_references: list[str] = []
        for sample in tqdm(self.data, desc="Evaluating zero-shot (accuracy)"):
            result = self.pipe(sample[self._input_col], **pipe_kwargs)
            candidate_predictions.append(result["labels"][0])
            raw = sample[self._label_col]
            candidate_references.append(class_names[int(raw)] if class_names else str(raw))

        return ClassificationMetric().compute(
            candidate_predictions,
            candidate_references,
            candidate_labels,
        )
