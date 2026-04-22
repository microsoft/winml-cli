# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model evaluation engine."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..datasets.config import DatasetConfig
from .base_evaluator import WinMLEvaluator
from .config import WinMLEvaluationConfig
from .feature_extraction_evaluator import WinMLFeatureExtractionEvaluator
from .fill_mask_evaluator import WinMLFillMaskEvaluator
from .image_segmentation_evaluator import WinMLImageSegmentationEvaluator
from .object_detection_evaluator import WinMLObjectDetectionEvaluator
from .question_answering_evaluator import WinMLQuestionAnsweringEvaluator
from .text_classification_evaluator import WinMLTextClassificationEvaluator
from .token_classification_evaluator import WinMLTokenClassificationEvaluator


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)

_EVALUATOR_REGISTRY: dict[str, type[WinMLEvaluator]] = {
    "text-classification": WinMLTextClassificationEvaluator,
    "sequence-classification": WinMLTextClassificationEvaluator,
    "next-sentence-prediction": WinMLTextClassificationEvaluator,
    "token-classification": WinMLTokenClassificationEvaluator,
    "object-detection": WinMLObjectDetectionEvaluator,
    "image-segmentation": WinMLImageSegmentationEvaluator,
    "question-answering": WinMLQuestionAnsweringEvaluator,
    "feature-extraction": WinMLFeatureExtractionEvaluator,
    "sentence-similarity": WinMLFeatureExtractionEvaluator,
    "fill-mask": WinMLFillMaskEvaluator,
}

_FE_DEFAULT = DatasetConfig(
    path="mteb/stsbenchmark-sts",
    split="test",
    samples=100,
    shuffle=True,
    streaming=True,
    columns_mapping={
        "input_column_1": "sentence1",
        "input_column_2": "sentence2",
        "score_column": "score",
    },
)

_DEFAULT_DATASETS: dict[str, DatasetConfig] = {
    "image-classification": DatasetConfig(
        path="timm/mini-imagenet",
        split="test",
        samples=100,
        shuffle=True,
    ),
    "text-classification": DatasetConfig(
        path="nyu-mll/glue",
        name="mrpc",
        split="validation",
        samples=100,
        shuffle=True,
        columns_mapping={
            "input_column": "sentence1",
            "second_input_column": "sentence2",
        },
    ),
    "token-classification": DatasetConfig(
        path="BramVanroy/conll2003",
        split="validation",
        samples=100,
        shuffle=True,
        columns_mapping={
            "label_column": "ner_tags",
        },
    ),
    "object-detection": DatasetConfig(
        path="detection-datasets/coco",
        split="val",
        samples=100,
        shuffle=True,
        columns_mapping={
            "annotation_column": "objects",
            "bbox_key": "bbox",
            "category_key": "category",
            "box_format": "xyxy",
        },
    ),
    "question-answering": DatasetConfig(
        path="rajpurkar/squad",
        split="validation",
        samples=100,
        shuffle=True,
        columns_mapping={
            "question_column": "question",
            "context_column": "context",
            "id_column": "id",
            "label_column": "answers",
        },
    ),
    "feature-extraction": _FE_DEFAULT,
    "sentence-similarity": _FE_DEFAULT,
    "fill-mask": DatasetConfig(
        path="Salesforce/wikitext",
        name="wikitext-2-raw-v1",
        split="test",
        samples=100,
        shuffle=True,
        streaming=True,
        columns_mapping={"input_column": "text"},
    ),
}


@dataclass
class EvalResult:
    """Results from model evaluation."""

    config: WinMLEvaluationConfig
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **self.config.to_dict(),
            "metrics": self.metrics,
        }


def _load_model(config: WinMLEvaluationConfig) -> WinMLPreTrainedModel:
    """Load model from ONNX path or HF model ID."""
    from ..models import WinMLAutoModel

    if config.model_id is None:
        raise ValueError("model_id is required.")

    if config.model_path is not None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(config.model_id)
        model = WinMLAutoModel.from_onnx(
            onnx_path=Path(config.model_path),
            task=config.task,
            device=config.device,
            skip_build=True,
        )
        model.config = hf_config
        return model

    return WinMLAutoModel.from_pretrained(
        config.model_id,
        task=config.task,
        device=config.device,
    )


def _resolve_task(config: WinMLEvaluationConfig) -> str:
    """Resolve task from config or model's HF config."""
    if config.task is not None:
        return config.task

    if config.model_id is None:
        raise ValueError("Cannot infer task without model_id. Provide --task.")

    from transformers import AutoConfig

    from ..loader.task import _detect_task_from_config

    hf_config = AutoConfig.from_pretrained(config.model_id)
    return _detect_task_from_config(hf_config)


def evaluate(config: WinMLEvaluationConfig) -> EvalResult:
    """Run model evaluation.

    This function does not mutate the caller's config. It creates internal
    copies via ``dataclasses.replace`` and ``deepcopy`` so the original
    config and any module-level defaults remain untouched.
    """
    config = replace(config, task=_resolve_task(config), dataset=deepcopy(config.dataset))
    model = _load_model(config)

    if config.dataset.path is None:
        default = _DEFAULT_DATASETS.get(config.task)
        if default is None:
            raise ValueError(
                f"No dataset provided and no default for task '{config.task}'. Use --dataset."
            )
        config.dataset = deepcopy(default)
        logger.info(
            "Using default dataset for %s: %s",
            config.task,
            default.path,
        )

    cls = _EVALUATOR_REGISTRY.get(config.task, WinMLEvaluator)
    task_evaluator = cls(config, model)
    metrics = task_evaluator.compute()

    return EvalResult(config=config, metrics=metrics)
