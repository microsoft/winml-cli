# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model evaluation engine."""

from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from rich.console import Console

from .config import WinMLEvaluationConfig


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from .base_evaluator import WinMLEvaluator

logger = logging.getLogger(__name__)

# Map task -> "module_path:ClassName"; modules are imported lazily by
# get_evaluator_class() to improve command latency.
# Keep the key/value-per-line layout: collapsing each entry onto one line (the
# default formatter layout) yields >100-char lines that trip E501.
# fmt: off
_EVALUATOR_REGISTRY: dict[str, str] = {
    "image-classification":
        "winml.modelkit.eval.base_evaluator:WinMLEvaluator",
    "text-classification":
        "winml.modelkit.eval.text_classification_evaluator:WinMLTextClassificationEvaluator",
    "sequence-classification":
        "winml.modelkit.eval.text_classification_evaluator:WinMLTextClassificationEvaluator",
    "next-sentence-prediction":
        "winml.modelkit.eval.text_classification_evaluator:WinMLTextClassificationEvaluator",
    "token-classification":
        "winml.modelkit.eval.token_classification_evaluator:WinMLTokenClassificationEvaluator",
    "object-detection":
        "winml.modelkit.eval.object_detection_evaluator:WinMLObjectDetectionEvaluator",
    "image-segmentation":
        "winml.modelkit.eval.image_segmentation_evaluator:WinMLImageSegmentationEvaluator",
    "question-answering":
        "winml.modelkit.eval.question_answering_evaluator:WinMLQuestionAnsweringEvaluator",
    "feature-extraction":
        "winml.modelkit.eval.feature_extraction_evaluator:WinMLFeatureExtractionEvaluator",
    "sentence-similarity":
        "winml.modelkit.eval.feature_extraction_evaluator:WinMLFeatureExtractionEvaluator",
    "image-feature-extraction":
        "winml.modelkit.eval.image_feature_extraction_evaluator:WinMLImageFeatureExtractionEvaluator",
    "image-to-text":
        "winml.modelkit.eval.image_to_text_evaluator:WinMLImageToTextEvaluator",
    "fill-mask":
        "winml.modelkit.eval.fill_mask_evaluator:WinMLFillMaskEvaluator",
    "zero-shot-classification":
        "winml.modelkit.eval.zero_shot_classification_evaluator:WinMLZeroShotClassificationEvaluator",
    "zero-shot-image-classification":
        "winml.modelkit.eval.zero_shot_image_classification_evaluator:WinMLZeroShotImageClassificationEvaluator",
    "depth-estimation":
        "winml.modelkit.eval.depth_estimation_evaluator:WinMLDepthEstimationEvaluator",
    "compare-tensor":
        "winml.modelkit.eval.tensor_similarity_evaluator:TensorSimilarityEvaluator",
}
# fmt: on


def get_evaluator_class(config: WinMLEvaluationConfig) -> type[WinMLEvaluator]:
    """Return the evaluator class for *task*, or raise ValueError if unsupported."""
    key = "compare-tensor" if config.mode == "compare" else config.task
    spec = _EVALUATOR_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(sorted(_EVALUATOR_REGISTRY))
        raise ValueError(
            f"Task '{key}' is not supported by `winml eval`. Supported tasks: {supported}."
        )
    module_path, class_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


_FE_DEFAULT = {
    "path": "mteb/stsbenchmark-sts",
    "split": "test",
    "streaming": True,
    "columns_mapping": {
        "input_column_1": "sentence1",
        "input_column_2": "sentence2",
        "score_column": "score",
    },
}

_DEFAULT_DATASETS: dict[str, dict] = {
    "image-classification": {
        "path": "timm/mini-imagenet",
        "split": "test",
    },
    "text-classification": {
        "path": "nyu-mll/glue",
        "name": "mrpc",
        "split": "validation",
        "columns_mapping": {
            "input_column": "sentence1",
            "second_input_column": "sentence2",
        },
    },
    "token-classification": {
        "path": "BramVanroy/conll2003",
        "split": "validation",
        "columns_mapping": {
            "label_column": "ner_tags",
        },
    },
    "object-detection": {
        "path": "detection-datasets/coco",
        "split": "val",
        "columns_mapping": {
            "annotation_column": "objects",
            "bbox_key": "bbox",
            "category_key": "category",
            "box_format": "xyxy",
        },
    },
    "question-answering": {
        "path": "rajpurkar/squad",
        "split": "validation",
        "columns_mapping": {
            "question_column": "question",
            "context_column": "context",
            "id_column": "id",
            "label_column": "answers",
        },
    },
    "feature-extraction": dict(_FE_DEFAULT),
    "sentence-similarity": dict(_FE_DEFAULT),
    "image-feature-extraction": {
        "path": "timm/mini-imagenet",
        "split": "test",
    },
    "fill-mask": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "streaming": True,
        "columns_mapping": {"input_column": "text"},
    },
    "zero-shot-classification": {
        "path": "fancyzhx/ag_news",
        "split": "test",
        "columns_mapping": {
            "input_column": "text",
            "label_column": "label",
            "candidate_labels": "World,Sports,Business,Sci/Tech",
            "hypothesis_template": "This text is about {}.",
        },
    },
    "zero-shot-image-classification": {
        "path": "uoft-cs/cifar100",
        "split": "test",
        "columns_mapping": {
            "input_column": "img",
            "label_column": "fine_label",
        },
    },
    "depth-estimation": {
        "path": "sayakpaul/nyu_depth_v2",
        "split": "validation",
        # Loaded via the parquet-mirror revision so the dataset works without
        # the legacy `nyu_depth_v2.py` loader script.
        "revision": "refs/convert/parquet",
    },
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
        # Pre-built ONNX: precision is already baked into the model and is
        # ignored here (mirrors winml perf's ONNX path).
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(config.model_id)
        model = WinMLAutoModel.from_onnx(
            onnx_path=config.model_path,
            task=config.task,
            device=config.device,
            ep=config.ep,
            skip_build=config.skip_build,
            hf_config=hf_config,
        )
        model.config = hf_config
        return model

    return WinMLAutoModel.from_pretrained(
        config.model_id,
        task=config.task,
        device=config.device,
        precision=config.precision,
        ep=config.ep,
        allow_unsupported_nodes=config.allow_unsupported_nodes,
    )


def _resolve_task(config: WinMLEvaluationConfig) -> str:
    """Resolve the eval task and validate it is supported.

    An explicit ``config.task`` is surfaced verbatim (explicit means explicit).
    When omitted, the modality-aware :func:`detect_task` infers it from the model's
    HF config — an image-embedding model resolves to ``image-feature-extraction``
    (not the lossy ``feature-extraction``), so the evaluator-registry lookup picks
    the image evaluator without any reverse io_config reconstruction.
    """
    console = Console()
    console.print("[bold]Resolving task...[/bold]")

    if config.task is not None:
        task = config.task
    else:
        if config.model_id is None:
            raise ValueError("Cannot infer task without model_id. Provide --task.")

        from transformers import AutoConfig

        from ..loader.resolution import resolve_task

        hf_config = AutoConfig.from_pretrained(config.model_id)
        task = resolve_task(hf_config).task

    console.print(f"[dim]Use[/dim] {task} [dim]to evaluate[/dim]")

    if task not in _EVALUATOR_REGISTRY:
        supported = ", ".join(sorted(_EVALUATOR_REGISTRY))
        raise ValueError(f"Task '{task}' is not supported. Supported tasks: {supported}.")
    return task


def evaluate(config: WinMLEvaluationConfig) -> EvalResult:
    """Run model evaluation.

    This function does not mutate the caller's config. It creates internal
    copies via ``dataclasses.replace`` and ``deepcopy`` so the original
    config and any module-level defaults remain untouched.
    """
    from ..utils.eval_utils import EVAL_MODES

    mode = config.mode if config.mode is not None else "onnx"
    if mode not in EVAL_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {EVAL_MODES} or None.")
    config = replace(
        config, mode=mode, task=_resolve_task(config), dataset=deepcopy(config.dataset)
    )
    if config.mode != "compare" and config.dataset.path is None:
        default = _DEFAULT_DATASETS.get(config.task)
        if default is None:
            raise ValueError(
                f"No dataset provided and no default for task '{config.task}'. Use --dataset."
            )
        for k, v in default.items():
            setattr(config.dataset, k, deepcopy(v))
        logger.warning(
            "--dataset not specified; attempting default dataset '%s' for task '%s'. "
            "Any --split / --column / --streaming / --dataset-name options are ignored.",
            config.dataset.path,
            config.task,
        )

    print_config(config)
    console = Console()

    console.print("\n[bold]Loading model...[/bold]")
    try:
        model = _load_model(config)
    except Exception as error:
        raise ValueError(
            f"Failed to load model '{config.model_id}'. "
            "Check --model, --model-id, --task, device, and EP settings. "
            f"For composite models, run 'winml eval --schema --task {config.task}' "
            "to see supported role=path model options.",
        ) from error

    from ..utils.eval_utils import DatasetValidationError

    cls = get_evaluator_class(config)
    try:
        console.print("[bold]Loading dataset and evaluating...[/bold]")
        task_evaluator = cls(config, model)
        metrics = task_evaluator.compute()
    except DatasetValidationError as error:
        raise ValueError(
            f"Dataset '{config.dataset.path}' is not compatible with task "
            f"'{config.task}': {error}. Use --dataset to specify a different dataset, "
            f"or run 'winml eval --schema --task {config.task}' to see the expected schema.",
        ) from error
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Failed to compute metrics for task '{config.task}' on dataset "
            f"'{config.dataset.path}'. "
            f"Run 'winml eval --schema --task {config.task}' to see the expected schema.",
        ) from error

    return EvalResult(config=config, metrics=metrics)


def print_config(config: WinMLEvaluationConfig) -> None:
    """Print effective evaluation config to the console (quantize.py style)."""
    ds = config.dataset
    output_console = Console()
    output_console.print(f"[bold blue]Model:[/bold blue] {config.model_id}")
    if config.model_path is not None:
        output_console.print(f"[bold blue]Model path:[/bold blue] {config.model_path}")
    output_console.print(f"[bold blue]Task:[/bold blue] {config.task}")
    output_console.print(f"[bold blue]Device:[/bold blue] {config.device}")
    if config.ep is not None:
        output_console.print(f"[bold blue]EP:[/bold blue] {config.ep}")
    output_console.print(f"[bold blue]Precision:[/bold blue] {config.precision}")
    if config.mode != "compare":
        output_console.print(f"[bold blue]Dataset:[/bold blue] {ds.path}")
        if ds.name:
            output_console.print(f"[bold blue]Dataset name:[/bold blue] {ds.name}")
        output_console.print(f"[bold blue]Split:[/bold blue] {ds.split}")
        output_console.print(f"[bold blue]Samples:[/bold blue] {ds.samples}")
        output_console.print(f"[bold blue]Shuffle:[/bold blue] {ds.shuffle} (seed={ds.seed})")
        output_console.print(f"[bold blue]Streaming:[/bold blue] {ds.streaming}")
        if ds.columns_mapping:
            cols = ", ".join(f"{k}={v}" for k, v in ds.columns_mapping.items())
            output_console.print(f"[bold blue]Columns:[/bold blue] {cols}")
    if config.output_path is not None:
        output_console.print(f"[bold blue]Output:[/bold blue] {config.output_path}")
