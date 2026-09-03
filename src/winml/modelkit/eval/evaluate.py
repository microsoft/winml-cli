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
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console

from .config import WinMLEvaluationConfig


if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from ..models.winml import HFCausalLM
    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel
    from ..models.winml.genai_causal_lm import WinMLGenaiCausalLM
    from .base_evaluator import WinMLEvaluator

logger = logging.getLogger(__name__)


class _ModelLoaderKind(Enum):
    PYTORCH = auto()
    GENAI = auto()
    EVALUATOR_MANAGED = auto()
    ONNX = auto()
    PRETRAINED = auto()


def _select_model_loader(config: WinMLEvaluationConfig) -> _ModelLoaderKind:
    """Select the model-loading path shared by loading and CLI diagnostics."""
    if config.runtime == "pytorch":
        return _ModelLoaderKind.PYTORCH
    if config.task == "text-generation":
        return _ModelLoaderKind.GENAI
    if isinstance(config.model_path, dict) and config.task == "mask-generation":
        return _ModelLoaderKind.EVALUATOR_MANAGED
    if config.model_path is not None:
        return _ModelLoaderKind.ONNX
    return _ModelLoaderKind.PRETRAINED


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
    "keypoint-detection":
        "winml.modelkit.eval.keypoint_detection_evaluator:WinMLKeypointDetectionEvaluator",
    "compare-tensor":
        "winml.modelkit.eval.tensor_similarity_evaluator:TensorSimilarityEvaluator",
    "mask-generation":
        "winml.modelkit.eval.mask_generation_evaluator:WinMLMaskGenerationEvaluator",
    "text-generation":
        "winml.modelkit.eval.text_generation_evaluator:WinMLTextGenerationEvaluator",
    "automatic-speech-recognition":
        "winml.modelkit.eval.ctc_asr_evaluator:WinMLCTCASREvaluator",
}
# fmt: on


def get_evaluator_class(config: WinMLEvaluationConfig) -> type[WinMLEvaluator]:
    """Return the evaluator class for *task*, or raise ValueError if unsupported."""
    key = "compare-tensor" if config.mode == "compare" else config.task
    spec = _EVALUATOR_REGISTRY.get(key) if key is not None else None
    if spec is None:
        supported = ", ".join(sorted(_EVALUATOR_REGISTRY))
        raise ValueError(
            f"Task '{key}' is not supported by `winml eval`. Supported tasks: {supported}."
        )
    module_path, class_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return cast("type[WinMLEvaluator]", getattr(module, class_name))


def _validate_pytorch_runtime_config(config: WinMLEvaluationConfig) -> None:
    """Validate state that cannot apply to the PyTorch runtime."""
    if config.runtime == "winml-ort":
        return

    incompatible: list[str] = []
    mode = config.mode if config.mode is not None else "onnx"
    if mode != "onnx":
        incompatible.append("mode")
    if config.model_path is not None:
        incompatible.append("model_path")
    if config.input_data is not None:
        incompatible.append("input_data")
    if config.reference_path is not None:
        incompatible.append("reference_path")
    if config.ep is not None:
        incompatible.append("ep")
    if config.precision != "auto":
        incompatible.append("precision")
    if not config.quant:
        incompatible.append("quant")
    if not config.optimize:
        incompatible.append("optimize")
    if not config.analyze:
        incompatible.append("analyze")
    if config.max_optim_iterations is not None:
        incompatible.append("max_optim_iterations")
    if config.shape_config is not None:
        incompatible.append("shape_config")
    if config.export_overrides is not None:
        incompatible.append("export_overrides")
    if config.allow_unsupported_nodes:
        incompatible.append("allow_unsupported_nodes")
    if not config.skip_build:
        incompatible.append("skip_build")
    if not config.use_cache:
        incompatible.append("use_cache")
    if config.rebuild:
        incompatible.append("rebuild")
    if incompatible:
        raise ValueError(
            f"The PyTorch runtime cannot use WinML-only configuration: {', '.join(incompatible)}."
        )


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
    "keypoint-detection": {
        # Built locally by scripts/build_coco_keypoints.py (COCO has no
        # script-free HF mirror for person keypoints). Run that script first,
        # or pass --dataset-path to point at your own build.
        "path": "~/.cache/winml/datasets/coco_keypoints_val2017",
        "split": "validation",
        "columns_mapping": {
            "input_column": "image",
            "annotation_column": "objects",
            "keypoints_key": "keypoints",
            "bbox_key": "bbox",
            "area_key": "area",
            "box_format": "xywh",
        },
    },
    "mask-generation": {
        # LIP-derived multi-class body-part labels, collapsed to a single
        # binary foreground/background mask by ``MaskGenerationDataset``.
        # Same dataset used by ``scripts/sam3_smoke_eval.py``.
        "path": "mattmdjaga/human_parsing_dataset",
        "split": "train",
    },
    "text-generation": {
        # Raw wikitext-2 test split scored token-by-token for perplexity;
        # ``input_column`` names the text field the corpus is built from.
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "columns_mapping": {"input_column": "text"},
    },
    "automatic-speech-recognition": {
        "path": "google/fleurs",
        "name": "en_us",
        "split": "validation",
        "samples": 2,
        "shuffle": False,
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "columns_mapping": {
            "input_column": "audio",
            "label_column": "transcription",
        },
    },
}


@dataclass
class EvalResult:
    """Results from model evaluation."""

    config: WinMLEvaluationConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    # Effective number of samples actually run, when it differs from
    # ``config.dataset.samples`` (e.g. ``--mode compare --input-data`` derives
    # it from the archive). ``None`` means "use ``config.dataset.samples``".
    num_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            **self.config.to_dict(),
            "metrics": self.metrics,
        }
        # Reflect the real sample count without mutating the config: override the
        # serialized dataset's ``samples`` so the JSON matches the console report.
        if self.num_samples is not None and isinstance(result.get("dataset"), dict):
            result["dataset"] = {**result["dataset"], "samples": self.num_samples}
        return result


def load_model(
    config: WinMLEvaluationConfig,
    pytorch_model: nn.Module | None = None,
    *,
    torch_dtype: Any = "auto",
) -> (
    nn.Module | HFCausalLM | WinMLPreTrainedModel | WinMLCompositeModel | WinMLGenaiCausalLM | None
):
    """Load model from ONNX path or HF model ID.

    For evaluators that handle their own ORT session construction from a
    composite ``role=path`` model dict (currently only
    ``mask-generation``), returns ``None`` -- the evaluator reads
    ``config.model_path`` directly.  Going through ``WinMLAutoModel``'s
    composite registry would require registering the model type (e.g.,
    SAM 3), which is a heavier follow-up; this bypass lets standalone
    ONNX exports be evaluated today.
    """
    from ..models import WinMLAutoModel
    from ..session import EPDeviceTarget, WinMLEPRegistry, resolve_device
    from ..utils import cli as cli_utils

    loader = _select_model_loader(config)
    if loader is _ModelLoaderKind.PYTORCH:
        if config.model_id is None:
            raise ValueError("model_id is required for native Hugging Face evaluation.")
        if pytorch_model is not None:
            from ..loader.native import _adapt_native_hf_model

            return _adapt_native_hf_model(
                config.model_id,
                pytorch_model,
                task=config.task,
                device=config.pipeline_device,
                trust_remote_code=config.trust_remote_code,
            )
        from ..loader import load_native_hf_model

        loaded = load_native_hf_model(
            config.model_id,
            task=config.task,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        config.device = loaded.device.name
        return loaded.model

    if loader is _ModelLoaderKind.GENAI:
        return _load_genai_causal_lm(config)

    quant_override: Any = None
    if not config.quant:
        from ..config import WinMLBuildConfig

        quant_override = WinMLBuildConfig()
        quant_override.quant = None

    pipeline_kwargs = cli_utils.build_pipeline_extra_kwargs(
        optimize=config.optimize,
        analyze=config.analyze,
        max_optim_iterations=config.max_optim_iterations,
    )
    cache_kwargs = cli_utils.cache_extra_kwargs(
        use_cache=config.use_cache,
        rebuild=config.rebuild,
    )

    model_id = config.model_id
    if model_id is None and loader is not _ModelLoaderKind.ONNX:
        raise ValueError("model_id is required.")

    if loader is _ModelLoaderKind.EVALUATOR_MANAGED:
        # Evaluator-driven session loading; skip WinMLAutoModel entirely.
        return None

    # Resolve EPDeviceTarget then bind a WinMLEPDevice at the boundary. Eval
    # config carries an optional ep field; resolve_device deduces device/ep
    # when either is 'auto'.
    device = (config.device or "auto").lower()
    target = resolve_device(EPDeviceTarget(ep=config.ep or "auto", device=device))
    ep_device = WinMLEPRegistry.instance().auto_device(target)

    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    try:
        if loader is _ModelLoaderKind.ONNX:
            # Pre-built ONNX: precision is already baked into the model and is
            # ignored here (mirrors winml perf's ONNX path).
            from transformers import AutoConfig

            from ..loader import load_hf_config

            hf_config = (
                load_hf_config(
                    AutoConfig,
                    model_id,
                    trust_remote_code=config.trust_remote_code,
                )
                if model_id is not None
                else None
            )
            model = WinMLAutoModel.from_onnx(
                # ``model_path`` is narrowed to ``str | dict[str, str]`` here;
                # cast bridges dict value-type invariance (str vs str | Path).
                onnx_path=cast("str | dict[str, str | Path]", config.model_path),
                ep_device=ep_device,
                task=config.task,
                skip_build=config.skip_build,
                config=quant_override,
                hf_config=hf_config,
                **cache_kwargs,
                **pipeline_kwargs,
            )
            model.config = hf_config
            return model

        assert model_id is not None

        # HuggingFace build path — export overrides (--input-specs/
        # --export-config/--dynamic-axes) are merged under the build config's
        # ``export`` section as a sparse dict so from_pretrained routes them
        # through merge_export_overrides (patching auto-resolved input_tensors
        # by name / re-deriving dynamic_axes). Passing a dict rather than a
        # WinMLBuildConfig avoids clobbering the auto-resolved export config
        # with default fields. Mirrors winml build/perf.
        build_override: Any = quant_override
        if config.export_overrides:
            override_dict: dict[str, Any] = {"export": config.export_overrides}
            if not config.quant:
                override_dict["quant"] = None
            build_override = override_dict

        return WinMLAutoModel.from_pretrained(
            model_id,
            ep_device,
            task=config.task,
            device=config.device,
            ep=config.ep,
            precision=config.precision,
            allow_unsupported_nodes=config.allow_unsupported_nodes,
            config=build_override,
            shape_config=config.shape_config,
            **cache_kwargs,
            **pipeline_kwargs,
        )
    except RuntimeException as error:
        auto_device = (
            config._auto_device_selected or config.device is None or config.device.lower() == "auto"
        )
        auto_ep = config.ep is None or config.ep.lower() == "auto"
        if not (auto_device and auto_ep) or target.device.lower() == "cpu":
            raise
        logger.warning(
            "Automatically selected %s on %s could not initialize an ORT session: %s. "
            "Retrying with CPUExecutionProvider.",
            target.ep,
            target.device,
            error,
        )
        config.device = "cpu"
        config.ep = "cpu"
        config._auto_device_selected = False
        return load_model(config, torch_dtype=torch_dtype)


def _load_genai_causal_lm(config: WinMLEvaluationConfig) -> WinMLGenaiCausalLM:
    """Load a causal LM from an onnxruntime-genai bundle directory.

    ``-m <bundle_dir>`` resolves to ``config.model_path`` (a local directory),
    so the bundle directory is read from there. ``ep`` / ``device`` pass straight
    through: an explicit ``--ep`` (or an explicit ``--device`` resolved to an EP
    by the CLI) forces the whole decoder pipeline onto that EP, while ``ep=None``
    respects the bundle's ``genai_config.json`` routing. The session compiles the
    bundle to EPContext when needed (reusing a cached ``_compiled/``), matching
    the safety path ``winml perf`` relies on.

    Raises:
        ValueError: no bundle directory was provided, the path is not a
            directory, or it is missing the ``genai_config.json`` / ONNX files
            that mark a genai bundle.
    """
    from pathlib import Path

    from ..models.winml.genai_causal_lm import WinMLGenaiCausalLM

    bundle_path = config.model_path
    if not bundle_path or isinstance(bundle_path, dict):
        raise ValueError(
            "text-generation evaluation requires a genai bundle *directory* via -m <bundle_dir>."
        )

    bundle_dir = Path(bundle_path).expanduser()
    if not bundle_dir.is_dir():
        raise ValueError(f"Genai bundle directory not found: {bundle_dir}")
    if not (bundle_dir / "genai_config.json").is_file():
        raise ValueError(
            f"'{bundle_dir}' is not a genai bundle: no genai_config.json found. "
            "Point -m at a bundle built with 'winml build ... --device npu --ep qnn'."
        )
    if not any(bundle_dir.rglob("*.onnx")):
        raise ValueError(f"'{bundle_dir}' contains no .onnx files; not a valid genai bundle.")

    return WinMLGenaiCausalLM(
        bundle_dir,
        config.ep,
        device=config.device,
    )


def _resolve_task(
    config: WinMLEvaluationConfig,
    pytorch_model: nn.Module | None = None,
) -> str:
    """Resolve the eval task and validate it is supported.

    An explicit ``config.task`` is surfaced verbatim (explicit means explicit).
    When omitted, the modality-aware :func:`resolve_task` infers it from the model's
    HF config — an image-embedding model resolves to ``image-feature-extraction``
    (not the lossy ``feature-extraction``), so the evaluator-registry lookup picks
    the image evaluator without any reverse io_config reconstruction.
    """
    console = Console()
    console.print("[bold]Resolving task...[/bold]")

    task = config.task if config.task is not None else _infer_task(config, pytorch_model)

    console.print(f"[dim]Use[/dim] {task} [dim]to evaluate[/dim]")

    if task not in _EVALUATOR_REGISTRY:
        supported = ", ".join(sorted(_EVALUATOR_REGISTRY))
        raise ValueError(f"Task '{task}' is not supported. Supported tasks: {supported}.")
    return task


def _infer_task(
    config: WinMLEvaluationConfig,
    pytorch_model: nn.Module | None = None,
) -> str:
    """Infer the evaluation task from the model's Hugging Face config."""
    hf_config = getattr(pytorch_model, "config", None)
    if hf_config is None:
        if config.model_id is None:
            raise ValueError("Cannot infer task without model_id or model.config. Provide --task.")

        from transformers import AutoConfig

        from ..loader import load_hf_config

        hf_config = load_hf_config(
            AutoConfig,
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )

    from ..loader.resolution import resolve_task

    return resolve_task(hf_config).task


def _pytorch_model_devices(model: nn.Module) -> tuple[str, str] | None:
    """Return WinML and exact pipeline devices for a placed PyTorch model."""
    device = getattr(model, "device", None)
    if device is None:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                device = next(iter(parameters())).device
            except StopIteration:
                device = None

    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        return "gpu", str(device)
    if device_type == "cpu":
        return "cpu", str(device)
    if device_type is not None:
        raise ValueError(
            f"PyTorch model device {device!s} is not supported; use a CPU or CUDA model."
        )
    return None


def _prepare_supplied_pytorch_model(
    config: WinMLEvaluationConfig,
    model: nn.Module,
) -> WinMLEvaluationConfig:
    """Resolve runtime metadata without reloading a supplied PyTorch model."""
    model_id = config.model_id
    if not model_id:
        model_config = getattr(model, "config", None)
        inferred_model_id = getattr(model_config, "_name_or_path", None)
        if isinstance(inferred_model_id, str) and inferred_model_id.strip():
            model_id = inferred_model_id
    if not model_id:
        raise ValueError(
            "A supplied PyTorch model requires model_id or a non-empty "
            "model.config._name_or_path for tokenizer or processor loading."
        )

    requested_device = config.device.lower()
    if requested_device not in ("auto", "cpu", "gpu"):
        raise ValueError(
            f"Device {config.device!r} is not supported for a supplied PyTorch model; "
            "use auto, cpu, or gpu."
        )
    model_devices = _pytorch_model_devices(model)
    model_device = model_devices[0] if model_devices is not None else None
    if model_device is not None and requested_device not in ("auto", model_device):
        raise ValueError(
            f"Supplied PyTorch model is on {model_device}, but config.device is {config.device!r}."
        )

    return replace(
        config,
        model_id=model_id,
        runtime="pytorch",
        device=model_device or config.device,
        _pipeline_device_override=(model_devices[1] if model_devices is not None else None),
    )


def evaluate(
    config: WinMLEvaluationConfig,
    *,
    pytorch_model: nn.Module | None = None,
) -> EvalResult:
    """Run model evaluation.

    A supplied ``pytorch_model`` is passed directly to the selected evaluator.
    ``config.model_id`` remains an explicit processor/tokenizer override; when
    omitted, it is inferred from ``pytorch_model.config._name_or_path``.

    This function does not mutate the caller's config. It creates internal
    copies via ``dataclasses.replace`` and ``deepcopy`` so the original
    config and any module-level defaults remain untouched.
    """
    from ..utils.eval_utils import EVAL_MODES

    if config.runtime not in ("winml-ort", "pytorch"):
        raise ValueError(f"Invalid runtime {config.runtime!r}; expected 'winml-ort' or 'pytorch'.")
    if pytorch_model is not None:
        config = _prepare_supplied_pytorch_model(config, pytorch_model)
    mode = config.mode if config.mode is not None else "onnx"
    if mode not in EVAL_MODES:
        raise ValueError(f"Invalid mode {mode!r}; expected one of {EVAL_MODES} or None.")
    # Two-ONNX compare does not require HF task resolution or model metadata.
    # With no explicit task, WinMLAutoModel selects its generic raw-output model.
    onnx_compare = mode == "compare" and config.reference_path is not None
    config = replace(
        config,
        mode=mode,
        task=(None if onnx_compare else _resolve_task(config, pytorch_model)),
        dataset=deepcopy(config.dataset),
    )
    _validate_pytorch_runtime_config(config)
    if config.mode != "compare" and config.dataset.path is None:
        default = _DEFAULT_DATASETS.get(config.task) if config.task is not None else None
        if default is None:
            raise ValueError(
                f"No dataset provided and no default for task '{config.task}'. Use --dataset."
            )
        user_columns = dict(config.dataset.columns_mapping or {})
        for k, v in default.items():
            setattr(config.dataset, k, deepcopy(v))
        # Preserve user-supplied --column values (e.g. the text-generation
        # scoring parameters num_tokens / seqlen): the default mapping only
        # fills columns the user did not provide, so an explicit --column wins.
        config.dataset.columns_mapping = {
            **deepcopy(default.get("columns_mapping", {})),
            **user_columns,
        }
        logger.warning(
            "--dataset not specified; attempting default dataset '%s' for task '%s'. "
            "Any --split / --streaming / --dataset-name options are ignored; "
            "explicit --column values are preserved.",
            config.dataset.path,
            config.task,
        )

    print_config(config)
    console = Console()

    model: Any
    if pytorch_model is not None:
        console.print("\n[bold]Using supplied PyTorch model...[/bold]")
        model = load_model(config, pytorch_model)
    else:
        console.print("\n[bold]Loading model...[/bold]")
        try:
            model = load_model(config)
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

    # For --input-data compare, the real sample count comes from the archive
    # (leading axis, chunked to the model's batch). Surface it on the result so
    # the report/JSON reflect N without writing back into the (copied) config.
    num_samples: int | None = None
    if config.input_data is not None:
        data = getattr(task_evaluator, "data", None)
        if data is not None:
            num_samples = len(data)

    return EvalResult(config=config, metrics=metrics, num_samples=num_samples)


def print_config(config: WinMLEvaluationConfig) -> None:
    """Print effective evaluation config to the console (quantize.py style)."""
    ds = config.dataset
    output_console = Console()
    if config.model_id is not None:
        output_console.print(f"[bold blue]Model:[/bold blue] {config.model_id}")
    if config.model_path is not None:
        output_console.print(f"[bold blue]Model path:[/bold blue] {config.model_path}")
    if config.input_data is not None:
        output_console.print(f"[bold blue]Input data:[/bold blue] {config.input_data}")
    if config.reference_path is not None:
        output_console.print(f"[bold blue]Reference:[/bold blue] {config.reference_path}")
    if config.task is not None:
        output_console.print(f"[bold blue]Task:[/bold blue] {config.task}")
    output_console.print(f"[bold blue]Runtime:[/bold blue] {config.runtime}")
    output_console.print(f"[bold blue]Device:[/bold blue] {config.device}")
    if config.ep is not None:
        output_console.print(f"[bold blue]EP:[/bold blue] {config.ep}")
    if config.runtime == "winml-ort":
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
