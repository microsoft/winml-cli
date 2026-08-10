# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Export I/O Specification - Fully Leverages Optimum.

This module provides config-only utilities for ONNX export I/O specification
and dummy input generation. No model weights needed - all functions work
with just (model_type, task, hf_config).

Key Components:
- InputTensorSpec / OutputTensorSpec: Re-exported from winml.modelkit.onnx.io
- generate_dummy_inputs: Generate input tensors from hf_config (no weights)
- resolve_io_specs: Get I/O metadata (names, shapes, dtypes, axes, value_ranges)
- MaxLengthTextInputGenerator: Uses max_position_embeddings as sequence_length

Example:
    >>> from winml.modelkit.export.io import resolve_io_specs, generate_dummy_inputs
    >>> from transformers import AutoConfig
    >>>
    >>> hf_config = AutoConfig.from_pretrained("bert-base-uncased")
    >>>
    >>> # Get I/O metadata
    >>> specs = resolve_io_specs("bert", "fill-mask", hf_config)
    >>> specs["input_names"]   # ["input_ids", "attention_mask", "token_type_ids"]
    >>>
    >>> # Generate actual tensors
    >>> inputs = generate_dummy_inputs("bert", "fill-mask", hf_config)
    >>> inputs["input_ids"].shape  # torch.Size([1, 512])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import (
    DEFAULT_DUMMY_SHAPES,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
)

from ..loader import to_optimum_task
from .value_range import intercept_value_ranges


if TYPE_CHECKING:
    import torch
    from optimum.exporters.onnx import OnnxConfig
    from optimum.utils import NormalizedTextConfig
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class ONNXConfigNotFoundError(ValueError):
    """Raised when no OnnxConfig is registered for a model_type/task combination."""


# Create register with overwrite_existing=True to override Optimum's defaults.
# Optimum's register_tasks_manager_onnx uses overwrite_existing=False, which means
# registrations are silently skipped if a config already exists for the model/task.
# Explicit annotation: the RHS is Any (optimum is untyped), and the import cycle
# export.io -> models.hf -> export makes the *inferred* type undeterminable in a
# combined mypy run ([has-type] at every @register_onnx_overwrite site). Declaring
# the type breaks that ordering dependency.
register_onnx_overwrite: Any = TasksManager.create_register("onnx", overwrite_existing=True)


def ensure_hf_models_registered() -> None:
    """Trigger HF model ONNX config registrations (idempotent).

    With lazy loading in ``modelkit/__init__.py``, the HF model files
    (bert.py, clip.py, etc.) and their ``@register_onnx_overwrite``
    decorators are not executed until explicitly imported. This function
    forces that import so registrations are in place before any
    ``TasksManager.get_exporter_config_constructor()`` call.
    """
    if getattr(ensure_hf_models_registered, "_done", False):
        return
    from ..models import hf as _hf  # noqa: F401

    ensure_hf_models_registered._done = True  # type: ignore[attr-defined]


# =============================================================================
# Task Synonym Extensions (relocated to loader.task — single source of truth)
# =============================================================================
# ``TASK_SYNONYM_EXTENSIONS`` and the WinML -> Optimum collapse now live in
# ``loader.task`` as ``to_optimum_task``. Both are imported above and re-exported
# here; ``map_task_synonym`` is kept as a backward-compatible alias for existing
# importers (and is identical to ``to_optimum_task``).
map_task_synonym = to_optimum_task


# =============================================================================
# Custom Input Generators
# =============================================================================
class MaxLengthTextInputGenerator(DummyTextInputGenerator):  # type: ignore[misc]
    """Text input generator that uses max_position_embeddings as sequence_length.

    Optimum's DummyTextInputGenerator uses a hardcoded default of 16 for
    sequence_length and ignores normalized_config.sequence_length.

    This generator reads sequence_length from normalized_config, which should
    be configured via NORMALIZED_CONFIG_CLASS with:
        sequence_length="max_position_embeddings"

    Usage in OnnxConfig:
        class BertIOConfig(BertOnnxConfig):
            NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
                sequence_length="max_position_embeddings",
                allow_new=True,
            )
            DUMMY_INPUT_GENERATOR_CLASSES = (MaxLengthTextInputGenerator,)
    """

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with sequence_length from normalized_config.

        Args:
            task: Task name
            normalized_config: Normalized config with sequence_length attribute
            sequence_length: Override sequence_length (if None, reads from normalized_config)
            **kwargs: Additional arguments passed to parent
        """
        # Read sequence_length from normalized_config if not explicitly provided
        if sequence_length is None:
            sequence_length = getattr(
                normalized_config,
                "sequence_length",
                DEFAULT_DUMMY_SHAPES["sequence_length"],
            )
        super().__init__(
            task,
            normalized_config,
            sequence_length=sequence_length,
            **kwargs,
        )


def _get_onnx_config(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig,
    library_name: str = "transformers",
    exporter: str = "onnx",
) -> OnnxConfig:
    """Get instantiated OnnxConfig from model_type, task, and hf_config (internal).

    Single internal helper used by both generate_dummy_inputs() and
    resolve_io_specs(). Handles task synonym normalization and
    TasksManager lookup.

    Args:
        model_type: HF model type (e.g., "bert", "clip_vision_model")
        task: Task name (will be collapsed to Optimum-canonical via to_optimum_task)
        hf_config: HuggingFace PretrainedConfig for OnnxConfig instantiation
        library_name: Source library (default: "transformers")
        exporter: Export backend (default: "onnx")

    Returns:
        Instantiated OnnxConfig for the model

    Raises:
        ValueError: If no OnnxConfig is registered for the model_type/task combination
    """
    ensure_hf_models_registered()

    normalized_task = to_optimum_task(task)

    # Route model_types whose Optimum OnnxConfig is registered under another
    # library (e.g. timm via "timm_wrapper" -> "timm") so the lookup succeeds
    # from every call site without an explicit --library flag.
    from ..loader import resolve_optimum_library

    library_name = resolve_optimum_library(model_type, library_name)

    logger.debug(
        "Getting OnnxConfig: model_type=%s, task=%s -> %s (library=%s)",
        model_type,
        task,
        normalized_task,
        library_name,
    )

    try:
        config_constructor = TasksManager.get_exporter_config_constructor(
            exporter=exporter,
            model_type=model_type,
            task=normalized_task,
            library_name=library_name,
        )
    except KeyError as e:
        raise ONNXConfigNotFoundError(
            f"No OnnxConfig registered for model_type='{model_type}' with task='{task}'. "
            f"Ensure the model's ONNX config is registered with TasksManager. "
            f"Original error: {e}"
        ) from e

    return config_constructor(hf_config, task=normalized_task)


def _populate_image_size_from_preprocessor(
    model_id: str | None,
    shape_kwargs: dict,
    hf_config: PretrainedConfig | None = None,
) -> None:
    """Populate height/width in shape_kwargs from preprocessor metadata.

    Optimum's DummyVisionInputGenerator falls back to 64x64 when model config
    lacks image_size (e.g., ResNet, timm). This reads the correct size from
    a preprocessor_config-style dict obtained via :func:`_get_preprocessor_dict`
    (which consults the hub's ``preprocessor_config.json`` first and, when that
    is unavailable, synthesizes one from wrapper-config metadata such as
    ``TimmWrapperConfig.pretrained_cfg``).

    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/resnet-50")
        shape_kwargs: Mutable dict to update with height/width if found
        hf_config: HuggingFace PretrainedConfig used to synthesize a
            preprocessor dict when ``preprocessor_config.json`` is missing.
    """
    if not model_id:
        return

    if "height" in shape_kwargs or "width" in shape_kwargs:
        return

    config = _get_preprocessor_dict(model_id, hf_config)
    size = config.get("size")

    if isinstance(size, int):
        shape_kwargs["height"] = size
        shape_kwargs["width"] = size
    elif isinstance(size, dict):
        if "height" in size:
            shape_kwargs["height"] = size["height"]
            shape_kwargs["width"] = size["width"]
        elif "shortest_edge" in size:
            shape_kwargs["height"] = size["shortest_edge"]
            shape_kwargs["width"] = size["shortest_edge"]

    if "height" in shape_kwargs:
        logger.debug(
            "Loaded image size from preprocessor dict: %dx%d",
            shape_kwargs["height"],
            shape_kwargs["width"],
        )


def _get_preprocessor_dict(
    model_id: str | None,
    hf_config: PretrainedConfig | None,
) -> dict:
    """Return a ``preprocessor_config.json``-style dict for the model.

    Resolution order:

    1. ``preprocessor_config.json`` fetched from the hub (standard HF vision),
       used only when it carries a ``size`` key.
    2. Synthesized from a nested plain-dict attribute on ``hf_config``
       carrying ``input_size`` or ``image_size`` (e.g.
       ``TimmWrapperConfig.pretrained_cfg``). Reached when the hub file is
       unavailable *or* present but missing ``size`` (a partial config).

    Returns the dict in the standard preprocessor schema (``{"size": ...}``)
    so downstream parsing logic does not need to know which source it came
    from. Returns an empty dict when neither source yields a usable size.
    """
    try:
        if model_id is None:
            raise OSError("No model_id provided")
        from transformers.image_processing_utils import (  # type: ignore[attr-defined]
            ImageProcessingMixin,
        )

        config, _ = ImageProcessingMixin.get_image_processor_dict(model_id)
        if "size" in config:
            return config
        # Partial preprocessor_config.json without a "size" key: fall through
        # to synthesis so we don't silently use Optimum's 64x64 default.
    except (OSError, ValueError, KeyError) as e:
        logger.debug("Could not load preprocessor_config.json for %s: %s", model_id, e)

    if hf_config is not None:
        return _synthesize_preprocessor_dict(hf_config)
    return {}


def _synthesize_preprocessor_dict(hf_config: PretrainedConfig) -> dict:
    """Build a ``preprocessor_config.json``-style dict from ``hf_config.pretrained_cfg``.

    timm wrapper configs (``TimmWrapperConfig``) stash shape metadata in a
    ``pretrained_cfg`` dict carrying ``input_size = [C, H, W]``. Optimum's
    NormalizedConfig only walks ``PretrainedConfig`` children, so this
    dict-wrapped value is invisible to the dummy-input generator and it
    falls back to 64x64.

    Preprocessing keys (``mean``/``std``/``interpolation``/``crop_pct``)
    don't affect export tensor shapes and are intentionally ignored.
    """
    pretrained_cfg = getattr(hf_config, "pretrained_cfg", None)
    if not isinstance(pretrained_cfg, dict):
        return {}

    input_size = pretrained_cfg.get("input_size")
    if isinstance(input_size, (list, tuple)):
        if len(input_size) == 3:
            return {"size": {"height": input_size[1], "width": input_size[2]}}
        if len(input_size) == 1:
            return {"size": input_size[0]}

    return {}


# Practical cap for export dummy input sequence length.
# LLMs have max_position_embeddings of 40K-131K which would OOM during export.
_MAX_EXPORT_SEQ_LEN = 1024


def _populate_sequence_length_from_config(
    hf_config: PretrainedConfig | None,
    shape_kwargs: dict,
) -> None:
    """Populate sequence_length in shape_kwargs from model config.

    Optimum defaults to sequence_length=16 for all models. For text models
    with max_position_embeddings, use a practical value capped at
    ``_MAX_EXPORT_SEQ_LEN`` (1024) to avoid OOM during export tracing.

    Skips if the user already provided sequence_length in shape_kwargs.

    Args:
        hf_config: HuggingFace model config (may be None).
        shape_kwargs: Mutable dict to update with sequence_length if needed.
    """
    if not hf_config:
        return

    if "sequence_length" in shape_kwargs:
        return  # User explicitly set it

    max_pos = getattr(hf_config, "max_position_embeddings", None)
    if max_pos is not None and max_pos > 16:
        # Cap at practical export size
        seq_len = min(max_pos, _MAX_EXPORT_SEQ_LEN)
        shape_kwargs["sequence_length"] = seq_len
        logger.debug(
            "Set sequence_length=%d from max_position_embeddings=%d (cap=%d)",
            seq_len,
            max_pos,
            _MAX_EXPORT_SEQ_LEN,
        )


def _find_unique_config_value(hf_config: PretrainedConfig, key: str) -> Any | None:
    """Return a metadata value when it is unambiguous across a nested config."""
    values: list[Any] = []

    def _visit(value: Any) -> None:
        if isinstance(value, dict):
            if key in value:
                values.append(value[key])
            for child in value.values():
                _visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _visit(child)

    _visit(hf_config.to_dict())
    unique_values = {value for value in values if isinstance(value, (int, float, str))}
    return unique_values.pop() if len(unique_values) == 1 else None


def _is_vision_onnx_config(onnx_config: OnnxConfig) -> bool:
    """Return whether vendor metadata declares a vision dummy-input generator."""
    generator_classes = getattr(onnx_config, "DUMMY_INPUT_GENERATOR_CLASSES", ())
    return any(
        isinstance(generator_class, type)
        and issubclass(generator_class, DummyVisionInputGenerator)
        for generator_class in generator_classes
    )


def _declared_image_shape(
    hf_config: PretrainedConfig,
    shape_kwargs: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Resolve a BCHW image shape from model and processor metadata."""
    dimensions = {
        "batch": shape_kwargs.get("batch_size"),
        "channels": _find_unique_config_value(hf_config, "num_channels"),
        "height": shape_kwargs.get("height"),
        "width": shape_kwargs.get("width"),
    }
    if not all(isinstance(value, int) and value > 0 for value in dimensions.values()):
        return None
    return cast("tuple[int, int, int, int]", tuple(dimensions.values()))


def _normalized_image_value_range(
    preprocessor: dict[str, Any],
    *,
    num_channels: int,
    float_dtype: str,
) -> tuple[float, float] | None:
    """Derive the processor-normalized range for an image tensor when fully specified."""
    if preprocessor.get("do_rescale") is not True or preprocessor.get("do_normalize") is not True:
        return None

    rescale_factor = preprocessor.get("rescale_factor")
    image_mean = preprocessor.get("image_mean")
    image_std = preprocessor.get("image_std")
    if (
        not isinstance(rescale_factor, (int, float))
        or not isinstance(image_mean, (list, tuple))
        or not isinstance(image_std, (list, tuple))
        or len(image_mean) != num_channels
        or len(image_std) != num_channels
        or not all(isinstance(value, (int, float)) for value in [*image_mean, *image_std])
        or not all(value > 0 for value in image_std)
    ):
        return None

    if float_dtype == "fp16":
        means_fp16 = np.asarray(image_mean, dtype=np.float16)
        stds_fp16 = np.asarray(image_std, dtype=np.float16)
        scaled_max_fp16 = np.asarray(255, dtype=np.float16) * np.asarray(
            rescale_factor, dtype=np.float16
        )
        lower_fp16 = np.min((np.asarray(0, dtype=np.float16) - means_fp16) / stds_fp16)
        upper_fp16 = np.max((scaled_max_fp16 - means_fp16) / stds_fp16)
        upper_exclusive_fp16 = np.nextafter(upper_fp16, np.asarray(np.inf, dtype=np.float16))
        return float(lower_fp16), float(upper_exclusive_fp16)

    means_fp32 = np.asarray(image_mean, dtype=np.float32)
    stds_fp32 = np.asarray(image_std, dtype=np.float32)
    scaled_max_fp32 = np.asarray(255, dtype=np.float32) * np.asarray(
        rescale_factor, dtype=np.float32
    )
    lower_fp32 = np.min((np.asarray(0, dtype=np.float32) - means_fp32) / stds_fp32)
    upper_fp32 = np.max((scaled_max_fp32 - means_fp32) / stds_fp32)
    upper_exclusive_fp32 = np.nextafter(upper_fp32, np.asarray(np.inf, dtype=np.float32))
    return float(lower_fp32), float(upper_exclusive_fp32)


def generate_dummy_inputs(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig,
    library_name: str = "transformers",
    model_id: str | None = None,
    batch_size: int = 1,
    int_dtype: str = "int32",
    float_dtype: str = "fp32",
    **shape_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Generate dummy inputs using Optimum's OnnxConfig (no weights needed).

    Works with just hf_config - no model weights required. For vision models,
    if model_id is provided and height/width are not in shape_kwargs, reads
    preprocessor_config.json for correct image dimensions (avoids Optimum's
    64x64 fallback).

    Args:
        model_type: HF model type (e.g., "bert", "clip_vision_model")
        task: Task name (e.g., "image-feature-extraction", "feature-extraction")
        hf_config: HuggingFace PretrainedConfig
        library_name: Source library (default: "transformers")
        model_id: HuggingFace model identifier for preprocessor_config.json
        batch_size: Batch size for input shapes (default: 1 for QNN compatibility)
        int_dtype: Integer dtype for text inputs (default: "int32").
            Optimum notation: "int64", "int32", "int8".
        float_dtype: Float dtype for vision inputs (default: "fp32").
            Optimum notation: "fp32", "fp16", "bf16".
        **shape_kwargs: Override default shapes (sequence_length, height, width, etc.)

    Returns:
        Dictionary of input tensors ready for export

    Example:
        >>> from transformers import AutoConfig
        >>> hf_config = AutoConfig.from_pretrained("microsoft/resnet-50")
        >>> inputs = generate_dummy_inputs(
        ...     "resnet", "image-classification", hf_config,
        ...     model_id="microsoft/resnet-50",
        ... )
        >>> inputs["pixel_values"].shape  # torch.Size([1, 3, 224, 224])
    """
    onnx_config = _get_onnx_config(model_type, task, hf_config, library_name)

    onnx_config.int_dtype = int_dtype
    onnx_config.float_dtype = float_dtype

    shape_kwargs["batch_size"] = batch_size
    _populate_image_size_from_preprocessor(model_id, shape_kwargs, hf_config)
    _populate_sequence_length_from_config(hf_config, shape_kwargs)

    logger.debug(
        "Generating dummy inputs with config %s, shapes=%s",
        type(onnx_config).__name__,
        shape_kwargs,
    )

    # Optimum's OnnxConfig is untyped; the dummy-inputs dict matches our return type.
    return cast(
        "dict[str, torch.Tensor]",
        onnx_config.generate_dummy_inputs(framework="pt", **shape_kwargs),
    )


def resolve_io_specs(
    model_type: str,
    task: str,
    hf_config: PretrainedConfig,
    *,
    library_name: str = "transformers",
    model_id: str | None = None,
    batch_size: int = 1,
    int_dtype: str = "int32",
    float_dtype: str = "fp32",
    **shape_kwargs: Any,
) -> dict[str, Any]:
    """Resolve I/O specs from OnnxConfig (no model weights needed).

    Uses OnnxConfig's inputs/outputs for tensor names and dynamic axes,
    and generates dummy inputs for concrete shapes and dtypes.

    Args:
        model_type: HF model type (e.g., "bert", "clip_text_model").
        task: Export task (e.g., "fill-mask", "image-classification").
        hf_config: HuggingFace PretrainedConfig.
        library_name: Source library ("transformers", "diffusers", "timm").
        model_id: HF model ID for preprocessor_config.json (correct image sizes).
        batch_size: Batch size for input shapes (default: 1 for QNN).
        int_dtype: Integer dtype for text inputs (default: "int32").
            Optimum notation: "int64", "int32", "int8".
        float_dtype: Float dtype for vision inputs (default: "fp32").
            Optimum notation: "fp32", "fp16", "bf16".
        **shape_kwargs: Override default shapes (sequence_length, height, width).

    Returns:
        Dict with:
            - inputs: dict mapping input names to dynamic axes
            - outputs: dict mapping output names to dynamic axes
            - input_names: list of input tensor names
            - output_names: list of output tensor names
            - dynamic_axes: combined dict for torch.onnx.export
            - input_shapes: list of input tensor shapes
            - input_dtypes: list of dtype strings (e.g., "float32", "int64")

    Raises:
        ValueError: If no OnnxConfig is registered for the model_type/task.
    """
    onnx_config = _get_onnx_config(model_type, task, hf_config, library_name)

    # Set dtypes on onnx_config (passed to generators via generate_dummy_inputs)
    onnx_config.int_dtype = int_dtype
    onnx_config.float_dtype = float_dtype

    # Populate shapes from model config / preprocessor
    shape_kwargs["batch_size"] = batch_size
    _populate_image_size_from_preprocessor(model_id, shape_kwargs, hf_config)
    _populate_sequence_length_from_config(hf_config, shape_kwargs)
    preprocessor = _get_preprocessor_dict(model_id, hf_config)

    # Generate dummy inputs for concrete shapes and dtypes,
    # intercepting value ranges from Optimum's tensor gen methods
    with intercept_value_ranges() as value_ranges:
        try:
            dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", **shape_kwargs)
        except IndexError:
            if not (
                onnx_config.inputs
                and onnx_config.outputs
                and _is_vision_onnx_config(onnx_config)
            ):
                raise
            logger.debug(
                "Vendor vision dummy-input generation failed; recovering from "
                "declared OnnxConfig metadata",
                exc_info=True,
            )
            dummy_inputs = {}

    input_shapes = [tuple(t.shape) for t in dummy_inputs.values()]
    input_dtypes = [str(t.dtype).replace("torch.", "") for t in dummy_inputs.values()]

    # Build value_range dict: {name: (min, max)} from intercepted data
    value_range_tuples = {
        name: (info["min"], info["max"]) for name, info in value_ranges.items()
    }

    if len(onnx_config.inputs) == 1 and _is_vision_onnx_config(onnx_config):
        input_name = next(iter(onnx_config.inputs))
        image_shape = _declared_image_shape(hf_config, shape_kwargs)
        if image_shape is not None:
            resolved_dtype = "float16" if float_dtype == "fp16" else "float32"
            if not dummy_inputs:
                input_shapes = [image_shape]
                input_dtypes = [resolved_dtype]
            if input_shapes == [image_shape] and input_dtypes == [resolved_dtype]:
                normalized_range = _normalized_image_value_range(
                    preprocessor,
                    num_channels=image_shape[1],
                    float_dtype=float_dtype,
                )
                if normalized_range is not None:
                    value_range_tuples[input_name] = normalized_range

    return {
        "inputs": onnx_config.inputs,
        "outputs": onnx_config.outputs,
        "input_names": list(onnx_config.inputs.keys()),
        "output_names": list(onnx_config.outputs.keys()),
        "dynamic_axes": {**onnx_config.inputs, **onnx_config.outputs},
        "input_shapes": input_shapes,
        "input_dtypes": input_dtypes,
        "value_ranges": value_range_tuples,
    }
