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
from typing import TYPE_CHECKING, Any

from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import (
    DEFAULT_DUMMY_SHAPES,
    DummyTextInputGenerator,
)

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
register_onnx_overwrite = TasksManager.create_register("onnx", overwrite_existing=True)


_hf_models_registered = False


def ensure_hf_models_registered() -> None:
    """Trigger HF model ONNX config registrations (idempotent).

    With lazy loading in ``modelkit/__init__.py``, the HF model files
    (bert.py, clip.py, etc.) and their ``@register_onnx_overwrite``
    decorators are not executed until explicitly imported. This function
    forces that import so registrations are in place before any
    ``TasksManager.get_exporter_config_constructor()`` call.
    """
    global _hf_models_registered
    if _hf_models_registered:
        return
    from ..models import hf as _hf  # noqa: F401

    _hf_models_registered = True


# =============================================================================
# Task Synonym Extensions (extends Optimum's TasksManager.map_from_synonym)
# =============================================================================

# Extends Optimum's built-in task synonym mapping for tasks it doesn't recognize.
# Optimum's map_from_synonym handles known synonyms like:
#   - "image-feature-extraction" → "feature-extraction"
# This dict adds mappings for tasks Optimum doesn't support at all.
TASK_SYNONYM_EXTENSIONS: dict[str, str] = {
    # next-sentence-prediction has same I/O as text-classification: input_ids → logits
    "next-sentence-prediction": "text-classification",
    # mask-generation is registered via register_onnx_overwrite for SAM2.
    # Optimum incorrectly maps it to "feature-extraction"; preserve as-is.
    "mask-generation": "mask-generation",
}


def _map_task_synonym(task: str) -> str:
    """Map task name to canonical form, extending Optimum's synonym mapping.

    Our extensions take priority over Optimum's built-in synonym map.
    If a task is found in ``TASK_SYNONYM_EXTENSIONS``, return immediately
    without passing through Optimum (which may incorrectly normalize
    custom-registered tasks like ``mask-generation``).

    Args:
        task: Task name (e.g., "next-sentence-prediction", "image-feature-extraction")

    Returns:
        Canonical task name (e.g., "text-classification", "feature-extraction")

    Example:
        >>> map_task_synonym("next-sentence-prediction")  # Our extension
        'text-classification'
        >>> map_task_synonym("mask-generation")  # Preserved (not Optimum-normalized)
        'mask-generation'
        >>> map_task_synonym("image-feature-extraction")  # Optimum's synonym
        'feature-extraction'
        >>> map_task_synonym("text-classification")  # Already canonical
        'text-classification'
    """
    # Our extensions take priority — return early to prevent Optimum from
    # incorrectly normalizing custom-registered tasks.
    if task in TASK_SYNONYM_EXTENSIONS:
        return TASK_SYNONYM_EXTENSIONS[task]

    # Fallback: normalize via Optimum's built-in synonym mapping
    return TasksManager.map_from_synonym(task)


# =============================================================================
# Custom Input Generators
# =============================================================================
class MaxLengthTextInputGenerator(DummyTextInputGenerator):
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
        **kwargs,
    ):
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
        task: Task name (will be normalized via map_task_synonym)
        hf_config: HuggingFace PretrainedConfig for OnnxConfig instantiation
        library_name: Source library (default: "transformers")
        exporter: Export backend (default: "onnx")

    Returns:
        Instantiated OnnxConfig for the model

    Raises:
        ValueError: If no OnnxConfig is registered for the model_type/task combination
    """
    ensure_hf_models_registered()

    normalized_task = _map_task_synonym(task)

    logger.debug(
        "Getting OnnxConfig: model_type=%s, task=%s -> %s",
        model_type,
        task,
        normalized_task,
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
) -> None:
    """Populate height/width in shape_kwargs from preprocessor_config.json.

    Optimum's DummyVisionInputGenerator falls back to 64x64 when model config
    lacks image_size (e.g., ResNet). This reads the correct size from
    preprocessor_config.json and injects it into shape_kwargs.

    Args:
        model_id: HuggingFace model identifier (e.g., "microsoft/resnet-50")
        shape_kwargs: Mutable dict to update with height/width if found
    """
    if not model_id:
        return

    if "height" in shape_kwargs or "width" in shape_kwargs:
        return

    try:
        from transformers.image_processing_utils import ImageProcessingMixin

        config, _ = ImageProcessingMixin.get_image_processor_dict(model_id)
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
                "Loaded image size from preprocessor_config.json: %dx%d",
                shape_kwargs["height"],
                shape_kwargs["width"],
            )
    except (OSError, ValueError, KeyError) as e:
        logger.debug("Could not load preprocessor_config.json for %s: %s", model_id, e)


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
    _populate_image_size_from_preprocessor(model_id, shape_kwargs)
    _populate_sequence_length_from_config(hf_config, shape_kwargs)

    logger.debug(
        "Generating dummy inputs with config %s, shapes=%s",
        type(onnx_config).__name__,
        shape_kwargs,
    )

    return onnx_config.generate_dummy_inputs(framework="pt", **shape_kwargs)


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
    _populate_image_size_from_preprocessor(model_id, shape_kwargs)
    _populate_sequence_length_from_config(hf_config, shape_kwargs)

    # Generate dummy inputs for concrete shapes and dtypes,
    # intercepting value ranges from Optimum's tensor gen methods
    with intercept_value_ranges() as value_ranges:
        dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", **shape_kwargs)

    input_shapes = [tuple(t.shape) for t in dummy_inputs.values()]
    input_dtypes = [str(t.dtype).replace("torch.", "") for t in dummy_inputs.values()]

    # Build value_range dict: {name: (min, max)} from intercepted data
    value_ranges = {name: (info["min"], info["max"]) for name, info in value_ranges.items()}

    return {
        "inputs": onnx_config.inputs,
        "outputs": onnx_config.outputs,
        "input_names": list(onnx_config.inputs.keys()),
        "output_names": list(onnx_config.outputs.keys()),
        "dynamic_axes": {**onnx_config.inputs, **onnx_config.outputs},
        "input_shapes": input_shapes,
        "input_dtypes": input_dtypes,
        "value_ranges": value_ranges,
    }
