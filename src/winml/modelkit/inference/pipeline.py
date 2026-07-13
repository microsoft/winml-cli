# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared HF Pipeline factory for WinML models.

Used by both ``winml serve`` (InferenceEngine) and ``winml eval`` (WinMLEvaluator)
to create a ``transformers.pipeline`` backed by a WinMLPreTrainedModel.

The pipeline handles all preprocessing and postprocessing; the WinML model
only provides the ONNX Runtime inference session.

ONNX models have fixed input shapes. This module adapts the pipeline's
tokenizer/image_processor to match those shapes so inputs are correctly
padded/resized before hitting the ONNX runtime.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from transformers.pipelines.image_to_text import ImageToTextPipeline

from ..models.winml.composite_model import PipelineCapability


if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers.pipelines.base import GenericTensor

    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel

logger = logging.getLogger(__name__)

# Tasks that WinML recognises but HF ``transformers.pipeline`` does not.
# Mapped to their HF pipeline equivalent before calling ``pipeline()``.
_HF_PIPELINE_TASK_MAP: dict[str, str] = {
    "sentence-similarity": "feature-extraction",
}


class SupportsPipelineCapabilities(Protocol):
    """Model protocol for selecting non-default preprocessing pipelines."""

    pipeline_capabilities: frozenset[PipelineCapability]


class SupportsCombinedProcessor(SupportsPipelineCapabilities, Protocol):
    """Model protocol for combined image/text processor construction."""

    def create_combined_processor(self, model_id: str) -> Any:
        """Load the processor that satisfies the model's declared contract."""


class SupportsTokenDecoding(Protocol):
    """Tokenizer capability required by image-to-text postprocessing."""

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str:
        """Decode generated token IDs."""


class SupportsTokenizer(Protocol):
    """Processor capability for supplying the postprocessing tokenizer."""

    tokenizer: SupportsTokenDecoding


class SupportsCombinedProcessorInputs(Protocol):
    """Processor output that can be transferred to the pipeline tensor dtype."""

    def to(self, device: object) -> SupportsCombinedProcessorInputs:
        """Move the processor output to a tensor device or dtype."""


class SupportsCombinedImageTextProcessor(SupportsTokenizer, Protocol):
    """Combined image/text processor surface required by the custom pipeline."""

    def __call__(
        self, *, images: object, text: str, return_tensors: str
    ) -> SupportsCombinedProcessorInputs:
        """Process an image and its text prompt together."""


class CombinedProcessorImageToTextPipeline(ImageToTextPipeline):
    """Image-to-text pipeline that preserves a processor's joint image/text contract."""

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    # Transformers' Pipeline stub uses ``input_`` plus ``**dict`` while its
    # ImageToTextPipeline override uses image/prompt/timeout. Preserve the
    # latter's public API and narrow only the incompatible base-stub override.
    def preprocess(  # type: ignore[override]
        self, image: Any, prompt: Any = None, timeout: Any = None
    ) -> dict[str, GenericTensor]:
        """Create model inputs with one combined processor invocation."""
        from transformers.image_utils import load_image

        if prompt is None:
            raise ValueError("A prompt is required by the combined image/text processor.")
        processor = self.processor
        if processor is None or not callable(processor):
            raise TypeError("A combined image/text processor is required.")
        image = load_image(image, timeout=timeout)
        model_inputs = cast("SupportsCombinedImageTextProcessor", processor)(
            images=image,
            text=prompt,
            return_tensors=self.framework,
        )
        if self.framework == "pt":
            model_inputs = model_inputs.to(self.dtype)
        return cast("dict[str, GenericTensor]", model_inputs)


def _pipeline_class_for(model: Any) -> type | None:
    """Resolve an HF pipeline implementation from declared model capabilities."""
    capabilities = inspect.getattr_static(model, "pipeline_capabilities", frozenset())
    if not isinstance(capabilities, frozenset):
        raise TypeError("pipeline_capabilities must be a frozenset of PipelineCapability values")
    if not all(isinstance(capability, PipelineCapability) for capability in capabilities):
        raise TypeError("pipeline_capabilities must contain PipelineCapability values")
    if PipelineCapability.COMBINED_IMAGE_TEXT_PROCESSOR in capabilities:
        return CombinedProcessorImageToTextPipeline
    return None


def _combined_processor_for(
    model: Any, model_id: str | None
) -> SupportsCombinedImageTextProcessor:
    """Load the declared combined processor with explicit capability errors."""
    loader = getattr(model, "create_combined_processor", None)
    if not callable(loader):
        raise TypeError(
            "Models declaring combined-image-text-processor must implement "
            "create_combined_processor(model_id)."
        )
    if model_id is None:
        raise ValueError(
            "A model ID is required to load a combined image/text processor."
        )
    processor = loader(model_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if not callable(processor) or not callable(getattr(tokenizer, "decode", None)):
        raise TypeError(
            "Combined image/text processors must be callable and expose a tokenizer "
            "with a decode method."
        )
    return cast("SupportsCombinedImageTextProcessor", processor)


def _tokenizer_for(processor: SupportsTokenizer) -> SupportsTokenDecoding:
    """Return the processor-owned tokenizer required by image-to-text decoding."""
    return processor.tokenizer


def create_pipeline(
    task: str,
    model: WinMLPreTrainedModel | WinMLCompositeModel,
    model_id: str | None = None,
) -> Any:
    """Create an HF pipeline for a WinML model.

    Automatically adapts tokenizer padding and image processor size
    to match the ONNX model's fixed input shapes.

    Args:
        task: HF task name (e.g. "image-classification")
        model: Loaded WinMLPreTrainedModel instance
        model_id: HF model ID for loading processors (tokenizer, image processor).
                  If None, pipeline will attempt auto-detection.

    Returns:
        A configured ``transformers.Pipeline`` ready for inference.
    """
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "framework": "pt",
        # "device" is for HF pipeline tensor placement, not ORT EP.
        # WinMLSession handles device delegation internally.
        "device": "cpu",
    }
    pipeline_class = _pipeline_class_for(model)
    if pipeline_class is not None:
        processor = _combined_processor_for(model, model_id)
        kwargs["pipeline_class"] = pipeline_class
        kwargs["processor"] = processor
        kwargs["tokenizer"] = _tokenizer_for(processor)
    elif model_id:
        kwargs["tokenizer"] = model_id
        kwargs["feature_extractor"] = model_id
        kwargs["image_processor"] = model_id
        kwargs["processor"] = model_id

    hf_task = _HF_PIPELINE_TASK_MAP.get(task, task)
    # transformers.pipeline has 60+ Literal overloads — runtime task strings can't
    # be statically matched. The string-task fallback handles unknown tasks safely.
    pipe = pipeline(hf_task, model=model, **kwargs)  # type: ignore[call-overload]

    # Adapt pipeline to fixed ONNX input shapes
    _adapt_tokenizer_padding(pipe, task, model)
    _adapt_image_processor_size(pipe, task, model)

    logger.info("Created HF pipeline: task=%s model=%s", task, model_id)
    return pipe


def _adapt_tokenizer_padding(pipe: Any, task: str, model: Any) -> None:
    """Pad tokenizer output to match ONNX fixed sequence length.

    ONNX models are exported with a fixed sequence_length dimension.
    Without padding, the tokenizer produces variable-length tensors
    that cause INVALID_ARGUMENT errors at inference time.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has a tokenizer AND the
    model's first input shape is 2-D with a fixed integer second
    dimension (batch, sequence_length).  4-D shapes (N, C, H, W) are
    image tensors and are explicitly skipped.
    """
    if pipe.tokenizer is None:
        return

    io_config = getattr(model, "io_config", None) or {}
    shapes = io_config.get("input_shapes", [[]])
    # Find the first 2-D shape (batch, seq_len) — multi-modal models like CLIP
    # have both 2-D text inputs and 4-D image inputs; scanning all shapes ensures
    # tokenizer padding is applied regardless of input ordering.
    max_length = None
    for shape in shapes:
        if len(shape) == 2 and isinstance(shape[1], int):
            max_length = shape[1]
            break
    if max_length is None:
        return

    # HF pipeline classes consume tokenizer settings in three patterns:
    #
    # A) Direct **kwargs → tokenizer (TextClassification, FeatureExtraction)
    #    e.g. self.tokenizer(text, **tokenizer_kwargs)
    #    → set top-level padding/max_length/truncation in _preprocess_params
    #
    # B) Nested tokenizer dict (TokenClassification, FillMask)
    #    e.g. tok_params = preprocess_params.pop("tokenizer_params", {})
    #         self.tokenizer(text, truncation=truncation, **tok_params)
    #    or:  self.tokenizer(text, **tokenizer_kwargs)  [named param]
    #    → set padding/max_length inside a dict param
    #
    # C) Explicit named params only (QuestionAnswering: max_seq_len)
    #    No **kwargs — only accepts specific named params
    #    → set only params that appear in the signature

    preprocess = getattr(type(pipe), "preprocess", None)
    if not callable(preprocess):
        return
    preprocess_sig = inspect.signature(preprocess)
    sig_params = preprocess_sig.parameters

    tok_dict_key = _detect_tokenizer_dict_param(pipe, sig_params)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())

    if tok_dict_key:
        # Pattern B: tokenizer settings go in a nested dict
        pipe._preprocess_params.setdefault(tok_dict_key, {})
        tok = pipe._preprocess_params[tok_dict_key]
        tok.setdefault("padding", "max_length")
        tok.setdefault("max_length", max_length)
        # TokenClassification pops "truncation" separately from **kwargs
        if tok_dict_key == "tokenizer_params":
            pipe._preprocess_params.setdefault("truncation", True)
        else:
            tok.setdefault("truncation", True)
    elif has_varkw:
        # Pattern A: **kwargs forwarded directly to tokenizer
        pipe._preprocess_params.setdefault("padding", "max_length")
        pipe._preprocess_params.setdefault("max_length", max_length)
        pipe._preprocess_params.setdefault("truncation", True)
    else:
        # Pattern C: no **kwargs — only set params the signature accepts
        if "max_seq_len" in sig_params:
            pipe._preprocess_params.setdefault("max_seq_len", max_length)
        elif "max_length" in sig_params:
            pipe._preprocess_params.setdefault("max_length", max_length)
        if "padding" in sig_params:
            pipe._preprocess_params.setdefault("padding", "max_length")
        if "truncation" in sig_params:
            pipe._preprocess_params.setdefault("truncation", True)

    pipe.tokenizer.model_max_length = max_length


def _detect_tokenizer_dict_param(
    pipe: Any, sig_params: Mapping[str, inspect.Parameter]
) -> str | None:
    """Detect if preprocess() consumes tokenizer settings via a nested dict.

    Returns the dict key name (e.g. "tokenizer_kwargs", "tokenizer_params"),
    or None if the pipeline uses direct **kwargs or explicit named params.
    """
    # Check for a named (non-**kwargs) parameter like tokenizer_kwargs=None
    # (e.g. FillMaskPipeline)
    for name, param in sig_params.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if name != "self" and ("tokenizer" in name or "tokenize" in name):
            return name

    # Check if preprocess() pops "tokenizer_params" from **kwargs
    # (e.g. TokenClassificationPipeline).  Source inspection is fragile —
    # it fails for compiled (.pyc-only) code or C extensions — but there
    # is no runtime API to detect dict-style consumption of **kwargs.
    # The except clause degrades gracefully to "no nested dict detected".
    try:
        src = inspect.getsource(type(pipe).preprocess)
    except (OSError, TypeError):
        return None
    if "tokenizer_params" in src:
        return "tokenizer_params"

    return None


def _adapt_image_processor_size(pipe: Any, task: str, model: Any) -> None:
    """Match image processor size to ONNX fixed input shape (NCHW).

    Models with 4D input shapes have fixed spatial dimensions.
    The image processor must resize to exactly those dimensions.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has an image_processor AND
    the model's first input shape is 4D (N, C, H, W).

    Size dict format varies by processor class:
      - ``{"height": h, "width": w}`` — direct resize (ViT, DETR, …)
      - ``{"shortest_edge": n}`` — aspect-preserving resize, usually
        followed by a center crop (ResNet, ConvNeXt, …)
    We preserve the processor's original format to avoid validation errors.
    """
    if not hasattr(pipe, "image_processor"):
        return

    io_config = getattr(model, "io_config", None) or {}
    input_shapes = io_config.get("input_shapes", [])
    # Find the first 4-D shape (N, C, H, W) — multi-modal models may have
    # both 2-D text and 4-D image inputs in any order.
    image_shape = None
    for shape in input_shapes:
        if len(shape) == 4:
            image_shape = shape
            break
    if image_shape is None:
        return

    _, _, h, w = image_shape
    proc = pipe.image_processor
    original_size = getattr(proc, "size", {}) or {}

    if "shortest_edge" in original_size and "longest_edge" not in original_size:
        # Processor only accepts shortest_edge format (e.g. ConvNeXt).
        # These processors use crop_pct internally to resize then
        # center-crop to (shortest_edge, shortest_edge), so setting
        # shortest_edge = min(h, w) produces the correct output for
        # square ONNX shapes.  Forcing {"height", "width"} would raise
        # a validation error in their resize() method.
        proc.size = {"shortest_edge": min(h, w)}
    else:
        # Processors with height/width (ViT) or shortest_edge+longest_edge
        # (DETR) all accept explicit height/width for exact dimensions.
        proc.size = {"height": h, "width": w}

    if hasattr(proc, "do_pad"):
        proc.do_pad = False
