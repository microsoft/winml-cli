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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel

logger = logging.getLogger(__name__)

# Tasks that WinML recognises but HF ``transformers.pipeline`` does not.
# Mapped to their HF pipeline equivalent before calling ``pipeline()``.
_HF_PIPELINE_TASK_MAP: dict[str, str] = {
    "sentence-similarity": "feature-extraction",
}


class KeypointDetectionPipeline:
    """Minimal ViTPose inference adapter.

    Transformers does not expose a ``keypoint-detection`` pipeline. ViTPose is
    top-down, so callers provide person boxes and this adapter runs
    image_processor.preprocess -> WinML model -> post_process_pose_estimation.
    """

    def __init__(self, model: Any, model_id: str | None = None) -> None:
        from transformers import AutoImageProcessor

        self.model = model
        self.image_processor = AutoImageProcessor.from_pretrained(model_id) if model_id else None
        if self.image_processor is None:
            raise ValueError("keypoint-detection requires model_id to load the image processor")
        self._adapt_processor_size()

    def __call__(
        self,
        image: Any,
        *,
        boxes: list[list[float]],
        box_format: str = "xywh",
        dataset_index: int | float = 0,
    ) -> list[dict[str, Any]]:
        import torch

        if not isinstance(boxes, list) or not boxes:
            raise ValueError("boxes must be a non-empty list of [x, y, w, h] boxes")
        xywh_boxes = [self._to_xywh(box, box_format) for box in boxes]
        inputs = self.image_processor.preprocess(
            images=image,
            boxes=[xywh_boxes],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]

        extra_inputs: dict[str, Any] = {}
        if self._declares_input("dataset_index"):
            extra_inputs["dataset_index"] = torch.tensor([int(dataset_index)], dtype=torch.long)

        heatmaps = []
        with torch.no_grad():
            for i in range(pixel_values.shape[0]):
                outputs = self.model(pixel_values=pixel_values[i : i + 1], **extra_inputs)
                heatmaps.append(self._extract_heatmaps(outputs))

        wrapped = SimpleNamespace(heatmaps=torch.cat(heatmaps, dim=0))
        poses = self.image_processor.post_process_pose_estimation(wrapped, boxes=[xywh_boxes])[0]
        return [self._serialize_pose(pose) for pose in poses]

    def _sanitize_parameters(
        self,
        boxes: tuple[Any, ...] = (),
        box_format: str = "xywh",
        dataset_index: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Expose accepted kwargs for InferenceEngine parameter filtering."""
        forward_kwargs = {
            "boxes": boxes,
            "box_format": box_format,
            "dataset_index": dataset_index,
        }
        return {}, forward_kwargs, {}

    def _adapt_processor_size(self) -> None:
        io_config = getattr(self.model, "io_config", None) or {}
        for shape in io_config.get("input_shapes", []):
            if len(shape) == 4:
                _, _, h, w = shape
                self.image_processor.size = {"height": h, "width": w}
                break

    def _declares_input(self, name: str) -> bool:
        io_config = getattr(self.model, "io_config", None) or {}
        return name in (io_config.get("input_names") or [])

    @staticmethod
    def _extract_heatmaps(outputs: Any) -> Any:
        if not isinstance(outputs, dict):
            return outputs.heatmaps
        heatmaps = outputs.get("heatmaps")
        if heatmaps is None:
            heatmaps = next(iter(outputs.values()))
        return heatmaps

    @staticmethod
    def _serialize_pose(pose: dict[str, Any]) -> dict[str, Any]:
        keypoints = pose["keypoints"].detach().cpu().numpy().tolist()
        scores = pose["scores"].detach().cpu().numpy().tolist()
        return {
            "keypoints": keypoints,
            "scores": scores,
            "score": float(sum(scores) / len(scores)) if scores else 0.0,
        }

    @staticmethod
    def _to_xywh(box: list[float], box_format: str) -> list[float]:
        if len(box) != 4:
            raise ValueError("each box must contain exactly four coordinates")
        x0, y0, a, b = (float(v) for v in box)
        if box_format == "xyxy":
            return [x0, y0, a - x0, b - y0]
        if box_format != "xywh":
            raise ValueError("box_format must be 'xywh' or 'xyxy'")
        return [x0, y0, a, b]


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
    if task == "keypoint-detection":
        return KeypointDetectionPipeline(model, model_id)

    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "framework": "pt",
        # "device" is for HF pipeline tensor placement, not ORT EP.
        # WinMLSession handles device delegation internally.
        "device": "cpu",
    }
    if model_id:
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

    preprocess_sig = inspect.signature(type(pipe).preprocess)
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
