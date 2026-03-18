"""Optimum integration utilities for loading ONNX models with HuggingFace configurations.

This module provides seamless integration with HuggingFace Optimum for inference
using exported ONNX models with preserved hierarchy and Hub metadata.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .hub_utils import load_hf_components_from_onnx


logger = logging.getLogger(__name__)


class OptimumONNXModel:
    """Wrapper for seamless Optimum integration with Hub metadata."""

    @classmethod
    def from_onnx(
        cls, onnx_path: str, task: str = "auto", device: str = "cpu", **kwargs: Any
    ) -> tuple[Any, Any]:
        """Load Optimum model from ONNX with Hub metadata.

        Args:
            onnx_path: Path to ONNX model
            task: Task type or "auto" to detect
            device: Device to run on
            **kwargs: Additional arguments for ORTModel

        Returns:
            Tuple of (model, preprocessor)
        """
        # Load config and preprocessor
        config, preprocessor = load_hf_components_from_onnx(onnx_path)

        # Auto-detect task if needed
        if task == "auto":
            task = cls._detect_task(config, onnx_path)

        # Get appropriate ORTModel class
        ort_model_class = cls._get_ort_model_class(task)

        # Create temporary directory with required files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save config
            config.save_pretrained(temp_path)

            # Save preprocessor if available
            if preprocessor:
                preprocessor.save_pretrained(temp_path)

            # Copy ONNX model
            shutil.copy(onnx_path, temp_path / "model.onnx")

            # Load with Optimum
            model = ort_model_class.from_pretrained(
                temp_path,
                provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
                **kwargs,
            )

        return model, preprocessor

    @staticmethod
    def _detect_task(config: Any, onnx_path: str) -> str:
        """Detect task from config and metadata."""
        from ..onnx import load_onnx

        # Try to get task from metadata
        try:
            onnx_model = load_onnx(onnx_path, validate=False)
            for prop in onnx_model.metadata_props:
                if prop.key == "hf_pipeline_tag":
                    return prop.value
        except Exception:
            pass

        # Check architectures
        if hasattr(config, "architectures"):
            arch = config.architectures[0] if config.architectures else ""

            task_mapping = {
                "ForSequenceClassification": "text-classification",
                "ForTokenClassification": "token-classification",
                "ForQuestionAnswering": "question-answering",
                "ForCausalLM": "text-generation",
                "ForConditionalGeneration": "text2text-generation",
                "ForImageClassification": "image-classification",
                "ForObjectDetection": "object-detection",
                "ForAudioClassification": "audio-classification",
            }

            for pattern, task in task_mapping.items():
                if pattern in arch:
                    return task

        # Default
        return "feature-extraction"

    @staticmethod
    def _get_ort_model_class(task: str) -> type[Any]:
        """Get appropriate ORTModel class for task."""
        from optimum.onnxruntime import (
            ORTModel,
            ORTModelForAudioClassification,
            ORTModelForCausalLM,
            ORTModelForFeatureExtraction,
            ORTModelForImageClassification,
            ORTModelForQuestionAnswering,
            ORTModelForSeq2SeqLM,
            ORTModelForSequenceClassification,
            ORTModelForTokenClassification,
        )

        task_to_model = {
            "text-classification": ORTModelForSequenceClassification,
            "token-classification": ORTModelForTokenClassification,
            "question-answering": ORTModelForQuestionAnswering,
            "text-generation": ORTModelForCausalLM,
            "text2text-generation": ORTModelForSeq2SeqLM,
            "translation": ORTModelForSeq2SeqLM,
            "summarization": ORTModelForSeq2SeqLM,
            "image-classification": ORTModelForImageClassification,
            "audio-classification": ORTModelForAudioClassification,
            "feature-extraction": ORTModelForFeatureExtraction,
        }

        return task_to_model.get(task, ORTModel)


def load_optimum_model(
    onnx_path: str, task: str = "auto", device: str = "cpu", **kwargs: Any
) -> tuple[Any, Any]:
    """Convenience function to load an ONNX model for Optimum inference.

    Args:
        onnx_path: Path to ONNX model exported with ModelExport
        task: Task type (auto-detected if not specified)
        device: Device to run on ('cpu' or 'cuda')
        **kwargs: Additional arguments for ORTModel

    Returns:
        Tuple of (model, preprocessor)

    Example:
        >>> model, tokenizer = load_optimum_model("bert.onnx")
        >>> inputs = tokenizer("Hello world!", return_tensors="pt")
        >>> outputs = model(**inputs)
    """
    return OptimumONNXModel.from_onnx(onnx_path, task, device, **kwargs)
