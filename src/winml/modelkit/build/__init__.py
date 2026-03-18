"""Build module — core pipeline API for building ONNX models.

This module owns the build pipeline (export -> optimize -> quantize -> compile).
CLI commands and WinMLAutoModel.from_pretrained() are consumers of this API.

Usage:
    from winml.modelkit.build import build_hf_model, build_onnx_model, BuildResult

    # Build from HuggingFace model
    result = build_hf_model(
        config=config,
        output_dir=Path("output/"),
        model_id="microsoft/resnet-50",
    )

    # Build from pre-exported ONNX model
    result = build_onnx_model(
        onnx_path=Path("model.onnx"),
        config=config,
        output_dir=Path("output/"),
    )
"""

from .hf import BuildResult, build_hf_model
from .onnx import build_onnx_model


__all__ = ["BuildResult", "build_hf_model", "build_onnx_model"]
