# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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

from typing import Any

from .hf import BuildResult, build_hf_model
from .onnx import build_onnx_model


__all__ = [
    "BuildResult",
    "build_hf_model",
    "build_onnx_model",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "run_optimize_analyze_loop": (".common", "run_optimize_analyze_loop"),
    "write_module_summary": (".module_summary", "write_module_summary"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load build helpers to avoid pulling in heavy deps at import time."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(set(list(globals()) + __all__))
