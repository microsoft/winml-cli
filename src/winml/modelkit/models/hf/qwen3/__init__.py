# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Qwen3 transformer-only export + genai bundle support.

Modules:
  qwen_transformer_only  — OnnxConfig, build config, composite model class.
  qwen3_modeling         — winml-owned Qwen3 module definitions (forward bindings).
  qwen3_export_ops       — custom ONNX symbolic ops (LpNorm, GQA, 1x1 Conv).
  genai                  — genai_config.json generator + bundle assembler.
"""

from .genai import (
    DecoderIOMapping,
    PipelineStage,
    build_genai_config,
    build_qwen3_transformer_only_stages,
    write_genai_bundle,
)


__all__ = [
    "DecoderIOMapping",
    "PipelineStage",
    "build_genai_config",
    "build_qwen3_transformer_only_stages",
    "write_genai_bundle",
]
