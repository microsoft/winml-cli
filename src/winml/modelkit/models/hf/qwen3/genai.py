# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Qwen3 genai bundle support — thin shim over :mod:`winml.modelkit.utils.genai`.

All generic logic (``PipelineStage``, ``DecoderIOMapping``, ``build_genai_config``,
``build_decoder_pipeline_stages``, ``write_genai_bundle``) lives in
:mod:`winml.modelkit.utils.genai` so it can be reused by other model families.

This module re-exports that API unchanged and adds
``build_qwen3_transformer_only_stages`` as a backward-compatible alias for
``build_decoder_pipeline_stages``.  New code should prefer the generic names.
"""

from __future__ import annotations

from ....utils.genai import (
    DEFAULT_CONTEXT_FILENAME,
    DEFAULT_EMBEDDINGS_FILENAME,
    DEFAULT_ITERATOR_FILENAME,
    DEFAULT_LM_HEAD_FILENAME,
    DecoderIOMapping,
    PipelineStage,
    build_decoder_pipeline_stages,
    build_genai_config,
    qnn_stage_session_options,
    write_genai_bundle,
)


# Backward-compatible alias: existing callers that import
# ``build_qwen3_transformer_only_stages`` continue to work unchanged.
build_qwen3_transformer_only_stages = build_decoder_pipeline_stages

# Keep the private EP helper importable under its old name for any callers
# that referenced it before the rename.
_qnn_stage_session_options = qnn_stage_session_options

__all__ = [
    "DEFAULT_CONTEXT_FILENAME",
    "DEFAULT_EMBEDDINGS_FILENAME",
    "DEFAULT_ITERATOR_FILENAME",
    "DEFAULT_LM_HEAD_FILENAME",
    "DecoderIOMapping",
    "PipelineStage",
    "build_decoder_pipeline_stages",
    "build_genai_config",
    "build_qwen3_transformer_only_stages",
    "qnn_stage_session_options",
    "write_genai_bundle",
]
