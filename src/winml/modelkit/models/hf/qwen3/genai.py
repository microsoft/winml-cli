# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Qwen3 genai bundle support built on :mod:`winml.modelkit.utils.genai`.

The generic, execution-provider-agnostic machinery (``PipelineStage``,
``DecoderIOMapping``, ``build_genai_config``, ``build_decoder_pipeline_stages``,
``write_genai_bundle``) lives in :mod:`winml.modelkit.utils.genai` so it can be
reused by other model families.

This module adds the **Qwen3-specific** layer on top: the Qwen3 transformer
stages target the QNN HTP (NPU) backend, so this is where the QNN
``session_options`` are constructed.  Keeping the EP-specific logic here lets the
generic utilities stay universal while the Qwen3 bundle keeps emitting the exact
same ``genai_config.json`` as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....onnx import strip_node_attrs
from ....utils.genai import (
    DEFAULT_CONTEXT_FILENAME,
    DEFAULT_EMBEDDINGS_FILENAME,
    DEFAULT_ITERATOR_FILENAME,
    DEFAULT_LM_HEAD_FILENAME,
    DecoderIOMapping,
    PipelineStage,
    build_decoder_pipeline_stages,
    build_genai_config,
)
from ....utils.genai import (
    write_genai_bundle as _write_genai_bundle,
)
from ...winml.genai_bundle import (
    GenaiBundleRecipe,
    GenaiCompanionSpec,
    GenaiTransformerSpec,
    register_genai_bundle,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import onnx


# ---------------------------------------------------------------------------
# Qwen3-specific QNN execution-provider routing
# ---------------------------------------------------------------------------


def qnn_stage_session_options(log_id: str, soc_model: str = "60") -> dict:
    """Return the ``session_options`` block that routes a stage to QNN HTP.

    Args:
        log_id: ORT log identifier (shown in ORT logs), e.g.
            ``"onnxruntime-genai.context"``.
        soc_model: Snapdragon SoC model number passed to the QNN HTP backend.
            ``"60"`` targets Snapdragon 8 Gen 3 (X Elite).  Change for other
            SoCs (e.g. ``"55"`` for 8 Gen 2, ``"73"`` for 8 Elite).

    Returns:
        Dict suitable for the ``session_options`` key of a pipeline stage in
        ``genai_config.json``.
    """
    return {
        "log_id": log_id,
        "provider_options": [
            {
                "qnn": {
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "burst",
                    "htp_graph_finalization_optimization_mode": "3",
                    "soc_model": soc_model,
                }
            }
        ],
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    }


def _stage_session_options(ep: str, soc_model: str) -> tuple[dict | None, dict | None]:
    """Return ``(context, iterator)`` session_options for the given EP.

    ``ep="qnn"`` routes the transformer stages to the QNN HTP (NPU) backend; any
    other value (e.g. ``"cpu"``) leaves them on the default CPU provider.
    """
    if ep == "qnn":
        return (
            qnn_stage_session_options("onnxruntime-genai.context", soc_model=soc_model),
            qnn_stage_session_options("onnxruntime-genai.iterator", soc_model=soc_model),
        )
    return None, None


# ---------------------------------------------------------------------------
# Qwen3-specific ONNX graph passes
# ---------------------------------------------------------------------------

# Attributes that com.microsoft::GroupQueryAttention requires for Qwen3.
# Any other attributes (e.g. k_quant_type, local_window_size, qk_output,
# smooth_softmax, v_quant_type) are default-valued extras injected by the
# TorchScript exporter from the ORT op schema; strip them so the bundle
# matches the expected minimal attribute set.
_GQA_KEEP_ATTRS = frozenset({"do_rotary", "kv_num_heads", "num_heads"})


def strip_gqa_default_attrs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove exporter-injected default attributes from Qwen3 GQA nodes.

    A ``transformer_onnx_pass`` for :func:`write_genai_bundle`: strips every
    attribute from ``com.microsoft::GroupQueryAttention`` nodes except the ones
    Qwen3 actually needs (:data:`_GQA_KEEP_ATTRS`), removing the default-valued
    extras the TorchScript exporter injects from the ORT op schema.  Mutates
    *model* in-place and returns it for convenient chaining.
    """
    return strip_node_attrs(model, "GroupQueryAttention", _GQA_KEEP_ATTRS, domain="com.microsoft")


# ---------------------------------------------------------------------------
# Qwen3-specific stage factory + bundle assembler
# ---------------------------------------------------------------------------


def build_qwen3_transformer_only_stages(
    context_onnx: str | Path,
    iterator_onnx: str | Path,
    num_layers: int,
    *,
    context_filename: str = DEFAULT_CONTEXT_FILENAME,
    iterator_filename: str = DEFAULT_ITERATOR_FILENAME,
    embeddings_filename: str = DEFAULT_EMBEDDINGS_FILENAME,
    lm_head_filename: str = DEFAULT_LM_HEAD_FILENAME,
    ep: str = "cpu",
    soc_model: str = "60",
) -> tuple[list[PipelineStage], DecoderIOMapping]:
    """Build the Qwen3 4-stage pipeline, routing ctx/iter to QNN when ``ep="qnn"``.

    Qwen3-specific wrapper over
    :func:`winml.modelkit.utils.genai.build_decoder_pipeline_stages` that injects
    the QNN ``session_options`` for the transformer stages.  Tensor names are
    still discovered by introspecting the ONNX graphs, so nothing is hardcoded.

    Args:
        context_onnx: Path to the built prefill/context ONNX.
        iterator_onnx: Path to the built decode/iterator ONNX.
        num_layers: Number of transformer layers (``hf_config.num_hidden_layers``).
        context_filename: Bundle filename for the context model.
        iterator_filename: Bundle filename for the iterator model.
        embeddings_filename: Bundle filename for the embeddings model.
        lm_head_filename: Bundle filename for the lm_head model.
        ep: ``"qnn"`` injects QNN HTP ``session_options`` into the ``context``
            and ``iterator`` stages so they run on the NPU while ``embeddings``
            and ``lm_head`` stay on CPU.  ``"cpu"`` (default) omits them.
        soc_model: Snapdragon SoC model number forwarded to the QNN backend when
            ``ep="qnn"``.  Default ``"60"`` targets Snapdragon 8 Gen 3.

    Returns:
        ``(stages, decoder_io)`` — see
        :func:`~winml.modelkit.utils.genai.build_decoder_pipeline_stages`.
    """
    ctx_opts, iter_opts = _stage_session_options(ep, soc_model)
    return build_decoder_pipeline_stages(
        context_onnx,
        iterator_onnx,
        num_layers,
        context_filename=context_filename,
        iterator_filename=iterator_filename,
        embeddings_filename=embeddings_filename,
        lm_head_filename=lm_head_filename,
        context_session_options=ctx_opts,
        iterator_session_options=iter_opts,
    )


def write_genai_bundle(
    output_dir: str | Path,
    *,
    context_onnx: str | Path,
    iterator_onnx: str | Path,
    model_id: str,
    max_cache_len: int,
    prefill_seq_len: int,
    embeddings_src: str | Path | None = None,
    lm_head_src: str | Path | None = None,
    context_filename: str = DEFAULT_CONTEXT_FILENAME,
    iterator_filename: str = DEFAULT_ITERATOR_FILENAME,
    embeddings_filename: str = DEFAULT_EMBEDDINGS_FILENAME,
    lm_head_filename: str = DEFAULT_LM_HEAD_FILENAME,
    ep: str = "cpu",
    soc_model: str = "60",
    transformer_onnx_passes: Sequence[Callable[[onnx.ModelProto], onnx.ModelProto]] | None = None,
) -> Path:
    """Assemble a Qwen3 genai bundle, routing ctx/iter to QNN when ``ep="qnn"``.

    Qwen3-specific wrapper over
    :func:`winml.modelkit.utils.genai.write_genai_bundle` that supplies the QNN
    ``session_options`` for the transformer stages.  See the generic function for
    the description of every other argument.

    Args:
        ep: ``"qnn"`` routes the transformer (context/iterator) stages to the QNN
            HTP (NPU) backend; ``"cpu"`` (default) keeps every stage on CPU.
        soc_model: Snapdragon SoC model passed to the QNN backend when
            ``ep="qnn"``.  Default ``"60"`` = Snapdragon 8 Gen 3 / X Elite.
        transformer_onnx_passes: Optional ONNX graph transforms applied to the
            copied context/iterator models before ``genai_config.json`` is
            written.  Forwarded verbatim to the generic assembler.

    Returns:
        Path to the written ``genai_config.json``.
    """
    ctx_opts, iter_opts = _stage_session_options(ep, soc_model)
    return _write_genai_bundle(
        output_dir,
        context_onnx=context_onnx,
        iterator_onnx=iterator_onnx,
        model_id=model_id,
        max_cache_len=max_cache_len,
        prefill_seq_len=prefill_seq_len,
        embeddings_src=embeddings_src,
        lm_head_src=lm_head_src,
        context_filename=context_filename,
        iterator_filename=iterator_filename,
        embeddings_filename=embeddings_filename,
        lm_head_filename=lm_head_filename,
        context_session_options=ctx_opts,
        iterator_session_options=iter_opts,
        transformer_onnx_passes=transformer_onnx_passes,
    )


__all__ = [
    "DEFAULT_CONTEXT_FILENAME",
    "DEFAULT_EMBEDDINGS_FILENAME",
    "DEFAULT_ITERATOR_FILENAME",
    "DEFAULT_LM_HEAD_FILENAME",
    "QWEN3_GENAI_BUNDLE_RECIPE",
    "DecoderIOMapping",
    "PipelineStage",
    "build_decoder_pipeline_stages",
    "build_genai_config",
    "build_qwen3_transformer_only_stages",
    "qnn_stage_session_options",
    "strip_gqa_default_attrs",
    "write_genai_bundle",
]


# ---------------------------------------------------------------------------
# Genai-bundle recipe registration
# ---------------------------------------------------------------------------
#
# Register Qwen3 as a genai-bundle family so ``winml build`` can assemble the
# full onnxruntime-genai bundle in one command (see
# ``winml.modelkit.models.winml.genai_bundle``).  Registered at import time so
# merely importing ``winml.modelkit.models.hf`` populates the registry,
# mirroring the composite-model registration pattern.
QWEN3_GENAI_BUNDLE_RECIPE = register_genai_bundle(
    GenaiBundleRecipe(
        family="qwen3",
        transformer=GenaiTransformerSpec(
            model_type="qwen3_transformer_only",
            task="text-generation",
            precision="w8a16",
            context_sub_model="decoder_prefill",
            iterator_sub_model="decoder_gen",
        ),
        companions=(
            GenaiCompanionSpec(
                role="embeddings",
                model_type="qwen3_embeddings_only",
                task="feature-extraction",
                precision="fp32",
            ),
            GenaiCompanionSpec(
                role="lm_head",
                model_type="qwen3_lm_head_only",
                task="feature-extraction",
                precision="w4a32",
            ),
        ),
        assemble=write_genai_bundle,
        transformer_onnx_passes=(strip_gqa_default_attrs,),
        max_cache_len=2048,
        prefill_seq_len=64,
        soc_model="60",
    )
)
