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
stages run on an NPU backend, so this is where the per-EP ``session_options``
are constructed.  Three NPU execution providers are supported for the
transformer (context/iterator) stages:

* **QNN HTP** — Qualcomm Snapdragon NPU (``ep="qnn"``).
* **VitisAI** — AMD Ryzen AI NPU (``ep="vitisai"``).
* **OpenVINO** — Intel NPU via the plugin EP (``ep="openvino"``).

Keeping the EP-specific logic here lets the generic utilities stay universal
while the Qwen3 bundle emits the correct per-EP ``genai_config.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ....onnx import strip_node_attrs
from ....utils.constants import normalize_ep_name
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
    GenaiTarget,
    GenaiTransformerSpec,
    register_genai_bundle,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import onnx


# ---------------------------------------------------------------------------
# Qwen3-specific NPU execution-provider routing (QNN / VitisAI / OpenVINO)
# ---------------------------------------------------------------------------

_OPENVINO_CONFIG_ROLE_KEYS = frozenset({"CTX", "ITER", "HEAD"})


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


def vitisai_stage_session_options(log_id: str) -> dict:
    """Return the ``session_options`` block that routes a stage to the AMD NPU.

    Routes a Qwen3 transformer stage to the AMD Ryzen AI NPU via the VitisAI
    execution provider.  The provider options match the AMD reference inference
    configuration (``waic_target_vaiml_cpp_me`` VAIML C++ backend with the
    XMC runner and linear-slice disabled).

    Args:
        log_id: ORT log identifier (shown in ORT logs), e.g.
            ``"onnxruntime-genai.context"``.

    Returns:
        Dict suitable for the ``session_options`` key of a pipeline stage in
        ``genai_config.json``.
    """
    return {
        "log_id": log_id,
        "provider_options": [
            {
                "vitisai": {
                    "target": "waic_target_vaiml_cpp_me",
                    "xmc_runner_config": "1",
                    "no_linear_slice": "1",
                }
            }
        ],
        "intra_op_num_threads": 8,
        "inter_op_num_threads": 1,
    }


def build_npu_load_config(
    custom_config_path: str | Path | None = None,
    *,
    model_role: str | None = None,
    weights_path: str | Path | None = None,
) -> str:
    """Build the JSON-string ``load_config`` for the OpenVINO NPU plugin.

    Accepts a flat configuration (e.g. ``{"NPU": {...}}``) or a configuration
    with ``CTX``/``ITER`` sections. A role-based file must contain the requested
    role; missing sections are errors rather than silently using defaults.
    ``HEAD`` is recognized only to detect role-based files; the LM head stays
    on CPU. No legacy provider setup or tuning flags are injected.

    Args:
        custom_config_path: Optional UTF-8 JSON file. Without one, only the
            driver compiler default and an explicitly supplied weights path
            are included. Custom values take precedence over defaults.
        model_role: ``"CTX"`` or ``"ITER"``; required for role-based files.
        weights_path: Default external-weights directory. A ``WEIGHTS_PATH``
            in the file takes precedence; relative file values resolve against
            that file's directory. Emitted paths are absolute so loading a
            derived bundle does not change their meaning.

    Returns:
        Serialized OpenVINO load configuration, not a filename.
    """
    if model_role not in (None, "CTX", "ITER"):
        raise ValueError("OpenVINO model_role must be 'CTX' or 'ITER'")

    config: dict = {}
    config_path = (
        Path(custom_config_path).expanduser().resolve() if custom_config_path is not None else None
    )
    if config_path is not None:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid OpenVINO load config JSON in {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise TypeError("OpenVINO load config must be a JSON object")
        if _OPENVINO_CONFIG_ROLE_KEYS.intersection(config):
            if model_role is None:
                raise ValueError("A role-based OpenVINO load config requires model_role")
            if model_role not in config:
                raise ValueError(f"OpenVINO load config is missing the {model_role} section")
            config = config[model_role]
            if not isinstance(config, dict):
                raise TypeError(f"OpenVINO {model_role} configuration must be a JSON object")

    npu_config = config.setdefault("NPU", {})
    if not isinstance(npu_config, dict):
        raise TypeError("OpenVINO NPU configuration must be a JSON object")
    npu_config.setdefault("NPU_COMPILER_TYPE", "DRIVER")

    if "WEIGHTS_PATH" in npu_config:
        configured_weights = npu_config["WEIGHTS_PATH"]
        if not isinstance(configured_weights, str) or not configured_weights.strip():
            raise ValueError("OpenVINO WEIGHTS_PATH must be a non-empty string")
        resolved_weights = Path(configured_weights).expanduser()
        if not resolved_weights.is_absolute() and config_path is not None:
            resolved_weights = config_path.parent / resolved_weights
        npu_config["WEIGHTS_PATH"] = str(resolved_weights.resolve())
    elif weights_path is not None:
        npu_config["WEIGHTS_PATH"] = str(Path(weights_path).expanduser().resolve())

    # Stable serialization keeps equivalent CTX/ITER options shareable even
    # when their keys occur in a different order in the user's JSON file.
    return json.dumps(config, sort_keys=True)


def openvino_stage_session_options(
    log_id: str,
    *,
    custom_config_path: str | Path | None = None,
    model_role: str | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    """Return session options routing a transformer stage to the Intel NPU.

    ``load_config`` is a JSON string containing OpenVINO properties, not ORT
    session entries. Plugin registration, ABI device binding, and EPContext
    compilation remain owned by the existing session/compiler infrastructure.
    See :func:`build_npu_load_config` for the optional configuration arguments.
    """
    return {
        "log_id": log_id,
        "provider_options": [
            {
                "openvino": {
                    "device_type": "NPU",
                    "load_config": build_npu_load_config(
                        custom_config_path,
                        model_role=model_role,
                        weights_path=weights_path,
                    ),
                }
            }
        ],
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    }


def _stage_session_options(
    ep: str,
    soc_model: str,
    *,
    openvino_config_path: str | Path | None = None,
    openvino_weights_path: str | Path | None = None,
) -> tuple[dict | None, dict | None]:
    """Return ``(context, iterator)`` session_options for the given EP.

    Routes the Qwen3 transformer (context/iterator) stages to an NPU backend:

    * ``ep="qnn"`` -> Qualcomm QNN HTP (``soc_model`` selects the Snapdragon SoC).
    * ``ep="vitisai"`` -> AMD Ryzen AI NPU.
    * ``ep="openvino"`` -> Intel NPU, with optional per-role load configuration.

    Any other value (e.g. ``"cpu"``) leaves the stages on the default CPU
    provider.  Short aliases and full ``*ExecutionProvider`` names are both
    accepted (normalized via :func:`normalize_ep_name`).
    """
    canonical = normalize_ep_name(ep)
    if canonical != "OpenVINOExecutionProvider" and (
        openvino_config_path is not None or openvino_weights_path is not None
    ):
        raise ValueError("OpenVINO configuration requires ep='openvino'")
    if canonical == "QNNExecutionProvider":
        return (
            qnn_stage_session_options("onnxruntime-genai.context", soc_model=soc_model),
            qnn_stage_session_options("onnxruntime-genai.iterator", soc_model=soc_model),
        )
    if canonical == "VitisAIExecutionProvider":
        return (
            vitisai_stage_session_options("onnxruntime-genai.context"),
            vitisai_stage_session_options("onnxruntime-genai.iterator"),
        )
    if canonical == "OpenVINOExecutionProvider":
        return (
            openvino_stage_session_options(
                "onnxruntime-genai.context",
                custom_config_path=openvino_config_path,
                model_role="CTX",
                weights_path=openvino_weights_path,
            ),
            openvino_stage_session_options(
                "onnxruntime-genai.iterator",
                custom_config_path=openvino_config_path,
                model_role="ITER",
                weights_path=openvino_weights_path,
            ),
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
    openvino_config_path: str | Path | None = None,
    openvino_weights_path: str | Path | None = None,
) -> tuple[list[PipelineStage], DecoderIOMapping]:
    """Build the Qwen3 4-stage pipeline, routing ctx/iter to the NPU per ``ep``.

    Qwen3-specific wrapper over
    :func:`winml.modelkit.utils.genai.build_decoder_pipeline_stages` that injects
    the NPU ``session_options`` for the transformer stages.  Tensor names are
    still discovered by introspecting the ONNX graphs, so nothing is hardcoded.

    Args:
        context_onnx: Path to the built prefill/context ONNX.
        iterator_onnx: Path to the built decode/iterator ONNX.
        num_layers: Number of transformer layers (``hf_config.num_hidden_layers``).
        context_filename: Bundle filename for the context model.
        iterator_filename: Bundle filename for the iterator model.
        embeddings_filename: Bundle filename for the embeddings model.
        lm_head_filename: Bundle filename for the lm_head model.
        ep: NPU execution provider for the ``context``/``iterator`` stages —
            ``"qnn"`` (Qualcomm), ``"vitisai"`` (AMD), or ``"openvino"`` (Intel)
            injects that EP's ``session_options`` so those stages run on the NPU while
            ``embeddings`` and ``lm_head`` stay on CPU.  ``"cpu"`` (default)
            omits them.
        soc_model: Snapdragon SoC model number forwarded to the QNN backend when
            ``ep="qnn"``.  Default ``"60"`` targets Snapdragon 8 Gen 3.  Ignored
            for non-QNN EPs.
        openvino_config_path: Optional OpenVINO JSON file shared by both stages
            or containing separate ``CTX``/``ITER`` sections. Only for OpenVINO.
        openvino_weights_path: Optional default external-weights directory for
            OpenVINO. Explicit ``WEIGHTS_PATH`` values in the JSON take precedence.

    Returns:
        ``(stages, decoder_io)`` — see
        :func:`~winml.modelkit.utils.genai.build_decoder_pipeline_stages`.
    """
    ctx_opts, iter_opts = _stage_session_options(
        ep,
        soc_model,
        openvino_config_path=openvino_config_path,
        openvino_weights_path=openvino_weights_path,
    )
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
    openvino_config_path: str | Path | None = None,
    openvino_weights_path: str | Path | None = None,
) -> Path:
    """Assemble a Qwen3 genai bundle, routing ctx/iter to the NPU per ``ep``.

    Qwen3-specific wrapper over
    :func:`winml.modelkit.utils.genai.write_genai_bundle` that supplies the NPU
    ``session_options`` for the transformer stages.  See the generic function for
    the description of every other argument.

    Args:
        ep: NPU execution provider routing the transformer (context/iterator)
            stages — ``"qnn"`` (Qualcomm HTP), ``"vitisai"`` (AMD Ryzen AI),
            or ``"openvino"`` (Intel NPU);
            ``"cpu"`` (default) keeps every stage on CPU.
        soc_model: Snapdragon SoC model passed to the QNN backend when
            ``ep="qnn"``.  Default ``"60"`` = Snapdragon 8 Gen 3 / X Elite.
            Ignored for non-QNN EPs.
        transformer_onnx_passes: Optional ONNX graph transforms applied to the
            copied context/iterator models before ``genai_config.json`` is
            written.  Forwarded verbatim to the generic assembler.
        openvino_config_path: Optional flat or per-role OpenVINO JSON file.
            Its contents are embedded in stage provider options; the original
            configuration file is not required at runtime. Different role
            options prevent grouped compilation in the current runtime.
        openvino_weights_path: Default external-weights directory for OpenVINO;
            defaults to the output bundle directory. Explicit ``WEIGHTS_PATH``
            values in the JSON take precedence. Other EPs reject these options.

    Returns:
        Path to the written ``genai_config.json``.
    """
    if normalize_ep_name(ep) == "OpenVINOExecutionProvider" and openvino_weights_path is None:
        openvino_weights_path = output_dir
    ctx_opts, iter_opts = _stage_session_options(
        ep,
        soc_model,
        openvino_config_path=openvino_config_path,
        openvino_weights_path=openvino_weights_path,
    )
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
    "build_npu_load_config",
    "build_qwen3_transformer_only_stages",
    "openvino_stage_session_options",
    "qnn_stage_session_options",
    "strip_gqa_default_attrs",
    "vitisai_stage_session_options",
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
        supported_targets=(
            GenaiTarget(ep="qnn", device="npu"),  # Qualcomm Snapdragon NPU
            GenaiTarget(ep="vitisai", device="npu"),  # AMD Ryzen AI NPU
            GenaiTarget(ep="openvino", device="npu"),  # Intel NPU
            GenaiTarget(ep="cpu", device="cpu"),
        ),
        transformer_onnx_passes=(strip_gqa_default_attrs,),
        max_cache_len=2048,
        prefill_seq_len=64,
        soc_model="60",
    )
)
