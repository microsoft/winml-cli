# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Generate an onnxruntime-genai bundle for a transformer-only decoder pipeline.

The bundle is a directory that ``onnxruntime-genai`` can load directly via
``og.Config(str(bundle_dir))``.  It contains:

  genai_config.json    — pipeline config consumed by onnxruntime-genai
  ctx.onnx             — prefill/context ONNX (built by winml-cli)
  iter.onnx            — iteration/decode ONNX (built by winml-cli)
  embeddings.onnx      — embedding-lookup ONNX (placeholder; copy externally)
  lm_head.onnx         — lm_head ONNX (placeholder; copy externally)
  tokenizer.json       — HF tokenizer files (downloaded from the model repo)
  tokenizer_config.json
  vocab.json / merges.txt / generation_config.json

The pipeline follows the same 4-stage layout as the reference bundle:

  input_ids → [embeddings] → input_hidden_states
           → [context | iterator] → output_hidden_states + present KVs
           → [lm_head] → logits

The context stage runs on the prompt (prefill); the iterator stage runs on each
subsequent decode step.  Both share the same KV cache buffer via genai's
``past_present_share_buffer`` mode.

Public API::

    from winml.modelkit.models.hf.qwen3.genai import (
        build_genai_config,
        build_qwen3_transformer_only_stages,
        write_genai_bundle,
        DecoderIOMapping,
        PipelineStage,
    )

    # High-level: derive everything from the built ONNX files
    stages, decoder_io = build_qwen3_transformer_only_stages(
        ctx_path, iter_path, num_layers=hf_config.num_hidden_layers
    )
    cfg = build_genai_config(
        hf_config, max_cache_len=256, prefill_seq_len=64,
        pipeline=stages, decoder_io=decoder_io,
    )

    # Or one-shot bundle assembly
    write_genai_bundle(
        Path("out/bundle"),
        context_onnx=ctx_path,
        iterator_onnx=iter_path,
        model_id="Qwen/Qwen3-0.6B",
        max_cache_len=256,
        prefill_seq_len=64,
        embeddings_src=emb_path,   # None = skip (add later)
        lm_head_src=lmh_path,      # None = skip (add later)
    )
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Default filenames inside the bundle directory.
DEFAULT_EMBEDDINGS_FILENAME = "embeddings.onnx"
DEFAULT_CONTEXT_FILENAME = "ctx.onnx"
DEFAULT_ITERATOR_FILENAME = "iter.onnx"
DEFAULT_LM_HEAD_FILENAME = "lm_head.onnx"

# Tokenizer files written by AutoTokenizer.save_pretrained.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
]

# Regex for detecting indexed tensor names such as ``past_keys_3``.
_KV_INDEXED_RE = re.compile(r"^(.+?)(\d+)$")


# ---------------------------------------------------------------------------
# Pipeline data structures
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """One stage in an onnxruntime-genai multi-model pipeline.

    Attributes:
        name: Stage key used inside the ``pipeline`` list of ``genai_config.json``.
        filename: ONNX filename inside the bundle directory.
        run_on_prompt: Whether genai runs this stage during the prefill pass.
        run_on_token_gen: Whether genai runs this stage during decode steps.
        inputs: Actual ONNX input tensor names (not format strings).
        outputs: Actual ONNX output tensor names (not format strings).
        is_lm_head: Set ``True`` for the final language-model head stage.
    """

    name: str
    filename: str
    run_on_prompt: bool
    run_on_token_gen: bool
    inputs: list[str]
    outputs: list[str]
    is_lm_head: bool = False
    session_options: dict | None = None
    """Per-stage ORT session options (e.g. provider_options for QNN).

    When set, emitted verbatim as the ``session_options`` key in the
    ``genai_config.json`` pipeline stage.  Leave ``None`` (default) for
    stages that should run on the default (CPU) provider.
    """

    def to_dict(self) -> dict:
        """Serialize to the dict format expected by ``genai_config.json``."""
        d: dict = {
            "filename": self.filename,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "run_on_prompt": self.run_on_prompt,
            "run_on_token_gen": self.run_on_token_gen,
        }
        if self.session_options:
            d["session_options"] = self.session_options
        if self.is_lm_head:
            d["is_lm_head"] = True
        return d


@dataclass
class DecoderIOMapping:
    """Maps genai's abstract I/O concepts to ONNX tensor name format strings.

    The ``*_names`` fields use ``%d`` as the layer-index placeholder, which is
    the convention genai uses to expand per-layer KV cache tensor names
    (e.g. ``"past_keys_%d"`` → ``"past_keys_0"``, ``"past_keys_1"``, …).

    All fields default to the names produced by the Qwen3 transformer-only
    export.
    """

    input_ids: str = "input_ids"
    past_sequence_length: str = "past_seq_len"
    total_sequence_length: str = "total_seq_len"
    past_key_names: str = "past_keys_%d"
    past_value_names: str = "past_values_%d"
    logits: str = "logits"
    present_key_names: str = "present_keys_%d"
    present_value_names: str = "present_values_%d"

    def inputs_dict(self) -> dict:
        """Return the ``decoder.inputs`` mapping dict for ``genai_config.json``."""
        return {
            "input_ids": self.input_ids,
            "past_sequence_length": self.past_sequence_length,
            "total_sequence_length": self.total_sequence_length,
            "past_key_names": self.past_key_names,
            "past_value_names": self.past_value_names,
        }

    def outputs_dict(self) -> dict:
        """Return the ``decoder.outputs`` mapping dict for ``genai_config.json``."""
        return {
            "logits": self.logits,
            "present_key_names": self.present_key_names,
            "present_value_names": self.present_value_names,
        }


# ---------------------------------------------------------------------------
# Generic config builder
# ---------------------------------------------------------------------------


def build_genai_config(
    hf_config: Any,
    *,
    max_cache_len: int,
    prefill_seq_len: int | None = None,
    pipeline: list[PipelineStage],
    decoder_io: DecoderIOMapping | None = None,
) -> dict:
    """Build a ``genai_config.json`` dict for any decoder-pipeline model.

    This function is architecture-agnostic: the caller supplies the pipeline
    stages and the I/O name mapping so no tensor names are hardcoded here.

    Args:
        hf_config: A ``transformers.PretrainedConfig``.  Reads:
            ``num_hidden_layers``, ``hidden_size``, ``num_attention_heads``,
            ``num_key_value_heads``, ``head_dim`` (optional, falls back to
            ``hidden_size // num_attention_heads``), ``bos_token_id``,
            ``eos_token_id``, ``pad_token_id``, ``vocab_size``.
        max_cache_len: Static KV cache length → ``context_length`` and
            ``search.max_length``.
        prefill_seq_len: When given, emits a ``sliding_window`` section with
            ``window_size=prefill_seq_len``.  Pass ``None`` to omit.
        pipeline: Ordered list of :class:`PipelineStage` describing each
            model in the genai pipeline.
        decoder_io: Format-string mapping from genai's abstract I/O names to
            actual ONNX tensor names.  Defaults to
            :class:`DecoderIOMapping` (the Qwen3 default names).

    Returns:
        A ``dict`` suitable for ``json.dumps`` as ``genai_config.json``.
    """
    if decoder_io is None:
        decoder_io = DecoderIOMapping()

    num_layers: int = hf_config.num_hidden_layers
    head_size: int = getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
    )

    eos_token_id = hf_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    pad_token_id = getattr(hf_config, "pad_token_id", None) or hf_config.bos_token_id

    decoder_section: dict = {
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "num_hidden_layers": num_layers,
        "head_size": head_size,
    }

    if prefill_seq_len is not None:
        decoder_section["sliding_window"] = {
            "window_size": prefill_seq_len,
            "pad_value": 0,
            "alignment": "left",
            "slide_inputs": True,
            "slide_key_value_cache": False,
        }

    decoder_section["inputs"] = decoder_io.inputs_dict()
    decoder_section["outputs"] = decoder_io.outputs_dict()
    decoder_section["pipeline"] = [{s.name: s.to_dict()} for s in pipeline]

    return {
        "model": {
            "type": "decoder-pipeline",
            "bos_token_id": hf_config.bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "vocab_size": hf_config.vocab_size,
            "context_length": max_cache_len,
            "decoder": decoder_section,
        },
        "search": {
            "max_length": max_cache_len,
            "min_length": 0,
            "do_sample": False,
            "past_present_share_buffer": True,
        },
    }


# ---------------------------------------------------------------------------
# ONNX introspection helpers
# ---------------------------------------------------------------------------


def _introspect_onnx_io(onnx_path: Path) -> tuple[list[str], list[str]]:
    """Return ``(input_names, output_names)`` from an ONNX model graph header.

    External data is intentionally not loaded — only the graph topology is read,
    so this is fast even for large quantized models.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required for ONNX introspection. "
            "Install it with: pip install onnx"
        ) from exc
    model = onnx.load(str(onnx_path), load_external_data=False)
    return (
        [inp.name for inp in model.graph.input],
        [out.name for out in model.graph.output],
    )


def _detect_format_patterns(names: list[str], num_layers: int) -> dict[str, str]:
    """Detect ``prefix%d`` patterns from a list of indexed tensor names.

    Scans *names* for entries matching ``<prefix><integer>`` where exactly
    *num_layers* consecutive zero-based indices are present.

    Returns:
        ``{prefix: "prefix%d"}`` for each qualifying group, in the order the
        prefixes first appear in *names*.  Only groups covering the full
        ``[0, num_layers)`` index range are returned.

    Examples::

        >>> _detect_format_patterns(
        ...     ["past_keys_0", "past_keys_1", "past_values_0", "past_values_1"],
        ...     num_layers=2,
        ... )
        {"past_keys_": "past_keys_%d", "past_values_": "past_values_%d"}
    """
    groups: dict[str, list[int]] = {}
    for name in names:
        m = _KV_INDEXED_RE.match(name)
        if m:
            prefix, idx = m.group(1), int(m.group(2))
            groups.setdefault(prefix, []).append(idx)

    return {
        prefix: f"{prefix}%d"
        for prefix, indices in groups.items()
        if len(indices) == num_layers and sorted(indices) == list(range(num_layers))
    }


def _sort_patterns_by_first_occurrence(patterns: dict[str, str], names: list[str]) -> list[str]:
    """Sort *patterns* keys by when ``<prefix>0`` first appears in *names*."""

    def _key(prefix: str) -> int:
        try:
            return names.index(f"{prefix}0")
        except ValueError:
            return len(names)

    return sorted(patterns.keys(), key=_key)


# ---------------------------------------------------------------------------
# Per-EP stage session_options helpers
# ---------------------------------------------------------------------------


def _qnn_stage_session_options(log_id: str, soc_model: str = "60") -> dict:
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


# ---------------------------------------------------------------------------
# Qwen3 transformer-only pipeline factory
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
    """Build pipeline stages by introspecting the built ONNX models.

    Reads actual tensor names from *context_onnx* and *iterator_onnx* so the
    generated ``genai_config.json`` can never drift out of sync with the real
    model I/O — no tensor names are hardcoded.

    Args:
        context_onnx: Path to the built prefill/context ONNX.
        iterator_onnx: Path to the built decode/iterator ONNX.
        num_layers: Number of transformer layers (``hf_config.num_hidden_layers``).
        context_filename: Bundle filename for the context model.
        iterator_filename: Bundle filename for the iterator model.
        embeddings_filename: Bundle filename for the embeddings model.
        lm_head_filename: Bundle filename for the lm_head model.
        ep: Execution provider for the transformer stages.  ``"qnn"`` injects
            QNN HTP ``session_options`` into the ``context`` and ``iterator``
            stages so they run on the NPU while ``embeddings`` and ``lm_head``
            continue on CPU.  ``"cpu"`` (default) omits ``session_options``
            from all stages.
        soc_model: Snapdragon SoC model number forwarded to the QNN backend
            when ``ep="qnn"``.  Default ``"60"`` targets Snapdragon 8 Gen 3.

    Returns:
        ``(stages, decoder_io)`` — a 4-element :class:`PipelineStage` list and
        the :class:`DecoderIOMapping` derived from the introspected tensor names.
    """
    ctx_inputs, ctx_outputs = _introspect_onnx_io(Path(context_onnx))
    iter_inputs, iter_outputs = _introspect_onnx_io(Path(iterator_onnx))

    # Detect per-layer KV format-string patterns in the context model.
    input_patterns = _detect_format_patterns(ctx_inputs, num_layers)
    output_patterns = _detect_format_patterns(ctx_outputs, num_layers)

    in_sorted = _sort_patterns_by_first_occurrence(input_patterns, ctx_inputs)
    out_sorted = _sort_patterns_by_first_occurrence(output_patterns, ctx_outputs)

    past_key_fmt = input_patterns[in_sorted[0]] if len(in_sorted) > 0 else "past_keys_%d"
    past_val_fmt = input_patterns[in_sorted[1]] if len(in_sorted) > 1 else "past_values_%d"
    pres_key_fmt = output_patterns[out_sorted[0]] if len(out_sorted) > 0 else "present_keys_%d"
    pres_val_fmt = output_patterns[out_sorted[1]] if len(out_sorted) > 1 else "present_values_%d"

    # Non-indexed inputs: hidden-state tensor + scalar seq-length scalars.
    non_indexed = [n for n in ctx_inputs if not _KV_INDEXED_RE.match(n)]
    seq_len_names = [n for n in non_indexed if re.search(r"seq|len", n, re.IGNORECASE)]
    hidden_state_in = next(
        (n for n in non_indexed if n not in seq_len_names), "input_hidden_states"
    )
    past_seq_name = next((n for n in seq_len_names if "past" in n.lower()), "past_seq_len")
    total_seq_name = next((n for n in seq_len_names if "total" in n.lower()), "total_seq_len")

    # Non-indexed output: hidden-state output of the transformer stack.
    hidden_state_out = next(
        (n for n in ctx_outputs if not _KV_INDEXED_RE.match(n)), "output_hidden_states"
    )

    decoder_io = DecoderIOMapping(
        past_sequence_length=past_seq_name,
        total_sequence_length=total_seq_name,
        past_key_names=past_key_fmt,
        past_value_names=past_val_fmt,
        present_key_names=pres_key_fmt,
        present_value_names=pres_val_fmt,
    )

    # Per-stage session_options: NPU stages get QNN config; CPU and others get None.
    ctx_session_opts: dict | None = None
    iter_session_opts: dict | None = None
    if ep == "qnn":
        ctx_session_opts = _qnn_stage_session_options(
            "onnxruntime-genai.context", soc_model=soc_model
        )
        iter_session_opts = _qnn_stage_session_options(
            "onnxruntime-genai.iterator", soc_model=soc_model
        )

    stages: list[PipelineStage] = [
        PipelineStage(
            name="embeddings",
            filename=embeddings_filename,
            run_on_prompt=True,
            run_on_token_gen=True,
            inputs=[decoder_io.input_ids],
            outputs=[hidden_state_in],
        ),
        PipelineStage(
            name="context",
            filename=context_filename,
            run_on_prompt=True,
            run_on_token_gen=False,
            inputs=ctx_inputs,
            outputs=ctx_outputs,
            session_options=ctx_session_opts,
        ),
        PipelineStage(
            name="iterator",
            filename=iterator_filename,
            run_on_prompt=False,
            run_on_token_gen=True,
            inputs=iter_inputs,
            outputs=iter_outputs,
            session_options=iter_session_opts,
        ),
        PipelineStage(
            name="lm_head",
            filename=lm_head_filename,
            run_on_prompt=True,
            run_on_token_gen=True,
            inputs=[hidden_state_out],
            outputs=[decoder_io.logits],
            is_lm_head=True,
        ),
    ]
    return stages, decoder_io


# ---------------------------------------------------------------------------
# Bundle assembler
# ---------------------------------------------------------------------------


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
) -> Path:
    """Assemble a complete ``onnxruntime-genai`` bundle in *output_dir*.

    Copies the winml-built transformer ONNX files, placeholder embedding /
    lm_head models (when provided), HF tokenizer files, and writes
    ``genai_config.json``.  Tensor names in the config are derived by
    introspecting the built ONNX files rather than being hardcoded.

    Args:
        output_dir: Destination directory (created if absent).
        context_onnx: Path to the built prefill/context ONNX.
        iterator_onnx: Path to the built decode/iterator ONNX.
        model_id: HuggingFace model ID or local path for config + tokenizer.
        max_cache_len: Static KV cache length (= ``context_length`` in genai).
        prefill_seq_len: Prefill sequence length (= ``sliding_window.window_size``).
        embeddings_src: Source path of the embeddings ONNX.  ``None`` = skip.
        lm_head_src: Source path of the lm_head ONNX.  ``None`` = skip.
        context_filename: Bundle filename for the context model.
        iterator_filename: Bundle filename for the iterator model.
        embeddings_filename: Bundle filename for the embeddings model.
        lm_head_filename: Bundle filename for the lm_head model.
        ep: Execution provider for the transformer (context/iterator) stages.
            ``"qnn"`` injects QNN HTP ``session_options`` so those stages run
            on the NPU while embeddings and lm_head run on CPU.
            ``"cpu"`` (default) omits ``session_options`` (all stages on CPU).
        soc_model: Snapdragon SoC model passed to the QNN backend when
            ``ep="qnn"``.  Default ``"60"`` = Snapdragon 8 Gen 3 / X Elite.

    Returns:
        Path to the written ``genai_config.json``.
    """
    from transformers import AutoConfig, AutoTokenizer

    from winml.modelkit.onnx import copy_onnx_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_onnx = Path(context_onnx)
    iterator_onnx = Path(iterator_onnx)

    # 1. Copy winml-built transformer ONNX files.
    logger.info("Copying context ONNX: %s -> %s", context_onnx.name, context_filename)
    copy_onnx_model(context_onnx, output_dir / context_filename)

    logger.info("Copying iterator ONNX: %s -> %s", iterator_onnx.name, iterator_filename)
    copy_onnx_model(iterator_onnx, output_dir / iterator_filename)

    # 2. Copy placeholder models (embeddings + lm_head).
    if embeddings_src is not None:
        logger.info("Copying embeddings: %s -> %s", Path(embeddings_src).name, embeddings_filename)
        copy_onnx_model(Path(embeddings_src), output_dir / embeddings_filename)
    else:
        logger.warning(
            "embeddings_src not provided — '%s' is missing from bundle.",
            embeddings_filename,
        )

    if lm_head_src is not None:
        logger.info("Copying lm_head: %s -> %s", Path(lm_head_src).name, lm_head_filename)
        copy_onnx_model(Path(lm_head_src), output_dir / lm_head_filename)
    else:
        logger.warning(
            "lm_head_src not provided — '%s' is missing from bundle.",
            lm_head_filename,
        )

    # 3. Save tokenizer files from the HF snapshot.
    logger.info("Saving tokenizer from '%s' to %s", model_id, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    # 4. Build pipeline stages by introspecting the source ONNX files.
    hf_config = AutoConfig.from_pretrained(model_id)
    stages, decoder_io = build_qwen3_transformer_only_stages(
        context_onnx,
        iterator_onnx,
        num_layers=hf_config.num_hidden_layers,
        context_filename=context_filename,
        iterator_filename=iterator_filename,
        embeddings_filename=embeddings_filename,
        lm_head_filename=lm_head_filename,
        ep=ep,
        soc_model=soc_model,
    )

    # 5. Write genai_config.json.
    config = build_genai_config(
        hf_config,
        max_cache_len=max_cache_len,
        prefill_seq_len=prefill_seq_len,
        pipeline=stages,
        decoder_io=decoder_io,
    )
    config_path = output_dir / "genai_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Wrote genai_config.json -> %s", config_path)

    _log_bundle_summary(output_dir, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_bundle_summary(bundle_dir: Path, config_path: Path) -> None:
    """Print a human-readable summary of the assembled bundle."""
    files = sorted(bundle_dir.iterdir())
    lines = [f"\n=== genai bundle: {bundle_dir} ==="]
    for f in files:
        size_kb = f.stat().st_size / 1024
        tag = ""
        if f.name == "genai_config.json":
            tag = "  <- pipeline config"
        elif f.name.endswith(".onnx"):
            tag = "  <- ONNX graph"
        elif f.name.endswith(".data"):
            tag = "  <- ONNX external weights"
        lines.append(f"  {f.name:<45} {size_kb:>8.1f} KB{tag}")
    lines.append(f"\nConfig written to: {config_path}")
    logger.info("\n".join(lines))


__all__ = [
    "DEFAULT_CONTEXT_FILENAME",
    "DEFAULT_EMBEDDINGS_FILENAME",
    "DEFAULT_ITERATOR_FILENAME",
    "DEFAULT_LM_HEAD_FILENAME",
    "DecoderIOMapping",
    "PipelineStage",
    "build_genai_config",
    "build_qwen3_transformer_only_stages",
    "write_genai_bundle",
]
