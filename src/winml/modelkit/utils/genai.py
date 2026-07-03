# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Generic onnxruntime-genai bundle utilities for decoder-pipeline models.

The bundle is a directory that ``onnxruntime-genai`` can load directly via
``og.Config(str(bundle_dir))``.  It contains:

  genai_config.json    — pipeline config consumed by onnxruntime-genai
  ctx.onnx             — prefill/context ONNX
  iter.onnx            — iteration/decode ONNX
  embeddings.onnx      — embedding-lookup ONNX
  lm_head.onnx         — lm_head ONNX
  tokenizer.json       — HF tokenizer files (downloaded from the model repo)
  tokenizer_config.json
  vocab.json / merges.txt / generation_config.json

The pipeline follows the standard 4-stage decoder layout:

  input_ids → [embeddings] → input_hidden_states
           → [context | iterator] → output_hidden_states + present KVs
           → [lm_head] → logits

The context stage runs on the prompt (prefill); the iterator stage runs on each
subsequent decode step.  Both share the same KV cache buffer via genai's
``past_present_share_buffer`` mode.

Per-stage execution-provider routing (e.g. running the transformer stages on an
NPU) is expressed through the generic ``PipelineStage.session_options`` field and
is supplied by the caller — this module is itself execution-provider-agnostic and
hardcodes no EP-specific settings.

Public API::

    from winml.modelkit.utils.genai import (
        build_genai_config,
        build_decoder_pipeline_stages,
        write_genai_bundle,
        DecoderIOMapping,
        PipelineStage,
    )

    # Build stages by introspecting the ONNX I/O (no hardcoded tensor names)
    stages, decoder_io = build_decoder_pipeline_stages(
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
    """Per-stage ORT session options (e.g. execution-provider selection and
    provider_options).

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

    Defaults match the Qwen3 transformer-only export naming; override any field
    when building bundles for models with different tensor names.
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
            :class:`DecoderIOMapping` (the standard names).

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

    eos_token_id: int | list[int] = hf_config.eos_token_id
    # Pass lists through unchanged — ORT genai accepts a JSON array of EOS token
    # IDs and treats any of them as a valid stop signal.  Truncating to [0] would
    # silently discard secondary EOS tokens (e.g. Qwen3 uses [151645, 151643])
    # and cause generation to run until max_length instead of stopping early.

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
# Generic decoder-pipeline stage factory
# ---------------------------------------------------------------------------


def build_decoder_pipeline_stages(
    context_onnx: str | Path,
    iterator_onnx: str | Path,
    num_layers: int,
    *,
    context_filename: str = DEFAULT_CONTEXT_FILENAME,
    iterator_filename: str = DEFAULT_ITERATOR_FILENAME,
    embeddings_filename: str = DEFAULT_EMBEDDINGS_FILENAME,
    lm_head_filename: str = DEFAULT_LM_HEAD_FILENAME,
    context_session_options: dict | None = None,
    iterator_session_options: dict | None = None,
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
        context_session_options: Optional ORT ``session_options`` dict attached
            verbatim to the ``context`` stage (e.g. to route it to an
            accelerator EP).  ``None`` (default) runs the stage on CPU.  This
            function stays execution-provider-agnostic — the caller decides the
            contents; no EP-specific values are constructed here.
        iterator_session_options: Same as *context_session_options* but for the
            ``iterator`` stage.

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

    # Assign key/value patterns by name (look for "key"/"val" in the prefix),
    # falling back to positional order only when names are ambiguous.  Pure
    # positional assignment would silently swap KV if a model lists values
    # before keys in its ONNX graph.
    def _pick_kv(
        sorted_prefixes: list[str],
        patterns: dict[str, str],
        key_default: str,
        val_default: str,
    ) -> tuple[str, str]:
        key_prefix = next((p for p in sorted_prefixes if "key" in p.lower()), None)
        val_prefix = next((p for p in sorted_prefixes if "val" in p.lower()), None)
        if key_prefix and val_prefix:
            return patterns[key_prefix], patterns[val_prefix]
        # Fallback: positional (preserves original behaviour for unambiguous names)
        key_fmt = patterns[sorted_prefixes[0]] if len(sorted_prefixes) > 0 else key_default
        val_fmt = patterns[sorted_prefixes[1]] if len(sorted_prefixes) > 1 else val_default
        return key_fmt, val_fmt

    past_key_fmt, past_val_fmt = _pick_kv(
        in_sorted, input_patterns, "past_keys_%d", "past_values_%d"
    )
    pres_key_fmt, pres_val_fmt = _pick_kv(
        out_sorted, output_patterns, "present_keys_%d", "present_values_%d"
    )

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
            session_options=context_session_options,
        ),
        PipelineStage(
            name="iterator",
            filename=iterator_filename,
            run_on_prompt=False,
            run_on_token_gen=True,
            inputs=iter_inputs,
            outputs=iter_outputs,
            session_options=iterator_session_options,
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


def _patch_seq_dim_dynamic(onnx_path: Path, dim_index: int = 1) -> None:
    """Make dimension *dim_index* of all graph inputs/outputs symbolic.

    ort-genai calls the embeddings model with the full prompt on prefill
    (seq_len = prompt_len) and with a single token on each decode step
    (seq_len = 1).  The ONNX export may bake in a concrete value; this
    helper replaces it with the symbolic name ``"seq_len"`` so the runtime
    accepts any sequence length.

    The model weights (external data) are not touched — only the protobuf
    shape annotations are updated.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    changed = False
    for value_info in list(model.graph.input) + list(model.graph.output):
        shape = value_info.type.tensor_type.shape
        if shape and len(shape.dim) > dim_index:
            dim = shape.dim[dim_index]
            if dim.HasField("dim_value"):  # it's a fixed integer
                dim.ClearField("dim_value")
                dim.dim_param = "seq_len"
                changed = True
    if changed:
        onnx.save(model, str(onnx_path))
        logger.info("Patched seq_len dim to dynamic in %s", onnx_path.name)


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
    context_session_options: dict | None = None,
    iterator_session_options: dict | None = None,
) -> Path:
    """Assemble a complete ``onnxruntime-genai`` bundle in *output_dir*.

    Copies the winml-built transformer ONNX files, optional embedding /
    lm_head models, HF tokenizer files, and writes ``genai_config.json``.
    Tensor names in the config are derived by introspecting the built ONNX
    files rather than being hardcoded, so this works for any model that
    follows the 4-stage decoder-pipeline layout.

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
        context_session_options: Optional ORT ``session_options`` dict attached
            verbatim to the ``context`` stage.  ``None`` (default) runs it on
            CPU.  This assembler is execution-provider-agnostic; the caller
            supplies any EP-specific options.
        iterator_session_options: Same as *context_session_options* but for the
            ``iterator`` stage.

    Returns:
        Path to the written ``genai_config.json``.
    """
    from transformers import AutoConfig, AutoTokenizer

    from ..onnx import copy_onnx_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_onnx = Path(context_onnx)
    iterator_onnx = Path(iterator_onnx)

    # 1. Copy winml-built transformer ONNX files.
    logger.info("Copying context ONNX: %s -> %s", context_onnx.name, context_filename)
    copy_onnx_model(context_onnx, output_dir / context_filename)

    logger.info("Copying iterator ONNX: %s -> %s", iterator_onnx.name, iterator_filename)
    copy_onnx_model(iterator_onnx, output_dir / iterator_filename)

    # 2. Copy embeddings + lm_head models.
    if embeddings_src is not None:
        logger.info("Copying embeddings: %s -> %s", Path(embeddings_src).name, embeddings_filename)
        dst_embeddings = output_dir / embeddings_filename
        copy_onnx_model(Path(embeddings_src), dst_embeddings)
        # Patch seq_len to dynamic: ort-genai calls embeddings with the full
        # prompt on prefill and with a single token on every decode step, so the
        # seq_len dimension must be symbolic, not a fixed value.
        _patch_seq_dim_dynamic(dst_embeddings)
    else:
        logger.warning(
            "embeddings_src not provided — '%s' is missing from bundle.",
            embeddings_filename,
        )

    if lm_head_src is not None:
        logger.info("Copying lm_head: %s -> %s", Path(lm_head_src).name, lm_head_filename)
        dst_lm_head = output_dir / lm_head_filename
        copy_onnx_model(Path(lm_head_src), dst_lm_head)
        # Same reason as embeddings: lm_head is called with prefill seq_len
        # and with seq_len=1 on each decode step.
        _patch_seq_dim_dynamic(dst_lm_head)
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
    stages, decoder_io = build_decoder_pipeline_stages(
        context_onnx,
        iterator_onnx,
        num_layers=hf_config.num_hidden_layers,
        context_filename=context_filename,
        iterator_filename=iterator_filename,
        embeddings_filename=embeddings_filename,
        lm_head_filename=lm_head_filename,
        context_session_options=context_session_options,
        iterator_session_options=iterator_session_options,
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
    "build_decoder_pipeline_stages",
    "build_genai_config",
    "write_genai_bundle",
]
