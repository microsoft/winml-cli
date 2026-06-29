# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generate an onnxruntime-genai bundle for the Qwen3 transformer-only pipeline.

The bundle is a directory that ``onnxruntime-genai`` can load directly via
``og.Config(str(bundle_dir))``. It contains:

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
subsequent decode step. Both share the same KV cache buffer via genai's
``past_present_share_buffer`` mode.

Public API::

    from winml.modelkit.models.hf.qwen3.genai import build_genai_config, write_genai_bundle

    cfg = build_genai_config(hf_config, max_cache_len=256, prefill_seq_len=64)
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
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed tensor name constants — must match qwen_transformer_only.py I/O.
# ---------------------------------------------------------------------------
_INPUT_IDS = "input_ids"
_INPUT_HIDDEN_STATES = "input_hidden_states"
_OUTPUT_HIDDEN_STATES = "output_hidden_states"
_PAST_SEQ_LEN = "past_seq_len"
_TOTAL_SEQ_LEN = "total_seq_len"
_PAST_KEY_FMT = "past_keys_%d"
_PAST_VALUE_FMT = "past_values_%d"
_PRESENT_KEY_FMT = "present_keys_%d"
_PRESENT_VALUE_FMT = "present_values_%d"
_LOGITS = "logits"

# Default filenames inside the bundle directory.
DEFAULT_EMBEDDINGS_FILENAME = "embeddings.onnx"
DEFAULT_CONTEXT_FILENAME = "ctx.onnx"
DEFAULT_ITERATOR_FILENAME = "iter.onnx"
DEFAULT_LM_HEAD_FILENAME = "lm_head.onnx"

# Tokenizer files to save from the HF snapshot.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "special_tokens_map.json",
]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_genai_config(
    hf_config: Any,
    *,
    max_cache_len: int,
    prefill_seq_len: int,
    embeddings_filename: str = DEFAULT_EMBEDDINGS_FILENAME,
    context_filename: str = DEFAULT_CONTEXT_FILENAME,
    iterator_filename: str = DEFAULT_ITERATOR_FILENAME,
    lm_head_filename: str = DEFAULT_LM_HEAD_FILENAME,
) -> dict:
    """Build the ``genai_config.json`` dict for the transformer-only pipeline.

    Args:
        hf_config: A ``transformers.PretrainedConfig`` (e.g. from
            ``AutoConfig.from_pretrained``). Reads: ``num_hidden_layers``,
            ``hidden_size``, ``num_attention_heads``, ``num_key_value_heads``,
            ``head_dim`` (or derived), ``bos_token_id``, ``eos_token_id``,
            ``pad_token_id``, ``vocab_size``.
        max_cache_len: Static KV cache length.  Becomes ``context_length`` and
            ``search.max_length`` in the generated config.
        prefill_seq_len: Prefill / context sequence length.  Becomes
            ``decoder.sliding_window.window_size``.
        embeddings_filename: Filename of the embeddings ONNX in the bundle.
        context_filename: Filename of the context (prefill) ONNX.
        iterator_filename: Filename of the iterator (decode) ONNX.
        lm_head_filename: Filename of the lm_head ONNX.

    Returns:
        A ``dict`` ready for ``json.dumps`` as ``genai_config.json``.
    """
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

    # Build per-layer KV name lists (same ordering as the reference config).
    past_keys = [f"past_keys_{i}" for i in range(num_layers)]
    past_values = [f"past_values_{i}" for i in range(num_layers)]
    present_keys = [f"present_keys_{i}" for i in range(num_layers)]
    present_values = [f"present_values_{i}" for i in range(num_layers)]

    # Transformer stage I/O: hidden states + seq lens + KV buffers.
    transformer_inputs = [
        _INPUT_HIDDEN_STATES,
        _PAST_SEQ_LEN,
        _TOTAL_SEQ_LEN,
        *past_keys,
        *past_values,
    ]
    transformer_outputs = [_OUTPUT_HIDDEN_STATES, *present_keys, *present_values]

    return {
        "model": {
            "type": "decoder-pipeline",
            "bos_token_id": hf_config.bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "vocab_size": hf_config.vocab_size,
            "context_length": max_cache_len,
            "decoder": {
                "hidden_size": hf_config.hidden_size,
                "num_attention_heads": hf_config.num_attention_heads,
                "num_key_value_heads": hf_config.num_key_value_heads,
                "num_hidden_layers": num_layers,
                "head_size": head_size,
                "sliding_window": {
                    "window_size": prefill_seq_len,
                    "pad_value": 0,
                    "alignment": "left",
                    "slide_inputs": True,
                    "slide_key_value_cache": False,
                },
                "inputs": {
                    "input_ids": _INPUT_IDS,
                    "past_sequence_length": _PAST_SEQ_LEN,
                    "total_sequence_length": _TOTAL_SEQ_LEN,
                    "past_key_names": _PAST_KEY_FMT,
                    "past_value_names": _PAST_VALUE_FMT,
                },
                "outputs": {
                    "logits": _LOGITS,
                    "present_key_names": _PRESENT_KEY_FMT,
                    "present_value_names": _PRESENT_VALUE_FMT,
                },
                "pipeline": [
                    {
                        "embeddings": {
                            "filename": embeddings_filename,
                            "inputs": [_INPUT_IDS],
                            "outputs": [_INPUT_HIDDEN_STATES],
                            "run_on_prompt": True,
                            "run_on_token_gen": True,
                        }
                    },
                    {
                        "context": {
                            "filename": context_filename,
                            "inputs": transformer_inputs,
                            "outputs": transformer_outputs,
                            "run_on_prompt": True,
                            "run_on_token_gen": False,
                        }
                    },
                    {
                        "iterator": {
                            "filename": iterator_filename,
                            "inputs": transformer_inputs,
                            "outputs": transformer_outputs,
                            "run_on_prompt": False,
                            "run_on_token_gen": True,
                        }
                    },
                    {
                        "lm_head": {
                            "filename": lm_head_filename,
                            "inputs": [_OUTPUT_HIDDEN_STATES],
                            "outputs": [_LOGITS],
                            "is_lm_head": True,
                            "run_on_prompt": True,
                            "run_on_token_gen": True,
                        }
                    },
                ],
            },
        },
        "search": {
            "max_length": max_cache_len,
            "min_length": 0,
            "do_sample": False,
            "past_present_share_buffer": True,
        },
    }


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
) -> Path:
    """Assemble a complete ``onnxruntime-genai`` bundle in *output_dir*.

    Copies the winml-built transformer ONNX files, placeholder embedding /
    lm_head models (when provided), HF tokenizer files, and writes
    ``genai_config.json``.

    Args:
        output_dir: Destination directory (created if absent).
        context_onnx: Path to the built prefill/context ONNX
            (``decoder_prefill`` sub-model output).
        iterator_onnx: Path to the built iteration/decode ONNX
            (``decoder_gen`` sub-model output).
        model_id: HuggingFace model ID or local path used to download the HF
            config and tokenizer files.
        max_cache_len: Static KV cache length (= ``context_length`` in genai).
        prefill_seq_len: Prefill sequence length (= ``sliding_window.window_size``).
        embeddings_src: Source path of the embeddings ONNX to copy into the
            bundle.  Pass ``None`` to skip (the bundle will be incomplete until
            the embeddings model is added separately).
        lm_head_src: Source path of the lm_head ONNX to copy.  Pass ``None``
            to skip.
        context_filename: Filename used for the context ONNX inside the bundle.
        iterator_filename: Filename used for the iterator ONNX.
        embeddings_filename: Filename used for the embeddings ONNX.
        lm_head_filename: Filename used for the lm_head ONNX.

    Returns:
        Path to the written ``genai_config.json``.
    """
    from transformers import AutoConfig, AutoTokenizer

    from winml.modelkit.onnx import copy_onnx_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_onnx = Path(context_onnx)
    iterator_onnx = Path(iterator_onnx)

    # ------------------------------------------------------------------
    # 1. Copy winml-built transformer ONNX files.
    # ------------------------------------------------------------------
    logger.info("Copying context ONNX: %s -> %s", context_onnx.name, context_filename)
    copy_onnx_model(context_onnx, output_dir / context_filename)

    logger.info("Copying iterator ONNX: %s -> %s", iterator_onnx.name, iterator_filename)
    copy_onnx_model(iterator_onnx, output_dir / iterator_filename)

    # ------------------------------------------------------------------
    # 2. Copy placeholder models (embeddings + lm_head).
    # ------------------------------------------------------------------
    if embeddings_src is not None:
        logger.info("Copying embeddings: %s -> %s", Path(embeddings_src).name, embeddings_filename)
        copy_onnx_model(Path(embeddings_src), output_dir / embeddings_filename)
    else:
        logger.warning(
            "embeddings_src not provided — '%s' is missing from bundle; "
            "add it manually before running inference.",
            embeddings_filename,
        )

    if lm_head_src is not None:
        logger.info("Copying lm_head: %s -> %s", Path(lm_head_src).name, lm_head_filename)
        copy_onnx_model(Path(lm_head_src), output_dir / lm_head_filename)
    else:
        logger.warning(
            "lm_head_src not provided — '%s' is missing from bundle; "
            "add it manually before running inference.",
            lm_head_filename,
        )

    # ------------------------------------------------------------------
    # 3. Save tokenizer files from the HF snapshot.
    # ------------------------------------------------------------------
    logger.info("Saving tokenizer from '%s' to %s", model_id, output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))
    # Prune any extra files that save_pretrained creates but genai doesn't need
    # (e.g. tokenizer.model for sentencepiece models).  Keep only known files.
    onnx_filenames = {context_filename, iterator_filename, embeddings_filename, lm_head_filename}
    for path in output_dir.iterdir():
        if (
            path.name not in _TOKENIZER_FILES
            and path.suffix in (".json", ".txt", ".model")
            and path.name not in onnx_filenames
        ):
            logger.debug("Keeping extra tokenizer file: %s", path.name)

    # ------------------------------------------------------------------
    # 4. Write genai_config.json.
    # ------------------------------------------------------------------
    hf_config = AutoConfig.from_pretrained(model_id)
    config = build_genai_config(
        hf_config,
        max_cache_len=max_cache_len,
        prefill_seq_len=prefill_seq_len,
        embeddings_filename=embeddings_filename,
        context_filename=context_filename,
        iterator_filename=iterator_filename,
        lm_head_filename=lm_head_filename,
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
    "build_genai_config",
    "write_genai_bundle",
]
