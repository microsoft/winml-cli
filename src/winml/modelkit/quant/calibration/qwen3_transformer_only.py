# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Config-driven w8a8 calibration for the transformer-only Qwen3 build.

The transformer-only export (``models.hf.qwen3.qwen_transformer_only``) emits a graph
whose only quantization-relevant runtime inputs (the calibration feeds and the
``GroupQueryAttention`` node names to keep in float) can't be known until the
ONNX exists. Rather than a standalone post-build script that reaches into
``composite.sub_models[...]._onnx_path``, this module defines a quant policy
keyed on ``model_type`` (:class:`Qwen3TransformerOnlyQuantFinalizer`, named in
:data:`~winml.modelkit.quant.calibration.registry.QUANT_FINALIZERS`). The build
pipeline resolves it via :func:`~winml.modelkit.quant.get_quant_finalizer` and
calls :func:`finalize_transformer_only_quant_config` just before
``quantize_onnx`` runs (see ``build/hf.py``), populating the live
:class:`WinMLQuantizationConfig` with the right
:class:`~winml.modelkit.quant.config.CalibrationDataReader` and
``nodes_to_exclude``.

The two readers match the exported graph exactly:

  - ``input_hidden_states`` (FP32), ``past_seq_len`` / ``total_seq_len``
    (INT32), ``past_keys_{i}`` / ``past_values_{i}`` (FP16, full cache buffer).
  - The prefill reader (``seq_len > 1``) embeds real prompt prefixes.
  - The decode reader (``seq_len == 1``) drives a fresh FP reference model
    through a real prefill + decode trajectory so MinMax sees representative
    mid-generation activation ranges (a single repeated token + zeroed KV
    collapses the ranges and degenerates generation).

The export wrapper surgically replaces its own ``self.model`` (RMSNorm ->
LpNorm-identity, attention -> GQA placeholder, Linear -> 1x1 Conv), so it can't
run real inference; calibration loads a *fresh* ``AutoModelForCausalLM``.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import CalibrationDataReader, WinMLQuantizationConfig


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_PREFILL_SEQ = 64
DEFAULT_GEN_SEQ = 1
DEFAULT_NUM_SAMPLES = 30
DEFAULT_DECODE_STEPS = 16
DEFAULT_CALIB_DATASET = "openai/gsm8k"
DEFAULT_CALIB_DATASET_CONFIG = "main"
DEFAULT_CALIB_SPLIT = "train"
DEFAULT_CALIB_SEED = 42


def _load_gsm8k_prompts(num_samples: int) -> list[str]:
    """GSM8K train split, shuffled seed=42 for reproducible calibration."""
    from datasets import load_dataset

    ds = load_dataset(DEFAULT_CALIB_DATASET, DEFAULT_CALIB_DATASET_CONFIG)
    split = ds[DEFAULT_CALIB_SPLIT].shuffle(seed=DEFAULT_CALIB_SEED)
    return [row["question"] for row in split.select(range(num_samples))]


def _tokenize_prompts(tokenizer: Any, prompts: list[str], num_samples: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer([text], return_tensors="pt").input_ids
        out.append(ids)
    return out


def _gqa_node_names(onnx_path: Path) -> list[str]:
    """Return the names of every GroupQueryAttention node in ``onnx_path``.

    These nodes are excluded from quantization so ORT leaves both their
    inputs and output in float (``... -> Cast -> GQA -> Cast``), matching
    the reference graph which keeps attention entirely out of QDQ.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return [n.name for n in model.graph.node if n.op_type == "GroupQueryAttention" and n.name]


def _graph_shapes(onnx_path: Path) -> tuple[int, int]:
    """Read ``(seq_len, max_cache_len)`` from the exported graph's static inputs.

    ``seq_len`` is the query length (``input_hidden_states`` dim 1) and
    ``max_cache_len`` is the KV buffer length (``past_keys_0`` dim 2). The
    transformer-only export keeps both axes static, so these fully determine
    whether the sub-model is prefill (``seq_len > 1``) or decode (``seq_len == 1``)
    and the size of the fixed KV buffers the calibration feeds must match.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    seq_len: int | None = None
    max_cache_len: int | None = None
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if inp.name == "input_hidden_states" and len(dims) >= 2:
            seq_len = dims[1].dim_value
        elif inp.name == "past_keys_0" and len(dims) >= 3:
            max_cache_len = dims[2].dim_value
    # A symbolic/dynamic axis yields dim_value == 0 (not None), so treat any
    # non-positive value as "not a usable static shape" and fail loudly rather
    # than silently building zero-length calibration feeds.
    if not seq_len or not max_cache_len:
        raise ValueError(
            f"Could not read static seq_len/max_cache_len from {onnx_path.name}; "
            f"found seq_len={seq_len}, max_cache_len={max_cache_len}"
        )
    return seq_len, max_cache_len


def _layer_kv(past: Any, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract layer ``i``'s (key, value) from an HF cache, version-agnostic.

    Handles the legacy tuple-of-tuples cache, the older ``DynamicCache``
    (``.key_cache`` / ``.value_cache``), and the newer per-layer
    ``DynamicCache`` (``.layers[i].keys`` / ``.values``).
    """
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return past.key_cache[i], past.value_cache[i]
    if hasattr(past, "layers"):
        layer = past.layers[i]
        return layer.keys, layer.values
    return past[i][0], past[i][1]


class Qwen3TransformerOnlyCalibReader(CalibrationDataReader):
    """Prefill calibration feeds for the transformer-only ONNX.

    Feeds match the exported graph exactly: ``input_hidden_states`` (FP32),
    ``past_seq_len`` (INT32 ``[1,1]``), ``total_seq_len`` (INT32 ``[1]``),
    and ``past_keys_{i}`` / ``past_values_{i}`` (FP16, full cache buffer).
    """

    def __init__(
        self,
        embed_tokens: torch.nn.Module,
        config: Any,
        token_ids_list: list[torch.Tensor],
        *,
        seq_len: int,
        max_cache_len: int,
    ) -> None:
        self.embed = embed_tokens
        self.seq_len = seq_len
        self.max_cache_len = max_cache_len
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self._samples = list(self._build_samples(token_ids_list))
        self._iter: Iterator[dict[str, np.ndarray]] | None = None
        self.rewind()

    def _build_samples(self, token_ids_list: list[torch.Tensor]) -> Iterator[dict[str, np.ndarray]]:
        for ids in token_ids_list:
            ids = ids[:, : self.seq_len]
            real_len = ids.shape[1]
            if real_len < self.seq_len:
                pad = torch.zeros((1, self.seq_len - real_len), dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, pad], dim=1)

            with torch.no_grad():
                embeds = self.embed(ids).to(torch.float32).cpu().numpy()

            feed: dict[str, np.ndarray] = {
                "input_hidden_states": embeds.astype(np.float32),
                # seqlens_k for GQA = (valid context length - 1), i.e.
                # ``embeddings.shape[1] - 1``. We pad to seq_len, so the query
                # has seq_len valid positions -> past_seq_len = seq_len - 1.
                # (Using 0 here declares only 1 valid token while feeding a
                # seq_len-token query, which makes the GQA prefill kernel read
                # out of bounds -> native access violation.)
                "past_seq_len": np.array([[self.seq_len - 1]], dtype=np.int32),
                "total_seq_len": np.array([self.max_cache_len], dtype=np.int32),
            }
            kv_shape = (1, self.num_kv_heads, self.max_cache_len, self.head_dim)
            zeros = np.zeros(kv_shape, dtype=np.float16)
            for i in range(self.num_layers):
                feed[f"past_keys_{i}"] = zeros
                feed[f"past_values_{i}"] = zeros
            yield feed

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next calibration feed, or None when exhausted."""
        try:
            return next(self._iter) if self._iter is not None else None
        except StopIteration:
            return None

    def rewind(self) -> None:
        """Reset the iterator so calibration can run another pass."""
        self._iter = iter(self._samples)


class Qwen3DecodeTrajectoryCalibReader(CalibrationDataReader):
    """Calibrate the iter (seq_len=1) model on REAL decode-step states.

    The naive reader feeds one (repeated) token with a zeroed KV cache and
    ``past_seq_len=0`` — a state the model never sees during generation. With
    MinMax calibration this collapses the observed activation ranges far below
    the real decode distribution, so the resulting w8a16 model degenerates
    (e.g. ``Paris -> Parisammedammed...``).

    Instead, drive the HF FP reference model through a real prefill + decode
    trajectory and capture, at each decode step, the exact feed the iter ONNX
    would receive: the embedding of the *actually generated* token, the real
    accumulated KV cache (copied into the fixed ``[1, kv_heads, max_cache,
    head_dim]`` FP16 buffer), and the growing ``past_seq_len``. Token
    selection uses the HF model's true logits, so the trajectory matches
    greedy generation. The QDQ scheme is unchanged — only the calibration
    statistics become representative.
    """

    def __init__(
        self,
        hf_model: torch.nn.Module,
        embed_tokens: torch.nn.Module,
        config: Any,
        token_ids_list: list[torch.Tensor],
        *,
        prefill_seq: int,
        max_cache_len: int,
        decode_steps: int = DEFAULT_DECODE_STEPS,
    ) -> None:
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.max_cache_len = max_cache_len
        self._samples = list(
            self._build_samples(
                hf_model,
                embed_tokens,
                token_ids_list,
                prefill_seq=prefill_seq,
                decode_steps=decode_steps,
            )
        )
        self._iter: Iterator[dict[str, np.ndarray]] | None = None
        self.rewind()

    def _kv_buffers(self, past: Any, cur_len: int) -> dict[str, np.ndarray]:
        """Copy the ``cur_len`` valid KV positions into fixed FP16 buffers."""
        feed: dict[str, np.ndarray] = {}
        for i in range(self.num_layers):
            k, v = _layer_kv(past, i)
            kbuf = np.zeros((1, self.num_kv_heads, self.max_cache_len, self.head_dim), np.float16)
            vbuf = np.zeros_like(kbuf)
            kbuf[:, :, :cur_len, :] = k[:, :, :cur_len, :].to(torch.float16).cpu().numpy()
            vbuf[:, :, :cur_len, :] = v[:, :, :cur_len, :].to(torch.float16).cpu().numpy()
            feed[f"past_keys_{i}"] = kbuf
            feed[f"past_values_{i}"] = vbuf
        return feed

    def _build_samples(
        self,
        hf_model: torch.nn.Module,
        embed_tokens: torch.nn.Module,
        token_ids_list: list[torch.Tensor],
        *,
        prefill_seq: int,
        decode_steps: int,
    ) -> Iterator[dict[str, np.ndarray]]:
        for ids in token_ids_list:
            ids = ids[:, :prefill_seq]  # real prompt prefix (no pad-token KV)
            cur_len = ids.shape[1]

            # FP prefill once to seed a realistic KV cache + first token.
            with torch.no_grad():
                out = hf_model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            tok = int(out.logits[:, -1, :].argmax(-1))

            for _ in range(decode_steps):
                if cur_len >= self.max_cache_len:
                    break
                # The feed the iter model sees for THIS token: embedding of the
                # token to process, the KV of the `cur_len` preceding tokens,
                # and seqlens_k = (cur_len + 1) - 1 = cur_len.
                with torch.no_grad():
                    emb = embed_tokens(torch.tensor([[tok]])).to(torch.float32).cpu().numpy()
                feed: dict[str, np.ndarray] = {
                    "input_hidden_states": emb.astype(np.float32),
                    "past_seq_len": np.array([[cur_len]], dtype=np.int32),
                    "total_seq_len": np.array([self.max_cache_len], dtype=np.int32),
                }
                feed.update(self._kv_buffers(past, cur_len))
                yield feed

                # Advance the reference model one real decode step.
                with torch.no_grad():
                    out = hf_model(
                        input_ids=torch.tensor([[tok]]),
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                tok = int(out.logits[:, -1, :].argmax(-1))
                cur_len += 1

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next calibration feed, or None when exhausted."""
        try:
            return next(self._iter) if self._iter is not None else None
        except StopIteration:
            return None

    def rewind(self) -> None:
        """Reset the iterator so calibration can run another pass."""
        self._iter = iter(self._samples)


def finalize_transformer_only_quant_config(
    quant: WinMLQuantizationConfig,
    *,
    onnx_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    prefill_seq: int = DEFAULT_PREFILL_SEQ,
    decode_steps: int = DEFAULT_DECODE_STEPS,
) -> WinMLQuantizationConfig:
    """Populate ``quant`` with the transformer-only w8a16 scheme + runtime fields.

    The build pipeline's device/precision policy only enables quantization and
    picks generic dtypes; the transformer-only scheme is fixed and reference-
    matched, so this hook is authoritative:

      - **int8-symmetric weights** (zp=0) + **uint8 asymmetric activations**,
      - **MinMax** calibration, ``mode="static"`` (forces QDQ dispatch),
      - GroupQueryAttention nodes excluded from QDQ (read from the graph),
      - the matching :class:`CalibrationDataReader` (prefill vs. decode-trajectory,
        chosen by the graph's ``seq_len``).

    Reads static shapes + GQA nodes from ``onnx_path`` and loads a fresh FP
    reference model for calibration (the export wrapper's own weights are
    surgically replaced and can't run real inference).
    """
    onnx_path = Path(onnx_path)
    seq_len, max_cache_len = _graph_shapes(onnx_path)
    gqa_nodes = _gqa_node_names(onnx_path)

    # Fixed, reference-matched w8a8 scheme (authoritative over policy dtypes).
    # ``mode`` must be pinned to "static": the new precision-driven flow keys the
    # quantizer dispatch on ``config.mode`` (fp16/rtn/static), so a build whose
    # precision policy resolved to "fp16"/"rtn" would otherwise bypass QDQ and
    # silently ignore the calibration reader + GQA exclusion set below.
    # uint8 activations (matching the reference model) keep ctx/iter at opset 18;
    # uint16 would force opset 21 (ORT requires opset >= 21 for 16-bit QDQ).
    quant.mode = "static"
    quant.weight_type = "int8"
    quant.activation_type = "uint8"
    quant.weight_symmetric = True
    quant.activation_symmetric = False
    quant.calibration_method = "minmax"
    num_samples = quant.samples or DEFAULT_NUM_SAMPLES

    logger.info(
        "Finalizing transformer-only quant config for %s "
        "(seq_len=%d, max_cache_len=%d, %d GQA nodes excluded, %d samples)",
        onnx_path.name,
        seq_len,
        max_cache_len,
        len(gqa_nodes),
        num_samples,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    hf_model.eval()
    embed_tokens = hf_model.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompts = _load_gsm8k_prompts(num_samples)
    token_ids_list = _tokenize_prompts(tokenizer, prompts, num_samples)

    reader: CalibrationDataReader
    if seq_len == DEFAULT_GEN_SEQ:
        # Decode sub-model: calibrate on a real prefill+decode trajectory.
        reader = Qwen3DecodeTrajectoryCalibReader(
            hf_model,
            embed_tokens,
            hf_model.config,
            token_ids_list,
            prefill_seq=prefill_seq,
            max_cache_len=max_cache_len,
            decode_steps=decode_steps,
        )
    else:
        reader = Qwen3TransformerOnlyCalibReader(
            embed_tokens,
            hf_model.config,
            token_ids_list,
            seq_len=seq_len,
            max_cache_len=max_cache_len,
        )

    quant.calibration_data = reader
    quant.nodes_to_exclude = gqa_nodes

    # Readers materialize all samples eagerly, so the FP reference is no longer
    # needed once they're built.
    del hf_model, embed_tokens
    gc.collect()

    return quant


class Qwen3TransformerOnlyQuantFinalizer:
    """Quant policy for the ``qwen3_transformer_only`` model_type.

    Named in :data:`~winml.modelkit.quant.calibration.registry.QUANT_FINALIZERS`
    and resolved by :func:`~winml.modelkit.quant.get_quant_finalizer`. Adapts
    :func:`finalize_transformer_only_quant_config` to the
    :class:`~winml.modelkit.quant.calibration.base.QuantConfigFinalizer`
    protocol so the build pipeline applies the model-specific w8a16 scheme +
    calibration reader (keyed on ``model_type``) rather than a hardcoded hook on
    the export wrapper.
    """

    def finalize(
        self,
        quant: WinMLQuantizationConfig,
        *,
        onnx_path: Path,
        model_id: str | None = None,
    ) -> WinMLQuantizationConfig:
        """Populate ``quant`` with the transformer-only w8a16 scheme + reader."""
        return finalize_transformer_only_quant_config(
            quant, onnx_path=onnx_path, model_id=model_id or DEFAULT_MODEL_ID
        )
