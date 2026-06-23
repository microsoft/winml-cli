"""Transformer-only w8a16 quantization for Qwen3.

Targets the transformer-only ONNX produced by the
``qwen3_transformer_only`` build variant (see ``test_qwen.py``):

  - **No embedding/lm_head surgery.** The export already excludes both,
    so we feed ``WinMLQuantization`` the file directly.
  - **Transformer-shaped calibration feeds.** ``input_hidden_states`` (FP32),
    ``past_seq_len`` / ``total_seq_len`` (INT32), ``past_keys_{i}`` /
    ``past_values_{i}`` (FP16) — names + dtypes match the exported graph.

Run via ``test_qwen.py``.
"""

from __future__ import annotations

import logging
import gc
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from winml.modelkit.models.winml.composite_model import WinMLCompositeModel
from winml.modelkit.quant import WinMLQuantizationConfig, quantize_onnx
from winml.modelkit.quant.config import CalibrationDataReader


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CACHE = 256
DEFAULT_PREFILL_SEQ = 64
DEFAULT_GEN_SEQ = 1
DEFAULT_NUM_SAMPLES = 30
DEFAULT_DECODE_STEPS = 16
DEFAULT_CALIB_DATASET = "openai/gsm8k"
DEFAULT_CALIB_DATASET_CONFIG = "main"
DEFAULT_CALIB_SPLIT = "train"
DEFAULT_CALIB_SEED = 42

# Map an ONNX quantization dtype to the bit-width suffix used in artifact
# filenames (e.g. int8 -> "8", uint16 -> "16"), instead of brittle string
# slicing of the dtype name.
_DTYPE_BITS = {
    "int8": "8",
    "uint8": "8",
    "int16": "16",
    "uint16": "16",
}


def _load_gsm8k_prompts(num_samples: int) -> list[str]:
    """GSM8K train split, shuffled seed=42 for reproducible calibration."""
    from datasets import load_dataset

    ds = load_dataset(DEFAULT_CALIB_DATASET, DEFAULT_CALIB_DATASET_CONFIG)
    split = ds[DEFAULT_CALIB_SPLIT].shuffle(seed=DEFAULT_CALIB_SEED)
    return [row["question"] for row in split.select(range(num_samples))]


class Qwen3TransformerOnlyCalibReader(CalibrationDataReader):
    """Yields calibration feeds for the transformer-only ONNX.

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
                pad = torch.zeros(
                    (1, self.seq_len - real_len), dtype=ids.dtype, device=ids.device
                )
                ids = torch.cat([ids, pad], dim=1)

            with torch.no_grad():
                embeds = self.embed(ids).to(torch.float32).cpu().numpy()

            feed: dict[str, np.ndarray] = {
                "input_hidden_states": embeds.astype(np.float32),
                # seqlens_k for GQA = (valid context length - 1), i.e.
                # ``embeddings.shape[1] - 1``. We pad to seq_len, so the query
                # has seq_len valid positions → past_seq_len = seq_len - 1.
                # (Using 0 here declares only 1 valid token while feeding a
                # seq_len-token query, which makes the GQA prefill kernel read
                # out of bounds → native access violation.)
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
        try:
            return next(self._iter) if self._iter is not None else None
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._samples)


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
        decode_steps: int = 16,
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
            kbuf = np.zeros(
                (1, self.num_kv_heads, self.max_cache_len, self.head_dim), np.float16
            )
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
        try:
            return next(self._iter) if self._iter is not None else None
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._samples)


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
    return [
        n.name
        for n in model.graph.node
        if n.op_type == "GroupQueryAttention" and n.name
    ]


def quantize_built_model(
    model: WinMLCompositeModel,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_cache_len: int = DEFAULT_MAX_CACHE,
    prefill_seq: int = DEFAULT_PREFILL_SEQ,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    weight_type: str = "int8",
    activation_type: str = "uint16",
    decode_steps: int = DEFAULT_DECODE_STEPS,
) -> dict[str, Path]:
    """Quantize the transformer-only ONNX files in-place.

    Returns ``{sub_model_name: quantized_path}``.
    """
    # Locate the un-compiled ONNX for each sub-model (no surgery — file is
    # already transformer-only).
    sub_paths: dict[str, Path] = {}
    for name, sub in model.sub_models.items():
        final_path = Path(sub._onnx_path)
        if final_path.name.endswith("_model.onnx"):
            stem = final_path.name[: -len("_model.onnx")]
            optimized = final_path.with_name(f"{stem}_optimized.onnx")
            if optimized.exists():
                sub_paths[name] = optimized
                continue
            print(
                f"WARNING: {optimized.name} not found next to {final_path.name}; "
                "falling back to the compiled model."
            )
        sub_paths[name] = final_path

    for name, p in sub_paths.items():
        print(f"  {name}: {p}")

    print("\n=== Loading HF embed_tokens for calibration ===")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()
    embed_tokens = hf_model.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(
        f"=== Loading {num_samples} GSM8K calibration prompts "
        f"({DEFAULT_CALIB_DATASET}/{DEFAULT_CALIB_DATASET_CONFIG}, "
        f"split={DEFAULT_CALIB_SPLIT}, seed={DEFAULT_CALIB_SEED}) ==="
    )
    prompts = _load_gsm8k_prompts(num_samples)
    token_ids_list = _tokenize_prompts(tokenizer, prompts, num_samples)

    seq_by_sub = {
        "decoder_prefill": prefill_seq,
        "decoder_gen": DEFAULT_GEN_SEQ,
    }

    quant_paths: dict[str, Path] = {}
    for sub_name, fused_path in sub_paths.items():
        if sub_name not in seq_by_sub:
            print(f"\n--- Skipping unknown sub-model {sub_name!r} ---")
            continue

        seq_len = seq_by_sub[sub_name]
        quant_path = fused_path.with_name(
            fused_path.stem
            + f"_w{_DTYPE_BITS[weight_type]}a{_DTYPE_BITS[activation_type]}.quant.onnx"
        )

        print(f"\n=== Quantize (transformer-only): {sub_name} (seq_len={seq_len}) ===")
        print(f"  in : {fused_path}")
        print(f"  out: {quant_path}")
        gqa_nodes = _gqa_node_names(fused_path)
        print(
            f"  excluding {len(gqa_nodes)} GroupQueryAttention nodes from "
            "quantization (inputs + output stay float, Cast -> GQA -> Cast)"
        )
        if sub_name == "decoder_gen":
            # The iter model only sees mid-generation states. Calibrate it on a
            # real prefill+decode trajectory (true tokens, accumulated KV,
            # growing past_seq_len) instead of one token + zeroed KV, which
            # would under-range the MinMax activation scales and collapse
            # generation.
            print(
                f"  calibrating on decode trajectory ({decode_steps} steps/prompt, "
                f"prefill_seq={prefill_seq})"
            )
            reader: CalibrationDataReader = Qwen3DecodeTrajectoryCalibReader(
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
        cfg = WinMLQuantizationConfig(
            samples=num_samples,
            weight_type=weight_type,  # type: ignore[arg-type]
            activation_type=activation_type,  # type: ignore[arg-type]
            calibration_method="minmax",
            calibration_data=reader,
            # w8a16: symmetric int8 weights (zp=0) + asymmetric uint16
            # activations, matching the reference quantization.
            weight_symmetric=True,
            activation_symmetric=False,
            # ORT treats GroupQueryAttention as quantizable and wraps both its
            # inputs and output in QDQ. The reference keeps attention entirely
            # in float (Cast -> GQA -> Cast), so exclude the GQA nodes from
            # quantization so no QDQ is inserted around them.
            nodes_to_exclude=gqa_nodes,
        )
        result = quantize_onnx(fused_path, output_path=quant_path, config=cfg)
        if not result.success:
            print("  FAILED:")
            for err in result.errors:
                print(f"    {err}")
            raise SystemExit(1)
        print(
            f"  ok — {result.nodes_quantized} QDQ nodes inserted in "
            f"{result.total_time_seconds:.1f}s"
        )
        quant_paths[sub_name] = quant_path

    # Free the FP reference model now that calibration is done.
    del hf_model, embed_tokens
    gc.collect()

    print("\n=== Done ===")
    return quant_paths
