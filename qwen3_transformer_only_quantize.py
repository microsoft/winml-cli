"""Transformer-only w8a16 quantization for Qwen3.

Targets the transformer-only ONNX produced by
``qwen_transformer_only.install() + test_qwen.py``:

  - **No embedding/lm_head surgery.** The export already excludes both,
    so we feed ``WinMLQuantization`` the file directly.
  - **Transformer-shaped calibration feeds.** ``input_hidden_states`` (FP32),
    ``past_seq_len`` / ``total_seq_len`` (INT32), ``past_keys_{i}`` /
    ``past_values_{i}`` (FP16) — names + dtypes match the exported graph.

Run via ``test_qwen.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from winml.modelkit.models.winml.composite_model import WinMLCompositeModel
from winml.modelkit.quant import WinMLQuantizationConfig, quantize_onnx


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CACHE = 256
DEFAULT_PREFILL_SEQ = 64
DEFAULT_GEN_SEQ = 1
DEFAULT_NUM_SAMPLES = 30
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


class Qwen3TransformerOnlyCalibReader:
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


def quantize_built_model(
    model: WinMLCompositeModel,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_cache_len: int = DEFAULT_MAX_CACHE,
    prefill_seq: int = DEFAULT_PREFILL_SEQ,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    weight_type: str = "uint8",
    activation_type: str = "uint16",
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
            fused_path.stem + f"_w{weight_type[-1]}a{activation_type[-2:]}.quant.onnx"
        )

        print(f"\n=== Quantize (transformer-only): {sub_name} (seq_len={seq_len}) ===")
        print(f"  in : {fused_path}")
        print(f"  out: {quant_path}")
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

    print("\n=== Done ===")
    return quant_paths
