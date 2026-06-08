"""Qwen3 transformer-only quantization.

Must be called after the composite Qwen3 model has been built (e.g. by
``test_qwen 2.py``) so that ``decoder_prefill`` / ``decoder_gen`` ONNX files
exist in the winml cache.

Pipeline:

  1. Apply ``make_transformer_only`` surgery to each sub-model, producing
     ``*_transformer.onnx`` with ``inputs_embeds`` input and
     ``output_hidden_states`` output — embeddings and lm_head are stripped
     out (ignored, not quantized).
  2. Quantize those transformer-only files via winml-cli's ``quantize_onnx``
     using a calibration reader that runs ``embed_tokens`` in PyTorch on
     real text samples.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from winml.modelkit.models.winml.composite_model import WinMLCompositeModel
from winml.modelkit.onnx import make_transformer_only
from winml.modelkit.quant import WinMLQuantizationConfig, quantize_onnx


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CACHE = 256
DEFAULT_PREFILL_SEQ = 64
DEFAULT_GEN_SEQ = 1
DEFAULT_NUM_SAMPLES = 16
DEFAULT_PROMPTS = [
    "Solve: 8 * 7 = ?",
    "Translate to French: The weather is nice today.",
    "Write a short poem about the ocean.",
    "Explain gradient descent in one paragraph.",
    "What is the capital of Japan?",
    "List three uses of magnesium.",
    "Summarize the plot of Hamlet in two sentences.",
    "Give a Python one-liner to reverse a string.",
]


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------


class Qwen3TransformerCalibReader:
    """Yields calibration feeds for the transformer-only Qwen3 ONNX.

    Runs HF ``embed_tokens`` in PyTorch to produce ``inputs_embeds`` since the
    embedding layer was stripped from the ONNX graph. All other inputs
    (attention_mask, position_ids, past_{i}_key/value) follow the conventions
    used by winml-cli's ``WinMLQwen3Model`` runtime.
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
        self.cfg = config
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

    def _build_samples(
        self, token_ids_list: list[torch.Tensor]
    ) -> Iterator[dict[str, np.ndarray]]:
        for ids in token_ids_list:
            # Right-truncate / pad to seq_len so we feed the static graph shape.
            ids = ids[:, : self.seq_len]
            real_len = ids.shape[1]
            if real_len < self.seq_len:
                pad = torch.zeros(
                    (1, self.seq_len - real_len), dtype=ids.dtype, device=ids.device
                )
                ids = torch.cat([ids, pad], dim=1)

            with torch.no_grad():
                embeds = self.embed(ids).to(torch.float32).cpu().numpy()

            # attention_mask: ones for real prompt positions placed at the
            # END of the max_cache buffer (sliding-window cache convention),
            # zeros elsewhere.
            attn_mask = np.zeros((1, self.max_cache_len), dtype=np.int64)
            attn_mask[0, -real_len:] = 1

            # position_ids: 0..seq_len-1 (clamped for padding).
            position_ids = np.arange(self.seq_len, dtype=np.int64)[None, :]

            feed: dict[str, np.ndarray] = {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": attn_mask,
                "position_ids": position_ids,
            }
            kv_shape = (1, self.num_kv_heads, self.max_cache_len, self.head_dim)
            zeros = np.zeros(kv_shape, dtype=np.float32)
            for i in range(self.num_layers):
                feed[f"past_{i}_key"] = zeros
                feed[f"past_{i}_value"] = zeros
            yield feed

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return next(self._iter) if self._iter is not None else None
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._samples)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _tokenize_prompts(
    tokenizer: Any, prompts: list[str], num_samples: int
) -> list[torch.Tensor]:
    # Cycle through prompts up to num_samples; apply chat template like the
    # runtime so calibration distribution matches inference inputs.
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
    """Run surgery + transformer-only quantization on an already-built composite.

    Reuses the ONNX files produced by ``WinMLCompositeModel.from_pretrained``
    so this can be called after a build step without re-exporting.

    Returns: mapping of sub-model name → quantized ONNX path.
    """
    sub_paths: dict[str, Path] = {}
    for name, sub in model.sub_models.items():
        final_path = Path(sub._onnx_path)
        # ``_model.onnx`` is the *compiled* QNN EPContext blob — surgery needs
        # the uncompiled fp16 graph.  ``build.hf`` emits ``{cache_key}_optimized.onnx``
        # alongside it in the same artifacts directory.
        if final_path.name.endswith("_model.onnx"):
            stem = final_path.name[: -len("_model.onnx")]
            optimized = final_path.with_name(f"{stem}_optimized.onnx")
            if optimized.exists():
                sub_paths[name] = optimized
                continue
            print(
                f"WARNING: {optimized.name} not found next to {final_path.name}; "
                "falling back to the compiled model (surgery will likely fail)."
            )
        sub_paths[name] = final_path

    for name, p in sub_paths.items():
        print(f"  {name}: {p}")

    print("\n=== Loading HF embed_tokens for calibration ===")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()
    embed_tokens = hf_model.get_input_embeddings()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    token_ids_list = _tokenize_prompts(tokenizer, DEFAULT_PROMPTS, num_samples)

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
        transformer_path = fused_path.with_name(fused_path.stem + "_transformer.onnx")
        quant_path = transformer_path.with_name(
            transformer_path.stem + f"_w{weight_type[-1]}a{activation_type[-2:]}.quant.onnx"
        )

        print(f"\n=== Surgery: {sub_name} (seq_len={seq_len}) ===")
        print(f"  in : {fused_path}")
        print(f"  out: {transformer_path}")
        make_transformer_only(fused_path, transformer_path)

        print(f"\n=== Quantize (transformer only): {sub_name} ===")
        print(f"  out: {quant_path}")
        reader = Qwen3TransformerCalibReader(
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
        result = quantize_onnx(transformer_path, output_path=quant_path, config=cfg)
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

