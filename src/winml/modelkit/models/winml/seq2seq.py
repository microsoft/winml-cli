# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLModelForSeq2SeqLM.

Inference wrapper for encoder-decoder ONNX models with KV cache.
Supports T5 and similar architectures exported as split encoder + decoder.

The encoder runs once. The decoder runs per token with a StaticCache
that persists across generation steps (same object, mutated in-place via
``index_copy_`` at ``cache_position``).

Both encoder and decoder are built and held as WinMLAutoModel instances.

Usage:
    model = WinMLModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
    pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    result = pipe("Hello, how are you?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from transformers import Cache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .base import PreTrainedModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)


class WinMLModelForSeq2SeqLM(PreTrainedModel, GenerationMixin):
    """WinML model for seq2seq tasks (translation, summarization).

    Composes two WinMLAutoModel instances (encoder + decoder).
    The encoder runs once at the start of generate(). The decoder runs
    per token with a HF ``StaticCache`` that is mutated in-place each step.

    Use ``from_pretrained()`` to build both ONNX models automatically.
    """

    main_input_name = "input_ids"
    base_model_prefix = ""
    _is_stateful = False
    _supports_cache_class = False

    def __init__(
        self,
        encoder: WinMLPreTrainedModel,
        decoder: WinMLPreTrainedModel,
        config: PretrainedConfig,
    ) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self.config = config

        # Read shapes from decoder ONNX
        dec_io = decoder.io_config
        dec_shapes = dict(
            zip(dec_io.get("input_names", []), dec_io.get("input_shapes", []), strict=False)
        )
        kv_shape = dec_shapes.get("past_0_key", [1, 8, 32, 64])
        self._max_dec = kv_shape[2] if len(kv_shape) > 2 else 32

        # Read encoder seq len from encoder ONNX
        enc_io = encoder.io_config
        enc_shapes = enc_io.get("input_shapes", [])
        self._enc_seq = enc_shapes[0][1] if enc_shapes and len(enc_shapes[0]) > 1 else 16

        # Model dims from config
        self._nl = getattr(config, "num_layers", 6)
        self._nh = getattr(config, "num_heads", 8)
        self._dk = getattr(config, "d_kv", 64)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "cpu",
        use_cache: bool = True,
        force_rebuild: bool = False,
        **kwargs: Any,
    ) -> WinMLModelForSeq2SeqLM:
        """Build encoder + decoder ONNX and return ready-to-use model.

        Args:
            model_id: HuggingFace model ID (e.g., "google-t5/t5-small").
            device: Target device.
            use_cache: Use persistent build cache.
            force_rebuild: Force rebuild.
        """
        from transformers import AutoConfig

        from ..auto import WinMLAutoModel

        hf_config = AutoConfig.from_pretrained(model_id)

        logger.info("Building encoder for %s...", model_id)
        encoder = WinMLAutoModel.from_pretrained(
            model_id,
            task="feature-extraction",
            device=device,
            use_cache=use_cache,
            force_rebuild=force_rebuild,
            **kwargs,
        )

        logger.info("Building decoder for %s...", model_id)
        decoder = WinMLAutoModel.from_pretrained(
            model_id,
            task="text2text-generation",
            device=device,
            use_cache=use_cache,
            force_rebuild=force_rebuild,
            **kwargs,
        )

        return cls(encoder=encoder, decoder=decoder, config=hf_config)

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def _run_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run encoder via WinMLAutoModel, return hidden states."""
        out = self._encoder(
            input_ids=self._pad_to(input_ids, self._enc_seq, 0),
            attention_mask=self._pad_to(attention_mask, self._enc_seq, 0),
        )
        return out["encoder_hidden_states"]

    class _EncoderProxy(torch.nn.Module):
        """Proxy returned by get_encoder() for GenerationMixin."""

        def __init__(self, model: WinMLModelForSeq2SeqLM) -> None:
            super().__init__()
            self._model = model

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kw: Any
        ) -> BaseModelOutput:
            return BaseModelOutput(
                last_hidden_state=self._model._run_encoder(input_ids, attention_mask)
            )

    def get_encoder(self) -> torch.nn.Module:
        """Return encoder proxy for GenerationMixin."""
        return self._EncoderProxy(self)

    # -----------------------------------------------------------------
    # GenerationMixin interface
    # -----------------------------------------------------------------

    @property
    def device(self) -> torch.device:  # noqa: D102
        return torch.device("cpu")

    def can_generate(self) -> bool:  # noqa: D102
        return True

    @property
    def generation_config(self):  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kw: dict[str, Any] = {}
            if self.config is not None:
                for attr in (
                    "decoder_start_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ):
                    val = getattr(self.config, attr, None)
                    if val is not None:
                        gc_kw[attr] = val
            gc_kw.setdefault("max_new_tokens", self._max_dec - 1)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build decoder inputs for each generate() step."""
        return {
            "decoder_input_ids": input_ids[:, -1:],
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def to(self, *args: Any, **kwargs: Any) -> WinMLModelForSeq2SeqLM:
        """No-op for HF pipeline compatibility."""
        return self

    @property
    def dtype(self) -> torch.dtype:  # noqa: D102
        return torch.float32

    def __call__(self, **kwargs: Any) -> Any:  # noqa: D102
        return self.forward(**kwargs)

    # -----------------------------------------------------------------
    # Forward (decoder via WinMLAutoModel + KV cache)
    # -----------------------------------------------------------------

    def forward(
        self,
        decoder_input_ids: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Run decoder with static KV cache.

        ``past_key_values`` is a HF ``Cache`` — a pre-allocated
        fixed-size buffer mutated in-place via ``index_copy_`` at
        ``cache_position``. The same object flows through GenerationMixin's
        loop across steps, just like HF's DynamicCache.
        """
        # Encoder hidden states
        if encoder_outputs is not None:
            enc_h = (
                encoder_outputs[0]
                if isinstance(encoder_outputs, tuple)
                else encoder_outputs.last_hidden_state
            )
        elif input_ids is not None:
            enc_h = self._run_encoder(input_ids, attention_mask)
        else:
            raise ValueError("Either encoder_outputs or input_ids required")

        # Initialize cache on first call
        if past_key_values is None:
            past_key_values = StaticCache(self.config, max_cache_len=self._max_dec)
            past_key_values.early_initialization(
                batch_size=1,
                num_heads=self._nh,
                head_dim=self._dk,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        # Determine write position from cache occupancy
        fc = past_key_values.get_seq_length()
        dec_mask = torch.zeros(1, self._max_dec, dtype=torch.int64)
        dec_mask[0, : fc + 1] = 1

        # Build feeds for decoder WinMLAutoModel
        feeds: dict[str, torch.Tensor] = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_h.detach(),
            "attention_mask": self._pad_to(attention_mask, self._enc_seq, 0),
            "decoder_attention_mask": dec_mask,
            "cache_position": torch.tensor([fc], dtype=torch.int64),
        }
        for i in range(self._nl):
            layer = past_key_values.layers[i]
            feeds[f"past_{i}_key"] = layer.keys.detach()
            feeds[f"past_{i}_value"] = layer.values.detach()

        outputs = self._decoder(**feeds)

        # Write new token's KV into the StaticCache in-place.
        # StaticCache.update() calls index_copy_ at cache_position.
        cache_kwargs = {"cache_position": torch.tensor([fc], dtype=torch.int64)}
        for i in range(self._nl):
            past_key_values.update(
                outputs[f"present_{i}_key"],
                outputs[f"present_{i}_value"],
                layer_idx=i,
                cache_kwargs=cache_kwargs,
            )

        return Seq2SeqLMOutput(
            logits=outputs["logits"],
            past_key_values=past_key_values,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _pad_to(t: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
        s = t.shape[-1]
        if s == target_len:
            return t
        if s > target_len:
            return t[..., :target_len]
        return torch.nn.functional.pad(t, (0, target_len - s), value=pad_value)
