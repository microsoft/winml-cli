# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Pipeline Models for multi-component architectures.

Provides a three-level class hierarchy for multi-ONNX-model inference:

- WinMLPipelineModel: Base for any model composed of multiple WinMLAutoModel
  sub-components (e.g., encoder+decoder, text_encoder+unet+vae).
- WinMLGenerationModel: Adds GenerationMixin support (encoder/decoder generate
  loop, KV cache management).
- WinMLT5Model: T5-specific config, cache shapes, and forward logic.

Usage:
    from winml.modelkit.models.winml.seq2seq import WinMLT5Model
    from transformers import AutoTokenizer, pipeline

    model = WinMLT5Model.from_pretrained("google-t5/t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
    result = pipe("Hello, how are you?", num_beams=1)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from transformers import Cache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .base import PreTrainedModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


# =========================================================================
# Layer 1: WinMLPipelineModel — multi-component base
# =========================================================================


class WinMLPipelineModel(PreTrainedModel):
    """Base class for models composed of multiple WinMLAutoModel sub-components.

    Subclasses declare ``_SUB_MODEL_CONFIG``: a mapping of component name to
    the HF task used to build it via ``WinMLAutoModel.from_pretrained``.

    After construction, sub-components are available in ``self.sub_models``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
    ) -> None:
        self.sub_models = sub_models
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "cpu",
        use_cache: bool = True,
        force_rebuild: bool = False,
        **kwargs: Any,
    ) -> WinMLPipelineModel:
        """Build all sub-components and return ready-to-use model."""
        from transformers import AutoConfig

        from ..auto import WinMLAutoModel

        hf_config = AutoConfig.from_pretrained(model_id)

        sub_models: dict[str, Any] = {}
        for name, task in cls._SUB_MODEL_CONFIG.items():
            logger.info("Building %s for %s...", name, model_id)
            sub_models[name] = WinMLAutoModel.from_pretrained(
                model_id,
                task=task,
                device=device,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                **kwargs,
            )

        return cls(sub_models=sub_models, config=hf_config)

    @property
    def device(self) -> torch.device:
        """Device (CPU — ORT handles actual placement)."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype for HF compatibility."""
        return torch.float32

    def to(self, *args: Any, **kwargs: Any) -> WinMLPipelineModel:
        """No-op for HF pipeline compatibility."""
        return self

    def __call__(self, **kwargs: Any) -> Any:
        """Inference entry point."""
        return self.forward(**kwargs)

    def forward(self, **kwargs: Any) -> Any:
        """Subclasses implement task-specific logic."""
        raise NotImplementedError


# =========================================================================
# Layer 2: WinMLGenerationModel — encoder-decoder generation
# =========================================================================


class WinMLGenerationModel(WinMLPipelineModel, GenerationMixin):
    """Pipeline model with HF GenerationMixin support.

    Expects sub-components ``"encoder"`` and ``"decoder"`` in
    ``_SUB_MODEL_CONFIG``. Provides the full interface required by
    ``GenerationMixin.generate()`` for encoder-decoder models with
    static KV cache.
    """

    main_input_name = "input_ids"
    base_model_prefix = ""
    _is_stateful = False
    _supports_cache_class = False

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
    ) -> None:
        super().__init__(sub_models, config)
        self._encoder = sub_models["encoder"]
        self._decoder = sub_models["decoder"]

        # Read shapes from ONNX I/O metadata
        dec_io = self._decoder.io_config
        dec_shapes = dict(
            zip(dec_io.get("input_names", []), dec_io.get("input_shapes", []), strict=False)
        )
        kv_shape = dec_shapes.get("past_0_key", [1, 8, 32, 64])
        self._max_dec = kv_shape[2] if len(kv_shape) > 2 else 32

        enc_io = self._encoder.io_config
        enc_shapes = enc_io.get("input_shapes", [])
        self._enc_seq = enc_shapes[0][1] if enc_shapes and len(enc_shapes[0]) > 1 else 16

    # ----- Encoder -----

    def _run_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder sub-model, return hidden states."""
        out = self._encoder(
            input_ids=self._pad_to(input_ids, self._enc_seq, 0),
            attention_mask=self._pad_to(attention_mask, self._enc_seq, 0),
        )
        # WinMLAutoModel may wrap output in BaseModelOutput (last_hidden_state)
        # or return a raw dict (encoder_hidden_states). Handle both.
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out["encoder_hidden_states"]

    class _EncoderProxy(torch.nn.Module):
        """Proxy returned by get_encoder() for GenerationMixin."""

        def __init__(self, model: WinMLGenerationModel) -> None:
            super().__init__()
            self._model = model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **kw: Any,
        ) -> BaseModelOutput:
            return BaseModelOutput(
                last_hidden_state=self._model._run_encoder(input_ids, attention_mask),
            )

    def get_encoder(self) -> torch.nn.Module:
        """Return encoder proxy for GenerationMixin."""
        return self._EncoderProxy(self)

    def can_generate(self) -> bool:  # noqa: D102
        return True

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

    # ----- Forward (decoder via WinMLAutoModel + KV cache) -----

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

        ``past_key_values`` is a HF ``StaticCache`` — a pre-allocated
        fixed-size buffer mutated in-place via ``index_copy_`` at
        ``cache_position``. The same object flows through GenerationMixin's
        loop across steps.
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

        # Resolve the self-attention cache.
        # GenerationMixin may pass None, a StaticCache, or an
        # EncoderDecoderCache wrapping a DynamicCache (auto-created).
        # We need our own StaticCache for the static-buffer ONNX decoder.
        cache = None
        if isinstance(past_key_values, StaticCache):
            cache = past_key_values
        elif hasattr(past_key_values, "self_attention_cache"):
            sa = past_key_values.self_attention_cache
            if isinstance(sa, StaticCache):
                cache = sa
        if cache is None:
            cache = StaticCache(self.config, max_cache_len=self._max_dec)
            cache.early_initialization(
                batch_size=1,
                num_heads=self.config.num_heads,
                head_dim=self.config.d_kv,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        # Determine write position from cache occupancy
        fc = cache.get_seq_length()
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
        for i in range(self.config.num_layers):
            layer = cache.layers[i]
            feeds[f"past_{i}_key"] = layer.keys.detach()
            feeds[f"past_{i}_value"] = layer.values.detach()

        outputs = self._decoder(**feeds)

        # Write new token's KV into the StaticCache in-place.
        # StaticCache.update() calls index_copy_ at cache_position.
        cache_kwargs = {"cache_position": torch.tensor([fc], dtype=torch.int64)}
        for i in range(self.config.num_layers):
            cache.update(
                outputs[f"present_{i}_key"],
                outputs[f"present_{i}_value"],
                layer_idx=i,
                cache_kwargs=cache_kwargs,
            )

        return Seq2SeqLMOutput(
            logits=outputs["logits"],
            past_key_values=cache,
        )

    # ----- Helpers -----

    @staticmethod
    def _pad_to(t: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
        s = t.shape[-1]
        if s == target_len:
            return t
        if s > target_len:
            return t[..., :target_len]
        return torch.nn.functional.pad(t, (0, target_len - s), value=pad_value)


# =========================================================================
# Layer 3: WinMLT5Model — T5-specific forward + cache logic
# =========================================================================


class WinMLT5Model(WinMLGenerationModel):
    """T5 encoder-decoder model.

    Declares T5 sub-component tasks and generation config defaults.
    All encoder-decoder forward/cache logic lives in ``WinMLGenerationModel``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

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
            # Static batch=1 ONNX models don't support beam search
            gc_kw.setdefault("num_beams", 1)
            gc_kw.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


# Backward compat alias
WinMLModelForSeq2SeqLM = WinMLT5Model
