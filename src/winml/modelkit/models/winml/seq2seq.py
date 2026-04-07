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
- WinMLT5Model: T5-specific sub-model tasks and generation config.

Design principles:
- NEVER guard config access with default values. Use ``self.config.param``
  directly and let AttributeError raise if the config is missing a field.
  Silent defaults hide bugs and make misconfiguration invisible.
- ONNX I/O names and shapes are read from ``io_config``, never hardcoded.
  Encoder/decoder input names are not assumed — kwargs are matched against
  ONNX metadata at runtime.
- Inputs smaller than the ONNX expected shape are zero-padded automatically.
  Inputs larger than expected are NOT truncated — the ONNX runtime will raise
  a clear shape mismatch error.

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

    Input/output names and shapes are read from ONNX I/O metadata — no
    model-specific names are assumed.
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
        raw_encoder = sub_models["encoder"]
        self._decoder = sub_models["decoder"]

        # Build {name: shape} lookups from ONNX I/O metadata
        enc_io = raw_encoder.io_config
        enc_expected = dict(
            zip(enc_io.get("input_names", []), enc_io.get("input_shapes", []), strict=False)
        )
        # Wrap encoder with auto-padding so all callsites just use self._encoder(...)
        self._encoder = self._EncoderWithInputPadding(raw_encoder, enc_expected)

        dec_io = self._decoder.io_config
        self._dec_expected = dict(
            zip(dec_io.get("input_names", []), dec_io.get("input_shapes", []), strict=False)
        )

        # Max decode length from decoder ONNX KV input shape
        self._max_dec = self._dec_expected["past_0_key"][2]
        self._num_kv_layers = sum(
            1 for n in self._dec_expected if n.startswith("past_") and n.endswith("_key")
        )

    # ----- Encoder -----

    class _EncoderWithInputPadding(torch.nn.Module):
        """Wraps an encoder sub-model with auto-padding to ONNX expected shapes.

        Matches kwargs against ONNX input names, pads undersized tensors,
        and forwards to the underlying WinMLAutoModel. Used as both
        ``self._encoder`` (direct calls) and the return value of
        ``get_encoder()`` (GenerationMixin contract).
        """

        def __init__(self, encoder: Any, expected: dict[str, list[int]]) -> None:
            super().__init__()
            self._encoder = encoder
            self._expected = expected

        def forward(self, **kwargs: Any) -> BaseModelOutput:
            feeds = WinMLGenerationModel._pad_inputs(kwargs, self._expected)
            return self._encoder(**feeds)

    def get_encoder(self) -> torch.nn.Module:
        """Return encoder for GenerationMixin (already wrapped with padding)."""
        return self._encoder

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
        *,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        past_key_values: Cache | None = None,
        input_ids: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Run decoder with static KV cache.

        Args:
            encoder_outputs: Pre-computed encoder hidden states.
            past_key_values: StaticCache (or wrapper) from previous step.
            input_ids: Fallback — run encoder if encoder_outputs is None.
            **model_kwargs: Remaining kwargs forwarded to the decoder ONNX
                (e.g., decoder_input_ids, attention_mask). Each tensor is
                auto-padded to match the ONNX model's expected input shape.
        """
        # Encoder hidden states
        if encoder_outputs is None and input_ids is not None:
            encoder_outputs = self._encoder(input_ids=input_ids, **model_kwargs)
        if encoder_outputs is None:
            raise ValueError("Either encoder_outputs or input_ids required")
        enc_h = encoder_outputs["last_hidden_state"]

        # Resolve the self-attention cache.
        # GenerationMixin may pass None, a StaticCache, or an
        # EncoderDecoderCache wrapping a DynamicCache (auto-created).
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

        # Build feeds: model_kwargs first, then fill in generated inputs
        feeds: dict[str, Any] = dict(model_kwargs)
        feeds.setdefault("encoder_hidden_states", enc_h.detach())
        feeds.setdefault("decoder_attention_mask", dec_mask)
        feeds.setdefault("cache_position", torch.tensor([fc], dtype=torch.int64))
        for i in range(self._num_kv_layers):
            layer = cache.layers[i]
            feeds[f"past_{i}_key"] = layer.keys.detach()
            feeds[f"past_{i}_value"] = layer.values.detach()

        # Filter to decoder ONNX inputs and pad any undersized tensors
        outputs = self._decoder(**self._pad_inputs(feeds, self._dec_expected))

        # Write new token's KV into the StaticCache in-place
        cache_kwargs = {"cache_position": torch.tensor([fc], dtype=torch.int64)}
        for i in range(self._num_kv_layers):
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
    def _pad_inputs(
        source: dict[str, Any],
        expected: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Filter *source* to keys in *expected* and pad undersized tensors.

        For each name in *expected*, if *source* has a tensor for it, pad
        any dimension smaller than the ONNX expected shape (skips batch dim).
        Non-tensor values are passed through. Missing names are skipped.
        """
        result: dict[str, Any] = {}
        for name, expected_shape in expected.items():
            val = source.get(name)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                # TODO: support dynamic shape ONNX models (None in expected_shape)
                ndim = min(len(val.shape), len(expected_shape))
                pad: list[int] = []
                for dim in reversed(range(1, ndim)):
                    deficit = expected_shape[dim] - val.shape[dim]
                    pad.extend([0, max(deficit, 0)])
                if any(p > 0 for p in pad):
                    val = torch.nn.functional.pad(val, pad)
            result[name] = val
        return result


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
