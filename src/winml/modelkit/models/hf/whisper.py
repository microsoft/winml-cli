# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Whisper split encoder/decoder export and inference support."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

from ...export import register_onnx_overwrite
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderCore
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

logger = logging.getLogger(__name__)


class WhisperEncoderWrapper(nn.Module):
    """Expose the Whisper audio encoder as a standalone model."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> WhisperEncoderWrapper:
        """Load a conditional-generation checkpoint and retain its encoder."""
        full_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model.get_encoder())
        wrapper.eval()
        return wrapper

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Return the encoded audio hidden states."""
        return cast(
            "torch.Tensor",
            self.encoder(input_features=input_features).last_hidden_state,
        )


class WhisperDecoderWrapper(nn.Module):
    """Expose Whisper decoding with WinML's static self-attention KV cache."""

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.config: PretrainedConfig = cast("PretrainedConfig", model.config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> WhisperDecoderWrapper:
        """Load a conditional-generation checkpoint for decoder export."""
        full_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model, full_model.config.decoder_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert named dummy inputs to the positional export protocol."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode one token and return logits plus new-token self-attention KV."""
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        decoder_attention_mask = args[2]
        cache_position = args[3]
        kv_start = 4

        max_cache_len = args[kv_start].size(2)
        self_attn_cache = WinMLStaticCache(self.config, max_cache_len=max_cache_len)
        self_attn_cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=args[kv_start].size(1),
            head_dim=args[kv_start].size(3),
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for layer_index in range(self.num_layers):
            layer = self_attn_cache.layers[layer_index]
            layer.keys = args[kv_start + layer_index * 2]
            layer.values = args[kv_start + layer_index * 2 + 1]
        self_attn_cache.set_trace_position(cache_position)

        cache = EncoderDecoderCache(self_attn_cache, DynamicCache())
        output = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        result: list[torch.Tensor] = [output.logits]
        for layer_index in range(self.num_layers):
            key, value = self_attn_cache.captured[layer_index]
            result.extend((key, value))
        return tuple(result)


class WhisperEncoderInputGenerator(DummyInputGenerator):  # type: ignore[misc]
    """Generate Whisper log-Mel features at the architecture's fixed length."""

    SUPPORTED_INPUT_NAMES = ("input_features",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.feature_size = normalized_config.feature_size
        self.audio_sequence_length = normalized_config.audio_sequence_length

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate the fixed-size feature tensor required by WhisperEncoder."""
        if input_name != "input_features":
            raise ValueError(f"Unknown input: {input_name}")
        return cast(
            "torch.Tensor",
            self.random_float_tensor(
                (self.batch_size, self.feature_size, self.audio_sequence_length),
                framework=framework,
                dtype=float_dtype,
            ),
        )


class _WhisperEncoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]
    FEATURE_SIZE = "num_mel_bins"
    MAX_SOURCE_POSITIONS = "max_source_positions"

    @property
    def audio_sequence_length(self) -> int:
        return cast("int", self.max_source_positions * 2)


class _WhisperDecoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]
    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "d_model"
    NUM_LAYERS = "decoder_layers"
    NUM_ATTENTION_HEADS = "decoder_attention_heads"
    MAX_CACHE_LEN = "max_target_positions"
    ENCODER_SEQUENCE_LENGTH = "max_source_positions"

    @property
    def head_dim(self) -> int:
        return cast("int", self.hidden_size // self.num_attention_heads)

    @property
    def sequence_length(self) -> int:
        return cast("int", self.encoder_sequence_length)


@register_onnx_overwrite("whisper", "feature-extraction", library_name="transformers")
class WhisperEncoderIOConfig(OnnxConfig):  # type: ignore[misc]
    """ONNX configuration for the fixed-shape Whisper audio encoder."""

    NORMALIZED_CONFIG_CLASS = _WhisperEncoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (WhisperEncoderInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {"input_features": {0: "batch_size"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {"encoder_hidden_states": {0: "batch_size"}}


@register_onnx_overwrite("whisper", "text2text-generation", library_name="transformers")
class WhisperDecoderIOConfig(OnnxConfig):  # type: ignore[misc]
    """ONNX configuration for token-at-a-time Whisper decoding."""

    NORMALIZED_CONFIG_CLASS = _WhisperDecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "decoder_attention_mask": {0: "batch_size"},
            "cache_position": {},
        }
        for layer_index in range(self._normalized_config.num_layers):
            result[f"past_{layer_index}_key"] = {0: "batch_size"}
            result[f"past_{layer_index}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
        for layer_index in range(self._normalized_config.num_layers):
            result[f"present_{layer_index}_key"] = {0: "batch_size"}
            result[f"present_{layer_index}_value"] = {0: "batch_size"}
        return result


MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("whisper", "feature-extraction"): WhisperEncoderWrapper,
    ("whisper", "text2text-generation"): WhisperDecoderWrapper,
}


@register_composite_model("whisper", "automatic-speech-recognition")
class WinMLWhisperModel(  # type: ignore[misc]
    WinMLEncoderDecoderCore, WhisperGenerationMixin
):
    """Split Whisper transcription model backed by encoder and decoder ONNX graphs."""

    main_input_name = "input_features"
    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
        device: str = "cpu",
    ) -> None:
        super().__init__(sub_models, config, device)
        encoder_frames = self._encoder._expected["input_features"][-1]
        input_stride = encoder_frames // config.max_source_positions
        self.model = SimpleNamespace(
            config=config,
            encoder=SimpleNamespace(
                conv1=SimpleNamespace(stride=(1,)),
                conv2=SimpleNamespace(stride=(input_stride,)),
            ),
        )
        model_name_or_path = getattr(config, "_name_or_path", "")
        if model_name_or_path:
            from transformers import GenerationConfig

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
            except OSError:
                logger.warning(
                    "Could not load Whisper generation_config.json from %s; "
                    "language/task controls may be unavailable.",
                    model_name_or_path,
                )

    @classmethod
    def get_cache_class(cls) -> type:  # noqa: D102
        return WinMLStaticCache

    def forward(
        self,
        *,
        input_features: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Run the audio encoder when callers have not precomputed its outputs."""
        if encoder_outputs is None and input_features is not None:
            encoder_outputs = self._encoder(input_features=input_features)
        return super().forward(encoder_outputs=encoder_outputs, **kwargs)

    @property
    def generation_config(self) -> GenerationConfig:  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            values: dict[str, Any] = {}
            for name in (
                "decoder_start_token_id",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "forced_decoder_ids",
                "suppress_tokens",
                "begin_suppress_tokens",
            ):
                value = getattr(self.config, name, None)
                if value is not None:
                    values[name] = value
            values.setdefault("max_new_tokens", self._max_dec - 1)
            values.setdefault("num_beams", 1)
            values.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**values)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


__all__ = [
    "MODEL_CLASS_MAPPING",
    "WhisperDecoderIOConfig",
    "WhisperDecoderWrapper",
    "WhisperEncoderIOConfig",
    "WhisperEncoderInputGenerator",
    "WhisperEncoderWrapper",
    "WinMLWhisperModel",
]
