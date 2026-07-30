# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Native Florence-2 split image-to-text export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator
from transformers import Florence2Config, Florence2ForConditionalGeneration, Florence2Processor
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import Transpose, WeightConverter, WeightRenaming

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache
from .decoder_wrapper import WinMLDecoderWrapper, WinMLStaticCacheDecoderIOConfig


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig


FLORENCE2_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        layer_norm_fusion=True,
        matmul_add_fusion=True,
    ),
)

_FLORENCE2_IMAGE_TOKEN = "<image>"  # noqa: S105 - model vocabulary token
_FLORENCE2_LEGACY_IMAGE_TOKEN_ID = 50265
_FLORENCE2_CAPTION_TOKEN_IDS = (0, 2264, 473, 5, 2274, 6190, 116, 2)
_FLORENCE2_ENCODER_SEQUENCE_LENGTH = 585
_PROCESSOR_HUB_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
}


def _legacy_florence2_weight_conversions() -> list[Any]:
    return [
        WeightRenaming(r"\.convs\.(\d+)\.proj\.", r".convs.\1.conv."),
        WeightRenaming(r"\.conv1\.fn\.dw\.", ".conv1."),
        WeightRenaming(r"\.conv2\.fn\.dw\.", ".conv2."),
        WeightRenaming(r"\.ffn\.fn\.net\.", ".ffn."),
        WeightRenaming(r"\.window_attn\.fn\.", ".window_attn."),
        WeightRenaming(r"\.window_attn\.norm\.", ".norm1."),
        WeightRenaming(r"\.channel_attn\.fn\.", ".channel_attn."),
        WeightRenaming(r"\.channel_attn\.norm\.", ".norm1."),
        WeightRenaming(r"\.ffn\.norm\.", ".norm2."),
        WeightRenaming(r"^vision_tower\.", "model.vision_tower."),
        WeightRenaming(r"^language_model\.model\.", "model.language_model."),
        WeightRenaming(
            r"^image_pos_embed\.",
            "model.multi_modal_projector.image_position_embed.",
        ),
        WeightRenaming(
            r"^image_proj_norm\.", "model.multi_modal_projector.image_proj_norm."
        ),
        WeightRenaming(
            r"^visual_temporal_embed\.",
            "model.multi_modal_projector.visual_temporal_embed.",
        ),
        WeightConverter(
            source_patterns="^image_projection$",
            target_patterns="model.multi_modal_projector.image_projection.weight",
            operations=[Transpose()],
        ),
    ]


class _WinMLFlorence2ForConditionalGeneration(Florence2ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = [  # noqa: RUF012 - Transformers class contract
        r"language_model\.final_logits_bias"
    ]


register_checkpoint_conversion_mapping(
    _WinMLFlorence2ForConditionalGeneration.__name__,
    _legacy_florence2_weight_conversions(),
)


def _load_florence2_processor(model_name_or_path: str, **kwargs: Any) -> Florence2Processor:
    processor_kwargs = {key: value for key, value in kwargs.items() if key in _PROCESSOR_HUB_KWARGS}
    return Florence2Processor.from_pretrained(
        model_name_or_path,
        extra_special_tokens={"image_token": _FLORENCE2_IMAGE_TOKEN},
        **processor_kwargs,
    )


def _load_florence2_model(
    model_name_or_path: str, **kwargs: Any
) -> Florence2ForConditionalGeneration:
    model_kwargs = dict(kwargs)
    model_kwargs.setdefault("dtype", torch.float32)
    config = model_kwargs.pop("config", None)
    if config is None:
        config_kwargs = {
            key: value for key, value in model_kwargs.items() if key in _PROCESSOR_HUB_KWARGS
        }
        config = Florence2Config.from_pretrained(model_name_or_path, **config_kwargs)
    if config.image_token_id >= config.text_config.vocab_size:
        config.image_token_id = _load_florence2_processor(
            model_name_or_path, **model_kwargs
        ).image_token_id
    model_kwargs.pop("output_loading_info", None)
    model, loading_info = cast(
        "tuple[Florence2ForConditionalGeneration, dict[str, Any]]",
        cast("Any", _WinMLFlorence2ForConditionalGeneration.from_pretrained)(
            model_name_or_path,
            config=config,
            output_loading_info=True,
            **model_kwargs,
        ),
    )
    unresolved = {
        key: loading_info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if unresolved:
        raise RuntimeError(
            f"Florence-2 checkpoint conversion left unresolved weights: {unresolved}"
        )
    return model


class _Florence2EncoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]
    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.num_channels = config.vision_config.in_channels
        self.image_size = 768
        self.image_token_id = (
            config.image_token_id
            if config.image_token_id < config.text_config.vocab_size
            else _FLORENCE2_LEGACY_IMAGE_TOKEN_ID
        )
        self.bos_token_id = config.text_config.bos_token_id
        self.eos_token_id = config.text_config.eos_token_id


class _Florence2EncoderInputGenerator(DummyInputGenerator):  # type: ignore[misc]
    SUPPORTED_INPUT_NAMES = ("input_ids", "pixel_values", "attention_mask")

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        del task
        self.batch_size = kwargs.get("batch_size", 1)
        self.image_size = normalized_config.image_size
        self.num_channels = normalized_config.num_channels
        self.image_token_id = normalized_config.image_token_id
        self.bos_token_id = normalized_config.bos_token_id
        self.eos_token_id = normalized_config.eos_token_id

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        del framework, int_dtype, float_dtype
        sequence_length = _FLORENCE2_ENCODER_SEQUENCE_LENGTH
        if input_name == "input_ids":
            image_tokens = torch.full(
                (self.batch_size, 577), self.image_token_id, dtype=torch.long
            )
            caption_tokens = torch.tensor(
                _FLORENCE2_CAPTION_TOKEN_IDS, dtype=torch.long
            ).repeat(self.batch_size, 1)
            return torch.cat((image_tokens, caption_tokens), dim=1)
        if input_name == "pixel_values":
            return torch.zeros(
                (self.batch_size, self.num_channels, self.image_size, self.image_size),
                dtype=torch.float32,
            )
        if input_name == "attention_mask":
            return torch.ones((self.batch_size, sequence_length), dtype=torch.long)
        raise ValueError(f"Unknown input: {input_name}")


class Florence2EncoderWrapper(nn.Module):
    """Export the native Florence image/text encoder as one ONNX component."""

    def __init__(self, model: Florence2ForConditionalGeneration) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Florence2EncoderWrapper:
        """Load converted native weights and return an evaluation-mode wrapper."""
        model = _load_florence2_model(model_name_or_path, **kwargs)
        wrapper = cls(model)
        wrapper.eval()
        return wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Merge image features into placeholder embeddings and run the encoder."""
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        image_features = self.model.get_image_features(pixel_values).pooler_output
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        placeholder_mask = self.model.get_placeholder_mask(
            cast("torch.LongTensor", input_ids),
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(placeholder_mask, image_features)
        outputs = self.model.model.language_model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        return cast("torch.Tensor", outputs.last_hidden_state)


@register_onnx_overwrite("florence2", "feature-extraction", library_name="transformers")
class Florence2EncoderIOConfig(OnnxConfig):  # type: ignore[misc]
    """Declare semantic Florence encoder inputs and hidden-state output."""

    NORMALIZED_CONFIG_CLASS = _Florence2EncoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (_Florence2EncoderInputGenerator,)
    PRESERVE_DUMMY_VALUE_RUNS = True

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return encoder input dynamic-axis metadata."""
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return encoder output dynamic-axis metadata."""
        return {"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}


class _Florence2DecoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]
    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._text_config = config.text_config

    @property
    def hidden_size(self) -> int:
        return cast("int", self._text_config.hidden_size)

    @property
    def num_layers(self) -> int:
        return cast("int", self._text_config.decoder_layers)

    @property
    def num_attention_heads(self) -> int:
        return cast("int", self._text_config.decoder_attention_heads)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def max_cache_len(self) -> int:
        return cast("int", self._text_config.max_position_embeddings)

    @property
    def vocab_size(self) -> int:
        return cast("int", self._text_config.vocab_size)


class _Florence2DecoderInputGenerator(EncoderDecoderInputGenerator):
    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        kwargs.setdefault("sequence_length", _FLORENCE2_ENCODER_SEQUENCE_LENGTH)
        super().__init__(task, normalized_config, **kwargs)


@register_onnx_overwrite("florence2", "text2text-generation", library_name="transformers")
class Florence2DecoderIOConfig(WinMLStaticCacheDecoderIOConfig):
    """Declare the six-layer Florence decoder static-cache contract."""

    NORMALIZED_CONFIG_CLASS = _Florence2DecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        _Florence2DecoderInputGenerator,
        PastKeyValueInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return decoder and past-cache input metadata."""
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "decoder_attention_mask": {0: "batch_size"},
            "cache_position": {},
        }
        for index in range(self._normalized_config.num_layers):
            result[f"past_{index}_key"] = {0: "batch_size"}
            result[f"past_{index}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return logits and present-cache output metadata."""
        result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
        for index in range(self._normalized_config.num_layers):
            result[f"present_{index}_key"] = {0: "batch_size"}
            result[f"present_{index}_value"] = {0: "batch_size"}
        return result


class Florence2DecoderWrapper(WinMLDecoderWrapper):
    """Export the native Florence language decoder with a static KV cache."""

    _HF_MODEL_CLS = Florence2ForConditionalGeneration
    _IO_CONFIG_CLS = Florence2DecoderIOConfig

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Florence2DecoderWrapper:
        """Load converted native weights and initialize shared decoder state."""
        model = _load_florence2_model(model_name_or_path, **kwargs)
        wrapper = cls()
        wrapper.model = model
        wrapper.config = model.config
        wrapper.onnx_config = cls._IO_CONFIG_CLS(model.config, task=cls._TASK)
        wrapper.num_layers = wrapper.onnx_config._normalized_config.num_layers
        wrapper.eval()
        return wrapper

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        model = cast("Any", self.model)
        encoder_hidden_states = inputs["encoder_hidden_states"]
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2], dtype=torch.long, device=encoder_hidden_states.device
        )
        outputs = model.model.language_model.decoder(
            input_ids=inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=EncoderDecoderCache(cache, DynamicCache()),
            use_cache=True,
            cache_position=inputs["cache_position"],
            return_dict=True,
        )
        return cast("torch.Tensor", model.lm_head(outputs.last_hidden_state))


@register_composite_model("florence2", "image-text-to-text")
@register_composite_model("florence2", "image-to-text")
class WinMLFlorence2ImageToText(WinMLEncoderDecoderModel):
    """Run split Florence encoder and decoder components for image-to-text."""

    main_input_name = "input_ids"
    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }
    _SUB_MODEL_PRECISION_OVERRIDES: ClassVar[dict[str, str]] = {"encoder": "fp32"}

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
        device: str = "cpu",
    ) -> None:
        super().__init__(sub_models, config, device)
        self.config.is_encoder_decoder = True
        self.config.num_hidden_layers = config.text_config.decoder_layers

    @classmethod
    def get_cache_class(cls) -> type:
        """Use a fixed-capacity cache matching the exported decoder graph."""
        return WinMLStaticCache

    def load_pipeline_processor(self, model_id: str) -> Florence2Processor:
        """Load the native processor with the Florence image token registered."""
        return _load_florence2_processor(model_id)

    @property
    def generation_config(self) -> GenerationConfig:
        """Return deterministic greedy generation defaults within cache capacity."""
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            text_config = self.config.text_config
            self._generation_config = GenerationConfig(
                decoder_start_token_id=text_config.decoder_start_token_id,
                bos_token_id=text_config.bos_token_id,
                eos_token_id=text_config.eos_token_id,
                pad_token_id=text_config.pad_token_id,
                max_new_tokens=self._max_dec - 1,
                num_beams=1,
                do_sample=False,
            )
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("florence2", None): WinMLFlorence2ImageToText,
    ("florence2", "image-to-text"): WinMLFlorence2ImageToText,
    ("florence2", "image-text-to-text"): WinMLFlorence2ImageToText,
    ("florence2", "feature-extraction"): Florence2EncoderWrapper,
    ("florence2", "text2text-generation"): Florence2DecoderWrapper,
}


__all__ = [
    "FLORENCE2_CONFIG",
    "MODEL_CLASS_MAPPING",
    "Florence2DecoderIOConfig",
    "Florence2DecoderWrapper",
    "Florence2EncoderIOConfig",
    "Florence2EncoderWrapper",
    "WinMLFlorence2ImageToText",
]
