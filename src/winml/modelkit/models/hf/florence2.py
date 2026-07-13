# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Florence-2 split image-to-text export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import PipelineCapability, register_composite_model
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


class _NativeFlorence2ForConditionalGeneration:
    """Load the checkpoint with its upstream model implementation."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: Any
    ) -> nn.Module:
        """Load all checkpoint tensors through the model's upstream contract."""
        from transformers import AutoModelForCausalLM

        kwargs["output_loading_info"] = True
        kwargs.setdefault("attn_implementation", "eager")
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        unresolved = {
            name: loading_info[name]
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys")
            if loading_info[name]
        }
        if unresolved:
            counts = ", ".join(f"{name}={len(keys)}" for name, keys in unresolved.items())
            raise RuntimeError(f"Checkpoint reconciliation failed: {counts}.")
        return model


def _load_native_combined_processor(model_id: str, *, trust_remote_code: bool = False) -> Any:
    """Load the processor that accompanies the upstream model implementation."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)


class _Florence2EncoderInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Generate image placeholders together with a caption prompt."""

    SUPPORTED_INPUT_NAMES = ("input_ids", "pixel_values", "attention_mask")

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        self.batch_size = kwargs.get("batch_size", 1)
        self.image_size = normalized_config.image_size
        self.num_channels = normalized_config.num_channels

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        del framework, int_dtype, float_dtype
        sequence_length = 8
        if input_name == "input_ids":
            return torch.zeros((self.batch_size, sequence_length), dtype=torch.long)
        if input_name == "pixel_values":
            return torch.zeros(
                (self.batch_size, self.num_channels, self.image_size, self.image_size),
                dtype=torch.float32,
            )
        if input_name == "attention_mask":
            return torch.ones((self.batch_size, sequence_length), dtype=torch.long)
        raise ValueError(f"Unknown input: {input_name}")


class _Florence2EncoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]  # optimum base is untyped
    """Normalize Florence-2's image and prompt dimensions."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        vision_config = config.vision_config
        self.num_channels = getattr(
            vision_config, "num_channels", getattr(vision_config, "in_channels", 3)
        )
        self.image_size = 768


class Florence2EncoderWrapper(nn.Module):
    """Export Florence-2's image-aware text encoder."""

    def __init__(self, model: nn.Module, config: Any) -> None:
        super().__init__()
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Florence2EncoderWrapper:
        """Load Florence-2 and retain its image-aware encoder."""
        full = _NativeFlorence2ForConditionalGeneration.from_pretrained(
            model_name_or_path, **kwargs
        )
        wrapper = cls(full, full.config)
        wrapper.eval()
        return wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the caption prompt after replacing image placeholders."""
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = self.model._encode_image(pixel_values).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        inputs_embeds, _ = self.model._merge_input_ids_with_image_features(
            image_features, inputs_embeds
        )
        image_attention_mask = attention_mask.new_ones(
            (attention_mask.size(0), image_features.size(1))
        )
        attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        outputs = self.model.get_encoder()(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        return cast("torch.Tensor", outputs.last_hidden_state)


@register_onnx_overwrite("florence2", "feature-extraction", library_name="transformers")
class Florence2EncoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for Florence-2's prompt-aware encoder."""

    NORMALIZED_CONFIG_CLASS = _Florence2EncoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (_Florence2EncoderInputGenerator,)
    PRESERVE_DUMMY_VALUE_RUNS = True

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {"last_hidden_state": {0: "batch_size", 1: "sequence_length"}}


class _Florence2DecoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]  # optimum base is untyped
    """Normalize Florence-2's BART decoder configuration."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._text_config = config.text_config

    @property
    def hidden_size(self) -> int:
        return cast("int", self._text_config.d_model)

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


@register_onnx_overwrite("florence2", "text2text-generation", library_name="transformers")
class Florence2DecoderIOConfig(WinMLStaticCacheDecoderIOConfig):
    """ONNX config for Florence-2's static-cache BART decoder."""

    NORMALIZED_CONFIG_CLASS = _Florence2DecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "decoder_attention_mask": {0: "batch_size"},
            "cache_position": {},
        }
        for i in range(self._normalized_config.num_layers):
            result[f"past_{i}_key"] = {0: "batch_size"}
            result[f"past_{i}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
        for i in range(self._normalized_config.num_layers):
            result[f"present_{i}_key"] = {0: "batch_size"}
            result[f"present_{i}_value"] = {0: "batch_size"}
        return result


class Florence2DecoderWrapper(WinMLDecoderWrapper):
    """Export Florence-2's BART decoder and language-model head."""

    _HF_MODEL_CLS = _NativeFlorence2ForConditionalGeneration
    _IO_CONFIG_CLS = Florence2DecoderIOConfig

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        cache_position = inputs["cache_position"].squeeze()
        legacy_cache = tuple(
            (
                layer.keys[:, :, :cache_position, :],
                layer.values[:, :, :cache_position, :],
            )
            for layer in cache.layers
        )
        decoder_outputs = self.model.get_decoder()(
            input_ids=inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"][
                :, : cache_position + inputs["decoder_input_ids"].size(1)
            ],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_attention_mask=None,
            past_key_values=legacy_cache,
            use_cache=True,
            return_dict=True,
        )
        for index, (key, value, *_) in enumerate(decoder_outputs.past_key_values):
            cache.captured[index] = (
                key[:, :, cache_position:, :],
                value[:, :, cache_position:, :],
            )
        return cast(
            "torch.Tensor", self.model.language_model.lm_head(decoder_outputs.last_hidden_state)
        )


@register_composite_model("florence2", "image-to-text")
class WinMLFlorence2ImageToText(WinMLEncoderDecoderModel):
    """Florence-2 image-to-text inference model."""

    main_input_name = "pixel_values"
    pipeline_capabilities = frozenset({PipelineCapability.COMBINED_IMAGE_TEXT_PROCESSOR})
    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }

    def __init__(self, sub_models: dict[str, Any], config: PretrainedConfig) -> None:
        super().__init__(sub_models, config)
        self.config.is_encoder_decoder = True

    def create_combined_processor(self, model_id: str) -> Any:
        """Load the processor required by the declared preprocessing capability."""
        return _load_native_combined_processor(
            model_id,
            trust_remote_code=self._trust_remote_code,
        )

    @classmethod
    def get_cache_class(cls) -> type:  # noqa: D102
        return WinMLStaticCache

    @property
    def generation_config(self) -> GenerationConfig:  # noqa: D102
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
