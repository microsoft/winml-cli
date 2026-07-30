# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for WinMLCompositeModel.from_pretrained (ep_device signature).

Guards:
  * Sub-model construction dispatches to WinMLAutoModel.from_pretrained without
    a TypeError from missing ``ep_device`` (B3).
  * The caller-supplied ``device`` value is forwarded into ``__init__`` rather
    than being silently defaulted to ``"cpu"`` (B6).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from winml.modelkit.models.winml.composite_model import (
    COMPOSITE_MODEL_REGISTRY,
    WinMLCompositeModel,
)


class _StubComposite(WinMLCompositeModel):
    """Concrete subclass with a single sub-component for isolation."""

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {"encoder": "feature-extraction"}


@pytest.fixture
def stub_registry_hit():
    """Register _StubComposite under a synthetic (model_type, task) pair."""
    key = ("_test_model_type", "_test_task")
    COMPOSITE_MODEL_REGISTRY[key] = _StubComposite
    try:
        yield key
    finally:
        COMPOSITE_MODEL_REGISTRY.pop(key, None)


def _fake_ep_device() -> object:
    device = SimpleNamespace(device_type="NPU", ep_name="QNNExecutionProvider")
    return SimpleNamespace(device=device)


def test_from_pretrained_does_not_raise_typeerror() -> None:
    """Sub-model dispatch must call WinMLAutoModel.from_pretrained without TypeError.

    Mirrors the production caller (:meth:`WinMLAutoModel.from_pretrained` at
    ``models/auto.py:329-341``) which passes ``device=`` and no ``ep_device``.
    The composite must derive/forward ``ep_device`` to the sub-model call.
    """
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "winml.modelkit.loader.load_hf_config",
            return_value=hf_cfg,
        ),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
        ) as mock_from_pretrained,
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="cpu", device="cpu", source=None),
        ),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
    ):
        mock_from_pretrained.return_value = MagicMock()
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        # Direct-subclass path (skips the registry lookup at line 156).
        # Production callers pass device= only — the composite must resolve
        # ep_device internally and pass it to the sub-model dispatch.
        _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="cpu",
        )

    # WinMLAutoModel.from_pretrained must have been reached for the sub-model
    # without raising TypeError on ep_device.
    assert mock_from_pretrained.called
    call = mock_from_pretrained.call_args
    assert fake_ep_device in call.args or call.kwargs.get("ep_device") is fake_ep_device, (
        f"Expected ep_device in call, got args={call.args!r} kwargs={list(call.kwargs)!r}"
    )


def test_device_is_forwarded_from_from_pretrained() -> None:
    """The device kwarg supplied to from_pretrained must reach __init__ (not default to cpu)."""
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "winml.modelkit.loader.load_hf_config",
            return_value=hf_cfg,
        ),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        result = _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="npu",
            ep_device=fake_ep_device,
        )

    # Caller intent must survive — not clobbered by the "cpu" default.
    assert result._device == "npu"


def test_from_pretrained_resolves_explicit_ep_without_ep_device() -> None:
    """Direct composite construction must not discard an explicit EP request."""
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()
    targets: list[object] = []

    def resolve_target(target: object) -> object:
        targets.append(target)
        return target

    with (
        patch("winml.modelkit.loader.load_hf_config", return_value=hf_cfg),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
        patch("winml.modelkit.session.resolve_device", side_effect=resolve_target),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
    ):
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="npu",
            ep="qnn",
        )

    assert targets
    assert targets[0].ep == "qnn"


def test_sub_models_receive_original_request_target() -> None:
    """Composite sub-model builds must not infer export policy from the runtime handle."""
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()

    with (
        patch("winml.modelkit.loader.load_hf_config", return_value=hf_cfg),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as mock_from_pretrained,
    ):
        _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="auto",
            ep_device=fake_ep_device,
        )

    call_kwargs = mock_from_pretrained.call_args.kwargs
    assert call_kwargs["device"] == "auto"
    assert call_kwargs["ep"] is None


def test_device_is_forwarded_from_from_onnx() -> None:
    """The resolved component target must also reach the composite wrapper."""
    fake_ep_device = _fake_ep_device()

    with patch(
        "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
        return_value=MagicMock(),
    ) as mock_from_onnx:
        result = _StubComposite.from_onnx(
            {"encoder": "encoder.onnx"},
            hf_config=SimpleNamespace(model_type="_test_model_type"),
            ep_device=fake_ep_device,
        )

    assert mock_from_onnx.call_args.kwargs["ep_device"] is fake_ep_device
    assert result.ort_device == "npu"


def test_composite_reports_no_switchable_pytorch_experts() -> None:
    model = _StubComposite({}, MagicMock())

    assert model.get_experts_implementation() == {"": None}


def test_encoder_decoder_accepts_resolved_device() -> None:
    from winml.modelkit.models.winml.encoder_decoder import WinMLEncoderDecoderModel

    encoder = SimpleNamespace(io_config={})
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )

    model = WinMLEncoderDecoderModel(
        {"encoder": encoder, "decoder": decoder},
        MagicMock(),
        device="npu",
    )

    assert model._device == "npu"


def test_encoder_decoder_generation_accepts_declared_encoder_input() -> None:
    from winml.modelkit.models.winml.encoder_decoder import WinMLEncoderDecoderModel

    encoder = SimpleNamespace(
        io_config={
            "input_names": ["encoder_input"],
            "input_shapes": [[1, 3, 8, 8]],
        }
    )
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )
    model = WinMLEncoderDecoderModel(
        {"encoder": encoder, "decoder": decoder},
        MagicMock(is_encoder_decoder=True),
    )

    model._validate_model_kwargs({"encoder_input": MagicMock()})


def test_encoder_decoder_slices_input_for_populated_wrapped_cache() -> None:
    from winml.modelkit.models.winml.encoder_decoder import WinMLEncoderDecoderModel
    from winml.modelkit.models.winml.kv_cache import WinMLCache, WinMLStaticCache

    model = object.__new__(WinMLEncoderDecoderModel)
    inner_cache = MagicMock(spec=WinMLStaticCache)
    assert isinstance(inner_cache, WinMLCache)
    inner_cache.get_seq_length.return_value = 2
    wrapped_cache = SimpleNamespace(self_attention_cache=inner_cache)

    prepared = model.prepare_inputs_for_generation(
        torch.tensor([[11, 22, 33]]),
        past_key_values=wrapped_cache,
    )

    torch.testing.assert_close(
        prepared["decoder_input_ids"],
        torch.tensor([[33]]),
    )


def test_encoder_decoder_prefills_every_prompt_token() -> None:
    from winml.modelkit.models.winml.encoder_decoder import WinMLEncoderDecoderModel

    class _RecordingDecoder:
        def __init__(self) -> None:
            self.io_config = {
                "input_names": [
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "past_0_key",
                    "past_0_value",
                ],
                "input_shapes": [[1, 1], [1, 8], [1, 2, 8, 4], [1, 2, 8, 4]],
                "input_types": [np.int64, np.int64, np.float32, np.float32],
            }
            self.tokens: list[int] = []
            self.masks: list[torch.Tensor] = []

        def __call__(self, **feeds):
            token = int(feeds["decoder_input_ids"].item())
            self.tokens.append(token)
            self.masks.append(feeds["decoder_attention_mask"].clone())
            return {
                "logits": torch.full((1, 1, 4), float(token)),
                "present_0_key": torch.zeros(1, 2, 1, 4),
                "present_0_value": torch.zeros(1, 2, 1, 4),
            }

    decoder = _RecordingDecoder()
    model = WinMLEncoderDecoderModel(
        {
            "encoder": SimpleNamespace(io_config={}),
            "decoder": decoder,
        },
        MagicMock(is_encoder_decoder=True),
    )
    cache = MagicMock()
    cache.step = 0
    cache.layers = [
        SimpleNamespace(
            keys=torch.zeros(1, 2, 8, 4),
            values=torch.zeros(1, 2, 8, 4),
        )
    ]

    def _build_decoder_mask(max_len, num_new_tokens=1):
        mask = torch.zeros(1, max_len)
        mask[:, : cache.step + num_new_tokens] = 1
        return mask

    cache.build_decoder_mask.side_effect = _build_decoder_mask
    cache.get_query_cache_position.side_effect = lambda max_len, num_new_tokens=1: torch.arange(
        cache.step,
        cache.step + num_new_tokens,
    )

    def _advance(outputs):
        cache.step += outputs["present_0_key"].shape[2]

    cache.update_all_layers.side_effect = _advance

    decoder_attention_mask = torch.tensor([[0, 1, 1]])
    prepared = model.prepare_inputs_for_generation(
        torch.tensor([[11, 22, 33]]),
        decoder_attention_mask=decoder_attention_mask,
    )
    assert prepared["decoder_attention_mask"] is decoder_attention_mask

    with patch.object(model, "_resolve_cache", return_value=cache):
        result = model.forward(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=torch.zeros(1, 2, 4),
            ),
            decoder_input_ids=torch.tensor([[11, 22, 33]]),
            decoder_attention_mask=decoder_attention_mask,
        )

    assert decoder.tokens == [11, 22, 33]
    assert [mask.tolist() for mask in decoder.masks] == [
        [[0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0]],
        [[0, 1, 1, 0, 0, 0, 0, 0]],
    ]
    torch.testing.assert_close(
        result.logits[:, :, 0],
        torch.tensor([[11.0, 22.0, 33.0]]),
    )
    assert cache.step == 3


def test_static_cache_rejects_out_of_range_query_positions() -> None:
    from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

    cache = object.__new__(WinMLStaticCache)
    cache.step = 7

    with pytest.raises(ValueError, match="capacity"):
        cache.get_query_cache_position(max_len=8, num_new_tokens=2)


def test_blip_composite_accepts_resolved_device() -> None:
    from transformers import BlipConfig

    model_cls = COMPOSITE_MODEL_REGISTRY[("blip", "image-to-text")]
    encoder = SimpleNamespace(io_config={})
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )

    model = model_cls(
        {"encoder": encoder, "decoder": decoder},
        BlipConfig(),
        device="gpu",
    )

    assert model._device == "gpu"


def _make_blip_generation_model() -> object:
    from transformers import BlipConfig

    model_cls = COMPOSITE_MODEL_REGISTRY[("blip", "image-to-text")]
    encoder = MagicMock()
    encoder.io_config = {
        "input_names": ["pixel_values"],
        "input_shapes": [[1, 3, 2, 2]],
    }
    encoder.return_value = BaseModelOutput(last_hidden_state=torch.zeros(1, 4, 8))
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )
    return model_cls(
        {"encoder": encoder, "decoder": decoder},
        BlipConfig(),
    )


class _GenerationLengthPreparedError(Exception):
    """Stop generation after Hugging Face has normalized decoder lengths."""


def _capture_prepared_generation(
    model: object,
    *generate_args: object,
    **generation_kwargs: object,
) -> tuple[int, object]:
    original_prepare = model._prepare_generated_length  # type: ignore[attr-defined]
    captured: dict[str, object] = {}

    def capture(
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        prepared = original_prepare(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            input_ids_length=input_ids_length,
            inputs_tensor=inputs_tensor,
        )
        captured["input_ids_length"] = input_ids_length
        captured["generation_config"] = prepared
        raise _GenerationLengthPreparedError

    with (
        patch.object(model, "_prepare_generated_length", side_effect=capture),
        pytest.raises(_GenerationLengthPreparedError),
    ):
        model.generate(*generate_args, **generation_kwargs)  # type: ignore[attr-defined]

    return int(captured["input_ids_length"]), captured["generation_config"]


def test_static_generation_budgets_normalized_decoder_input_alias() -> None:
    model = _make_blip_generation_model()

    input_ids_length, prepared_config = _capture_prepared_generation(
        model,
        pixel_values=torch.zeros(1, 3, 2, 2),
        input_ids=torch.tensor([[11, 22]]),
        max_new_tokens=100,
    )

    assert input_ids_length == 3
    assert prepared_config.max_new_tokens == 5
    assert prepared_config.max_length == 8


def test_static_generation_honors_positional_generation_config() -> None:
    from transformers import GenerationConfig

    model = _make_blip_generation_model()
    model.generation_config.max_new_tokens = 100
    generation_config = GenerationConfig(
        max_new_tokens=2,
        decoder_start_token_id=model.generation_config.decoder_start_token_id,
    )
    original_max_length = generation_config.max_length

    input_ids_length, prepared_config = _capture_prepared_generation(
        model,
        torch.zeros(1, 3, 2, 2),
        generation_config,
        input_ids=torch.tensor([[11, 22]]),
    )

    assert input_ids_length == 3
    assert prepared_config.max_new_tokens == 2
    assert prepared_config.max_length == 5
    assert generation_config.max_new_tokens == 2
    assert generation_config.max_length == original_max_length


def test_static_generation_preserves_hf_length_conflict_precedence() -> None:
    from transformers import GenerationConfig

    model = _make_blip_generation_model()
    generation_config = GenerationConfig(
        max_length=6,
        max_new_tokens=4,
        decoder_start_token_id=model.generation_config.decoder_start_token_id,
    )
    _, prepared_config = _capture_prepared_generation(
        model,
        torch.zeros(1, 3, 2, 2),
        generation_config,
        input_ids=torch.tensor([[model.generation_config.decoder_start_token_id, 22, 33]]),
        max_length=5,
    )

    assert prepared_config.max_new_tokens == 4
    assert prepared_config.max_length == 7
    assert generation_config.max_length == 6
    assert generation_config.max_new_tokens == 4


def test_static_generation_rejects_normalized_prompt_at_capacity() -> None:
    model = _make_blip_generation_model()

    with pytest.raises(ValueError, match="exhausts the static KV cache"):
        _capture_prepared_generation(
            model,
            pixel_values=torch.zeros(1, 3, 2, 2),
            decoder_input_ids=torch.tensor([[11, 22, 33, 44, 55, 66, 77]]),
            max_new_tokens=1,
        )


def test_sliding_window_generation_preserves_requested_budget() -> None:
    """A sliding-window cache evicts history instead of imposing a hard generation limit."""
    from winml.modelkit.models.hf.mu2 import WinMLMu2Model

    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )
    model = WinMLMu2Model(
        {"encoder": SimpleNamespace(io_config={}), "decoder": decoder},
        MagicMock(),
    )
    from transformers import GenerationConfig

    generation_config = GenerationConfig(max_new_tokens=12)
    prepared_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=True,
        has_default_min_length=True,
        model_input_name="input_ids",
        input_ids_length=3,
        inputs_tensor=torch.tensor([[11, 22, 33]]),
    )

    assert prepared_config.max_new_tokens == 12
    assert prepared_config.max_length == 15


def test_decoder_only_accepts_resolved_device() -> None:
    from winml.modelkit.models.winml.decoder_only import WinMLDecoderOnlyModel

    prefill = SimpleNamespace(
        io_config={
            "input_names": ["input_ids"],
            "input_shapes": [[1, 4]],
        }
    )
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["past_0_key"],
            "input_shapes": [[1, 2, 8, 4]],
            "input_types": [np.float32],
        }
    )

    model = WinMLDecoderOnlyModel(
        {"decoder_prefill": prefill, "decoder_gen": decoder},
        MagicMock(),
        device="gpu",
    )

    assert model._device == "gpu"


def test_zero_shot_image_classification_accepts_resolved_device() -> None:
    from winml.modelkit.models.winml.zero_shot_image_classification import (
        WinMLModelForZeroShotImageClassification,
    )

    model = WinMLModelForZeroShotImageClassification(
        {},
        MagicMock(),
        device="gpu",
    )

    assert model._device == "gpu"
