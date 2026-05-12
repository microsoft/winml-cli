# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``apply_loader_config_overrides``.

The helper deep-merges an overrides dict onto a HF :class:`PretrainedConfig`
by going through ``to_dict`` / ``from_dict``. Tests exercise:

* No-op behaviour for empty / ``None`` overrides.
* Simple scalar overrides.
* Recursive merge into nested sub-configs (CLIP-style).
* Non-mutation of the original config.
* Concrete return type matches the input config type.
"""

from __future__ import annotations

import pytest

from winml.modelkit.loader.config import apply_loader_config_overrides


@pytest.fixture
def bert_config():
    from transformers import BertConfig

    return BertConfig(hidden_size=768, num_attention_heads=12)


@pytest.fixture
def clip_config():
    from transformers import CLIPConfig

    return CLIPConfig()


class TestNoOp:
    """``None`` / empty overrides return the same instance unchanged."""

    def test_none_overrides_returns_same_instance(self, bert_config):
        result = apply_loader_config_overrides(bert_config, None)
        assert result is bert_config

    def test_empty_dict_returns_same_instance(self, bert_config):
        result = apply_loader_config_overrides(bert_config, {})
        assert result is bert_config


class TestScalarOverride:
    """Single-level scalar patches."""

    def test_scalar_value_applied(self, bert_config):
        result = apply_loader_config_overrides(bert_config, {"hidden_size": 1024})
        assert result.hidden_size == 1024

    def test_does_not_mutate_original(self, bert_config):
        _ = apply_loader_config_overrides(bert_config, {"hidden_size": 1024})
        assert bert_config.hidden_size == 768

    def test_returned_config_is_same_concrete_type(self, bert_config):
        result = apply_loader_config_overrides(bert_config, {"hidden_size": 512})
        assert type(result) is type(bert_config)

    def test_returned_config_is_new_instance(self, bert_config):
        result = apply_loader_config_overrides(bert_config, {"hidden_size": 512})
        assert result is not bert_config

    def test_untouched_fields_preserved(self, bert_config):
        result = apply_loader_config_overrides(bert_config, {"hidden_size": 1024})
        # num_attention_heads was set on the original; should round-trip
        assert result.num_attention_heads == bert_config.num_attention_heads


class TestNestedOverride:
    """Recursive merge into nested :class:`PretrainedConfig` attributes."""

    def test_nested_dict_recurses_into_subconfig(self, clip_config):
        original_size = clip_config.vision_config.image_size
        result = apply_loader_config_overrides(
            clip_config, {"vision_config": {"image_size": original_size + 16}}
        )
        assert result.vision_config.image_size == original_size + 16

    def test_nested_override_preserves_sibling_fields(self, clip_config):
        original_hidden = clip_config.vision_config.hidden_size
        result = apply_loader_config_overrides(
            clip_config, {"vision_config": {"image_size": 320}}
        )
        # ``image_size`` patched; ``hidden_size`` of the same sub-config preserved
        assert result.vision_config.hidden_size == original_hidden

    def test_nested_override_does_not_mutate_original(self, clip_config):
        original_size = clip_config.vision_config.image_size
        _ = apply_loader_config_overrides(
            clip_config, {"vision_config": {"image_size": original_size + 16}}
        )
        assert clip_config.vision_config.image_size == original_size

    def test_top_level_and_nested_in_one_call(self, clip_config):
        result = apply_loader_config_overrides(
            clip_config,
            {
                "logit_scale_init_value": 3.0,
                "vision_config": {"image_size": 320},
            },
        )
        assert result.logit_scale_init_value == 3.0
        assert result.vision_config.image_size == 320


class TestESRGANUseCase:
    """End-to-end sanity check on the actual ESRGAN config (drives this feature)."""

    def test_scale_override_on_esrgan(self):
        # Triggers HF registrations that ESRGANConfig depends on.
        import winml.modelkit.models.hf  # noqa: F401
        from winml.modelkit.models.hf.esrgan import ESRGANConfig

        cfg = ESRGANConfig()
        assert cfg.scale == 4  # default

        result = apply_loader_config_overrides(cfg, {"scale": 2})
        assert result.scale == 2
        assert isinstance(result, ESRGANConfig)
        # Original untouched
        assert cfg.scale == 4
