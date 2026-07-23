# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HF config loading with transformers-4-style tolerance for model_type-less configs.

transformers>=5 dropped the lenient fallback that let ``AutoConfig.from_pretrained``
load a config lacking a ``model_type`` key (older Hub models such as
``prajjwal1/bert-tiny``); it now raises ``ValueError: Unrecognized model ...``.
transformers 4 returned a base :class:`~transformers.PretrainedConfig` in that
case. :func:`load_hf_config` restores that behavior so such models stay loadable.

There is no architecture-specific inference here — a generic ``PretrainedConfig``
simply carries the raw config fields, and downstream resolution keys off the
task / model_class rather than ``model_type`` for these inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from transformers import PretrainedConfig


def load_hf_config(
    auto_config: Any,
    model_id: str,
    *,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> PretrainedConfig:
    """Load an HF config, tolerating configs that omit a ``model_type`` key.

    Args:
        auto_config: The caller's own ``AutoConfig`` reference (its module-level
            name). Passing it in — rather than importing ``AutoConfig`` here —
            keeps each call site's ``AutoConfig`` monkeypatchable in tests.
        model_id: HuggingFace model ID or local path.
        trust_remote_code: Forwarded to the transformers loaders.
        **kwargs: Additional keyword arguments forwarded verbatim (e.g.
            ``revision``).

    Returns:
        The resolved config. Prefers ``auto_config.from_pretrained`` (the
        architecture-specific subclass); falls back to an identifier-inferred
        concrete config only when the model omits ``model_type`` and
        AutoConfig would otherwise raise.
    """
    try:
        return cast(
            "PretrainedConfig",
            auto_config.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs),
        )
    except ValueError as auto_err:
        if "model_type" not in str(auto_err):
            raise

        from transformers import PretrainedConfig

        try:
            config_dict, unused_kwargs = PretrainedConfig.get_config_dict(
                model_id, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:
            raise auto_err from None

        if "model_type" in config_dict:
            raise

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        model_id_lower = model_id.lower()
        candidates = sorted(
            (name for name in CONFIG_MAPPING if name.lower() in model_id_lower),
            key=lambda name: (-len(name), name),
        )
        if not candidates:
            raise

        return cast(
            "PretrainedConfig",
            CONFIG_MAPPING[candidates[0]].from_dict(config_dict, **unused_kwargs),
        )
