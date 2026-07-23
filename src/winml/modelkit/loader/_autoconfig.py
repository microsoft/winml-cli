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
        architecture-specific subclass); falls back to a base
        ``PretrainedConfig`` only when the model omits ``model_type`` and
        AutoConfig would otherwise raise.
    """
    try:
        return cast(
            "PretrainedConfig",
            auto_config.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, **kwargs
            ),
        )
    except ValueError as auto_err:
        from transformers import PretrainedConfig

        try:
            return PretrainedConfig.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:
            # Not the model_type case (e.g. missing/invalid model id) — surface
            # the original, more informative AutoConfig error.
            raise auto_err from None
