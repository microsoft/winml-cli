# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Export-time attention compatibility helpers."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch.nn as nn


@contextlib.contextmanager
def use_eager_attention_for_export(model: nn.Module) -> Iterator[None]:
    """Temporarily prefer eager attention on HF-style module configs."""
    restored: list[tuple[object, object]] = []
    seen_configs: set[int] = set()

    for module in model.modules():
        config = getattr(module, "config", None)
        if config is None or id(config) in seen_configs:
            continue
        seen_configs.add(id(config))
        if not hasattr(config, "_attn_implementation"):
            continue

        current = config._attn_implementation
        if current in (None, "eager"):
            continue

        config._attn_implementation = "eager"
        restored.append((config, current))

    try:
        yield
    finally:
        for config, previous in reversed(restored):
            config._attn_implementation = previous
