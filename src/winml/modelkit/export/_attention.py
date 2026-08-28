# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utilities for controlling Transformers attention during export."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch.nn as nn


@contextlib.contextmanager
def use_eager_attention_for_export(model: nn.Module) -> Iterator[None]:
    """Temporarily prefer eager attention on HF-style module configs."""
    restored: list[tuple[int, Any, Any]] = []
    configs: dict[int, Any] = {}
    children: dict[int, set[int]] = {}
    seen_configs: set[int] = set()

    for module in model.modules():
        _collect_attention_configs(getattr(module, "config", None), configs, children, seen_configs)

    for config_id, config in configs.items():
        current = config._attn_implementation
        restored.append((config_id, config, current))

    for config in configs.values():
        if config._attn_implementation != "eager":
            config._attn_implementation = "eager"

    try:
        yield
    finally:
        for _config_id, config, previous in _parent_before_child(restored, children):
            config._attn_implementation = previous


def _collect_attention_configs(
    config: Any,
    configs: dict[int, Any],
    children: dict[int, set[int]],
    seen_configs: set[int],
) -> None:
    if config is None or id(config) in seen_configs:
        return
    seen_configs.add(id(config))

    config = cast("Any", config)
    config_id = id(config)
    if hasattr(config, "_attn_implementation"):
        configs[config_id] = config

    for child_config in _iter_sub_configs(config):
        if hasattr(config, "_attn_implementation") and hasattr(
            child_config, "_attn_implementation"
        ):
            children.setdefault(config_id, set()).add(id(child_config))
        _collect_attention_configs(child_config, configs, children, seen_configs)


def _iter_sub_configs(config: Any) -> Iterator[Any]:
    sub_configs = getattr(config, "sub_configs", None)
    if isinstance(sub_configs, dict):
        for key, value in sub_configs.items():
            if isinstance(key, str):
                child = getattr(config, key, None)
                if child is not None:
                    yield child
            if not isinstance(value, type):
                yield value
    elif isinstance(sub_configs, (list, tuple, set)):
        yield from sub_configs


def _parent_before_child(
    restored: list[tuple[int, Any, Any]],
    children: dict[int, set[int]],
) -> list[tuple[int, Any, Any]]:
    parents: dict[int, set[int]] = {}
    restored_ids = {config_id for config_id, _config, _previous in restored}
    for parent_id, child_ids in children.items():
        if parent_id not in restored_ids:
            continue
        for child_id in child_ids:
            if child_id in restored_ids:
                parents.setdefault(child_id, set()).add(parent_id)

    depths: dict[int, int] = {}

    def depth(config_id: int, visiting: set[int]) -> int:
        if config_id in depths:
            return depths[config_id]
        if config_id in visiting:
            return 0
        visiting.add(config_id)
        config_depth = 0
        if config_id in parents:
            config_depth = 1 + max(depth(parent_id, visiting) for parent_id in parents[config_id])
        visiting.remove(config_id)
        depths[config_id] = config_depth
        return config_depth

    return sorted(restored, key=lambda item: depth(item[0], set()))
