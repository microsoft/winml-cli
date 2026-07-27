# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Data utilities for input preparation and padding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def pad_inputs(
    source: dict[str, Any],
    # Covariant Mapping/Sequence so existing dict[str, list[int]] callers still
    # fit, while modelling ONNX dynamic dims (None / str symbol) the body handles.
    expected: Mapping[str, Sequence[int | str | None]],
    mode: Literal["left", "right"] = "right",
) -> dict[str, Any]:
    """Filter *source* to keys in *expected* and fit tensors to static shapes.

    For each name in *expected*, if *source* has a tensor for it, pad any
    dimension smaller than the ONNX expected shape (skips batch dim). When
    Non-tensor values are passed through. Missing names are skipped.

    Args:
        source: Input tensors keyed by name.
        expected: ONNX expected shapes keyed by input name.
        mode: Padding side — ``"right"`` (default, pad at end) or
            ``"left"`` (pad at start).

    Returns:
        Filtered and padded tensors matching *expected* keys.
    """
    if mode not in ("right", "left"):
        raise ValueError(f"mode must be 'right' or 'left', got {mode!r}")

    result: dict[str, Any] = {}
    for name, expected_shape in expected.items():
        val = source.get(name)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            ndim = min(len(val.shape), len(expected_shape))
            # torch.nn.functional.pad takes pairs (low, high) from the LAST
            # dim backwards. Skip batch dim (dim 0).
            pad: list[int] = []
            for dim in reversed(range(1, ndim)):
                exp = expected_shape[dim]
                # Dynamic ONNX dims may be None or a string symbol; emit a
                # (0, 0) pair so later pairs stay aligned with their dim index.
                if not isinstance(exp, int):
                    pad.extend([0, 0])
                    continue
                deficit = max(exp - val.shape[dim], 0)
                if mode == "right":
                    pad.extend([0, deficit])
                else:  # left
                    pad.extend([deficit, 0])
            if any(p > 0 for p in pad):
                val = torch.nn.functional.pad(val, pad)
        result[name] = val
    return result
