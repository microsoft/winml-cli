# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Multi-input/output capture for nn.Module submodules.

Uses PyTorch forward hooks with with_kwargs=True to capture ALL input
and output tensors per module, including kwargs like attention_mask.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class ModuleIOInfo:
    """Captured I/O information for a single module."""

    class_name: str
    module_path: str
    input_shapes: list[list[int]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    output_dtypes: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


def _extract_tensors(obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursively extract tensors from nested structures.

    Handles: Tensor, tuple, list, dict, NamedTuple, dataclass (ModelOutput).
    """
    results: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        results.append((prefix or "tensor", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_extract_tensors(v, k))
    elif isinstance(obj, (tuple, list)):
        # Check for NamedTuple (has _fields)
        if hasattr(obj, "_fields"):
            for fname, val in zip(obj._fields, obj, strict=True):
                results.extend(_extract_tensors(val, fname))
        else:
            for i, val in enumerate(obj):
                label = f"{prefix}[{i}]" if prefix else str(i)
                results.extend(_extract_tensors(val, label))
    elif hasattr(obj, "__dataclass_fields__"):
        # HuggingFace ModelOutput (dataclass-like)
        for k in obj.__dataclass_fields__:
            v = getattr(obj, k, None)
            if v is not None:
                results.extend(_extract_tensors(v, k))
    return results


def capture_module_io(
    model: nn.Module,
    example_inputs: dict[str, torch.Tensor] | tuple,
    target_class: str | None = None,
) -> dict[str, ModuleIOInfo]:
    """Capture all input/output tensors for each module during a forward pass.

    Uses register_forward_hook(with_kwargs=True) to capture both positional
    args and keyword args (critical for HF modules that pass attention_mask etc.
    as kwargs).

    Args:
        model: PyTorch model to trace.
        example_inputs: Dummy inputs (dict for kwargs, tuple for positional).
        target_class: If set, only capture modules matching this class name.

    Returns:
        Dict mapping module_path -> ModuleIOInfo.
    """
    captured: dict[str, ModuleIOInfo] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for name, module in model.named_modules():
        if name == "":  # skip root
            continue
        class_name = type(module).__name__
        if target_class and class_name != target_class:
            continue

        # Get forward parameter names for input naming
        try:
            sig = inspect.signature(module.forward)
            param_names = [
                p.name for p in sig.parameters.values() if p.name != "self"
            ]
        except (ValueError, TypeError):
            param_names = []

        def make_hook(
            mod_name: str, cls_name: str, fwd_params: list[str],
        ):
            def hook(
                mod: nn.Module, args: tuple, kwargs: dict, output: Any,
            ) -> None:
                info = ModuleIOInfo(class_name=cls_name, module_path=mod_name)

                # Capture inputs from positional args
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor):
                        pname = fwd_params[i] if i < len(fwd_params) else f"arg_{i}"
                        info.input_shapes.append(list(a.shape))
                        info.input_dtypes.append(str(a.dtype).replace("torch.", ""))
                        info.input_names.append(pname)

                # Capture inputs from kwargs
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        info.input_shapes.append(list(v.shape))
                        info.input_dtypes.append(str(v.dtype).replace("torch.", ""))
                        info.input_names.append(k)

                # Capture outputs
                out_tensors = _extract_tensors(output)
                for oname, t in out_tensors:
                    info.output_shapes.append(list(t.shape))
                    info.output_dtypes.append(str(t.dtype).replace("torch.", ""))
                    info.output_names.append(oname)

                captured[mod_name] = info

            return hook

        h = module.register_forward_hook(
            make_hook(name, class_name, param_names),
            with_kwargs=True,
        )
        handles.append(h)

    # Run forward pass (with guaranteed hook cleanup)
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                try:
                    model(**example_inputs)
                except TypeError:
                    model(*example_inputs.values())
            else:
                model(*example_inputs)
    finally:
        for h in handles:
            h.remove()

    return captured
