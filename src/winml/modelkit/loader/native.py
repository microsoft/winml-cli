# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Native Hugging Face PyTorch loading and device placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class NativeDevice:
    """Resolved PyTorch device and its WinML CLI name."""

    name: str
    torch_device: Any


@dataclass(frozen=True)
class NativeHFModel:
    """Loaded native Hugging Face model and resolved runtime metadata."""

    model: "nn.Module"
    config: Any
    task: str
    device: NativeDevice


def resolve_native_device(device: str) -> NativeDevice:
    """Map a WinML device name to a PyTorch CPU or CUDA device."""
    import torch

    requested = device.lower()
    if requested == "auto":
        requested = "gpu" if torch.cuda.is_available() else "cpu"
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise ValueError(
                "--device gpu with --runtime pytorch requires a CUDA-enabled PyTorch "
                "installation and an available CUDA device."
            )
        return NativeDevice(name="gpu", torch_device=torch.device("cuda"))
    if requested == "cpu":
        return NativeDevice(name="cpu", torch_device=torch.device("cpu"))
    raise ValueError(
        f"--device {device} is not supported with --runtime pytorch; use auto, cpu, or gpu."
    )


def load_native_hf_model(
    model_id: str,
    *,
    task: str | None = None,
    device: str = "auto",
    trust_remote_code: bool = False,
) -> NativeHFModel:
    """Load a checkpoint-declared Hugging Face class without ONNX export."""
    from .hf import load_hf_model

    resolved_device = resolve_native_device(device)
    model, hf_config, resolved_task = load_hf_model(
        model_id,
        task=task,
        trust_remote_code=trust_remote_code,
        use_checkpoint_class=True,
        torch_dtype="auto",
    )
    model = model.to(resolved_device.torch_device).eval()
    return NativeHFModel(
        model=model,
        config=hf_config,
        task=resolved_task,
        device=resolved_device,
    )
