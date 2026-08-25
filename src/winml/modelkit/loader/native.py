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

    from ..models.winml import HFCausalLM


@dataclass(frozen=True)
class NativeDevice:
    """Resolved PyTorch device and its WinML CLI name."""

    name: str
    torch_device: Any


@dataclass(frozen=True)
class NativeHFModel:
    """Loaded native Hugging Face model and its resolved device."""

    model: "nn.Module | HFCausalLM"
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
    torch_dtype: Any = "auto",
) -> NativeHFModel:
    """Load the task-resolved Hugging Face model without ONNX export."""
    resolved_device = resolve_native_device(device)
    if task == "text-generation":
        from ..models.winml import HFCausalLM

        causal_model = HFCausalLM(
            model_id,
            resolved_device.torch_device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        return NativeHFModel(model=causal_model, device=resolved_device)

    from .hf import load_hf_model

    model, _, _ = load_hf_model(
        model_id,
        task=task,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model = model.to(resolved_device.torch_device).eval()
    return NativeHFModel(
        model=model,
        device=resolved_device,
    )


def _adapt_native_hf_model(
    model_id: str,
    model: nn.Module,
    *,
    task: str | None,
    device: str,
    trust_remote_code: bool = False,
) -> nn.Module | HFCausalLM:
    """Prepare a caller-owned model for the selected evaluator contract."""
    if task == "text-generation":
        from ..models.winml import HFCausalLM

        return HFCausalLM.from_model(
            model_id,
            model,
            device,
            trust_remote_code=trust_remote_code,
        )

    return model.eval()
