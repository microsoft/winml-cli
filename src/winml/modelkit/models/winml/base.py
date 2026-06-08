# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base Model Class for WinML Inference.

Provides WinMLPreTrainedModel - minimal base class for HF pipeline compatibility.

HF pipeline uses duck typing, NOT isinstance checks. Required interface:
- forward() / __call__() - inference
- to() - device placement
- device property
- dtype property
- config attribute

Design: Leverages WinMLSession only for ORT operations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch


if TYPE_CHECKING:
    import contextlib

from ...session.session import WinMLSession


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...utils.constants import EPName, EPNameOrAlias

logger = logging.getLogger(__name__)


class PreTrainedModel:
    """Name shim so HF ``infer_framework()`` recognizes WinML models as "pt".

    HF's ``transformers.utils.generic.infer_framework`` walks the MRO looking
    for a class **named** ``PreTrainedModel``.  By giving our abstract base
    this exact name, ``pipeline(task, model=winml_model)`` works without
    requiring ``torch.nn.Module`` inheritance.
    """


class WinMLPreTrainedModel(PreTrainedModel, ABC):
    """Base class for WinML inference models.

    Minimal interface for HuggingFace pipeline compatibility.
    Delegates all ORT operations to WinMLSession.
    Does NOT inherit from nn.Module (not required - HF uses duck typing).

    Subclasses implement forward() for task-specific inference.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        config: PretrainedConfig | None = None,
        device: str = "auto",
        session_options: Any | None = None,
        ep: EPNameOrAlias | None = None,
    ) -> None:
        """Initialize inference model.

        Args:
            onnx_path: Path to ONNX model file
            config: HuggingFace PretrainedConfig (num_labels, id2label, etc.)
            device: Target device ("auto", "npu", "gpu", "cpu")
            session_options: Factory returning an ORT SessionOptions (e.g., for
                graph_optimization_level). Called fresh per ORT session.
            ep: Explicit EP short name (e.g., "dml", "qnn"). Forwarded to WinMLSession.
        """
        self._onnx_path = Path(onnx_path)
        self.config = config
        self._device = device

        # Set by WinMLAutoModel.from_pretrained() after construction
        self._build_config: Any = None

        # Create WinMLSession (delegates ORT operations)
        self._session = WinMLSession(
            onnx_path=self._onnx_path,
            device=device,
            session_options=session_options,
            ep=ep,
        )

    @property
    def io_config(self) -> dict:
        """ONNX I/O metadata (delegated to session)."""
        return self._session.io_config

    @property
    def onnx_path(self) -> Path:
        """Path to the ONNX model file."""
        return self._onnx_path

    def _format_inputs(
        self,
        data: torch.Tensor | np.ndarray | list | dict | None = None,
        **kwargs: torch.Tensor | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Normalize inputs to dict and convert torch tensors to numpy.

        Accepts multiple input formats:
            - Single tensor: zipped with first input name
            - List of tensors: zipped with input names in order
            - Dict: used directly
            - Kwargs: used directly

        Note:
            - Validation (missing inputs) is done by WinMLSession._validate_inputs()
            - Dtype enforcement is done by WinMLSession._prepare_inputs()

        Args:
            data: Input data (tensor, list, or dict)
            **kwargs: Named input tensors (alternative to data)

        Returns:
            Dict of input_name -> numpy array

        Raises:
            TypeError: If data format is not supported
        """
        input_names = self.io_config["input_names"]

        # Normalize to dict - handle all input formats
        if data is not None:
            if isinstance(data, list):
                # List of tensors -> zip with input names
                data = dict(zip(input_names, data, strict=True))
            elif isinstance(data, torch.Tensor | np.ndarray):
                # Single tensor -> first input name
                data = {input_names[0]: data}
            elif not isinstance(data, dict):
                raise TypeError(f"Expected tensor, list, or dict, got {type(data)}")
            # else: already dict, use as-is
        else:
            data = kwargs

        # Convert torch tensors to numpy
        # (validation + dtype enforcement done by WinMLSession)
        inputs = {}
        for name, value in data.items():
            if isinstance(value, torch.Tensor):
                inputs[name] = value.numpy()
            elif isinstance(value, np.ndarray):
                inputs[name] = value
            else:
                inputs[name] = np.asarray(value)

        return inputs

    def _run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Run inference via WinMLSession.

        WinMLSession handles:
            - Auto-compilation on first run
            - Re-batching for static batch models
            - Dtype enforcement

        Args:
            inputs: Formatted input dict from _format_inputs()

        Returns:
            Dict of output_name -> torch.Tensor
        """
        # Delegate to WinMLSession (handles compile, re-batching, run)
        outputs = self._session.run(inputs)
        return {name: torch.from_numpy(out) for name, out in outputs.items()}

    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
        """Forward pass - subclasses implement task-specific logic."""

    def __call__(self, **kwargs: Any) -> Any:
        """Inference entry point."""
        return self.forward(**kwargs)

    def to(self, *args: Any, **kwargs: Any) -> WinMLPreTrainedModel:
        """No-op for HF pipeline compatibility.

        FIXME: HF pipeline calls model.to(torch.device(...)) to move the model.
        WinML models are ORT-backed — device placement is handled by the EP
        policy set at session creation, not by moving tensors.  We ignore
        .to() calls so the pipeline doesn't break compiled EPContext models
        by trying to recreate the session on CPU.
        """
        return self

    def perf(self, warmup: int = 0) -> contextlib.AbstractContextManager:
        """Context manager for scoped performance tracking.

        Delegates to the underlying WinMLSession.perf(). Every inference
        call within the context records timing in PerfStats.

        Args:
            warmup: Number of initial samples to exclude from statistics.

        Example::

            with model.perf(warmup=5) as stats:
                for img in images:
                    model(pixel_values=img)
            print(f"P99: {stats.p99_ms:.2f} ms")
        """
        return self._session.perf(warmup=warmup)

    @property
    def device(self) -> str:
        """Current device (delegates to session, resolved after compile)."""
        return self._session.device

    @property
    def ep_name(self) -> EPName | None:
        """Primary EP bound by ORT (delegates to session, None before compile)."""
        return self._session.ep_name

    @property
    def task(self) -> str | None:
        """Resolved task from build config, or None if unavailable."""
        build_config = getattr(self, "_build_config", None)
        if build_config is not None:
            loader = getattr(build_config, "loader", None)
            if loader:
                return loader.task
        return None

    @property
    def precision(self) -> str | None:
        """Resolved precision from build config, or None if unavailable.

        TODO: derive from _build_config.quant.weight_type when ready.
        """
        return None

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype (for HF compatibility)."""
        return torch.float32


class WinMLModelForGenericTask(WinMLPreTrainedModel):
    """Generic fallback for unknown/unsupported tasks.

    Returns raw ONNX outputs without task-specific wrapping.
    Useful for:
    - Benchmarking models with unsupported tasks
    - Prototyping before implementing task-specific wrappers
    - Running inference when output format doesn't matter

    Example:
        >>> model = WinMLModelForGenericTask("model.onnx")
        >>> outputs = model(input_ids=tokens, attention_mask=mask)
        >>> # outputs is dict[str, torch.Tensor] with raw ONNX output names
    """

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Generic inference - returns raw ONNX outputs.

        Args:
            **kwargs: Input tensors matching ONNX model inputs

        Returns:
            Dict mapping output names to torch.Tensor
        """
        formatted = self._format_inputs(**kwargs)
        return self._run_inference(formatted)
