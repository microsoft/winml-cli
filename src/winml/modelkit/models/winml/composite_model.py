# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Pipeline Model base and registry.

Provides ``WinMLCompositeModel`` — a base class for models composed of
multiple ``WinMLAutoModel`` sub-components (e.g., encoder + decoder,
prefill + gen).  Each subclass declares ``_SUB_MODEL_CONFIG`` mapping
component names to HF tasks; ``from_pretrained()`` builds them all.

Registry
--------
``@register_composite_model(model_type, task)`` registers a pipeline class.
``wmk config`` checks the registry to generate one config file per component::

    wmk config -m google-t5/t5-small --task translation -o t5.json
    # → t5_encoder.json (feature-extraction) + t5_decoder.json (text2text-generation)

    wmk build -c t5_encoder.json -m google-t5/t5-small -o output/encoder
    wmk build -c t5_decoder.json -m google-t5/t5-small -o output/decoder

Per-component kwargs
--------------------
``sub_model_kwargs`` in ``from_pretrained`` allows different ``shape_config``
per sub-model (e.g., different ``max_cache_len`` for prefill vs gen)::

    WinMLCompositeModel.from_pretrained(model_id, task="text-generation",
        sub_model_kwargs={
            "decoder_prefill": {"shape_config": {"max_cache_len": 256, "seq_len": 64}},
            "decoder_gen":     {"shape_config": {"max_cache_len": 256, "seq_len": 1}},
        })

Concrete pipeline models live alongside their export configs:

- ``models.hf.t5.WinMLT5Model`` (encoder-decoder, T5)
- ``models.hf.mu2.WinMLMu2Model`` (encoder-decoder, Mu2)
- ``models.hf.qwen.WinMLQwen3Model`` (decoder-only, Qwen3)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from .base import PreTrainedModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


# =========================================================================
# Pipeline Model Registry
# =========================================================================

# Maps (model_type, task) → pipeline class with _SUB_MODEL_CONFIG.
# Used by `wmk config` to generate one config file per sub-component.
PIPELINE_MODEL_REGISTRY: dict[tuple[str, str], type] = {}


def register_composite_model(model_type: str, task: str):
    """Class decorator that registers a pipeline model for `wmk config`."""

    def decorator(cls: type) -> type:
        PIPELINE_MODEL_REGISTRY[(model_type, task)] = cls
        return cls

    return decorator


# =========================================================================
# WinMLCompositeModel — multi-component base
# =========================================================================


class WinMLCompositeModel(PreTrainedModel):
    """Base class for models composed of multiple WinMLAutoModel sub-components.

    Subclasses declare ``_SUB_MODEL_CONFIG``: a mapping of component name to
    the HF task used to build it via ``WinMLAutoModel.from_pretrained``.

    After construction, sub-components are available in ``self.sub_models``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
    ) -> None:
        self.sub_models = sub_models
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: str,
        *,
        device: str = "cpu",
        use_cache: bool = True,
        force_rebuild: bool = False,
        sub_model_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> WinMLCompositeModel:
        """Build all sub-components and return ready-to-use model.

        When called on ``WinMLCompositeModel`` directly (not a subclass),
        ``task`` is required to resolve the concrete class from
        ``PIPELINE_MODEL_REGISTRY``.  When called on a registered subclass
        (e.g., ``WinMLT5Model``), ``task`` is optional.

        Args:
            model_id: HuggingFace model ID or local path.
            task: Pipeline task name (e.g., ``"translation"``,
                ``"text-generation"``). Required when calling on the base
                class; ignored when calling on a registered subclass.
            device: Target device.
            use_cache: Use persistent cache.
            force_rebuild: Force rebuild even if cached.
            sub_model_kwargs: Per-component kwargs forwarded to
                ``WinMLAutoModel.from_pretrained()``.  Keys are component
                names from ``_SUB_MODEL_CONFIG`` (e.g., ``"decoder_prefill"``,
                ``"decoder_gen"``).  Values are dicts merged on top of the
                shared ``**kwargs``.  Use this to pass different
                ``shape_config`` per sub-model.
            **kwargs: Forwarded to ``WinMLAutoModel.from_pretrained()``
                for every sub-component (overridden by ``sub_model_kwargs``).
        """
        from transformers import AutoConfig

        trust_remote_code = kwargs.get("trust_remote_code", False)
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_type = hf_config.model_type

        if not cls._SUB_MODEL_CONFIG:
            # Resolve concrete class from registry when called on the base class
            resolved_cls = PIPELINE_MODEL_REGISTRY.get((model_type, task))
            if resolved_cls is None:
                raise ValueError(
                    f"No pipeline model registered for ({model_type!r}, {task!r}). "
                    f"Registered: {list(PIPELINE_MODEL_REGISTRY.keys())}"
                )
            return resolved_cls.from_pretrained(
                model_id,
                task,
                device=device,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                sub_model_kwargs=sub_model_kwargs,
                **kwargs,
            )
        from ..auto import WinMLAutoModel

        per_component = sub_model_kwargs or {}
        sub_models: dict[str, Any] = {}
        for name, component_task in cls._SUB_MODEL_CONFIG.items():
            logger.info("Building %s for %s...", name, model_id)
            merged = {**kwargs, **per_component.get(name, {})}
            sub_models[name] = WinMLAutoModel.from_pretrained(
                model_id,
                task=component_task,
                device=device,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                **merged,
            )

        return cls(sub_models=sub_models, config=hf_config)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: dict[str, str],
        *,
        task: str | None = None,
        **kwargs: Any,
    ) -> WinMLCompositeModel:
        """Load composite model from pre-built ONNX files.

        Resolves the concrete model class from the registry using *task*
        and ``hf_config.model_type``, then builds each sub-component via
        ``WinMLAutoModel.from_onnx``.

        Args:
            onnx_path: Maps component name (e.g., ``"encoder"``,
                ``"decoder_prefill"``) to its ONNX file path.
            task: Pipeline task (e.g., ``"translation"``,
                ``"text-generation"``).
            **kwargs: Must include ``hf_config`` (``PretrainedConfig``).
                May include ``sub_model_kwargs`` for per-component
                overrides.  Remaining kwargs are forwarded to
                ``WinMLAutoModel.from_onnx`` for every component.
        """
        from pathlib import Path

        hf_config = kwargs.pop("hf_config", None)
        sub_model_kwargs = kwargs.pop("sub_model_kwargs", None) or {}

        # Resolve concrete class from registry
        model_type = getattr(hf_config, "model_type", None) if hf_config else None
        if not cls._SUB_MODEL_CONFIG:
            resolved_cls = PIPELINE_MODEL_REGISTRY.get((model_type, task))
            if resolved_cls is None:
                raise ValueError(
                    f"No composite model for ({model_type!r}, {task!r}). "
                    f"Registered: {list(PIPELINE_MODEL_REGISTRY.keys())}"
                )
        else:
            resolved_cls = cls

        from ..auto import WinMLAutoModel

        sub_models: dict[str, Any] = {}
        for name, path in onnx_path.items():
            component_task = resolved_cls._SUB_MODEL_CONFIG.get(name)
            merged = {**kwargs, "task": component_task, **sub_model_kwargs.get(name, {})}
            sub_models[name] = WinMLAutoModel.from_onnx(Path(path), **merged)

        return resolved_cls(sub_models=sub_models, config=hf_config)

    @property
    def device(self) -> torch.device:
        """Device (CPU — ORT handles actual placement)."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype for HF compatibility."""
        return torch.float32

    def to(self, *args: Any, **kwargs: Any) -> WinMLCompositeModel:
        """No-op for HF pipeline compatibility."""
        return self

    def __call__(self, **kwargs: Any) -> Any:
        """Inference entry point."""
        return self.forward(**kwargs)

    def forward(self, **kwargs: Any) -> Any:
        """Subclasses implement task-specific logic."""
        raise NotImplementedError

    @staticmethod
    def _pad_inputs(
        source: dict[str, Any],
        expected: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Filter *source* to keys in *expected* and pad undersized tensors.

        For each name in *expected*, if *source* has a tensor for it, pad
        any dimension smaller than the ONNX expected shape (skips batch dim).
        Non-tensor values are passed through. Missing names are skipped.
        """
        result: dict[str, Any] = {}
        for name, expected_shape in expected.items():
            val = source.get(name)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                # TODO: support dynamic shape ONNX models (None in expected_shape)
                ndim = min(len(val.shape), len(expected_shape))
                pad: list[int] = []
                for dim in reversed(range(1, ndim)):
                    deficit = expected_shape[dim] - val.shape[dim]
                    pad.extend([0, max(deficit, 0)])
                if any(p > 0 for p in pad):
                    val = torch.nn.functional.pad(val, pad)
            result[name] = val
        return result
