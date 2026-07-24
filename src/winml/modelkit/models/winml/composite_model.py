# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML composite model base and registry.

Provides ``WinMLCompositeModel`` — a base class for models composed of
multiple ``WinMLAutoModel`` sub-components (e.g., encoder + decoder,
prefill + gen).  Each subclass declares ``_SUB_MODEL_CONFIG`` mapping
component names to HF tasks; ``from_pretrained()`` builds them all.

Registry
--------
``@register_composite_model(model_type, task)`` registers a pipeline class.
``winml config`` checks the registry to generate one config file per component::

    winml config -m google-t5/t5-small --task translation -o t5.json
    # → t5_encoder.json (feature-extraction) + t5_decoder.json (text2text-generation)

    winml build -c t5_encoder.json -m google-t5/t5-small -o output/encoder
    winml build -c t5_decoder.json -m google-t5/t5-small -o output/decoder

Per-component kwargs
--------------------
``sub_model_kwargs`` in ``from_pretrained`` allows different ``shape_config``
per sub-model (e.g., different ``max_cache_len`` for prefill vs gen)::

    WinMLCompositeModel.from_pretrained(model_id, task="text-generation",
        sub_model_kwargs={
            "decoder_prefill": {"shape_config": {"max_cache_len": 256, "seq_len": 64}},
            "decoder_gen":     {"shape_config": {"max_cache_len": 256, "seq_len": 1}},
        })

Concrete composite models live alongside their export configs:

- ``models.hf.t5.WinMLT5Model`` (encoder-decoder, T5)
- ``models.hf.mu2.WinMLMu2Model`` (encoder-decoder, Mu2)
- ``models.hf.qwen.WinMLQwen3Model`` (decoder-only, Qwen3)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from .base import PreTrainedModel


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from transformers import PretrainedConfig

    from ...session import WinMLEPDevice

logger = logging.getLogger(__name__)


# =========================================================================
# composite model Registry
# =========================================================================

# Maps (model_type, task) → pipeline class with _SUB_MODEL_CONFIG.
# Used by `winml config` to generate one config file per sub-component.
COMPOSITE_MODEL_REGISTRY: dict[tuple[str, str], type[WinMLCompositeModel]] = {}


def register_composite_model(model_type: str, task: str) -> Callable[[type], type]:
    """Class decorator that registers a composite model for `winml config`."""

    def decorator(cls: type) -> type:
        if not issubclass(cls, WinMLCompositeModel):
            raise TypeError(
                f"{cls.__name__} cannot register as a composite model for "
                f"{(model_type, task)!r}: it must subclass WinMLCompositeModel. "
                f"This invariant lets every registry consumer trust the registry "
                f"without re-filtering by type."
            )
        key = (model_type, task)
        if key in COMPOSITE_MODEL_REGISTRY:
            raise ValueError(
                f"Composite model already registered for {key!r}: "
                f"{COMPOSITE_MODEL_REGISTRY[key].__name__}. "
                f"Cannot register {cls.__name__}."
            )
        COMPOSITE_MODEL_REGISTRY[key] = cls
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
        device: str = "cpu",
    ) -> None:
        self.sub_models = sub_models
        self.config = config
        self._device = device

    def get_experts_implementation(self) -> dict[str, None]:
        """Report no switchable PyTorch experts for immutable ORT graphs."""
        return {"": None}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: str,
        *,
        device: str = "cpu",
        ep_device: WinMLEPDevice | None = None,
        use_cache: bool = True,
        force_rebuild: bool = False,
        sub_model_kwargs: dict[str, dict[str, Any]] | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> WinMLCompositeModel:
        """Build all sub-components and return ready-to-use model.

        When called on ``WinMLCompositeModel`` directly (not a subclass),
        ``task`` is required to resolve the concrete class from
        ``COMPOSITE_MODEL_REGISTRY``.  When called on a registered subclass
        (e.g., ``WinMLT5Model``), ``task`` is optional.

        Args:
            model_id: HuggingFace model ID or local path.
            task: Pipeline task name (e.g., ``"translation"``,
                ``"text-generation"``). Required when calling on the base
                class; ignored when calling on a registered subclass.
            device: Target device short name (e.g. ``"npu"``, ``"cpu"``).
                Forwarded to ``__init__`` so ``self._device`` reflects the
                caller's intent.
            ep_device: Optional pre-resolved ``WinMLEPDevice`` handle. When
                ``None``, derived from ``device`` via
                ``resolve_device`` + ``WinMLEPRegistry.auto_device`` so the
                sub-model call always receives one.
            use_cache: Use persistent cache.
            force_rebuild: Force rebuild even if cached.
            sub_model_kwargs: Per-component kwargs forwarded to
                ``WinMLAutoModel.from_pretrained()``.  Keys are component
                names from ``_SUB_MODEL_CONFIG`` (e.g., ``"decoder_prefill"``,
                ``"decoder_gen"``).  Values are dicts merged on top of the
                shared ``**kwargs``.  Use this to pass different
                ``shape_config`` per sub-model.
            trust_remote_code: Forward to ``AutoConfig.from_pretrained``
                and each sub-model's ``WinMLAutoModel.from_pretrained``.
                Required for custom-code HF models (e.g., Mu2).
            **kwargs: Forwarded to ``WinMLAutoModel.from_pretrained()``
                for every sub-component (overridden by ``sub_model_kwargs``).
        """
        from transformers import AutoConfig

        if trust_remote_code:
            from ...utils.cli import warn_trust_remote_code

            warn_trust_remote_code()

        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_type = hf_config.model_type

        if not cls._SUB_MODEL_CONFIG:
            # Resolve concrete class from registry when called on the base class
            resolved_cls = COMPOSITE_MODEL_REGISTRY.get((model_type, task))
            if resolved_cls is None:
                raise ValueError(
                    f"No composite model registered for ({model_type!r}, {task!r}). "
                    f"Registered: {list(COMPOSITE_MODEL_REGISTRY.keys())}"
                )
            return resolved_cls.from_pretrained(
                model_id,
                task,
                device=device,
                ep_device=ep_device,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                sub_model_kwargs=sub_model_kwargs,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        from ..auto import WinMLAutoModel

        # Sub-model API requires a WinMLEPDevice — derive one from the
        # device short name when the caller didn't hand one in.
        if ep_device is None:
            from ...session import EPDeviceTarget, WinMLEPRegistry, resolve_device

            target = resolve_device(EPDeviceTarget(ep="auto", device=device))
            ep_device = WinMLEPRegistry.instance().auto_device(target)

        per_component = sub_model_kwargs or {}
        sub_models: dict[str, Any] = {}
        for name, component_task in cls._SUB_MODEL_CONFIG.items():
            logger.info("Building %s for %s...", name, model_id)
            merged = {**kwargs, **per_component.get(name, {})}
            sub_models[name] = WinMLAutoModel.from_pretrained(
                model_id,
                ep_device=ep_device,
                task=component_task,
                use_cache=use_cache,
                force_rebuild=force_rebuild,
                trust_remote_code=trust_remote_code,
                **merged,
            )

        return cls(sub_models=sub_models, config=hf_config, device=device)

    @classmethod
    def from_onnx(
        cls,
        # Mapping (not dict) so dict[str, str] from configs is accepted
        # without a cast — dict is invariant on value type, Mapping is covariant.
        onnx_path: Mapping[str, str | Path],
        *,
        task: str | None = None,
        hf_config: PretrainedConfig | None = None,
        sub_model_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> WinMLCompositeModel:
        """Load composite model from pre-built ONNX files.

        Resolves the concrete model class from the registry using *task*
        and ``hf_config.model_type``, then builds each sub-component via
        ``WinMLAutoModel.from_onnx``.

        Args:
            onnx_path: Maps component name (e.g., ``"encoder"``,
                ``"decoder_prefill"``) to its ONNX file path. Values may
                be ``str`` or ``pathlib.Path``; coerced via ``Path(path)``
                inside the dispatch loop.
            task: Pipeline task (e.g., ``"translation"``,
                ``"text-generation"``).
            hf_config: HF ``PretrainedConfig`` for the model. Used to
                resolve the concrete class from the registry via
                ``hf_config.model_type``.
            sub_model_kwargs: Per-component kwargs merged on top of
                ``**kwargs`` for each sub-model's ``from_onnx`` call.
            **kwargs: Forwarded to ``WinMLAutoModel.from_onnx`` for every
                component (overridden by ``sub_model_kwargs``). A resolved
                ``ep_device`` also determines the composite wrapper's device.
        """
        from pathlib import Path

        per_component = sub_model_kwargs or {}

        # Resolve concrete class from registry
        model_type = getattr(hf_config, "model_type", None) if hf_config else None
        if not cls._SUB_MODEL_CONFIG:
            # model_type/task may be None; the str-keyed registry simply misses
            # (returns None, handled below). dict.get tolerates any hashable key.
            resolved_cls = COMPOSITE_MODEL_REGISTRY.get(cast("tuple[str, str]", (model_type, task)))
            if resolved_cls is None:
                raise ValueError(
                    f"No composite model for ({model_type!r}, {task!r}). "
                    f"Registered: {list(COMPOSITE_MODEL_REGISTRY.keys())}"
                )
        else:
            resolved_cls = cls

        from ..auto import WinMLAutoModel

        sub_models: dict[str, Any] = {}
        for name, path in onnx_path.items():
            component_task = resolved_cls._SUB_MODEL_CONFIG.get(name)
            if component_task is None:
                valid = list(resolved_cls._SUB_MODEL_CONFIG.keys())
                raise ValueError(
                    f"Unknown component {name!r}. Valid names for {resolved_cls.__name__}: {valid}"
                )
            merged = {**kwargs, "task": component_task, **per_component.get(name, {})}
            sub_models[name] = WinMLAutoModel.from_onnx(Path(path), **merged)

        if hf_config is None:
            raise ValueError("Composite model construction requires an HF config (hf_config).")
        ep_device = kwargs.get("ep_device")
        device = (
            ep_device.device.device_type.lower()
            if ep_device is not None
            else str(kwargs.get("device") or "cpu").lower()
        )
        return resolved_cls(sub_models=sub_models, config=hf_config, device=device)

    @property
    def device(self) -> torch.device:
        """Device (CPU — ORT handles actual placement)."""
        return torch.device("cpu")

    @property
    def ort_device(self) -> str:
        """ORT execution provider target (e.g. "npu", "gpu", "cpu", "auto")."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype for HF compatibility."""
        return torch.float32

    def to(self, *args: Any, **kwargs: Any) -> WinMLCompositeModel:
        """No-op for HF pipeline compatibility; sub-models remain on their original device."""
        if args or kwargs:
            # debug (not warning) — HF pipelines routinely call `.to("cpu")` as a
            # setup step; surfacing that as a warning would spam normal usage.
            logger.debug(
                "WinMLCompositeModel.to(...) is a no-op; sub-models remain on their original "
                "device. Use WinMLSession options to control device placement."
            )
        return self

    def __call__(self, **kwargs: Any) -> Any:
        """Inference entry point."""
        return self.forward(**kwargs)

    def forward(self, **kwargs: Any) -> Any:
        """Subclasses implement task-specific logic."""
        raise NotImplementedError
