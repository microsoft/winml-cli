# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Registry mapping ``model_type`` to its quantization policy.

Mirrors the project's other ``model_type``-keyed registries (e.g.
``COMPOSITE_MODEL_REGISTRY``): a finalizer registers itself with
``@register_quant_finalizer(model_type)`` and the build pipeline resolves it
with :func:`get_quant_finalizer`.

The registry is intentionally lazy. Importing :mod:`winml.modelkit.quant`
must stay free of heavy deps (torch/transformers); the per-model finalizer
modules — which do pull those in — are only imported the first time their
``model_type`` is actually quantized.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import QuantConfigFinalizer


# Populated by the ``@register_quant_finalizer`` decorator at import time.
_QUANT_FINALIZER_REGISTRY: dict[str, type[QuantConfigFinalizer]] = {}

# ``model_type`` -> submodule that defines (and self-registers) its finalizer.
# Looked up lazily so the heavy module loads only when needed. Keys must match
# the live ``model_type`` string verbatim (no ``_`` -> ``-`` normalization),
# since lookup is keyed on the exported model's ``config.model_type``.
_KNOWN_FINALIZER_MODULES: dict[str, str] = {
    "qwen3_transformer_only": ".qwen3_transformer_only",
}


def register_quant_finalizer(model_type: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`QuantConfigFinalizer` for ``model_type``."""

    def decorator(cls: type) -> type:
        if not hasattr(cls, "finalize"):
            raise TypeError(
                f"{cls.__name__} cannot register as a quant finalizer for "
                f"{model_type!r}: it must define a ``finalize`` method."
            )
        if model_type in _QUANT_FINALIZER_REGISTRY:
            raise ValueError(
                f"Quant finalizer already registered for {model_type!r}: "
                f"{_QUANT_FINALIZER_REGISTRY[model_type].__name__}. "
                f"Cannot register {cls.__name__}."
            )
        _QUANT_FINALIZER_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_quant_finalizer(model_type: str | None) -> QuantConfigFinalizer | None:
    """Return a finalizer instance for ``model_type``, or ``None`` if unregistered.

    ``None`` means "no model-specific policy" — the quantizer then uses its
    standard task-aware ``DatasetCalibrationReader``.
    """
    if not model_type:
        return None
    if model_type not in _QUANT_FINALIZER_REGISTRY and model_type in _KNOWN_FINALIZER_MODULES:
        # Triggers the module's ``@register_quant_finalizer`` side effect.
        importlib.import_module(_KNOWN_FINALIZER_MODULES[model_type], __package__)
    cls = _QUANT_FINALIZER_REGISTRY.get(model_type)
    return cls() if cls is not None else None
