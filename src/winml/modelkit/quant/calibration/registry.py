# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Registry mapping ``model_type`` to its quantization policy.

A model type with a fixed, reference-matched quant scheme (calibration reader,
``nodes_to_exclude``, dtypes) names its :class:`QuantConfigFinalizer` in the
plain ``QUANT_FINALIZERS`` dict below; the build pipeline resolves it with
:func:`get_quant_finalizer`. This mirrors the other ``model_type``-keyed tables
in the repo — a simple dict, no decorator/self-registration machinery.

The lookup is intentionally lazy. Importing :mod:`winml.modelkit.quant` must
stay free of heavy deps (torch/transformers); the per-model finalizer modules —
which do pull those in — are only imported the first time their ``model_type``
is actually quantized.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import QuantConfigFinalizer


# ``model_type`` -> ``(submodule, class name)`` of its QuantConfigFinalizer.
# Imported lazily by ``get_quant_finalizer`` so the heavy module loads only when
# needed. Keys must match the live ``model_type`` string verbatim (no ``_`` ->
# ``-`` normalization), since lookup is keyed on the exported model's
# ``config.model_type``.
QUANT_FINALIZERS: dict[str, tuple[str, str]] = {
    "qwen3_transformer_only": (".qwen3_transformer_only", "Qwen3TransformerOnlyQuantFinalizer"),
}


def get_quant_finalizer(model_type: str | None) -> QuantConfigFinalizer | None:
    """Return a finalizer instance for ``model_type``, or ``None`` if unregistered.

    ``None`` means "no model-specific policy" — the quantizer then uses its
    standard task-aware ``DatasetCalibrationReader``.
    """
    if not model_type:
        return None
    entry = QUANT_FINALIZERS.get(model_type)
    if entry is None:
        return None
    module_name, class_name = entry
    module = importlib.import_module(module_name, __package__)
    finalizer_cls = getattr(module, class_name)
    finalizer: QuantConfigFinalizer = finalizer_cls()
    return finalizer
