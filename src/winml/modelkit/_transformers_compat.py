# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Re-inject symbols removed in transformers 5.x that optimum-onnx 0.1.0 imports.

optimum-onnx 0.1.0 (last PyPI release as of 2026-04-30) hardcodes imports
against transformers 4.x internals. This module re-injects those symbols so
optimum-onnx imports succeed. Importing this module has side effects; the
public package __init__ does so before any optimum.* import cascade.

Implementation note: transformers 5.x's top-level package is a `_LazyModule`,
and `from transformers import <SomeClass>` triggers `_LazyModule.__getattr__`,
which can replace `sys.modules["transformers"]` with a fresh `_LazyModule`
instance as a side effect. Therefore we (1) perform every required
`from transformers import …` upfront so all replacements settle, then (2)
capture the live module's `_objects` dict and inject the missing symbols.
`_LazyModule.__getattr__` consults `_objects` first, so this is the durable
injection point — `setattr(transformers, name, value)` would write to a
now-orphaned `__dict__` if a replacement happened later.

Drop this file (and the corresponding override in pyproject.toml) once
optimum-onnx 0.2+ ships with transformers 5.x compatibility.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import transformers
import transformers.modeling_utils
import transformers.utils
import transformers.utils.generic

# Pre-load any successor classes we need; each `from transformers import …`
# may swap sys.modules["transformers"]. After these calls all replacements
# have settled.
from transformers import (
    AutoModelForImageTextToText,
    CLIPImageProcessor,
)


# Capture the live module + _objects dict AFTER all replacements have happened.
_t = sys.modules["transformers"]
_top_objects: dict[str, Any] = _t._objects  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# transformers.CLIPFeatureExtractor — removed in transformers 5.x; the long-
# deprecated successor is CLIPImageProcessor (same role, different class
# hierarchy). diffusers' HunYuanDiT pipelines and optimum-onnx's
# modeling_diffusion still import it at module top.
#
# Subclass-with-warning so per-instantiation callers see a deprecation nudge.
# Trade-off: `isinstance(obj, CLIPFeatureExtractor)` on objects built directly
# via `CLIPImageProcessor(...)` returns False — no consumer in this codebase
# performs that check.
# ---------------------------------------------------------------------------
class CLIPFeatureExtractor(CLIPImageProcessor):  # type: ignore[misc, valid-type]
    """Compat shim: CLIPImageProcessor with a deprecation warning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Warn and forward to CLIPImageProcessor."""
        warnings.warn(
            "CLIPFeatureExtractor was removed in transformers 5.x; "
            "use CLIPImageProcessor instead. This shim exists to keep "
            "optimum-onnx 0.1.0 imports working until optimum-onnx 0.2.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# transformers.MT5Tokenizer — removed in transformers 5.x. Three diffusers
# files (hunyuandit, controlnet_hunyuandit, pag_hunyuandit) import it at
# module top and fail. Aliasing to T5Tokenizer is unsafe at runtime: T5's
# SentencePiece vocab is ~32K tokens (English-focused) while MT5's is ~250K
# (multilingual, ~101 languages). A silent alias would tokenize Chinese
# HunYuanDiT prompts with the wrong vocab and produce garbage output.
#
# This stub satisfies `from transformers import MT5Tokenizer` so importers
# can register their pipelines, but raises on any actual instantiation —
# preserving loud failure for real HunYuanDiT usage. We guard both __new__
# and __init__ because HF's from_pretrained may bypass __init__ via
# cls.__new__(cls).
# ---------------------------------------------------------------------------
class MT5Tokenizer:
    """Compat stub: import-time placeholder, raises on instantiation."""

    _ERROR = (
        "MT5Tokenizer was removed in transformers 5.x and is not safely "
        "shimmable: aliasing to T5Tokenizer would silently produce wrong "
        "tokenization for multilingual HunYuanDiT prompts (T5: ~32K English "
        "vocab; MT5: ~250K multilingual vocab). HunYuanDiT and related "
        "pipelines are unsupported in this branch. Use upstream diffusers "
        "once it ships a transformers-5-compatible variant."
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> MT5Tokenizer:
        """Block construction at the lowest level."""
        raise RuntimeError(cls._ERROR)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Defense in depth: also block if __new__ is bypassed."""
        raise RuntimeError(self._ERROR)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> MT5Tokenizer:
        """Block the standard HF tokenizer entry point."""
        raise RuntimeError(cls._ERROR)


# ---------------------------------------------------------------------------
# Inject all top-level shims at once. Each is conditional so re-imports of
# this module are idempotent. transformers 5's _LazyModule.__getattr__
# consults `_objects` before its lazy import structure, so this is the
# correct injection point.
# ---------------------------------------------------------------------------
_top_objects.setdefault("CLIPFeatureExtractor", CLIPFeatureExtractor)
_top_objects.setdefault("MT5Tokenizer", MT5Tokenizer)
# AutoModelForVision2Seq — replaced by AutoModelForImageTextToText in
# transformers 5. Used by optimum-onnx's modeling_seq2seq.py (top-level
# import); without the alias, every `from optimum.onnxruntime import
# ORTModelFor*` cascades and fails. The successor's model registry is not a
# 1:1 superset, so ORTModelForVision2Seq.from_pretrained may fail to load
# certain Vision2Seq checkpoints — Vision2Seq is not exercised by
# winml.modelkit consumers, so the alias only unblocks the import cascade.
_top_objects.setdefault("AutoModelForVision2Seq", AutoModelForImageTextToText)


# ---------------------------------------------------------------------------
# Submodule-level shims. Submodules of transformers are regular ModuleType
# objects (not _LazyModule), so plain setattr lands in __dict__ and persists.
# These are runtime-reachable — see comments per shim.
# ---------------------------------------------------------------------------

# transformers.utils.is_offline_mode — called at runtime by
# optimum/onnxruntime/modeling.py:527 inside ORTModel._from_pretrained.
# Replacement matches the 4.57 implementation: env-var gated bool.
if not hasattr(transformers.utils, "is_offline_mode"):

    def is_offline_mode() -> bool:
        """Compat shim: HF offline mode flag."""
        return os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in ("1", "true", "yes", "on")

    transformers.utils.is_offline_mode = is_offline_mode


# transformers.modeling_utils.get_parameter_dtype — called at runtime by
# optimum/exporters/onnx/convert.py:933 inside onnx_export_from_model.
# Replacement walks the model's parameters and returns the first dtype seen.
if not hasattr(transformers.modeling_utils, "get_parameter_dtype"):

    def get_parameter_dtype(parameter: Any) -> Any:
        """Compat shim: dtype of the first model parameter."""
        try:
            return next(parameter.parameters()).dtype
        except (StopIteration, AttributeError):
            pass
        try:
            return parameter.dtype
        except AttributeError:
            import torch

            return torch.float32

    transformers.modeling_utils.get_parameter_dtype = get_parameter_dtype


# transformers.utils.generic._CAN_RECORD_REGISTRY — registry read at runtime
# by optimum/exporters/onnx/_traceable_decorator.py:57 (every HF ONNX export
# touches this). Empty dict means the recorder branch never fires for any
# model; output_attentions / output_hidden_states capture during export are
# silently skipped (acceptable: not requested by winml.modelkit consumers).
if not hasattr(transformers.utils.generic, "_CAN_RECORD_REGISTRY"):
    transformers.utils.generic._CAN_RECORD_REGISTRY = {}


# transformers.utils.generic.OutputRecorder — referenced inside the
# traceable_check_model_inputs decorator. With _CAN_RECORD_REGISTRY = {}, the
# branch that constructs OutputRecorder is unreachable, so this stub only
# satisfies `from transformers.utils.generic import OutputRecorder` — no
# fidelity work needed.
if not hasattr(transformers.utils.generic, "OutputRecorder"):

    class OutputRecorder:
        """Compat shim: never instantiated at runtime; satisfies the import."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize a recordings dict (defensive; not exercised)."""
            self.recordings: dict[str, Any] = {}

        def __enter__(self) -> OutputRecorder:
            """Enter the recording context."""
            return self

        def __exit__(self, *args: Any) -> None:
            """Exit the recording context."""

        def record(self, name: str, value: Any) -> None:
            """Capture a named output value."""
            self.recordings[name] = value

    transformers.utils.generic.OutputRecorder = OutputRecorder
