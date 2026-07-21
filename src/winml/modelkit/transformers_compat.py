# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compat shim for optimum-onnx 0.1.0 against transformers 5.x, armed lazily.

optimum-onnx 0.1.0 (last PyPI release as of 2026-04-30) hardcodes imports
against transformers 4.x internals. This module re-injects those symbols so
optimum-onnx's imports succeed on transformers 5.x.

Module load only inserts a ``sys.meta_path`` finder — it does NOT load
transformers. The finder calls :func:`install` the first time anything
imports ``optimum.*``; :func:`install` is idempotent. Lightweight commands
that never touch optimum (``winml sys``, ``winml --help``) pay zero
transformers cost.

Drop this file (and the corresponding override in pyproject.toml) once
optimum-onnx 0.2+ ships with transformers 5.x compatibility.
"""
from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec


_installed = False


def install() -> None:
    """Apply the transformers 5.x ↔ optimum-onnx 0.1.0 shim. Idempotent."""
    global _installed
    if _installed:
        return
    # Set the guard flag NOW to prevent re-entrancy: install() itself
    # imports optimum.exporters.onnx.model_patcher below, which
    # re-triggers the meta-path hook → install(). Without this guard,
    # that recursive call would loop forever. On exception we roll the
    # flag back so the next install() call retries with a fresh state.
    _installed = True
    try:
        _install_impl()
    except BaseException:
        _installed = False
        raise


def _install_impl() -> None:
    import os
    import warnings

    import transformers
    import transformers.modeling_utils
    import transformers.utils
    import transformers.utils.generic

    # transformers 5.x's top-level package is a ``_LazyModule``; each
    # ``from transformers import X`` may swap ``sys.modules["transformers"]``
    # with a fresh instance. Force replacements to settle before capturing
    # the live module's ``_objects`` dict (``_LazyModule.__getattr__``
    # consults ``_objects`` first, so this is the durable injection point).
    from transformers import (
        AutoModelForImageTextToText,
        CLIPImageProcessor,
    )

    _t = sys.modules["transformers"]
    _top_objects: dict[str, Any] = _t._objects

    class CLIPFeatureExtractor(CLIPImageProcessor):
        """Successor is CLIPImageProcessor; kept for optimum-onnx / diffusers imports."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                "CLIPFeatureExtractor was removed in transformers 5.x; "
                "use CLIPImageProcessor instead.",
                UserWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    class MT5Tokenizer:
        """Import-time placeholder; instantiation raises.

        Aliasing to T5Tokenizer would silently produce wrong tokenization
        for multilingual HunYuanDiT prompts (T5: ~32K English vocab;
        MT5: ~250K multilingual). Preserve loud failure at instantiation.
        """

        _ERROR = (
            "MT5Tokenizer was removed in transformers 5.x and is not safely "
            "shimmable. HunYuanDiT pipelines are unsupported in this branch."
        )

        def __new__(cls, *args: Any, **kwargs: Any) -> MT5Tokenizer:
            raise RuntimeError(cls._ERROR)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(self._ERROR)

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> MT5Tokenizer:
            raise RuntimeError(cls._ERROR)

    _top_objects.setdefault("CLIPFeatureExtractor", CLIPFeatureExtractor)
    _top_objects.setdefault("MT5Tokenizer", MT5Tokenizer)
    # optimum-onnx's modeling_seq2seq.py imports AutoModelForVision2Seq at
    # top level; alias to successor unblocks the import cascade. Vision2Seq
    # is not exercised by winml.modelkit consumers.
    _top_objects.setdefault("AutoModelForVision2Seq", AutoModelForImageTextToText)

    # Submodules of transformers are regular ModuleType (not _LazyModule),
    # so plain setattr lands in __dict__ and persists.
    if not hasattr(transformers.utils, "is_offline_mode"):

        def is_offline_mode() -> bool:
            return os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in (
                "1", "true", "yes", "on",
            )

        # Monkey-patch an attribute the transformers.utils stubs don't export.
        transformers.utils.is_offline_mode = is_offline_mode  # type: ignore[attr-defined]  # untyped lib monkey-patch

    if not hasattr(transformers.modeling_utils, "get_parameter_dtype"):

        def get_parameter_dtype(parameter: Any) -> Any:
            try:
                return next(parameter.parameters()).dtype
            except (StopIteration, AttributeError):
                pass
            try:
                return parameter.dtype
            except AttributeError:
                import torch
                return torch.float32

        transformers.modeling_utils.get_parameter_dtype = get_parameter_dtype  # type: ignore[attr-defined]  # untyped lib monkey-patch

    # Empty dict means the recorder branch never fires; output_attentions /
    # output_hidden_states capture during export is silently skipped.
    if not hasattr(transformers.utils.generic, "_CAN_RECORD_REGISTRY"):
        transformers.utils.generic._CAN_RECORD_REGISTRY = {}  # type: ignore[attr-defined]  # untyped lib monkey-patch

    if not hasattr(transformers.utils.generic, "OutputRecorder"):

        class OutputRecorder:
            """Never instantiated with _CAN_RECORD_REGISTRY={}; satisfies the import."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.recordings: dict[str, Any] = {}

            def __enter__(self) -> OutputRecorder:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def record(self, name: str, value: Any) -> None:
                self.recordings[name] = value

        # Monkey-patch the untyped transformers module: it declares
        # OutputRecorder as a type, so mypy rejects rebinding it.
        transformers.utils.generic.OutputRecorder = OutputRecorder  # type: ignore[attr-defined]  # untyped lib monkey-patch

    # optimum-onnx 0.1.0's sdpa_mask_without_vmap was written against
    # transformers 4.x's mask_interface (cache_position tensor); tf 5.x
    # passes q_length/q_offset/device directly. Replace optimum's binding.
    try:
        import optimum.exporters.onnx.model_patcher as _optimum_mp
    except ImportError:
        return
    import torch as torch
    from transformers.masking_utils import (
        _ignore_causal_mask_sdpa,
        and_masks,
        causal_mask_function,
        padding_mask_function,
        prepare_padding_mask,
    )

    def _sdpa_mask_without_vmap_tf5(
        batch_size: int,
        q_length: int,
        kv_length: int,
        q_offset: int = 0,
        kv_offset: int = 0,
        mask_function: Any | None = None,
        attention_mask: Any | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        device: Any = "cpu",
        **kwargs: Any,
    ) -> Any:
        if mask_function is None:
            mask_function = causal_mask_function
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, q_offset, kv_offset, local_size
        ):
            return None
        if padding_mask is not None:
            mask_function = and_masks(mask_function, padding_mask_function(padding_mask))
        if isinstance(device, str):
            device = torch.device(device)
        q_indices = (
            torch.arange(q_length, dtype=torch.long, device=device)[None, None, :, None]
            + q_offset
        )
        head_indices = torch.arange(1, dtype=torch.long, device=device)[None, :, None, None]
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[
            :, None, None, None
        ]
        kv_indices = (
            torch.arange(kv_length, dtype=torch.long, device=device)[None, None, None, :]
            + kv_offset
        )
        # mask_function is a dynamically-selected callable from the untyped
        # transformers masking API; cast to Any so mypy doesn't bind it to a
        # concrete signature.
        causal_mask = cast("Any", mask_function)(
            batch_indices, head_indices, q_indices, kv_indices
        )
        return causal_mask.expand(batch_size, -1, q_length, kv_length)

    _optimum_mp.sdpa_mask_without_vmap = _sdpa_mask_without_vmap_tf5


class _OptimumImportHook(MetaPathFinder):
    """sys.meta_path finder: calls install() when anything imports optimum.*."""

    def find_spec(
        self,
        name: str,
        path: Sequence[str] | None,
        target: object = None,
    ) -> ModuleSpec | None:
        if name == "optimum" or name.startswith("optimum."):
            # Guard: if transformers is still initializing (e.g. its own
            # dependency_versions_check probes for optimum), running
            # install() now would circular-import transformers.utils.
            # Wait until transformers has finished loading.
            _tf = sys.modules.get("transformers")
            if _tf is None or getattr(_tf, "__spec__", None) is None:
                return None
            _tf_utils = sys.modules.get("transformers.utils")
            if _tf_utils is None or not hasattr(_tf_utils, "HF_MODULES_CACHE"):
                return None
            install()
        return None  # never claim ownership; fall through to normal finders


def arm() -> None:
    """Insert the hook at meta_path[0]. Idempotent — safe to call repeatedly."""
    if not any(isinstance(f, _OptimumImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _OptimumImportHook())
