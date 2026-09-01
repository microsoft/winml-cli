# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility helpers for transformers-dependent export paths.

The import hook below provides an optimum-onnx 0.1.0 shim against transformers
5.x, armed lazily.

optimum-onnx 0.1.0 (last PyPI release as of 2026-04-30) hardcodes imports
against transformers 4.x internals. This module re-injects those symbols so
optimum-onnx's imports succeed on transformers 5.x.

Module load only defines lightweight helpers and inserts a ``sys.meta_path``
finder — it does NOT load transformers. The finder calls :func:`install` the
first time anything imports ``optimum.*``; :func:`install` is idempotent.
Lightweight commands that never touch optimum (``winml sys``, ``winml --help``)
pay zero transformers cost.

Drop this file (and the corresponding override in pyproject.toml) once
optimum-onnx 0.2+ ships with transformers 5.x compatibility.
"""

from __future__ import annotations

import contextlib
import sys
from importlib.abc import Loader, MetaPathFinder
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import torch.nn as nn


_installed = False

# optimum-onnx's model_patcher module is handled specially: unlike the rest
# of optimum.*, our sdpa_mask_without_vmap replacement must be applied after
# it finishes executing (see _PatchModelPatcherLoader), not while install()
# is still running.
_MODEL_PATCHER_MODULE = "optimum.exporters.onnx.model_patcher"


@contextlib.contextmanager
def use_eager_attention_for_export(model: nn.Module) -> Iterator[None]:
    """Temporarily prefer eager attention and normalize SDPA mask dtypes.

    Some exporter paths feed integer attention masks into SDPA, which newer
    torch versions reject. During export-only tracing, coerce integral masks
    to bool before dispatching to the currently-installed SDPA implementation.
    """
    import torch

    restored: list[tuple[int, Any, Any]] = []
    configs: dict[int, Any] = {}
    children: dict[int, set[int]] = {}
    seen_configs: set[int] = set()
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def _coerce_integer_mask_sdpa(*args: Any, **kwargs: Any) -> Any:
        attn_mask: Any | None = None
        use_positional = len(args) >= 4

        if use_positional:
            attn_mask = args[3]
        elif "attn_mask" in kwargs:
            attn_mask = kwargs["attn_mask"]

        if isinstance(attn_mask, torch.Tensor) and attn_mask.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            cast_mask = attn_mask.to(dtype=torch.bool)
            if use_positional:
                args_list = list(args)
                args_list[3] = cast_mask
                args = tuple(args_list)
            else:
                kwargs["attn_mask"] = cast_mask

        return original_sdpa(*args, **kwargs)

    for module in model.modules():
        _collect_attention_configs(getattr(module, "config", None), configs, children, seen_configs)

    for config_id, config in configs.items():
        current = config._attn_implementation
        restored.append((config_id, config, current))

    for config in configs.values():
        if config._attn_implementation != "eager":
            config._attn_implementation = "eager"
    torch.nn.functional.scaled_dot_product_attention = _coerce_integer_mask_sdpa

    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
        for _config_id, config, previous in _parent_before_child(restored, children):
            config._attn_implementation = previous


def _collect_attention_configs(
    config: Any,
    configs: dict[int, Any],
    children: dict[int, set[int]],
    seen_configs: set[int],
) -> None:
    if config is None or id(config) in seen_configs:
        return
    seen_configs.add(id(config))

    config = cast("Any", config)
    config_id = id(config)
    if hasattr(config, "_attn_implementation"):
        configs[config_id] = config

    for child_config in _iter_sub_configs(config):
        if hasattr(config, "_attn_implementation") and hasattr(
            child_config, "_attn_implementation"
        ):
            children.setdefault(config_id, set()).add(id(child_config))
        _collect_attention_configs(child_config, configs, children, seen_configs)


def _iter_sub_configs(config: Any) -> Iterator[Any]:
    sub_configs = getattr(config, "sub_configs", None)
    if isinstance(sub_configs, dict):
        for key, value in sub_configs.items():
            if isinstance(key, str):
                child = getattr(config, key, None)
                if child is not None:
                    yield child
            if not isinstance(value, type):
                yield value
    elif isinstance(sub_configs, (list, tuple, set)):
        yield from sub_configs


def _parent_before_child(
    restored: list[tuple[int, Any, Any]],
    children: dict[int, set[int]],
) -> list[tuple[int, Any, Any]]:
    parents: dict[int, set[int]] = {}
    restored_ids = {config_id for config_id, _config, _previous in restored}
    for parent_id, child_ids in children.items():
        if parent_id not in restored_ids:
            continue
        for child_id in child_ids:
            if child_id in restored_ids:
                parents.setdefault(child_id, set()).add(parent_id)

    depths: dict[int, int] = {}

    def depth(config_id: int, visiting: set[int]) -> int:
        if config_id in depths:
            return depths[config_id]
        if config_id in visiting:
            return 0
        visiting.add(config_id)
        config_depth = 0
        if config_id in parents:
            config_depth = 1 + max(depth(parent_id, visiting) for parent_id in parents[config_id])
        visiting.remove(config_id)
        depths[config_id] = config_depth
        return config_depth

    return sorted(restored, key=lambda item: depth(item[0], set()))


def install() -> None:
    """Apply the transformers 5.x ↔ optimum-onnx 0.1.0 shim. Idempotent."""
    global _installed
    if _installed:
        return
    # Set the guard flag NOW to prevent re-entrancy: _install_impl() imports
    # transformers, which may itself probe for optimum's availability (e.g.
    # via importlib.util.find_spec("optimum")) and re-trigger the meta-path
    # hook → install(). Without this guard, that recursive call would loop
    # forever. On exception we roll the flag back so the next install()
    # call retries with a fresh state.
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

    utils_mod = cast("Any", transformers.utils)
    modeling_utils_mod = cast("Any", transformers.modeling_utils)
    generic_utils_mod = cast("Any", transformers.utils.generic)

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
    if not hasattr(utils_mod, "is_offline_mode"):

        def is_offline_mode() -> bool:
            return os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        # Monkey-patch an attribute the transformers.utils stubs don't export.
        utils_mod.is_offline_mode = is_offline_mode

    if not hasattr(modeling_utils_mod, "get_parameter_dtype"):

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

        modeling_utils_mod.get_parameter_dtype = get_parameter_dtype

    # Empty dict means the recorder branch never fires; output_attentions /
    # output_hidden_states capture during export is silently skipped.
    if not hasattr(generic_utils_mod, "_CAN_RECORD_REGISTRY"):
        generic_utils_mod._CAN_RECORD_REGISTRY = {}

    if not hasattr(generic_utils_mod, "OutputRecorder"):

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
        generic_utils_mod.OutputRecorder = OutputRecorder

    # optimum-onnx 0.1.0's sdpa_mask_without_vmap was written against
    # transformers 4.x's mask_interface (cache_position tensor); tf 5.x
    # passes q_length/q_offset/device directly. Replace optimum's binding.
    #
    # Do NOT import optimum.exporters.onnx.model_patcher here: install() is
    # itself invoked from _OptimumImportHook.find_spec, which may still be
    # resolving that very module. A nested import at that point re-enters
    # the import machinery, runs model_patcher's module body to completion,
    # and returns — but the *outer* import (the one that triggered this
    # find_spec call in the first place) then proceeds to load the module a
    # second time from scratch, discarding the freshly patched instance and
    # replacing it with an unpatched one. See _PatchModelPatcherLoader below
    # for the fix: it applies the patch after model_patcher's *own* load
    # completes, so there's exactly one execution to worry about.
    #
    # This fallback only covers the case where the module is already fully
    # loaded (e.g. re-running install() after resetting state, or
    # model_patcher having been imported through a path that bypassed the
    # wrapping loader).
    _existing_model_patcher = sys.modules.get(_MODEL_PATCHER_MODULE)
    if _existing_model_patcher is not None:
        _patch_model_patcher(_existing_model_patcher)


def _patch_model_patcher(module: Any) -> None:
    """Replace ``sdpa_mask_without_vmap`` on an already-loaded model_patcher module.

    Must only be called after ``module`` has finished executing its own
    top-level code (its own definition is assigned near the end of the
    module body, so patching any earlier would just get overwritten).
    """
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
        try:
            # transformers >=5 uses _slice to preserve compile-friendly indexing.
            padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        except TypeError:
            # Older helper signatures do not expose _slice.
            padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)
        ignore_causal_mask_sdpa = cast("Any", _ignore_causal_mask_sdpa)
        should_skip = False
        if allow_is_causal_skip:
            try:
                # transformers >=5.1 signature
                should_skip = bool(
                    ignore_causal_mask_sdpa(
                        padding_mask,
                        q_length,
                        kv_length,
                        kv_offset,
                        local_size,
                    )
                )
            except TypeError:
                # transformers 5.0 / late 4.x signature
                should_skip = bool(
                    ignore_causal_mask_sdpa(
                        padding_mask,
                        q_length,
                        kv_length,
                        q_offset,
                        kv_offset,
                        local_size,
                    )
                )
        if should_skip:
            return None
        if padding_mask is not None:
            mask_function = and_masks(mask_function, padding_mask_function(padding_mask))
        if isinstance(device, str):
            device = torch.device(device)
        q_indices = (
            torch.arange(q_length, dtype=torch.long, device=device)[None, None, :, None] + q_offset
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
        causal_mask = cast("Any", mask_function)(batch_indices, head_indices, q_indices, kv_indices)
        return causal_mask.expand(batch_size, -1, q_length, kv_length)

    module.sdpa_mask_without_vmap = _sdpa_mask_without_vmap_tf5


class _PatchModelPatcherLoader(Loader):
    """Wraps model_patcher's real loader to apply the sdpa_mask_without_vmap patch.

    The patch is applied only once the module body has fully executed.
    Patching any earlier (e.g. via a nested import while the module is
    still mid-exec, as install() used to do) risks the module's own
    definition — assigned near the end of its body — silently overwriting
    the replacement.
    """

    def __init__(self, wrapped: Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create_module = getattr(self._wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        _patch_model_patcher(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _OptimumImportHook(MetaPathFinder):
    """sys.meta_path finder: calls install() when anything imports optimum.*."""

    # Duck-typed marker (rather than isinstance/class identity) so stale
    # instances left behind by a module reload — a distinct class object,
    # despite the same name — are still recognized as "one of ours".
    _is_optimum_import_hook = True

    def find_spec(
        self,
        name: str,
        path: Sequence[str] | None,
        target: object = None,
    ) -> ModuleSpec | None:
        if name != "optimum" and not name.startswith("optimum."):
            return None
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

        if name == _MODEL_PATCHER_MODULE:
            # Unlike the rest of optimum.*, this module needs a patch
            # applied strictly after it finishes loading (see
            # _PatchModelPatcherLoader), so — uniquely for this one module
            # — claim ownership by delegating to whichever finder would
            # normally handle it and wrapping the resulting loader.
            spec = self._find_delegate_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchModelPatcherLoader(spec.loader)
            return spec

        return None  # never claim ownership; fall through to normal finders

    def _find_delegate_spec(
        self,
        name: str,
        path: Sequence[str] | None,
        target: object,
    ) -> ModuleSpec | None:
        for finder in sys.meta_path:
            # Skip every _OptimumImportHook instance, not just self: repeated
            # module reloads (e.g. test fixtures that re-import winml.modelkit)
            # can leave multiple stale hook instances in sys.meta_path, each a
            # distinct object of a distinct (reloaded) class. Delegating to
            # one of those would just bounce back into this same branch.
            if getattr(finder, "_is_optimum_import_hook", False):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(name, path, target)
            if spec is not None:
                return cast("ModuleSpec | None", spec)
        return None


def arm() -> None:
    """Insert the hook at meta_path[0]. Idempotent — safe to call repeatedly."""
    if not any(isinstance(f, _OptimumImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _OptimumImportHook())
