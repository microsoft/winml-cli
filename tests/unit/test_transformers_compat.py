# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for the transformers-5 / optimum-onnx-0.1.0 compat shim.

``bart.py`` (and friends) import ``optimum.exporters.onnx.model_patcher``
directly. The ``sys.meta_path`` hook in ``transformers_compat`` must patch
that module's ``sdpa_mask_without_vmap`` with a transformers-5 compatible
replacement regardless of *when* the module first gets imported relative to
explicit compatibility setup. These tests exercise the real transformers and
optimum packages (no fabricated model output) and pin both import orders
called out in the PR #1019 review thread.
"""

from __future__ import annotations

import inspect
import sys

import pytest
import torch

from winml.modelkit import transformers_compat


_MODEL_PATCHER_MODULE = "optimum.exporters.onnx.model_patcher"


def _clear_optimum_modules() -> None:
    # Force a fresh (re-)import of optimum on the next `import optimum...`
    # statement so each test can independently control ordering relative to
    # the compat hook.
    for name in [m for m in sys.modules if m == "optimum" or m.startswith("optimum.")]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _reset_compat_state():
    _clear_optimum_modules()
    transformers_compat._installed = False
    yield
    _clear_optimum_modules()
    transformers_compat._installed = False


def _is_transformers5_patched(module: object) -> bool:
    sig = inspect.signature(module.sdpa_mask_without_vmap)  # type: ignore[attr-defined]
    return "q_length" in sig.parameters and "cache_position" not in sig.parameters


def test_direct_model_patcher_import_gets_transformers5_patch():
    """bart.py's exact import order: nothing has touched optimum.* yet.

    This is the order that previously broke: the meta-path hook's install()
    recursively re-imported model_patcher while the outer import machinery
    was still resolving the very same module, so model_patcher finished
    executing a second time afterward and clobbered the patch with its own
    transformers-4 implementation.
    """
    import optimum.exporters.onnx.model_patcher as model_patcher

    assert _is_transformers5_patched(model_patcher)


def test_explicit_install_before_model_patcher_import_still_patches():
    """Compatibility setup triggered (via importing optimum) before the
    direct model_patcher import must keep working."""
    transformers_compat.install()
    import optimum.exporters.onnx.model_patcher as model_patcher

    assert _is_transformers5_patched(model_patcher)


def test_explicit_install_patches_model_patcher_imported_before_setup():
    """model_patcher already sitting in sys.modules (imported through a path
    that bypassed the wrapping loader) must still get patched once explicit
    compatibility setup runs.
    """
    # Apply the generic transformers-level shims so a raw model_patcher
    # import can succeed at all, then force a fresh, hook-free import to
    # simulate a caller that imported model_patcher without ever routing
    # through the compat meta-path finder.
    transformers_compat.install()
    transformers_compat._installed = False
    _clear_optimum_modules()
    # Remove every hook-like finder, not just one instance: test fixtures
    # elsewhere in the suite may reimport winml.modelkit and leave stale
    # hook instances (from earlier module generations) sitting in
    # sys.meta_path; any one of them would otherwise still intercept this
    # import and wrap it.
    hooks = [f for f in sys.meta_path if getattr(f, "_is_optimum_import_hook", False)]
    for hook in hooks:
        sys.meta_path.remove(hook)
    try:
        import optimum.exporters.onnx.model_patcher as model_patcher
    finally:
        for hook in hooks:
            sys.meta_path.insert(0, hook)

    assert not _is_transformers5_patched(model_patcher)

    transformers_compat.install()

    assert _is_transformers5_patched(model_patcher)


def test_install_is_idempotent_and_reentrant_safe():
    import optimum.exporters.onnx.model_patcher as model_patcher

    assert _is_transformers5_patched(model_patcher)

    # Calling install() again (e.g. a second optimum.* import elsewhere)
    # must not raise and must leave the module correctly patched.
    transformers_compat.install()
    transformers_compat.install()

    assert _is_transformers5_patched(model_patcher)


def test_traceable_sdpa_accepts_integral_visibility_mask() -> None:
    """Optimum must normalize the integral masks emitted by Transformers 4.57."""
    import optimum.exporters.onnx.model_patcher as model_patcher

    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    integral_mask = torch.tensor(
        [[[[1, 1, 0], [1, 1, 0], [1, 1, 1]]]],
        dtype=torch.int64,
    )

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=integral_mask.to(torch.bool),
    )
    actual = model_patcher.traceable_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=integral_mask,
    )

    torch.testing.assert_close(actual, expected)


def test_integral_sdpa_mask_patch_is_idempotent() -> None:
    """Repeated compatibility setup must not stack SDPA wrappers."""
    import optimum.exporters.onnx.model_patcher as model_patcher

    first = model_patcher.original_scaled_dot_product_attention
    transformers_compat._patch_model_patcher(model_patcher)

    assert model_patcher.original_scaled_dot_product_attention is first
