# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for ep forwarding in WinMLAutoModel.from_pretrained().

The HF build path used to derive the analyzer EP solely from
``config.compile.ep_config.provider``. On CPU (and other compile-less paths)
``config.compile`` is None, so the user-supplied ``--ep cpu`` was dropped and
the static analyzer fell back to its all-EP aggregation mode.

The fix prefers ``kwargs["ep"]`` over the compile-derived value.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


class _StopAfterEpCheckError(Exception):
    """Sentinel raised from the stubbed build_hf_model to abort from_pretrained."""


def _install_stubs(monkeypatch: pytest.MonkeyPatch, *, compile_provider: str | None) -> dict:
    """Wire monkeypatches around from_pretrained so the only real code that runs
    is the EP-resolution logic. Returns a dict that will be populated with the
    kwargs that reach build_hf_model.
    """
    import transformers

    from winml.modelkit import build as build_pkg
    from winml.modelkit import config as config_pkg

    fake_build_config = MagicMock()
    if compile_provider is None:
        fake_build_config.compile = None
    else:
        fake_build_config.compile.ep_config.provider = compile_provider
    fake_build_config.loader.task = "image-classification"
    fake_build_config.loader.trust_remote_code = False
    fake_build_config.generate_cache_key.return_value = "deadbeef"
    monkeypatch.setattr(config_pkg, "generate_hf_build_config", lambda *a, **k: fake_build_config)

    fake_hf_config = MagicMock()
    fake_hf_config.model_type = "resnet"
    monkeypatch.setattr(
        transformers,
        "AutoConfig",
        MagicMock(from_pretrained=lambda *a, **k: fake_hf_config),
    )

    received: dict[str, Any] = {}

    def stub_build(**kwargs: Any) -> None:
        received.update(kwargs)
        raise _StopAfterEpCheckError

    monkeypatch.setattr(build_pkg, "build_hf_model", stub_build)
    return received


def test_explicit_ep_reaches_build_when_compile_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User passes --ep cpu and config.compile is None — ep must still propagate."""
    from winml.modelkit.models import WinMLAutoModel

    received = _install_stubs(monkeypatch, compile_provider=None)

    with pytest.raises(_StopAfterEpCheckError):
        WinMLAutoModel.from_pretrained("microsoft/resnet-50", ep="cpu", device="cpu")

    assert received.get("ep") == "cpu", (
        f"Expected ep='cpu' to reach build_hf_model, got {received.get('ep')!r}. "
        "Without this, analyze_onnx defaults to ep=None and aggregates across "
        "all EPs."
    )


def test_compile_provider_used_when_user_ep_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User omits --ep — fall back to config.compile.ep_config.provider."""
    from winml.modelkit.models import WinMLAutoModel

    received = _install_stubs(monkeypatch, compile_provider="QNNExecutionProvider")

    with pytest.raises(_StopAfterEpCheckError):
        WinMLAutoModel.from_pretrained("microsoft/resnet-50", device="npu")

    assert received.get("ep") == "QNNExecutionProvider"


def test_explicit_ep_overrides_compile_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-supplied ep wins over the compile-derived value."""
    from winml.modelkit.models import WinMLAutoModel

    received = _install_stubs(monkeypatch, compile_provider="QNNExecutionProvider")

    with pytest.raises(_StopAfterEpCheckError):
        WinMLAutoModel.from_pretrained("microsoft/resnet-50", ep="cpu", device="npu")

    assert received.get("ep") == "cpu"


def test_both_absent_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No user ep and no compile config — ep stays None (legacy behavior)."""
    from winml.modelkit.models import WinMLAutoModel

    received = _install_stubs(monkeypatch, compile_provider=None)

    with pytest.raises(_StopAfterEpCheckError):
        WinMLAutoModel.from_pretrained("microsoft/resnet-50", device="cpu")

    assert received.get("ep") is None


@pytest.mark.parametrize("flag", [True, False])
def test_allow_unsupported_nodes_reaches_build(
    monkeypatch: pytest.MonkeyPatch, flag: bool
) -> None:
    """``allow_unsupported_nodes`` propagates to build_hf_model (HF path)."""
    from winml.modelkit.models import WinMLAutoModel

    received = _install_stubs(monkeypatch, compile_provider=None)

    with pytest.raises(_StopAfterEpCheckError):
        WinMLAutoModel.from_pretrained(
            "microsoft/resnet-50", device="cpu", allow_unsupported_nodes=flag
        )

    assert received.get("allow_unsupported_nodes") is flag


def test_allow_unsupported_nodes_reaches_composite(monkeypatch: pytest.MonkeyPatch) -> None:
    """``allow_unsupported_nodes`` reaches the composite-model dispatch path."""
    import transformers

    from winml.modelkit.models import WinMLAutoModel
    from winml.modelkit.models.winml import composite_model as cm_mod

    # One fake composite registered for (model_type, task).
    monkeypatch.setattr(cm_mod, "COMPOSITE_MODEL_REGISTRY", {("faketype", "faketask"): object})

    fake_cfg = MagicMock()
    fake_cfg.model_type = "faketype"
    monkeypatch.setattr(
        transformers, "AutoConfig", MagicMock(from_pretrained=lambda *a, **k: fake_cfg)
    )

    received: dict[str, Any] = {}

    def _stub_composite(*_args: Any, **kwargs: Any) -> str:
        received.update(kwargs)
        return "COMPOSITE_SENTINEL"

    monkeypatch.setattr(
        cm_mod.WinMLCompositeModel, "from_pretrained", staticmethod(_stub_composite)
    )

    result = WinMLAutoModel.from_pretrained(
        "some/composite", task="faketask", allow_unsupported_nodes=True
    )

    assert result == "COMPOSITE_SENTINEL"
    assert received.get("allow_unsupported_nodes") is True
