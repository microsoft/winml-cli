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

from types import SimpleNamespace
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
    from winml.modelkit import session as session_pkg
    from winml.modelkit.session import EPDeviceTarget

    fake_build_config = MagicMock()
    if compile_provider is None:
        fake_build_config.compile = None
    else:
        fake_build_config.compile.ep_config.provider = compile_provider
    fake_build_config.loader.task = "image-classification"
    fake_build_config.loader.trust_remote_code = False
    fake_build_config.generate_cache_key.return_value = "deadbeef"
    monkeypatch.setattr(config_pkg, "generate_hf_build_config", lambda *a, **k: fake_build_config)

    fake_ep_device = MagicMock()
    fake_ep_device.device.device_type = "CPU"
    fake_ep_device.device.ep_name = "CPUExecutionProvider"
    monkeypatch.setattr(
        session_pkg,
        "resolve_device",
        lambda target: EPDeviceTarget(
            ep=target.ep if target.ep != "auto" else "QNNExecutionProvider",
            device=target.device,
        ),
    )
    monkeypatch.setattr(
        session_pkg.WinMLEPRegistry,
        "instance",
        classmethod(lambda _cls: MagicMock(auto_device=lambda _target: fake_ep_device)),
    )

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
def test_allow_unsupported_nodes_reaches_build(monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
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

    received: dict[str, Any] = {}

    class _FakeComposite:
        @staticmethod
        def from_pretrained(*_args: Any, **kwargs: Any) -> str:
            received.update(kwargs)
            return "COMPOSITE_SENTINEL"

    # One fake composite registered for (model_type, task).  The dispatch
    # resolves the concrete class from the registry and calls *its*
    # ``from_pretrained`` (honouring an explicit ``model_type`` override),
    # rather than routing through the base ``WinMLCompositeModel``.
    monkeypatch.setattr(
        cm_mod, "COMPOSITE_MODEL_REGISTRY", {("faketype", "faketask"): _FakeComposite}
    )

    fake_cfg = MagicMock()
    fake_cfg.model_type = "faketype"
    monkeypatch.setattr(
        transformers, "AutoConfig", MagicMock(from_pretrained=lambda *a, **k: fake_cfg)
    )

    result = WinMLAutoModel.from_pretrained(
        "some/composite", task="faketask", allow_unsupported_nodes=True
    )

    assert result == "COMPOSITE_SENTINEL"
    assert received.get("allow_unsupported_nodes") is True


def test_cache_reuse_does_not_eagerly_load_hf_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The build API owns cache reuse, so AutoModel must not preload weights."""
    from winml.modelkit.models import WinMLAutoModel
    from winml.modelkit.models import auto as auto_module

    build_config = MagicMock()
    build_config.loader.task = "image-classification"
    build_config.compile = None
    build_config.generate_cache_key.return_value = "cache-key"
    hf_config = SimpleNamespace(model_type="unit-type")
    build_result = SimpleNamespace(final_onnx_path=tmp_path / "cached.onnx")
    build_result.final_onnx_path.write_bytes(b"cached")

    monkeypatch.setattr(
        "winml.modelkit.config.generate_hf_build_config",
        lambda *_args, **_kwargs: build_config,
    )
    monkeypatch.setattr(
        "winml.modelkit.loader._autoconfig.load_hf_config",
        lambda *_args, **_kwargs: hf_config,
    )
    monkeypatch.setattr(auto_module, "get_cache_dir", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(auto_module, "get_model_dir", lambda *_args, **_kwargs: tmp_path)
    received: dict[str, Any] = {}

    def reuse_build(**kwargs: Any) -> SimpleNamespace:
        received.update(kwargs)
        return build_result

    monkeypatch.setattr(
        "winml.modelkit.build.build_hf_model",
        reuse_build,
    )
    wrapper = SimpleNamespace()
    monkeypatch.setattr(auto_module, "get_winml_class", lambda *_args: lambda **_kwargs: wrapper)

    assert (
        WinMLAutoModel.from_pretrained(
            "unit/model",
            ep_device=MagicMock(
                device=MagicMock(device_type="CPU", ep_name="CPUExecutionProvider")
            ),
            task="image-classification",
        )
        is wrapper
    )
    assert received.get("pytorch_model") is None
