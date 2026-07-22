# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Registry-driven composite inspect support tests."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from winml.modelkit.inspect.types import (
    ExporterInfo,
    IOConfigInfo,
    LoaderInfo,
    ProcessorInfo,
    SupportLevel,
    WinMLInfo,
)


def _supported_loader() -> LoaderInfo:
    return LoaderInfo("ComponentModel", "registry", SupportLevel.SUPPORTED)


def _supported_exporter() -> ExporterInfo:
    return ExporterInfo(
        "ComponentOnnxConfig",
        "registry",
        SupportLevel.SUPPORTED,
    )


def test_composite_exporter_aggregates_every_registered_component(
    monkeypatch,
) -> None:
    """A composite exporter is supported only after every registered component resolves."""
    from winml.modelkit.inspect import resolver
    from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

    class Composite:
        _SUB_MODEL_CONFIG: ClassVar = {"first": "component-one", "second": "component-two"}

    calls: list[tuple[str, str]] = []

    def resolve_loader(model_type: str, task: str) -> LoaderInfo:
        calls.append(("loader", task))
        return _supported_loader()

    def resolve_exporter(
        model_type: str,
        task: str,
        hf_config: object | None = None,
        *,
        model_id: str | None = None,
    ) -> ExporterInfo:
        calls.append(("exporter", task))
        return _supported_exporter()

    monkeypatch.setitem(COMPOSITE_MODEL_REGISTRY, ("test-composite", "pipeline"), Composite)
    monkeypatch.setattr(resolver, "resolve_loader", resolve_loader)
    monkeypatch.setattr(resolver, "resolve_exporter", resolve_exporter)

    info = resolver.resolve_composite_exporter("test-composite", "pipeline")

    assert info is not None
    assert info.support_level is SupportLevel.SUPPORTED
    assert info.onnx_config_source == "COMPOSITE_MODEL_REGISTRY"
    assert calls == [
        ("loader", "component-one"),
        ("exporter", "component-one"),
        ("loader", "component-two"),
        ("exporter", "component-two"),
    ]


def test_composite_exporter_is_unsupported_when_a_component_cannot_export(
    monkeypatch,
) -> None:
    """One unsupported component prevents the registered composite from being supported."""
    from winml.modelkit.inspect import resolver
    from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

    class Composite:
        _SUB_MODEL_CONFIG: ClassVar = {"first": "component-one", "second": "component-two"}

    unsupported = ExporterInfo(None, "none", SupportLevel.UNSUPPORTED)
    monkeypatch.setitem(COMPOSITE_MODEL_REGISTRY, ("test-composite", "pipeline"), Composite)
    monkeypatch.setattr(resolver, "resolve_loader", lambda *_: _supported_loader())
    monkeypatch.setattr(
        resolver,
        "resolve_exporter",
        lambda _model_type, task, **_: unsupported
        if task == "component-two"
        else _supported_exporter(),
    )

    info = resolver.resolve_composite_exporter("test-composite", "pipeline")

    assert info is not None
    assert info.support_level is SupportLevel.UNSUPPORTED


def test_optimization_only_build_config_is_not_an_exporter(monkeypatch) -> None:
    """Legacy exporter lookup must continue to fallback when a build config has no export."""
    from winml.modelkit.config import WinMLBuildConfig
    from winml.modelkit.inspect import resolver
    from winml.modelkit.optim import WinMLOptimizationConfig

    monkeypatch.setitem(
        resolver.MODEL_BUILD_CONFIGS,
        "test-optim-only",
        WinMLBuildConfig(optim=WinMLOptimizationConfig()),
    )
    with monkeypatch.context() as context:
        tasks_manager = MagicMock()
        tasks_manager.get_exporter_config_constructor.return_value = None
        context.setattr(
            "optimum.exporters.tasks.TasksManager",
            tasks_manager,
        )
        info = resolver.resolve_exporter("test-optim-only", "component-one")

    assert info.support_level is SupportLevel.UNSUPPORTED


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_legacy_inspect_forwards_trust_remote_code_to_hierarchy(
    monkeypatch,
    trust_remote_code: bool,
) -> None:
    """The legacy inspect API must forward trust consent to hierarchy loading."""
    import winml.modelkit.inspect as inspect_module

    hf_config = MagicMock()
    hf_config.model_type = "test-composite"
    hf_config.architectures = []
    config_loader = MagicMock(return_value=hf_config)
    hierarchy_loader = MagicMock(return_value=MagicMock(hf_module_count=1))
    monkeypatch.setattr(inspect_module.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(
        "winml.modelkit.inspect.hierarchy.extract_hierarchy",
        hierarchy_loader,
    )
    monkeypatch.setattr(inspect_module, "resolve_loader", lambda *_: _supported_loader())
    monkeypatch.setattr(
        inspect_module,
        "resolve_exporter",
        lambda *_args, **_kwargs: _supported_exporter(),
    )
    monkeypatch.setattr(
        inspect_module,
        "resolve_winml",
        lambda *_: WinMLInfo("Composite", "registry", SupportLevel.SUPPORTED),
    )
    monkeypatch.setattr(inspect_module, "resolve_cache", lambda *_: MagicMock())
    monkeypatch.setattr(
        inspect_module, "resolve_processor", lambda *_args, **_kwargs: ProcessorInfo()
    )
    monkeypatch.setattr(
        inspect_module, "resolve_io_config", lambda *_args, **_kwargs: IOConfigInfo()
    )
    monkeypatch.setattr(inspect_module, "get_build_config", lambda *_: None)
    monkeypatch.setattr(inspect_module, "resolve_composite_info", lambda *_: None)
    monkeypatch.setattr(
        inspect_module,
        "resolve_composite_exporter",
        lambda *_args, **_kwargs: None,
    )

    inspect_module.inspect_model(
        "test/model",
        include_hierarchy=True,
        trust_remote_code=trust_remote_code,
    )

    config_loader.assert_called_once_with("test/model", trust_remote_code=trust_remote_code)
    hierarchy_loader.assert_called_once_with("test/model", trust_remote_code=trust_remote_code)
