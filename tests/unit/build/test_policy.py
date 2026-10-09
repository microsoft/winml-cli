# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for target-driven CGC build configuration."""

import pytest

from winml.modelkit.config import WinMLBuildConfig
from winml.modelkit.config.build import _apply_cgc_config
from winml.modelkit.export import WinMLExportConfig
from winml.modelkit.optim import WinMLOptimizationConfig
from winml.modelkit.utils.constants import RuntimeBackend


@pytest.mark.parametrize(
    ("backend", "ep"),
    [
        (None, "cpu"),
        ("ort", "openvino"),
        ("ort", "nvtensorrtrtx"),
    ],
)
def test_non_cgc_target_preserves_build_stages(
    backend: RuntimeBackend | None, ep: str
) -> None:
    config = WinMLBuildConfig()
    original_export = config.export
    original_optim = config.optim
    original_quant = config.quant
    original_compile = config.compile
    original_auto = config.auto

    _apply_cgc_config(config, backend=backend, ep=ep)

    assert config.export is original_export
    assert config.optim is original_optim
    assert config.quant is original_quant
    assert config.compile is original_compile
    assert config.auto is original_auto
    assert config.skip_optimize is False
    assert config.convert is None


@pytest.mark.parametrize(
    ("backend", "ep"),
    [
        ("cgc", None),
        ("cgc", "winmlcg"),
        ("cgc", "WinMLCGExecutionProvider"),
        (None, "winmlcg"),
        (None, "WinMLCGExecutionProvider"),
    ],
)
def test_cgc_target_enables_compatibility_optimization(
    backend: RuntimeBackend | None, ep: str | None
) -> None:
    config = WinMLBuildConfig()
    original_export = config.export
    original_quant = config.quant

    _apply_cgc_config(config, backend=backend, ep=ep)

    assert config.export is original_export
    assert config.quant is original_quant
    assert config.auto is False
    assert config.skip_optimize is False
    assert config.optim == WinMLOptimizationConfig.for_cgc()
    assert config.compile is None

    if backend == "cgc":
        assert config.convert is not None
        assert config.convert.target == "cgir"
        assert not config.convert.options
    else:
        assert config.convert is None


def test_cgc_target_preserves_existing_conversion_config() -> None:
    conversion = WinMLExportConfig(target="cgir", options={})
    config = WinMLBuildConfig(convert=conversion)

    _apply_cgc_config(config, backend="cgc", ep=None)

    assert config.convert is conversion
    assert config.convert.options == {}
