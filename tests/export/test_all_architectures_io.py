# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parametrized tests across ALL Optimum architectures for export/io.py.

Validates that ``_get_onnx_config`` returns a valid OnnxConfig (with .inputs
and .outputs) for every transformers architecture in the OPTIMUM_ARCHITECTURES
catalog.

Architectures where ``AutoConfig.for_model()`` fails (custom/rare models that
require specific config classes not registered with AutoConfig) are skipped
via ``pytest.skip()`` since they genuinely cannot be tested without a network
fetch or model-specific config construction.

Diffusers, sentence_transformers, and timm architectures are excluded because
they require different config types (pipeline configs, not PretrainedConfig).
"""

from __future__ import annotations

import pytest

# Trigger OnnxConfig registration with TasksManager (custom overrides).
import winml.modelkit.models  # noqa: F401
from tests.assets.optimum_architectures import OPTIMUM_ARCHITECTURES


# ---------------------------------------------------------------------------
# Build parametrize list: transformers-only
# ---------------------------------------------------------------------------

# Architectures with known Optimum bugs that prevent .inputs/.outputs access
# with a default AutoConfig. These are xfailed so the user can track and fix later.
_XFAIL_ARCHS: dict[str, str] = {
    # SpeechT5OnnxConfig._behavior is unset when task="text-to-audio" with default config;
    # accessing .inputs raises ValueError in Optimum's model_configs.py.
    "speecht5": "Optimum bug: SpeechT5OnnxConfig._behavior unset for text-to-audio default config",
}

TRANSFORMERS_ARCHS = [
    pytest.param(
        key,
        info,
        id=key,
        marks=[pytest.mark.xfail(reason=_XFAIL_ARCHS[key], strict=False)]
        if key in _XFAIL_ARCHS
        else [],
    )
    for key, info in OPTIMUM_ARCHITECTURES.items()
    if info.library == "transformers"
]


# ---------------------------------------------------------------------------
# Test: _get_onnx_config returns valid config for every architecture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch_key,arch_info", TRANSFORMERS_ARCHS)
def test_get_onnx_config_for_all_architectures(arch_key, arch_info):
    """Verify _get_onnx_config works for every registered transformers architecture."""
    from transformers import AutoConfig

    from winml.modelkit.export.io import _get_onnx_config

    # Handle compound keys like "clip:sentence_transformers" -> extract model_type
    model_type = arch_key.split(":")[0] if ":" in arch_key else arch_key
    first_task = arch_info.tasks[0]

    try:
        hf_config = AutoConfig.for_model(model_type)
    except (KeyError, ValueError):
        pytest.skip(f"AutoConfig.for_model('{model_type}') not available")

    onnx_config = _get_onnx_config(
        model_type,
        first_task,
        hf_config,
        library_name=arch_info.library,
    )
    assert hasattr(onnx_config, "inputs")
    assert hasattr(onnx_config, "outputs")
    assert len(onnx_config.inputs) > 0
