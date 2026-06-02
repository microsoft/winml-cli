# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for timm library routing during OnnxConfig resolution.

timm checkpoints load through transformers' TimmWrapper (model_type=
"timm_wrapper"), but Optimum registers their OnnxConfig (TimmDefaultOnnxConfig)
only under library_name="timm". ``resolve_optimum_library`` reroutes the lookup
so ``resolve_io_specs`` / ``_get_onnx_config`` resolve it under the default
"transformers" library, with no --library flag. See loader/task.py and
export/io.py.
"""

from __future__ import annotations

import pytest

# Trigger OnnxConfig registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import resolve_io_specs
from winml.modelkit.export.io import _get_onnx_config  # internal: routing under test


@pytest.fixture(scope="module")
def timm_wrapper_config():
    """Minimal offline TimmWrapperConfig (no hub download)."""
    from transformers import TimmWrapperConfig

    return TimmWrapperConfig(num_labels=10)


class TestTimmLibraryRouting:
    """timm_wrapper resolves to Optimum's TimmDefaultOnnxConfig via library routing."""

    def test_get_onnx_config_routes_to_timm_default(self, timm_wrapper_config) -> None:
        """A default (transformers) lookup reroutes to Optimum's TimmDefaultOnnxConfig."""
        from optimum.exporters.onnx.model_configs import TimmDefaultOnnxConfig

        onnx_config = _get_onnx_config("timm_wrapper", "image-classification", timm_wrapper_config)
        assert isinstance(onnx_config, TimmDefaultOnnxConfig), (
            "timm_wrapper did not route to Optimum's timm OnnxConfig; "
            "resolve_optimum_library routing may be inactive."
        )

    def test_io_specs_pixel_values_to_logits(self, timm_wrapper_config) -> None:
        """resolve_io_specs yields the timm image-classifier I/O without a --library flag."""
        specs = resolve_io_specs("timm_wrapper", "image-classification", timm_wrapper_config)
        assert specs["input_names"] == ["pixel_values"]
        assert "logits" in specs["output_names"]

    def test_pixel_values_is_4d_nchw(self, timm_wrapper_config) -> None:
        specs = resolve_io_specs("timm_wrapper", "image-classification", timm_wrapper_config)
        shape = specs["input_shapes"][0]
        assert len(shape) == 4, f"pixel_values should be 4D NCHW, got {shape}"
        assert shape[1] == 3, f"expected 3 channels, got {shape[1]}"
