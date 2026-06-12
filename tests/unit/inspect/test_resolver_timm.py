# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""timm (wrapped-library) resolution in the public `inspect` path.

`inspect_model` resolves task/exporter via `resolve_task` +
`resolver.resolve_exporter` — a separate path from the CLI's `_inspect_model_v2`.
timm checkpoints load as `TimmWrapperConfig` (model_type="timm_wrapper",
architectures=None). Without wrapped-library handling, `resolve_task` mislabels
the task (HF_TASK_DEFAULTS fallback) and `resolve_exporter` hardcodes
library_name="transformers" so the OnnxConfig lookup fails (UNSUPPORTED).

These cover the fix that routes both through the timm library, matching the CLI.
"""

from __future__ import annotations

import pytest

from winml.modelkit.inspect import SupportLevel, resolve_exporter
from winml.modelkit.loader.resolution import TaskSource, resolve_task


@pytest.fixture(scope="module")
def timm_wrapper_config():
    """Minimal offline TimmWrapperConfig (no hub download)."""
    from transformers import TimmWrapperConfig

    return TimmWrapperConfig(num_labels=10)


class TestDetectTaskTimm:
    def test_timm_detects_image_classification(self, timm_wrapper_config) -> None:
        """timm_wrapper (no architectures) resolves to image-classification, not a fallback."""
        r = resolve_task(timm_wrapper_config)
        assert r.task == "image-classification", f"got task={r.task!r} source={r.source!r}"
        assert r.source == TaskSource.WRAPPED_LIBRARY


class TestResolveExporterTimm:
    def test_timm_resolves_optimum_onnx_config(self, timm_wrapper_config) -> None:
        """resolve_exporter routes timm_wrapper to Optimum's timm OnnxConfig + real I/O."""
        info = resolve_exporter(
            "timm_wrapper", "image-classification", hf_config=timm_wrapper_config
        )
        assert info.onnx_config_class == "TimmDefaultOnnxConfig", info.onnx_config_class
        assert info.support_level is not SupportLevel.UNSUPPORTED
        names = [t.name for t in info.input_tensors]
        assert "pixel_values" in names, names
