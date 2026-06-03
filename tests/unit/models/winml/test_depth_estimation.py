# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ``WinMLModelForDepthEstimation``.

Verifies the depth-estimation forward pass wraps raw ONNX outputs in
a ``DepthEstimatorOutput`` so HF's ``DepthEstimationPipeline`` and
``image_processor.post_process_depth_estimation`` can use attribute access.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import DepthEstimatorOutput

from winml.modelkit.models.winml import (
    TASK_TO_WINML_CLASS,
    WinMLModelForDepthEstimation,
    get_winml_class,
)


# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_task_mapped(self):
        assert (
            TASK_TO_WINML_CLASS["depth-estimation"]
            == "WinMLModelForDepthEstimation"
        )

    def test_get_winml_class_returns_depth_estimation(self):
        cls = get_winml_class(model_type="depth_anything", task="depth-estimation")
        assert cls is WinMLModelForDepthEstimation


# ---------------------------------------------------------------------------
# forward() output wrapping
# ---------------------------------------------------------------------------


def _make_model(onnx_outputs: dict[str, torch.Tensor]) -> WinMLModelForDepthEstimation:
    """Construct a ``WinMLModelForDepthEstimation`` with ``_run_inference`` stubbed."""
    model = object.__new__(WinMLModelForDepthEstimation)
    model._format_inputs = lambda **kw: kw
    model._run_inference = lambda formatted: onnx_outputs
    return model


class TestForward:
    def test_returns_depth_estimator_output(self):
        depth = torch.zeros((1, 518, 518))
        model = _make_model({"predicted_depth": depth})

        out = model.forward(pixel_values=torch.zeros((1, 3, 518, 518)))

        assert isinstance(out, DepthEstimatorOutput)

    def test_predicted_depth_passthrough(self):
        depth = torch.full((1, 32, 32), 5.0)
        model = _make_model({"predicted_depth": depth})

        out = model.forward(pixel_values=torch.zeros((1, 3, 32, 32)))

        assert out.predicted_depth is depth

    def test_attribute_and_dict_access(self):
        """``ModelOutput`` supports both ``.attr`` and ``["key"]`` access."""
        depth = torch.zeros((1, 16, 16))
        model = _make_model({"predicted_depth": depth})

        out = model.forward(pixel_values=torch.zeros((1, 3, 16, 16)))

        assert out.predicted_depth is depth
        assert out["predicted_depth"] is depth

    def test_falls_back_to_first_output_when_name_differs(self):
        """Non-standard output names use the first tensor (architecture-agnostic)."""
        depth = torch.full((1, 8, 8), 2.5)
        model = _make_model({"some_unconventional_name": depth})

        out = model.forward(pixel_values=torch.zeros((1, 3, 8, 8)))

        assert out.predicted_depth is depth
