# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ``WinMLModelForKeypointDetection``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from winml.modelkit.models.winml import (
    TASK_TO_WINML_CLASS,
    KeypointDetectionOutput,
    WinMLModelForKeypointDetection,
    get_winml_class,
)


class TestRegistry:
    def test_task_mapped(self) -> None:
        assert (
            TASK_TO_WINML_CLASS["keypoint-detection"]
            == "WinMLModelForKeypointDetection"
        )

    def test_get_winml_class_returns_keypoint_detection(self) -> None:
        cls = get_winml_class(model_type="vitpose", task="keypoint-detection")
        assert cls is WinMLModelForKeypointDetection


def _make_model(
    onnx_outputs: dict[str, torch.Tensor],
    input_names: list[str] | None = None,
) -> WinMLModelForKeypointDetection:
    model = object.__new__(WinMLModelForKeypointDetection)
    model._session = SimpleNamespace(io_config={"input_names": input_names or ["pixel_values"]})
    model._run_inference = lambda formatted: onnx_outputs
    model.formatted = None

    def _format_inputs(**kw):
        model.formatted = kw
        return kw

    model._format_inputs = _format_inputs
    return model


class TestForward:
    def test_returns_keypoint_detection_output(self) -> None:
        heatmaps = torch.zeros((1, 17, 64, 48))
        model = _make_model({"heatmaps": heatmaps})

        out = model.forward(pixel_values=torch.zeros((1, 3, 256, 192)))

        assert isinstance(out, KeypointDetectionOutput)
        assert out.heatmaps is heatmaps
        assert out["heatmaps"] is heatmaps

    def test_falls_back_to_first_output_when_name_differs(self) -> None:
        heatmaps = torch.ones((1, 17, 64, 48))
        model = _make_model({"unusual_output": heatmaps})

        out = model.forward(pixel_values=torch.zeros((1, 3, 256, 192)))

        assert out.heatmaps is heatmaps

    def test_dataset_index_only_passed_when_declared(self) -> None:
        heatmaps = torch.zeros((1, 17, 64, 48))
        dataset_index = torch.tensor([2])
        model = _make_model({"heatmaps": heatmaps}, ["pixel_values", "dataset_index"])

        model.forward(pixel_values=torch.zeros((1, 3, 256, 192)), dataset_index=dataset_index)

        assert model.formatted["dataset_index"] is dataset_index

    def test_dataset_index_ignored_when_not_declared(self) -> None:
        heatmaps = torch.zeros((1, 17, 64, 48))
        model = _make_model({"heatmaps": heatmaps}, ["pixel_values"])

        model.forward(pixel_values=torch.zeros((1, 3, 256, 192)), dataset_index=torch.tensor([2]))

        assert "dataset_index" not in model.formatted
