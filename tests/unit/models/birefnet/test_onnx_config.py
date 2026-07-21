# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import torch
from torchvision.ops import deform_conv2d
from transformers import PretrainedConfig

from winml.modelkit.export import generate_dummy_inputs
from winml.modelkit.export.io import _get_onnx_config
from winml.modelkit.models.hf.birefnet import BiRefNetIOConfig, _exportable_deform_conv2d


class BiRefNetTestConfig(PretrainedConfig):
    model_type = "SegformerForSemanticSegmentation"


def test_birefnet_onnx_config_is_registered() -> None:
    config = _get_onnx_config(
        "SegformerForSemanticSegmentation",
        "image-segmentation",
        BiRefNetTestConfig(),
    )

    assert isinstance(config, BiRefNetIOConfig)
    assert list(config.inputs) == ["x"]
    assert list(config.outputs) == ["logits"]
    assert config.outputs["logits"] == {0: "batch_size", 2: "height", 3: "width"}


def test_birefnet_dummy_input_uses_remote_forward_name_and_size() -> None:
    inputs = generate_dummy_inputs(
        "SegformerForSemanticSegmentation",
        "image-segmentation",
        BiRefNetTestConfig(),
    )

    assert list(inputs) == ["x"]
    assert inputs["x"].shape == torch.Size([1, 3, 1024, 1024])
    assert inputs["x"].dtype == torch.float32
    assert inputs["x"].min() >= -2.1179039301310043
    assert inputs["x"].max() < 2.64


def test_birefnet_dummy_input_honors_explicit_shape() -> None:
    inputs = generate_dummy_inputs(
        "SegformerForSemanticSegmentation",
        "image-segmentation",
        BiRefNetTestConfig(),
        height=256,
        width=384,
    )

    assert inputs["x"].shape == torch.Size([1, 3, 256, 384])


def test_exportable_deform_conv_matches_torchvision() -> None:
    torch.manual_seed(7)
    input_tensor = torch.randn(1, 4, 7, 8)
    weight = torch.randn(6, 4, 3, 3)
    bias = torch.randn(6)
    offset = torch.randn(1, 18, 7, 8) * 0.2
    mask = torch.sigmoid(torch.randn(1, 9, 7, 8))

    expected = deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias=bias,
        padding=1,
        mask=mask,
    )
    actual = _exportable_deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias=bias,
        padding=1,
        mask=mask,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_exportable_deform_conv_supports_convolution_and_offset_groups() -> None:
    torch.manual_seed(11)
    input_tensor = torch.randn(1, 4, 6, 5)
    weight = torch.randn(6, 2, 3, 3)
    bias = torch.randn(6)
    offset = torch.randn(1, 36, 6, 5) * 0.2
    mask = torch.sigmoid(torch.randn(1, 18, 6, 5))

    expected = deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias=bias,
        padding=1,
        mask=mask,
    )
    actual = _exportable_deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias=bias,
        padding=1,
        mask=mask,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
