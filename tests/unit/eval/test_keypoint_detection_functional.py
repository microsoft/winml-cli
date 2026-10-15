# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Functional regression for the local keypoint-detection Eval path."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import PngImagePlugin

from winml.modelkit.eval import WinMLEvaluationConfig, WinMLKeypointDetectionEvaluator
from winml.modelkit.eval.config import DatasetConfig
from winml.modelkit.models import WinMLModelForGenericTask
from winml.modelkit.session.ep_device import EPDeviceTarget
from winml.modelkit.session.ep_registry import WinMLEPRegistry


_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures"
_IMAGE_PAYLOAD = _FIXTURE_DIR / "generated_pose_24x17.png.b64"
_IMAGE_SHA256 = "c73364487e82350382835a236fa9baf8ca2152b7b808d4827459e76b02291d1f"

# The RGB payload is generated for this test with channels
# ((11*x + 3*y) % 256, (5*x + 17*y) % 256, (19*x + 7*y) % 256).
# It has no external media source. These COCO-style metric annotations are also
# generated test data, not official COCO image or annotation content.
_OBJECTS = {
    "bbox": [[5.0, 1.0, 14.0, 14.0]],
    "area": [196.0],
    "keypoints": [[0.0] * 45 + [9.0, 6.0, 2.0] + [15.0, 6.0, 2.0]],
}

_PROCESSOR_CONFIG = {
    "do_affine_transform": True,
    "do_normalize": True,
    "do_rescale": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "VitPoseImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "normalize_factor": 200.0,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 256, "width": 192},
}


def _write_saved_dataset(tmp_path: Path) -> Path:
    image_bytes = base64.b64decode(_IMAGE_PAYLOAD.read_text(encoding="ascii"))
    assert hashlib.sha256(image_bytes).hexdigest() == _IMAGE_SHA256
    image_path = tmp_path / "generated_pose.png"
    image_path.write_bytes(image_bytes)

    features = Features(
        {
            "image": Image(),
            "objects": {
                "keypoints": Sequence(Sequence(Value("float32"))),
                "bbox": Sequence(Sequence(Value("float32"))),
                "area": Sequence(Value("float32")),
            },
        }
    )
    dataset = Dataset.from_list(
        [{"image": str(image_path), "objects": _OBJECTS}],
        features=features,
    )
    dataset_path = tmp_path / "coco_keypoints_val2017_one"
    dataset.save_to_disk(str(dataset_path))
    return dataset_path


def _write_vitpose_processor(tmp_path: Path) -> Path:
    processor_path = tmp_path / "vitpose-processor"
    processor_path.mkdir()
    (processor_path / "preprocessor_config.json").write_text(
        json.dumps(_PROCESSOR_CONFIG), encoding="utf-8"
    )
    return processor_path


def _write_keypoint_model(tmp_path: Path) -> Path:
    pixel_values = onnx.helper.make_tensor_value_info(
        "pixel_values", onnx.TensorProto.FLOAT, [1, 3, 256, 192]
    )
    heatmaps = onnx.helper.make_tensor_value_info(
        "heatmaps", onnx.TensorProto.FLOAT, [1, 17, 64, 48]
    )
    reshape_shape = onnx.helper.make_tensor(
        "reshape_shape", onnx.TensorProto.INT64, [4], [1, 1, 1, 1]
    )
    heatmap_shape = onnx.helper.make_tensor(
        "heatmap_shape", onnx.TensorProto.INT64, [4], [1, 17, 64, 48]
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ReduceMean", ["pixel_values"], ["mean"], axes=[0, 1, 2, 3], keepdims=0
            ),
            onnx.helper.make_node("Reshape", ["mean", "reshape_shape"], ["mean_4d"]),
            onnx.helper.make_node("Expand", ["mean_4d", "heatmap_shape"], ["heatmaps"]),
        ],
        "KeypointEvalFixture",
        [pixel_values],
        [heatmaps],
        [reshape_shape, heatmap_shape],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    model_path = tmp_path / "keypoint_fixture.onnx"
    onnx.save(model, model_path)
    return model_path


def _cpu_model(model_path: Path) -> WinMLModelForGenericTask:
    target = EPDeviceTarget(ep="cpu", device="cpu")
    ep_device = WinMLEPRegistry.instance().auto_device(target)
    return WinMLModelForGenericTask(model_path, ep_device=ep_device)


def test_saved_coco_row_runs_complete_vitpose_onnx_eval_path(tmp_path: Path) -> None:
    dataset_path = _write_saved_dataset(tmp_path)
    processor_path = _write_vitpose_processor(tmp_path)
    model = _cpu_model(_write_keypoint_model(tmp_path))
    config = WinMLEvaluationConfig(
        model_id=str(processor_path),
        task="keypoint-detection",
        dataset=DatasetConfig(
            path=str(dataset_path),
            split="validation",
            samples=1,
            shuffle=False,
        ),
    )
    evaluator = WinMLKeypointDetectionEvaluator(config, model)

    row = evaluator.data[0]
    assert isinstance(row["image"], PngImagePlugin.PngImageFile)
    assert row["image"].mode == "RGB"
    assert len(row["objects"]["bbox"]) == len(row["objects"]["keypoints"]) == 1

    observed: dict[str, Any] = {}
    preprocess = evaluator.pipe.preprocess
    postprocess = evaluator.pipe.post_process_pose_estimation

    def record_preprocess(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        observed["boxes"] = kwargs["boxes"]
        inputs = preprocess(*args, **kwargs)
        observed["pixel_values"] = inputs["pixel_values"]
        return inputs

    def record_postprocess(*args: Any, **kwargs: Any) -> Any:
        observed["heatmaps"] = args[0].heatmaps
        result = postprocess(*args, **kwargs)
        observed["poses"] = result
        return result

    evaluator.pipe.preprocess = record_preprocess
    evaluator.pipe.post_process_pose_estimation = record_postprocess
    result = evaluator.compute()

    np.testing.assert_allclose(observed["boxes"], [_OBJECTS["bbox"]], atol=1e-5)
    pixel_values = observed["pixel_values"]
    assert pixel_values.dtype == torch.float32
    assert tuple(pixel_values.shape) == (1, 3, 256, 192)
    assert torch.isfinite(pixel_values).all()
    assert pixel_values.min() < 0 < pixel_values.max()
    assert model.ep_name == "CPUExecutionProvider"
    assert tuple(observed["heatmaps"].shape) == (1, 17, 64, 48)
    assert torch.isfinite(observed["heatmaps"]).all()

    pose = observed["poses"][0][0]
    assert tuple(pose["keypoints"].shape) == (17, 2)
    assert tuple(pose["scores"].shape) == (17,)
    assert torch.isfinite(pose["keypoints"]).all()
    assert torch.isfinite(pose["scores"]).all()

    assert all(math.isfinite(result[key]) for key in ("map", "map_50", "map_75", "mar"))
    assert result["num_predictions"] == 1
    assert result["num_ground_truths"] == 1
    assert result["num_images"] == 1
