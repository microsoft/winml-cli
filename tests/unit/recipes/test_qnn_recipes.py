# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from winml.modelkit.config import WinMLBuildConfig


REPO_ROOT = Path(__file__).resolve().parents[3]

recipes = [
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "facebook_dinov2-base"
        / "qnn"
        / "npu"
        / "image-feature-extraction_w8a16_config.json",
        "loader_task": "image-feature-extraction",
        "optim_key": "bias_softmax_fusion",
        "optim_value": True,
        "expected_opset": 21,
        "quant_expected": {"weight_type": "uint8", "activation_type": "uint16"},
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "facebook_dinov2-small"
        / "qnn"
        / "npu"
        / "image-feature-extraction_w8a16_config.json",
        "loader_task": "image-feature-extraction",
        "optim_key": "bias_softmax_fusion",
        "optim_value": True,
        "expected_opset": 21,
        "quant_expected": {"weight_type": "uint8", "activation_type": "uint16"},
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "microsoft_swinv2-tiny-patch4-window16-256"
        / "qnn"
        / "npu"
        / "image-classification_fp16_config.json",
        "loader_task": "image-classification",
        "optim_key": "matmul_transpose_fusion",
        "optim_value": True,
        "expected_opset": 21,
        "quant_expected": None,
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim"
        / "qnn"
        / "npu"
        / "audio-classification_w8a16_config.json",
        "loader_task": "audio-classification",
        "loader_class": "EmotionModel",
        "loader_model_type": "wav2vec2_emotion_regression",
        "expected_opset": 17,
        "optim_key": "trim_split_grouped_conv",
        "optim_value": True,
        "quant_expected": {
            "mode": "static",
            "samples": 10,
            "calibration_method": "minmax",
            "weight_type": "uint8",
            "activation_type": "uint16",
            "per_channel": False,
            "symmetric": False,
            "save_calibration": False,
            "distribution": "uniform",
            "task": "audio-classification",
            "model_id": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        },
        "compile_expected": {
            "execution_provider": "qnn",
            "provider_options": {
                "device_type": "NPU",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
            },
            "enable_ep_context": True,
            "embed_context": False,
            "compiler": "ort",
            "device": "npu",
            "validate": True,
        },
    },
]


@pytest.mark.parametrize(
    "rec",
    recipes,
    ids=["dinov2-base", "dinov2-small", "swinv2-tiny", "audeering-wav2vec2-emotion-w8a16"],
)
def test_qnn_recipes(rec):
    path: Path = rec["path"]
    assert path.exists(), f"Recipe file missing: {path}"

    data = json.loads(path.read_text(encoding="utf-8"))

    # Construct the validated config from the recipe dict
    config = WinMLBuildConfig.from_dict(data)

    # export.opset_version matches expected
    assert config.export is not None
    assert config.export.opset_version == rec["expected_opset"]

    # loader.task exact
    assert config.loader.task == rec["loader_task"]

    # optional loader fields
    if rec.get("loader_class") is not None:
        assert config.loader.model_class == rec["loader_class"]
    if rec.get("loader_model_type") is not None:
        assert config.loader.model_type == rec["loader_model_type"]

    # optim key/value
    # config.optim supports dict-like access in the approved API
    assert rec["optim_key"] in config.optim
    assert config.optim[rec["optim_key"]] == rec["optim_value"]

    # quant expectations
    if rec["quant_expected"] is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        # check only the keys provided in the expectation
        for k, v in rec["quant_expected"].items():
            # Use public attributes of WinMLQuantizationConfig directly
            assert getattr(config.quant, k) == v

    # compile expectations (only present for recipes that declare them)
    if rec.get("compile_expected") is not None:
        assert config.compile is not None
        cdict = config.compile.to_dict()
        # Assert top-level expected keys
        for k, v in rec["compile_expected"].items():
            if k == "provider_options":
                # Ensure nested provider options exactly match declared values
                for pk, pv in v.items():
                    assert cdict.get("provider_options", {}).get(pk) == pv
            else:
                assert cdict.get(k) == v
