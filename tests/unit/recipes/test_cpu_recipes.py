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
        / "audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim"
        / "cpu"
        / "cpu"
        / "audio-classification_fp32_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "EmotionModel",
        "loader_model_type": "wav2vec2_emotion_regression",
        "opset_version": 17,
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim"
        / "cpu"
        / "cpu"
        / "audio-classification_fp16_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "EmotionModel",
        "loader_model_type": "wav2vec2_emotion_regression",
        "opset_version": 17,
    },
]


@pytest.mark.parametrize(
    "rec",
    recipes,
    ids=["audeering-wav2vec2-emotion-fp32", "audeering-wav2vec2-emotion-fp16"],
)
def test_cpu_recipes(rec):
    path: Path = rec["path"]
    assert path.exists(), f"Recipe file missing: {path}"

    # EP/device is encoded by folder layout: <model>/<ep>/<device>/<recipe>.json
    assert path.parent.name == "cpu"  # device
    assert path.parent.parent.name == "cpu"  # ep

    data = json.loads(path.read_text(encoding="utf-8"))

    # Construct the validated config from the recipe dict
    config = WinMLBuildConfig.from_dict(data)

    # export.opset_version exact
    assert config.export is not None
    assert config.export.opset_version == rec["opset_version"]

    # loader routes to the emotion-regression head
    assert config.loader.task == rec["loader_task"]
    assert config.loader.model_class == rec["loader_model_class"]
    assert config.loader.model_type == rec["loader_model_type"]

    # float CPU recipe (fp32/fp16): no quantization
    assert config.quant is None
