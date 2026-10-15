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
        / "dima806_music_genres_classification"
        / "cpu"
        / "cpu"
        / "audio-classification_fp32_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "AutoModelForAudioClassification",
        "loader_model_type": "wav2vec2",
        "opset_version": 17,
        "quant_mode": None,
        "transformers_attention": "eager",
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "dima806_music_genres_classification"
        / "cpu"
        / "cpu"
        / "audio-classification_fp16_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "AutoModelForAudioClassification",
        "loader_model_type": "wav2vec2",
        "opset_version": 17,
        "quant_mode": "fp16",
        "transformers_attention": "eager",
    },
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
        "quant_mode": None,
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
        "quant_mode": "fp16",
    },
]


@pytest.mark.parametrize(
    "rec",
    recipes,
    ids=[
        "dima806-music-genres-fp32",
        "dima806-music-genres-fp16",
        "audeering-wav2vec2-emotion-fp32",
        "audeering-wav2vec2-emotion-fp16",
    ],
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
    assert config.export.input_tensors[0].dtype == "float32"
    if "transformers_attention" in rec:
        assert (
            config.export.compatibility.transformers_attention
            == rec["transformers_attention"]
        )

    # loader routes to the emotion-regression head
    assert config.loader.task == rec["loader_task"]
    assert config.loader.model_class == rec["loader_model_class"]
    assert config.loader.model_type == rec["loader_model_type"]

    if rec["quant_mode"] is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        assert config.quant.mode == rec["quant_mode"]
