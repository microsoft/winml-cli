# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from winml.modelkit.config import WinMLBuildConfig


REPO_ROOT = Path(__file__).resolve().parents[3]

MXBAI_XSMALL_FP16_BOUNDARY_CASTS = [
    "InsertedPrecisionFreeCast_/deberta/embeddings/LayerNorm/LayerNormalization_output_0",
    *[
        f"InsertedPrecisionFreeCast_/deberta/encoder/layer.{layer}/attention/self/{boundary}_output_0"
        for layer in range(12)
        for boundary in (
            "Transpose_3",
            "Reshape_1",
            "Reshape_3",
            "Transpose_8",
            "Reshape_5",
            "Reshape_13",
        )
    ],
    "InsertedPrecisionFreeCast_/pooler/Gather_output_0",
]
MXBAI_XSMALL_EMBEDDINGS_INT32_CAST = "InsertedPrecisionFreeCast_/deberta/embeddings/Cast_output_0"

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

    if rec["quant_mode"] is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        assert config.quant.mode == rec["quant_mode"]


def test_mxbai_xsmall_reranking_recipes() -> None:
    recipe_root = (
        REPO_ROOT / "examples" / "recipes" / "mixedbread-ai_mxbai-rerank-xsmall-v1" / "cpu" / "cpu"
    )
    fp32_data = json.loads((recipe_root / "reranking_fp32_config.json").read_text(encoding="utf-8"))
    fp16_data = json.loads((recipe_root / "reranking_fp16_config.json").read_text(encoding="utf-8"))

    assert fp32_data["quant"] is None
    assert fp16_data["quant"]["fp16_nodes_to_exclude"] == MXBAI_XSMALL_FP16_BOUNDARY_CASTS
    assert len(MXBAI_XSMALL_FP16_BOUNDARY_CASTS) == 74
    assert len(set(MXBAI_XSMALL_FP16_BOUNDARY_CASTS)) == 74
    assert MXBAI_XSMALL_EMBEDDINGS_INT32_CAST not in MXBAI_XSMALL_FP16_BOUNDARY_CASTS

    for data in (fp32_data, fp16_data):
        config = WinMLBuildConfig.from_dict(data)
        assert config.loader.task == "reranking"
        assert config.loader.model_class == "DebertaV2ForSequenceClassification"
        assert config.loader.model_type == "deberta-v2"

    fp16_config = WinMLBuildConfig.from_dict(fp16_data)
    assert fp16_config.quant is not None
    assert fp16_config.quant.mode == "fp16"
    assert fp16_config.quant.fp16_nodes_to_exclude == MXBAI_XSMALL_FP16_BOUNDARY_CASTS


@pytest.mark.parametrize(
    "relative_path",
    [
        Path(
            "examples/recipes/typeform_distilbert-base-uncased-mnli/cpu/cpu/"
            "text-classification_fp16_config.json"
        ),
        Path(
            "examples/recipes/audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim/"
            "cpu/cpu/audio-classification_fp16_config.json"
        ),
    ],
    ids=["bert-family", "non-text"],
)
def test_existing_fp16_recipes_default_to_no_node_exclusions(relative_path: Path) -> None:
    data = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))

    assert "fp16_nodes_to_exclude" not in data["quant"]
    config = WinMLBuildConfig.from_dict(data)
    assert config.quant is not None
    assert config.quant.fp16_nodes_to_exclude is None
