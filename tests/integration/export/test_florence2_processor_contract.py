# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Native Florence-2 processor contract checks for the exported encoder."""

from __future__ import annotations

import gc
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from click.testing import CliRunner
from PIL import Image
from transformers import AutoConfig

from winml.modelkit.commands.build import build
from winml.modelkit.eval.metrics.tensor_similarity import TensorSimilarityMetric
from winml.modelkit.export import resolve_io_specs
from winml.modelkit.models.hf.florence2 import (
    _load_native_combined_processor,
    _NativeFlorence2ForConditionalGeneration,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


_MODEL_ID = "microsoft/Florence-2-base"
_RECIPE_DIR = Path("examples/recipes/microsoft_Florence-2-base")
_ARTIFACT_ROOT = Path("temp/microsoft_Florence-2-base/processor-contract")

pytestmark = [pytest.mark.integration, pytest.mark.network, pytest.mark.slow]


class FlorenceArtifacts(NamedTuple):
    """Fresh model and component artifacts owned by the module fixture."""

    model_dir: Path
    encoder_onnx_path: Path
    encoder_export_path: Path
    decoder_onnx_path: Path


def _build_recipe_component(recipe_path: Path, model_dir: Path, output_dir: Path) -> Path:
    """Build one recipe component through the public precision-variant command path."""
    result = CliRunner().invoke(
        build,
        [
            "--config",
            str(recipe_path),
            "--model",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--precision",
            "fp16",
            "--no-compile",
            "--rebuild",
        ],
        obj={"debug": False},
    )
    assert result.exit_code == 0, result.output
    onnx_path = output_dir / "model.onnx"
    assert onnx_path.is_file()
    return onnx_path


@pytest.fixture(scope="module")
def florence_artifacts() -> Iterator[FlorenceArtifacts]:
    """Download and build component artifacts once in fixture-owned storage."""
    from huggingface_hub import snapshot_download

    root = _ARTIFACT_ROOT
    if root.exists():
        rmtree(root)
    root.mkdir(parents=True)
    model_dir = root / "model"
    build_dir = root / "build"
    try:
        snapshot_download(
            _MODEL_ID,
            local_dir=model_dir,
            cache_dir=root / "hf-cache",
        )

        encoder_output_dir = build_dir / "encoder"
        encoder_onnx_path = _build_recipe_component(
            _RECIPE_DIR / "image-to-text_fp16_config_encoder.json",
            model_dir,
            encoder_output_dir,
        )
        decoder_onnx_path = _build_recipe_component(
            _RECIPE_DIR / "image-to-text_fp16_config_decoder.json",
            model_dir,
            build_dir / "decoder",
        )
        yield FlorenceArtifacts(
            model_dir=model_dir,
            encoder_onnx_path=encoder_onnx_path,
            encoder_export_path=encoder_output_dir / "export.onnx",
            decoder_onnx_path=decoder_onnx_path,
        )
    finally:
        gc.collect()
        rmtree(root)
        if not any(root.parent.iterdir()):
            root.parent.rmdir()


def _session_inputs(session: ort.InferenceSession, batch: dict[str, torch.Tensor]) -> dict:
    input_names = {input_.name for input_ in session.get_inputs()}
    return {
        name: value.detach().cpu().numpy() for name, value in batch.items() if name in input_names
    }


def test_native_model_loads_the_complete_checkpoint(florence_artifacts: FlorenceArtifacts) -> None:
    """Independent native loads must not retain random initialized parameters."""
    first = _NativeFlorence2ForConditionalGeneration.from_pretrained(
        florence_artifacts.model_dir,
        trust_remote_code=True,
    )
    second = _NativeFlorence2ForConditionalGeneration.from_pretrained(
        florence_artifacts.model_dir,
        trust_remote_code=True,
    )

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(first_state[key], second_state[key]), key


def test_native_processor_matches_exported_encoder_contract(
    florence_artifacts: FlorenceArtifacts,
) -> None:
    config = AutoConfig.from_pretrained(florence_artifacts.model_dir)
    processor = _load_native_combined_processor(
        str(florence_artifacts.model_dir),
        trust_remote_code=True,
    )
    batch = processor(
        text="<CAPTION>",
        images=Image.new("RGB", (768, 768)),
        return_tensors="pt",
    )
    specs = resolve_io_specs("florence2", "feature-extraction", config)
    shapes = dict(zip(specs["input_names"], specs["input_shapes"], strict=True))

    assert set(batch) == set(shapes)
    for name, shape in shapes.items():
        assert tuple(batch[name].shape) == shape
    assert tuple(batch["input_ids"].shape) == (1, 8)


def test_native_padded_prompts_match_exported_encoder(
    florence_artifacts: FlorenceArtifacts,
) -> None:
    processor = _load_native_combined_processor(
        str(florence_artifacts.model_dir),
        trust_remote_code=True,
    )
    batch = processor(
        text=["<CAPTION>", "<DETAILED_CAPTION>"],
        images=[Image.new("RGB", (768, 768)), Image.new("RGB", (768, 768))],
        padding=True,
        return_tensors="pt",
    )
    assert torch.any(batch["attention_mask"] == 0)
    session = ort.InferenceSession(
        str(florence_artifacts.encoder_export_path), providers=["CPUExecutionProvider"]
    )
    assert {input_.name for input_ in session.get_inputs()} == set(batch)
    exported = session.run(None, _session_inputs(session, batch))[0]
    optimized_session = ort.InferenceSession(
        str(florence_artifacts.encoder_onnx_path), providers=["CPUExecutionProvider"]
    )
    optimized = optimized_session.run(None, _session_inputs(optimized_session, batch))[0]

    optimized_model = onnx.load(florence_artifacts.encoder_onnx_path, load_external_data=False)
    assert any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in optimized_model.graph.initializer
    )
    assert optimized.shape == exported.shape
    assert optimized.dtype == exported.dtype
    assert np.isfinite(optimized).all()

    raw_metrics = TensorSimilarityMetric()
    raw_metrics.update(optimized, exported)
    metrics = raw_metrics.compute()
    assert {"cosine_similarity_mean", "max_abs_diff_mean"} <= metrics.keys()


def test_decoder_accepts_real_caption_encoder_states(florence_artifacts: FlorenceArtifacts) -> None:
    processor = _load_native_combined_processor(
        str(florence_artifacts.model_dir),
        trust_remote_code=True,
    )
    batch = processor(
        text="<CAPTION>",
        images=Image.new("RGB", (768, 768)),
        return_tensors="pt",
    )
    encoder_session = ort.InferenceSession(
        str(florence_artifacts.encoder_onnx_path), providers=["CPUExecutionProvider"]
    )
    encoder_hidden_states = encoder_session.run(
        None, _session_inputs(encoder_session, batch)
    )[0]
    decoder_session = ort.InferenceSession(
        str(florence_artifacts.decoder_onnx_path), providers=["CPUExecutionProvider"]
    )
    decoder_inputs = {
        "decoder_input_ids": torch.zeros((1, 1), dtype=torch.int32).numpy(),
        "encoder_hidden_states": encoder_hidden_states,
        "decoder_attention_mask": torch.ones((1, 1024), dtype=torch.long).numpy(),
        "cache_position": torch.zeros((1,), dtype=torch.long).numpy(),
    }
    for index in range(6):
        decoder_inputs[f"past_{index}_key"] = torch.zeros((1, 12, 1024, 64)).numpy()
        decoder_inputs[f"past_{index}_value"] = torch.zeros((1, 12, 1024, 64)).numpy()

    outputs = decoder_session.run(None, decoder_inputs)

    assert encoder_hidden_states.shape == (1, 585, 768)
    assert outputs[0].shape[:2] == (1, 1)
