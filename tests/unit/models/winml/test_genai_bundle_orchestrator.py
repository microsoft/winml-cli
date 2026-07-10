# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the genai-bundle orchestrator (``build_genai_bundle``).

Wiring-only: ``WinMLAutoModel.from_pretrained`` and the recipe assembler are
stubbed so no model is downloaded.  The tests verify the orchestrator maps
recipe data + caller overrides onto the component builders and the assembler,
and that it stays architecture-agnostic (a synthetic, non-Qwen recipe is used).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from onnx import TensorProto, helper, save

from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.models.winml import (
    GenaiBundleRecipe,
    GenaiCompanionSpec,
    GenaiTransformerSpec,
    build_genai_bundle,
)


def _write_tiny_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


class _StubModel:
    def __init__(self, onnx_path: Path, sub_models: dict | None = None) -> None:
        self.onnx_path = str(onnx_path)
        self.sub_models = sub_models or {}


def _dummy_pass(model):
    return model


def _make_recipe(assemble) -> GenaiBundleRecipe:
    return GenaiBundleRecipe(
        family="testfam",
        transformer=GenaiTransformerSpec(
            model_type="T-transformer",
            task="text-generation",
            precision="w8a16",
            context_sub_model="ctx_sub",
            iterator_sub_model="iter_sub",
        ),
        companions=(
            GenaiCompanionSpec(
                role="embeddings", model_type="T-emb", task="feature-extraction", precision="fp32"
            ),
            GenaiCompanionSpec(
                role="lm_head", model_type="T-lmh", task="feature-extraction", precision="w4a32"
            ),
        ),
        assemble=assemble,
        transformer_onnx_passes=(_dummy_pass,),
        max_cache_len=2048,
        prefill_seq_len=64,
        soc_model="60",
    )


def _by_model_type(calls: list[dict], model_type: str) -> dict:
    return next(c for c in calls if c.get("model_type") == model_type)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    onnx_file = tmp_path / "tiny.onnx"
    _write_tiny_onnx(onnx_file)

    calls: list[dict] = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        if kwargs.get("model_type") == "T-transformer":
            return _StubModel(
                onnx_file,
                sub_models={"ctx_sub": _StubModel(onnx_file), "iter_sub": _StubModel(onnx_file)},
            )
        return _StubModel(onnx_file)

    monkeypatch.setattr(WinMLAutoModel, "from_pretrained", staticmethod(fake_from_pretrained))

    assemble_kwargs: dict = {}

    def fake_assemble(output_dir, **kwargs):
        assemble_kwargs.clear()
        assemble_kwargs.update(kwargs)
        assemble_kwargs["output_dir"] = output_dir
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = out / "genai_config.json"
        cfg.write_text("{}")
        return cfg

    return {
        "onnx_file": onnx_file,
        "calls": calls,
        "assemble_kwargs": assemble_kwargs,
        "recipe": _make_recipe(fake_assemble),
        "tmp_path": tmp_path,
    }


def test_returns_genai_config_path(harness):
    out = harness["tmp_path"] / "bundle"
    result = build_genai_bundle("some/model", out, harness["recipe"], ep="qnn", device="npu")
    assert result == out / "genai_config.json"
    assert result.exists()


def test_transformer_built_with_recipe_defaults(harness):
    out = harness["tmp_path"] / "bundle"
    build_genai_bundle("some/model", out, harness["recipe"], ep="qnn", device="npu")

    t = _by_model_type(harness["calls"], "T-transformer")
    assert t["task"] == "text-generation"
    assert t["device"] == "npu"
    assert t["precision"] == "w8a16"
    assert t["ep"] == "QNNExecutionProvider"  # normalized from short "qnn"
    assert t["sub_model_kwargs"]["ctx_sub"]["shape_config"] == {
        "max_cache_len": 2048,
        "seq_len": 64,
    }
    assert t["sub_model_kwargs"]["iter_sub"]["shape_config"] == {
        "max_cache_len": 2048,
        "seq_len": 1,
    }


def test_companions_built_on_cpu(harness):
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"])
    emb = _by_model_type(harness["calls"], "T-emb")
    lmh = _by_model_type(harness["calls"], "T-lmh")
    for companion in (emb, lmh):
        assert companion["device"] == "cpu"
        assert companion["ep"] == "CPUExecutionProvider"
        assert companion["task"] == "feature-extraction"
    assert emb["precision"] == "fp32"
    assert lmh["precision"] == "w4a32"


def test_assembler_receives_paths_ep_and_passes(harness):
    onnx_file = harness["onnx_file"]
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], ep="qnn", device="npu")
    ak = harness["assemble_kwargs"]
    assert Path(ak["context_onnx"]) == onnx_file
    assert Path(ak["iterator_onnx"]) == onnx_file
    assert Path(ak["embeddings_src"]) == onnx_file
    assert Path(ak["lm_head_src"]) == onnx_file
    assert ak["ep"] == "qnn"  # short token forwarded verbatim to the assembler
    assert ak["soc_model"] == "60"
    assert ak["model_id"] == "m"
    assert ak["max_cache_len"] == 2048
    assert ak["prefill_seq_len"] == 64
    assert ak["transformer_onnx_passes"] == [_dummy_pass]


def test_precision_override_only_affects_transformer(harness):
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="w4a16")
    assert _by_model_type(harness["calls"], "T-transformer")["precision"] == "w4a16"
    assert _by_model_type(harness["calls"], "T-emb")["precision"] == "fp32"
    assert _by_model_type(harness["calls"], "T-lmh")["precision"] == "w4a32"


def test_length_overrides_flow_to_shapes_and_assembler(harness):
    build_genai_bundle(
        "m", harness["tmp_path"] / "b", harness["recipe"], max_cache_len=1024, prefill_seq_len=32
    )
    t = _by_model_type(harness["calls"], "T-transformer")
    assert t["sub_model_kwargs"]["ctx_sub"]["shape_config"] == {
        "max_cache_len": 1024,
        "seq_len": 32,
    }
    assert t["sub_model_kwargs"]["iter_sub"]["shape_config"] == {
        "max_cache_len": 1024,
        "seq_len": 1,
    }
    assert harness["assemble_kwargs"]["max_cache_len"] == 1024
    assert harness["assemble_kwargs"]["prefill_seq_len"] == 32


def test_companion_override_skips_build(harness):
    prebuilt = harness["tmp_path"] / "prebuilt_emb.onnx"
    _write_tiny_onnx(prebuilt)
    build_genai_bundle(
        "m",
        harness["tmp_path"] / "b",
        harness["recipe"],
        companion_overrides={"embeddings": prebuilt},
    )
    # embeddings companion NOT built; lm_head still built.
    assert all(c.get("model_type") != "T-emb" for c in harness["calls"])
    assert any(c.get("model_type") == "T-lmh" for c in harness["calls"])
    assert Path(harness["assemble_kwargs"]["embeddings_src"]) == prebuilt


def test_emit_receives_progress(harness):
    lines: list[str] = []
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], emit=lines.append)
    joined = "\n".join(lines)
    assert "assembling bundle" in joined
    assert "genai_config.json" in joined
