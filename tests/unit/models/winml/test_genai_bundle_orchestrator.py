# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the genai-bundle orchestrator (``build_genai_bundle``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from onnx import TensorProto, helper, save

from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.models.winml import (
    GenaiBundleRecipe,
    GenaiCompanionSpec,
    GenaiTarget,
    GenaiTransformerSpec,
    build_genai_bundle,
)


def _write_tiny_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


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
        supported_targets=(GenaiTarget(ep="qnn", device="npu"),),
        transformer_onnx_passes=(_dummy_pass,),
        max_cache_len=2048,
        prefill_seq_len=64,
        soc_model="60",
    )


def _by_model_type(calls: list[dict], model_type: str) -> dict:
    return next(c for c in calls if c.get("model_type") == model_type)


def _by_model_type_and_task(calls: list[dict], model_type: str, task: str) -> dict:
    return next(c for c in calls if c.get("model_type") == model_type and c.get("task") == task)


def _by_transformer_task(calls: list[dict], task: str) -> dict:
    return _by_model_type_and_task(calls, "T-transformer", task)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    onnx_file = tmp_path / "tiny.onnx"
    _write_tiny_onnx(onnx_file)

    calls: list[dict] = []

    def fake_build_artifact(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        return SimpleNamespace(result=SimpleNamespace(final_onnx_path=onnx_file))

    monkeypatch.setattr(
        WinMLAutoModel, "_build_pretrained_artifact", staticmethod(fake_build_artifact)
    )
    monkeypatch.setattr(
        WinMLAutoModel,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: pytest.fail("runtime wrapper was created")),
    )

    import winml.modelkit.models.winml.composite_model as cm_mod

    class _FakeTransformerComposite:
        _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
            "ctx_sub": "feature-extraction",
            "iter_sub": "text2text-generation",
        }

    monkeypatch.setattr(
        cm_mod,
        "COMPOSITE_MODEL_REGISTRY",
        {("T-transformer", "text-generation"): _FakeTransformerComposite},
    )

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

    ctx = _by_transformer_task(harness["calls"], "feature-extraction")
    it = _by_transformer_task(harness["calls"], "text2text-generation")
    for transformer_call in (ctx, it):
        assert transformer_call["device"] == "npu"
        assert transformer_call["precision"] == "w8a16"
        assert transformer_call["ep"] == "QNNExecutionProvider"
    assert ctx["shape_config"] == {
        "max_cache_len": 2048,
        "seq_len": 64,
    }
    assert it["shape_config"] == {
        "max_cache_len": 2048,
        "seq_len": 1,
    }


def test_components_use_artifact_builds_without_runtime_wrappers(harness):
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"])

    assert all("build_only" not in call for call in harness["calls"])


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
    # ``T-transformer`` has no registered quant finalizer, so a precision
    # override is honored (forwarded to the transformer build).
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="w4a16")
    assert _by_transformer_task(harness["calls"], "feature-extraction")["precision"] == "w4a16"
    assert _by_transformer_task(harness["calls"], "text2text-generation")["precision"] == "w4a16"
    assert _by_model_type(harness["calls"], "T-emb")["precision"] == "fp32"
    assert _by_model_type(harness["calls"], "T-lmh")["precision"] == "w4a32"


def test_precision_override_rejected_when_transformer_finalizer_pinned(harness, monkeypatch):
    """A precision override that conflicts with a finalizer-pinned transformer errors.

    When the transformer's ``model_type`` has a registered quant finalizer, its
    scheme is authoritative — a differing ``--precision`` would be silently
    reverted, so it is rejected up-front instead. Architecture-agnostic: the
    guard keys on the finalizer registry, not on any model name.
    """
    import winml.modelkit.models.winml.genai_bundle as gb

    monkeypatch.setattr(gb, "has_quant_finalizer", lambda model_type: True)
    with pytest.raises(ValueError, match="precision"):
        build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="w4a16")
    # Fail-fast: rejected before any component is built.
    assert harness["calls"] == []


def test_matching_precision_override_allowed_when_finalizer_pinned(harness, monkeypatch):
    """Passing exactly the recipe transformer precision is a no-op, not rejected."""
    import winml.modelkit.models.winml.genai_bundle as gb

    monkeypatch.setattr(gb, "has_quant_finalizer", lambda model_type: True)
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="w8a16")
    assert _by_transformer_task(harness["calls"], "feature-extraction")["precision"] == "w8a16"
    assert _by_transformer_task(harness["calls"], "text2text-generation")["precision"] == "w8a16"


def test_no_precision_override_uses_recipe_default_when_finalizer_pinned(harness, monkeypatch):
    """Omitting a precision override builds at the recipe default even when pinned."""
    import winml.modelkit.models.winml.genai_bundle as gb

    monkeypatch.setattr(gb, "has_quant_finalizer", lambda model_type: True)
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"])
    assert _by_transformer_task(harness["calls"], "feature-extraction")["precision"] == "w8a16"
    assert _by_transformer_task(harness["calls"], "text2text-generation")["precision"] == "w8a16"


def test_auto_precision_override_allowed_when_finalizer_pinned(harness, monkeypatch):
    """``--precision auto`` defers to the recipe scheme, so it is not a conflict.

    ``auto`` resolves to the device default (``w8a16`` on NPU), which equals the
    pinned recipe precision, so a generic ``auto`` invocation must be accepted
    and collapse to the canonical recipe precision — not rejected on the raw
    ``"auto" != "w8a16"`` string comparison.
    """
    import winml.modelkit.models.winml.genai_bundle as gb

    monkeypatch.setattr(gb, "has_quant_finalizer", lambda model_type: True)
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="auto")
    assert _by_transformer_task(harness["calls"], "feature-extraction")["precision"] == "w8a16"
    assert _by_transformer_task(harness["calls"], "text2text-generation")["precision"] == "w8a16"


def test_case_variant_precision_override_allowed_when_finalizer_pinned(harness, monkeypatch):
    """A case variant like ``W8A16`` matches the pinned precision (case-insensitive).

    Downstream precision resolution is case-insensitive, so the guard must not
    reject an equivalent upper/mixed-case override; it collapses to the
    canonical recipe precision.
    """
    import winml.modelkit.models.winml.genai_bundle as gb

    monkeypatch.setattr(gb, "has_quant_finalizer", lambda model_type: True)
    build_genai_bundle("m", harness["tmp_path"] / "b", harness["recipe"], precision="W8A16")
    assert _by_transformer_task(harness["calls"], "feature-extraction")["precision"] == "w8a16"
    assert _by_transformer_task(harness["calls"], "text2text-generation")["precision"] == "w8a16"


def test_length_overrides_flow_to_shapes_and_assembler(harness):
    build_genai_bundle(
        "m", harness["tmp_path"] / "b", harness["recipe"], max_cache_len=1024, prefill_seq_len=32
    )
    ctx = _by_transformer_task(harness["calls"], "feature-extraction")
    it = _by_transformer_task(harness["calls"], "text2text-generation")
    assert ctx["shape_config"] == {
        "max_cache_len": 1024,
        "seq_len": 32,
    }
    assert it["shape_config"] == {
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
