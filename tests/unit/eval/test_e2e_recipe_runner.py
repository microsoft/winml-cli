# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for target-aware E2E recipe execution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[3]
_E2E_DIR = _REPO_ROOT / "scripts" / "e2e_eval"
sys.path.insert(0, str(_E2E_DIR))

from utils.recipes import RecipeComponent, RecipeVariant, discover_recipe_variants  # noqa: E402


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("e2e_run_eval", _E2E_DIR / "run_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_recipe_variants_selects_nested_target_and_fp32(tmp_path: Path) -> None:
    model_dir = tmp_path / "org_model"
    cpu_dir = model_dir / "cpu" / "cpu"
    qnn_dir = model_dir / "qnn" / "npu"
    cpu_dir.mkdir(parents=True)
    qnn_dir.mkdir(parents=True)

    for precision in ("fp32", "fp16"):
        for role in ("encoder", "decoder"):
            (cpu_dir / f"image-to-text_{precision}_config_{role}.json").write_text("{}")
    (qnn_dir / "image-to-text_fp16_config_encoder.json").write_text("{}")

    variants = discover_recipe_variants(
        tmp_path,
        "org/model",
        "image-to-text",
        ep="CPUExecutionProvider",
        device="cpu",
    )

    assert [variant.precision for variant in variants] == ["fp32", "fp16"]
    assert all(len(variant.components) == 2 for variant in variants)
    assert all(
        component.path.parent == cpu_dir
        for variant in variants
        for component in variant.components
    )

    auto_dir = tmp_path / "org_auto-model" / "cpu" / "cpu"
    auto_dir.mkdir(parents=True)
    for precision in ("fp32", "fp16"):
        for role in ("encoder", "decoder"):
            (auto_dir / f"image-to-text_{precision}_config_{role}.json").write_text("{}")

    auto_variants = discover_recipe_variants(tmp_path, "org/auto-model", "image-to-text")
    assert [variant.precision for variant in auto_variants] == ["fp32", "fp16"]
    assert {(variant.ep, variant.device) for variant in auto_variants} == {("cpu", "cpu")}

    alias_variants = discover_recipe_variants(
        tmp_path,
        "org/model",
        "image-to-text",
        ep="CPUExecutionProvider",
        device="cpu",
    )
    assert [variant.precision for variant in alias_variants] == ["fp32", "fp16"]


def test_recipe_build_passes_fp16_precision_to_every_component(
    tmp_path: Path, monkeypatch
) -> None:
    run_eval = _load_run_eval()
    encoder = tmp_path / "image-to-text_fp16_config_encoder.json"
    decoder = tmp_path / "image-to-text_fp16_config_decoder.json"
    encoder.write_text("{}")
    decoder.write_text("{}")
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"onnx")
    commands: list[list[str]] = []

    def fake_run(args: list[str], timeout: int) -> dict:
        commands.append(args)
        return {"stdout": "", "stderr": "", "exit_code": 0, "elapsed": 0.0, "timeout": False}

    monkeypatch.setattr(run_eval, "_run_subprocess", fake_run)
    monkeypatch.setattr(run_eval, "_extract_onnx_path", lambda *args: str(artifact))
    variant = RecipeVariant(
        precision="fp16",
        components=[
            RecipeComponent(encoder, "encoder"),
            RecipeComponent(decoder, "decoder"),
        ],
    )

    result = run_eval._run_recipe_build(
        SimpleNamespace(hf_id="org/model", task="image-to-text"),
        variant,
        timeout=30,
        model_dir=tmp_path / "output",
        ep="cpu",
        device="cpu",
    )

    assert result["success"] is True
    assert len(commands) == 2
    assert all(command[-2:] == ["--precision", "fp16"] for command in commands)
    assert all("--ep" in command and "--device" in command for command in commands)
