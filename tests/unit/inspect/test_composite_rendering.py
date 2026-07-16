# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for inspect composite (Variant 1) rendering — offline, no network.

Covers the registry-driven ``resolve_composite_info`` and the Variant 1
table/JSON output: composites surface their pipeline tasks (Task row + Export
row + Composite Pipeline panel), non-composites are unchanged, and the JSON
``task`` field stays the granular machine task (additive ``pipeline_tasks`` /
``composite`` fields).
"""

from __future__ import annotations

import io
import json

from rich.console import Console
from transformers import AutoConfig, ViTConfig

from winml.modelkit.inspect import (
    CompositeInfo,
    ExporterInfo,
    InspectResult,
    IOConfigInfo,
    LoaderInfo,
    SupportLevel,
    WinMLInfo,
    resolve_composite_info,
)
from winml.modelkit.inspect.formatter import output_json, output_table
from winml.modelkit.loader import resolve_task


_BART_COMPONENTS = {"encoder": "feature-extraction", "decoder": "text2text-generation"}
_BART_PIPELINES = ["summarization", "table-question-answering"]


def _make_result(
    composite: CompositeInfo | None = None, task: str = "text2text-generation"
) -> InspectResult:
    """Minimal real InspectResult (no mocks) for formatter tests."""
    return InspectResult(
        model_id="microsoft/tapex-base-finetuned-wikisql",
        model_type="bart",
        architectures=["BartForConditionalGeneration"],
        task=task,
        task_source="tasks-manager",
        loader=LoaderInfo("BartDecoderWrapper", "MODEL_CLASS_MAPPING", SupportLevel.SUPPORTED),
        exporter=ExporterInfo("BartDecoderIOConfig", "TasksManager", SupportLevel.DEFAULT),
        winml=WinMLInfo("WinMLModelForGenericTask", "Generic", SupportLevel.GENERIC),
        overall_support=SupportLevel.DEFAULT,
        composite=composite,
    )


def _render(result: InspectResult) -> str:
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    output_table(console, result)
    return console.file.getvalue()


# --- resolve_composite_info (gated on the resolver's detected composite) -----


def test_resolve_composite_info_uses_detected_components():
    info = resolve_composite_info("bart", _BART_COMPONENTS)
    assert info is not None
    assert info.pipeline_tasks == _BART_PIPELINES  # higher-level pipelines from registry
    assert info.components == _BART_COMPONENTS  # breakdown from the resolver


def test_resolve_composite_info_none_without_detected_composite():
    # Gate: no composite view unless the *resolved* task bridged to a composite,
    # even for a model_type that merely could serve one (CLIP @ feature-extraction).
    assert resolve_composite_info("bart", None) is None
    assert resolve_composite_info("clip", None) is None
    assert resolve_composite_info("bert", None) is None


def test_resolve_composite_info_none_when_pipeline_tasks_empty(monkeypatch):
    # Registry-divergence guard: a detected composite whose model_type yields no
    # pipeline tasks must NOT surface a broken "[composite]" Task row -> return None.
    # (composite_pipeline_tasks is resolved from the loader package at call time.)
    monkeypatch.setattr("winml.modelkit.loader.composite_pipeline_tasks", lambda mt: [])
    assert resolve_composite_info("bart", _BART_COMPONENTS) is None


def test_resolve_composite_info_empty_components_not_suppressed(monkeypatch):
    # `{}` is not None: an empty dict means the composite path WAS taken (detection
    # succeeded) but yielded no component breakdown — it must NOT be suppressed the
    # way None is. The pipeline_tasks guard, not the falsiness of components, governs.
    monkeypatch.setattr(
        "winml.modelkit.loader.composite_pipeline_tasks", lambda mt: ["summarization"]
    )
    info = resolve_composite_info("bart", {})
    assert info is not None
    assert info.components == {}
    assert info.pipeline_tasks == ["summarization"]


# --- table rendering (Variant 1: pipeline-led) ------------------------------


def test_composite_table_is_pipeline_led():
    out = _render(_make_result(CompositeInfo(_BART_PIPELINES, _BART_COMPONENTS)))
    # Task row surfaces the pipeline tasks + a [composite] tag
    assert "summarization · table-question-answering" in out
    assert "[composite]" in out
    # Export row shows the component -> export-task breakdown
    assert "encoder: feature-extraction" in out
    assert "decoder: text2text-generation" in out
    # Dedicated Composite Pipeline panel is rendered
    assert "Composite Pipeline" in out


def test_non_composite_table_unchanged():
    out = _render(_make_result(composite=None, task="fill-mask"))
    assert "Composite Pipeline" not in out  # no composite panel
    assert "[composite]" not in out  # no composite tag on the Task row
    assert "encoder: feature-extraction" not in out  # no Export row breakdown
    assert "fill-mask" in out  # granular task shown as-is


def test_composite_with_empty_pipeline_tasks_renders_plain_task():
    # output_table is public: a directly-constructed CompositeInfo with empty
    # pipeline_tasks must fall back to the plain Task row (no bare " [composite]")
    # and skip the Composite Pipeline panel — robust independent of the resolver guard.
    out = _render(_make_result(CompositeInfo([], _BART_COMPONENTS), task="text2text-generation"))
    assert "[composite]" not in out
    assert "Composite Pipeline" not in out
    assert "text2text-generation" in out  # plain task shown as-is


# --- JSON output (additive; machine `task` unchanged) -----------------------


def test_composite_json_is_additive():
    data = json.loads(output_json(_make_result(CompositeInfo(_BART_PIPELINES, _BART_COMPONENTS))))
    # The machine contract `task` stays the granular export task.
    assert data["task"] == "text2text-generation"
    assert data["pipeline_tasks"] == _BART_PIPELINES
    assert data["composite"] == {
        "pipeline_tasks": _BART_PIPELINES,
        "components": _BART_COMPONENTS,
    }


def test_non_composite_json_has_null_composite():
    data = json.loads(output_json(_make_result(composite=None, task="fill-mask")))
    assert data["task"] == "fill-mask"
    assert data["pipeline_tasks"] is None
    assert data["composite"] is None


def test_json_serializes_nested_pretrained_config_in_io_extra():
    result = _make_result(CompositeInfo(["image-to-text"], _BART_COMPONENTS))
    result.io_config = IOConfigInfo(extra={"encoder": ViTConfig(hidden_size=768)})

    data = json.loads(output_json(result))

    assert data["io_config"]["extra"]["encoder"]["model_type"] == "vit"
    assert data["io_config"]["extra"]["encoder"]["hidden_size"] == 768


# --- resolver -> render seam (offline integration, real resolve_task) --------


def _bart_config() -> AutoConfig:
    cfg = AutoConfig.for_model("bart")
    cfg.architectures = ["BartForConditionalGeneration"]
    cfg.is_encoder_decoder = True
    return cfg


def test_seam_autodetected_bart_renders_composite():
    """End-to-end (offline): resolve_task -> resolve_composite_info -> render.

    Unlike the formatter tests above (hardcoded CompositeInfo), this stitches the
    REAL resolver output through, so a model_type-key mismatch — gate set but
    composite_pipeline_tasks() empty -> orphaned `[composite]` tag — would be caught.
    """
    resolution = resolve_task(_bart_config())
    assert resolution.composite is not None  # auto-detect bridges seq2seq to a composite
    info = resolve_composite_info("bart", resolution.composite)
    assert info is not None
    assert info.pipeline_tasks == _BART_PIPELINES  # non-empty -> no orphaned tag
    out = _render(_make_result(info))
    assert "summarization · table-question-answering" in out
    assert "[composite]" in out
    assert "Composite Pipeline" in out


def test_seam_explicit_task_pins_no_composite():
    """Gate contract: an explicit *granular* --task (text2text-generation) pins
    composite=None (USER_TASK, single-decoder export), so no composite view — even for a
    genuine seq2seq. Pins the design the inline comment documents, so a revert to a
    registry-fallback (which would show it) can't slip through silently. Contrast with
    test_seam_explicit_composite_task_shows_composite below.
    """
    resolution = resolve_task(_bart_config(), task="text2text-generation")
    assert resolution.composite is None
    assert resolve_composite_info("bart", resolution.composite) is None
    out = _render(_make_result(composite=None, task="text2text-generation"))
    assert "Composite Pipeline" not in out
    assert "[composite]" not in out


def test_seam_explicit_composite_task_shows_composite():
    """An explicit --task naming a composite pipeline task (summarization) now populates
    composite, so inspect renders the Composite Pipeline panel — the #1069 behavior."""
    resolution = resolve_task(_bart_config(), task="summarization")
    assert resolution.composite == _BART_COMPONENTS
    info = resolve_composite_info("bart", resolution.composite)
    assert info is not None
    out = _render(_make_result(info, task="summarization"))
    assert "Composite Pipeline" in out
    assert "[composite]" in out
