# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml perf support of composite (multi-session) models.

Composite models (e.g. CLIP/SigLIP dual-encoders) have no single ONNX
session; they orchestrate several sub-models. The perf benchmark must
aggregate their io_configs and time the full ``forward()`` pass rather
than reaching for a single ``_session``.

Regression guard: previously ``PerfBenchmark`` assumed every model exposed
``io_config`` / ``_session`` and raised ``AttributeError`` on composites.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    PerfBenchmark,
    _aggregate_io_config,
    _describe_outputs,
)


def _make_sub_model(
    input_names: list[str],
    input_shapes: list[list[int | None]],
    input_types: list[str],
    output_names: list[str],
    output_shapes: list[list[int | None]],
    *,
    device: str = "GPU",
    ep_name: str = "OpenVINOExecutionProvider",
    precision: str | None = "fp16",
) -> Any:
    """Build a minimal stand-in for a WinMLAutoModel sub-component."""
    io_config = {
        "input_names": input_names,
        "input_shapes": input_shapes,
        "input_types": input_types,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "output_types": ["float32"] * len(output_names),
        "precision": precision,
    }
    compiled: dict[str, bool] = {"compiled": False}

    def _compile() -> None:
        compiled["compiled"] = True

    return SimpleNamespace(
        io_config=io_config,
        device=device,
        ep_name=ep_name,
        _session=SimpleNamespace(compile=_compile),
        _compiled_flag=compiled,
    )


class _FakeComposite:
    """Stand-in for a WinMLCompositeModel (duck-typed via ``sub_models``)."""

    def __init__(self, sub_models: dict[str, Any]) -> None:
        self.sub_models = sub_models
        self.call_log: list[dict[str, np.ndarray]] = []

    def __call__(self, **kwargs: np.ndarray) -> dict[str, np.ndarray]:
        self.call_log.append(kwargs)
        # Mimics a composite's task-level forward() output (e.g. SigLIP):
        # tensors that exist on no single sub-model's ONNX graph.
        return {
            "logits_per_image": np.zeros((1, 1), dtype=np.float32),
            "image_embeds": np.zeros((1, 768), dtype=np.float32),
            "text_embeds": np.zeros((1, 768), dtype=np.float32),
        }


def _siglip_like() -> _FakeComposite:
    image_encoder = _make_sub_model(
        input_names=["pixel_values"],
        input_shapes=[[1, 3, 224, 224]],
        input_types=["float32"],
        output_names=["image_embeds"],
        output_shapes=[[1, 768]],
    )
    text_encoder = _make_sub_model(
        input_names=["input_ids", "attention_mask"],
        input_shapes=[[1, 64], [1, 64]],
        input_types=["int64", "int64"],
        output_names=["text_embeds"],
        output_shapes=[[1, 768]],
    )
    return _FakeComposite({"image-encoder": image_encoder, "text-encoder": text_encoder})


class TestAggregateIoConfig:
    """Unit tests for the io_config union helper."""

    def test_union_dedupes_by_name_preserving_order(self) -> None:
        model = _siglip_like()
        agg = _aggregate_io_config(model.sub_models.values())

        assert agg["input_names"] == ["pixel_values", "input_ids", "attention_mask"]
        assert agg["input_shapes"] == [[1, 3, 224, 224], [1, 64], [1, 64]]
        assert agg["input_types"] == ["float32", "int64", "int64"]
        assert agg["output_names"] == ["image_embeds", "text_embeds"]

    def test_shared_input_name_is_not_duplicated(self) -> None:
        # Both encoders consume "attention_mask" -> it must appear once.
        a = _make_sub_model(
            ["input_ids", "attention_mask"],
            [[1, 8], [1, 8]],
            ["int64", "int64"],
            ["a"],
            [[1, 4]],
        )
        b = _make_sub_model(
            ["attention_mask", "token_type_ids"],
            [[1, 8], [1, 8]],
            ["int64", "int64"],
            ["b"],
            [[1, 4]],
        )
        agg = _aggregate_io_config([a, b])
        assert agg["input_names"] == ["input_ids", "attention_mask", "token_type_ids"]

    def test_precision_taken_from_first_sub_model(self) -> None:
        a = _make_sub_model(["x"], [[1]], ["float32"], ["y"], [[1]], precision="int8")
        b = _make_sub_model(["z"], [[1]], ["float32"], ["w"], [[1]], precision="fp16")
        assert _aggregate_io_config([a, b])["precision"] == "int8"


class TestPerfBenchmarkComposite:
    """PerfBenchmark must transparently handle composite models."""

    def _benchmark(self) -> tuple[PerfBenchmark, _FakeComposite]:
        config = BenchmarkConfig(
            model_id="google/siglip-base-patch16-224",
            task="zero-shot-image-classification",
            device="gpu",
            iterations=3,
            warmup=1,
        )
        bench = PerfBenchmark(config)
        model = _siglip_like()
        bench._model = model  # bypass _load_model (no HF download in unit tests)
        return bench, model

    def test_detects_composite(self) -> None:
        bench, _ = self._benchmark()
        assert bench._is_composite is True

    def test_resolved_io_config_is_aggregated_and_cached(self) -> None:
        bench, _ = self._benchmark()
        io = bench._resolved_io_config()
        assert io["input_names"] == ["pixel_values", "input_ids", "attention_mask"]
        # Cached: second call returns the same object.
        assert bench._resolved_io_config() is io

    def test_compile_compiles_every_sub_session(self) -> None:
        bench, model = self._benchmark()
        bench._compile_model()
        assert all(s._compiled_flag["compiled"] for s in model.sub_models.values())

    def test_generate_inputs_covers_all_sub_model_inputs(self) -> None:
        bench, _ = self._benchmark()
        bench._generate_inputs()
        assert set(bench._inputs) == {"pixel_values", "input_ids", "attention_mask"}
        assert bench._inputs["pixel_values"].shape == (1, 3, 224, 224)
        assert bench._inputs["input_ids"].shape == (1, 64)

    def test_resolved_device_ep_task_from_sub_model(self) -> None:
        bench, _ = self._benchmark()
        assert bench._resolved_device() == "GPU"
        assert bench._resolved_ep() == "OpenVINOExecutionProvider"
        assert bench._resolved_task() == "zero-shot-image-classification"

    def test_simple_benchmark_times_full_forward(self) -> None:
        bench, model = self._benchmark()
        bench._generate_inputs()
        stats = bench._run_benchmark_simple()

        # warmup(1) + iterations(3) == 4 forward() calls; stats excludes warmup.
        assert len(model.call_log) == 4
        assert stats.count == 3
        # forward() received the generated inputs as kwargs.
        assert set(model.call_log[0]) == {"pixel_values", "input_ids", "attention_mask"}

    def test_probe_replaces_outputs_with_real_forward_result(self) -> None:
        # The aggregated view reports the image encoder's raw ONNX outputs;
        # probing must replace them with the composite forward()'s outputs.
        bench, _ = self._benchmark()
        bench._generate_inputs()
        assert bench._resolved_io_config()["output_names"] == ["image_embeds", "text_embeds"]

        bench._probe_composite_outputs()
        io = bench._resolved_io_config()
        assert io["output_names"] == ["logits_per_image", "image_embeds", "text_embeds"]
        assert io["output_shapes"] == [[1, 1], [1, 768], [1, 768]]

    def test_collect_results_reports_probed_outputs(self) -> None:
        bench, _ = self._benchmark()
        bench._generate_inputs()
        bench._probe_composite_outputs()
        stats = bench._run_benchmark_simple()
        result = bench._collect_results(stats)

        assert result.input_names == ["pixel_values", "input_ids", "attention_mask"]
        # Real composite outputs, not the deduped sub-model ONNX outputs.
        assert result.output_names == ["logits_per_image", "image_embeds", "text_embeds"]
        assert result.actual_device == "GPU"
        assert result.actual_ep == "OpenVINOExecutionProvider"
        assert result.actual_task == "zero-shot-image-classification"


class TestDescribeOutputs:
    """Unit tests for the architecture-agnostic forward()-output describer."""

    def test_dict_output_named_fields(self) -> None:
        out = {
            "logits": np.zeros((2, 5), dtype=np.float32),
            "embeds": np.zeros((2, 8), dtype=np.float32),
        }
        names, shapes, types = _describe_outputs(out)
        assert names == ["logits", "embeds"]
        assert shapes == [[2, 5], [2, 8]]
        assert all("float32" in t for t in types)

    def test_skips_none_and_non_array_fields(self) -> None:
        out = {"a": np.zeros((1, 3)), "b": None, "c": "not-an-array"}
        names, shapes, _ = _describe_outputs(out)
        assert names == ["a"]
        assert shapes == [[1, 3]]

    def test_sequence_output_positional_names(self) -> None:
        names, shapes, _ = _describe_outputs([np.zeros((1, 4)), np.zeros((1, 2))])
        assert names == ["output_0", "output_1"]
        assert shapes == [[1, 4], [1, 2]]

    def test_single_tensor_output(self) -> None:
        names, shapes, _ = _describe_outputs(np.zeros((3, 3)))
        assert names == ["output_0"]
        assert shapes == [[3, 3]]
