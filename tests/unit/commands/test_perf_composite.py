# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml perf support of composite (multi-session) models.

Composite models (e.g. CLIP/SigLIP dual-encoders) have no single ONNX
session; they orchestrate several sub-models. ``winml perf`` benchmarks
each sub-model individually (like ``--module``) and reports one row per
sub-model rather than timing the aggregate ``forward()`` pass.

Regression guard: previously ``PerfBenchmark`` assumed every model exposed
``io_config`` / ``_session`` and raised ``AttributeError`` on composites.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    BenchmarkResult,
    PerfBenchmark,
    report_composite_results,
)
from winml.modelkit.ep_path import EPEntry, PyPISource
from winml.modelkit.models.winml import WinMLCompositeModel
from winml.modelkit.session import WinMLDevice, WinMLEP, WinMLEPDevice
from winml.modelkit.session.stats import PerfStats


if TYPE_CHECKING:
    from collections.abc import Generator


class _FakeSession:
    """Stand-in for a WinMLSession that times runs via a real PerfStats."""

    def __init__(self, io_config: dict[str, Any], device: str, ep_name: str) -> None:
        self.io_config = io_config
        self.device = device
        self.ep_name = ep_name
        self.running_model_path = "model.onnx"
        self.compiled = False
        self.run_log: list[dict[str, Any]] = []
        self._perf_stats: PerfStats | None = None

    def compile(self) -> None:
        self.compiled = True

    @contextmanager
    def perf(self, warmup: int = 0) -> Generator[Any, None, None]:
        # WinMLSession.perf yields a PerfContext (``.stats`` + ``.monitor``),
        # not the PerfStats directly. Mirror that shape so the benchmark's
        # ``ctx.stats`` access works against the fake.
        self._perf_stats = PerfStats(warmup=warmup)
        try:
            yield SimpleNamespace(stats=self._perf_stats, monitor=None)
        finally:
            self._perf_stats = None

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.run_log.append(inputs)
        if self._perf_stats is not None:
            self._perf_stats.record(lambda: None)
        return {}


class _FakeSubModel:
    """Stand-in for a single-session WinMLAutoModel sub-component."""

    def __init__(
        self,
        io_config: dict[str, Any],
        task: str,
        *,
        device: str = "GPU",
        ep_name: str = "OpenVINOExecutionProvider",
    ) -> None:
        self._session = _FakeSession(io_config, device, ep_name)
        self.task = task

    @property
    def io_config(self) -> dict[str, Any]:
        return self._session.io_config

    @property
    def device(self) -> str:
        return self._session.device

    @property
    def ep_name(self) -> str:
        return self._session.ep_name

    @property
    def running_model_path(self) -> str:
        return self._session.running_model_path


class _FakeComposite(WinMLCompositeModel):
    """Stand-in for a WinMLCompositeModel.

    Subclasses the real base so ``PerfBenchmark._is_composite``'s concrete
    ``isinstance`` check matches, but skips the torch-backed ``__init__`` and
    just carries the sub-model mapping the benchmark needs.
    """

    def __init__(self, sub_models: dict[str, Any]) -> None:
        self.sub_models = sub_models


def _io_config(
    input_names: list[str],
    input_shapes: list[list[int]],
    input_types: list[str],
    output_names: list[str],
    output_shapes: list[list[int]],
    *,
    precision: str | None = "fp16",
) -> dict[str, Any]:
    return {
        "input_names": input_names,
        "input_shapes": input_shapes,
        "input_types": input_types,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "output_types": ["float32"] * len(output_names),
        "precision": precision,
    }


def _siglip_like() -> _FakeComposite:
    image_encoder = _FakeSubModel(
        _io_config(
            ["pixel_values"],
            [[1, 3, 224, 224]],
            ["float32"],
            ["image_embeds"],
            [[1, 768]],
        ),
        task="image-feature-extraction",
    )
    text_encoder = _FakeSubModel(
        _io_config(
            ["input_ids", "attention_mask"],
            [[1, 64], [1, 64]],
            ["int64", "int64"],
            ["text_embeds"],
            [[1, 768]],
        ),
        task="feature-extraction",
    )
    return _FakeComposite({"image-encoder": image_encoder, "text-encoder": text_encoder})


def _resolved_ep_device() -> WinMLEPDevice:
    ep_name = "OpenVINOExecutionProvider"
    ort_device = MagicMock()
    ort_device.ep_name = ep_name
    ort_device.device = SimpleNamespace(
        type=SimpleNamespace(name="GPU"),
        metadata={"Description": "Fake GPU"},
    )
    ort_device.ep_metadata = {"FULL_DEVICE_NAME": "Fake GPU"}
    ort_device.ep_vendor = "Intel"
    device = WinMLDevice(ort_device)
    ep = WinMLEP(
        source=EPEntry(
            ep_name=ep_name,
            dll_path=Path("fake-openvino.dll"),
            source=PyPISource(
                distribution="fake-openvino",
                relative_dll="fake-openvino.dll",
                eps=(ep_name,),
            ),
            version="1.2.3",
        ),
        devices=(device,),
        arg0=ep_name,
    )
    return WinMLEPDevice(ep=ep, device=device)


def _composite_benchmark() -> tuple[PerfBenchmark, _FakeComposite]:
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
    bench._ep_device = _resolved_ep_device()
    bench._resolved_device = "gpu"
    bench._resolved_ep = "OpenVINOExecutionProvider"
    return bench, model


class TestPerfBenchmarkComposite:
    """PerfBenchmark benchmarks each sub-model of a composite individually."""

    def test_detects_composite(self) -> None:
        bench, _ = _composite_benchmark()
        assert bench._is_composite is True

    def test_run_returns_result_per_sub_model(self) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()

        assert set(results) == {"image-encoder", "text-encoder"}
        assert all(isinstance(r, BenchmarkResult) for r in results.values())

    def test_each_sub_model_reports_its_own_io(self) -> None:
        # No aggregation: each result carries only its sub-model's inputs.
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()

        assert results["image-encoder"].input_names == ["pixel_values"]
        assert results["text-encoder"].input_names == ["input_ids", "attention_mask"]
        assert results["image-encoder"].output_names == ["image_embeds"]
        assert results["text-encoder"].output_names == ["text_embeds"]

    def test_each_sub_model_reports_its_own_task(self) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()

        assert results["image-encoder"].actual_task == "image-feature-extraction"
        assert results["text-encoder"].actual_task == "feature-extraction"

    def test_resolved_device_and_ep_per_sub_model(self) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()

        for result in results.values():
            assert result.actual_device == "GPU"
            assert result.actual_ep == "OpenVINOExecutionProvider"

    def test_each_sub_model_inherits_parent_ep_identity(self) -> None:
        bench, _ = _composite_benchmark()

        with patch("winml.modelkit.commands.perf.print_pre_bench_block") as print_block:
            bench._run_sub_models()

        assert len(print_block.call_args_list) == 2
        for call in print_block.call_args_list:
            assert call.kwargs["ep_source"] == "pypi"
            assert call.kwargs["ep_version"] == "1.2.3"
            assert call.kwargs["hardware_name"] == "Fake GPU"

    def test_compiles_and_runs_every_sub_session(self) -> None:
        bench, model = _composite_benchmark()
        bench._run_sub_models()

        for sub in model.sub_models.values():
            assert sub._session.compiled is True
            # warmup(1) + iterations(3) == 4 run() calls per sub-session.
            assert len(sub._session.run_log) == 4

    def test_each_sub_model_stats_exclude_warmup(self) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()

        for result in results.values():
            assert len(result.raw_samples_ms) == 3

    def test_sub_model_failure_names_component(self) -> None:
        # A failing sub-model surfaces a RuntimeError naming the component,
        # chained (``from``) from the original exception so the cause is kept.
        bench, model = _composite_benchmark()

        def _boom() -> None:
            raise ValueError("compile failed")

        # image-encoder is first in dict order, so it fails before text-encoder.
        model.sub_models["image-encoder"]._session.compile = _boom

        with pytest.raises(RuntimeError, match="image-encoder") as excinfo:
            bench._run_sub_models()
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_empty_sub_models_returns_empty_dict(self) -> None:
        # Boundary: a composite with zero sub-models must not crash; it
        # yields an empty mapping that downstream reporting handles.
        config = BenchmarkConfig(model_id="empty/composite", device="gpu")
        bench = PerfBenchmark(config)
        bench._model = _FakeComposite({})

        assert bench._is_composite is True
        assert bench._run_sub_models() == {}


class TestReportCompositeResults:
    """report_composite_results writes a combined per-component JSON report."""

    def test_combined_json_nests_each_component(self, tmp_path: Path) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()
        output = tmp_path / "perf.json"

        report_composite_results(
            results,
            console=Console(),
            json_mode=False,
            output_path=output,
            model_id="google/siglip-base-patch16-224",
            task="zero-shot-image-classification",
        )

        data = json.loads(output.read_text())
        assert data["model_id"] == "google/siglip-base-patch16-224"
        assert data["task"] == "zero-shot-image-classification"
        assert data["component_count"] == 2
        assert set(data["components"]) == {"image-encoder", "text-encoder"}
        # Each component holds a full BenchmarkResult.to_dict() payload.
        img = data["components"]["image-encoder"]
        assert img["model_info"]["input_names"] == ["pixel_values"]
        assert "latency_ms" in img

    def test_json_mode_emits_combined_payload_to_stdout(self, tmp_path: Path, capsys: Any) -> None:
        bench, _ = _composite_benchmark()
        results = bench._run_sub_models()
        output = tmp_path / "perf.json"

        report_composite_results(
            results,
            console=Console(stderr=True),
            json_mode=True,
            output_path=output,
            model_id="google/siglip-base-patch16-224",
            task="zero-shot-image-classification",
        )

        payload = json.loads(capsys.readouterr().out)
        assert set(payload["components"]) == {"image-encoder", "text-encoder"}
        # File is written regardless of json_mode.
        assert output.exists()

    def test_empty_components_writes_zero_count(self, tmp_path: Path) -> None:
        # Boundary: zero sub-models produces a valid report with no rows.
        output = tmp_path / "perf.json"

        report_composite_results(
            {},
            console=Console(),
            json_mode=False,
            output_path=output,
            model_id="empty/composite",
            task="zero-shot-image-classification",
        )

        data = json.loads(output.read_text())
        assert data["component_count"] == 0
        assert data["components"] == {}
