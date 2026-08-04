# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for native Hugging Face PyTorch benchmarking in ``winml perf``."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from click.testing import CliRunner

import winml.modelkit.commands.perf as perf_module
from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    BenchmarkResult,
    PyTorchPerfBenchmark,
    _PyTorchForwardRunner,
    perf,
)
from winml.modelkit.export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.session.stats import PerfStats


class _RecordingModel(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 2, dtype=dtype)
        self.inference_modes: list[bool] = []
        self.input_devices: list[torch.device] = []
        self.input_dtypes: list[torch.dtype] = []
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.input_devices.append(pixel_values.device)
        self.input_dtypes.append(pixel_values.dtype)
        self.input_shapes.append(tuple(pixel_values.shape))
        return {"logits": self.projection(pixel_values)}


class _DecoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.dummy_inputs = {
            "input_ids": torch.ones((3, 5), dtype=torch.int64),
        }
        self.seen_inputs: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        self.seen_inputs.append(inputs)
        return {"logits": self.embedding(input_ids)}


class _MultimodalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.dummy_inputs = {"input_ids": torch.ones((3, 5), dtype=torch.int64)}
        self.seen_inputs: list[set[str]] = []

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids is None or pixel_values is None:
            raise ValueError("both modalities are required")
        self.seen_inputs.append({"input_ids", "pixel_values"})
        return {"logits": (input_ids.float().mean() + pixel_values.mean()) * self.scale}


class _OptionalMultimodalModel(torch.nn.Module):
    main_input_name = "input_ids"

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {}
        if input_ids is not None:
            outputs["text"] = input_ids.float() * self.scale
        if pixel_values is not None:
            outputs["image"] = pixel_values * self.scale
        if not outputs:
            raise ValueError("at least one modality is required")
        return outputs


class _Seq2SeqDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.dummy_inputs = {
            "input_ids": torch.ones((3, 5), dtype=torch.int64),
            "attention_mask": torch.ones((3, 5), dtype=torch.int64),
        }
        self.seen_shapes: list[dict[str, tuple[int, ...]]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.seen_shapes.append(
            {
                "input_ids": tuple(input_ids.shape),
                "attention_mask": tuple(attention_mask.shape),
            }
        )
        return {"logits": self.embedding(input_ids)}


class _AlternativeInputModel(torch.nn.Module):
    main_input_name = "raw_input"

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.paths: list[str] = []

    def forward(
        self,
        raw_input: torch.Tensor | None = None,
        precomputed_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if precomputed_input is not None:
            self.paths.append("precomputed")
            return {"output": precomputed_input * self.scale}
        if raw_input is None:
            raise ValueError("raw input is required")
        self.paths.append("raw")
        return {"output": raw_input * self.scale}


def _fake_build_config() -> SimpleNamespace:
    return SimpleNamespace(
        loader=SimpleNamespace(
            task="image-classification",
            model_class="AutoModelForImageClassification",
            trust_remote_code=False,
        ),
        export=WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(
                    name="pixel_values",
                    dtype="float32",
                    shape=(1, 4),
                )
            ],
            output_tensors=[OutputTensorSpec(name="logits")],
        ),
    )


def _patch_hf_loading(
    monkeypatch: pytest.MonkeyPatch,
    model: torch.nn.Module,
) -> tuple[MagicMock, MagicMock]:
    generate = MagicMock(return_value=_fake_build_config())
    load = MagicMock(return_value=(model, SimpleNamespace(), "image-classification"))
    monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
    monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
    return generate, load


class TestNoExportCli:
    def test_help_shows_export_pair(self) -> None:
        result = CliRunner().invoke(perf, ["--help"])

        assert result.exit_code == 0
        assert "--export" in result.output
        assert "--no-export" in result.output

    def test_no_export_dispatches_pytorch_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        captured: dict[str, object] = {}

        def run(config: BenchmarkConfig, **_: object) -> None:
            captured["config"] = config

        monkeypatch.setattr(perf_module, "_run_pytorch_perf_command", run)
        output = tmp_path / "result.json"

        result = CliRunner().invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--no-export",
                "--device",
                "cpu",
                "--iterations",
                "2",
                "-o",
                str(output),
            ],
            obj={},
        )

        assert result.exit_code == 0, result.output
        config = captured["config"]
        assert isinstance(config, BenchmarkConfig)
        assert config.backend == "pytorch"
        assert config.device == "cpu"
        assert config.iterations == 2

    @pytest.mark.parametrize(
        "args, expected_flag",
        [
            (["--ep", "cpu"], "--ep"),
            (["--precision", "fp16"], "--precision"),
            (["--prompt", "hello"], "--prompt"),
            (["--max-new-tokens", "4"], "--max-new-tokens"),
            (["--no-quant"], "--quant/--no-quant"),
            (["--compile"], "--compile/--no-compile"),
            (["--module", "Linear"], "--module"),
            (["--submodel", "encoder"], "--submodel"),
            (["--op-tracing", "basic"], "--op-tracing"),
        ],
    )
    def test_rejects_onnx_only_options(
        self,
        args: list[str],
        expected_flag: str,
    ) -> None:
        result = CliRunner().invoke(
            perf,
            ["-m", "fake/model", "--no-export", *args],
            obj={},
        )

        assert result.exit_code == 2
        assert expected_flag in result.output

    def test_rejects_onnx_input(self, tmp_path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"not used")

        result = CliRunner().invoke(
            perf,
            ["-m", str(model_path), "--no-export"],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a Hugging Face model" in result.output

    def test_rejects_genai_runtime(self) -> None:
        result = CliRunner().invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--no-export",
                "--runtime",
                "winml-genai",
            ],
            obj={},
        )

        assert result.exit_code == 2
        assert "only supported with --runtime winml" in result.output

    def test_rejects_npu_device(self) -> None:
        result = CliRunner().invoke(
            perf,
            ["-m", "fake/model", "--no-export", "--device", "npu"],
            obj={},
        )

        assert result.exit_code == 2
        assert "use auto, cpu, or gpu" in result.output

    def test_gpu_requires_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = CliRunner().invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--no-export",
                "--device",
                "gpu",
                "-o",
                str(tmp_path / "result.json"),
            ],
            obj={},
        )

        assert result.exit_code == 2
        assert "requires a CUDA-enabled PyTorch" in result.output


class TestPyTorchPerfBenchmark:
    def test_runs_raw_forward_on_cpu(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _RecordingModel(dtype=torch.float64)
        generate, load = _patch_hf_loading(monkeypatch, model)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/model",
                backend="pytorch",
                device="cpu",
                iterations=2,
                warmup=1,
                batch_size=3,
                memory=False,
            )
        )

        result = benchmark.run()

        assert len(model.inference_modes) == 3
        assert all(model.inference_modes)
        assert model.input_devices == [torch.device("cpu")] * 3
        assert model.input_dtypes == [torch.float64] * 3
        assert model.input_shapes == [(3, 4)] * 3
        assert result.config.backend == "pytorch"
        assert result.actual_device == "cpu"
        assert result.actual_ep is None
        assert result.actual_task == "image-classification"
        assert result.model_precision == "float64"
        assert result.input_shapes == [[3, 4]]
        assert result.input_types == ["float64"]
        assert result.output_names == ["logits"]
        assert result.output_shapes == [[3, 2]]
        assert len(result.raw_samples_ms) == 2
        assert result.mean_ms > 0
        assert result.effective_batch_size == 3
        generate.assert_called_once()
        load.assert_called_once_with(
            "fake/model",
            task=None,
            use_checkpoint_class=True,
            torch_dtype="auto",
        )

    def test_uses_native_dummy_input_for_flattened_export_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _DecoderModel()
        build_config = _fake_build_config()
        build_config.loader.task = "text-generation"
        build_config.export.input_tensors = [
            InputTensorSpec(name="past_key_values.0.key", dtype="float32", shape=(1, 2, 4, 2)),
        ]
        generate = MagicMock(return_value=build_config)
        load = MagicMock(return_value=(model, SimpleNamespace(), "text-generation"))
        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/decoder",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                batch_size=2,
                memory=False,
            )
        )

        result = benchmark.run()

        assert list(model.seen_inputs) == [{"input_ids": model.seen_inputs[0]["input_ids"]}]
        assert tuple(model.seen_inputs[0]["input_ids"].shape) == (2, 5)
        assert result.input_names == ["input_ids"]

    def test_uses_native_dummy_input_when_export_specs_cannot_resolve(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _DecoderModel()
        generate = MagicMock(side_effect=AttributeError("missing normalized attribute"))
        load = MagicMock(return_value=(model, SimpleNamespace(), "text-generation"))
        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/decoder",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                memory=False,
            )
        )

        result = benchmark.run()

        assert result.input_names == ["input_ids"]
        assert len(result.raw_samples_ms) == 1

    def test_merges_composite_inputs_for_full_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _MultimodalModel()
        text_config = _fake_build_config()
        text_config.loader.task = "feature-extraction"
        text_config.export.input_tensors = [
            InputTensorSpec(name="input_ids", dtype="int64", shape=(1, 5)),
        ]
        full_config = _fake_build_config()
        full_config.loader.task = "zero-shot-image-classification"
        full_config.export.input_tensors = [
            InputTensorSpec(name="input_ids", dtype="int64", shape=(1, 5)),
            InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 4, 4)),
        ]
        generate = MagicMock(side_effect=[text_config, full_config])
        load = MagicMock(
            return_value=(model, SimpleNamespace(model_type="multimodal"), "feature-extraction")
        )
        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        monkeypatch.setattr(
            "winml.modelkit.loader.composite_pipeline_tasks",
            lambda _model_type: ["zero-shot-image-classification"],
        )
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/multimodal",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                memory=False,
            )
        )

        result = benchmark.run()

        assert model.seen_inputs == [{"input_ids", "pixel_values"}]
        assert result.input_names == ["input_ids", "pixel_values"]

    def test_merges_inputs_from_unregistered_nested_configs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from transformers import PretrainedConfig

        model = _MultimodalModel()
        parent_config = PretrainedConfig()
        parent_config.model_type = "multimodal"
        parent_config.text_config = PretrainedConfig()
        parent_config.text_config.model_type = "text-encoder"
        parent_config.vision_config = PretrainedConfig()
        parent_config.vision_config.model_type = "vision-encoder"
        generate = MagicMock(side_effect=ValueError("no top-level exporter"))
        load = MagicMock(return_value=(model, parent_config, "feature-extraction"))

        def resolve_specs(model_type: str, *_args: object, **_kwargs: object) -> dict[str, object]:
            if model_type == "text-encoder":
                return {
                    "input_names": ["input_ids"],
                    "input_shapes": [(1, 5)],
                    "input_dtypes": ["int64"],
                    "value_ranges": {},
                }
            return {
                "input_names": ["pixel_values"],
                "input_shapes": [(1, 3, 4, 4)],
                "input_dtypes": ["float32"],
                "value_ranges": {},
            }

        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        monkeypatch.setattr("winml.modelkit.loader.composite_pipeline_tasks", lambda _: [])
        monkeypatch.setattr("winml.modelkit.export.resolve_io_specs", resolve_specs)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/multimodal",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                memory=False,
            )
        )

        result = benchmark.run()

        assert model.seen_inputs == [{"input_ids", "pixel_values"}]
        assert set(result.input_names) == {"input_ids", "pixel_values"}

    def test_shape_config_resizes_all_native_sequence_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _Seq2SeqDummyModel()
        generate = MagicMock(side_effect=ValueError("no export specs"))
        load = MagicMock(
            return_value=(model, SimpleNamespace(model_type="decoder"), "text2text-generation")
        )
        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        monkeypatch.setattr("winml.modelkit.loader.composite_pipeline_tasks", lambda _: [])
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/decoder",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                batch_size=2,
                shape_config={"sequence_length": 7},
                memory=False,
            )
        )

        result = benchmark.run()

        assert model.seen_shapes == [
            {
                "input_ids": (2, 7),
                "attention_mask": (2, 7),
            }
        ]
        assert result.input_shapes == [[2, 7], [2, 7]]

    def test_preflight_keeps_checkpoint_main_input_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _AlternativeInputModel()
        build_config = _fake_build_config()
        build_config.export.input_tensors = [
            InputTensorSpec(name="precomputed_input", dtype="float32", shape=(1, 4)),
            InputTensorSpec(name="raw_input", dtype="float32", shape=(1, 4)),
        ]
        generate = MagicMock(return_value=build_config)
        load = MagicMock(
            return_value=(model, SimpleNamespace(model_type="alternative"), "feature-extraction")
        )
        monkeypatch.setattr("winml.modelkit.config.generate_hf_build_config", generate)
        monkeypatch.setattr("winml.modelkit.loader.load_hf_model", load)
        monkeypatch.setattr("winml.modelkit.loader.composite_pipeline_tasks", lambda _: [])
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/alternative",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                memory=False,
            )
        )

        result = benchmark.run()

        assert model.paths[-1] == "raw"
        assert result.input_names == ["raw_input"]

    def test_preflight_keeps_all_independent_modalities(self) -> None:
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(model_id="fake/multimodal", backend="pytorch")
        )
        benchmark._model = _OptionalMultimodalModel()
        benchmark._main_input_name = "input_ids"
        inputs = {
            "input_ids": torch.ones((1, 4), dtype=torch.int64),
            "pixel_values": torch.ones((1, 3, 4, 4)),
        }

        selected = benchmark._select_checkpoint_forward_inputs(torch, inputs)

        assert list(selected) == ["input_ids", "pixel_values"]
        assert selected["input_ids"] is inputs["input_ids"]
        assert selected["pixel_values"] is inputs["pixel_values"]

    def test_input_data_owns_batch_size(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        model = _RecordingModel()
        _patch_hf_loading(monkeypatch, model)
        input_path = tmp_path / "inputs.npz"
        np.savez(input_path, pixel_values=np.ones((2, 4), dtype=np.float32))
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/model",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                batch_size=9,
                input_data=input_path,
                memory=False,
            )
        )

        result = benchmark.run()

        assert model.input_shapes == [(2, 4)]
        assert result.effective_batch_size == 2
        assert result.input_shapes == [[2, 4]]

    def test_auto_uses_cpu_without_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(model_id="fake/model", backend="pytorch", device="auto")
        )

        benchmark._resolve_device(torch)

        assert benchmark._actual_device == "cpu"
        assert benchmark._torch_device == torch.device("cpu")

    def test_duration_runs_until_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _RecordingModel()
        _patch_hf_loading(monkeypatch, model)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/model",
                backend="pytorch",
                device="cpu",
                duration=0.001,
                warmup=0,
                memory=False,
            )
        )

        result = benchmark.run()

        assert result.raw_samples_ms
        assert result.to_dict()["benchmark_info"]["iterations"] == len(result.raw_samples_ms)

    def test_monitor_falls_back_when_hw_monitor_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from winml.modelkit.session.monitor.hw_monitor import HWMonitor

        model = _RecordingModel()
        _patch_hf_loading(monkeypatch, model)
        monkeypatch.setattr(HWMonitor, "is_available", lambda: False)
        benchmark = PyTorchPerfBenchmark(
            BenchmarkConfig(
                model_id="fake/model",
                backend="pytorch",
                device="cpu",
                iterations=1,
                warmup=0,
                monitor=True,
                memory=False,
            )
        )

        result = benchmark.run()

        assert len(result.raw_samples_ms) == 1
        assert result.hw_monitor is None

    def test_memory_profile_uses_model_load_phase_names(self) -> None:
        profile = PyTorchPerfBenchmark._build_memory_profile(
            (100.0, 10.0, 0.0),
            (140.0, 30.0, 0.0),
            (150.0, 35.0, 0.0),
        )

        assert profile["rss_after_model_load_mb"] == 140.0
        assert profile["rss_model_load_delta_mb"] == 40.0
        assert profile["vram_local_model_load_delta_mb"] == 20.0
        assert profile["vram_local_inference_delta_mb"] == 5.0


class TestPyTorchTiming:
    def test_cuda_runner_synchronizes_before_and_after_forward(self) -> None:
        model = MagicMock(return_value={"logits": torch.ones(1, 2)})
        synchronize = MagicMock()
        stats = PerfStats()
        runner = _PyTorchForwardRunner(model, stats, synchronize)
        inputs = {"pixel_values": torch.ones(1, 4)}

        runner.run(inputs)

        assert synchronize.call_count == 2
        model.assert_called_once_with(**inputs)
        assert len(stats.samples_ms) == 1
        assert runner.output_metadata == [("logits", [1, 2], "float32")]

    def test_cuda_luid_uses_windows_pdh_format(self) -> None:
        raw_luid = bytes.fromhex("0200000001000000")

        assert PyTorchPerfBenchmark._format_cuda_luid(raw_luid) == "0x00000001_0x00000002"


def test_report_identifies_pytorch_backend() -> None:
    result = BenchmarkResult(
        config=BenchmarkConfig(model_id="fake/model", backend="pytorch"),
        actual_device="gpu",
    )

    report = result.to_dict()

    assert report["benchmark_info"]["backend"] == "pytorch"
    assert report["benchmark_info"]["ep"] is None
    assert report["benchmark_info"]["precision"] is None
    assert report["benchmark_info"]["running_model_path"] == ""
    json.dumps(report)
