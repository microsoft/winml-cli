"""Tests for the standalone LLM benchmark scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "scripts" / "e2e_eval" / "schemas" / "llm_eval_result.schema.json"


def _load_script(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_script("_cpu_llm_runner", "scripts/e2e_eval/run_llm_eval.py")


@pytest.fixture(scope="module")
def reporter():
    return _load_script("_cpu_llm_reporter", "scripts/e2e_eval/build_llm_report.py")


def _perf_report(
    prompt_tokens: int = 256,
    *,
    device: str = "cpu",
    ep: str = "cpu",
    accelerator_util_pct: float = 0.0,
    local_memory_mb: float = 0.0,
    shared_memory_mb: float = 0.0,
) -> dict:
    report = {
        "benchmark_info": {
            "runtime": "winml-genai",
            "ep": ep,
            "device": device,
            "compile": False,
            "monitor": True,
            "iterations": 3,
            "warmup": 1,
            "max_new_tokens": 128,
            "apply_template": False,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": 128,
        },
        "ttft_ms": {"mean": 1000.0},
        "prefill_ms": {"mean": 900.0},
        "decode": {"tokens_per_sec": 8.0, "tpot_ms": 125.0},
        "total_generation_ms": {"mean": 17000.0},
        "raw": {
            "ttft_ms": [980.0, 1000.0, 1020.0],
            "prefill_ms": [880.0, 900.0, 920.0],
            "decode_tokens_per_sec": [7.9, 8.0, 8.1],
            "tpot_ms": [126.0, 125.0, 123.5],
            "total_ms": [17100.0, 17000.0, 16900.0],
        },
        "hw_monitor": {
            "device_kind": None if device == "cpu" else device,
            "cpu": {"process_mean_pct": 350.0, "sample_count": 40},
            "ram": {"mean_mb": 2048.0},
            "device_memory": {
                "local_mean_mb": local_memory_mb,
                "shared_mean_mb": shared_memory_mb,
            },
        },
    }
    if device != "cpu":
        report["hw_monitor"][device] = {
            "mean_pct": accelerator_util_pct,
            "sample_count": 40,
        }
    return report


class TestBundleValidation:
    def test_provider_free_bundle_is_accepted(self, runner, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        config = {
            "model": {"decoder": {"session_options": {"provider_options": []}}}
        }
        (bundle / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")

        assert runner.validate_bundle(bundle) == config

    def test_hardware_provider_bundle_is_accepted(self, runner, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        config = {
            "model": {
                "decoder": {"session_options": {"provider_options": [{"qnn": {}}]}}
            }
        }
        (bundle / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")

        assert runner.validate_bundle(bundle) == config

    def test_missing_config_is_rejected(self, runner, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"No genai_config\.json"):
            runner.validate_bundle(tmp_path)


class TestPerfResultMapping:
    @staticmethod
    def _point(runner, report: dict | None = None):
        return runner._context_point(
            256,
            report or _perf_report(),
            expected_device="cpu",
            expected_ep="cpu",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

    def test_perf_args_forward_target_and_enable_monitor(self, runner, tmp_path: Path) -> None:
        args = runner._perf_args(
            bundle_dir=tmp_path / "bundle",
            report_path=tmp_path / "report.json",
            prompt="hello",
            max_new_tokens=128,
            iterations=3,
            warmup=1,
            device="npu",
            ep="qnn",
        )

        assert args[args.index("--device") + 1] == "npu"
        assert args[args.index("--ep") + 1] == "qnn"
        assert "--no-compile" in args
        assert "--no-apply-template" in args
        assert "--monitor" in args

    def test_cpu_context_point_maps_schema_metrics(self, runner) -> None:
        point = self._point(runner)

        assert point["context_length_tokens"] == 256
        assert point["tokens_per_second"] == 8.0
        assert point["prefill_tokens_per_second"] == pytest.approx(256 / 0.9)
        assert point["ttft_s"] == 1.0
        assert point["total_elapsed_s"] == 17.0
        assert point["inter_token_latency_ms"]["avg"] == pytest.approx(124.8333)
        assert point["gpu_util_avg_pct"] == 0.0
        assert point["vram"] == {"util_avg_pct": 0.0, "used_avg_mb": 0.0}
        assert point["process_cpu_util_avg_pct"] == 350.0
        assert point["process_mem"] == {"util_avg_pct": 12.5, "used_avg_mb": 2048.0}

    def test_npu_context_point_maps_adapter_and_shared_memory(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(
                device="npu",
                ep="qnn",
                accelerator_util_pct=42.5,
                shared_memory_mb=512.0,
            ),
            expected_device="npu",
            expected_ep="qnn",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

        assert point["gpu_util_avg_pct"] == 42.5
        assert point["vram"] == {"util_avg_pct": 3.125, "used_avg_mb": 512.0}

    def test_gpu_context_point_uses_dedicated_vram_capacity(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(
                device="gpu",
                ep="qnn",
                accelerator_util_pct=75.0,
                local_memory_mb=1024.0,
            ),
            expected_device="gpu",
            expected_ep="qnn",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

        assert point["gpu_util_avg_pct"] == 75.0
        assert point["vram"] == {"util_avg_pct": 25.0, "used_avg_mb": 1024.0}

    def test_context_length_mismatch_fails(self, runner) -> None:
        with pytest.raises(ValueError, match="perf measured 255"):
            self._point(runner, _perf_report(255))

    def test_non_cpu_perf_report_fails(self, runner) -> None:
        report = _perf_report()
        report["benchmark_info"]["device"] = "gpu"

        with pytest.raises(ValueError, match=r"benchmark_info\.device"):
            self._point(runner, report)

    def test_missing_hw_monitor_fails(self, runner) -> None:
        report = _perf_report()
        report.pop("hw_monitor")

        with pytest.raises(TypeError, match="no hw_monitor"):
            self._point(runner, report)

    def test_missing_raw_sample_fails(self, runner) -> None:
        report = _perf_report()
        report["raw"]["ttft_ms"].pop()

        with pytest.raises(ValueError, match=r"raw\.ttft_ms"):
            self._point(runner, report)


class TestResultContract:
    @pytest.mark.parametrize(("device", "ep"), [("cpu", "cpu"), ("npu", "qnn")])
    def test_result_validates_against_schema(self, runner, device: str, ep: str) -> None:
        report = _perf_report(
            device=device,
            ep=ep,
            accelerator_util_pct=42.0,
            shared_memory_mb=512.0,
        )
        point = runner._context_point(
            256,
            report,
            expected_device=device,
            expected_ep=ep,
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16000.0,
            total_vram_mb=4000.0,
        )
        result = runner.build_result(
            model="organization/model",
            model_type="causal-lm",
            task="text-generation",
            quantization="w8a16",
            device=device,
            ep=ep,
            started_at="2026-07-17T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "windows",
                "hardware": {
                    "cpu_name": "Test CPU",
                    "logical_cores": 8,
                    "total_ram_mb": 16000.0,
                },
                "total_ram_mb": 16000.0,
                "total_vram_mb": 4000.0,
                "gpu_memory_gb": 4.0,
            },
        )
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        jsonschema.validate(result, schema)
        assert result["device"] == device
        assert result["ep"] == ep
        assert result["run"]["passed"] is True

    def test_distribution_empty_is_none(self, runner) -> None:
        assert runner._distribution([]) is None


class TestReport:
    @staticmethod
    def _result(runner, *, device: str, ep: str) -> dict:
        point = runner._context_point(
            256,
            _perf_report(
                device=device,
                ep=ep,
                accelerator_util_pct=42.0,
                shared_memory_mb=512.0,
            ),
            expected_device=device,
            expected_ep=ep,
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16000.0,
            total_vram_mb=4000.0,
        )
        return runner.build_result(
            model="organization/model",
            model_type="causal-lm",
            task="text-generation",
            quantization="w8a16",
            device=device,
            ep=ep,
            started_at="2026-07-17T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "windows",
                "hardware": {
                    "cpu_name": "Test CPU",
                    "cpu_logical_cores": 8,
                    "total_ram_mb": 16000.0,
                },
                "total_ram_mb": 16000.0,
                "total_vram_mb": 4000.0,
                "gpu_memory_gb": 4.0,
            },
        )

    def test_render_contains_cpu_metrics_and_method(self, runner, reporter) -> None:
        result = self._result(runner, device="cpu", ep="cpu")

        document = reporter.render_html(result)

        assert "organization/model CPU Benchmark" in document
        assert "8.00" in document
        assert "deterministic repeated filler" in document
        assert "Accelerator %" in document
        assert "Process RAM MB" in document

    def test_reporter_accepts_npu_result(self, runner, reporter) -> None:
        result = self._result(runner, device="npu", ep="qnn")

        reporter._validate_result(result)
        document = reporter.render_html(result)

        assert "organization/model NPU Benchmark" in document
        assert "QNN" in document


class TestProcessLifecycle:
    def test_interruption_kills_process_tree(self, runner, monkeypatch) -> None:
        killed: list[int] = []

        class InterruptedProcess:
            pid = 1234
            returncode = None

            def communicate(self, timeout=None):
                raise KeyboardInterrupt

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(
            runner.subprocess,
            "Popen",
            lambda *_args, **_kwargs: InterruptedProcess(),
        )
        monkeypatch.setattr(runner, "_kill_process_tree", killed.append)

        with pytest.raises(KeyboardInterrupt):
            runner._run_process(["benchmark"], timeout=10)

        assert killed == [1234]
