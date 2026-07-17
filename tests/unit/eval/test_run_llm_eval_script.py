"""Tests for the standalone CPU LLM benchmark scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "scripts" / "e2e_eval" / "schemas" / "llm_benchmark.schema.json"


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


def _perf_report(prompt_tokens: int = 256) -> dict:
    return {
        "benchmark_info": {
            "runtime": "winml-genai",
            "ep": "cpu",
            "device": "cpu",
            "compile": False,
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
    }


def _process_result(runner):
    return runner.ProcessResult(
        args=["winml", "perf"],
        exit_code=0,
        elapsed_s=20.0,
        stdout="",
        stderr="",
        timed_out=False,
        cpu_avg_pct=350.0,
        memory_avg_mb=2048.0,
        memory_avg_pct=12.5,
        resource_sample_count=40,
    )


class TestCpuBundleValidation:
    def test_provider_free_bundle_is_accepted(self, runner, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        config = {
            "model": {"decoder": {"session_options": {"provider_options": []}}}
        }
        (bundle / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")

        assert runner.validate_cpu_bundle(bundle) == config

    def test_hardware_provider_is_rejected(self, runner, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        config = {
            "model": {
                "decoder": {"session_options": {"provider_options": [{"qnn": {}}]}}
            }
        }
        (bundle / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")

        with pytest.raises(ValueError, match="not CPU-only"):
            runner.validate_cpu_bundle(bundle)


class TestPerfResultMapping:
    @staticmethod
    def _point(runner, report: dict | None = None):
        return runner._context_point(
            256,
            report or _perf_report(),
            _process_result(runner),
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
        )

    def test_perf_args_force_cpu_and_disable_compile(self, runner, tmp_path: Path) -> None:
        args = runner._perf_args(
            bundle_dir=tmp_path / "bundle",
            report_path=tmp_path / "report.json",
            prompt="hello",
            max_new_tokens=128,
            iterations=3,
            warmup=1,
        )

        assert args[args.index("--device") + 1] == "cpu"
        assert "--no-compile" in args
        assert "--no-apply-template" in args

    def test_context_point_preserves_raw_samples(self, runner) -> None:
        point = self._point(runner)

        assert point["context_length_tokens"] == 256
        assert point["decode_tokens_per_second"] == 8.0
        assert point["prefill_tokens_per_second"] == pytest.approx(256 / 0.9)
        assert point["ttft_s"] == 1.0
        assert point["generation_compute_s"] == 17.0
        assert point["raw"]["decode_tokens_per_second"] == [7.9, 8.0, 8.1]
        assert point["process_cpu_avg_pct"] == 350.0

    def test_context_length_mismatch_fails(self, runner) -> None:
        with pytest.raises(ValueError, match="perf measured 255"):
            self._point(runner, _perf_report(255))

    def test_non_cpu_perf_report_fails(self, runner) -> None:
        report = _perf_report()
        report["benchmark_info"]["device"] = "gpu"

        with pytest.raises(ValueError, match=r"benchmark_info\.device"):
            self._point(runner, report)

    def test_missing_raw_sample_fails(self, runner) -> None:
        report = _perf_report()
        report["raw"]["ttft_ms"].pop()

        with pytest.raises(ValueError, match=r"raw\.ttft_ms"):
            self._point(runner, report)


class TestResultContract:
    def test_result_validates_against_schema(self, runner) -> None:
        point = TestPerfResultMapping._point(runner)
        result = runner.build_result(
            model="organization/model",
            dtype="f16",
            bundle_dir=Path("output/bundle"),
            config_sha256="a" * 64,
            provider_names=[],
            context_lengths=[256],
            max_new_tokens=128,
            iterations=3,
            warmup=1,
            started_at="2026-07-17T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "Windows",
                "cpu": "Test CPU",
                "logical_cores": 8,
                "total_ram_mb": 16000.0,
                "python": "3.11.0",
            },
        )
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        jsonschema.validate(result, schema)
        assert result["device"] == "cpu"
        assert result["bundle"]["provider_names"] == []
        assert result["run"]["passed"] is True

    def test_distribution_empty_is_none(self, runner) -> None:
        assert runner._distribution([]) is None


class TestReport:
    def test_render_contains_cpu_metrics_and_method(self, runner, reporter) -> None:
        point = TestPerfResultMapping._point(runner)
        result = runner.build_result(
            model="organization/model",
            dtype="f16",
            bundle_dir=Path("output/bundle"),
            config_sha256="b" * 64,
            provider_names=[],
            context_lengths=[256],
            max_new_tokens=128,
            iterations=3,
            warmup=1,
            started_at="2026-07-17T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "Windows",
                "cpu": "Test CPU",
                "logical_cores": 8,
                "total_ram_mb": 16000.0,
                "python": "3.11.0",
            },
        )

        document = reporter.render_html(result)

        assert "organization/model CPU Benchmark" in document
        assert "8.00" in document
        assert "3 timed / 1 warmup" in document
        assert "deterministic repeated filler" in document
        assert "GPU util" not in document

    def test_reporter_rejects_non_cpu_result(self, runner, reporter) -> None:
        point = TestPerfResultMapping._point(runner)
        result = runner.build_result(
            model="organization/model",
            dtype="f16",
            bundle_dir=Path("output/bundle"),
            config_sha256="c" * 64,
            provider_names=[],
            context_lengths=[256],
            max_new_tokens=128,
            iterations=3,
            warmup=1,
            started_at="2026-07-17T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "Windows",
                "cpu": "Test CPU",
                "logical_cores": 8,
                "total_ram_mb": 16000.0,
                "python": "3.11.0",
            },
        )
        result["device"] = "gpu"

        with pytest.raises(jsonschema.ValidationError):
            reporter._validate_result(result)


class TestProcessSampler:
    def test_sampler_starts_and_stops(self, runner) -> None:
        sampler = runner._ProcessTreeSampler(os.getpid(), 0.05, total_ram_mb=16000.0)
        sampler.start()
        time.sleep(0.2)
        sampler.stop()
        sampler.join(timeout=5)

        assert not sampler.is_alive()
        assert sampler.summary()["sample_count"] >= 1

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
