# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ``scripts/e2e_eval/run_llm_eval.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import jsonschema
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "scripts" / "e2e_eval" / "schemas" / "llm_eval_result.schema.json"
TEST_BUNDLE_DIR = REPO_ROOT / "test-bundle"


def _load_runner():
    path = REPO_ROOT / "scripts" / "e2e_eval" / "run_llm_eval.py"
    spec = importlib.util.spec_from_file_location("_e2e_run_llm_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_e2e_run_llm_eval"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _perf_report(
    prompt_tokens: int = 256,
    *,
    device: str = "npu",
    ep: str = "qnn",
    generated_tokens: int = 128,
    bundle_dir: Path = TEST_BUNDLE_DIR,
) -> dict:
    return {
        "benchmark_info": {
            "runtime": "ort-genai",
            "bundle_dir": str(bundle_dir),
            "ep": ep,
            "device": device,
            "effective_device": device,
            "compile": True,
            "monitor": True,
            "iterations": 3,
            "warmup": 1,
            "max_new_tokens": 128,
            "apply_template": False,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
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
            "device_kind": device,
            "adapter": {"mean_pct": 42.5, "sample_count": 40},
            "cpu": {"process_mean_pct": 350.0, "sample_count": 40},
            "ram": {"mean_mb": 2048.0},
            "device_memory": {"local_mean_mb": 0.0, "shared_mean_mb": 512.0},
        },
    }


class TestPerfResultMapping:
    def test_perf_args_match_genai_command(self, runner, tmp_path: Path) -> None:
        args = runner._perf_args(
            bundle_dir=tmp_path / "bundle",
            report_path=tmp_path / "report.json",
            prompt_path=tmp_path / "prompt.txt",
            max_new_tokens=128,
            iterations=3,
            warmup=1,
            compile_timeout=1800,
            device="npu",
            ep="qnn",
        )

        assert args[args.index("--runtime") + 1] == "ort-genai"
        assert args[args.index("--device") + 1] == "npu"
        assert args[args.index("--ep") + 1] == "qnn"
        assert args[args.index("--compile-timeout") + 1] == "1800"
        assert "--compile" in args
        assert "--monitor" in args

    def test_context_point_maps_schema_metrics(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(),
            expected_bundle_dir=TEST_BUNDLE_DIR,
            expected_device="npu",
            expected_ep="qnn",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

        assert point["context_length_tokens"] == 256
        assert point["tokens_per_second"] == 8.0
        assert point["prefill_tokens_per_second"] == pytest.approx(256 / 0.9)
        assert point["ttft_s"] == 1.0
        assert point["total_elapsed_s"] == 17.0
        assert point["inter_token_latency_ms"]["avg"] == pytest.approx(124.8333)
        assert point["gpu_util_avg_pct"] == 42.5
        assert point["vram"] == {"util_avg_pct": None, "used_avg_mb": 512.0}
        assert point["process_cpu_util_avg_pct"] == 350.0
        assert point["process_mem"] == {"util_avg_pct": 12.5, "used_avg_mb": 2048.0}

    def test_context_length_mismatch_fails(self, runner) -> None:
        with pytest.raises(ValueError, match="perf measured 255"):
            runner._context_point(
                256,
                _perf_report(255),
                expected_bundle_dir=TEST_BUNDLE_DIR,
                expected_device="npu",
                expected_ep="qnn",
                expected_max_new_tokens=128,
                expected_iterations=3,
                expected_warmup=1,
                total_ram_mb=16384.0,
                total_vram_mb=4096.0,
            )

    def test_canonical_ep_aliases_match(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(ep="nvtensorrtrtx"),
            expected_bundle_dir=TEST_BUNDLE_DIR,
            expected_device="npu",
            expected_ep="nv_tensorrt_rtx@catalog",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

        assert point["generated_tokens"] == 128

    def test_early_eos_generated_count_is_accepted(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(generated_tokens=17),
            expected_bundle_dir=TEST_BUNDLE_DIR,
            expected_device="npu",
            expected_ep="qnn",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16384.0,
            total_vram_mb=4096.0,
        )

        assert point["generated_tokens"] == 17

    @pytest.mark.parametrize("generated_tokens", [0, 129])
    def test_invalid_generated_count_fails(self, runner, generated_tokens: int) -> None:
        with pytest.raises(ValueError, match=r"Expected 1\.\.128"):
            runner._context_point(
                256,
                _perf_report(generated_tokens=generated_tokens),
                expected_bundle_dir=TEST_BUNDLE_DIR,
                expected_device="npu",
                expected_ep="qnn",
                expected_max_new_tokens=128,
                expected_iterations=3,
                expected_warmup=1,
                total_ram_mb=16384.0,
                total_vram_mb=4096.0,
            )

    def test_cpu_result_rejects_gpu_monitor(self, runner) -> None:
        report = _perf_report(device="cpu", ep="dml")
        report["hw_monitor"]["device_kind"] = "gpu"

        with pytest.raises(ValueError, match="Expected CPU-only monitoring"):
            runner._context_point(
                256,
                report,
                expected_bundle_dir=TEST_BUNDLE_DIR,
                expected_device="cpu",
                expected_ep="dml",
                expected_max_new_tokens=128,
                expected_iterations=3,
                expected_warmup=1,
                total_ram_mb=16384.0,
                total_vram_mb=4096.0,
            )

    def test_bundle_mismatch_fails(self, runner) -> None:
        report = _perf_report(bundle_dir=TEST_BUNDLE_DIR / "other")

        with pytest.raises(ValueError, match=r"Expected benchmark_info\.bundle_dir"):
            runner._context_point(
                256,
                report,
                expected_bundle_dir=TEST_BUNDLE_DIR,
                expected_device="npu",
                expected_ep="qnn",
                expected_max_new_tokens=128,
                expected_iterations=3,
                expected_warmup=1,
                total_ram_mb=16384.0,
                total_vram_mb=4096.0,
            )

    def test_zero_token_prompt_filler_fails_fast(self, runner, monkeypatch, tmp_path) -> None:
        class EmptyTokenizer:
            def encode(self, _text, *, add_special_tokens=False):
                return []

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: EmptyTokenizer(),
        )

        with pytest.raises(ValueError, match="tokenize to at least one token"):
            runner._make_prompt(tmp_path, 256, "")


class TestResultContract:
    def test_result_validates_against_schema(self, runner) -> None:
        point = runner._context_point(
            256,
            _perf_report(),
            expected_bundle_dir=TEST_BUNDLE_DIR,
            expected_device="npu",
            expected_ep="qnn",
            expected_max_new_tokens=128,
            expected_iterations=3,
            expected_warmup=1,
            total_ram_mb=16000.0,
            total_vram_mb=4000.0,
        )
        result = runner.build_result(
            model="Qwen/Qwen3-1.7B",
            model_type="llm",
            task="text-generation",
            quantization="w8a16",
            device="npu",
            ep="qnn",
            group=None,
            priority="P0",
            machine_label=None,
            started_at="2026-08-04T00:00:00+00:00",
            elapsed_s=60.0,
            points=[point],
            errors=[],
            environment={
                "os": "windows",
                "hardware": {"cpu_name": "Test CPU", "cpu_logical_cores": 8},
                "total_ram_mb": 16000.0,
                "total_vram_mb": 4000.0,
                "gpu_memory_gb": 4.0,
            },
            command="winml perf -m bundle --runtime ort-genai --device npu",
            timed_out=False,
        )
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        jsonschema.validate(result, schema)
        assert result["run"]["passed"] is True

    def test_main_writes_schema_valid_result(
        self, runner, tmp_path: Path, monkeypatch
    ) -> None:
        bundle = tmp_path / "qwen3-bundle"
        bundle.mkdir()
        (bundle / "genai_config.json").write_text("{}", encoding="utf-8")
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        stale_failure = output_dir / runner.FAILURE_FILENAME
        stale_failure.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(runner, "_make_prompt", lambda *_args: "prompt")
        monkeypatch.setattr(
            runner,
            "_collect_environment",
            lambda _gpu_memory_gb: {
                "os": "windows",
                "hardware": {"cpu_name": "Test CPU", "cpu_logical_cores": 8},
                "total_ram_mb": 16384.0,
                "total_vram_mb": 4096.0,
                "gpu_memory_gb": 4.0,
            },
        )

        def fake_run(args: list[str], *, timeout: int):
            prompt_path = Path(args[args.index("--prompt-file") + 1])
            assert prompt_path.read_text(encoding="utf-8") == "prompt"
            assert "prompt" not in args
            report_path = Path(args[args.index("-o") + 1])
            model_arg = args.index("-m", args.index("perf") + 1)
            bundle_dir = Path(args[model_arg + 1])
            report_path.write_text(
                json.dumps(_perf_report(bundle_dir=bundle_dir)), encoding="utf-8"
            )
            return runner.ProcessResult(args, 0, 1.0, "", "", False)

        monkeypatch.setattr(runner, "_run_process", fake_run)

        exit_code = runner.main(
            [
                "-m",
                str(bundle),
                "--model-id",
                "Qwen/Qwen3-1.7B",
                "--output-dir",
                str(output_dir),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--context-lengths",
                "256",
            ]
        )

        result = json.loads((output_dir / runner.RESULT_FILENAME).read_text(encoding="utf-8"))
        runner._validate_result(result)
        assert exit_code == 0
        assert result["model"] == "Qwen/Qwen3-1.7B"
        assert len(result["context_sweep"]) == 1
        assert "winml.modelkit.cli perf" in result["run"]["command"]
        assert not stale_failure.exists()

    def test_failed_rerun_removes_stale_result(
        self, runner, tmp_path: Path, monkeypatch
    ) -> None:
        bundle = tmp_path / "qwen3-bundle"
        bundle.mkdir()
        (bundle / "genai_config.json").write_text("{}", encoding="utf-8")
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        stale_result = output_dir / runner.RESULT_FILENAME
        stale_result.write_text('{"run": {"passed": true}}', encoding="utf-8")

        monkeypatch.setattr(runner, "_make_prompt", lambda *_args: "prompt")
        monkeypatch.setattr(
            runner,
            "_collect_environment",
            lambda _gpu_memory_gb: {
                "os": "windows",
                "hardware": {},
                "total_ram_mb": 16384.0,
                "total_vram_mb": 0.0,
                "gpu_memory_gb": None,
            },
        )
        monkeypatch.setattr(
            runner,
            "_run_process",
            lambda args, *, timeout: runner.ProcessResult(args, 1, 1.0, "", "failed", False),
        )

        exit_code = runner.main(
            [
                "-m",
                str(bundle),
                "--output-dir",
                str(output_dir),
                "--context-lengths",
                "256",
            ]
        )

        assert exit_code == 1
        assert not stale_result.exists()
        assert (output_dir / runner.FAILURE_FILENAME).exists()

    def test_run_start_removes_stale_artifacts_before_bundle_validation(
        self, runner, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        stale_result = output_dir / runner.RESULT_FILENAME
        stale_failure = output_dir / runner.FAILURE_FILENAME
        stale_result.write_text("{}", encoding="utf-8")
        stale_failure.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="genai_config"):
            runner.main(
                [
                    "-m",
                    str(tmp_path / "missing-bundle"),
                    "--output-dir",
                    str(output_dir),
                ]
            )

        assert not stale_result.exists()
        assert not stale_failure.exists()

    def test_concurrent_run_rejects_shared_output_directory(
        self, runner, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        stale_result = output_dir / runner.RESULT_FILENAME
        stale_result.write_text('{"run": {"passed": true}}', encoding="utf-8")

        with (
            runner._OutputDirectoryLock(output_dir),
            pytest.raises(RuntimeError, match="already in use"),
        ):
            runner.main(
                [
                    "-m",
                    str(tmp_path / "bundle"),
                    "--output-dir",
                    str(output_dir),
                ]
            )

        assert stale_result.exists()


class TestProcessLifecycle:
    def test_guard_setup_failure_cleans_spawned_process(self, runner, monkeypatch) -> None:
        cleaned: list[int] = []
        original_cleanup = runner._kill_process_tree

        class BrokenGuard:
            def __init__(self, _process) -> None:
                raise OSError("job assignment failed")

        monkeypatch.setattr(runner, "_ProcessTreeGuard", BrokenGuard)
        monkeypatch.setattr(
            runner,
            "_kill_process_tree",
            lambda pid: (cleaned.append(pid), original_cleanup(pid)),
        )

        with pytest.raises(OSError, match="job assignment failed"):
            runner._run_process([sys.executable, "-c", "pass"], timeout=10)

        assert len(cleaned) == 1

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_gpu_memory_is_rejected(self, runner, tmp_path, value: float) -> None:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "genai_config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="must be positive"):
            runner.main(
                [
                    "-m",
                    str(bundle),
                    "--output-dir",
                    str(tmp_path / "results"),
                    f"--gpu-memory-gb={value}",
                ]
            )

    def test_json_writer_rejects_nonfinite_values(self, runner, tmp_path) -> None:
        with pytest.raises(ValueError, match="Out of range float values"):
            runner._write_json(tmp_path / "result.json", {"value": float("nan")})

    def test_timeout_kills_descendant_holding_output_pipe(self, runner) -> None:
        child_code = "import threading; threading.Event().wait(60)"
        parent_code = (
            "import subprocess, sys; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}])"
        )
        started = time.perf_counter()

        result = runner._run_process([sys.executable, "-c", parent_code], timeout=1)

        assert result.timed_out is True
        assert result.exit_code == -1
        assert time.perf_counter() - started < 10
