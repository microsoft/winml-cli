"""Build and benchmark a provider-free ONNX Runtime GenAI bundle on CPU.

The runner intentionally supports one execution target: CPU. It can build a
portable bundle with Mobius or consume an existing provider-free bundle, then
runs a fixed-token context sweep through ``winml perf --runtime winml-genai``.
Results include raw timing samples plus full-subprocess process-tree CPU and
RSS averages.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import psutil


SCHEMA_VERSION = "1.0"
RUNTIME = "onnxruntime-genai"
RESULT_FILENAME = "llm_cpu_benchmark.json"
DEFAULT_FILLER = "The quick brown fox jumps over the lazy dog. "
WINML_CLI = [sys.executable, "-m", "winml.modelkit.cli"]
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "llm_benchmark.schema.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str | None, limit: int = 4000) -> str:
    value = (text or "").strip()
    return value if len(value) <= limit else "...(truncated)...\n" + value[-limit:]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_version(distribution: str) -> str | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(distribution)
    return None


@dataclass
class ProcessResult:
    """Captured subprocess output, timing, and optional resource averages."""

    args: list[str]
    exit_code: int
    elapsed_s: float
    stdout: str
    stderr: str
    timed_out: bool
    cpu_avg_pct: float | None = None
    memory_avg_mb: float | None = None
    memory_avg_pct: float | None = None
    resource_sample_count: int = 0


class _ProcessTreeSampler(threading.Thread):
    """Sample aggregate CPU percentage and RSS for a process tree."""

    def __init__(self, pid: int, interval: float, total_ram_mb: float) -> None:
        super().__init__(daemon=True)
        self._pid = pid
        self._interval = max(interval, 0.05)
        self._total_ram_mb = total_ram_mb
        self._stop_event = threading.Event()
        self._processes: dict[int, psutil.Process] = {}
        self.cpu_samples: list[float] = []
        self.memory_samples_mb: list[float] = []

    def _refresh(self) -> None:
        try:
            root = psutil.Process(self._pid)
        except psutil.Error:
            self._processes.clear()
            return

        live = {root.pid: root}
        with contextlib.suppress(psutil.Error):
            live.update({child.pid: child for child in root.children(recursive=True)})

        for pid, process in live.items():
            if pid not in self._processes:
                with contextlib.suppress(psutil.Error):
                    process.cpu_percent(None)
                self._processes[pid] = process
        self._processes = {
            pid: process for pid, process in self._processes.items() if pid in live
        }

    def run(self) -> None:
        self._refresh()
        while not self._stop_event.wait(self._interval):
            self._refresh()
            cpu_pct = 0.0
            memory_bytes = 0
            for process in list(self._processes.values()):
                with contextlib.suppress(psutil.Error):
                    cpu_pct += process.cpu_percent(None)
                    memory_bytes += process.memory_info().rss
            self.cpu_samples.append(cpu_pct)
            self.memory_samples_mb.append(memory_bytes / (1024 * 1024))

    def stop(self) -> None:
        self._stop_event.set()

    def summary(self) -> dict[str, float | int | None]:
        memory_avg_mb = (
            statistics.fmean(self.memory_samples_mb) if self.memory_samples_mb else None
        )
        return {
            "cpu_avg_pct": (
                round(statistics.fmean(self.cpu_samples), 2) if self.cpu_samples else None
            ),
            "memory_avg_mb": round(memory_avg_mb, 2) if memory_avg_mb is not None else None,
            "memory_avg_pct": (
                round(memory_avg_mb / self._total_ram_mb * 100, 2)
                if memory_avg_mb is not None and self._total_ram_mb > 0
                else None
            ),
            "sample_count": len(self.cpu_samples),
        }


def _kill_process_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return
    processes = [root]
    with contextlib.suppress(psutil.Error):
        processes.extend(root.children(recursive=True))
    for process in reversed(processes):
        with contextlib.suppress(psutil.Error):
            process.kill()
    with contextlib.suppress(Exception):
        psutil.wait_procs(processes, timeout=5)


def _run_process(
    args: list[str],
    *,
    timeout: int,
    sample_interval: float | None = None,
    total_ram_mb: float = 0.0,
) -> ProcessResult:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    started = time.perf_counter()
    process = subprocess.Popen(args, **kwargs)  # noqa: S603
    sampler = (
        _ProcessTreeSampler(process.pid, sample_interval, total_ram_mb)
        if sample_interval is not None
        else None
    )
    if sampler is not None:
        sampler.start()

    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        exit_code = -1
    except BaseException:
        _kill_process_tree(process.pid)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        raise
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.join(timeout=5)

    result = ProcessResult(
        args=args,
        exit_code=exit_code,
        elapsed_s=round(time.perf_counter() - started, 2),
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )
    if sampler is not None:
        summary = sampler.summary()
        result.cpu_avg_pct = cast("float | None", summary["cpu_avg_pct"])
        result.memory_avg_mb = cast("float | None", summary["memory_avg_mb"])
        result.memory_avg_pct = cast("float | None", summary["memory_avg_pct"])
        result.resource_sample_count = int(summary["sample_count"] or 0)
    return result


def _bundle_provider_names(config: dict[str, Any]) -> set[str]:
    """Return every execution-provider name declared in a GenAI config."""
    providers: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "provider_options" and isinstance(child, list):
                    for entry in child:
                        if isinstance(entry, dict):
                            providers.update(str(name).lower() for name in entry)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    return providers


def validate_cpu_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate that a bundle exists and declares no hardware provider."""
    config_path = bundle_dir / "genai_config.json"
    if not config_path.is_file():
        raise ValueError(f"No genai_config.json found under {bundle_dir}")
    config = _load_json(config_path)
    providers = _bundle_provider_names(config)
    non_cpu = providers - {"cpu", "cpuexecutionprovider"}
    if non_cpu:
        names = ", ".join(sorted(non_cpu))
        raise ValueError(
            f"Bundle is not CPU-only; genai_config.json declares provider(s): {names}"
        )
    return config


def _build_bundle(
    *,
    model: str,
    dtype: str,
    mobius_python: Path,
    bundle_dir: Path,
    timeout: int,
) -> ProcessResult:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(mobius_python),
        "-m",
        "mobius",
        "build",
        "--model",
        model,
        str(bundle_dir),
        "--dtype",
        dtype,
        "--runtime",
        "ort-genai",
    ]
    return _run_process(args, timeout=timeout)


def _make_prompt(bundle_dir: Path, target_tokens: int, filler: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir), local_files_only=True)
    repeats = max(16, target_tokens // 4)
    token_ids: list[int] = []
    while len(token_ids) < target_tokens:
        token_ids = tokenizer.encode(filler * repeats, add_special_tokens=False)
        repeats *= 2
    return tokenizer.decode(
        token_ids[:target_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _perf_args(
    *,
    bundle_dir: Path,
    report_path: Path,
    prompt: str,
    max_new_tokens: int,
    iterations: int,
    warmup: int,
) -> list[str]:
    return [
        *WINML_CLI,
        "perf",
        "-m",
        str(bundle_dir),
        "--runtime",
        "winml-genai",
        "--device",
        "cpu",
        "--prompt",
        prompt,
        "--no-apply-template",
        "--max-new-tokens",
        str(max_new_tokens),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--no-compile",
        "-o",
        str(report_path),
        "--overwrite",
    ]


def _distribution(values: list[float]) -> dict[str, float] | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None

    def percentile(percent: float) -> float:
        index = min(int(len(samples) * percent / 100), len(samples) - 1)
        return round(samples[index], 4)

    return {
        "mean": round(statistics.fmean(samples), 4),
        "p50": percentile(50),
        "p90": percentile(90),
        "min": round(samples[0], 4),
        "max": round(samples[-1], 4),
        "std": round(statistics.pstdev(samples), 4) if len(samples) > 1 else 0.0,
    }


def _context_point(
    target_tokens: int,
    report: dict[str, Any],
    process: ProcessResult,
    *,
    expected_max_new_tokens: int,
    expected_iterations: int,
    expected_warmup: int,
) -> dict[str, Any]:
    info = report.get("benchmark_info") or {}
    expected_info = {
        "runtime": "winml-genai",
        "device": "cpu",
        "compile": False,
        "iterations": expected_iterations,
        "warmup": expected_warmup,
        "max_new_tokens": expected_max_new_tokens,
        "apply_template": False,
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            raise ValueError(f"Expected benchmark_info.{key}={expected!r}, got {info.get(key)!r}")
    if str(info.get("ep") or "").lower() not in {"cpu", "cpuexecutionprovider"}:
        raise ValueError(f"Expected CPU execution provider, got {info.get('ep')!r}")

    prompt_tokens = int(info.get("prompt_tokens") or 0)
    if prompt_tokens != target_tokens:
        raise ValueError(
            f"Target context was {target_tokens} tokens, but perf measured {prompt_tokens}"
        )

    generated_tokens = int(info.get("generated_tokens") or 0)
    if generated_tokens != expected_max_new_tokens:
        raise ValueError(
            f"Expected {expected_max_new_tokens} generated tokens, got {generated_tokens}"
        )

    raw = report.get("raw") or {}
    raw_keys = (
        "ttft_ms",
        "prefill_ms",
        "decode_tokens_per_sec",
        "tpot_ms",
        "total_ms",
    )
    for key in raw_keys:
        samples = raw.get(key)
        if not isinstance(samples, list) or len(samples) != expected_iterations:
            actual = len(samples) if isinstance(samples, list) else None
            raise ValueError(
                f"Expected {expected_iterations} raw.{key} samples, got {actual}"
            )

    prefill_ms = float((report.get("prefill_ms") or {}).get("mean") or 0.0)
    return {
        "context_length_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens_per_second": round(
            float((report.get("decode") or {}).get("tokens_per_sec") or 0.0), 4
        ),
        "prefill_tokens_per_second": (
            round(prompt_tokens / (prefill_ms / 1000.0), 4) if prefill_ms > 0 else None
        ),
        "ttft_s": round(float((report.get("ttft_ms") or {}).get("mean") or 0.0) / 1000, 4),
        "generation_compute_s": round(
            float((report.get("total_generation_ms") or {}).get("mean") or 0.0) / 1000,
            4,
        ),
        "tpot_ms": _distribution(raw.get("tpot_ms") or []),
        "raw": {
            "ttft_ms": raw.get("ttft_ms") or [],
            "prefill_ms": raw.get("prefill_ms") or [],
            "decode_tokens_per_second": raw.get("decode_tokens_per_sec") or [],
            "tpot_ms": raw.get("tpot_ms") or [],
            "generation_compute_ms": raw.get("total_ms") or [],
        },
        "process_cpu_avg_pct": process.cpu_avg_pct,
        "process_memory_avg_mb": process.memory_avg_mb,
        "process_memory_avg_pct": process.memory_avg_pct,
        "resource_sample_count": process.resource_sample_count,
    }


def _collect_environment() -> dict[str, Any]:
    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.uname().processor,
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_mb": round(total_ram_mb, 1),
        "python": platform.python_version(),
        "winml_cli": _package_version("winml-cli"),
        "onnxruntime": _package_version("onnxruntime"),
        "onnxruntime_genai": _package_version("onnxruntime-genai")
        or _package_version("onnxruntime-genai-winml"),
    }


def build_result(
    *,
    model: str,
    dtype: str,
    bundle_dir: Path,
    config_sha256: str,
    provider_names: list[str],
    context_lengths: list[int],
    max_new_tokens: int,
    iterations: int,
    warmup: int,
    started_at: str,
    elapsed_s: float,
    points: list[dict[str, Any]],
    errors: list[str],
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "runtime": RUNTIME,
        "precision": dtype,
        "device": "cpu",
        "bundle": {
            "path": str(bundle_dir),
            "genai_config_sha256": config_sha256,
            "provider_names": provider_names,
        },
        "benchmark": {
            "context_lengths": context_lengths,
            "max_new_tokens": max_new_tokens,
            "iterations": iterations,
            "warmup": warmup,
            "prompt_kind": "synthetic repeated filler",
        },
        "environment": environment,
        "run_timestamp": started_at,
        "run": {
            "passed": len(errors) == 0 and len(points) == len(context_lengths),
            "elapsed_s": round(elapsed_s, 2),
            "errors": errors,
        },
        "context_sweep": points,
    }


def _validate_result(result: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID to benchmark.")
    parser.add_argument(
        "--mobius-python",
        type=Path,
        help="Python executable from an environment with Mobius installed.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Existing provider-free GenAI bundle. Skips the Mobius build.",
    )
    parser.add_argument("--reuse-bundle", action="store_true", help="Reuse <output-dir>/bundle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", default="f16", choices=["f16", "f32", "bf16"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--prompt-filler", default=DEFAULT_FILLER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(length <= 0 for length in args.context_lengths):
        raise ValueError("--context-lengths values must be positive")
    if args.iterations <= 0 or args.warmup < 0 or args.max_new_tokens <= 0:
        raise ValueError("iterations and max-new-tokens must be positive; warmup must be >= 0")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = (args.bundle_dir or output_dir / "bundle").resolve()
    started_at = _utc_now()
    started = time.perf_counter()
    errors: list[str] = []

    if args.bundle_dir is not None or args.reuse_bundle:
        bundle_config = validate_cpu_bundle(bundle_dir)
    else:
        if args.mobius_python is None:
            raise ValueError(
                "--mobius-python is required unless --bundle-dir or --reuse-bundle is used"
            )
        mobius_python = args.mobius_python.resolve()
        if not mobius_python.is_file():
            raise FileNotFoundError(f"Mobius Python executable not found: {mobius_python}")
        print(f"Building provider-free bundle: {bundle_dir}")
        build = _build_bundle(
            model=args.model,
            dtype=args.dtype,
            mobius_python=mobius_python,
            bundle_dir=bundle_dir,
            timeout=args.timeout,
        )
        if build.exit_code != 0:
            raise RuntimeError(f"Mobius build failed:\n{_tail(build.stderr or build.stdout)}")
        bundle_config = validate_cpu_bundle(bundle_dir)

    config_path = bundle_dir / "genai_config.json"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    provider_names = sorted(_bundle_provider_names(bundle_config))
    environment = _collect_environment()
    total_ram_mb = float(environment["total_ram_mb"])
    points: list[dict[str, Any]] = []

    for context_length in args.context_lengths:
        print(f"Benchmarking CPU context={context_length} tokens")
        prompt = _make_prompt(bundle_dir, context_length, args.prompt_filler)
        report_path = output_dir / f"perf_ctx{context_length}.json"
        report_path.unlink(missing_ok=True)
        process = _run_process(
            _perf_args(
                bundle_dir=bundle_dir,
                report_path=report_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                iterations=args.iterations,
                warmup=args.warmup,
            ),
            timeout=args.timeout,
            sample_interval=args.sample_interval,
            total_ram_mb=total_ram_mb,
        )
        if process.exit_code != 0 or not report_path.is_file():
            reason = "timeout" if process.timed_out else f"exit {process.exit_code}"
            errors.append(f"context {context_length}: perf {reason}: {_tail(process.stderr)}")
            continue
        try:
            point = _context_point(
                context_length,
                _load_json(report_path),
                process,
                expected_max_new_tokens=args.max_new_tokens,
                expected_iterations=args.iterations,
                expected_warmup=args.warmup,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"context {context_length}: invalid perf report: {exc}")
            continue
        points.append(point)
        print(
            f"  decode={point['decode_tokens_per_second']:.2f} tok/s "
            f"ttft={point['ttft_s']:.3f}s generation={point['generation_compute_s']:.3f}s"
        )

    result = build_result(
        model=args.model,
        dtype=args.dtype,
        bundle_dir=bundle_dir,
        config_sha256=config_sha256,
        provider_names=provider_names,
        context_lengths=args.context_lengths,
        max_new_tokens=args.max_new_tokens,
        iterations=args.iterations,
        warmup=args.warmup,
        started_at=started_at,
        elapsed_s=time.perf_counter() - started,
        points=points,
        errors=errors,
        environment=environment,
    )
    result_path = output_dir / RESULT_FILENAME
    _validate_result(result)
    _write_json(result_path, result)
    status = "PASS" if result["run"]["passed"] else "FAIL"
    print(f"[{status}] {result_path}")
    return 0 if result["run"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
