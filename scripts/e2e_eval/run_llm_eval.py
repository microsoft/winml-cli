# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Run a schema-normalized ONNX Runtime GenAI context sweep.

The runner consumes an existing GenAI bundle and invokes ``winml perf
--runtime ort-genai`` for each requested prompt length. Generation timing
and generation-window resource metrics are normalized into one
``llm_eval_result.json`` document.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


SCHEMA_VERSION = "1.0"
RUNTIME = "onnxruntime-genai"
RESULT_FILENAME = "llm_eval_result.json"
FAILURE_FILENAME = "llm_eval_failure.json"
DEFAULT_FILLER = "The quick brown fox jumps over the lazy dog. "
WINML_CLI = [sys.executable, "-m", "winml.modelkit.cli"]
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "llm_eval_result.schema.json"


class _OutputDirectoryLock:
    """Prevent concurrent runners from sharing mutable result artifacts."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._path = output_dir / ".llm_eval.lock"
        self._fd: int | None = None

    def __enter__(self) -> _OutputDirectoryLock:
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            if platform.system() == "Windows":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                f"Output directory is already in use: {self._output_dir}"
            ) from exc
        self._fd = fd
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fd is None:
            return
        try:
            if platform.system() == "Windows":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str | None, limit: int = 4000) -> str:
    value = (text or "").strip()
    return value if len(value) <= limit else "...(truncated)...\n" + value[-limit:]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


@dataclass
class ProcessResult:
    """Captured subprocess output and timing."""

    args: list[str]
    exit_code: int
    elapsed_s: float
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def command(self) -> str:
        """Return the subprocess arguments as a reproducible command line."""
        return subprocess.list2cmdline(self.args)


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


class _ProcessTreeGuard:
    """Own a subprocess tree through a Windows Job Object or POSIX process group."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._job: int | None = None
        if platform.system() == "Windows":
            self._job = self._create_windows_job(process)

    @staticmethod
    def _create_windows_job(process: subprocess.Popen) -> int:
        import ctypes
        import ctypes.wintypes as wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise ctypes.WinError(error)
        process_handle = wintypes.HANDLE(process._handle)
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise ctypes.WinError(error)
        status = ctypes.WinDLL("ntdll").NtResumeProcess(process_handle)
        if status != 0:
            kernel32.CloseHandle(job)
            raise OSError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
        return int(job)

    def terminate(self) -> None:
        if self._job is not None:
            import ctypes
            import ctypes.wintypes as wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject(wintypes.HANDLE(self._job), 1)
            return
        if platform.system() != "Windows":
            import signal

            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGKILL)
            return
        _kill_process_tree(self._process.pid)

    def close(self) -> None:
        if self._job is None:
            return
        import ctypes
        import ctypes.wintypes as wintypes

        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(wintypes.HANDLE(self._job))
        self._job = None


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _run_process(args: list[str], *, timeout: int) -> ProcessResult:
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
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | 0x00000004  # CREATE_SUSPENDED
    else:
        kwargs["start_new_session"] = True

    started = time.perf_counter()
    process = subprocess.Popen(args, **kwargs)  # noqa: S603
    try:
        tree = _ProcessTreeGuard(process)
    except Exception:
        _kill_process_tree(process.pid)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
        raise
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        exit_code = process.returncode
    except subprocess.TimeoutExpired as initial_timeout:
        timed_out = True
        tree.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as drain_timeout:
            stdout = _timeout_output(drain_timeout.output or initial_timeout.output)
            stderr = _timeout_output(drain_timeout.stderr or initial_timeout.stderr)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    with contextlib.suppress(OSError):
                        pipe.close()
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        exit_code = -1
    except BaseException:
        tree.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        raise
    finally:
        tree.close()

    return ProcessResult(
        args=args,
        exit_code=exit_code,
        elapsed_s=round(time.perf_counter() - started, 2),
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load and minimally validate an ONNX Runtime GenAI bundle."""
    config_path = bundle_dir / "genai_config.json"
    if not config_path.is_file():
        raise ValueError(f"No genai_config.json found under {bundle_dir}")
    return _load_json(config_path)


def _make_prompt(bundle_dir: Path, target_tokens: int, filler: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir), local_files_only=True)
    if not tokenizer.encode(filler, add_special_tokens=False):
        raise ValueError("--prompt-filler must tokenize to at least one token")
    repeats = max(16, target_tokens // 4)
    token_ids: list[int] = []
    previous_count = -1
    while len(token_ids) < target_tokens:
        token_ids = tokenizer.encode(filler * repeats, add_special_tokens=False)
        if len(token_ids) <= previous_count:
            raise ValueError("--prompt-filler token count stopped growing")
        previous_count = len(token_ids)
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
    prompt_path: Path,
    max_new_tokens: int,
    iterations: int,
    warmup: int,
    compile_timeout: int,
    device: str,
    ep: str | None,
) -> list[str]:
    args = [
        *WINML_CLI,
        "perf",
        "-m",
        str(bundle_dir),
        "--runtime",
        "ort-genai",
        "--device",
        device,
        "--prompt-file",
        str(prompt_path),
        "--no-apply-template",
        "--max-new-tokens",
        str(max_new_tokens),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--compile",
        "--compile-timeout",
        str(compile_timeout),
        "--monitor",
        "--no-color",
        "--quiet",
        "-o",
        str(report_path),
        "--overwrite",
    ]
    if ep:
        args.extend(["--ep", ep])
    return args


def _distribution(values: list[float]) -> dict[str, float] | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None

    def percentile(percent: float) -> float:
        index = min(int(len(samples) * percent / 100), len(samples) - 1)
        return round(samples[index], 4)

    return {
        "avg": round(statistics.fmean(samples), 4),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "min": round(samples[0], 4),
        "max": round(samples[-1], 4),
        "std": round(statistics.pstdev(samples), 4) if len(samples) > 1 else 0.0,
    }


def _context_label(tokens: int) -> str:
    return f"{tokens // 1000}k" if tokens >= 1000 and tokens % 1000 == 0 else str(tokens)


def _context_point(
    target_tokens: int,
    report: dict[str, Any],
    *,
    expected_bundle_dir: Path,
    expected_device: str,
    expected_ep: str | None,
    expected_max_new_tokens: int,
    expected_iterations: int,
    expected_warmup: int,
    total_ram_mb: float,
    total_vram_mb: float,
) -> dict[str, Any]:
    info = report.get("benchmark_info") or {}
    reported_bundle_dir = info.get("bundle_dir")
    if not isinstance(reported_bundle_dir, str) or (
        Path(reported_bundle_dir).resolve() != expected_bundle_dir.resolve()
    ):
        raise ValueError(
            f"Expected benchmark_info.bundle_dir={str(expected_bundle_dir)!r}, "
            f"got {reported_bundle_dir!r}"
        )
    expected_info = {
        "runtime": "ort-genai",
        "device": expected_device,
        "effective_device": expected_device,
        "compile": True,
        "iterations": expected_iterations,
        "warmup": expected_warmup,
        "max_new_tokens": expected_max_new_tokens,
        "apply_template": False,
        "monitor": True,
    }
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            raise ValueError(f"Expected benchmark_info.{key}={expected!r}, got {info.get(key)!r}")

    reported_ep = str(info.get("ep") or "")
    if expected_ep:
        from winml.modelkit.utils.constants import normalize_ep_name

        expected_canonical = normalize_ep_name(expected_ep.split("@", 1)[0])
        reported_canonical = normalize_ep_name(reported_ep)
        if reported_canonical != expected_canonical:
            raise ValueError(f"Expected execution provider {expected_ep!r}, got {info.get('ep')!r}")

    prompt_tokens = int(info.get("prompt_tokens") or 0)
    if prompt_tokens != target_tokens:
        raise ValueError(
            f"Target context was {target_tokens} tokens, but perf measured {prompt_tokens}"
        )
    generated_tokens = int(info.get("generated_tokens") or 0)
    if not 0 < generated_tokens <= expected_max_new_tokens:
        raise ValueError(
            f"Expected 1..{expected_max_new_tokens} generated tokens, got {generated_tokens}"
        )

    raw = report.get("raw") or {}
    for key in ("ttft_ms", "prefill_ms", "decode_tokens_per_sec", "tpot_ms", "total_ms"):
        samples = raw.get(key)
        if not isinstance(samples, list) or len(samples) != expected_iterations:
            actual = len(samples) if isinstance(samples, list) else None
            raise ValueError(f"Expected {expected_iterations} raw.{key} samples, got {actual}")

    hw_monitor = report.get("hw_monitor")
    if not isinstance(hw_monitor, dict):
        raise TypeError("perf report has no hw_monitor metrics")
    cpu_metrics = hw_monitor.get("cpu") or {}
    ram_metrics = hw_monitor.get("ram") or {}
    process_cpu_pct = float(cpu_metrics.get("process_mean_pct") or 0.0)
    process_memory_mb = float(ram_metrics.get("mean_mb") or 0.0)
    if int(cpu_metrics.get("sample_count") or 0) <= 0:
        raise ValueError("hw_monitor collected no process CPU samples")
    if process_memory_mb <= 0:
        raise ValueError("hw_monitor collected no process memory samples")

    device_kind = str(hw_monitor.get("device_kind") or "").lower()
    if expected_device == "cpu":
        if device_kind not in ("", "cpu"):
            raise ValueError(
                f"Expected CPU-only monitoring, got device_kind={device_kind!r}"
            )
        accelerator_util_pct = 0.0
        device_memory_mb = 0.0
        device_memory_util_pct = 0.0
    else:
        if device_kind != expected_device:
            raise ValueError(
                f"Expected hw_monitor device_kind={expected_device!r}, got {device_kind!r}"
            )
        adapter_metrics = hw_monitor.get("adapter") or hw_monitor.get(device_kind) or {}
        if int(adapter_metrics.get("sample_count") or 0) <= 0:
            raise ValueError(f"hw_monitor collected no {expected_device} samples")
        accelerator_util_pct = float(adapter_metrics.get("mean_pct") or 0.0)
        memory_metrics = hw_monitor.get("device_memory") or {}
        local_memory_mb = float(memory_metrics.get("local_mean_mb") or 0.0)
        shared_memory_mb = float(memory_metrics.get("shared_mean_mb") or 0.0)
        device_memory_mb = local_memory_mb if local_memory_mb > 0 else shared_memory_mb
        capacity_mb = total_vram_mb if local_memory_mb > 0 and total_vram_mb > 0 else None
        device_memory_util_pct = (
            device_memory_mb / capacity_mb * 100 if capacity_mb is not None else None
        )

    prefill_ms = float((report.get("prefill_ms") or {}).get("mean") or 0.0)
    return {
        "context_length_tokens": target_tokens,
        "context_length_label": _context_label(target_tokens),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(
            float((report.get("decode") or {}).get("tokens_per_sec") or 0.0), 4
        ),
        "prefill_tokens_per_second": (
            round(prompt_tokens / (prefill_ms / 1000.0), 4) if prefill_ms > 0 else None
        ),
        "total_elapsed_s": round(
            float((report.get("total_generation_ms") or {}).get("mean") or 0.0) / 1000,
            4,
        ),
        "ttft_s": round(float((report.get("ttft_ms") or {}).get("mean") or 0.0) / 1000, 4),
        "inter_token_latency_ms": _distribution(raw.get("tpot_ms") or []),
        "gpu_util_avg_pct": round(accelerator_util_pct, 4),
        "vram": {
            "util_avg_pct": (
                round(device_memory_util_pct, 4) if device_memory_util_pct is not None else None
            ),
            "used_avg_mb": round(device_memory_mb, 4),
        },
        "process_cpu_util_avg_pct": round(process_cpu_pct, 4),
        "process_mem": {
            "util_avg_pct": round(process_memory_mb / total_ram_mb * 100, 4),
            "used_avg_mb": round(process_memory_mb, 4),
        },
    }


def _collect_environment(gpu_memory_gb: float | None = None) -> dict[str, Any]:
    total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
    gpu_name: str | None = None
    npu_name: str | None = None
    if platform.system() == "Windows":
        with contextlib.suppress(Exception):
            from winml.modelkit.sysinfo.hardware import GPU, NPU

            gpus = GPU.get_all()
            npus = NPU.get_all()
            if gpus:
                gpu_name = gpus[0].name
            if npus:
                npu_name = npus[0].name
    total_vram_mb = gpu_memory_gb * 1024 if gpu_memory_gb is not None else 0.0
    hardware: dict[str, Any] = {
        "cpu_name": platform.processor() or platform.uname().processor,
        "total_vram_mb": round(total_vram_mb, 1),
        "total_ram_mb": round(total_ram_mb, 1),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
    }
    if gpu_name:
        hardware["gpu_name"] = gpu_name
    if npu_name:
        hardware["npu_name"] = npu_name
    return {
        "os": platform.system().lower(),
        "hardware": hardware,
        "total_ram_mb": round(total_ram_mb, 1),
        "total_vram_mb": round(total_vram_mb, 1),
        "gpu_memory_gb": round(total_vram_mb / 1024, 4) if total_vram_mb > 0 else None,
    }


def build_result(
    *,
    model: str,
    model_type: str | None,
    task: str,
    quantization: str | None,
    device: str,
    ep: str | None,
    group: str | None,
    priority: str | None,
    machine_label: str | None,
    started_at: str,
    elapsed_s: float,
    points: list[dict[str, Any]],
    errors: list[str],
    environment: dict[str, Any],
    command: str | None,
    timed_out: bool,
) -> dict[str, Any]:
    passed = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "model_type": model_type,
        "task": task,
        "runtime": RUNTIME,
        "quantization": quantization,
        "device": device,
        "ep": ep,
        "os": environment["os"],
        "gpu_memory_gb": environment["gpu_memory_gb"],
        "machine_label": machine_label,
        "hardware": environment["hardware"],
        "group": group,
        "priority": priority,
        "eval_types_run": ["perf"],
        "run_timestamp": started_at,
        "run": {
            "passed": passed,
            "elapsed_s": round(elapsed_s, 2),
            "exit_code": 0 if passed else 1,
            "timeout": timed_out,
            "error": "\n".join(errors) if errors else None,
            "command": command,
        },
        "context_sweep": points,
    }


def _validate_result(result: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", type=Path, required=True, help="Existing GenAI bundle.")
    parser.add_argument("--model-id", help="Model identifier recorded in the result.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantization", help="Artifact quantization label, e.g. w8a16 or fp16.")
    parser.add_argument("--model-type", default="llm")
    parser.add_argument("--task", default="text-generation")
    parser.add_argument("--device", choices=["cpu", "gpu", "npu"], default="npu")
    parser.add_argument("--ep", help="Execution provider passed to winml perf, e.g. qnn.")
    parser.add_argument("--group")
    parser.add_argument("--priority")
    parser.add_argument("--machine-label")
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        help="Override dedicated GPU memory capacity used for VRAM percentage.",
    )
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--compile-timeout",
        type=int,
        default=1800,
        help="Maximum seconds to compile each EPContext stage.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--prompt-filler", default=DEFAULT_FILLER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(length <= 0 for length in args.context_lengths):
        raise ValueError("--context-lengths values must be positive")
    if args.iterations <= 0 or args.warmup < 0 or args.max_new_tokens <= 0:
        raise ValueError("iterations and max-new-tokens must be positive; warmup must be >= 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.compile_timeout <= 0:
        raise ValueError("--compile-timeout must be positive")
    if args.gpu_memory_gb is not None and (
        not math.isfinite(args.gpu_memory_gb) or args.gpu_memory_gb <= 0
    ):
        raise ValueError("--gpu-memory-gb must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _OutputDirectoryLock(output_dir):
        return _run(args, output_dir)


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    (output_dir / RESULT_FILENAME).unlink(missing_ok=True)
    (output_dir / FAILURE_FILENAME).unlink(missing_ok=True)
    bundle_dir = args.model.resolve()
    validate_bundle(bundle_dir)
    started_at = _utc_now()
    started = time.perf_counter()
    environment = _collect_environment(args.gpu_memory_gb)
    total_ram_mb = float(environment["total_ram_mb"])
    total_vram_mb = float(environment["total_vram_mb"])
    points: list[dict[str, Any]] = []
    errors: list[str] = []
    commands: list[str] = []
    reported_eps: set[str] = set()
    any_timeout = False

    for context_length in args.context_lengths:
        print(f"Benchmarking {args.device.upper()} context={context_length} tokens")
        prompt = _make_prompt(bundle_dir, context_length, args.prompt_filler)
        prompt_path = output_dir / f"prompt_ctx{context_length}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        report_path = output_dir / f"perf_ctx{context_length}.json"
        report_path.unlink(missing_ok=True)
        process = _run_process(
            _perf_args(
                bundle_dir=bundle_dir,
                report_path=report_path,
                prompt_path=prompt_path,
                max_new_tokens=args.max_new_tokens,
                iterations=args.iterations,
                warmup=args.warmup,
                compile_timeout=args.compile_timeout,
                device=args.device,
                ep=args.ep,
            ),
            timeout=args.timeout,
        )
        commands.append(process.command)
        any_timeout = any_timeout or process.timed_out
        if process.exit_code != 0 or not report_path.is_file():
            reason = "timeout" if process.timed_out else f"exit {process.exit_code}"
            errors.append(f"context {context_length}: perf {reason}: {_tail(process.stderr)}")
            continue
        try:
            report = _load_json(report_path)
            point = _context_point(
                context_length,
                report,
                expected_bundle_dir=bundle_dir,
                expected_device=args.device,
                expected_ep=args.ep,
                expected_max_new_tokens=args.max_new_tokens,
                expected_iterations=args.iterations,
                expected_warmup=args.warmup,
                total_ram_mb=total_ram_mb,
                total_vram_mb=total_vram_mb,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"context {context_length}: invalid perf report: {exc}")
            continue
        reported_ep = str((report.get("benchmark_info") or {}).get("ep") or "").lower()
        if reported_ep and reported_ep != "config":
            reported_eps.add(reported_ep)
        points.append(point)
        print(
            f"  decode={point['tokens_per_second']:.2f} tok/s "
            f"ttft={point['ttft_s']:.3f}s total={point['total_elapsed_s']:.3f}s"
        )

    if not points:
        failure_path = output_dir / FAILURE_FILENAME
        (output_dir / RESULT_FILENAME).unlink(missing_ok=True)
        _write_json(
            failure_path,
            {
                "model": args.model_id or bundle_dir.name,
                "device": args.device,
                "ep": args.ep,
                "errors": errors or ["No context points completed"],
                "run_timestamp": started_at,
            },
        )
        print(f"[FAIL] {failure_path}")
        return 1

    effective_ep = args.ep
    if effective_ep is None and len(reported_eps) == 1:
        effective_ep = next(iter(reported_eps))
    result = build_result(
        model=args.model_id or bundle_dir.name,
        model_type=args.model_type,
        task=args.task,
        quantization=args.quantization,
        device=args.device,
        ep=effective_ep,
        group=args.group,
        priority=args.priority,
        machine_label=args.machine_label,
        started_at=started_at,
        elapsed_s=time.perf_counter() - started,
        points=points,
        errors=errors,
        environment=environment,
        command=" && ".join(commands) if commands else None,
        timed_out=any_timeout,
    )
    result_path = output_dir / RESULT_FILENAME
    _validate_result(result)
    (output_dir / FAILURE_FILENAME).unlink(missing_ok=True)
    _write_json(result_path, result)
    status = "PASS" if result["run"]["passed"] else "FAIL"
    print(f"[{status}] {result_path}")
    return 0 if result["run"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
