# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""E2E evaluation runner — unified perf + accuracy.

Batch-runs winml perf (and optionally winml eval + pytorch baseline) for models
in a JSON registry, writes unified eval_result.json per model, and generates
combined reports.

Strategy B cache sharing: winml perf runs first (build + benchmark, populates
model cache). winml eval then reuses the cache — no redundant build step.

Usage:
    # Perf only (default)
    python scripts/e2e_eval/run_eval.py --priority P0

    # Both perf and accuracy in one batch
    python scripts/e2e_eval/run_eval.py --eval-type both --priority P0

    # Accuracy only (winml perf is skipped; winml eval will build the model if cache is missing)
    python scripts/e2e_eval/run_eval.py --eval-type accuracy --hf-model microsoft/resnet-50

    # Single model
    python scripts/e2e_eval/run_eval.py --hf-model microsoft/resnet-50 --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path


# Ensure utils is importable when invoked directly
sys.path.insert(0, str(Path(__file__).parent))

from utils.accuracy import (
    AccuracyVerdict,
    compute_delta,
    derive_verdict,
    derive_verdicts,
    format_delta,
)
from utils.dataset_config import get_dataset_config, register_from_registry
from utils.registry import ModelEntry, filter_registry, load_registry, make_adhoc_entry
from utils.reporter import (
    build_eval_result,
    classify_result,
    classify_results,
    format_text_summary,
    generate_html_report,
    generate_summary,
    load_result_json,
    write_result_json,
    write_summary_json,
    write_summary_md,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINML_CLI = [sys.executable, "-m", "winml.modelkit.cli"]
BASELINE_SCRIPT = Path(__file__).parent / "run_pytorch_baseline.py"
BASELINE_CACHE_PATH = Path(__file__).parent / "cache" / "baseline_cache.json"
EVAL_DATASETS_CACHE = Path.home() / ".cache" / "winml" / "eval_datasets"
TIMEOUT_SKIP_LIST_PATH = Path(__file__).parent / "cache" / "timeout_skip_list.json"
_DEFAULT_SAMPLES = 1000
_DEFAULT_PRECISION = "w8a16"
_HW_POLL_INTERVAL_MS = 200


def _load_timeout_skip_set() -> set[tuple[str, str]]:
    """Load the timeout skip list as a set of (hf_id, task) tuples."""
    if not TIMEOUT_SKIP_LIST_PATH.exists():
        return set()
    with TIMEOUT_SKIP_LIST_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)
    return {(e["hf_id"], e.get("task", "")) for e in entries}


def _get_timeout_skip_reason(hf_id: str, task: str) -> str:
    """Get the skip reason for a timeout-skipped model."""
    if not TIMEOUT_SKIP_LIST_PATH.exists():
        return "timeout"
    with TIMEOUT_SKIP_LIST_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        if e["hf_id"] == hf_id and e.get("task", "") == task:
            return e.get("reason", "timeout")
    return "timeout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patterns that indicate the disk is full (cross-platform).
_NO_SPACE_PATTERNS = (
    "no space left on device",  # Linux/macOS OSError
    "oserror: [errno 28]",  # Python errno string
    "there is not enough space on the disk",  # Windows
    "winerror 112",  # Windows disk-full error code
    "disk full",
)

_HF_CACHE = Path.home() / ".cache" / "huggingface"
_WML_CACHE = Path.home() / ".cache" / "winml"
_TEMP_DIR = Path(os.environ.get("TEMP", os.environ.get("TMP", tempfile.gettempdir())))
_TEMP_PREFIXES = ("wmk_", "modelkit_compat_")


def _is_no_space_error(proc: dict) -> bool:
    """Return True if subprocess output indicates a disk-full condition."""
    combined = (proc.get("stdout", "") + proc.get("stderr", "")).lower()
    return any(pat in combined for pat in _NO_SPACE_PATTERNS)


def _clear_disk_caches() -> None:
    """Delete HuggingFace, WML cache directories and leaked temp files."""
    for cache_dir in (_HF_CACHE, _WML_CACHE):
        if cache_dir.exists():
            safe_print(f"  [cleanup] Removing cache: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                safe_print(f"  [cleanup] Removed: {cache_dir}")
            except OSError as exc:
                safe_print(f"  [cleanup] Warning: could not remove {cache_dir}: {exc}")

    # Clean leaked temp directories/files (wmk_*, modelkit_compat_*, tmp*.onnx*)
    if _TEMP_DIR.is_dir():
        cleaned = 0
        for entry in _TEMP_DIR.iterdir():
            name = entry.name
            should_clean = False
            if any(name.startswith(p) for p in _TEMP_PREFIXES):
                should_clean = (
                    entry.is_dir()
                    or entry.suffix in (".onnx", ".out", ".err")
                    or name.endswith(".onnx.data")
                )
            elif name.startswith("tmp") and name.endswith((".onnx", ".onnx.data")):
                # Python tempfile creates tmp* prefixed files; only clean ONNX artifacts
                should_clean = True
            if should_clean:
                safe_print(f"  [cleanup] Leaked temp: {entry}")
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    cleaned += 1
                except OSError:
                    pass  # Best-effort cleanup; ignore if file is locked or already removed
        if cleaned:
            safe_print(f"  [cleanup] Removed {cleaned} leaked temp entries from {_TEMP_DIR}")


def safe_print(text: str) -> None:
    """Cross-platform safe print (handles Windows Unicode issues)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children.

    On Windows, taskkill /T may miss grandchildren spawned without job objects.
    We use psutil if available for reliable tree kill, falling back to taskkill.
    """
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        with contextlib.suppress(psutil.NoSuchProcess):
            parent.kill()
        # Wait briefly for processes to terminate
        psutil.wait_procs([*children, parent], timeout=5)
    except (ImportError, psutil.NoSuchProcess):
        # Fallback: taskkill on Windows, killpg on Unix
        if platform.system() == "Windows":
            subprocess.run(  # noqa: S603
                ["taskkill", "/F", "/T", "/PID", str(pid)],  # noqa: S607
                capture_output=True,
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already exited; nothing to kill


# ---------------------------------------------------------------------------
# Windows Job Object — caps a subprocess tree's committed memory so a
# runaway model build (e.g. QNN w8a16 calibration on a vision-encoder-decoder)
# can't thrash a low-RAM host into total unresponsiveness.
# ---------------------------------------------------------------------------

# Module-level cap. Set once from CLI args via _set_subprocess_memory_limit_gb.
_SUBPROCESS_MEMORY_LIMIT_BYTES: int | None = None


def _set_subprocess_memory_limit_gb(gb: float | None) -> None:
    """Set (or clear) the per-subprocess Windows Job Object memory cap.

    When set on Windows, every subprocess launched via ``_run_subprocess`` is
    wrapped in a Job Object with ``JOB_OBJECT_LIMIT_JOB_MEMORY``.  If the
    process tree exceeds the cap, the OS refuses further commits — the
    offender typically crashes with a clean OOM, and the host stays
    responsive so the next model can run.

    Passing ``None`` (or a non-positive value) disables the cap.  No-op on
    non-Windows.
    """
    global _SUBPROCESS_MEMORY_LIMIT_BYTES
    _SUBPROCESS_MEMORY_LIMIT_BYTES = None if gb is None or gb <= 0 else int(gb * (1024**3))


class _JobMemoryCap:
    """Best-effort Windows Job Object that caps a subprocess tree's commit.

    Gracefully degrades to a no-op when:
      * The host is not Windows;
      * ``ctypes`` / kernel32 setup fails;
      * The parent process already lives in a non-nestable Job Object (e.g.
        some ADO Agent configurations) and ``AssignProcessToJobObject``
        returns ``ERROR_ACCESS_DENIED``;
      * The target process has already exited before we can assign.

    Detection via ``query_peak_bytes()`` reads ``PeakJobMemoryUsed`` from
    ``JobObjectExtendedLimitInformation`` (works on all supported Windows
    versions).
    """

    # JobObjectInformationClass enum values
    _JobObjectExtendedLimitInformation = 9
    # LimitFlags bits
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    # OpenProcess access rights
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        self.assign_failed = False
        self.assign_error: str | None = None
        self._job_handle: int | None = None
        self._proc_handle: int | None = None
        self._kernel32 = None
        self._info_class = None

        if platform.system() != "Windows":
            return

        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:
            return

        class _IO_COUNTERS(ctypes.Structure):  # noqa: N801 (Win32 struct name)
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 (Win32 struct name)
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_void_p),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 (Win32 struct name)
            _fields_ = [
                ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._info_class = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        try:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except OSError:
            return

        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        h = self._kernel32.CreateJobObjectW(None, None)
        if not h:
            self.assign_error = f"CreateJobObjectW failed (err={ctypes.get_last_error()})"
            return

        info = self._info_class()
        info.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_JOB_MEMORY | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.JobMemoryLimit = ctypes.c_size_t(limit_bytes)
        ok = self._kernel32.SetInformationJobObject(
            h,
            self._JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self.assign_error = f"SetInformationJobObject failed (err={ctypes.get_last_error()})"
            self._kernel32.CloseHandle(h)
            return

        self._job_handle = h

    @property
    def active(self) -> bool:
        """True when the job is created and ready for assign()."""
        return self._job_handle is not None

    def assign(self, pid: int) -> bool:
        """Open the process by PID and add it to the job. Returns False on failure."""
        if self._job_handle is None or self._kernel32 is None:
            return False
        import ctypes

        proc_h = self._kernel32.OpenProcess(
            self._PROCESS_SET_QUOTA | self._PROCESS_TERMINATE,
            False,
            pid,
        )
        if not proc_h:
            self.assign_error = f"OpenProcess({pid}) failed (err={ctypes.get_last_error()})"
            self.assign_failed = True
            return False
        ok = self._kernel32.AssignProcessToJobObject(self._job_handle, proc_h)
        if not ok:
            self.assign_error = (
                f"AssignProcessToJobObject failed (err={ctypes.get_last_error()})"
            )
            self.assign_failed = True
            self._kernel32.CloseHandle(proc_h)
            return False
        self._proc_handle = proc_h
        return True

    def query_peak_bytes(self) -> int | None:
        """Read ``PeakJobMemoryUsed`` from the job. Returns ``None`` on failure."""
        if self._job_handle is None or self._kernel32 is None or self._info_class is None:
            return None
        import ctypes

        info = self._info_class()
        ok = self._kernel32.QueryInformationJobObject(
            self._job_handle,
            self._JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        )
        if not ok:
            return None
        return int(info.PeakJobMemoryUsed)

    def close(self) -> None:
        """Close handles. ``KILL_ON_JOB_CLOSE`` reaps any survivors."""
        if self._kernel32 is None:
            return
        if self._proc_handle is not None:
            with contextlib.suppress(OSError):
                self._kernel32.CloseHandle(self._proc_handle)
            self._proc_handle = None
        if self._job_handle is not None:
            with contextlib.suppress(OSError):
                self._kernel32.CloseHandle(self._job_handle)
            self._job_handle = None


def _start_hw_monitor() -> object | None:
    """Start an HWMonitor for the duration of a subprocess. Returns the live monitor or None.

    Captures system-wide CPU/RAM/NPU metrics from the parent process so even
    a crashing child leaves data behind. ``None`` is returned when the monitor
    is unavailable (non-Windows or import failure) so callers can no-op.
    """
    try:
        from winml.modelkit.session import HWMonitor
    except ImportError:
        return None
    if not HWMonitor.is_available():
        return None
    try:
        monitor = HWMonitor(poll_interval_ms=_HW_POLL_INTERVAL_MS)
        monitor.__enter__()
        return monitor
    except Exception as exc:
        safe_print(f"  [monitor] failed to start: {exc}")
        return None


def _stop_hw_monitor(monitor: object | None) -> dict | None:
    """Stop a live HWMonitor and return its summary dict (or None on error)."""
    if monitor is None:
        return None
    try:
        monitor.__exit__(None, None, None)
        return monitor.to_dict()  # type: ignore[attr-defined]
    except Exception as exc:
        safe_print(f"  [monitor] failed to collect metrics: {exc}")
        return None


def _merge_hw_monitor(parts: list[dict]) -> dict:
    """Merge multiple HWMonitor dicts (e.g. composite sub-runs).

    Peaks take ``max`` across parts; means take a sample-count-weighted average
    when counts are present, falling back to a simple average otherwise.
    Sample counts and ``running_time_ns`` are summed.
    """

    def _peak(section: str, key: str) -> float:
        return max((p.get(section, {}).get(key, 0.0) or 0.0) for p in parts)

    def _weighted_mean(section: str, mean_key: str, count_key: str) -> float:
        total_w = 0.0
        total = 0.0
        fallback = []
        for p in parts:
            sec = p.get(section, {})
            mean = sec.get(mean_key, 0.0) or 0.0
            count = sec.get(count_key, 0) or 0
            if count > 0:
                total += mean * count
                total_w += count
            fallback.append(mean)
        if total_w > 0:
            return total / total_w
        return sum(fallback) / len(fallback) if fallback else 0.0

    return {
        "monitor": "HWMonitor",
        "npu_luid": next((p.get("npu_luid") for p in parts if p.get("npu_luid")), None),
        "merged_from": len(parts),
        "cpu": {
            "mean_pct": round(_weighted_mean("cpu", "mean_pct", "sample_count"), 2),
            "peak_pct": round(_peak("cpu", "peak_pct"), 2),
            "sample_count": sum(p.get("cpu", {}).get("sample_count", 0) for p in parts),
        },
        "ram": {
            "used_mb": round(max(p.get("ram", {}).get("used_mb", 0.0) for p in parts), 2),
            "peak_mb": round(_peak("ram", "peak_mb"), 2),
        },
        "npu": {
            "mean_pct": round(_weighted_mean("npu", "mean_pct", "sample_count"), 2),
            "peak_pct": round(_peak("npu", "peak_pct"), 2),
            "sample_count": sum(p.get("npu", {}).get("sample_count", 0) for p in parts),
        },
        "device_memory": {
            "local_peak_mb": round(_peak("device_memory", "local_peak_mb"), 2),
            "shared_peak_mb": round(_peak("device_memory", "shared_peak_mb"), 2),
        },
        "running_time_ns": sum(p.get("running_time_ns", 0) for p in parts),
    }


def _collect_op_traces(model_dir: Path) -> list[dict]:
    """Load all ``*_op_trace.json`` files written by ``wmk perf --op-tracing``.

    Returns one dict per file (already a structured ``OpTraceResult.to_dict()``
    payload). Returns an empty list if no traces were produced (e.g. tracing
    was disabled, or the underlying perf run failed before tracing).
    """
    traces: list[dict] = []
    for trace_path in sorted(model_dir.glob("*_op_trace.json")):
        try:
            with trace_path.open(encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = trace_path.name
            traces.append(data)
        except (OSError, json.JSONDecodeError) as exc:
            safe_print(f"  [op-trace] failed to load {trace_path.name}: {exc}")
    return traces


def _format_hw_summary(perf_proc: dict | None, accuracy_result: dict | None) -> str:
    """Build a one-line HW peak summary across whichever phases produced data."""
    parts: list[dict] = []
    if perf_proc and perf_proc.get("hw_monitor"):
        parts.append(perf_proc["hw_monitor"])
    if accuracy_result and accuracy_result.get("hw_monitor"):
        parts.append(accuracy_result["hw_monitor"])
    if not parts:
        return ""
    merged = _merge_hw_monitor(parts) if len(parts) > 1 else parts[0]
    ram = merged.get("ram", {}).get("peak_mb", 0.0)
    cpu = merged.get("cpu", {}).get("peak_pct", 0.0)
    npu = merged.get("npu", {}).get("peak_pct", 0.0)
    npu_mem = merged.get("device_memory", {}).get("local_peak_mb", 0.0)
    bits: list[str] = []
    if ram:
        bits.append(f"RAM={ram / 1024:.1f}GB")
    if cpu:
        bits.append(f"CPU={cpu:.0f}%")
    if npu:
        bits.append(f"NPU={npu:.0f}%")
    if npu_mem:
        bits.append(f"NPUmem={npu_mem:.0f}MB")
    return f"  hw[{' '.join(bits)}]" if bits else ""


def _run_subprocess(args: list[str], timeout: int, monitor: bool = False) -> dict:
    """Run a subprocess with three-layer timeout protection.

    Returns a dict with: stdout, stderr, exit_code, elapsed, timeout, command.
    When ``monitor`` is True, the dict also carries an ``hw_monitor`` key with
    the HWMonitor.to_dict() summary captured during the subprocess lifetime.

    Windows fix: On Windows, child processes can inherit pipe handles, causing
    ``proc.communicate()`` to block indefinitely even after ``taskkill`` kills
    the process tree.  We work around this by:
    1. Using ``CREATE_NO_WINDOW`` to prevent console inheritance issues.
    2. Reading stdout/stderr in background threads so the main thread can
       enforce the timeout independently of pipe EOF.
    3. Using a hard watchdog timer that forcefully closes pipes.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    timed_out = False
    hw_handle = _start_hw_monitor() if monitor else None

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
    }
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, **popen_kwargs)  # noqa: S603

    # Wrap the subprocess tree in a Job Object with a memory cap so a runaway
    # build can't thrash the host. Must happen ASAP after Popen so the cap
    # applies to (almost) all grandchildren — they only inherit job membership
    # if the parent is in the job at the moment of CreateProcess.
    job_cap: _JobMemoryCap | None = None
    if _SUBPROCESS_MEMORY_LIMIT_BYTES is not None:
        job_cap = _JobMemoryCap(_SUBPROCESS_MEMORY_LIMIT_BYTES)
        if job_cap.active:
            if not job_cap.assign(proc.pid):
                safe_print(
                    f"  [memory-cap] WARNING: could not enforce "
                    f"{_SUBPROCESS_MEMORY_LIMIT_BYTES / (1024**3):.1f} GB cap "
                    f"({job_cap.assign_error}). Continuing without cap."
                )
        elif job_cap.assign_error:
            safe_print(f"  [memory-cap] disabled: {job_cap.assign_error}")

    # Read pipes in background threads so communicate() timeout works even
    # when grandchild processes keep pipe handles alive (Windows issue).
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _reader(pipe, dest: list[bytes]) -> None:
        try:
            while True:
                chunk = pipe.read(8192)
                if not chunk:
                    break
                dest.append(chunk)
        except (OSError, ValueError):
            pass  # Pipe closed or broken; stop reading

    stdout_thread = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True)
    stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    def _watchdog() -> None:
        try:
            _kill_process_tree(proc.pid)
            proc.kill()
            for pipe in (proc.stdout, proc.stderr):
                if pipe:
                    try:
                        pipe.close()
                    except OSError:
                        pass  # Pipe already closed
        except Exception:
            pass  # Best-effort cleanup; ignore all errors in watchdog

    watchdog = threading.Timer(timeout + 30, _watchdog)
    watchdog.daemon = True
    watchdog.start()

    try:
        proc.wait(timeout=timeout)
        exit_code = proc.returncode
        # Give reader threads a moment to finish draining
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        proc.kill()
        # Give threads a short time to drain after kill
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        exit_code = -1
        timed_out = True
    except KeyboardInterrupt:
        safe_print("\n  [Ctrl+C] Killing subprocess...")
        _kill_process_tree(proc.pid)
        proc.kill()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        raise
    finally:
        watchdog.cancel()
        # Force-close pipes to unblock any stuck reader threads
        for pipe in (proc.stdout, proc.stderr):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass  # Pipe already closed
        # Final attempt: if reader threads are still alive after pipe close,
        # don't block forever — just proceed with whatever was collected.
        if stdout_thread.is_alive():
            stdout_thread.join(timeout=2)
        if stderr_thread.is_alive():
            stderr_thread.join(timeout=2)
        # Always stop the HW monitor here so its background thread is reaped
        # even when the outer try raises (e.g. KeyboardInterrupt).  Capture
        # the metrics so the post-finally code can attach them to the result.
        _hw_metrics_final = _stop_hw_monitor(hw_handle)
        hw_handle = None
        # Read peak commit and release the job handle here too — KILL_ON_JOB_CLOSE
        # will reap any survivors so the parent never leaks them.
        _peak_bytes_final: int | None = None
        if job_cap is not None:
            _peak_bytes_final = job_cap.query_peak_bytes()
            job_cap.close()

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    elapsed = round(time.perf_counter() - start, 1)
    hw_metrics = _hw_metrics_final

    result = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "elapsed": elapsed,
        "timeout": timed_out,
        "command": " ".join(str(a) for a in args),
    }
    if hw_metrics is not None:
        result["hw_monitor"] = hw_metrics
    if _SUBPROCESS_MEMORY_LIMIT_BYTES is not None:
        mem_info: dict = {"limit_bytes": _SUBPROCESS_MEMORY_LIMIT_BYTES}
        if _peak_bytes_final is not None:
            mem_info["peak_bytes"] = _peak_bytes_final
            # Heuristic: peak >= 95% of cap implies the cap was very likely
            # the cause of a non-zero exit (commits were rejected).
            if (
                exit_code != 0
                and _peak_bytes_final >= int(_SUBPROCESS_MEMORY_LIMIT_BYTES * 0.95)
            ):
                mem_info["limit_hit"] = True
        if job_cap is not None and job_cap.assign_failed:
            mem_info["assign_failed"] = True
            if job_cap.assign_error:
                mem_info["assign_error"] = job_cap.assign_error
        result["memory_cap"] = mem_info

    # Retry once after clearing caches if the failure was due to disk full.
    if exit_code != 0 and not timed_out and _is_no_space_error(result):
        safe_print("  [disk-full] Detected 'no space left' — clearing caches and retrying...")
        _clear_disk_caches()
        safe_print(f"  [disk-full] Retrying: {result['command']}")
        result = _run_subprocess(args, timeout, monitor=monitor)

    return result


# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------


def _run_build(
    entry: ModelEntry,
    device: str,
    precision: str,
    timeout: int,
    model_dir: Path,
    ep: str | None = None,
    monitor: bool = False,
) -> dict:
    """Run winml config + winml build for one model. Returns build result dict.

    Flow: winml config → list of config JSONs → winml build each → ONNX paths.

    Single models produce one config; composite models (e.g., T5 translation)
    produce one per sub-component (suffixed names). Both go through the same
    build loop — single model is just the list-of-1 case.
    """
    config_path = model_dir / "build_config.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale suffixed sub-configs BEFORE `wmk config` runs.
    # For composite models `wmk config` writes files matching {stem}_*.json
    # (e.g., build_config_encoder.json); cleaning those AFTER the command would
    # delete the freshly-written configs and silently degrade composite builds
    # to single-model. Running cleanup first removes prior-run artifacts without
    # touching the current run's output.
    for _stale in config_path.parent.glob(f"{config_path.stem}_*.json"):
        safe_print(f"    [config] Removing stale sub-config from prior run: {_stale.name}")
        _stale.unlink(missing_ok=True)

    # Step 1: winml config
    config_args = [
        *WINML_CLI,
        "config",
        "-m",
        entry.hf_id,
        "--device",
        device,
        "--precision",
        precision,
        "-o",
        str(config_path),
    ]
    if entry.task:
        config_args += ["--task", entry.task]
    if ep:
        config_args += ["--ep", ep]

    config_proc = _run_subprocess(config_args, timeout, monitor=monitor)
    if config_proc["exit_code"] != 0:
        return {
            "success": False,
            "onnx_paths": {},
            "stage": "config",
            "proc": config_proc,
        }

    # Collect config files: composite models produce suffixed files
    # (e.g., build_config_encoder.json); single models produce config_path itself.
    sub_configs = sorted(config_path.parent.glob(f"{config_path.stem}_*.json"))
    if not sub_configs:
        sub_configs = [config_path]

    # Step 2: build each sub-config
    # Map component label → ONNX path. Single model uses "" as label.
    onnx_paths: dict[str, str] = {}
    last_proc = config_proc

    # TODO: remove for loop once wimnl build supports building composite model to multiple onnx files
    for sub_cfg in sub_configs:
        label = sub_cfg.stem.removeprefix(f"{config_path.stem}_") if len(sub_configs) > 1 else ""
        if label:
            safe_print(f"    building component: {label}")

        build_args = [
            *WINML_CLI,
            "build",
            "-c",
            str(sub_cfg),
            "-m",
            entry.hf_id,
            "--use-cache",
        ]

        build_proc = _run_subprocess(build_args, timeout, monitor=monitor)
        last_proc = build_proc
        if build_proc["exit_code"] != 0:
            stage = f"build_{label}" if label else "build"
            return {
                "success": False,
                "onnx_paths": onnx_paths,
                "stage": stage,
                "proc": build_proc,
            }

        task_hint = _extract_task_from_config(sub_cfg) or entry.task
        path = _extract_onnx_path(build_proc, entry.hf_id, task_hint)
        if path:
            onnx_paths[label] = path

    return {
        "success": len(onnx_paths) == len(sub_configs),
        "onnx_paths": onnx_paths,
        "stage": "complete",
        "proc": last_proc,
    }


def _extract_onnx_path(build_proc: dict, hf_id: str, task: str | None) -> str | None:
    """Extract ONNX path from build subprocess output."""
    # Patterns used by winml build to report the artifact path
    markers = ("Final artifact:", "Existing artifact found:", "Artifact:")
    onnx_path = None
    for line in (build_proc["stderr"] + build_proc["stdout"]).splitlines():
        for marker in markers:
            if marker in line:
                candidate = line.split(marker)[-1].strip()
                if candidate and Path(candidate).exists():
                    onnx_path = candidate
                    break
        if onnx_path:
            break

    if not onnx_path or not Path(onnx_path).exists():
        onnx_path = _find_cached_model(hf_id, build_proc, task)

    return onnx_path


def _extract_task_from_config(config_path: Path) -> str | None:
    """Read the task from a build config JSON file."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        loader = data.get("loader", {})
        return loader.get("task")
    except (OSError, json.JSONDecodeError):
        return None


def _find_cached_model(hf_id: str, build_proc: dict, task: str | None = None) -> str | None:
    """Try to find the built ONNX model in the WinML cache.

    Requires task to safely identify the correct artifact when a model has
    multiple cached tasks (e.g. feat_* and txtcls_*). Returns None if task is
    not provided to avoid picking the wrong model.
    """
    if not task:
        return None

    slug = hf_id.replace("/", "_").replace("\\", "_")
    cache_dir = Path.home() / ".cache" / "winml" / "artifacts" / slug
    if not cache_dir.exists():
        return None

    from winml.modelkit.loader.task import get_task_abbrev

    prefix = get_task_abbrev(task) + "_"

    model_files = sorted(
        (p for p in cache_dir.glob("*_model.onnx") if p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(model_files[0]) if model_files else None


# ---------------------------------------------------------------------------
# Perf phase
# ---------------------------------------------------------------------------


def run_model(
    entry: ModelEntry,
    device: str,
    timeout: int,
    onnx_paths: dict[str, str] | None = None,
    ep: str | None = None,
    monitor: bool = False,
    op_tracing: str | None = None,
    model_dir: Path | None = None,
) -> dict:
    """Execute winml perf for one or more ONNX models. Returns merged result dict.

    When onnx_paths is provided, benchmarks each pre-built ONNX directly.
    Single model is the {"": path} case. Results are merged (worst exit
    code, concatenated stdout/stderr, summed elapsed).

    When ``op_tracing`` is set ("basic" or "detail"), passes ``--op-tracing``
    to ``wmk perf``; the resulting ``*_op_trace.json`` files (one per sub-run)
    are loaded and attached under ``result['op_trace']`` as a list of dicts.
    Requires QNN EP — a non-QNN EP triggers a SystemExit inside ``wmk perf``.
    """
    if not onnx_paths:
        # No pre-built paths: fall back to HF model ID (single model only)
        args = [
            *WINML_CLI,
            "perf",
            "-m",
            entry.hf_id,
            "--device",
            device,
            "--precision",
            _DEFAULT_PRECISION,
        ]
        if entry.task:
            args += ["--task", entry.task]
        if ep:
            args += ["--ep", ep]
        args += ["--iterations", "10", "--warmup", "2"]
        if model_dir is not None:
            args += ["--output", str(model_dir / "perf_output.json")]
        if op_tracing:
            args += ["--op-tracing", op_tracing]
        args += entry.perf_args

        proc = _run_subprocess(args, timeout, monitor=monitor)
        proc["device"] = device
        proc["timestamp"] = _utc_now()
        proc["error_summary"] = (
            ""
            if proc["exit_code"] == 0
            else f"timeout ({timeout}s)"
            if proc["timeout"]
            else f"exit code {proc['exit_code']}"
        )
        if op_tracing and model_dir is not None:
            traces = _collect_op_traces(model_dir)
            if traces:
                proc["op_trace"] = traces
        return proc

    # Run perf for each sub-model and merge results
    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_elapsed = 0.0
    worst_exit = 0
    any_timeout = False
    commands: list[str] = []
    sub_hw_metrics: list[dict] = []

    for label, path in onnx_paths.items():
        if label:
            safe_print(f"    perf: {label}")

        args = [*WINML_CLI, "perf", "-m", path, "--device", device]
        if ep:
            args += ["--ep", ep]
        args += ["--iterations", "10", "--warmup", "2"]
        if model_dir is not None:
            # Per-sub-run output: keeps op_trace files distinct for composite models.
            suffix = f"_{label}" if label else ""
            args += ["--output", str(model_dir / f"perf_output{suffix}.json")]
        if op_tracing:
            args += ["--op-tracing", op_tracing]
        args += entry.perf_args

        proc = _run_subprocess(args, timeout, monitor=monitor)
        if label:
            all_stdout.append(f"=== {label} ===\n{proc['stdout']}")
            all_stderr.append(f"=== {label} ===\n{proc['stderr']}")
        else:
            all_stdout.append(proc["stdout"])
            all_stderr.append(proc["stderr"])
        total_elapsed += proc["elapsed"]
        commands.append(proc["command"])
        if proc["exit_code"] != 0:
            worst_exit = proc["exit_code"]
        if proc["timeout"]:
            any_timeout = True
        if proc.get("hw_monitor"):
            sub_hw_metrics.append(proc["hw_monitor"])

    merged: dict = {
        "stdout": "\n".join(all_stdout),
        "stderr": "\n".join(all_stderr),
        "exit_code": worst_exit,
        "elapsed": round(total_elapsed, 1),
        "timeout": any_timeout,
        "command": commands[0] if len(commands) == 1 else " | ".join(commands),
        "device": device,
        "timestamp": _utc_now(),
        "error_summary": (
            ""
            if worst_exit == 0
            else f"timeout ({timeout}s)"
            if any_timeout
            else f"exit code {worst_exit}"
        ),
    }
    if sub_hw_metrics:
        merged["hw_monitor"] = _merge_hw_monitor(sub_hw_metrics)
    if op_tracing and model_dir is not None:
        traces = _collect_op_traces(model_dir)
        if traces:
            merged["op_trace"] = traces
    return merged


# ---------------------------------------------------------------------------
# Accuracy phase helpers
# ---------------------------------------------------------------------------


def _parse_metric_from_stdout(stdout: str) -> dict | None:
    """Find the last valid JSON object with a 'value' key in stdout."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "value" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _parse_metric_from_winml_output(
    output_path: Path, metric_name: str, num_samples: int
) -> dict | None:
    """Parse winml eval --output JSON file into the canonical metric dict."""
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = data.get("metrics", {})
    value = metrics.get(metric_name)
    if value is None:
        return None
    return {"metric": metric_name, "value": float(value), "num_samples": num_samples}


def _build_dataset(ds_config: dict, timeout: int) -> None:
    """If ds_config has a build_script, run it to build the dataset on disk.

    The dataset path is already set in ds_config["dataset"] (from the JSON
    registry).  This function only ensures the data exists on disk.
    """
    build_script = ds_config.get("build_script")
    if not build_script:
        return

    script_path = Path(build_script)
    cache_dir = Path(ds_config.get("dataset", EVAL_DATASETS_CACHE / script_path.stem))

    if (cache_dir / "dataset_info.json").exists():
        safe_print(f"    dataset: cached ({cache_dir})")
    else:
        safe_print(f"    dataset: building via {script_path.name} ...")
        proc = _run_subprocess(
            [sys.executable, str(script_path), "--output", str(cache_dir)],
            timeout,
        )
        if proc["exit_code"] != 0:
            safe_print(f"    dataset build FAILED (exit {proc['exit_code']})")
            for line in proc["stderr"].strip().splitlines()[-5:]:
                safe_print(f"      {line}")


def _run_winml_eval(
    entry: ModelEntry,
    device: str,
    timeout: int,
    ds_config: dict,
    model_dir: Path,
    onnx_paths: dict[str, str] | None = None,
    ep: str | None = None,
    monitor: bool = False,
) -> dict:
    """Invoke winml eval for one model. Returns process result + parsed metric."""
    output_path = model_dir / "winml_eval_output.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    # winml eval requires explicit device ('cpu'/'gpu'/'npu'); 'auto' is not accepted
    eval_device = "npu" if device == "auto" else device
    if onnx_paths:
        args = [
            *WINML_CLI,
            "eval",
            "--model-id",
            entry.hf_id,
            "--device",
            eval_device,
        ]
        # Single model uses {"": path}; composite uses {role: path, ...}.
        for label, path in onnx_paths.items():
            args += ["-m", f"{label}={path}" if label else path]
    else:
        args = [
            *WINML_CLI,
            "eval",
            "-m",
            entry.hf_id,
            "--device",
            eval_device,
        ]
    if entry.task:
        args += ["--task", entry.task]
    if ep:
        args += ["--ep", ep]
    # When ds_config is provided, pass explicit dataset args;
    # otherwise winml eval uses its built-in task defaults.
    if ds_config.get("dataset"):
        args += ["--dataset", ds_config["dataset"]]
    if ds_config.get("split"):
        args += ["--split", ds_config["split"]]
    num_samples = ds_config.get("num_samples", _DEFAULT_SAMPLES)
    args += ["--samples", str(num_samples)]
    if ds_config.get("dataset_config"):
        args += ["--dataset-name", ds_config["dataset_config"]]
    for k, v in ds_config.get("columns_mapping", {}).items():
        args += ["--column", f"{k}={v}"]
    if ds_config.get("label_mapping_file"):
        args += ["--label-mapping", ds_config["label_mapping_file"]]
    if ds_config.get("streaming"):
        args += ["--streaming"]
    args += ["--output", str(output_path)]
    args += entry.eval_args

    proc = _run_subprocess(args, timeout, monitor=monitor)

    metric = None
    if proc["exit_code"] == 0 and output_path.exists():
        winml_key = ds_config.get("winml_metric_key") or ds_config.get("metric", "accuracy")
        num_samples = ds_config.get("num_samples", _DEFAULT_SAMPLES)
        metric = _parse_metric_from_winml_output(output_path, winml_key, num_samples)
    status = "PASS" if (proc["exit_code"] == 0 and metric is not None) else "FAIL"

    return {
        "status": status,
        "metric": metric,
        "exit_code": proc["exit_code"],
        "stdout": proc["stdout"],
        "stderr": proc["stderr"],
        "elapsed": proc["elapsed"],
        "timeout": proc["timeout"],
        "command": proc["command"],
        "hw_monitor": proc.get("hw_monitor"),
    }


# ---------------------------------------------------------------------------
# Baseline cache
# ---------------------------------------------------------------------------


def _baseline_cache_key(hf_id: str, task: str, ds_config: dict) -> str:
    """Build a deterministic cache key from model id, task, and dataset params."""
    parts = [
        hf_id,
        task,
        ds_config.get("dataset", ""),
        ds_config.get("dataset_config", ""),
        ds_config.get("split", ""),
        str(ds_config.get("num_samples", _DEFAULT_SAMPLES)),
    ]
    return "|".join(parts)


def _load_baseline_cache() -> dict:
    """Load baseline cache from disk. Returns {} on any error."""
    try:
        return json.loads(BASELINE_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_baseline_cache(cache: dict) -> None:
    """Persist baseline cache to disk."""
    BASELINE_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _lookup_baseline_cache(hf_id: str, task: str, ds_config: dict) -> dict | None:
    """Return cached baseline result dict, or None if not cached."""
    cache = _load_baseline_cache()
    key = _baseline_cache_key(hf_id, task, ds_config)
    entry = cache.get(key)
    if entry and isinstance(entry, dict) and entry.get("status") == "PASS":
        return entry
    return None


def _shorten_command(cmd: str) -> str:
    """Strip absolute paths from a command string for portable caching."""
    parts = cmd.split()
    shortened = []
    for p in parts:
        # Replace absolute python/script paths with just the filename
        if os.sep in p or (os.altsep and os.altsep in p):
            shortened.append(Path(p).name)
        else:
            shortened.append(p)
    return " ".join(shortened)


def _store_baseline_cache(hf_id: str, task: str, ds_config: dict, result: dict) -> None:
    """Store a successful baseline result in cache."""
    if result.get("status") != "PASS":
        return
    cache = _load_baseline_cache()
    key = _baseline_cache_key(hf_id, task, ds_config)
    cache[key] = {
        "status": result["status"],
        "metric": result["metric"],
        "elapsed": result["elapsed"],
        "command": _shorten_command(result.get("command", "")),
    }
    _save_baseline_cache(cache)


def _run_pytorch_baseline(entry: ModelEntry, device: str, timeout: int) -> dict:
    """Invoke run_pytorch_baseline.py for one model."""
    ds_config = get_dataset_config(entry.hf_id, entry.task) or {}
    args = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--model",
        entry.hf_id,
        "--task",
        entry.task,
        "--device",
        "cpu",  # baseline always on CPU for reproducibility
    ]
    args += ["--num-samples", str(ds_config.get("num_samples", _DEFAULT_SAMPLES))]
    if ds_config.get("dataset"):
        args += ["--dataset", ds_config["dataset"]]
    if ds_config.get("split"):
        args += ["--split", ds_config["split"]]
    if ds_config.get("dataset_config"):
        args += ["--dataset-config", ds_config["dataset_config"]]
    if ds_config.get("columns_mapping"):
        args += ["--columns-mapping", json.dumps(ds_config["columns_mapping"])]
    if ds_config.get("label_mapping_file"):
        args += ["--label-mapping-file", ds_config["label_mapping_file"]]

    proc = _run_subprocess(args, timeout)
    metric = _parse_metric_from_stdout(proc["stdout"]) if proc["exit_code"] == 0 else None
    status = "PASS" if (proc["exit_code"] == 0 and metric is not None) else "FAIL"

    return {
        "status": status,
        "metric": metric,
        "exit_code": proc["exit_code"],
        "stderr": proc["stderr"],
        "elapsed": proc["elapsed"],
        "timeout": proc["timeout"],
        "command": proc["command"],
    }


def _run_accuracy_phase(
    entry: ModelEntry,
    device: str,
    timeout: int,
    model_dir: Path,
    onnx_paths: dict[str, str] | None = None,
    ep: str | None = None,
    monitor: bool = False,
) -> dict:
    """Run winml eval + pytorch baseline for one model. Returns accuracy sub-section dict."""
    ds_config = get_dataset_config(entry.hf_id, entry.task) or {}

    # Build local dataset if a build_script is configured
    _build_dataset(ds_config, timeout)

    winml = _run_winml_eval(
        entry, device, timeout, ds_config, model_dir, onnx_paths, ep=ep, monitor=monitor
    )

    # Check baseline cache before running the expensive PyTorch baseline
    cached = _lookup_baseline_cache(entry.hf_id, entry.task, ds_config)
    if cached is not None:
        safe_print(f"    baseline: cached ({cached['metric']})")
        baseline = cached
    else:
        baseline = _run_pytorch_baseline(entry, device, timeout)
        _store_baseline_cache(entry.hf_id, entry.task, ds_config, baseline)

    delta_abs, delta_rel = compute_delta(winml["metric"], baseline["metric"])

    return {
        "skipped": False,
        "skip_reason": None,
        "winml_eval_status": winml["status"],
        "winml_metric": winml["metric"],
        "winml_eval_exit_code": winml.get("exit_code"),
        "winml_eval_stdout": winml.get("stdout", ""),
        "winml_eval_stderr": winml.get("stderr", ""),
        "elapsed_winml": winml["elapsed"],
        "pytorch_baseline_status": baseline["status"],
        "pytorch_baseline_metric": baseline["metric"],
        "pytorch_baseline_exit_code": baseline.get("exit_code"),
        "pytorch_baseline_stderr": baseline.get("stderr", ""),
        "elapsed_pytorch": baseline["elapsed"],
        "delta_absolute": delta_abs,
        "delta_relative": delta_rel,
        "dataset_config": {k: v for k, v in ds_config.items() if k != "hf_token_required"},
        "winml_eval_command": winml["command"],
        "pytorch_baseline_command": baseline["command"],
        "hw_monitor": winml.get("hw_monitor"),
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def save_environment_info(path: Path) -> None:
    """Save environment metadata for reproducibility."""
    info = {
        "timestamp": _utc_now(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    for pkg in ("onnxruntime", "torch", "transformers", "optimum"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    # Git HEAD commit info
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H%n%s%n%ai"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            info["git_commit"] = lines[0] if lines else ""
            info["git_commit_message"] = lines[1] if len(lines) > 1 else ""
            info["git_commit_date"] = lines[2] if len(lines) > 2 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # git not available or timed out; commit info stays empty

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _get_disk_free_gb() -> float:
    """Get free disk space in GB on the drive where HF cache lives."""
    anchor = HF_CACHE_DIR.anchor or Path.home().anchor
    return shutil.disk_usage(anchor).free / (1024**3)


def _should_skip_existing(existing: dict, retry_types: set[str] | None, eval_type: str) -> bool:
    """Return True if an existing eval_result should be skipped (not re-run).

    Used by both --list-json and the main eval loop to share continue/retry logic.
    """
    if retry_types is None:
        return True  # --continue without --retry-failed: skip all existing

    perf = existing.get("perf") or {}
    acc = existing.get("accuracy")

    # Check perf failure (only when perf ran)
    if eval_type != "accuracy" and not perf.get("passed"):
        cls = classify_result(existing) or "UNKNOWN"
        if not retry_types or cls in retry_types:
            return False  # Should retry

    # Check accuracy verdict
    if acc is not None and not acc.get("skipped"):
        verdict = derive_verdict(acc).value
        if not retry_types or verdict in retry_types:
            return False  # Should retry

    return True  # No retry criteria matched — skip


def model_result_dir(output_dir: Path, hf_id: str, task: str = "") -> Path:
    """Convert model ID + task to directory slug."""
    slug = hf_id.replace("/", "__")
    if task:
        slug += f"__{task}"
    return output_dir / "models" / slug


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="E2E evaluation runner — unified perf + accuracy")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent / "testsets" / "models_all.json",
        help="Model registry JSON (default: scripts/e2e_eval/testsets/models_all.json)",
    )
    parser.add_argument("--hf-model", help="Single model (overrides registry)")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--eval-type",
        choices=["perf", "accuracy", "both"],
        default="perf",
        help=(
            "Evaluation signals to run (default: perf). "
            "accuracy/both: winml perf runs first to populate cache, "
            "then winml eval + pytorch baseline."
        ),
    )
    parser.add_argument("--task", help="Filter by HF task")
    parser.add_argument(
        "--priority",
        nargs="+",
        choices=["P0", "P1", "P2", "P3"],
        default=["P0", "P1", "P2"],
        metavar="{P0,P1,P2,P3}",
        help=(
            "Filter by priority. Pass one or more, e.g. --priority P0 P1. "
            "Default: P0 P1 P2 (P3 excluded from default runs)."
        ),
    )
    parser.add_argument("--model-type", help="Filter by model_type")
    parser.add_argument("--group", help="Filter by group")
    parser.add_argument("--device", default="auto", help="Target device (default: auto)")
    parser.add_argument("--ep", default=None, help="Execution provider (e.g. qnn, dml, ov)")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-subprocess timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--clean-cache",
        dest="clean_cache",
        action="store_true",
        help="Delete caches and leaked temp files after each model evaluation (saves disk space)",
    )
    parser.add_argument(
        "--monitor",
        dest="monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample system-wide CPU/RAM/NPU metrics (Windows PDH) during each "
            "subprocess and persist them in eval_result.json. Use --no-monitor "
            "to disable. Default: enabled."
        ),
    )
    parser.add_argument(
        "--op-tracing",
        dest="op_tracing",
        choices=["basic", "detail"],
        default=None,
        help=(
            "Forward --op-tracing to `wmk perf` for per-operator profiling "
            "(QNN EP only). Adds extra trace iterations and writes "
            "<model_dir>/<slug>_op_trace.json. Default: disabled."
        ),
    )
    parser.add_argument(
        "--memory-limit-gb",
        dest="memory_limit_gb",
        type=float,
        default=None,
        metavar="GB",
        help=(
            "Cap each subprocess tree's committed memory via a Windows Job "
            "Object. When the cap is exceeded the OS refuses further commits "
            "so the host stays responsive instead of thrashing. The offender "
            "typically crashes with a clean OOM and the next model can run. "
            "Recommended for low-RAM agents (e.g. 24 on a 16 GB machine). "
            "No-op on non-Windows. Default: disabled."
        ),
    )
    parser.add_argument("--list", action="store_true", help="List filtered models and exit")
    parser.add_argument(
        "--list-json",
        type=Path,
        metavar="PATH",
        help="Write filtered model list as JSON to PATH and exit (for pipeline orchestration)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation (useful when running per-model in a pipeline loop)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip models that already have eval_result.json",
    )
    parser.add_argument(
        "--retry-failed",
        nargs="*",
        metavar="TYPE",
        help=(
            "Re-run models matching given failure types or accuracy verdicts "
            "(e.g. ENVIRONMENT, ACCURACY_REGRESSION). "
            "Use without args to retry ALL non-PASS models. "
            "Implies --continue for passing models."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run E2E evaluation pipeline."""
    args = parse_args()

    # 1. Load registry
    if args.hf_model:
        # Try to find the model in the registry (preserves dataset_config, etc.)
        matched_entry: ModelEntry | None = None
        try:
            registry_entries = load_registry(args.registry)
            for e in registry_entries:
                if e.hf_id == args.hf_model and (not args.task or e.task == args.task):
                    matched_entry = e
                    break
            # Fallback: match by hf_id only if task-specific match not found
            if matched_entry is None:
                for e in registry_entries:
                    if e.hf_id == args.hf_model:
                        matched_entry = e
                        break
        except Exception as e:
            safe_print(f"  [registry] Optional enrichment skipped: {e}")
        if matched_entry is not None:
            # Override task if explicitly provided on CLI
            if args.task and args.task != matched_entry.task:
                matched_entry = ModelEntry(
                    hf_id=matched_entry.hf_id,
                    task=args.task,
                    model_type=matched_entry.model_type,
                    group=matched_entry.group,
                    priority=matched_entry.priority,
                    dataset_config=matched_entry.dataset_config,
                )
            entries = [matched_entry]
        else:
            entries = [make_adhoc_entry(args.hf_model, args.task)]
    else:
        entries = load_registry(args.registry)
        entries = filter_registry(
            entries,
            task=args.task,
            priority=args.priority,
            model_type=args.model_type,
            group=args.group,
        )

    if not entries:
        safe_print("No models matched the filters.")
        sys.exit(1)

    # Register dataset configs from registry entries as fallback
    register_from_registry(entries)

    # --list mode
    if args.list:
        safe_print(f"Registry: {len(entries)} models  (eval-type: {args.eval_type})")
        for e in entries:
            ds = get_dataset_config(e.hf_id, e.task)
            skip_acc = "" if args.eval_type == "perf" else "  [task_default]" if ds is None else ""
            safe_print(
                f"  [{e.priority}] {e.hf_id} / {e.task}  ({e.model_type}, {e.group}){skip_acc}"
            )
        sys.exit(0)

    # --list-json mode: write machine-readable JSON and exit
    if args.list_json:
        # --continue / --retry-failed: filter out already-evaluated models
        if args.continue_run or args.retry_failed is not None:
            output_dir = args.output_dir or Path(f"eval_results/{date.today().isoformat()}")
            retry_types: set[str] | None = None
            if args.retry_failed is not None:
                args.continue_run = True
                retry_types = {t.upper() for t in args.retry_failed} if args.retry_failed else set()

            filtered: list[ModelEntry] = []
            skipped_count = 0
            for e in entries:
                result_path = model_result_dir(output_dir, e.hf_id, e.task) / "eval_result.json"
                if args.continue_run and result_path.exists():
                    try:
                        existing = load_result_json(result_path)
                        if _should_skip_existing(existing, retry_types, args.eval_type):
                            skipped_count += 1
                            continue
                    except (OSError, json.JSONDecodeError, KeyError) as exc:
                        safe_print(
                            f"  [continue] Corrupt result file {result_path}: {exc} — re-evaluating"
                        )
                filtered.append(e)
            if skipped_count:
                safe_print(
                    f"--continue: skipped {skipped_count} already-evaluated models "
                    f"(output_dir: {output_dir})"
                )
            entries = filtered

        model_list = [
            {
                "hf_id": e.hf_id,
                "task": e.task,
                "model_type": e.model_type,
                "group": e.group,
                "priority": e.priority,
            }
            for e in entries
        ]
        args.list_json.parent.mkdir(parents=True, exist_ok=True)
        args.list_json.write_text(json.dumps(model_list, indent=2), encoding="utf-8")
        safe_print(f"Wrote {len(model_list)} models to {args.list_json}")
        sys.exit(0)

    # 2. Setup output directory
    output_dir = args.output_dir or Path(f"eval_results/{date.today().isoformat()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_environment_info(output_dir / "environment.json")

    # eval_types_run reflects what actually runs for each model:
    #   "perf"     → winml perf only
    #   "accuracy" → winml eval + pytorch baseline only (perf skipped)
    #   "both"     → Strategy B: winml perf first (populates cache), then winml eval + baseline
    eval_types_run = (
        ["accuracy"]
        if args.eval_type == "accuracy"
        else ["perf", "accuracy"]
        if args.eval_type == "both"
        else ["perf"]
    )

    # --retry-failed implies --continue for passing models
    retry_types: set[str] | None = None
    if args.retry_failed is not None:
        args.continue_run = True
        retry_types = {t.upper() for t in args.retry_failed} if args.retry_failed else set()

    safe_print(f"E2E Evaluation: {len(entries)} models -> {output_dir}")
    ep_label = args.ep or "auto"
    safe_print(
        f"Device: {args.device} | EP: {ep_label} | Timeout: {args.timeout}s | Eval: {args.eval_type}"
    )
    safe_print(f"Disk free: {_get_disk_free_gb():.1f} GB")
    if args.clean_cache:
        safe_print("Cache cleanup: ON (caches + temp files cleaned after each model)")
    if args.monitor:
        try:
            from winml.modelkit.session import HWMonitor

            mon_status = "ON" if HWMonitor.is_available() else "OFF (unsupported platform)"
        except ImportError:
            mon_status = "OFF (winml.modelkit.session not importable)"
        safe_print(f"HW monitor: {mon_status} (CPU/RAM/NPU sampled per subprocess)")
    else:
        safe_print("HW monitor: OFF (--no-monitor)")
    if args.op_tracing:
        try:
            from winml.modelkit.optracing import is_qnn_profiling_available

            if is_qnn_profiling_available():
                op_status = f"ON (level={args.op_tracing}, QNN EP)"
            else:
                op_status = (
                    f"REQUESTED (level={args.op_tracing}) but QNN EP not available "
                    "— wmk perf will fail; install onnxruntime-qnn"
                )
        except ImportError:
            op_status = f"REQUESTED (level={args.op_tracing}) — optracing module not importable"
        safe_print(f"Op-tracing: {op_status}")
        if args.eval_type == "accuracy":
            safe_print(
                "  [warn] --op-tracing has no effect with --eval-type accuracy "
                "(only the perf phase invokes wmk perf)."
            )
    if args.memory_limit_gb:
        _set_subprocess_memory_limit_gb(args.memory_limit_gb)
        if platform.system() == "Windows":
            safe_print(
                f"Memory cap: {args.memory_limit_gb:.1f} GB per subprocess tree "
                "(Windows Job Object)"
            )
        else:
            safe_print(
                f"Memory cap: REQUESTED ({args.memory_limit_gb:.1f} GB) but "
                "non-Windows host — ignored"
            )
    else:
        _set_subprocess_memory_limit_gb(None)
    if retry_types is not None:
        if retry_types:
            safe_print(f"Retry mode: {', '.join(sorted(retry_types))}")
        else:
            safe_print("Retry mode: ALL non-PASS models")
    elif args.continue_run:
        safe_print("Continue mode: skipping models with existing eval_result.json")

    # 3. Run evaluation
    results: list[dict] = []
    skipped = 0
    run_start = time.perf_counter()
    interrupted = False
    timeout_skip_set = _load_timeout_skip_set()
    if timeout_skip_set:
        safe_print(f"Timeout skip list: {len(timeout_skip_set)} models will be auto-skipped")

    for i, entry in enumerate(entries, 1):
        label = f"{entry.hf_id} / {entry.task}" if entry.task else entry.hf_id
        model_dir = model_result_dir(output_dir, entry.hf_id, entry.task)
        result_path = model_dir / "eval_result.json"

        # Timeout skip list: skip known-timeout models and write a TIMEOUT result
        if (entry.hf_id, entry.task or "") in timeout_skip_set:
            reason = _get_timeout_skip_reason(entry.hf_id, entry.task or "")
            safe_print(f"\n[{i}/{len(entries)}] {label}  (SKIP - TIMEOUT: {reason})")
            model_dir.mkdir(parents=True, exist_ok=True)
            timeout_result = build_eval_result(
                entry=entry,
                perf_proc={
                    "stdout": "",
                    "stderr": f"Skipped: known timeout model. Reason: {reason}",
                    "exit_code": -1,
                    "elapsed": 0,
                    "timeout": True,
                    "command": "skipped",
                },
                device=args.device,
                eval_types_run=[args.eval_type],
                accuracy_result=None,
                ep=args.ep,
            )
            write_result_json(timeout_result, result_path)
            results.append(timeout_result)
            skipped += 1
            continue

        # --continue / --retry-failed: check existing eval_result.json
        if args.continue_run and result_path.exists():
            try:
                existing = load_result_json(result_path)

                if _should_skip_existing(existing, retry_types, args.eval_type):
                    results.append(existing)
                    skipped += 1
                    perf = existing.get("perf") or {}
                    acc = existing.get("accuracy")
                    perf_cls = classify_result(existing) or "UNKNOWN"
                    perf_tag = "PASS" if perf.get("passed") else f"FAIL/{perf_cls}"
                    acc_tag = ""
                    if acc is not None:
                        acc_tag = f"  acc={derive_verdict(acc).value}"
                    safe_print(
                        f"\n[{i}/{len(entries)}] {label}  (SKIP - {perf_tag}{acc_tag}, cached)"
                    )
                    continue

                retry_label = classify_result(existing) or (
                    derive_verdict(existing.get("accuracy")).value
                    if existing.get("accuracy")
                    else "?"
                )
                safe_print(f"\n[{i}/{len(entries)}] {label}  (RETRY - was {retry_label})")
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted result file — re-run

        safe_print(f"\n[{i}/{len(entries)}] {label}  ({entry.priority}, {entry.group})")

        try:
            perf_proc: dict | None = None
            accuracy_result: dict | None = None

            # Build phase: winml config + winml build → list of ONNX paths
            # Build is shared by perf and eval, avoiding redundant builds.
            build_result = _run_build(
                entry,
                args.device,
                _DEFAULT_PRECISION,
                args.timeout,
                model_dir,
                ep=args.ep,
                monitor=args.monitor,
            )
            onnx_paths = build_result["onnx_paths"] if build_result["success"] else {}

            if not build_result["success"]:
                # Build failed — synthesize failed result for downstream phases
                fail_proc = build_result["proc"]
                fail_proc["device"] = args.device
                fail_proc["timestamp"] = _utc_now()
                fail_proc["error_summary"] = f"build_{build_result['stage']}_failed"

                if args.eval_type != "accuracy":
                    perf_proc = fail_proc
                if args.eval_type != "perf":
                    accuracy_result = {"skipped": True, "skip_reason": "build_failed"}
            elif args.eval_type == "accuracy":
                accuracy_result = _run_accuracy_phase(
                    entry,
                    args.device,
                    args.timeout,
                    model_dir,
                    onnx_paths,
                    ep=args.ep,
                    monitor=args.monitor,
                )
            elif args.eval_type == "perf":
                perf_proc = run_model(
                    entry,
                    args.device,
                    args.timeout,
                    onnx_paths,
                    ep=args.ep,
                    monitor=args.monitor,
                    op_tracing=args.op_tracing,
                    model_dir=model_dir,
                )
            else:
                # "both": perf → eval
                perf_proc = run_model(
                    entry,
                    args.device,
                    args.timeout,
                    onnx_paths,
                    ep=args.ep,
                    monitor=args.monitor,
                    op_tracing=args.op_tracing,
                    model_dir=model_dir,
                )
                if perf_proc["exit_code"] != 0:
                    accuracy_result = {"skipped": True, "skip_reason": "perf_failed"}
                else:
                    accuracy_result = _run_accuracy_phase(
                        entry,
                        args.device,
                        args.timeout,
                        model_dir,
                        onnx_paths,
                        ep=args.ep,
                        monitor=args.monitor,
                    )

        except KeyboardInterrupt:
            safe_print("\n\n[Ctrl+C] Interrupted — generating reports for completed models...")
            interrupted = True
            break

        result = build_eval_result(
            entry, perf_proc, args.device, eval_types_run, accuracy_result, ep=args.ep
        )
        results.append(result)

        # Write eval_result.json immediately (crash-safe, facts only)
        write_result_json(result, result_path)

        # Print status line
        acc_tag = ""
        if accuracy_result is not None:
            if accuracy_result.get("skipped"):
                acc_tag = f"  acc=SKIP/{accuracy_result['skip_reason']}"
            else:
                verdict = derive_verdict(accuracy_result).value
                delta_str = format_delta(accuracy_result)
                if delta_str:
                    delta_str = f" {delta_str}"
                acc_tag = f"  acc={verdict}{delta_str}"

        # Build a compact HW summary tag (peaks across whichever phases ran)
        hw_tag = _format_hw_summary(perf_proc, accuracy_result)

        # Op-trace tag: count how many trace files were captured for this model.
        op_tag = ""
        if perf_proc and perf_proc.get("op_trace"):
            op_tag = f"  op_trace={len(perf_proc['op_trace'])}"

        if perf_proc is not None:
            perf_passed = perf_proc["exit_code"] == 0
            perf_cls = classify_result(result) or "UNKNOWN"
            perf_tag = "PASS" if perf_passed else f"FAIL ({perf_cls})"
            safe_print(f"  [{perf_tag}] {result['perf']['elapsed']}s{acc_tag}{hw_tag}{op_tag}")
            if args.verbose and not perf_passed:
                combined = (perf_proc["stdout"] + perf_proc["stderr"]).strip()
                for line in combined.splitlines()[-10:]:
                    safe_print(f"    {line}")
        else:
            safe_print(f"  [acc only]{acc_tag}{hw_tag}")

        if args.clean_cache:
            _clear_disk_caches()

    run_duration = time.perf_counter() - run_start

    if not results:
        safe_print("\nNo results to report.")
        sys.exit(1)

    # 4. Generate reports
    classify_results(results)
    if args.eval_type != "perf":
        derive_verdicts(results)

    if not args.no_report:
        summary = generate_summary(results, run_duration)
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")

        # JSON report
        report_json_path = output_dir / f"eval_report_{timestamp_slug}.json"
        write_summary_json(summary, report_json_path)

        # Text summary (perf-focused)
        text_report = format_text_summary(results)
        safe_print(text_report)
        report_txt_path = output_dir / f"eval_report_{timestamp_slug}.txt"
        report_txt_path.write_text(text_report, encoding="utf-8")

        # Markdown summary
        write_summary_md(results, summary, output_dir / "summary.md")

        # HTML report
        generate_html_report(summary, output_dir / "eval_report.html", args.registry)

        safe_print(f"\nResults saved to: {output_dir}")
        safe_print(f"  report: {report_json_path.name}")
        safe_print("  summary: summary.md")
        safe_print("  html: eval_report.html")

        ps = summary["perf_summary"]
        total = ps["total"]
        rate = (ps["passed"] / total * 100) if total else 0
        safe_print(f"\nPerf pass rate: {ps['passed']}/{total} ({rate:.1f}%)")

        if args.eval_type != "perf":
            acc_s = summary.get("accuracy_summary", {})
            evaluated = acc_s.get("evaluated", 0)
            acc_pass = acc_s.get("accuracy_pass", 0)
            acc_rate = acc_s.get("pass_rate", 0)
            safe_print(
                f"Accuracy pass rate: {acc_pass}/{evaluated} ({acc_rate:.1%})  "
                f"[at-risk={acc_s.get('accuracy_at_risk', 0)} "
                f"regression={acc_s.get('accuracy_regression', 0)} "
                f"error={acc_s.get('eval_error', 0)}]"
            )

    if skipped:
        safe_print(f"  ({skipped} cached from previous run)")
    if interrupted:
        safe_print(f"  (interrupted — {len(entries) - len(results)} models not evaluated)")

    all_perf_pass = all((r.get("perf") or {}).get("passed", False) for r in results)
    all_acc_pass = args.eval_type == "perf" or all(
        derive_verdict(r.get("accuracy")) == AccuracyVerdict.ACCURACY_PASS
        for r in results
        if r.get("accuracy") and not (r.get("accuracy") or {}).get("skipped")
    )

    sys.exit(0 if not interrupted and all_perf_pass and all_acc_pass else 1)


if __name__ == "__main__":
    main()
