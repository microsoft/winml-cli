"""E2E evaluation runner — unified perf + accuracy.

Batch-runs wmk perf (and optionally wmk eval + pytorch baseline) for models
in a JSON registry, writes unified eval_result.json per model, and generates
combined reports.

Strategy B cache sharing: wmk perf runs first (build + benchmark, populates
model cache). wmk eval then reuses the cache — no redundant build step.

Usage:
    # Perf only (default)
    python scripts/e2e_eval/run_eval.py --priority P0

    # Both perf and accuracy in one batch
    python scripts/e2e_eval/run_eval.py --eval-type both --priority P0

    # Accuracy only (wmk perf is skipped; wmk eval will build the model if cache is missing)
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

WMK = [sys.executable, "-m", "winml.modelkit.cli"]
BASELINE_SCRIPT = Path(__file__).parent / "run_pytorch_baseline.py"
BASELINE_CACHE_PATH = Path(__file__).parent / "cache" / "baseline_cache.json"
EVAL_DATASETS_CACHE = Path.home() / ".cache" / "winml" / "eval_datasets"
TIMEOUT_SKIP_LIST_PATH = Path(__file__).parent / "cache" / "timeout_skip_list.json"
_DEFAULT_SAMPLES = 1000
_DEFAULT_PRECISION = "w8a16"


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
    "no space left on device",          # Linux/macOS OSError
    "oserror: [errno 28]",              # Python errno string
    "there is not enough space on the disk",  # Windows
    "winerror 112",                     # Windows disk-full error code
    "disk full",
)

_HF_CACHE = Path.home() / ".cache" / "huggingface"
_WML_CACHE = Path.home() / ".cache" / "winml"


def _is_no_space_error(proc: dict) -> bool:
    """Return True if subprocess output indicates a disk-full condition."""
    combined = (proc.get("stdout", "") + proc.get("stderr", "")).lower()
    return any(pat in combined for pat in _NO_SPACE_PATTERNS)


def _clear_disk_caches() -> None:
    """Delete HuggingFace and WML cache directories to free disk space."""
    for cache_dir in (_HF_CACHE, _WML_CACHE):
        if cache_dir.exists():
            safe_print(f"  [disk-full] Removing cache: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                safe_print(f"  [disk-full] Removed: {cache_dir}")
            except OSError as exc:
                safe_print(f"  [disk-full] Warning: could not remove {cache_dir}: {exc}")
        else:
            safe_print(f"  [disk-full] Cache not found (skipping): {cache_dir}")


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
                ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True
            )
        else:
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already exited; nothing to kill


def _run_subprocess(args: list[str], timeout: int) -> dict:
    """Run a subprocess with three-layer timeout protection.

    Returns a dict with: stdout, stderr, exit_code, elapsed, timeout, command.

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

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    elapsed = round(time.perf_counter() - start, 1)

    result = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "elapsed": elapsed,
        "timeout": timed_out,
        "command": " ".join(str(a) for a in args),
    }

    # Retry once after clearing caches if the failure was due to disk full.
    if exit_code != 0 and not timed_out and _is_no_space_error(result):
        safe_print("  [disk-full] Detected 'no space left' — clearing caches and retrying...")
        _clear_disk_caches()
        safe_print(f"  [disk-full] Retrying: {result['command']}")
        result = _run_subprocess(args, timeout)

    return result


# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------


def _run_build(
    entry: ModelEntry, device: str, precision: str, timeout: int, model_dir: Path,
) -> dict:
    """Run wmk config + wmk build for one model. Returns build result dict.

    Flow: wmk config → config.json → wmk build --use-cache → ONNX path.
    """
    config_path = model_dir / "build_config.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: wmk config
    config_args = [
        *WMK, "config",
        "-m", entry.hf_id,
        "--device", device,
        "--precision", precision,
        "-o", str(config_path),
    ]
    if entry.task:
        config_args += ["--task", entry.task]

    config_proc = _run_subprocess(config_args, timeout)
    if config_proc["exit_code"] != 0:
        return {
            "success": False,
            "onnx_path": None,
            "stage": "config",
            "proc": config_proc,
        }

    # Step 2: wmk build --use-cache
    build_args = [
        *WMK, "build",
        "-c", str(config_path),
        "-m", entry.hf_id,
        "--use-cache",
    ]

    build_proc = _run_subprocess(build_args, timeout)
    if build_proc["exit_code"] != 0:
        return {
            "success": False,
            "onnx_path": None,
            "stage": "build",
            "proc": build_proc,
        }

    # Extract ONNX path from build output
    # wmk build prints "Final artifact: <path>" in stderr
    onnx_path = None
    for line in build_proc["stderr"].splitlines():
        if "Final artifact:" in line:
            onnx_path = line.split("Final artifact:")[-1].strip()
            break

    # Fallback: search cache for the built model
    if not onnx_path:
        for line in build_proc["stdout"].splitlines():
            if "Final artifact:" in line:
                onnx_path = line.split("Final artifact:")[-1].strip()
                break

    if not onnx_path or not Path(onnx_path).exists():
        # Last resort: find _model.onnx in the cache
        onnx_path = _find_cached_model(entry.hf_id, build_proc, entry.task)

    return {
        "success": onnx_path is not None,
        "onnx_path": onnx_path,
        "stage": "complete",
        "proc": build_proc,
        "config_path": str(config_path),
    }


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
    entry: ModelEntry, device: str, timeout: int, onnx_path: str | None = None,
) -> dict:
    """Execute wmk perf for one model. Returns raw subprocess result dict.

    When onnx_path is provided, benchmarks the pre-built ONNX directly
    (skips internal build). Otherwise falls back to HF model ID.
    """
    if onnx_path:
        args = [*WMK, "perf", "-m", onnx_path, "--device", device]
    else:
        args = [
            *WMK, "perf", "-m", entry.hf_id,
            "--device", device, "--precision", _DEFAULT_PRECISION,
        ]
        if entry.task:
            args += ["--task", entry.task]

    args += ["--iterations", "10", "--warmup", "2"]
    args += entry.perf_args

    proc = _run_subprocess(args, timeout)
    # Attach device and timestamp for build_eval_result
    proc["device"] = device
    proc["timestamp"] = _utc_now()
    proc["error_summary"] = (
        ""
        if proc["exit_code"] == 0
        else f"timeout ({timeout}s)"
        if proc["timeout"]
        else f"exit code {proc['exit_code']}"
    )
    return proc


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


def _parse_metric_from_wmk_output(
    output_path: Path, metric_name: str, num_samples: int
) -> dict | None:
    """Parse wmk eval --output JSON file into the canonical metric dict."""
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


def _run_wmk_eval(
    entry: ModelEntry,
    device: str,
    timeout: int,
    ds_config: dict,
    model_dir: Path,
    onnx_path: str | None = None,
) -> dict:
    """Invoke wmk eval for one model. Returns process result + parsed metric."""
    output_path = model_dir / "wmk_eval_output.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    # wmk eval requires explicit device ('cpu'/'gpu'/'npu'); 'auto' is not accepted
    eval_device = "npu" if device == "auto" else device
    if onnx_path:
        args = [
            *WMK, "eval", "-m", onnx_path,
            "--model-id", entry.hf_id,
            "--device", eval_device,
        ]
    else:
        args = [
            *WMK, "eval", "-m", entry.hf_id,
            "--device", eval_device,
        ]
    if entry.task:
        args += ["--task", entry.task]
    # When ds_config is provided, pass explicit dataset args;
    # otherwise wmk eval uses its built-in task defaults.
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

    proc = _run_subprocess(args, timeout)

    metric = None
    if proc["exit_code"] == 0 and output_path.exists():
        wmk_key = (
            ds_config.get("wmk_metric_key")
            or ds_config.get("metric", "accuracy")
        )
        num_samples = ds_config.get("num_samples", _DEFAULT_SAMPLES)
        metric = _parse_metric_from_wmk_output(
            output_path, wmk_key, num_samples
        )
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


def _lookup_baseline_cache(
    hf_id: str, task: str, ds_config: dict
) -> dict | None:
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


def _store_baseline_cache(
    hf_id: str, task: str, ds_config: dict, result: dict
) -> None:
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
    onnx_path: str | None = None,
) -> dict:
    """Run wmk eval + pytorch baseline for one model. Returns accuracy sub-section dict."""
    ds_config = get_dataset_config(entry.hf_id, entry.task) or {}

    # Build local dataset if a build_script is configured
    _build_dataset(ds_config, timeout)

    wmk = _run_wmk_eval(entry, device, timeout, ds_config, model_dir, onnx_path)

    # Check baseline cache before running the expensive PyTorch baseline
    cached = _lookup_baseline_cache(entry.hf_id, entry.task, ds_config)
    if cached is not None:
        safe_print(f"    baseline: cached ({cached['metric']})")
        baseline = cached
    else:
        baseline = _run_pytorch_baseline(entry, device, timeout)
        _store_baseline_cache(entry.hf_id, entry.task, ds_config, baseline)

    delta_abs, delta_rel = compute_delta(wmk["metric"], baseline["metric"])

    return {
        "skipped": False,
        "skip_reason": None,
        "wmk_eval_status": wmk["status"],
        "wmk_metric": wmk["metric"],
        "wmk_eval_exit_code": wmk.get("exit_code"),
        "wmk_eval_stdout": wmk.get("stdout", ""),
        "wmk_eval_stderr": wmk.get("stderr", ""),
        "elapsed_wmk": wmk["elapsed"],
        "pytorch_baseline_status": baseline["status"],
        "pytorch_baseline_metric": baseline["metric"],
        "pytorch_baseline_exit_code": baseline.get("exit_code"),
        "pytorch_baseline_stderr": baseline.get("stderr", ""),
        "elapsed_pytorch": baseline["elapsed"],
        "delta_absolute": delta_abs,
        "delta_relative": delta_rel,
        "dataset_config": {k: v for k, v in ds_config.items() if k != "hf_token_required"},
        "wmk_eval_command": wmk["command"],
        "pytorch_baseline_command": baseline["command"],
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
        except ImportError:  # noqa: PERF203
            info[f"{pkg}_version"] = "not installed"

    # Git HEAD commit info
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H%n%s%n%ai"],  # noqa: S603, S607
            capture_output=True, text=True, timeout=5,
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


def _clean_model_hf_cache(hf_id: str) -> None:
    """Delete cached HuggingFace files for a specific model."""
    slug = f"models--{hf_id.replace('/', '--')}"
    cache_path = HF_CACHE_DIR / slug
    if not cache_path.exists():
        return
    size_mb = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file()) / (1024 * 1024)
    shutil.rmtree(cache_path, ignore_errors=True)
    safe_print(f"  [cache] Cleaned {slug} ({size_mb:.0f} MB freed)")


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
            "accuracy/both: wmk perf runs first to populate cache, "
            "then wmk eval + pytorch baseline."
        ),
    )
    parser.add_argument("--task", help="Filter by HF task")
    parser.add_argument("--priority", choices=["P0", "P1", "P2"], help="Filter by priority")
    parser.add_argument("--model-type", help="Filter by model_type")
    parser.add_argument("--group", help="Filter by group")
    parser.add_argument("--device", default="auto", help="Target device (default: auto)")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-subprocess timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--clean-hf-cache",
        action="store_true",
        help="Delete HuggingFace hub cache for each model after evaluation (saves disk space)",
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
        except Exception:
            pass  # Registry is optional for single-model mode; proceed without enrichment
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
            skip_acc = (
                ""
                if args.eval_type == "perf"
                else "  [task_default]"
                if ds is None
                else ""
            )
            safe_print(
                f"  [{e.priority}] {e.hf_id} / {e.task}  ({e.model_type}, {e.group}){skip_acc}"
            )
        sys.exit(0)

    # --list-json mode: write machine-readable JSON and exit
    if args.list_json:
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
    #   "perf"     → wmk perf only
    #   "accuracy" → wmk eval + pytorch baseline only (perf skipped)
    #   "both"     → Strategy B: wmk perf first (populates cache), then wmk eval + baseline
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
    safe_print(f"Device: {args.device} | Timeout: {args.timeout}s | Eval: {args.eval_type}")
    safe_print(f"Disk free: {_get_disk_free_gb():.1f} GB")
    if args.clean_hf_cache:
        safe_print("Cache cleanup: ON (HF cache cleaned after each model)")
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
            safe_print(
                f"\n[{i}/{len(entries)}] {label}  (SKIP - TIMEOUT: {reason})"
            )
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
            )
            write_result_json(timeout_result, result_path)
            results.append(timeout_result)
            skipped += 1
            continue

        # --continue / --retry-failed: check existing eval_result.json
        if args.continue_run and result_path.exists():
            try:
                existing = load_result_json(result_path)
                skip = True

                perf = existing.get("perf") or {}
                acc = existing.get("accuracy")

                # Derive current classification / verdict to check against retry types
                if retry_types is not None:
                    should_retry = False
                    # Check perf failure (only when perf ran)
                    if args.eval_type != "accuracy" and not perf.get("passed"):
                        cls = classify_result(existing) or "UNKNOWN"
                        if not retry_types or cls in retry_types:
                            should_retry = True
                    # Check accuracy verdict
                    if not should_retry and acc is not None and not acc.get("skipped"):
                        verdict = derive_verdict(acc).value
                        if not retry_types or verdict in retry_types:
                            should_retry = True
                    skip = not should_retry

                if skip:
                    results.append(existing)
                    skipped += 1
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

            # Build phase: wmk config + wmk build → ONNX path
            # Build is shared by perf and eval, avoiding redundant builds.
            onnx_path: str | None = None
            if args.eval_type in ("perf", "both"):
                build_result = _run_build(
                    entry, args.device, _DEFAULT_PRECISION, args.timeout, model_dir,
                )
                if build_result["success"]:
                    onnx_path = build_result["onnx_path"]

            if args.eval_type == "accuracy":
                # Accuracy-only: build + eval (no perf)
                build_result = _run_build(
                    entry, args.device, _DEFAULT_PRECISION, args.timeout, model_dir,
                )
                if build_result["success"]:
                    onnx_path = build_result["onnx_path"]
                    accuracy_result = _run_accuracy_phase(
                        entry, args.device, args.timeout, model_dir, onnx_path,
                    )
                else:
                    accuracy_result = {"skipped": True, "skip_reason": "build_failed"}
            elif args.eval_type == "perf":
                if onnx_path:
                    perf_proc = run_model(entry, args.device, args.timeout, onnx_path)
                else:
                    # Build failed — synthesize a failed perf result
                    perf_proc = build_result["proc"]
                    perf_proc["device"] = args.device
                    perf_proc["timestamp"] = _utc_now()
                    perf_proc["error_summary"] = f"build_{build_result['stage']}_failed"
            else:
                # "both": build → perf → eval
                if onnx_path:
                    perf_proc = run_model(entry, args.device, args.timeout, onnx_path)
                    if perf_proc["exit_code"] != 0:
                        accuracy_result = {"skipped": True, "skip_reason": "perf_failed"}
                    else:
                        accuracy_result = _run_accuracy_phase(
                            entry, args.device, args.timeout, model_dir, onnx_path,
                        )
                else:
                    # Build failed
                    perf_proc = build_result["proc"]
                    perf_proc["device"] = args.device
                    perf_proc["timestamp"] = _utc_now()
                    perf_proc["error_summary"] = f"build_{build_result['stage']}_failed"
                    accuracy_result = {"skipped": True, "skip_reason": "build_failed"}

        except KeyboardInterrupt:
            safe_print("\n\n[Ctrl+C] Interrupted — generating reports for completed models...")
            interrupted = True
            break

        result = build_eval_result(entry, perf_proc, args.device, eval_types_run, accuracy_result)
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
                delta_rel = accuracy_result.get("delta_relative")
                delta_str = f" {delta_rel:.1%}" if delta_rel is not None else ""
                acc_tag = f"  acc={verdict}{delta_str}"

        if perf_proc is not None:
            perf_passed = perf_proc["exit_code"] == 0
            perf_cls = classify_result(result) or "UNKNOWN"
            perf_tag = "PASS" if perf_passed else f"FAIL ({perf_cls})"
            safe_print(f"  [{perf_tag}] {result['perf']['elapsed']}s{acc_tag}")
            if args.verbose and not perf_passed:
                combined = (perf_proc["stdout"] + perf_proc["stderr"]).strip()
                for line in combined.splitlines()[-10:]:
                    safe_print(f"    {line}")
        else:
            safe_print(f"  [acc only]{acc_tag}")

        if args.clean_hf_cache:
            _clean_model_hf_cache(entry.hf_id)

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
