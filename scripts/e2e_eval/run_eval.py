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
import hashlib
import json
import logging
import os
import platform
import re
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


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINML_CLI = [sys.executable, "-m", "winml.modelkit.cli"]
BASELINE_SCRIPT = Path(__file__).parent / "run_pytorch_baseline.py"
BASELINE_CACHE_PATH = Path(__file__).parent / "cache" / "baseline_cache.json"
EVAL_DATASETS_CACHE = Path.home() / ".cache" / "winml" / "eval_datasets"
TIMEOUT_SKIP_LIST_PATH = Path(__file__).parent / "cache" / "timeout_skip_list.json"
_DEFAULT_SAMPLES = 1000
_DEFAULT_PRECISION_NPU = "w8a16"

# EPs whose eval track keeps the model unquantized (the "fp" variant)
# rather than running winml's QDQ pass on top.  This is an eval-setup
# choice -- e.g. VitisAI / AMD Ryzen AI is benchmarked on the fp32/fp16
# model -- not a claim about the EP's internal pipeline.  For these EPs
# the harness passes ``--no-quant`` to both ``winml config`` and
# ``winml build`` (see :func:`_run_build` and :func:`run_model`).
#
# Entries are canonical ``EPName`` values (the ``*ExecutionProvider`` form);
# user-facing aliases like ``vitisai`` are normalised via
# ``normalize_ep_name`` in :func:`_should_skip_winml_quant` so each EP only
# needs to be listed once.
_EPS_SKIP_WINML_QUANT = frozenset({"VitisAIExecutionProvider"})


def _should_skip_winml_quant(ep: str | None) -> bool:
    """True if the eval harness should run this EP on the unquantized model."""
    # Lazy import: keeps ``scripts/e2e_eval`` cheap to load (winml.modelkit
    # transitively imports onnxruntime) and matches the existing in-function
    # import pattern used elsewhere in this script.
    from winml.modelkit.utils.constants import normalize_ep_name

    return normalize_ep_name(ep) in _EPS_SKIP_WINML_QUANT


def _resolve_precision(device: str, explicit: str | None, ep: str | None = None) -> str | None:
    """Return the precision to pass to winml config/perf, or None to omit the flag.

    w8a16 is only applied by default on NPU.  For CPU/GPU the flag is omitted
    so winml config uses its own auto-detection.  Forcing w8a16 on GPU produces
    a QDQ-quantized model that fails at ORT session creation with QNN GPU EP
    (NHWC layout transformer inserts Conv nodes that QNN GPU's GetCapability
    does not claim).

    For EPs in :data:`_EPS_SKIP_WINML_QUANT` (e.g. VitisAI) the flag is forced
    off regardless of ``explicit``: the harness pairs these EPs with
    ``--no-quant`` at config/build time, so a non-empty ``--precision`` would
    produce a config that says "quantize to X" while the build says "skip
    quantization" -- a contradiction.  An explicit value is dropped with a
    one-line warning so the override is visible in the log.

    Otherwise an explicit per-model precision always takes precedence.
    """
    if _should_skip_winml_quant(ep):
        if explicit:
            safe_print(
                f"  [precision] Ignoring explicit precision={explicit!r} for EP {ep!r}: "
                "this EP is run on the unquantized variant (--no-quant)."
            )
        return None
    if explicit:
        return explicit
    return _DEFAULT_PRECISION_NPU if device == "npu" else None


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
_WINML_CACHE = Path.home() / ".cache" / "winml"
_TEMP_DIR = Path(os.environ.get("TEMP", os.environ.get("TMP", tempfile.gettempdir())))
_TEMP_PREFIXES = ("winmlcli_", "winmlcli_compat_")


def _is_no_space_error(proc: dict) -> bool:
    """Return True if subprocess output indicates a disk-full condition."""
    combined = (proc.get("stdout", "") + proc.get("stderr", "")).lower()
    return any(pat in combined for pat in _NO_SPACE_PATTERNS)


def _clear_disk_caches() -> None:
    """Delete HuggingFace, WinML cache directories and leaked temp files."""
    for cache_dir in (_HF_CACHE, _WINML_CACHE):
        if cache_dir.exists():
            safe_print(f"  [cleanup] Removing cache: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                safe_print(f"  [cleanup] Removed: {cache_dir}")
            except OSError as exc:
                safe_print(f"  [cleanup] Warning: could not remove {cache_dir}: {exc}")

    # Clean leaked temp directories/files (winmlcli_*, winmlcli_compat_*, tmp*.onnx*)
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


def _compute_onnx_size(onnx_paths: dict[str, str]) -> int | None:
    """Return combined size in bytes of all ONNX files + their external data companions.

    Parses the ONNX proto to discover all referenced external data files (not just
    the conventional `.data` suffix). Falls back to the `.data` companion heuristic
    if proto parsing is unavailable.

    Returns None if onnx_paths is empty or no files exist on disk.
    """
    if not onnx_paths:
        return None
    total = 0
    found_any = False
    for path_str in onnx_paths.values():
        p = Path(path_str)
        if not p.exists():
            continue
        total += p.stat().st_size
        found_any = True
        # Try to enumerate all external data files from the proto
        try:
            from winml.modelkit.onnx.external_data import get_external_data_files

            ext_files = get_external_data_files(p)
            for ext_name in ext_files:
                ext_path = p.parent / ext_name
                if ext_path.exists():
                    total += ext_path.stat().st_size
        except Exception:
            # Fallback: check conventional .data companion
            data_p = p.with_suffix(p.suffix + ".data")
            if data_p.exists():
                total += data_p.stat().st_size
    return total if found_any else None


# Lines that carry no diagnostic value in eval_result.json.
# Matching is case-insensitive, anchored at line start.
_NOISE_PATTERNS = (
    "benchmarking onnx",
    "device:",
    "task:",
    "latency (ms)",
    "throughput:",
    "results saved to",
    "inputs:",
    "outputs:",
    "samples/sec",
)
_NOISE_RE = re.compile("|".join(re.escape(p) for p in _NOISE_PATTERNS), re.IGNORECASE)

# Box-drawing characters used by Rich tables.
_BOX_CHARS = frozenset("─│┌┐└┘├┤┬┴┼")


def _sanitize_output(text: str) -> str:
    """Strip routine CLI chrome from subprocess output, keeping error content.

    Removes Rich benchmark tables, device/IO banners, and path lines that
    bloat eval_result.json without aiding failure diagnosis. All classifier
    patterns (see classifier.py) are error-related and survive this filter.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop box-drawing table rows
        if stripped[0] in _BOX_CHARS:
            continue
        if _NOISE_RE.match(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


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
    entry: ModelEntry,
    device: str,
    precision: str | None,
    timeout: int,
    model_dir: Path,
    ep: str | None = None,
    build_only: bool = False,
) -> dict:
    """Run winml config + winml build for one model. Returns build result dict.

    Flow: winml config → list of config JSONs → winml build each → ONNX paths.

    Single models produce one config; composite models (e.g., T5 translation)
    produce one per sub-component (suffixed names). Both go through the same
    build loop — single model is just the list-of-1 case.

    When ``build_only`` is set, each build writes its artifacts to ``model_dir``
    via ``-o`` (preserving the intermediate export/optimized/quantized ONNX) and
    skips compile (``--no-compile``) — no execution provider is required.
    Otherwise the build populates the global cache (``--use-cache``).
    """
    config_path = model_dir / "build_config.json"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale suffixed sub-configs BEFORE `winml config` runs.
    # For composite models `winml config` writes files matching {stem}_*.json
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
        "-o",
        str(config_path),
    ]
    if precision:
        config_args += ["--precision", precision]
    if entry.task:
        config_args += ["--task", entry.task]
    if ep:
        config_args += ["--ep", ep]
    # EPs in _EPS_SKIP_WINML_QUANT are evaluated on the unquantized variant.
    # Pass --no-quant to winml config so the generated build_config.json is
    # written with quant=None up-front; otherwise on NPU the config command
    # would still apply its default precision (w8a16) and we'd be relying on
    # --no-quant at build time alone to override it.
    if _should_skip_winml_quant(ep):
        config_args += ["--no-quant"]

    config_proc = _run_subprocess(config_args, timeout)
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
        ]
        if build_only:
            # Write artifacts to disk and skip compile (no EP required).
            # Composite components get a subdir to avoid name collisions.
            build_out = model_dir / label if label else model_dir
            build_out.mkdir(parents=True, exist_ok=True)
            build_args += ["-o", str(build_out), "--no-compile"]
        else:
            build_args += ["--use-cache"]
        build_args += ["--device", device]
        if ep:
            build_args += ["--ep", ep]
        # Mirror the --no-quant passed to winml config above so the build
        # stage also skips QDQ regardless of what the config carries (defence
        # in depth; see _EPS_SKIP_WINML_QUANT for the rationale).
        if _should_skip_winml_quant(ep):
            build_args += ["--no-quant"]

        build_proc = _run_subprocess(build_args, timeout)
        last_proc = build_proc
        if build_proc["exit_code"] != 0:
            stage = f"build_{label}" if label else "build"
            return {
                "success": False,
                "onnx_paths": onnx_paths,
                "stage": stage,
                "proc": build_proc,
            }

        if build_only:
            # In build-only mode the artifacts go to ``-o <build_out>`` (no
            # cache, no compile). There is no "Final artifact:" marker to
            # parse and no downstream consumer of the path -- exit-code 0 is
            # the success signal. Record build_out so the per-component
            # bookkeeping (len(onnx_paths) == len(sub_configs)) stays valid.
            onnx_paths[label] = str(build_out)
            continue

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
# Build-only phase (export + optimize + quantize, no compile / no EP hardware)
# ---------------------------------------------------------------------------

# EP matrix generated by --build-only when neither --ep nor --device is pinned.
# This is the eval test matrix (deliberately broader than the canonical
# get_ep_device_map): qnn on npu+gpu, OpenVINO on cpu+npu+gpu, MLAS (native CPU
# EP), DirectML on gpu, and VitisAI on npu. Each combo is built into its own
# <model_dir>/<label>/ subdir.
_BUILD_ONLY_EP_MATRIX: tuple[tuple[str, str, str], ...] = (
    ("qnn_npu", "qnn", "npu"),
    ("qnn_gpu", "qnn", "gpu"),
    ("ov_cpu", "openvino", "cpu"),
    ("ov_npu", "openvino", "npu"),
    ("ov_gpu", "openvino", "gpu"),
    ("mlas_cpu", "cpu", "cpu"),
    ("dml_gpu", "dml", "gpu"),
    ("vitisai_npu", "vitisai", "npu"),
)


# ---------------------------------------------------------------------------
# Build-only: export dedup
# ---------------------------------------------------------------------------


def _hash_files(paths: list[Path]) -> str:
    """SHA-256 over a set of files (name + streamed content), order-independent.

    Raises:
        OSError: if any file cannot be read. The caller must decide how to
            handle this (e.g. skip dedup) instead of hashing two unreadable
            files to the same value and deleting an artifact that was never
            verified to be identical.
    """
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _dedup_export(
    build_dir: Path, shared_dir: Path, canonical_hash: str | None, label: str
) -> str | None:
    """Deduplicate this combo's export.onnx(+sidecar) against a per-model canonical.

    The export stage is EP/device-independent, so every combo produces an
    identical ``export.onnx``. The first one is moved into ``shared_dir``
    (``_shared/``); later identical ones are deleted to keep one copy on disk.

    Returns the (possibly newly-set) canonical hash.
    """
    export_files = sorted(build_dir.glob("export.onnx*"))
    if not export_files:
        return canonical_hash  # composite/no top-level export — leave untouched
    try:
        h = _hash_files(export_files)
    except OSError as exc:
        # Never dedup on an unverified hash: an export we cannot read is kept in
        # place rather than risk deleting it as a false duplicate.
        safe_print(f"  [dedup] WARNING {label}: cannot hash export ({exc}) — keeping in place")
        return canonical_hash
    if canonical_hash is None:
        shared_dir.mkdir(parents=True, exist_ok=True)
        for f in export_files:
            shutil.move(str(f), str(shared_dir / f.name))
        safe_print(
            f"  [dedup] export -> {shared_dir.name}/ (canonical, {len(export_files)} file(s))"
        )
        return h
    if h == canonical_hash:
        for f in export_files:
            f.unlink(missing_ok=True)
        safe_print(f"  [dedup] {label}: export identical -> removed (using {shared_dir.name}/)")
        return canonical_hash
    safe_print(f"  [dedup] WARNING {label}: export differs from canonical — keeping in place")
    return canonical_hash


# ---------------------------------------------------------------------------
# Build-only: Azure Artifacts feed upload (Universal Packages, az CLI, no PAT)
# ---------------------------------------------------------------------------

# Azure DevOps AAD application ID. Used as the token audience (``--resource``)
# when querying the feed REST API with ``az rest``. Constant across all orgs and
# not a secret (it is the public first-party app id for Azure DevOps).
_AZURE_DEVOPS_RESOURCE = "499b84ac-1321-427f-aa17-267ca6975798"
_FEED_API_VERSION = "7.1-preview.1"


def _run_az(args: list[str], timeout: int = 600) -> dict:
    """Run an `az ...` command, returning the same dict shape as _run_subprocess."""
    az = shutil.which("az")
    if az is None:
        return {
            "stdout": "",
            "stderr": "az CLI not found on PATH",
            "exit_code": 127,
            "elapsed": 0.0,
            "timeout": False,
            "command": "az " + " ".join(args),
        }
    # Pass the az path (incl. az.cmd on Windows) directly in list form.
    # subprocess handles .cmd resolution and arg quoting correctly; wrapping in
    # `cmd /c` breaks when the az path has spaces (C:\Program Files\...) and an
    # arg also contains spaces (e.g. --description), because cmd.exe then mangles
    # the quotes and tries to run 'C:\Program'.
    return _run_subprocess([az, *args], timeout)


def _ensure_feed_ready(timeout: int = 180) -> str | None:
    """Verify az + azure-devops extension + login. Returns an error string or None."""
    if shutil.which("az") is None:
        return "az CLI not found. Install Azure CLI (https://aka.ms/azcli)."
    ext = _run_az(["extension", "show", "--name", "azure-devops"], timeout)
    if ext["exit_code"] != 0:
        safe_print("  [upload] Installing 'azure-devops' az extension...")
        add = _run_az(["extension", "add", "--name", "azure-devops"], timeout)
        if add["exit_code"] != 0:
            return f"Failed to install 'azure-devops' az extension: {add['stderr'][:300]}"
    acct = _run_az(["account", "show"], timeout)
    if acct["exit_code"] != 0:
        return "Not logged in to Azure. Run 'az login' (PAT not required), then retry."
    return None


def _slugify_version(text: str) -> str:
    """Lowercase + collapse non-[0-9a-z] runs to single dashes (semver prerelease-safe)."""
    s = re.sub(r"[^0-9a-z]+", "-", text.lower())
    return re.sub(r"-{2,}", "-", s).strip("-")


def _feed_version_for(entry: ModelEntry, run_stamp: str, combo_label: str) -> str:
    """Per-combo Universal Package version (one per EP/device pair).

    The version is ``0.0.0-<run-stamp>-<ep>-<device>-<model-slug>``. Universal
    Packages require a valid lowercase SemVer 2.0 version, so the
    ``major.minor.patch`` core is fixed at ``0.0.0`` and the batch stamp, the
    EP/device combo, and the model identity live in the pre-release segment. The
    run stamp (a date like ``20260609``) groups a batch under a common prefix;
    ``combo_label`` (e.g. ``qnn_npu``) scopes the version to a single EP/device
    pair so each model uploads one (smaller) package per combo instead of one
    large package for the whole matrix -- which both lowers the per-upload
    timeout risk and lets an interrupted combo be retried on its own. An
    interrupted batch resumes by re-using the same stamp (see ``--run-stamp`` /
    ``--continue``).
    """
    parts = [run_stamp, combo_label, entry.hf_id]
    if entry.task:
        parts.append(entry.task)
    return "0.0.0-" + _slugify_version("-".join(parts))


def _is_publish_conflict(proc: dict) -> bool:
    """True if a publish failed because the version already exists in the feed.

    Only specific version-exists / HTTP 409 markers are matched. A broad
    substring like ``"conflict"`` or a bare ``"409"`` is avoided on purpose: a
    false positive is treated as ``exists-skipped`` and deletes the local model
    dir, so an unrelated message mentioning those words would be a data-loss
    path.
    """
    blob = (proc.get("stdout", "") + proc.get("stderr", "")).lower()
    return any(
        marker in blob
        for marker in (
            "already exist",
            "packageversionexists",
            "status code: 409",
            "statuscode=409",
            "httpstatuscode: 409",
        )
    )


def _is_az_unavailable(proc: dict) -> bool:
    """True if an ``az`` invocation failed because the CLI/login is unavailable.

    Distinguishes a host-level Azure CLI problem (not installed, not logged in,
    token expired) from a per-package publish error (a one-off network blip, a
    slow/timed-out transfer, or a version conflict). A host-level problem recurs
    for every remaining combo, so the caller aborts the whole run; a per-combo
    error is recorded and the run continues.

    A bare timeout is deliberately *not* treated as "unavailable": with per-combo
    uploads a timeout is almost always a slow/large transfer, which the caller
    handles as a per-combo ``timeout`` (clean up + continue). Only an explicit
    auth marker (below) or ``az`` missing from PATH triggers an abort -- so a
    hung re-auth that still emits an auth marker is caught, while a plain slow
    upload is not.
    """
    if proc.get("exit_code") == 127:  # az not found on PATH (see _run_az)
        return True
    blob = (proc.get("stdout", "") + proc.get("stderr", "")).lower()
    return any(
        marker in blob
        for marker in (
            "az login",
            "not logged in",
            "no subscription found",
            "interactive authentication is needed",
            "authenticationfailed",
            "authentication failed",
            "aadsts",
            "token has expired",
            "expired token",
            "refresh token",
            "re-authenticate",
            "az account set",
        )
    )


def _upload_model_dir(
    args: argparse.Namespace, model_dir: Path, version: str, timeout: int
) -> dict:
    """Publish a model dir to the feed as a Universal Package version."""
    return _run_az(
        [
            "artifacts",
            "universal",
            "publish",
            "--organization",
            args.feed_org,
            "--project",
            args.feed_project,
            "--scope",
            "project",
            "--feed",
            args.feed,
            "--name",
            args.package_name,
            "--version",
            version,
            "--path",
            str(model_dir),
            "--description",
            f"build-only artifacts: {version}",
        ],
        timeout,
    )


def _classify_upload(up: dict, args: argparse.Namespace) -> str:
    """Classify an ``az universal publish`` result into an upload status.

    Pure (no side effects) so the caller owns printing, recording, counters, and
    cleanup. Returns one of:

    - ``uploaded``: published successfully.
    - ``exists-skipped``: version already on the feed and --upload-skip-existing.
    - ``auth-abort``: host-level az failure (not installed / not logged in /
      token expired) -> the caller cleans up and aborts the whole run.
    - ``timeout``: the publish was killed by the timeout (slow/large transfer)
      -> the caller cleans up and continues with the next combo.
    - ``failed``: any other per-combo publish error -> clean up and continue.
    """
    if up.get("exit_code") == 0:
        return "uploaded"
    conflict = _is_publish_conflict(up)
    if conflict and args.upload_skip_existing:
        return "exists-skipped"
    if not conflict and _is_az_unavailable(up):
        return "auth-abort"
    if up.get("timeout"):
        return "timeout"
    return "failed"


def _load_results(path: Path) -> dict:
    """Load the build_only_results.json log, or {} on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_results(path: Path, results: dict) -> None:
    """Persist the build-only results log (combo version -> outcome)."""
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree to bound disk usage; warn (never raise) on error."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        safe_print(f"  [cleanup] Removed local: {path}")
    except OSError as exc:
        safe_print(f"  [cleanup] Warning: could not remove {path}: {exc}")


def _feed_org_name(feed_org: str) -> str:
    """Extract the Azure DevOps org name from a feed-org URL.

    ``https://dev.azure.com/microsoft`` -> ``microsoft``
    ``https://microsoft.visualstudio.com`` -> ``microsoft``
    """
    host_and_path = feed_org.rstrip("/").split("://", 1)[-1]
    host, _, path = host_and_path.partition("/")
    if path:
        return path.split("/")[0]
    if host.endswith(".visualstudio.com"):
        return host.split(".", 1)[0]
    return host


def _fetch_feed_versions(
    args: argparse.Namespace, run_stamp: str, timeout: int = 180
) -> set[str] | None:
    """Return package versions already published to the feed for ``run_stamp``.

    The local manifest is only written after a successful upload, so a fresh
    ``--output-dir`` starts empty even when the feed already holds versions from
    a previous run. Querying the feed makes ``--continue`` authoritative: a model
    is skipped if its version exists on the feed, regardless of local state.

    Two ``&``-free REST GETs are used because ``az`` resolves to ``az.cmd``, which
    runs through cmd.exe and would split a query string on ``&`` (dropping every
    parameter after the first):
        1. list packages, find the UPack package by name -> package GUID,
        2. list that package's versions.

    Returns the set of lowercased versions matching the ``0.0.0-<run_stamp>-``
    prefix, an empty set if the feed is reachable but has no such versions yet,
    or ``None`` if the feed could not be queried (caller falls back to the local
    manifest only).
    """
    org = _feed_org_name(args.feed_org)
    base = (
        f"https://feeds.dev.azure.com/{org}/{args.feed_project}/_apis/packaging/feeds/{args.feed}"
    )

    def _get_json(url: str) -> dict | None:
        res = _run_az(
            ["rest", "--method", "get", "--resource", _AZURE_DEVOPS_RESOURCE, "--url", url],
            timeout,
        )
        if res["exit_code"] != 0:
            return None
        try:
            return json.loads(res["stdout"])
        except (json.JSONDecodeError, TypeError):
            return None

    listing = _get_json(f"{base}/packages?api-version={_FEED_API_VERSION}")
    if listing is None:
        return None
    pkg_id: str | None = None
    for pkg in listing.get("value", []):
        name_matches = (pkg.get("name") or "").lower() == args.package_name.lower()
        is_upack = (pkg.get("protocolType") or "").lower() == "upack"
        if name_matches and is_upack:
            pkg_id = pkg.get("id")
            break
    if pkg_id is None:
        return set()  # feed reachable, package not published yet

    versions_doc = _get_json(f"{base}/packages/{pkg_id}/versions?api-version={_FEED_API_VERSION}")
    if versions_doc is None:
        return None
    prefix = f"0.0.0-{run_stamp}-"
    published: set[str] = set()
    for entry in versions_doc.get("value", []):
        version = (entry.get("version") or "").lower()
        if version.startswith(prefix):
            published.add(version)
    return published


def _run_build_only(entries: list[ModelEntry], args: argparse.Namespace) -> None:
    """Build each model to disk with --no-compile (no execution provider needed).

    ``winml build -o <dir> --no-compile`` writes the per-stage artifacts
    (export.onnx, optimized.onnx, quantized.onnx). Perf and accuracy are skipped.

    When neither --ep nor --device is pinned, every model is built once per EP in
    :data:`_BUILD_ONLY_EP_MATRIX`, each into a ``<model_dir>/<ep>_<device>/``
    subdir. When --ep or --device is pinned, a single build is written directly
    into ``<model_dir>``. Precision per combo follows the same policy as the eval
    path (NPU defaults to w8a16; CPU/GPU omit the flag; native-quant EPs skip).

    Every (model, combo) outcome -- build status and (when uploading) upload
    status -- is recorded to ``build_only_results.json`` so a partially-failing or
    interrupted run can be audited afterwards.

    With ``--upload``, each combo's dir is published to the Azure Artifacts feed as
    its own Universal Package version
    (``<package-name>@0.0.0-<run-stamp>-<ep>-<device>-<model-slug>``) as soon as it
    is built, then deleted locally -- so peak disk stays at ~one combo and a
    failed/timed-out upload of one combo cannot fill the disk. The identical
    ``export.onnx`` dedup applies only to the non-upload matrix path (uploaded
    combos are self-contained and deleted, so there is nothing to share).
    """
    output_dir = args.output_dir or Path(f"eval_results/{date.today().isoformat()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_environment_info(output_dir / "environment.json")

    # Results are recorded for EVERY run (upload or not) so failures/timeouts are
    # auditable. --continue resume, however, is upload-driven: without --upload
    # there is no feed to resume from, so all models are rebuilt.
    if args.continue_run and not args.upload:
        safe_print(
            "[warn] --continue has no effect in --build-only without --upload: "
            "no feed resume exists, so all models will be rebuilt."
        )

    use_matrix = args.ep is None and args.device == "auto"
    # Single combo uses an empty label (no subdir); matrix uses (label, ep, device).
    # In the pinned single-combo path the device may be "auto": precision is then
    # delegated to winml config's own auto-detection (the same omit-the-flag policy
    # _resolve_precision applies to CPU/GPU), instead of forcing w8a16 the way the
    # matrix does for its explicit *_npu combos.
    combos = list(_BUILD_ONLY_EP_MATRIX) if use_matrix else [("", args.ep, args.device)]

    results_path = output_dir / "build_only_results.json"
    results: dict = _load_results(results_path)
    run_stamp = _slugify_version(args.run_stamp or date.today().strftime("%Y%m%d"))

    # Verify feed prerequisites once, up front. --upload exists to keep disk
    # bounded, so a broken az setup must abort (not silently fall back to
    # keeping everything local, which would fill the disk it was meant to save).
    if args.upload:
        err = _ensure_feed_ready()
        if err is not None:
            safe_print(f"[upload] Cannot upload: {err}")
            sys.exit(2)
        safe_print(
            f"[upload] Feed ready: {args.feed_org} / {args.feed_project} / "
            f"feed={args.feed} / package={args.package_name} | run-stamp={run_stamp}"
        )
        if args.continue_run:
            safe_print("[upload] Continue: skipping combos already uploaded for this run-stamp")
            # Seed results from the feed: a fresh --output-dir has no local
            # results, so the feed is the source of truth for what's published.
            # Best-effort — a query failure falls back to local-results-only.
            feed_versions = _fetch_feed_versions(args, run_stamp)
            if feed_versions is None:
                safe_print("[upload] Continue: could not query feed; relying on local results only")
            else:
                seeded = 0
                for feed_version in feed_versions:
                    prev = results.get(feed_version)
                    if not prev or prev.get("upload_status") not in ("uploaded", "exists-skipped"):
                        results[feed_version] = {"upload_status": "uploaded", "source": "feed"}
                        seeded += 1
                safe_print(
                    f"[upload] Continue: feed has {len(feed_versions)} version(s) for "
                    f"run-stamp {run_stamp}; {seeded} not in local results -> will skip"
                )

    safe_print(f"Build-only: {len(entries)} models -> {output_dir}")
    if use_matrix:
        safe_print(
            f"EP matrix ({len(combos)}): {', '.join(c[0] for c in combos)} | "
            f"Timeout: {args.timeout}s | Compile skipped (no EP required)"
        )
    else:
        safe_print(
            f"Device: {args.device} | EP: {args.ep or 'auto'} | "
            f"Timeout: {args.timeout}s | Compile skipped (no EP required)"
        )

    total_builds = len(entries) * len(combos)
    built_ok = 0
    build_fail = 0
    uploaded = 0
    upload_fail = 0
    upload_timeout = 0
    interrupted = False

    def _record(
        combo_version: str,
        entry: ModelEntry,
        ep: str | None,
        device: str,
        combo_label: str,
        build_status: str,
        build_stage: str,
        upload_status: str,
        error: str = "",
    ) -> None:
        """Record one (model, combo) outcome and flush the results log to disk."""
        rec: dict = {
            "hf_id": entry.hf_id,
            "task": entry.task,
            "ep": ep,
            "device": device,
            "combo": combo_label or "(pinned)",
            "package": args.package_name,
            "run_stamp": run_stamp,
            "build_status": build_status,
            "build_stage": build_stage,
            "upload_status": upload_status,
            "recorded_at": _utc_now(),
        }
        if error:
            rec["error"] = error
        results[combo_version] = rec
        _write_results(results_path, results)

    for i, entry in enumerate(entries, 1):
        label = f"{entry.hf_id} / {entry.task}" if entry.task else entry.hf_id
        model_dir = model_result_dir(output_dir, entry.hf_id, entry.task)
        shared_dir = model_dir / "_shared"
        canonical_hash: str | None = None

        safe_print(f"\n[{i}/{len(entries)}] {label}  ({entry.priority}, {entry.group})")

        for combo_label, ep, device in combos:
            build_dir = model_dir / combo_label if combo_label else model_dir
            tag = f"  [{combo_label}]" if combo_label else ""
            effective_label = combo_label or f"{ep or 'auto'}_{device}"
            combo_version = _feed_version_for(entry, run_stamp, effective_label)

            # Per-combo resume (upload mode): skip combos already in the feed so a
            # rerun only (re)builds the missing/failed ones, not the whole matrix.
            if args.upload and args.continue_run:
                prev = results.get(combo_version)
                if prev and prev.get("upload_status") in ("uploaded", "exists-skipped"):
                    origin = " via feed" if prev.get("source") == "feed" else ""
                    safe_print(f"  [skip]{tag} {prev['upload_status']}{origin}: {combo_version}")
                    continue

            precision = _resolve_precision(device, entry.precision, ep=ep)
            try:
                build = _run_build(
                    entry, device, precision, args.timeout, build_dir, ep=ep, build_only=True
                )
            except KeyboardInterrupt:
                safe_print("\n\n[Ctrl+C] Interrupted.")
                interrupted = True
                break

            # ---- Build failed (non-zero exit) or was killed by the timeout ----
            if not build["success"]:
                proc = build.get("proc") or {}
                is_timeout = bool(proc.get("timeout"))
                build_status = "timeout" if is_timeout else "failed"
                stage = build.get("stage", "build")
                build_fail += 1
                safe_print(f"  [BUILD {build_status.upper()} @ {stage}]{tag}")
                combined = (proc.get("stdout", "") + proc.get("stderr", "")).strip()
                err_tail = "\n".join(combined.splitlines()[-12:]) if combined else ""
                if args.verbose and combined:
                    for line in combined.splitlines()[-12:]:
                        safe_print(f"    {line}")
                # Upload mode bounds disk: drop the failed combo's partial
                # artifacts. Non-upload mode keeps them for inspection.
                if args.upload and not args.keep_local:
                    _safe_rmtree(build_dir)
                _record(
                    combo_version,
                    entry,
                    ep,
                    device,
                    combo_label,
                    build_status=build_status,
                    build_stage=stage,
                    upload_status="skipped" if args.upload else "n/a",
                    error=err_tail,
                )
                continue

            # ---- Build succeeded ----
            built_ok += 1
            safe_print(f"  [OK]{tag} artifacts -> {build_dir}")

            if not args.upload:
                # Dedup the EP-independent export into _shared/ (matrix only).
                if use_matrix:
                    canonical_hash = _dedup_export(
                        build_dir, shared_dir, canonical_hash, combo_label
                    )
                _record(
                    combo_version,
                    entry,
                    ep,
                    device,
                    combo_label,
                    build_status="ok",
                    build_stage="complete",
                    upload_status="n/a",
                )
                continue

            # ---- Upload this combo, then drop it locally to bound disk usage ----
            safe_print(f"  [upload]{tag} {args.package_name}@{combo_version} ...")
            up = _upload_model_dir(args, build_dir, combo_version, args.timeout)
            upload_status = _classify_upload(up, args)
            err_tail = ""
            if upload_status in ("timeout", "failed", "auth-abort"):
                blob = (up.get("stderr", "") or up.get("stdout", "")).strip()
                err_tail = "\n".join(blob.splitlines()[-12:])

            if upload_status == "uploaded":
                uploaded += 1
                safe_print(f"  [upload OK]{tag} {combo_version}")
            elif upload_status == "exists-skipped":
                uploaded += 1
                safe_print(f"  [upload SKIP]{tag} version exists: {combo_version}")
            elif upload_status == "timeout":
                # A timed-out upload is almost always a slow/large transfer, not a
                # host-level az problem. Drop the local copy to bound disk and
                # continue; retry later with --continue + the same --run-stamp.
                upload_timeout += 1
                safe_print(f"  [upload TIMEOUT]{tag} {combo_version} (cleaned, continuing)")
            else:  # "failed" or "auth-abort"
                upload_fail += 1
                safe_print(f"  [upload FAIL]{tag} {args.package_name}@{combo_version}")
                if args.verbose or upload_status == "auth-abort":
                    for line in err_tail.splitlines():
                        safe_print(f"    {line}")

            # An auth failure is recorded as a plain "failed" outcome (the combo is
            # not in the feed); the distinction only drives the abort below.
            _record(
                combo_version,
                entry,
                ep,
                device,
                combo_label,
                build_status="ok",
                build_stage="complete",
                upload_status="failed" if upload_status == "auth-abort" else upload_status,
                error=err_tail,
            )

            # Bound disk: drop the local copy after every outcome -- the artifacts
            # are either safely in the feed or intentionally discarded.
            # --keep-local opts out for debugging.
            if not args.keep_local:
                _safe_rmtree(build_dir)

            # A host-level az failure (not logged in, token expired) recurs for
            # every remaining combo, so abort now -- already-uploaded combos are
            # skipped on resume via --continue + the same --run-stamp.
            if upload_status == "auth-abort":
                safe_print(
                    "\n[upload] ABORT: Azure CLI is unavailable "
                    "(not logged in / token expired). Re-run 'az login', then "
                    f"resume with --continue and the same --run-stamp {run_stamp}."
                )
                sys.exit(3)

        if interrupted:
            break

        # Drop a now-empty model dir (all combos uploaded + deleted in upload mode).
        if args.upload and not args.keep_local and model_dir.exists():
            try:
                if not any(model_dir.iterdir()):
                    model_dir.rmdir()
            except OSError:
                pass

        # Clean caches once per model (after all EP combos finish), not per
        # combo: combos share the same HF download, so clearing between
        # combos forces redundant re-downloads of the same weights.
        if args.clean_cache:
            _clear_disk_caches()

    tail = (
        f" | uploaded {uploaded}, upload-failed {upload_fail}, upload-timeout {upload_timeout}"
        if args.upload
        else ""
    )
    safe_print(
        f"\nBuild-only complete: built {built_ok}/{total_builds} "
        f"(build-failed {build_fail}){tail} -> {output_dir}\n"
        f"Results log: {results_path}"
    )


# ---------------------------------------------------------------------------
# Perf phase
# ---------------------------------------------------------------------------


def run_model(
    entry: ModelEntry,
    device: str,
    timeout: int,
    onnx_paths: dict[str, str] | None = None,
    ep: str | None = None,
) -> dict:
    """Execute winml perf for one or more ONNX models. Returns merged result dict.

    When onnx_paths is provided, benchmarks each pre-built ONNX directly.
    Single model is the {"": path} case. Results are merged (worst exit
    code, concatenated stdout/stderr, summed elapsed).
    """
    if not onnx_paths:
        # No pre-built paths: fall back to HF model ID (single model only).
        # winml perf builds internally; the same --no-quant gating used by
        # _run_build must apply here so the EP sees the unquantized variant.
        precision = _resolve_precision(device, None, ep=ep)
        args = [
            *WINML_CLI,
            "perf",
            "-m",
            entry.hf_id,
            "--device",
            device,
        ]
        if precision:
            args += ["--precision", precision]
        if entry.task:
            args += ["--task", entry.task]
        if ep:
            args += ["--ep", ep]
        if _should_skip_winml_quant(ep):
            args += ["--no-quant"]
        args += ["--iterations", "10", "--warmup", "2"]
        args += entry.perf_args

        proc = _run_subprocess(args, timeout)
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

    # Run perf for each sub-model and merge results
    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_elapsed = 0.0
    worst_exit = 0
    any_timeout = False
    commands: list[str] = []

    for label, path in onnx_paths.items():
        if label:
            safe_print(f"    perf: {label}")

        args = [*WINML_CLI, "perf", "-m", path, "--device", device]
        if ep:
            args += ["--ep", ep]
        args += ["--iterations", "10", "--warmup", "2"]
        args += entry.perf_args

        proc = _run_subprocess(args, timeout)
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

    return {
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
    cache_dir = Path(ds_config.get("dataset", EVAL_DATASETS_CACHE / script_path.stem)).expanduser()

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
    if ds_config.get("revision"):
        args += ["--dataset-revision", ds_config["revision"]]
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
    if ds_config.get("revision"):
        args += ["--dataset-revision", ds_config["revision"]]
    if ds_config.get("columns_mapping"):
        args += ["--columns-mapping", json.dumps(ds_config["columns_mapping"])]
    if ds_config.get("label_mapping_file"):
        args += ["--label-mapping-file", ds_config["label_mapping_file"]]
    winml_key = ds_config.get("winml_metric_key") or ds_config.get("metric")
    if winml_key:
        args += ["--winml-metric-key", winml_key]

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
) -> dict:
    """Run winml eval + pytorch baseline for one model. Returns accuracy sub-section dict."""
    ds_config = get_dataset_config(entry.hf_id, entry.task) or {}

    # Build local dataset if a build_script is configured
    _build_dataset(ds_config, timeout)

    winml = _run_winml_eval(entry, device, timeout, ds_config, model_dir, onnx_paths, ep=ep)

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

    # `winml sys --format json` captures hardware details (devices, EPs,
    # backends) that the lightweight package-version probes above miss.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "winml", "sys", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            info["winml_sys"] = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as exc:
        logger.debug("winml sys skipped: %s", exc)

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
        "--build-only",
        dest="build_only",
        action="store_true",
        help=(
            "Build-only mode: run config + build with --no-compile and write each "
            "stage's ONNX (export/optimize/quantize) to the output dir. No execution "
            "provider required; perf/accuracy are skipped. When --ep/--device are "
            "omitted, builds once per EP in the build-only matrix "
            "(qnn/openvino/mlas/dml/vitisai) into <model_dir>/<ep>_<device>/ subdirs. "
            "Identical export.onnx is deduped into <model_dir>/_shared/."
        ),
    )
    # --- Build-only feed upload (Azure Artifacts Universal Packages) ---
    parser.add_argument(
        "--upload",
        dest="upload",
        action="store_true",
        help=(
            "Build-only only: as each EP/device combo is built, publish that combo "
            "to an Azure Artifacts feed as its own Universal Package version and "
            "delete it locally to bound disk usage (peak disk ~= one combo). Auth "
            "via 'az login' (no PAT)."
        ),
    )
    parser.add_argument(
        "--feed",
        default="Modelkit",
        help="Azure Artifacts feed name for --upload (default: Modelkit)",
    )
    parser.add_argument(
        "--feed-org",
        default="https://dev.azure.com/microsoft",
        help="Azure DevOps org URL for --upload (default: https://dev.azure.com/microsoft)",
    )
    parser.add_argument(
        "--feed-project",
        default="windows.ai.toolkit",
        help="Azure DevOps project for the project-scoped feed (default: windows.ai.toolkit)",
    )
    parser.add_argument(
        "--package-name",
        default="winml-cli-models",
        help="Universal Package name for --upload (default: winml-cli-models)",
    )
    parser.add_argument(
        "--run-stamp",
        dest="run_stamp",
        default=None,
        help=(
            "Batch stamp used as the feed version prefix "
            "(<stamp>-<ep>-<device>-<model-slug>). "
            "Defaults to today's date (YYYYMMDD). Pass the SAME stamp with "
            "--continue to resume an interrupted batch."
        ),
    )
    parser.add_argument(
        "--keep-local",
        dest="keep_local",
        action="store_true",
        help=(
            "With --upload, do NOT delete local combo dirs after any outcome "
            "(uploaded / failed / timed-out / build-failed). For debugging."
        ),
    )
    parser.add_argument(
        "--upload-skip-existing",
        dest="upload_skip_existing",
        action="store_true",
        help=(
            "With --upload: treat a 'version already exists' publish conflict as "
            "success (and delete the local copy) instead of a failure. This does "
            "NOT skip the build. To skip rebuilding already-uploaded models "
            "entirely, use --continue."
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-subprocess timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--clean-cache",
        dest="clean_cache",
        action="store_true",
        help="Delete caches and leaked temp files after each model evaluation (saves disk space)",
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
        "--raw-output",
        action="store_true",
        help="Keep raw subprocess output in eval_result.json without sanitization",
    )
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

    # Build-only mode: generate export+optimize+quantize artifacts only (no EP).
    # Loops the EP matrix unless --ep/--device pinned. Skips perf/accuracy.
    if args.build_only:
        _run_build_only(entries, args)
        return

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
                _resolve_precision(args.device, entry.precision, ep=args.ep),
                args.timeout,
                model_dir,
                ep=args.ep,
            )
            onnx_paths = build_result["onnx_paths"] if build_result["success"] else {}
            onnx_size = _compute_onnx_size(onnx_paths)

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
                )
            elif args.eval_type == "perf":
                perf_proc = run_model(entry, args.device, args.timeout, onnx_paths, ep=args.ep)
            else:
                # "both": perf → eval
                perf_proc = run_model(entry, args.device, args.timeout, onnx_paths, ep=args.ep)
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
                    )

        except KeyboardInterrupt:
            safe_print("\n\n[Ctrl+C] Interrupted — generating reports for completed models...")
            interrupted = True
            break

        result = build_eval_result(
            entry,
            perf_proc,
            args.device,
            eval_types_run,
            accuracy_result,
            ep=args.ep,
            onnx_size_bytes=onnx_size,
            sanitize_fn=None if args.raw_output else _sanitize_output,
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
