# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generate ONNX models for E2E evaluation using winml export.

Exports HuggingFace models to ONNX using the same task/config parameters
as run_eval.py, so the generated models are identical.

Usage:
    # Single model (looks up task from registry)
    uv run python scripts/e2e_eval/gen_eval_model.py --model microsoft/resnet-50

    # All models from registry
    uv run python scripts/e2e_eval/gen_eval_model.py --registry scripts/e2e_eval/testsets/models_all.json

    # Filter by priority/task/group
    uv run python scripts/e2e_eval/gen_eval_model.py --priority P0 --task image-classification

    # Custom output directory
    uv run python scripts/e2e_eval/gen_eval_model.py --model google/vit-base-patch16-224 --output-dir eval_models
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from utils.registry import ModelEntry, filter_registry, load_registry, make_adhoc_entry


WINML_CLI = [sys.executable, "-m", "winml.modelkit.cli"]
DEFAULT_REGISTRY = Path(__file__).parent / "testsets" / "models_all.json"
DEFAULT_OUTPUT_DIR = Path("eval_models")
_DEFAULT_TIMEOUT = 600


def safe_print(text: str) -> None:
    """Print text, replacing non-ASCII characters on encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _run_subprocess(args: list[str], timeout: int) -> dict:
    """Run a subprocess and return result dict with stdout, stderr, exit_code."""
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

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _read(stream, chunks):
        for chunk in iter(lambda: stream.read(4096), b""):
            chunks.append(chunk)

    t_out = threading.Thread(target=_read, args=(proc.stdout, stdout_chunks))
    t_err = threading.Thread(target=_read, args=(proc.stderr, stderr_chunks))
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass  # Already killed; ignore if OS is slow to reap

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    elapsed = time.perf_counter() - start
    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": proc.returncode or (1 if timed_out else 0),
        "elapsed": round(elapsed, 2),
        "timeout": timed_out,
        "command": " ".join(args),
    }


def _make_slug(hf_id: str, task: str) -> str:
    """Build a filesystem-safe slug from HF model ID and task."""
    slug = hf_id.replace("/", "__")
    if task:
        slug += f"__{task}"
    return slug


def export_model(
    entry: ModelEntry,
    output_dir: Path,
    timeout: int,
    config_path: Path | None = None,
) -> dict:
    """Run winml export for one model.

    Uses winml config to generate a build config (for consistent export
    settings with run_eval.py), then calls winml export with that config.

    For composite models (e.g., encoder-decoder), handles multiple
    sub-configs from winml config.

    Returns dict with: success, onnx_path, elapsed.
    """
    slug = _make_slug(entry.hf_id, entry.task)
    model_dir = output_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_dir / "export.onnx"

    # Skip if already exported (single model or composite sub-exports)
    composite_exports = list(model_dir.glob("export_*.onnx"))
    if output_path.exists() and output_path.stat().st_size > 0:
        safe_print(f"  [cached] {output_path}")
        return {"success": True, "onnx_path": str(output_path), "elapsed": 0}
    if composite_exports and all(f.stat().st_size > 0 for f in composite_exports):
        paths = ", ".join(str(f) for f in composite_exports)
        safe_print(f"  [cached] {paths}")
        return {"success": True, "onnx_path": paths, "elapsed": 0}

    # Step 1: winml config to get consistent export settings
    cfg_path = config_path or (model_dir / "build_config.json")

    # Clean stale sub-configs from prior runs
    for stale in cfg_path.parent.glob(f"{cfg_path.stem}_*.json"):
        stale.unlink(missing_ok=True)

    config_args = [
        *WINML_CLI,
        "config",
        "-m",
        entry.hf_id,
        "-o",
        str(cfg_path),
    ]
    if entry.task:
        config_args += ["--task", entry.task]

    safe_print(
        f"  [config] winml config -m {entry.hf_id}" + (f" -t {entry.task}" if entry.task else "")
    )
    config_proc = _run_subprocess(config_args, timeout)
    if config_proc["exit_code"] != 0:
        safe_print(f"  [ERROR] Config failed (rc={config_proc['exit_code']})")
        stderr_tail = config_proc["stderr"][-300:] if config_proc["stderr"] else ""
        if stderr_tail:
            safe_print(f"  {stderr_tail}")
        return {"success": False, "onnx_path": None, "elapsed": config_proc["elapsed"]}

    total_elapsed = config_proc["elapsed"]

    # Check for composite models (multiple sub-configs)
    sub_configs = sorted(cfg_path.parent.glob(f"{cfg_path.stem}_*.json"))
    if sub_configs:
        # Composite model: export each sub-config
        all_paths: list[str] = []
        for sub_cfg in sub_configs:
            label = sub_cfg.stem.removeprefix(f"{cfg_path.stem}_")
            sub_output = model_dir / f"export_{label}.onnx"

            export_args = [
                *WINML_CLI,
                "export",
                "-m",
                entry.hf_id,
                "-o",
                str(sub_output),
                "-c",
                str(sub_cfg),
            ]
            if entry.task:
                export_args += ["-t", entry.task]

            safe_print(f"  [export] {label}: winml export -m {entry.hf_id} -c {sub_cfg.name}")
            export_proc = _run_subprocess(export_args, timeout)
            total_elapsed += export_proc["elapsed"]

            if export_proc["exit_code"] != 0:
                safe_print(f"  [ERROR] Export {label} failed (rc={export_proc['exit_code']})")
                stderr_tail = export_proc["stderr"][-300:] if export_proc["stderr"] else ""
                if stderr_tail:
                    safe_print(f"  {stderr_tail}")
                return {"success": False, "onnx_path": None, "elapsed": total_elapsed}

            if sub_output.exists():
                all_paths.append(str(sub_output))

        return {
            "success": len(all_paths) == len(sub_configs),
            "onnx_path": ", ".join(all_paths) if all_paths else None,
            "elapsed": total_elapsed,
        }

    # Single model: export with config
    export_args = [
        *WINML_CLI,
        "export",
        "-m",
        entry.hf_id,
        "-o",
        str(output_path),
        "-c",
        str(cfg_path),
    ]
    if entry.task:
        export_args += ["-t", entry.task]

    safe_print(f"  [export] winml export -m {entry.hf_id} -o {output_path.name} -c {cfg_path.name}")
    export_proc = _run_subprocess(export_args, timeout)
    total_elapsed += export_proc["elapsed"]

    if export_proc["exit_code"] != 0:
        safe_print(f"  [ERROR] Export failed (rc={export_proc['exit_code']})")
        stderr_tail = export_proc["stderr"][-300:] if export_proc["stderr"] else ""
        if stderr_tail:
            safe_print(f"  {stderr_tail}")
        return {"success": False, "onnx_path": None, "elapsed": total_elapsed}

    if not output_path.exists():
        safe_print("  [ERROR] Export returned 0 but no ONNX file produced")
        return {"success": False, "onnx_path": None, "elapsed": total_elapsed}

    # Report model size
    size_mb = output_path.stat().st_size / 1024**2
    safe_print(f"  [export] {output_path.name} ({size_mb:.1f} MB)")

    return {"success": True, "onnx_path": str(output_path), "elapsed": total_elapsed}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ONNX models for E2E evaluation via winml export"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"Model registry JSON (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument("--model", help="Single model by HF ID (looks up task from registry)")
    parser.add_argument("--task", help="Filter by HF task (or override task for --model)")
    parser.add_argument(
        "--priority",
        nargs="+",
        choices=["P0", "P1", "P2", "P3"],
        default=["P0", "P1", "P2"],
        metavar="{P0,P1,P2,P3}",
        help="Filter by priority (default: P0 P1 P2)",
    )
    parser.add_argument("--model-type", help="Filter by model_type")
    parser.add_argument("--group", help="Filter by group")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for exported models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT,
        help=f"Per-subprocess timeout in seconds (default: {_DEFAULT_TIMEOUT})",
    )
    parser.add_argument("--list", action="store_true", help="List filtered models and exit")
    return parser.parse_args()


def main() -> None:
    """Generate ONNX models for E2E evaluation."""
    args = parse_args()

    # Load and filter models (same logic as run_eval.py)
    if args.model:
        matched_entry: ModelEntry | None = None
        try:
            registry_entries = load_registry(args.registry)
            for e in registry_entries:
                if e.hf_id == args.model and (not args.task or e.task == args.task):
                    matched_entry = e
                    break
            if matched_entry is None:
                for e in registry_entries:
                    if e.hf_id == args.model:
                        matched_entry = e
                        break
        except Exception as ex:
            safe_print(f"  [registry] Optional enrichment skipped: {ex}")
        if matched_entry is not None:
            if args.task and args.task != matched_entry.task:
                matched_entry = ModelEntry(
                    hf_id=matched_entry.hf_id,
                    task=args.task,
                    model_type=matched_entry.model_type,
                    group=matched_entry.group,
                    priority=matched_entry.priority,
                )
            entries = [matched_entry]
        else:
            entries = [make_adhoc_entry(args.model, args.task)]
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

    if args.list:
        for e in entries:
            safe_print(f"  {e.hf_id:50s}  {e.task:30s}  {e.priority}")
        safe_print(f"\n{len(entries)} models")
        return

    output_dir = args.output_dir.resolve()
    safe_print(f"Models to export: {len(entries)}")
    safe_print(f"Output: {output_dir}")
    safe_print("")

    t_start = time.perf_counter()
    success_count = 0
    fail_count = 0
    results: list[dict] = []

    for i, entry in enumerate(entries, 1):
        safe_print(f"[{i}/{len(entries)}] {entry.hf_id} ({entry.task})")

        result = export_model(
            entry,
            output_dir=output_dir,
            timeout=args.timeout,
        )
        results.append({"model": entry.hf_id, "task": entry.task, **result})

        if result["success"]:
            success_count += 1
            safe_print(f"  [OK] {result['onnx_path']}  ({result['elapsed']:.1f}s)")
        else:
            fail_count += 1
            safe_print(f"  [FAIL] ({result['elapsed']:.1f}s)")

    elapsed = time.perf_counter() - t_start
    safe_print(f"\nDone: {success_count} succeeded, {fail_count} failed, {elapsed:.1f}s total")

    # Write summary JSON
    summary_path = output_dir / "gen_eval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
