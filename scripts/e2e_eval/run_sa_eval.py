# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SA accuracy evaluation — four-stage self-contained pipeline.

Reads models from model_with_acc.json, runs export + graph optimize,
SA pre-check, capability-driven optimization, SA post-check, and optional
EPContext diff against cached compiled ONNX.

Pipeline per model:
  Stage 1: winml export + Python optimize_onnx (default)
           → graph_optimized.onnx
  Stage 2: ONNXStaticAnalyzer (enable_information=True)
           → sa_pre.json + optim_config
  Stage 3: Python optimize_onnx(**optim_config) → sa_optimized.onnx
  Stage 4: ONNXStaticAnalyzer → sa_post.json
  Stage 5: EPContext diff (optional, uses ~/.cache/winml/artifacts/)
           → epcontext comparison vs sa_post predictions

Usage:
    uv run python scripts/e2e_eval/run_sa_eval.py
    uv run python scripts/e2e_eval/run_sa_eval.py --model ProsusAI/finbert
    uv run python scripts/e2e_eval/run_sa_eval.py --output-dir sa_eval_results/2026-03-27
    uv run python scripts/e2e_eval/run_sa_eval.py --use-cache
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from sa_comparison import (
    compare_sa_vs_epcontext,
    compute_delta,
    get_level_patterns,
    get_sa_summary,
    parse_sa_json,
    resolve_auto_ep_device,
    run_sa_with_info,
)
from sa_report import generate_sa_html_report


MODELS_FILE = Path(__file__).parent / "testsets" / "models_with_acc.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_print(text: str) -> None:
    """Print text, replacing non-ASCII characters on encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def make_slug(hf_id: str, task: str) -> str:
    """Build a filesystem-safe slug from HF model ID and task."""
    slug = hf_id.replace("/", "__")
    if task:
        slug += f"__{task}"
    return slug


def is_cached(path: Path) -> bool:
    """Return True if path exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def run_winmlcli_export(hf_id: str, task: str, output: Path) -> tuple[int, str]:
    """Run winml export via subprocess. Returns (rc, stderr_tail)."""
    args = [
        sys.executable,
        "-m",
        "winml.modelkit.cli",
        "export",
        "--model",
        hf_id,
        "--output",
        str(output),
        "--clean-onnx",
    ]
    if task:
        args += ["--task", task]
    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.returncode, (result.stderr or "").strip()[-500:]


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage1_export_optimize(
    hf_id: str,
    task: str,
    model_dir: Path,
    use_cache: bool,
) -> tuple[Path | None, str | None]:
    """Export HF model and apply baseline graph optimization.

    Returns (graph_optimized_path, None) on success,
    or (None, skip_reason) on failure.
    """
    from winml.modelkit.optim import optimize_onnx

    exported_path = model_dir / "exported.onnx"
    graph_opt_path = model_dir / "graph_optimized.onnx"

    # Stage 1a: Export (subprocess — HF download + tracing)
    if use_cache and is_cached(exported_path):
        safe_print("  [Stage 1a] Export (cached)")
    else:
        safe_print(f"  [Stage 1a] Exporting {hf_id}...")
        rc, stderr = run_winmlcli_export(hf_id, task, exported_path)
        if rc != 0 or not is_cached(exported_path):
            safe_print(f"  [ERROR] Export failed (rc={rc}): {stderr}")
            return None, "SKIP_EXPORT"
        safe_print(f"  [Stage 1a] Exported: {exported_path.name}")

    # Stage 1b: Baseline graph optimization (Python API)
    if use_cache and is_cached(graph_opt_path):
        safe_print("  [Stage 1b] Graph optimize (cached)")
    else:
        safe_print("  [Stage 1b] Applying baseline graph optimization...")
        try:
            optimize_onnx(str(exported_path), str(graph_opt_path))
        except Exception as e:
            safe_print(f"  [ERROR] Graph optimize failed: {e}")
            return None, "SKIP_GRAPH_OPTIM"
        if not is_cached(graph_opt_path):
            safe_print("  [ERROR] Graph optimize produced no output")
            return None, "SKIP_GRAPH_OPTIM"
        safe_print(f"  [Stage 1b] Optimized: {graph_opt_path.name}")

    return graph_opt_path, None


def stage2_sa_pre(
    model_dir: Path,
    graph_opt_path: Path,
    use_cache: bool,
    ep: str = "auto",
    device: str = "auto",
) -> tuple[dict[str, str], dict, list[dict]] | None:
    """Run SA with information on graph_optimized.onnx.

    Returns (classifications, optim_config, info_items) or None on failure.
    """
    sa_pre_path = model_dir / "sa_pre.json"
    optim_record_path = model_dir / "optimization_flags.json"

    if use_cache and is_cached(sa_pre_path) and is_cached(optim_record_path):
        safe_print("  [Stage 2] SA pre-check (cached)")
        classifications = parse_sa_json(sa_pre_path, ep=ep)
        optim_record = json.loads(optim_record_path.read_text(encoding="utf-8"))
        optim_config = optim_record.get("optim_config", {})
        info_items = optim_record.get("info_items", [])
    else:
        safe_print("  [Stage 2] Running SA pre-check (with recommendations)...")
        try:
            classifications, optim_config, info_items = run_sa_with_info(
                graph_opt_path, sa_pre_path, ep=ep, device=device
            )
        except Exception as e:
            safe_print(f"  [ERROR] SA pre-check failed: {e}")
            return None
        # Persist optim_config + info so cache works
        optim_record_path.write_text(
            json.dumps({"optim_config": optim_config, "info_items": info_items}, indent=2),
            encoding="utf-8",
        )

    if not classifications:
        safe_print("  [ERROR] SA pre-check returned no classifications")
        return None

    summary = get_sa_summary(classifications)
    safe_print(
        f"  [Stage 2] Pre: SUPPORTED={summary['supported']} PARTIAL={summary['partial']} "
        f"UNSUPPORTED={summary['unsupported']} UNKNOWN={summary['unknown']} "
        f"({summary['supported_ratio']:.0%} supported)  optim_flags={list(optim_config.keys())}"
    )
    return classifications, optim_config, info_items


def stage3_capability_optimize(
    model_dir: Path,
    graph_opt_path: Path,
    optim_config: dict,
    use_cache: bool,
) -> Path | None:
    """Run capability-driven optimization using SA's recommended config.

    Uses optimize_onnx Python API directly with WinMLOptimizationConfig kwargs.
    Returns path to sa_optimized.onnx, or None on failure.
    """
    from winml.modelkit.optim import optimize_onnx

    sa_opt_path = model_dir / "sa_optimized.onnx"

    if use_cache and is_cached(sa_opt_path):
        safe_print(f"  [Stage 3] Capability optimize (cached, config={optim_config})")
        return sa_opt_path

    safe_print(f"  [Stage 3] Capability optimization (config={optim_config})...")
    try:
        optimize_onnx(str(graph_opt_path), str(sa_opt_path), **optim_config)
    except Exception as e:
        safe_print(f"  [ERROR] Capability optimize failed: {e}")
        return None

    if not is_cached(sa_opt_path):
        safe_print("  [ERROR] Capability optimize produced no output")
        return None

    safe_print(f"  [Stage 3] Optimized: {sa_opt_path.name}")
    return sa_opt_path


def stage4_sa_post(
    model_dir: Path,
    sa_opt_path: Path,
    use_cache: bool,
    ep: str = "auto",
    device: str = "auto",
) -> tuple[dict[str, str], list[dict]] | None:
    """Run SA on sa_optimized.onnx.

    Returns (classifications, info_items) or None on failure.
    """
    sa_post_path = model_dir / "sa_post.json"

    if use_cache and is_cached(sa_post_path):
        safe_print("  [Stage 4] SA post-check (cached)")
        classifications = parse_sa_json(sa_post_path, ep=ep)
        info_items = []
    else:
        safe_print("  [Stage 4] Running SA post-check...")
        try:
            classifications, _, info_items = run_sa_with_info(
                sa_opt_path, sa_post_path, ep=ep, device=device
            )
        except Exception as e:
            safe_print(f"  [ERROR] SA post-check failed: {e}")
            return None

    if not classifications:
        safe_print("  [ERROR] SA post-check returned no classifications")
        return None

    summary = get_sa_summary(classifications)
    safe_print(
        f"  [Stage 4] Post: SUPPORTED={summary['supported']} PARTIAL={summary['partial']} "
        f"UNSUPPORTED={summary['unsupported']} UNKNOWN={summary['unknown']} "
        f"({summary['supported_ratio']:.0%} supported)"
    )
    return classifications, info_items


def _run_compile(
    onnx_path: Path,
    output_dir: Path,
    device: str = "npu",
    ep: str | None = None,
) -> tuple[int, str]:
    """Run winml compile --device <device> --no-quantize. Returns (rc, stderr_tail)."""
    cmd = [
        sys.executable,
        "-m",
        "winml.modelkit.cli",
        "compile",
        "--model",
        str(onnx_path),
        "--device",
        device,
        "--no-quantize",
        "--output-dir",
        str(output_dir),
    ]
    if ep:
        cmd += ["--ep", ep]
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.returncode, (result.stderr or "").strip()[-500:]


def _compile_and_diff(
    label: str,
    onnx_path: Path,
    compiled_name: str,
    sa_predictions: dict[str, str],
    model_dir: Path,
    use_cache: bool,
    device: str = "npu",
    ep: str | None = None,
) -> dict | None:
    """Compile an ONNX and compare against SA predictions.

    Args:
        label: Log prefix, e.g. "5a (pre)" or "5b (post)".
        onnx_path: ONNX to compile (graph_optimized or sa_optimized).
        compiled_name: Expected output filename, e.g. "graph_optimized_qnn_ctx.onnx".
        sa_predictions: SA classifications to compare against the compilation result.
        model_dir: Directory for artifacts.
        use_cache: Skip compile if compiled_name already exists.

    Returns EPContext comparison dict or None on failure.
    """
    compiled_path = model_dir / compiled_name

    if use_cache and is_cached(compiled_path):
        safe_print(f"  [Stage {label}] Compile (cached): {compiled_path.name}")
    else:
        safe_print(f"  [Stage {label}] Compiling {onnx_path.name} → EPContext...")
        rc, _ = _run_compile(onnx_path, model_dir, device=device, ep=ep)
        if rc != 0 or not is_cached(compiled_path):
            safe_print(f"  [Stage {label}] Compile failed (rc={rc}) — skipping diff")
            return None
        safe_print(f"  [Stage {label}] Compiled: {compiled_path.name}")

    try:
        result = compare_sa_vs_epcontext(sa_predictions, compiled_path)
    except Exception as e:
        safe_print(f"  [Stage {label}] EPContext diff failed: {e}")
        return None

    s = result["summary"]
    safe_print(
        f"  [Stage {label}] TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']} "
        f"accuracy={s['accuracy']:.0%}"
    )
    return result


def stage5_compile_and_diff(
    model_dir: Path,
    graph_opt_path: Path,
    sa_opt_path: Path,
    sa_pre: dict[str, str],
    sa_post: dict[str, str],
    use_cache: bool,
    device: str = "npu",
    ep: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Stage 5: compile both graph_optimized and sa_optimized, diff each vs its SA.

    - 5a: graph_optimized.onnx  → compiled → compare vs sa_pre predictions
    - 5b: sa_optimized.onnx     → compiled → compare vs sa_post predictions

    Returns (epcontext_diff_pre, epcontext_diff_post).
    """
    diff_pre = _compile_and_diff(
        "5a (pre)",
        graph_opt_path,
        graph_opt_path.stem + "_qnn_ctx.onnx",
        sa_pre,
        model_dir,
        use_cache,
        device=device,
        ep=ep,
    )
    diff_post = _compile_and_diff(
        "5b (post)",
        sa_opt_path,
        sa_opt_path.stem + "_qnn_ctx.onnx",
        sa_post,
        model_dir,
        use_cache,
        device=device,
        ep=ep,
    )
    return diff_pre, diff_post


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_entry: dict,
    output_dir: Path,
    use_cache: bool,
    ep: str = "auto",
    device: str = "auto",
) -> dict | None:
    """Run the 4+1 stage SA eval pipeline for a single model."""
    hf_id = model_entry["hf_id"]
    task = model_entry.get("task", "")
    model_type = model_entry.get("model_type", "")

    safe_print(f"\n{'=' * 60}")
    safe_print(f"[sa_eval] {hf_id} ({task})")
    safe_print(f"{'=' * 60}")

    slug = make_slug(hf_id, task)
    model_dir = output_dir / "models" / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Stage 1
    graph_opt_path, skip_reason = stage1_export_optimize(hf_id, task, model_dir, use_cache)
    if graph_opt_path is None:
        return _skip_result(hf_id, task, model_type, skip_reason or "SKIP_EXPORT", model_dir)

    # Stage 2
    pre_result = stage2_sa_pre(model_dir, graph_opt_path, use_cache, ep=ep, device=device)
    if pre_result is None:
        return _skip_result(hf_id, task, model_type, "SKIP_SA_PRE", model_dir)
    sa_pre, optim_config, pre_info_items = pre_result

    # Stage 3
    sa_opt_path = stage3_capability_optimize(model_dir, graph_opt_path, optim_config, use_cache)
    if sa_opt_path is None:
        return _skip_result(hf_id, task, model_type, "SKIP_OPTIM", model_dir)

    # Stage 4
    post_result = stage4_sa_post(model_dir, sa_opt_path, use_cache, ep=ep, device=device)
    if post_result is None:
        return _skip_result(hf_id, task, model_type, "SKIP_SA_POST", model_dir)
    sa_post, post_info_items = post_result

    # Stage 5: compile both ONNXes → EPContext diff pre and post
    epcontext_diff_pre, epcontext_diff_post = stage5_compile_and_diff(
        model_dir,
        graph_opt_path,
        sa_opt_path,
        sa_pre,
        sa_post,
        use_cache,
        device=device.lower(),
        ep=ep,
    )

    elapsed = time.monotonic() - t0
    delta = compute_delta(sa_pre, sa_post)

    safe_print(
        f"  Delta: {len(delta['improved'])} improved, "
        f"{len(delta.get('fused_away', []))} fused_away, "
        f"{len(delta['regressed'])} regressed, "
        f"supported {delta['pre_supported_ratio']:.0%} -> {delta['post_supported_ratio']:.0%} "
        f"({delta['supported_ratio_delta']:+.0%})"
    )

    result: dict = {
        "model": hf_id,
        "task": task,
        "model_type": model_type,
        "status": "COMPLETE",
        "elapsed": round(elapsed, 2),
        "artifacts": {
            "exported_onnx": str(model_dir / "exported.onnx"),
            "graph_optimized_onnx": str(graph_opt_path),
            "sa_optimized_onnx": str(sa_opt_path),
        },
        "sa_pre": {
            "source_onnx": graph_opt_path.name,
            "classifications": sa_pre,
            "summary": get_sa_summary(sa_pre),
            "partial_patterns": get_level_patterns(sa_pre, "PARTIAL"),
            "unsupported_patterns": get_level_patterns(sa_pre, "UNSUPPORTED"),
            "unknown_patterns": get_level_patterns(sa_pre, "UNKNOWN"),
            "info_items": pre_info_items,
        },
        "optimization": {
            "optim_config": optim_config,
        },
        "sa_post": {
            "source_onnx": sa_opt_path.name,
            "classifications": sa_post,
            "summary": get_sa_summary(sa_post),
            "partial_patterns": get_level_patterns(sa_post, "PARTIAL"),
            "unsupported_patterns": get_level_patterns(sa_post, "UNSUPPORTED"),
            "unknown_patterns": get_level_patterns(sa_post, "UNKNOWN"),
            "info_items": post_info_items,
        },
        "delta": delta,
    }

    if epcontext_diff_pre:
        result["epcontext_diff_pre"] = epcontext_diff_pre
    if epcontext_diff_post:
        result["epcontext_diff_post"] = epcontext_diff_post

    out_file = model_dir / "sa_eval_result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"  Written: {out_file}")

    return result


def _skip_result(hf_id: str, task: str, model_type: str, status: str, model_dir: Path) -> dict:
    result = {"model": hf_id, "task": task, "model_type": model_type, "status": status}
    (model_dir / "sa_eval_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def build_aggregate_report(results: list[dict], models_input: Path) -> dict:
    """Build aggregate SA eval report from per-model results."""
    complete = [r for r in results if r.get("status") == "COMPLETE"]
    skipped: dict[str, int] = defaultdict(int)
    for r in results:
        if r.get("status", "COMPLETE") != "COMPLETE":
            skipped[r["status"]] += 1

    pre_ratios = [r["sa_pre"]["summary"]["supported_ratio"] for r in complete]
    post_ratios = [r["sa_post"]["summary"]["supported_ratio"] for r in complete]
    deltas = [r["delta"]["supported_ratio_delta"] for r in complete]
    pre_unknowns = [r["sa_pre"]["summary"]["unknown"] for r in complete]
    post_unknowns = [r["sa_post"]["summary"]["unknown"] for r in complete]

    n = len(complete) or 1
    avg_pre = sum(pre_ratios) / n
    avg_post = sum(post_ratios) / n
    avg_delta = sum(deltas) / n
    avg_pre_unknown = sum(pre_unknowns) / n
    avg_post_unknown = sum(post_unknowns) / n

    # improved = explicit level change OR fused away (implicit improvement)
    models_improved = sum(
        1 for r in complete if r["delta"]["improved"] or r["delta"].get("fused_away")
    )
    models_regressed = sum(1 for r in complete if r["delta"]["regressed"])
    models_unchanged = len(complete) - models_improved - models_regressed

    # Common improved / fused_away / unresolved patterns
    improved_counter: dict[str, int] = defaultdict(int)
    fused_counter: dict[str, int] = defaultdict(int)
    unresolved_counter: dict[str, int] = defaultdict(int)
    unknown_pre_counter: dict[str, int] = defaultdict(int)
    for r in complete:
        for pid in r["delta"]["improved"]:
            improved_counter[pid] += 1
        for pid in r["delta"].get("fused_away", []):
            fused_counter[pid] += 1
        for pid in r["delta"]["unchanged_partial_unsupported"]:
            unresolved_counter[pid] += 1
        for pid in r["sa_pre"].get("unknown_patterns", []):
            unknown_pre_counter[pid] += 1

    # EPContext accuracy (where available) — track pre and post separately
    epctx_pre = [r for r in complete if r.get("epcontext_diff_pre")]
    epctx_post = [r for r in complete if r.get("epcontext_diff_post")]
    epctx_summary: dict = {
        "models_with_pre_gt": len(epctx_pre),
        "models_with_post_gt": len(epctx_post),
        "avg_accuracy_pre": round(
            statistics.mean(r["epcontext_diff_pre"]["summary"]["accuracy"] for r in epctx_pre), 4
        )
        if epctx_pre
        else None,
        "avg_accuracy_post": round(
            statistics.mean(r["epcontext_diff_post"]["summary"]["accuracy"] for r in epctx_post), 4
        )
        if epctx_post
        else None,
    }

    return {
        "run_date": date.today().isoformat(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(models_input),
        "models_total": len(results),
        "models_complete": len(complete),
        "models_skipped": dict(skipped),
        "pre_optimization": {
            "avg_supported_ratio": round(avg_pre, 4),
            "avg_unknown_count": round(avg_pre_unknown, 2),
            "models_all_supported": sum(1 for r in pre_ratios if r == 1.0),
            "models_with_partial_unsupported": sum(
                1
                for r in complete
                if r["sa_pre"]["summary"]["partial"] + r["sa_pre"]["summary"]["unsupported"] > 0
            ),
        },
        "post_optimization": {
            "avg_supported_ratio": round(avg_post, 4),
            "avg_unknown_count": round(avg_post_unknown, 2),
            "models_all_supported": sum(1 for r in post_ratios if r == 1.0),
            "models_with_partial_unsupported": sum(
                1
                for r in complete
                if r["sa_post"]["summary"]["partial"] + r["sa_post"]["summary"]["unsupported"] > 0
            ),
        },
        "optimizer_effectiveness": {
            "models_improved": models_improved,
            "models_unchanged": models_unchanged,
            "models_regressed": models_regressed,
            "avg_supported_ratio_delta": round(avg_delta, 4),
        },
        "epcontext_ground_truth": epctx_summary,
        "common_improved_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in improved_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "common_fused_away_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in fused_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "unresolved_partial_unsupported_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in unresolved_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "common_unknown_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in unknown_pre_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "results": complete,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run SA accuracy evaluation pipeline for all models in the registry."""
    parser = argparse.ArgumentParser(
        description="SA accuracy evaluation — 4-stage self-contained pipeline"
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=MODELS_FILE,
        help=f"Input model list JSON (default: {MODELS_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: sa_eval_results/{date})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run a single model by HF ID",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip stages whose output artifacts already exist",
    )
    parser.add_argument(
        "--ep",
        default="auto",
        help="Execution provider (default: auto-detects based on available hardware/EPs)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device (default: auto-detects available device, NPU > GPU > CPU)",
    )
    args = parser.parse_args()

    # Resolve "auto" once up-front so all per-model invocations share the
    # same EP/device target (and the resolution is logged once).
    args.ep, args.device = resolve_auto_ep_device(args.ep, args.device)
    safe_print(f"Target: {args.ep} on {args.device}")

    output_dir = args.output_dir or Path(f"sa_eval_results/{date.today().isoformat()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_print(f"Output: {output_dir}")

    if not args.models_file.exists():
        safe_print(f"[ERROR] Models file not found: {args.models_file}")
        sys.exit(1)

    all_models: list[dict] = json.loads(args.models_file.read_text(encoding="utf-8"))

    if args.model:
        models_to_run = [m for m in all_models if m["hf_id"] == args.model]
        if not models_to_run:
            models_to_run = [{"hf_id": args.model, "task": "", "model_type": ""}]
    else:
        models_to_run = all_models

    safe_print(f"Models to evaluate: {len(models_to_run)}")

    t_start = time.monotonic()
    all_results: list[dict] = []

    for i, entry in enumerate(models_to_run, 1):
        safe_print(f"\n[{i}/{len(models_to_run)}]")
        result = evaluate_model(
            entry, output_dir, use_cache=args.use_cache, ep=args.ep, device=args.device
        )
        if result:
            all_results.append(result)

    elapsed = time.monotonic() - t_start
    complete = [r for r in all_results if r.get("status") == "COMPLETE"]

    safe_print(f"\n{'=' * 60}")
    safe_print("SA Eval Summary")
    safe_print(f"{'=' * 60}")
    safe_print(f"Models complete: {len(complete)}")
    safe_print(f"Models skipped:  {len(all_results) - len(complete)}")
    safe_print(f"Total time:      {elapsed:.1f}s")

    if complete:
        avg_pre = sum(r["sa_pre"]["summary"]["supported_ratio"] for r in complete) / len(complete)
        avg_post = sum(r["sa_post"]["summary"]["supported_ratio"] for r in complete) / len(complete)
        safe_print(
            f"Avg supported ratio: {avg_pre:.0%} -> {avg_post:.0%} ({avg_post - avg_pre:+.0%})"
        )

        report = build_aggregate_report(all_results, args.models_file)
        report["total_elapsed"] = round(elapsed, 2)

        report_json = output_dir / "sa_eval_report.json"
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        safe_print(f"\nReport JSON: {report_json}")

        report_html = output_dir / "sa_eval_report.html"
        generate_sa_html_report(report, report_html)
        safe_print(f"Report HTML: {report_html}")
    else:
        safe_print("No models completed successfully.")


if __name__ == "__main__":
    main()
