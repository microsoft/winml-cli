# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SA accuracy evaluation — four-stage self-contained pipeline.

Reads models from model_with_acc.json, runs export + graph optimize,
SA pre-check, capability-driven optimization, SA post-check, and optional
EPContext diff against cached compiled ONNX.

Pipeline per model:
  Stage 1: wmk export + Python optimize_onnx (default)
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


def run_wmk_export(hf_id: str, task: str, output: Path) -> tuple[int, str]:
    """Run wmk export via subprocess. Returns (rc, stderr_tail)."""
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


# Map full EP names to the short form accepted by `wmk perf --ep`
_EP_TO_PERF_ARG: dict[str, str] = {
    "QNNExecutionProvider": "qnn",
    "DmlExecutionProvider": "dml",
    "CPUExecutionProvider": "cpu",
    "MIGraphXExecutionProvider": "migraphx",
    "OpenVINOExecutionProvider": "openvino",
    "VitisAIExecutionProvider": "vitisai",
    "NvTensorRTRTXExecutionProvider": "nv_tensorrt_rtx",
}


def run_winml_perf(
    label: str,
    onnx_path: Path,
    output_json: Path,
    device: str,
    ep: str | None,
    iterations: int,
    warmup: int,
    use_cache: bool,
) -> dict | None:
    """Run wmk perf on onnx_path. Returns latency_ms dict or None on failure."""
    if use_cache and is_cached(output_json):
        safe_print(f"  [{label}] Perf (cached): {output_json.name}")
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            return data.get("latency_ms")
        except Exception:
            return None

    safe_print(f"  [{label}] Running perf on {onnx_path.name}...")
    cmd = [
        sys.executable,
        "-m",
        "winml.modelkit.cli",
        "perf",
        "-m",
        str(onnx_path),
        "--device",
        device.lower(),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--output",
        str(output_json),
    ]
    ep_arg = _EP_TO_PERF_ARG.get(ep, ep.lower() if ep else None) if ep else None
    if ep_arg:
        cmd += ["--ep", ep_arg]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0 or not is_cached(output_json):
        safe_print(f"  [{label}] Perf failed (rc={result.returncode})")
        return None

    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
        latency = data.get("latency_ms", {})
        mean_ms = latency.get("mean", 0)
        p90_ms = latency.get("p90", 0)
        safe_print(f"  [{label}] mean={mean_ms:.2f}ms  p90={p90_ms:.2f}ms")
        return latency
    except Exception as e:
        safe_print(f"  [{label}] Could not parse perf result: {e}")
        return None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_onnx_artifacts(model_dir: Path) -> None:
    """Delete intermediate ONNX files after eval, keeping only JSON/log results.

    Removes all ``*.onnx`` and ``*.onnx.data`` files (exported, graph_optimized,
    sa_optimized, quantized, compiled EPContext). JSON result files and perf
    logs are preserved so --report-only and --use-cache still work for the
    JSON-driven stages.
    """
    freed = 0
    for pattern in ("*.onnx", "*.onnx.data"):
        for f in model_dir.glob(pattern):
            size = f.stat().st_size
            f.unlink()
            freed += size
    safe_print(f"  [cleanup] Freed {freed / 1024**2:.1f} MB of ONNX artifacts")


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
        rc, stderr = run_wmk_export(hf_id, task, exported_path)
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
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
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
        safe_print(
            "  [WARN] SA pre-check: no classifications (no QNN rule data on this machine). "
            "SA-driven optimization will be skipped; pipeline continues."
        )
    else:
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
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
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
        safe_print("  [WARN] SA post-check: no classifications (no QNN rule data). Continuing.")
    else:
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
    """Run wmk compile --device <device> --no-quantize. Returns (rc, stderr_tail)."""
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


def stage6_quantize(
    model_dir: Path,
    sa_opt_path: Path,
    hf_id: str,
    task: str,
    use_cache: bool,
    precision: str = "int8",
    samples: int = 10,
) -> Path | None:
    """Stage 6: QDQ-quantize sa_optimized.onnx → quantized.onnx.

    Runs ``wmk quantize`` on the SA-optimized model. Skips if the output
    already exists and use_cache is True.

    Returns the path to the quantized ONNX on success, None on failure.
    """
    quantized_path = model_dir / "quantized.onnx"

    if use_cache and is_cached(quantized_path):
        safe_print(f"  [Stage 6] Quantize (cached): {quantized_path.name}")
        return quantized_path

    safe_print(f"  [Stage 6] Quantizing {sa_opt_path.name} → {quantized_path.name}...")
    cmd = [
        sys.executable,
        "-m",
        "winml.modelkit.cli",
        "quantize",
        "-m",
        str(sa_opt_path),
        "-o",
        str(quantized_path),
        "--precision",
        precision,
        "--samples",
        str(samples),
    ]
    if task:
        cmd += ["--task", task]
    if hf_id:
        cmd += ["--model-name", hf_id]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0 or not is_cached(quantized_path):
        safe_print(
            f"  [Stage 6] Quantize failed (rc={result.returncode}): {(result.stderr or '').strip()[-300:]}"
        )
        return None

    safe_print(f"  [Stage 6] Quantized: {quantized_path.name}")
    return quantized_path


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
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
    run_perf: bool = True,
    perf_iterations: int = 30,
    perf_warmup: int = 5,
    run_quantize: bool = True,
    quantize_precision: str = "int8",
    quantize_samples: int = 10,
    cleanup: bool = False,
) -> dict | None:
    """Run the 4+1+1 stage SA eval pipeline for a single model."""
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

    # Perf after export and after graph optimize
    perf_exported: dict | None = None
    perf_graph_opt: dict | None = None
    perf_sa_opt: dict | None = None

    if run_perf:
        exported_path = model_dir / "exported.onnx"
        if is_cached(exported_path):
            perf_exported = run_winml_perf(
                "Perf (exported)",
                exported_path,
                model_dir / "exported_perf.json",
                device=device,
                ep=ep,
                iterations=perf_iterations,
                warmup=perf_warmup,
                use_cache=use_cache,
            )
        perf_graph_opt = run_winml_perf(
            "Perf (graph_opt)",
            graph_opt_path,
            model_dir / "graph_optimized_perf.json",
            device=device,
            ep=ep,
            iterations=perf_iterations,
            warmup=perf_warmup,
            use_cache=use_cache,
        )

    # Stage 2
    pre_result = stage2_sa_pre(model_dir, graph_opt_path, use_cache, ep=ep, device=device)
    if pre_result is None:
        return _skip_result(hf_id, task, model_type, "SKIP_SA_PRE", model_dir)
    sa_pre, optim_config, pre_info_items = pre_result

    # Stage 3
    sa_opt_path = stage3_capability_optimize(model_dir, graph_opt_path, optim_config, use_cache)
    if sa_opt_path is None:
        return _skip_result(hf_id, task, model_type, "SKIP_OPTIM", model_dir)

    # Perf after SA capability optimization
    if run_perf:
        perf_sa_opt = run_winml_perf(
            "Perf (sa_opt)",
            sa_opt_path,
            model_dir / "sa_optimized_perf.json",
            device=device,
            ep=ep,
            iterations=perf_iterations,
            warmup=perf_warmup,
            use_cache=use_cache,
        )

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

    # Stage 6: QDQ quantize
    quantized_path: Path | None = None
    perf_quantized: dict | None = None
    if run_quantize:
        quantized_path = stage6_quantize(
            model_dir,
            sa_opt_path,
            hf_id,
            task,
            use_cache,
            precision=quantize_precision,
            samples=quantize_samples,
        )
        if run_perf and quantized_path is not None:
            perf_quantized = run_winml_perf(
                "Perf (quantized)",
                quantized_path,
                model_dir / "quantized_perf.json",
                device=device,
                ep=ep,
                iterations=perf_iterations,
                warmup=perf_warmup,
                use_cache=use_cache,
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

    if run_perf:

        def _fmt(p: dict | None) -> str:
            return f"{p['mean']:.2f}ms" if p else "N/A"

        safe_print(
            f"  Perf (mean): exported={_fmt(perf_exported)} "
            f"→ normalize={_fmt(perf_graph_opt)} "
            f"→ sa_opt={_fmt(perf_sa_opt)} "
            f"→ quantize={_fmt(perf_quantized)}"
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
            **({"quantized_onnx": str(quantized_path)} if quantized_path else {}),
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

    if run_perf:
        result["perf"] = {
            "exported": perf_exported,
            "graph_optimized": perf_graph_opt,
            "sa_optimized": perf_sa_opt,
            "quantized": perf_quantized,
        }

    out_file = model_dir / "sa_eval_result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"  Written: {out_file}")

    if cleanup:
        cleanup_onnx_artifacts(model_dir)

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
        default="QNNExecutionProvider",
        help="Execution provider (default: QNNExecutionProvider)",
    )
    parser.add_argument("--device", default="NPU", help="Target device (default: NPU)")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip winml perf benchmarks after each stage",
    )
    parser.add_argument(
        "--perf-iterations",
        type=int,
        default=30,
        help="Number of perf iterations per stage (default: 30)",
    )
    parser.add_argument(
        "--perf-warmup",
        type=int,
        default=5,
        help="Number of perf warmup iterations per stage (default: 5)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip QDQ quantize step (stage 6)",
    )
    parser.add_argument(
        "--quantize-precision",
        default="int8",
        help="Quantization precision (default: int8)",
    )
    parser.add_argument(
        "--quantize-samples",
        type=int,
        default=10,
        help="Number of calibration samples for quantize (default: 10)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate ONNX files after each model completes to free disk space. "
        "JSON result and perf files are preserved.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Skip all eval stages — collect existing sa_eval_result.json files from "
            "models/ subdirectories and regenerate the HTML report only."
        ),
    )
    args = parser.parse_args()

    output_dir = (args.output_dir or Path(f"sa_eval_results/{date.today().isoformat()}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_print(f"Output: {output_dir}")

    # --report-only: collect existing per-model JSONs and regenerate the report
    if args.report_only:
        models_dir = output_dir / "models"
        result_files = sorted(models_dir.glob("*/sa_eval_result.json"))
        if not result_files:
            safe_print(f"[ERROR] No sa_eval_result.json files found under {models_dir}")
            sys.exit(1)
        all_results = [json.loads(f.read_text(encoding="utf-8")) for f in result_files]
        safe_print(f"Collected {len(all_results)} model results from disk.")
        report = build_aggregate_report(all_results, args.models_file)
        report_json = output_dir / "sa_eval_report.json"
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        safe_print(f"Report JSON: {report_json}")
        report_html = output_dir / "sa_eval_report.html"
        generate_sa_html_report(report, report_html)
        safe_print(f"Report HTML: {report_html}")
        return

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
            entry,
            output_dir,
            use_cache=args.use_cache,
            ep=args.ep,
            device=args.device,
            run_perf=not args.no_perf,
            perf_iterations=args.perf_iterations,
            perf_warmup=args.perf_warmup,
            run_quantize=not args.no_quantize,
            quantize_precision=args.quantize_precision,
            quantize_samples=args.quantize_samples,
            cleanup=args.cleanup,
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
