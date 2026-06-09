# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML CLI component analysis evaluation pipeline.

Reads models from models_with_acc.json, runs winml build to produce all
pipeline artifacts, then runs SA pre/post analysis on the build outputs.

Pipeline per model:
  Stage 1: winml config        → build_config.json
  Stage 2: winml build         → export.onnx, optimized.onnx, quantized.onnx,
                                  compiled.onnx, winml_build_config.json
  Stage 3: winml optimize      → graph_optimized.onnx  (baseline, no SA flags)
  Stage 4: SA pre-check        → winml analyze on graph_optimized.onnx → sa_pre.json
  Stage 5: winml compile (pre) → compiled_pre.onnx  (baseline EPContext)
  Stage 6: EPCtx pre diff      → compare_sa_vs_epcontext(sa_pre, compiled_pre.onnx)
  Stage 7: SA post-check       → winml analyze on optimized.onnx → sa_post.json
  Stage 8: EPCtx post diff     → compare_sa_vs_epcontext(sa_post, compiled.onnx)
  Perf: run winml perf on export.onnx, optimized.onnx, quantized.onnx, compiled.onnx

Usage:
    uv run python scripts/e2e_eval/run_sa_eval.py
    uv run python scripts/e2e_eval/run_sa_eval.py --model microsoft/resnet-50
    uv run python scripts/e2e_eval/run_sa_eval.py --output-dir sa_eval_results/2026-03-27
    uv run python scripts/e2e_eval/run_sa_eval.py --use-cache
    uv run python scripts/e2e_eval/run_sa_eval.py --report-only --output-dir sa_eval_results/2026-05-12
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


def _count_onnx_nodes(path: Path) -> int | None:
    """Return the number of graph nodes in an ONNX model, or None if unavailable."""
    if not is_cached(path):
        return None
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        return len(model.graph.node)
    except Exception:
        return None


def _run_cli(args: list[str]) -> tuple[int, str]:
    """Run a winml CLI command via subprocess. Returns (rc, combined_output)."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "winml.modelkit.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    return result.returncode, combined


# Map full EP names to the short form accepted by `winml perf --ep`
_EP_TO_PERF_ARG: dict[str, str] = {
    "QNNExecutionProvider": "qnn",
    "DmlExecutionProvider": "dml",
    "CPUExecutionProvider": "cpu",
    "MIGraphXExecutionProvider": "migraphx",
    "OpenVINOExecutionProvider": "openvino",
    "VitisAIExecutionProvider": "vitisai",
    "NvTensorRTRTXExecutionProvider": "nv_tensorrt_rtx",
}


def _resolve_ep_arg(ep: str) -> str:
    """Resolve full ORT EP name to the short form accepted by winml CLI --ep."""
    try:
        return _EP_TO_PERF_ARG[ep]
    except KeyError:
        raise ValueError(f"Unknown EP {ep!r}; add it to _EP_TO_PERF_ARG.") from None


def _parse_info_items(json_path: Path, ep: str = "QNNExecutionProvider") -> list[dict]:
    """Parse information items from winml analyze JSON output.

    Returns list of {pattern_id, explanation, has_actions} for each info item
    in the EP result, or [] if none found.
    """
    if not json_path.exists():
        return []
    try:
        sa_data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    for ep_result in sa_data.get("results", []):
        if ep_result.get("ep_type") != ep:
            continue
        return [
            {
                "pattern_id": item.get("pattern_id", ""),
                "explanation": item.get("explanation", ""),
                "has_actions": bool(item.get("actions")),
            }
            for item in ep_result.get("information", [])
        ]
    return []


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
    """Run winml perf on onnx_path. Returns latency_ms dict or None on failure."""
    if use_cache and is_cached(output_json):
        safe_print(f"  [{label}] Perf (cached): {output_json.name}")
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            return data.get("latency_ms")
        except Exception:
            return None

    if not is_cached(onnx_path):
        safe_print(f"  [{label}] Skipping perf — {onnx_path.name} not found")
        return None

    safe_print(f"  [{label}] Running perf on {onnx_path.name}...")
    cmd = [
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
    ep_arg = _resolve_ep_arg(ep) if ep else None
    if ep_arg:
        cmd += ["--ep", ep_arg]

    rc, _ = _run_cli(cmd)
    if rc != 0 or not is_cached(output_json):
        safe_print(f"  [{label}] Perf failed (rc={rc})")
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
    """Delete intermediate ONNX and binary artifacts after eval.

    Removes all non-JSON files including ``*.onnx``, ``*.onnx.data``,
    ``*.bin``, and extensionless ONNX external data files (e.g. weight
    tensors like ``roberta.embeddings.word_embeddings.weight``).
    JSON result files and perf logs are preserved so --report-only and
    --use-cache still work for the JSON-driven stages.
    """
    freed = 0
    for f in model_dir.rglob("*"):
        if f.is_file() and f.suffix != ".json":
            try:
                size = f.stat().st_size
                f.unlink()
                freed += size
            except OSError:
                pass  # File may be locked by a subprocess; skip it
    safe_print(f"  [cleanup] Freed {freed / 1024**2:.1f} MB of artifacts")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_build(
    hf_id: str,
    task: str,
    model_dir: Path,
    use_cache: bool,
    precision: str = "int8",
    device: str = "npu",
    ep: str | None = None,
    run_compile: bool = True,
    run_quantize: bool = True,
) -> str | None:
    """Generate build config and run winml build.

    Runs ``winml config`` then ``winml build`` to produce all pipeline
    artifacts: export.onnx, optimized.onnx, quantized.onnx, compiled.onnx,
    and winml_build_config.json.

    Returns None on success, or a skip reason string on failure.
    """
    config_path = model_dir / "build_config.json"
    export_path = model_dir / "export.onnx"
    optimized_path = model_dir / "optimized.onnx"

    # Skip entire build if key artifacts already exist and cache is enabled
    if use_cache and is_cached(export_path) and is_cached(optimized_path):
        # Also check quantized/compiled if those stages are requested
        if run_quantize and not is_cached(model_dir / "quantized.onnx"):
            safe_print("  [Build] Cache incomplete (missing quantized.onnx), rebuilding...")
        elif run_compile and not is_cached(model_dir / "compiled.onnx"):
            safe_print("  [Build] Cache incomplete (missing compiled.onnx), rebuilding...")
        else:
            safe_print("  [Build] Using cached artifacts (export.onnx, optimized.onnx)")
            return None

    # Stage 1: Generate config
    safe_print(f"  [Build] Generating config for {hf_id}...")
    config_args = [
        "config",
        "-m",
        hf_id,
        "-d",
        device,
        "-p",
        precision,
        "-o",
        str(config_path),
    ]
    if task:
        config_args += ["-t", task]
    if run_compile:
        config_args += ["--compile"]
    if not run_quantize:
        config_args += ["--no-quant"]

    rc, output = _run_cli(config_args)
    if rc != 0 or not is_cached(config_path):
        safe_print(f"  [ERROR] Config generation failed (rc={rc}): {output[-300:]}")
        return "SKIP_EXPORT"
    safe_print(f"  [Build] Config written: {config_path.name}")

    # Stage 2: Run winml build
    safe_print("  [Build] Running winml build...")
    build_args = [
        "build",
        "-c",
        str(config_path),
        "-m",
        hf_id,
        "-o",
        str(model_dir),
        "--ep",
        _resolve_ep_arg(ep) if ep else "qnn",
    ]
    if not use_cache:
        build_args += ["--rebuild"]
    if not run_compile:
        build_args += ["--no-compile"]
    if not run_quantize:
        build_args += ["--no-quant"]

    rc, output = _run_cli(build_args)

    # Check for required artifacts regardless of rc (partial builds are useful)
    if not is_cached(export_path):
        safe_print(f"  [ERROR] Build failed — no export.onnx produced: {output[-300:]}")
        return "SKIP_EXPORT"

    if not is_cached(optimized_path):
        safe_print(f"  [ERROR] Build failed — no optimized.onnx produced: {output[-300:]}")
        return "SKIP_OPTIM"

    produced = [
        p.name
        for p in model_dir.iterdir()
        if p.suffix in (".onnx", ".json") and not p.name.startswith("sa_")
    ]
    safe_print(f"  [Build] Complete: {', '.join(sorted(produced))}")
    return None


def read_optim_flags(model_dir: Path) -> dict:
    """Read optim flags written by winml build into winml_build_config.json."""
    config_path = model_dir / "winml_build_config.json"
    if not config_path.exists():
        return {}
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        return cfg.get("optim", {})
    except Exception:
        return {}


def stage_graph_optimize(model_dir: Path, use_cache: bool) -> bool:
    """Run winml optimize on export.onnx to produce baseline graph_optimized.onnx.

    This applies default graph optimizations WITHOUT any SA-specific flags
    (gelu_fusion, matmul_add_fusion, etc.), giving a pre-SA baseline that
    can be analyzed and compiled for PRE EPCTX comparison.

    Returns True on success, False on failure.
    """
    export_path = model_dir / "export.onnx"
    graph_opt_path = model_dir / "graph_optimized.onnx"

    if use_cache and is_cached(graph_opt_path):
        safe_print("  [Graph Opt] Cached: graph_optimized.onnx")
        return True

    if not is_cached(export_path):
        safe_print("  [ERROR] Graph Opt: export.onnx not found")
        return False

    safe_print("  [Graph Opt] Optimizing export.onnx (baseline, no SA flags)...")
    rc, output = _run_cli(["optimize", "-m", str(export_path), "-o", str(graph_opt_path)])

    if rc != 0 or not is_cached(graph_opt_path):
        safe_print(f"  [ERROR] Graph Opt failed (rc={rc}): {output[-300:]}")
        return False

    safe_print(f"  [Graph Opt] Done: {graph_opt_path.name}")
    return True


def stage_compile_pre(
    model_dir: Path,
    use_cache: bool,
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
) -> bool:
    """Compile graph_optimized.onnx to EPContext for PRE EPCTX comparison.

    Returns True on success (compiled_pre.onnx produced), False otherwise.
    """
    graph_opt_path = model_dir / "graph_optimized.onnx"
    compiled_pre_path = model_dir / "compiled_pre.onnx"

    if use_cache and is_cached(compiled_pre_path):
        safe_print("  [Compile Pre] Cached: compiled_pre.onnx")
        return True

    if not is_cached(graph_opt_path):
        safe_print("  [Compile Pre] No graph_optimized.onnx — skipping")
        return False

    safe_print("  [Compile Pre] Compiling graph_optimized.onnx → EPContext...")
    ep_arg = _resolve_ep_arg(ep)
    rc, _out = _run_cli(
        [
            "compile",
            "-m",
            str(graph_opt_path),
            "-o",
            str(compiled_pre_path),
            "--ep",
            ep_arg,
            "--device",
            device.lower(),
        ]
    )

    if rc != 0 or not is_cached(compiled_pre_path):
        safe_print(f"  [Compile Pre] Compile failed (rc={rc}) — skipping pre diff")
        return False

    safe_print(f"  [Compile Pre] Done: {compiled_pre_path.name}")
    return True


def stage_compile_post(
    model_dir: Path,
    use_cache: bool,
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
) -> bool:
    """Compile optimized.onnx to EPContext for POST EPCTX comparison.

    Used when winml build does not produce compiled.onnx (e.g. non-NPU
    devices where quantize is skipped).

    Returns True on success (compiled.onnx produced), False otherwise.
    """
    optimized_path = model_dir / "optimized.onnx"
    compiled_path = model_dir / "compiled.onnx"

    if use_cache and is_cached(compiled_path):
        safe_print("  [Compile Post] Cached: compiled.onnx")
        return True

    if not is_cached(optimized_path):
        safe_print("  [Compile Post] No optimized.onnx — skipping")
        return False

    safe_print("  [Compile Post] Compiling optimized.onnx → EPContext...")
    ep_arg = _resolve_ep_arg(ep)
    rc, _out = _run_cli(
        [
            "compile",
            "-m",
            str(optimized_path),
            "-o",
            str(compiled_path),
            "--ep",
            ep_arg,
            "--device",
            device.lower(),
        ]
    )

    if rc != 0 or not is_cached(compiled_path):
        safe_print(f"  [Compile Post] Compile failed (rc={rc}) — skipping post diff")
        return False

    safe_print(f"  [Compile Post] Done: {compiled_path.name}")
    return True


def stage_sa_pre(
    model_dir: Path,
    use_cache: bool,
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
) -> tuple[dict[str, str], list[dict]] | None:
    """Run SA with information on graph_optimized.onnx (pre-optimization state).

    Returns (classifications, info_items) or None on failure.
    """
    graph_opt_path = model_dir / "graph_optimized.onnx"
    sa_pre_path = model_dir / "sa_pre.json"

    if use_cache and is_cached(sa_pre_path):
        safe_print("  [SA Pre] Cached")
        classifications = parse_sa_json(sa_pre_path, ep=ep)
        info_items = _parse_info_items(sa_pre_path, ep=ep)
        return classifications, info_items

    if not is_cached(graph_opt_path):
        safe_print("  [ERROR] SA pre: graph_optimized.onnx not found")
        return None

    safe_print("  [SA Pre] Analyzing graph_optimized.onnx...")
    ep_arg = _resolve_ep_arg(ep)
    rc, output = _run_cli(
        [
            "analyze",
            "-m",
            str(graph_opt_path),
            "--ep",
            ep_arg,
            "--device",
            device.lower(),
            "--output",
            str(sa_pre_path),
            "--information",
        ]
    )
    # rc=0: fully supported, rc=1: partial (both OK for us), rc=2: error
    if rc == 2 or not is_cached(sa_pre_path):
        safe_print(f"  [ERROR] SA pre failed (rc={rc}): {output[-300:]}")
        return None

    classifications = parse_sa_json(sa_pre_path, ep=ep)
    info_items = _parse_info_items(sa_pre_path, ep=ep)

    if not classifications:
        safe_print(
            "  [WARN] SA pre: no classifications (no QNN rule data on this machine). "
            "Pipeline continues without SA-driven optimization."
        )
    else:
        summary = get_sa_summary(classifications)
        safe_print(
            f"  [SA Pre] SUPPORTED={summary['supported']} PARTIAL={summary['partial']} "
            f"UNSUPPORTED={summary['unsupported']} UNKNOWN={summary['unknown']} "
            f"({summary['supported_ratio']:.0%} supported)"
        )
    return classifications, info_items


def stage_sa_post(
    model_dir: Path,
    use_cache: bool,
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
) -> tuple[dict[str, str], list[dict]] | None:
    """Run SA with information on optimized.onnx (post-optimization state).

    Returns (classifications, info_items) or None on failure.
    """
    optimized_path = model_dir / "optimized.onnx"
    sa_post_path = model_dir / "sa_post.json"

    if use_cache and is_cached(sa_post_path):
        safe_print("  [SA Post] Cached")
        classifications = parse_sa_json(sa_post_path, ep=ep)
        info_items = _parse_info_items(sa_post_path, ep=ep)
        return classifications, info_items

    if not is_cached(optimized_path):
        safe_print("  [ERROR] SA post: optimized.onnx not found")
        return None

    safe_print("  [SA Post] Analyzing optimized.onnx...")
    ep_arg = _resolve_ep_arg(ep)
    rc, output = _run_cli(
        [
            "analyze",
            "-m",
            str(optimized_path),
            "--ep",
            ep_arg,
            "--device",
            device.lower(),
            "--output",
            str(sa_post_path),
            "--information",
        ]
    )
    # rc=0: fully supported, rc=1: partial (both OK for us), rc=2: error
    if rc == 2 or not is_cached(sa_post_path):
        safe_print(f"  [ERROR] SA post failed (rc={rc}): {output[-300:]}")
        return None

    classifications = parse_sa_json(sa_post_path, ep=ep)
    info_items = _parse_info_items(sa_post_path, ep=ep)

    if not classifications:
        safe_print("  [WARN] SA post: no classifications. Continuing.")
    else:
        summary = get_sa_summary(classifications)
        safe_print(
            f"  [SA Post] SUPPORTED={summary['supported']} PARTIAL={summary['partial']} "
            f"UNSUPPORTED={summary['unsupported']} UNKNOWN={summary['unknown']} "
            f"({summary['supported_ratio']:.0%} supported)"
        )
    return classifications, info_items


def stage_epctx_diff(
    sa_predictions: dict[str, str],
    compiled_path: Path,
    label: str = "EPCtx",
) -> dict | None:
    """EPContext diff: compare SA predictions vs a compiled EPContext ONNX.

    Used for both PRE (sa_pre vs compiled_pre.onnx) and POST (sa_post vs
    compiled.onnx) comparisons to measure SA classifier accuracy.

    Returns comparison dict or None if the compiled file is unavailable.
    """
    if not is_cached(compiled_path):
        safe_print(f"  [{label}] No {compiled_path.name} — skipping diff")
        return None

    try:
        result = compare_sa_vs_epcontext(sa_predictions, compiled_path)
        s = result["summary"]
        safe_print(
            f"  [{label}] TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']} "
            f"accuracy={s['accuracy']:.0%}"
        )
        ufa = s.get("unsupported_false_alarms", [])
        pfa = s.get("partial_false_alarms", [])
        if ufa:
            safe_print(f"  [{label}] UNSUPPORTED false alarms: {ufa}")
        if pfa:
            safe_print(f"  [{label}] PARTIAL false alarms: {pfa}")
        return result
    except Exception as e:
        safe_print(f"  [{label}] Diff failed: {e}")
        return None


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
    run_compile: bool = True,
    cleanup: bool = False,
) -> dict | None:
    """Run the winml build + SA analysis pipeline for a single model."""
    hf_id = model_entry["hf_id"]
    task = model_entry.get("task", "")
    model_type = model_entry.get("model_type", "")

    # Non-NPU devices: skip quantize entirely
    if device.upper() != "NPU":
        run_quantize = False

    safe_print(f"\n{'=' * 60}")
    safe_print(f"[sa_eval] {hf_id} ({task})")
    safe_print(f"{'=' * 60}")

    slug = make_slug(hf_id, task)
    model_dir = output_dir / "models" / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Stage 1+2: Generate config + winml build
    skip_reason = stage_build(
        hf_id,
        task,
        model_dir,
        use_cache,
        precision=quantize_precision,
        device=device.lower(),
        ep=ep,
        run_compile=run_compile,
        run_quantize=run_quantize,
    )
    if skip_reason:
        return _skip_result(hf_id, task, model_type, skip_reason, model_dir)

    export_path = model_dir / "export.onnx"
    optimized_path = model_dir / "optimized.onnx"
    quantized_path = model_dir / "quantized.onnx"

    compiled_path = model_dir / "compiled.onnx"

    # Perf: export and optimized
    perf_exported: dict | None = None
    perf_sa_opt: dict | None = None
    perf_quantized: dict | None = None
    perf_compiled: dict | None = None

    if run_perf:
        perf_exported = run_winml_perf(
            "Perf (export)",
            export_path,
            model_dir / "exported_perf.json",
            device=device,
            ep=ep,
            iterations=perf_iterations,
            warmup=perf_warmup,
            use_cache=use_cache,
        )
        perf_sa_opt = run_winml_perf(
            "Perf (optimized)",
            optimized_path,
            model_dir / "optimized_perf.json",
            device=device,
            ep=ep,
            iterations=perf_iterations,
            warmup=perf_warmup,
            use_cache=use_cache,
        )

    # Stage 3: Baseline graph optimization (no SA flags) → graph_optimized.onnx
    if not stage_graph_optimize(model_dir, use_cache):
        return _skip_result(hf_id, task, model_type, "SKIP_GRAPH_OPT", model_dir)

    # Stage 4: SA pre-check (on graph_optimized.onnx — baseline, before SA optimization)
    pre_result = stage_sa_pre(model_dir, use_cache, ep=ep, device=device)
    if pre_result is None:
        return _skip_result(hf_id, task, model_type, "SKIP_SA_PRE", model_dir)
    sa_pre, pre_info_items = pre_result

    # Stage 5: Compile baseline model for PRE EPCTX diff
    compiled_pre_path = model_dir / "compiled_pre.onnx"
    if run_compile:
        stage_compile_pre(model_dir, use_cache, ep=ep, device=device)

    # Stage 6: EPContext diff PRE (sa_pre vs compiled_pre.onnx)
    epctx_pre_result = stage_epctx_diff(sa_pre, compiled_pre_path, label="EPCtx Pre")

    # Stage 7: SA post-check (on optimized.onnx — after SA optimization)
    post_result = stage_sa_post(model_dir, use_cache, ep=ep, device=device)
    if post_result is None:
        return _skip_result(hf_id, task, model_type, "SKIP_SA_POST", model_dir)
    sa_post, post_info_items = post_result

    # Read optimization flags from winml_build_config.json
    optim_config = read_optim_flags(model_dir)
    if optim_config:
        safe_print(f"  [Optim flags] {list(optim_config.keys())}")

    # Stage 8: EPContext diff POST (sa_post vs compiled.onnx)
    # If build didn't produce compiled.onnx (e.g. non-NPU, quantize skipped),
    # compile optimized.onnx directly for post EPCtx comparison.
    if run_compile and not is_cached(compiled_path):
        stage_compile_post(model_dir, use_cache, ep=ep, device=device)
    epctx_post_result = stage_epctx_diff(sa_post, compiled_path, label="EPCtx Post")

    # Perf: quantized and compiled
    if run_perf and run_quantize:
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
    if run_perf and run_compile:
        perf_compiled = run_winml_perf(
            "Perf (compiled)",
            compiled_path,
            model_dir / "compiled_perf.json",
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
            f"  Perf (mean): export={_fmt(perf_exported)} "
            f"→ optimized={_fmt(perf_sa_opt)} "
            f"→ quantized={_fmt(perf_quantized)} "
            f"→ compiled={_fmt(perf_compiled)}"
        )

    # Collect node counts for each stage
    graph_opt_path = model_dir / "graph_optimized.onnx"
    compiled_pre_path = model_dir / "compiled_pre.onnx"
    node_counts: dict = {
        "export": _count_onnx_nodes(export_path),
        "graph_optimized": _count_onnx_nodes(graph_opt_path),
        "optimized": _count_onnx_nodes(optimized_path),
        "quantized": _count_onnx_nodes(quantized_path),
        "compiled_pre": _count_onnx_nodes(compiled_pre_path),
        "compiled": _count_onnx_nodes(compiled_path),
    }
    node_parts = [f"{k}={v}" for k, v in node_counts.items() if v is not None]
    safe_print(f"  Nodes: {', '.join(node_parts)}")

    artifacts: dict = {
        "exported_onnx": str(export_path),
        "optimized_onnx": str(optimized_path),
    }
    if is_cached(quantized_path):
        artifacts["quantized_onnx"] = str(quantized_path)

    result: dict = {
        "model": hf_id,
        "task": task,
        "model_type": model_type,
        "status": "COMPLETE",
        "elapsed": round(elapsed, 2),
        "artifacts": artifacts,
        "node_counts": node_counts,
        "sa_pre": {
            "source_onnx": "graph_optimized.onnx",
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
            "source_onnx": optimized_path.name,
            "classifications": sa_post,
            "summary": get_sa_summary(sa_post),
            "partial_patterns": get_level_patterns(sa_post, "PARTIAL"),
            "unsupported_patterns": get_level_patterns(sa_post, "UNSUPPORTED"),
            "unknown_patterns": get_level_patterns(sa_post, "UNKNOWN"),
            "info_items": post_info_items,
        },
        "delta": delta,
        # perf keys match sa_report.py expectations:
        #   "exported"    → Export (ms) column
        #   "sa_optimized"→ Optimized (ms) column  [perf of optimized.onnx]
        #   "quantized"   → Quantize (ms) column
        #   "compiled"    → Compiled (ms) column   [perf of compiled.onnx / EPContext]
        "perf": {
            "exported": perf_exported,
            "sa_optimized": perf_sa_opt,
            "quantized": perf_quantized,
            "compiled": perf_compiled,
        },
    }

    if epctx_pre_result:
        result["epcontext_diff_pre"] = epctx_pre_result
    if epctx_post_result:
        result["epcontext_diff_post"] = epctx_post_result

    out_file = model_dir / "sa_eval_result.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    safe_print(f"  Written: {out_file}")

    if cleanup:
        cleanup_onnx_artifacts(model_dir)

    return result


def _should_skip_existing(existing: dict, retry_types: set[str] | None) -> bool:
    """Return True if an existing sa_eval_result should be skipped (not re-run).

    *retry_types* semantics (mirrors run_eval.py):
      - ``None``  → ``--continue`` only: skip every existing result.
      - ``set()`` → ``--retry-failed`` with no args: retry ALL non-COMPLETE.
      - ``{"SKIP_BUILD", ...}`` → retry only matching statuses.
    """
    if retry_types is None:
        return True  # --continue without --retry-failed: skip all existing

    status = existing.get("status", "COMPLETE")
    if status == "COMPLETE":
        return True  # completed models are always kept

    # Non-COMPLETE → check whether this status matches the retry filter
    return not (not retry_types or status in retry_types)


def _skip_result(hf_id: str, task: str, model_type: str, status: str, model_dir: Path) -> dict:
    result = {"model": hf_id, "task": task, "model_type": model_type, "status": status}
    (model_dir / "sa_eval_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def build_aggregate_report(
    results: list[dict],
    models_input: Path,
    ep: str = "",
    device: str = "",
) -> dict:
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

    # SA false alarms (post EPCtx: UNSUPPORTED/PARTIAL predicted but actually on NPU)
    unsup_fa_counter: dict[str, int] = defaultdict(int)
    partial_fa_counter: dict[str, int] = defaultdict(int)
    for r in complete:
        post_summary = r.get("epcontext_diff_post", {}).get("summary", {})
        for pid in post_summary.get("unsupported_false_alarms", []):
            unsup_fa_counter[pid] += 1
        for pid in post_summary.get("partial_false_alarms", []):
            partial_fa_counter[pid] += 1

    # EPContext accuracy — pre (compiled_pre.onnx) and post (compiled.onnx)
    epctx_pre = [r for r in complete if r.get("epcontext_diff_pre")]
    epctx_post = [r for r in complete if r.get("epcontext_diff_post")]
    epctx_summary: dict = {
        "models_with_pre_gt": len(epctx_pre),
        "models_with_post_gt": len(epctx_post),
        "avg_accuracy_pre": round(
            statistics.mean(r["epcontext_diff_pre"]["summary"]["accuracy"] for r in epctx_pre),
            4,
        )
        if epctx_pre
        else None,
        "avg_accuracy_post": round(
            statistics.mean(r["epcontext_diff_post"]["summary"]["accuracy"] for r in epctx_post),
            4,
        )
        if epctx_post
        else None,
    }

    return {
        "run_date": date.today().isoformat(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(models_input),
        "ep": ep,
        "device": device,
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
        "unsupported_false_alarm_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in unsup_fa_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "partial_false_alarm_patterns": sorted(
            [{"pattern": k, "count": v} for k, v in partial_fa_counter.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
        "results": complete,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run WinML component analysis evaluation pipeline."""
    parser = argparse.ArgumentParser(description="WinML CLI component analysis evaluation pipeline")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=MODELS_FILE,
        help=f"Input model list JSON (default: {MODELS_FILE})",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Model registry JSON (same format as run_eval.py --registry). "
        "Overrides --models-file when provided.",
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
        "--priority",
        nargs="+",
        choices=["P0", "P1", "P2", "P3"],
        default=["P0", "P1", "P2"],
        metavar="{P0,P1,P2,P3}",
        help="Filter by priority (default: P0 P1 P2, P3 excluded). Only used with --registry.",
    )
    parser.add_argument("--task", help="Filter by HF task. Only used with --registry.")
    parser.add_argument("--model-type", help="Filter by model_type. Only used with --registry.")
    parser.add_argument("--group", help="Filter by group. Only used with --registry.")
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
        help="Skip QDQ quantize step",
    )
    parser.add_argument(
        "--quantize-precision",
        default="int8",
        help="Quantization precision passed to winml config (default: int8)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip compilation step (no compiled.onnx / EPContext diff)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete ONNX and binary artifacts after each model completes. "
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
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip models that already have sa_eval_result.json",
    )
    parser.add_argument(
        "--retry-failed",
        nargs="*",
        metavar="STATUS",
        help=(
            "Re-run models matching given failure statuses "
            "(e.g. SKIP_BUILD, SKIP_SA_PRE, SKIP_SA_POST, SKIP_GRAPH_OPT). "
            "Use without args to retry ALL non-COMPLETE models. "
            "Implies --continue for completed models."
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
        report = build_aggregate_report(
            all_results, args.registry or args.models_file, ep=args.ep, device=args.device
        )
        report_json = output_dir / "sa_eval_report.json"
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        safe_print(f"Report JSON: {report_json}")
        report_html = output_dir / "sa_eval_report.html"
        generate_sa_html_report(report, report_html)
        safe_print(f"Report HTML: {report_html}")
        return

    # Load model list from --registry (preferred) or --models-file
    if args.registry:
        from utils.registry import filter_registry, load_registry

        if not args.registry.exists():
            safe_print(f"[ERROR] Registry not found: {args.registry}")
            sys.exit(1)
        registry_entries = load_registry(args.registry)
        registry_entries = filter_registry(
            registry_entries,
            task=args.task,
            priority=args.priority,
            model_type=getattr(args, "model_type", None),
            group=args.group,
        )
        all_models = [
            {
                "hf_id": e.hf_id,
                "task": e.task,
                "model_type": e.model_type,
                "group": e.group,
                "priority": e.priority,
            }
            for e in registry_entries
        ]
    else:
        models_file = args.models_file
        if not models_file.exists():
            safe_print(f"[ERROR] Models file not found: {models_file}")
            sys.exit(1)
        all_models = json.loads(models_file.read_text(encoding="utf-8"))

    if args.model:
        models_to_run = [m for m in all_models if m["hf_id"] == args.model]
        if not models_to_run:
            models_to_run = [{"hf_id": args.model, "task": "", "model_type": ""}]
    else:
        models_to_run = all_models

    # --retry-failed implies --continue for passing models
    retry_types: set[str] | None = None
    if args.retry_failed is not None:
        args.continue_run = True
        retry_types = {t.upper() for t in args.retry_failed} if args.retry_failed else set()

    safe_print(f"Models to evaluate: {len(models_to_run)}")
    if retry_types is not None:
        if retry_types:
            safe_print(f"Retry mode: {', '.join(sorted(retry_types))}")
        else:
            safe_print("Retry mode: ALL non-COMPLETE models")
    elif args.continue_run:
        safe_print("Continue mode: skipping models with existing sa_eval_result.json")

    t_start = time.monotonic()
    all_results: list[dict] = []
    cached_count = 0

    for i, entry in enumerate(models_to_run, 1):
        hf_id = entry["hf_id"]
        task = entry.get("task", "")
        slug = make_slug(hf_id, task)
        model_dir = output_dir / "models" / slug
        result_path = model_dir / "sa_eval_result.json"

        # --continue / --retry-failed: check existing sa_eval_result.json
        if args.continue_run and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))

                if _should_skip_existing(existing, retry_types):
                    all_results.append(existing)
                    cached_count += 1
                    status = existing.get("status", "COMPLETE")
                    safe_print(f"\n[{i}/{len(models_to_run)}] {hf_id}  (SKIP - {status}, cached)")
                    continue

                safe_print(
                    f"\n[{i}/{len(models_to_run)}] {hf_id}  "
                    f"(RETRY - was {existing.get('status', '?')})"
                )
            except (json.JSONDecodeError, KeyError):
                pass  # Corrupted result file — re-run

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
            run_compile=not args.no_compile,
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
    if cached_count:
        safe_print(f"Models cached:   {cached_count}")
    safe_print(f"Total time:      {elapsed:.1f}s")

    if complete:
        avg_pre = sum(r["sa_pre"]["summary"]["supported_ratio"] for r in complete) / len(complete)
        avg_post = sum(r["sa_post"]["summary"]["supported_ratio"] for r in complete) / len(complete)
        safe_print(
            f"Avg supported ratio: {avg_pre:.0%} -> {avg_post:.0%} ({avg_post - avg_pre:+.0%})"
        )

        report = build_aggregate_report(
            all_results, args.registry or args.models_file, ep=args.ep, device=args.device
        )
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
