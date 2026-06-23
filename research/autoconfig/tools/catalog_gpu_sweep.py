# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""catalog_gpu_sweep.py — QNN GPU optimization hypothesis sweep for winml catalog models.

QNN GPU differs fundamentally from QNN NPU:
  - NO quantization (gpu-004: QDQ graphs hang on QNN GPU EP)
  - NO compile (gpu-003: EPContext compilation regresses ~34% on GPU)
  - NO nhwc-transformer (gpu-002: Adreno X1-85 does not benefit)
  - CV gating IS reliable on GPU (no DVFS noise unlike NPU)
  - All findings from gpu-001..006 are ConvNext-specific — transformer fusions
    (attention, matmul_add, layer_norm) are UNTESTED and may help

Hypothesis matrix (per model):
  h0: baseline FP32 (auto-config, no quant, no compile)
  h1: opset 17 explicit
  h2: opset 19
  h3: opset 21  ← tests gpu-006 (unknown territory)

  Transformer/attention fusions (graph-analysis-driven):
  h4: opset 17 + matmul_transpose_fusion  (24-36× on transformer optimized.onnx)
  h5: opset 17 + attention_fusion
  h6: opset 17 + bias_softmax_fusion      (12× on BERT-family)
  h7: opset 17 + layer_norm_fusion
  h8: opset 17 + skip_layer_norm_fusion

  Combined bundles:
  h9:  opset 21 + matmul_transpose_fusion + attention_fusion
  h10: opset 17 + layer_norm_fusion + skip_layer_norm_fusion + matmul_transpose_fusion
  h11: opset 17 + gelu_fusion (already in autoconf baseline; test stability benefit — gpu-005)

  Layout (Conv-heavy models only):
  h12: opset 17 + transpose_optimizer

2-phase bench (CV-gated, GPU is stable unlike NPU):
  Phase A: 200-iter screen, CV < 15% required.
  Phase B: 3 sessions × 300 iters, 5s cool-down.
  Phase C (confirmation): KEEP candidates get 2 additional sessions.
    All 5 sessions must show improvement → KEEP_CONFIRMED.
    Fewer than 5/5 → MARGINAL_UNCONFIRMED.
  KEEP criterion: median p50 >= 5% improvement AND CV < 5%.

Results: catalog-gpu-sweep/<model_slug>/results.json
Summary: catalog-gpu-sweep/SUMMARY.md
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Agent package bootstrap: make the autoconfig root importable for sibling packages.
_AGENT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ep_knowledge").is_dir())
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

try:
    from lib.gen_model_report import generate_model_report  # noqa: E402
except Exception:
    generate_model_report = None


sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR = _AGENT_ROOT
WINML = str(BASE_DIR / ".venv" / "Scripts" / "winml.exe")
EP = "qnn"
DEVICE = "gpu"
RESULTS_DIR = BASE_DIR / "catalog-gpu-sweep"
CHAMPION_DIR = BASE_DIR / "champion-configs"

SCREEN_WARMUP = 20
SCREEN_ITERS = 200
SCREEN_CV_MAX = 0.15  # GPU is CV-stable, unlike NPU

FULL_WARMUP = 20
FULL_ITERS = 300
FULL_SESSIONS = 3  # baseline sessions per hypothesis
CONFIRM_SESSIONS = 2  # extra sessions for KEEP candidates (Phase C)
COOL_DOWN_S = 5  # GPU cools faster than NPU HTP

MIN_IMPROVEMENT_PCT = 5.0  # % gain required to declare KEEP

BUILD_TIMEOUT_S = 10 * 60
BENCH_TIMEOUT_S = 5 * 60

# gpu-004: no quantization allowed
# gpu-003: no compile
GPU_NO_QUANT = True
GPU_NO_COMPILE = True

# Hypotheses: (id, label, opset_override, extra_optim)
# extra_optim=None → keep auto-config optim unchanged
# extra_optim=dict → merge ON TOP of auto-config optim
HYPOTHESES = [
    ("h0", "baseline FP32 (no quant, no compile)", None, None),
    ("h1", "opset 17 explicit", 17, None),
    ("h2", "opset 19", 19, None),
    ("h3", "opset 21 (tests gpu-006)", 21, None),
    # ── transformer/attention fusions (graph-analysis-driven) ──────────────
    ("h4", "opset 17 + matmul_transpose_fusion", 17, {"matmul_transpose_fusion": True}),
    ("h5", "opset 17 + attention_fusion", 17, {"attention_fusion": True}),
    ("h6", "opset 17 + bias_softmax_fusion", 17, {"bias_softmax_fusion": True}),
    (
        "h7",
        "opset 17 + layer_norm_fusion",
        17,
        {"layer_norm_fusion": True},
    ),
    (
        "h8",
        "opset 17 + skip_layer_norm_fusion",
        17,
        {"skip_layer_norm_fusion": True},
    ),
    # ── combined bundles ────────────────────────────────────────────────────
    (
        "h9",
        "opset 21 + matmul_transpose + attention_fusion",
        21,
        {"matmul_transpose_fusion": True, "attention_fusion": True},
    ),
    (
        "h10",
        "opset 17 + ln + skip_ln + matmul_transpose",
        17,
        {
            "layer_norm_fusion": True,
            "skip_layer_norm_fusion": True,
            "matmul_transpose_fusion": True,
        },
    ),
    # ── gelu stability (gpu-005) ────────────────────────────────────────────
    # gelu_fusion is already in autoconf defaults, but test explicitly
    # to confirm p90/std stability benefit on non-ConvNext models
    ("h11", "opset 17 + gelu_fusion explicit", 17, {"gelu_fusion": True}),
    # ── layout ──────────────────────────────────────────────────────────────
    ("h12", "opset 17 + transpose_optimizer", 17, {"transpose_optimizer": True}),
]

# Catalog models (same as NPU sweep + recipe models)
ALL_MODELS: list[tuple[str, str, str]] = [
    # Catalog 8
    ("microsoft/resnet-18", "image-classification", "resnet"),
    ("google/vit-base-patch16-224", "image-classification", "vit"),
    ("apple/mobilevit-small", "image-classification", "mobilevit"),
    ("facebook/dinov2-small", "image-feature-extraction", "dinov2"),
    ("hustvl/yolos-small", "object-detection", "yolos"),
    (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "text-classification",
        "distilbert",
    ),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-similarity", "bert"),
    ("deepset/roberta-base-squad2", "question-answering", "roberta"),
    # Recipe models (from winml-cli examples/recipes)
    ("microsoft/rad-dino", "image-feature-extraction", "dinov2"),
    ("deepset/tinyroberta-squad2", "question-answering", "roberta"),
    ("BAAI/bge-small-en-v1.5", "sentence-similarity", "bert"),
]


# ── low-level helpers ─────────────────────────────────────────────────────────


def run_cmd(cmd: list[str], label: str = "", timeout: int = 600) -> tuple[int, str, float]:
    """Run a command; return (returncode, combined_output, elapsed_s)."""
    t0 = time.time()
    print(f"  >> {label or cmd[1]}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.time() - t0
        tag = "ok" if result.returncode == 0 else f"rc={result.returncode}"
        print(f"     {elapsed:.0f}s [{tag}]", flush=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f"     stderr: {stderr[:200]}", flush=True)
        return result.returncode, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"     TIMEOUT ({elapsed:.0f}s)", flush=True)
        return -1, "TIMEOUT", elapsed


def _get_p50(perf_json: Path) -> float | None:
    try:
        d = json.loads(perf_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", d)
        return float(lat.get("p50") or 0) or None
    except Exception:
        return None


def _get_cv(perf_json: Path) -> float | None:
    """Return CV (std/p50). Returns None on parse error."""
    try:
        d = json.loads(perf_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", d)
        p50 = float(lat.get("p50") or 0)
        std = float(lat.get("std") or 0)
        return std / p50 if p50 > 0 else None
    except Exception:
        return None


# ── config helpers ────────────────────────────────────────────────────────────


def _patch_for_gpu(cfg: dict) -> dict:
    """Strip quantization and compile from a base config for GPU EP."""
    cfg = copy.deepcopy(cfg)
    cfg["quant"] = None
    cfg["compile"] = None
    # Remove nhwc-transformer (gpu-002)
    optim = cfg.get("optim") or {}
    optim.pop("nhwc_transformer", None)
    cfg["optim"] = optim
    return cfg


def get_base_config(model_id: str, task: str, model_type: str) -> dict | None:
    """Call winml config for GPU EP and return the parsed config."""
    tmp_dir = RESULTS_DIR / "_tmp_config"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = tmp_dir / f"{model_id.replace('/', '--')}_gpu.json"

    rc, out, _ = run_cmd(
        [
            WINML,
            "config",
            "--model",
            model_id,
            "--task",
            task,
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--model-type",
            model_type,
            "--output",
            str(cfg_out),
        ],
        label="winml config --ep qnn --device gpu",
        timeout=300,
    )
    if rc != 0 or not cfg_out.exists():
        # Try without --output (some versions write to stdout)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    cfg = json.loads(line)
                    return _patch_for_gpu(cfg)
                except Exception:
                    pass
        return None

    cfg = json.loads(cfg_out.read_text(encoding="utf-8"))
    return _patch_for_gpu(cfg)


def make_hypothesis_config(
    base_config: dict, opset_override: int | None, extra_optim: dict | None
) -> dict:
    """Apply opset + extra_optim on top of base config."""
    cfg = copy.deepcopy(base_config)
    if opset_override is not None:
        cfg.setdefault("export", {})["opset_version"] = opset_override
    if extra_optim:
        existing = cfg.get("optim") or {}
        cfg["optim"] = {**existing, **extra_optim}
    return cfg


# ── build + bench ─────────────────────────────────────────────────────────────


def run_build(model_id: str, cfg_path: Path, out_dir: Path) -> tuple[bool, str]:
    """winml build --no-quant --no-compile --rebuild. Returns (ok, output)."""
    rc, out, _ = run_cmd(
        [
            WINML,
            "build",
            "-m",
            model_id,
            "-c",
            str(cfg_path),
            "-o",
            str(out_dir),
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--no-quant",
            "--no-compile",
            "--rebuild",
        ],
        label="winml build",
        timeout=BUILD_TIMEOUT_S,
    )
    return rc == 0, out


def run_perf_screen(onnx_path: Path, out_json: Path) -> tuple[float | None, float | None]:
    """Phase A: 200-iter screen. Returns (p50_ms, cv)."""
    rc, out, _ = run_cmd(
        [
            WINML,
            "perf",
            "-m",
            str(onnx_path),
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--warmup",
            str(SCREEN_WARMUP),
            "--iterations",
            str(SCREEN_ITERS),
            "--output",
            str(out_json),
        ],
        label="perf screen (200 iters)",
        timeout=BENCH_TIMEOUT_S,
    )
    if rc != 0 or not out_json.exists():
        return None, None
    p50 = _get_p50(out_json)
    cv = _get_cv(out_json)
    if p50:
        print(f"     screen: p50={p50:.2f}ms  CV={cv:.3f}", flush=True)
    return p50, cv


def run_perf_full(onnx_path: Path, hyp_dir: Path) -> list[float]:
    """Phase B: 2 × 300-iter sessions. Returns list of p50 values."""
    p50s = []
    for s in range(1, FULL_SESSIONS + 1):
        out_json = hyp_dir / f"full_s{s}.json"
        rc, out, _ = run_cmd(
            [
                WINML,
                "perf",
                "-m",
                str(onnx_path),
                "--ep",
                EP,
                "--device",
                DEVICE,
                "--warmup",
                str(FULL_WARMUP),
                "--iterations",
                str(FULL_ITERS),
                "--output",
                str(out_json),
            ],
            label=f"perf full s{s}/{FULL_SESSIONS} ({FULL_ITERS} iters)",
            timeout=BENCH_TIMEOUT_S,
        )
        p50 = _get_p50(out_json) if rc == 0 and out_json.exists() else None
        if p50:
            print(f"     full s{s}: p50={p50:.2f}ms", flush=True)
            p50s.append(p50)
        if s < FULL_SESSIONS:
            print(f"     cool-down {COOL_DOWN_S}s…", flush=True)
            time.sleep(COOL_DOWN_S)
    return p50s


# ── sweep logic ───────────────────────────────────────────────────────────────


def sweep_model(
    model_id: str,
    task: str,
    model_type: str,
    only_hyp_ids: "set[str] | None" = None,
    reuse_h0_config: bool = False,
) -> dict:
    """Run GPU hypotheses for one model. Returns results dict."""
    model_slug = model_id.replace("/", "--")
    model_dir = RESULTS_DIR / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    # Resume from partial run
    results_path = model_dir / "results.json"
    if only_hyp_ids and results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            print("  [resume] loaded existing results", flush=True)
        except Exception:
            results = {}
    else:
        results = {}

    results.update(
        {
            "model_id": model_id,
            "task": task,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ep": EP,
            "device": DEVICE,
        }
    )
    results.setdefault("baseline_opset", None)
    results.setdefault("hypotheses", {})
    results.setdefault("best_hypothesis", None)
    results.setdefault("baseline_p50_ms", None)
    results.setdefault("best_p50_ms", None)
    results.setdefault("best_gain_pct", None)
    results.setdefault("opset21_gain_pct", None)  # tests gpu-006
    results.setdefault("feature_gaps", [])
    results.setdefault("errors", [])

    print(f"\n{'=' * 64}", flush=True)
    print(f"  SWEEP [GPU]: {model_id}  [{task}]", flush=True)
    if only_hyp_ids:
        print(f"  (delta — only: {sorted(only_hyp_ids)})", flush=True)
    print(f"{'=' * 64}", flush=True)

    # ── Step 1: base config ────────────────────────────────────────────────
    print("\n[1/3] Generating base config…", flush=True)
    base_config = None

    if reuse_h0_config:
        h0_cfg = model_dir / "h0" / "build_config.json"
        if h0_cfg.exists():
            try:
                base_config = json.loads(h0_cfg.read_text(encoding="utf-8"))
                print("  [reuse] h0 config loaded", flush=True)
            except Exception:
                pass

    if base_config is None:
        base_config = get_base_config(model_id, task, model_type)

    if base_config is None:
        results["errors"].append("base config generation failed")
        _save_results(results, model_dir)
        _emit_model_artifacts(results, model_dir)
        return results

    baseline_opset = (base_config.get("export") or {}).get("opset_version", "?")
    results["baseline_opset"] = baseline_opset
    print(f"  baseline opset={baseline_opset}  quant=NONE (GPU EP)  compile=NONE", flush=True)

    # ── Step 2: hypothesis loop ────────────────────────────────────────────
    print(f"\n[2/3] Running {len(HYPOTHESES)} hypotheses…", flush=True)

    baseline_p50: float | None = results.get("baseline_p50_ms")

    for hyp_id, label, opset_override, extra_optim in HYPOTHESES:
        if only_hyp_ids is not None and hyp_id not in only_hyp_ids:
            continue

        sep = "─" * 56
        print(f"\n{sep}", flush=True)
        print(f"  {hyp_id}: {label}", flush=True)
        print(f"{sep}", flush=True)

        hyp_config = make_hypothesis_config(base_config, opset_override, extra_optim)
        opset_used = (hyp_config.get("export") or {}).get("opset_version", "?")
        print(f"  opset={opset_used}  extra_optim={extra_optim}", flush=True)

        hyp_dir = model_dir / hyp_id
        hyp_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = hyp_dir / "build_config.json"
        cfg_path.write_text(json.dumps(hyp_config, indent=2), encoding="utf-8")

        # Build
        build_ok, build_out = run_build(model_id, cfg_path, hyp_dir)
        if not build_ok:
            results["hypotheses"][hyp_id] = {
                "status": "BUILD_FAIL",
                "label": label,
                "opset": opset_used,
                "build_error": build_out[-300:] if build_out else "",
            }
            results["errors"].append(f"{hyp_id}: BUILD_FAIL")
            continue

        # Find output ONNX
        onnx_path = hyp_dir / "model.onnx"
        if not onnx_path.exists():
            candidates = list(hyp_dir.glob("*.onnx"))
            if candidates:
                onnx_path = candidates[0]
            else:
                results["hypotheses"][hyp_id] = {"status": "NO_ONNX", "label": label}
                results["errors"].append(f"{hyp_id}: build OK but no ONNX")
                continue

        # Phase A: screen
        screen_json = hyp_dir / "screen_perf.json"
        screen_p50, screen_cv = run_perf_screen(onnx_path, screen_json)

        if screen_p50 is None:
            results["hypotheses"][hyp_id] = {"status": "BENCH_FAIL", "label": label}
            results["errors"].append(f"{hyp_id}: screen bench failed")
            continue

        if screen_cv is not None and screen_cv > SCREEN_CV_MAX:
            print(
                f"  [warn] high CV={screen_cv:.3f} on GPU (unusual) — proceeding anyway", flush=True
            )

        # Phase B: full bench
        p50s = run_perf_full(onnx_path, hyp_dir)
        if not p50s:
            results["hypotheses"][hyp_id] = {
                "status": "BENCH_FAIL",
                "label": label,
                "screen_p50_ms": screen_p50,
            }
            continue

        median_p50 = sorted(p50s)[len(p50s) // 2]

        hyp_data: dict = {
            "status": "OK",
            "label": label,
            "opset": opset_used,
            "extra_optim": extra_optim or {},
            "screen_p50_ms": screen_p50,
            "screen_cv": screen_cv,
            "full_p50s_ms": p50s,
            "median_p50_ms": median_p50,
        }

        # Track baseline
        if hyp_id == "h0":
            baseline_p50 = median_p50
            results["baseline_p50_ms"] = baseline_p50
            print(f"  [baseline] p50={baseline_p50:.2f}ms", flush=True)

        # Compare to baseline
        if baseline_p50 and hyp_id != "h0":
            gain_pct = (baseline_p50 - median_p50) / baseline_p50 * 100
            hyp_data["gain_vs_baseline_pct"] = round(gain_pct, 2)
            verdict = (
                "KEEP"
                if gain_pct >= MIN_IMPROVEMENT_PCT
                else ("MARGINAL" if gain_pct > 0 else "DISCARD")
            )
            hyp_data["verdict"] = verdict
            print(
                f"  [{verdict}] gain={gain_pct:+.1f}% ({baseline_p50:.2f}ms → {median_p50:.2f}ms)",
                flush=True,
            )

            # Track best
            best_p50 = results.get("best_p50_ms")
            if best_p50 is None or median_p50 < best_p50:
                if gain_pct >= MIN_IMPROVEMENT_PCT:
                    results["best_p50_ms"] = median_p50
                    results["best_hypothesis"] = hyp_id
                    results["best_gain_pct"] = round(gain_pct, 2)

            # gpu-006: track opset21 result
            if opset_override == 21 and extra_optim is None:
                results["opset21_gain_pct"] = round(gain_pct, 2)
        else:
            hyp_data["verdict"] = "BASELINE"

        results["hypotheses"][hyp_id] = hyp_data

    # ── Step 2b: Phase C — confirmation runs for KEEP candidates ──────────────
    _run_confirmation_pass(results, model_dir, baseline_p50)

    # ── Step 3: finalise ───────────────────────────────────────────────────
    _post_process(results)
    _save_results(results, model_dir)
    _emit_model_artifacts(results, model_dir)
    return results


def _run_confirmation_pass(results: dict, model_dir: Path, baseline_p50: float | None) -> None:
    """Phase C: re-run CONFIRM_SESSIONS additional sessions for every KEEP candidate.

    If all (FULL_SESSIONS + CONFIRM_SESSIONS) sessions show >= MIN_IMPROVEMENT_PCT:
      verdict stays KEEP_CONFIRMED.
    Otherwise downgrade to MARGINAL_UNCONFIRMED.
    """
    if not baseline_p50:
        return
    hyps = results.get("hypotheses", {})
    keep_ids = [hid for hid, h in hyps.items() if h.get("verdict") == "KEEP"]
    if not keep_ids:
        return

    print(
        f"\n  ── Phase C: confirming {keep_ids} ({CONFIRM_SESSIONS} extra sessions each) ──",
        flush=True,
    )

    for hyp_id in keep_ids:
        hyp_data = hyps[hyp_id]
        onnx_path: Path | None = None
        hyp_dir = model_dir / hyp_id

        # Find built ONNX from the hypothesis dir
        for candidate in (hyp_dir / "optimized.onnx", hyp_dir / "quantized.onnx"):
            if candidate.exists():
                onnx_path = candidate
                break
        if onnx_path is None:
            print(f"  [confirm] {hyp_id}: no onnx found, skipping", flush=True)
            continue

        print(f"  [confirm] {hyp_id} ({hyp_data['label']})", flush=True)
        extra_p50s: list[float] = []
        for s in range(1, CONFIRM_SESSIONS + 1):
            out_json = hyp_dir / f"confirm_s{s}.json"
            rc, _, _ = run_cmd(
                [
                    WINML,
                    "perf",
                    "-m",
                    str(onnx_path),
                    "--ep",
                    EP,
                    "--device",
                    DEVICE,
                    "--warmup",
                    str(FULL_WARMUP),
                    "--iterations",
                    str(FULL_ITERS),
                    "--output",
                    str(out_json),
                ],
                label=f"confirm s{s}/{CONFIRM_SESSIONS}",
                timeout=BENCH_TIMEOUT_S,
            )
            p50 = _get_p50(out_json) if rc == 0 and out_json.exists() else None
            if p50:
                print(f"     confirm s{s}: p50={p50:.2f}ms", flush=True)
                extra_p50s.append(p50)
            if s < CONFIRM_SESSIONS:
                time.sleep(COOL_DOWN_S)

        if not extra_p50s:
            print(f"  [confirm] {hyp_id}: confirm bench failed, keeping KEEP", flush=True)
            continue

        all_p50s: list[float] = hyp_data.get("full_p50s_ms", []) + extra_p50s
        overall_median = sorted(all_p50s)[len(all_p50s) // 2]
        overall_gain = (baseline_p50 - overall_median) / baseline_p50 * 100
        wins = sum(
            1 for p in all_p50s if (baseline_p50 - p) / baseline_p50 * 100 >= MIN_IMPROVEMENT_PCT
        )

        hyp_data["confirm_p50s_ms"] = extra_p50s
        hyp_data["all_p50s_ms"] = all_p50s
        hyp_data["overall_median_p50_ms"] = round(overall_median, 3)
        hyp_data["overall_gain_pct"] = round(overall_gain, 2)
        hyp_data["sessions_above_threshold"] = wins
        hyp_data["total_sessions"] = len(all_p50s)

        if wins == len(all_p50s):
            hyp_data["verdict"] = "KEEP_CONFIRMED"
            print(
                f"  [KEEP_CONFIRMED] {hyp_id}: {wins}/{len(all_p50s)} sessions ≥ {MIN_IMPROVEMENT_PCT}%,"
                f" overall gain={overall_gain:+.1f}%",
                flush=True,
            )
        else:
            hyp_data["verdict"] = "MARGINAL_UNCONFIRMED"
            print(
                f"  [MARGINAL_UNCONFIRMED] {hyp_id}: only {wins}/{len(all_p50s)} sessions above threshold",
                flush=True,
            )

        # Update best_hypothesis tracking
        if hyp_data["verdict"] == "KEEP_CONFIRMED":
            best_p50 = results.get("best_p50_ms")
            if best_p50 is None or overall_median < best_p50:
                results["best_p50_ms"] = overall_median
                results["best_hypothesis"] = hyp_id
                results["best_gain_pct"] = round(overall_gain, 2)


def _post_process(results: dict) -> None:
    """Print summary and add cross-hypothesis notes."""
    hyps = results.get("hypotheses", {})
    baseline_p50 = results.get("baseline_p50_ms")
    if not baseline_p50:
        return

    keeps = [(hid, h) for hid, h in hyps.items() if h.get("verdict") in ("KEEP", "KEEP_CONFIRMED")]
    unconfirmed = [
        (hid, h) for hid, h in hyps.items() if h.get("verdict") == "MARGINAL_UNCONFIRMED"
    ]
    if keeps:
        print(f"\n  ✓ KEEP/KEEP_CONFIRMED: {[h[0] for h in keeps]}", flush=True)
    if unconfirmed:
        print(
            f"  ⚠ MARGINAL_UNCONFIRMED (failed confirmation): {[h[0] for h in unconfirmed]}",
            flush=True,
        )
    if not keeps and not unconfirmed:
        print("\n  No improvements found above 5% threshold.", flush=True)

    # gpu-006 summary
    opset21 = results.get("opset21_gain_pct")
    if opset21 is not None:
        if opset21 >= 5:
            print(f"  [gpu-006] opset21 HELPS GPU: +{opset21:.1f}%", flush=True)
        elif opset21 <= -5:
            print(f"  [gpu-006] opset21 HURTS GPU: {opset21:.1f}%", flush=True)
        else:
            print(f"  [gpu-006] opset21 NEUTRAL on GPU: {opset21:.1f}%", flush=True)


def _save_results(results: dict, model_dir: Path) -> None:
    out = model_dir / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results: {out}", flush=True)


def emit_champion_config(results: dict, model_dir: Path) -> None:
    """Write champion config to champion-configs/<slug>_<ep>_<device>_optimal.json."""
    best_h_id = results.get("best_hypothesis")
    if not best_h_id:
        return
    best_hyp = results.get("hypotheses", {}).get(best_h_id, {})

    cfg_path = model_dir / best_h_id / "build_config.json"
    build_config = None
    if cfg_path.exists():
        try:
            build_config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    model_slug = results["model_id"].replace("/", "--")
    ep = results.get("ep", "unknown")
    device = results.get("device", "unknown")

    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHAMPION_DIR / f"{model_slug}_{ep}_{device}_optimal.json"

    champion = {
        "model_id": results["model_id"],
        "ep": ep,
        "device": device,
        "champion_hypothesis": best_h_id,
        "champion_label": best_hyp.get("label", ""),
        "opset": best_hyp.get("opset"),
        "extra_optim": best_hyp.get("extra_optim", {}),
        "perf": {
            "baseline_p50_ms": results.get("baseline_p50_ms"),
            "champion_p50_ms": results.get("best_p50_ms"),
            "gain_pct": results.get("best_gain_pct"),
        },
        "build_config": build_config,
        "sweep_timestamp": results.get("timestamp"),
        "generated_by": Path(__file__).name,
    }
    out_path.write_text(json.dumps(champion, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Champion config: {out_path}", flush=True)


def _emit_model_artifacts(results: dict, model_dir: Path) -> None:
    emit_champion_config(results, model_dir)
    try:
        if generate_model_report is None:
            raise RuntimeError("gen_model_report import unavailable")
        generate_model_report(results, model_dir / "report.html")
    except Exception as e:
        print(f"  [warn] report generation failed: {e}", flush=True)


# ── summary writer ────────────────────────────────────────────────────────────


def write_summary(all_results: list[dict]) -> None:
    lines = [
        "# QNN GPU Optimization Sweep — Catalog Models",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"EP: `{EP}` / device: `{DEVICE}`  ",
        f"Protocol: screen {SCREEN_ITERS} iters (CV<{SCREEN_CV_MAX * 100:.0f}%),"
        f" full {FULL_ITERS}×{FULL_SESSIONS} sessions + {CONFIRM_SESSIONS} confirm sessions for KEEP  ",
        "Constraints: NO quant (gpu-004), NO compile (gpu-003), NO nhwc (gpu-002)  ",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
        "| Model | Baseline p50 | Best p50 | Best config | Gain% | opset21 gain% | Notes |",
        "|-------|-------------|----------|-------------|-------|--------------|-------|",
    ]

    for r in all_results:
        model_id = r["model_id"]
        baseline = f"{r['baseline_p50_ms']:.1f} ms" if r.get("baseline_p50_ms") else "N/A"
        best = f"{r['best_p50_ms']:.1f} ms" if r.get("best_p50_ms") else "N/A"
        best_h = r.get("best_hypothesis") or "N/A"
        best_label = ""
        if best_h != "N/A":
            best_label = r.get("hypotheses", {}).get(best_h, {}).get("label", "")
        gain = f"{r['best_gain_pct']:.1f}%" if r.get("best_gain_pct") is not None else "N/A"
        opset21 = r.get("opset21_gain_pct")
        opset21_str = f"{opset21:+.1f}%" if opset21 is not None else "N/A"
        errors = "; ".join(r.get("errors", []))[:80] or "none"
        lines.append(
            f"| `{model_id}` | {baseline} | {best} | {best_h} ({best_label}) | {gain} | {opset21_str} | {errors} |"
        )

    lines += [
        "",
        "## gpu-006: opset 21 on QNN GPU",
        "",
        "Previously untested. This sweep provides first data across multiple architectures.",
        "",
    ]

    opset21_helps = [r["model_id"] for r in all_results if (r.get("opset21_gain_pct") or 0) >= 5]
    opset21_hurts = [r["model_id"] for r in all_results if (r.get("opset21_gain_pct") or 0) <= -5]
    opset21_neutral = [
        r["model_id"]
        for r in all_results
        if r.get("opset21_gain_pct") is not None and -5 < (r.get("opset21_gain_pct") or 0) < 5
    ]
    lines += [
        f"- **Helps (≥5%):** {', '.join(opset21_helps) or 'none'}",
        f"- **Hurts (≤-5%):** {', '.join(opset21_hurts) or 'none'}",
        f"- **Neutral:** {', '.join(opset21_neutral) or 'none (no data yet)'}",
        "",
    ]

    lines += ["## Feature Gaps", ""]
    all_gaps = [
        f"- **`{r['model_id']}`**: {g}" for r in all_results for g in r.get("feature_gaps", [])
    ]
    lines += all_gaps if all_gaps else ["- None observed"]

    summary_path = RESULTS_DIR / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📄 Summary: {summary_path}", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QNN GPU hypothesis sweep for winml catalog models"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--model-type", default="auto")
    parser.add_argument(
        "--only-hypotheses", default=None, help="Comma-separated h IDs, e.g. h3,h4,h9"
    )
    parser.add_argument("--reuse-h0-config", action="store_true")
    args = parser.parse_args()

    only_hyp_ids: set[str] | None = None
    if args.only_hypotheses:
        only_hyp_ids = {h.strip() for h in args.only_hypotheses.split(",")}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm QNN GPU EP
    print("=== Confirming QNN GPU EP ===", flush=True)
    rc, out, _ = run_cmd([WINML, "sys", "--list-ep"], label="winml sys --list-ep", timeout=30)
    if "qnn" not in out.lower():
        print("❌ QNN EP not detected! Aborting.", flush=True)
        sys.exit(1)
    print("✓ QNN EP available\n", flush=True)

    if args.model:
        if not args.task:
            print("Error: --task required with --model", flush=True)
            sys.exit(1)
        models_to_run = [(args.model, args.task, args.model_type)]
    else:
        models_to_run = ALL_MODELS  # type: ignore[assignment]

    all_results: list[dict] = []

    for model_id, task, model_type in models_to_run:
        try:
            result = sweep_model(
                model_id,
                task,
                model_type,
                only_hyp_ids=only_hyp_ids,
                reuse_h0_config=args.reuse_h0_config,
            )
        except Exception as exc:
            print(f"\n❌ Unexpected error for {model_id}: {exc}", flush=True)
            result = {
                "model_id": model_id,
                "task": task,
                "model_type": model_type,
                "errors": [f"Unexpected exception: {exc}"],
                "hypotheses": {},
                "feature_gaps": [],
            }
        all_results.append(result)
        write_summary(all_results)

    print("\n" + "=" * 64, flush=True)
    print("  GPU SWEEP COMPLETE", flush=True)
    print("=" * 64, flush=True)
    write_summary(all_results)


if __name__ == "__main__":
    main()
