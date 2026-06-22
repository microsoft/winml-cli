#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""catalog_qnn_sweep.py — QNN NPU optimization hypothesis sweep for winml catalog models.

Hypothesis matrix (per model):
  h0: baseline (auto-config, default winml build for QNN NPU + W8A16)
  h1: opset 17 explicit (explicit opset, same optim as baseline)
  h2: opset 19
  h3: opset 21  <- tests npu-001 generalization

  Conv fusions (npu-006 hazard on Conv-dense models):
  h4: opset 17 + conv fusions (conv-bn, conv-add, conv-activation)
  h5: opset 21 + conv fusions

  Attention/transformer fusions (graph-analysis-driven; 2026-06-17):
  h6: opset 21 + matmul_transpose_fusion  (24-36× detected on all transformer models)
  h7: opset 21 + bias_softmax_fusion      (12× on BERT-family: roberta, bge, MiniLM)
  h8: opset 21 + attention_fusion         (12× Softmax nodes across all transformers)

  Rewrite hypotheses (graph-analysis-driven; 2026-06-17):
  h9: opset 21 + highdimRTR_lowdimRTR     (12× Reshape-Transpose-Reshape on MobileViT)
  h10: opset 17 + conv_add_fusion only    (11× on ResNet; safe subset of npu-006 convoy)

2-phase bench protocol (npu-007):
  Phase A: 200-iter screen — high CV is NORMAL on QNN NPU (DVFS), always proceed to Phase B.
  Phase B: 3 independent sessions x 500 iters, 30 s cool-down between sessions.
  KEEP criterion: all 3 sessions faster than baseline, ranges must not overlap.

Validated constraints applied:
  npu-006: conv fusions (conv-bn/add/activation) produce FusedConv ops that QNN EP cannot
    dispatch -> CPU fallback -> catastrophic regression on Conv-dense models. h4/h5 are
    annotated with npu006_expected_regression=True when Conv% of total ops > 20%.
  npu-001: opset21 speedup is architecture-specific. npu001_generalized uses range-overlap
    check (max(h3_p50s) < min(h1_p50s)), not just median comparison.

Results: catalog-qnn-sweep/<model_slug>/results.json
Summary: catalog-qnn-sweep/SUMMARY.md
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from gen_model_report import generate_model_report
except Exception:
    generate_model_report = None


sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# npu-006 guard: conv fusions produce FusedConv (ORT private op) that QNN EP cannot dispatch.
# On Conv-dense models (Conv% > this threshold), h4/h5 will catastrophically regress.
# Validated: ResNet-18 (Conv-dense) +4900%, DINOv2-base (1 Conv total) benign.
NPU006_CONV_PCT_THRESHOLD = 20.0  # percent of total ops; above this = high npu-006 risk

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WINML = str(BASE_DIR / ".venv" / "Scripts" / "winml.exe")
EP = "qnn"
DEVICE = "npu"
RESULTS_DIR = BASE_DIR / "catalog-qnn-sweep"
CHAMPION_DIR = BASE_DIR / "champion-configs"

SCREEN_WARMUP = 20
SCREEN_ITERS = 200
SCREEN_CV_MAX = 0.15

# Effect-size gate (anti-DVFS-artifact): a measured gain is only "reliable" if it is
# large relative to the run-to-run noise. We require gain_pct >= EFFECT_SIZE_CV_MULT * CV
# (CV = session-to-session spread of p50). On QNN NPU, CV is routinely 0.2-0.5, so a
# sub-5% "win" is almost always thermal/DVFS noise, not a real optimization. This guards
# against the npu-001/MobileViT failure where a thermally inflated baseline produced a
# fake +26% that did not reproduce on a clean run.
EFFECT_SIZE_CV_MULT = 2.0

FULL_WARMUP = 50
FULL_ITERS = 500
FULL_SESSIONS = 3
CONFIRM_SESSIONS = 2  # extra sessions for best hypothesis (Phase C confirmation)
COOL_DOWN_S = 30

MODEL_TIMEOUT_S = 180 * 60  # 3 hours per model — 6 hypotheses × ~30min each
BUILD_TIMEOUT_S = 8 * 60  # 8 min per individual build
BENCH_TIMEOUT_S = 8 * 60  # 8 min per bench run
EVAL_TIMEOUT_S = 6 * 60  # 6 min for accuracy eval
EVAL_SAMPLES = 50

# Hypotheses: (id, label, opset_override, extra_optim)
# opset_override=None → keep whatever auto-config chose
# extra_optim=None    → keep auto-config optim unchanged
# extra_optim=dict    → merge these flags ON TOP of auto-config optim
HYPOTHESES = [
    ("h0", "baseline (auto-config, W8A16)", None, None),
    ("h1", "opset 17 explicit", 17, None),
    ("h2", "opset 19", 19, None),
    ("h3", "opset 21 (tests npu-001 bypass)", 21, None),
    # ── conv fusions (npu-006) ──────────────────────────────────────────────
    (
        "h4",
        "opset 17 + conv fusions",
        17,
        {
            "conv_bn_fusion": True,
            "conv_add_fusion": True,
            "conv_activation_fusion": True,
        },
    ),
    (
        "h5",
        "opset 21 + conv fusions",
        21,
        {
            "conv_bn_fusion": True,
            "conv_add_fusion": True,
            "conv_activation_fusion": True,
        },
    ),
    # ── attention/transformer fusions (graph-analysis-driven, 2026-06-17) ──
    # matmul_transpose_fusion: 24-36× patterns detected on all transformer
    # models (dinov2, roberta, bge, mobilevit). Tests whether fusing
    # Transpose↔MatMul pairs helps QNN NPU dispatch.
    (
        "h6",
        "opset 21 + matmul_transpose_fusion",
        21,
        {"matmul_transpose_fusion": True},
    ),
    # bias_softmax_fusion: 12× Add→Softmax patterns in BERT-family models
    # (roberta, bge, MiniLM). Attention mask is added before softmax —
    # fusing may help QNN NPU kernel scheduling.
    (
        "h7",
        "opset 21 + bias_softmax_fusion",
        21,
        {"bias_softmax_fusion": True},
    ),
    # attention_fusion: 9-12× Softmax nodes across all transformers.
    # Full QK^T V attention fusion into a single op.
    (
        "h8",
        "opset 21 + attention_fusion",
        21,
        {"attention_fusion": True},
    ),
    # ── rewrite hypotheses (graph-analysis-driven, 2026-06-17) ─────────────
    # highdimRTR_lowdimRTR: 12× Reshape→Transpose→Reshape detected on
    # MobileViT. Reduces high-rank RTR chains to lower-rank equivalents,
    # potentially reducing Transpose overhead on QNN NPU.
    (
        "h9",
        "opset 21 + highdimRTR_lowdimRTR",
        21,
        {"highdimRTR_lowdimRTR": True},
    ),
    # conv_add_fusion only (safe subset of npu-006 convoy): 11× Conv→Add
    # on ResNet. Distinct from conv_add_activation_fusion (FusedConv) —
    # only fuses the Conv+bias Add, not the full 3-node chain.
    (
        "h10",
        "opset 17 + conv_add_fusion only",
        17,
        {"conv_add_fusion": True},
    ),
]

# Full catalog sweep list: (model_id, task, model_type, run_eval_on_baseline)
ALL_MODELS: list[tuple[str, str, str, bool]] = [
    # Vision
    ("microsoft/resnet-18", "image-classification", "resnet", True),
    ("google/vit-base-patch16-224", "image-classification", "vit", True),
    ("apple/mobilevit-small", "image-classification", "mobilevit", True),
    ("facebook/dinov2-small", "image-feature-extraction", "dinov2", False),  # no imagenet eval
    ("hustvl/yolos-small", "object-detection", "yolos", False),  # no imagenet eval
    # NLP
    (
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "text-classification",
        "distilbert",
        False,
    ),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-similarity", "bert", False),
    ("deepset/roberta-base-squad2", "question-answering", "roberta", False),
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
            snippet = (result.stderr or result.stdout or "")[-600:]
            print(f"     stderr: {snippet}", flush=True)
        return result.returncode, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"     TIMEOUT after {elapsed:.0f}s", flush=True)
        return -999, f"TIMEOUT after {timeout}s", elapsed


def _count_conv_pct(model_onnx: Path) -> tuple[float, int, int]:
    """Count Conv ops in a built ONNX model. Returns (conv_pct, conv_count, total_count).
    Used to assess npu-006 risk before running conv-fusion hypotheses.
    Falls back to (0.0, 0, 0) if onnx is not importable or file missing.

    WARNING: (0.0, 0, 0) means UNKNOWN, not SAFE. The caller must treat a zero
    result as unknown and emit a warning rather than silently skipping the guard.
    """
    if not model_onnx.exists():
        return 0.0, 0, 0
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        print(
            "  [ERROR] onnx package not installed — cannot assess npu-006 risk for conv fusions.\n"
            "          Install it: pip install onnx\n"
            "          Conv-fusion hypotheses (h4/h5) will be annotated as UNKNOWN risk.",
            flush=True,
        )
        return 0.0, 0, 0
    try:
        model = onnx.load(str(model_onnx))
        ops = [n.op_type for n in model.graph.node]
        total = len(ops)
        conv_count = sum(1 for o in ops if o == "Conv")
        pct = conv_count / total * 100 if total > 0 else 0.0
        return round(pct, 1), conv_count, total
    except Exception as e:
        print(f"  [warn] Conv% analysis failed: {e}", flush=True)
        return 0.0, 0, 0


# ── winml wrappers ────────────────────────────────────────────────────────────


def get_base_config(model_id: str, task: str, model_type: str) -> dict | None:
    """Generate the auto-config via `winml config` for QNN NPU.
    Returns the parsed config dict, or None on failure.
    """
    tmp_path = RESULTS_DIR / "_tmp_base_config.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    def _try(extra_args: list[str]) -> dict | None:
        cmd = [
            WINML,
            "config",
            "-m",
            model_id,
            "-t",
            task,
            "--device",
            DEVICE,
            "--ep",
            EP,
            "--no-compile",
            "-o",
            str(tmp_path),
        ] + extra_args
        rc, out, _ = run_cmd(cmd, label="winml config", timeout=120)
        if rc == 0 and tmp_path.exists():
            try:
                cfg = json.loads(tmp_path.read_text(encoding="utf-8"))
                tmp_path.unlink(missing_ok=True)
                return cfg
            except Exception as e:
                print(f"  [warn] config parse error: {e}", flush=True)
        tmp_path.unlink(missing_ok=True)
        return None

    # Try with explicit model-type first, fall back without it
    cfg = _try(["--model-type", model_type])
    if cfg is None:
        print("  [warn] config with --model-type failed, retrying without…", flush=True)
        cfg = _try([])
    return cfg


def make_hypothesis_config(
    base: dict, opset_override: int | None, extra_optim: dict | None
) -> dict:
    """Return a modified copy of base config for this hypothesis."""
    cfg = copy.deepcopy(base)
    if opset_override is not None:
        if cfg.get("export"):
            cfg["export"]["opset_version"] = opset_override
    if extra_optim is not None:
        existing = cfg.get("optim") or {}
        cfg["optim"] = {**existing, **extra_optim}
    return cfg


def run_build(model_id: str, cfg_path: Path, out_dir: Path) -> tuple[bool, str]:
    """Run `winml build -c cfg_path -m model_id -o out_dir --ep qnn --device npu --no-compile`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        WINML,
        "build",
        "-c",
        str(cfg_path),
        "-m",
        model_id,
        "-o",
        str(out_dir),
        "--ep",
        EP,
        "--device",
        DEVICE,
        "--no-compile",
        "--rebuild",
    ]
    rc, out, _ = run_cmd(cmd, label=f"winml build [{out_dir.name}]", timeout=BUILD_TIMEOUT_S)
    return rc == 0, out


def bench_screen(model_path: Path) -> tuple[float | None, float, bool]:
    """Phase A: 200-iter screen.
    Returns (p50_ms, cv, stable).
    p50_ms=None only on hard failure (rc!=0 or missing output file).
    QNN NPU DVFS routinely produces CV >> 0.15 — high CV is logged but does NOT
    block Phase B; Phase B's multi-session cool-down is the thermal control.
    """
    out_json = model_path.parent / "screen_perf.json"
    rc, _, _ = run_cmd(
        [
            WINML,
            "perf",
            "-m",
            str(model_path),
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--warmup",
            str(SCREEN_WARMUP),
            "--iterations",
            str(SCREEN_ITERS),
            "-o",
            str(out_json),
        ],
        label=f"perf screen ({SCREEN_ITERS} iters)",
        timeout=BENCH_TIMEOUT_S,
    )
    if rc != 0 or not out_json.exists():
        return None, 999.0, False
    try:
        data = json.loads(out_json.read_text())
        lat = data["latency_ms"]
        p50, std = lat["p50"], lat["std"]
        cv = std / p50 if p50 > 0 else 999.0
        stable = cv <= SCREEN_CV_MAX
        tag = "stable" if stable else "HIGH-CV (DVFS noise — proceeding to Phase B)"
        print(f"     screen: p50={p50:.2f}ms  std={std:.2f}ms  CV={cv:.3f}  [{tag}]", flush=True)
        return p50, cv, stable
    except Exception as e:
        print(f"     [warn] screen parse error: {e}", flush=True)
        return None, 999.0, False


def bench_full(model_path: Path) -> list[float]:
    """Phase B: 3 × 500-iter full bench with cool-down. Returns list of p50 values."""
    p50s: list[float] = []
    for s in range(1, FULL_SESSIONS + 1):
        out_json = model_path.parent / f"full_perf_s{s}.json"
        rc, _, _ = run_cmd(
            [
                WINML,
                "perf",
                "-m",
                str(model_path),
                "--ep",
                EP,
                "--device",
                DEVICE,
                "--warmup",
                str(FULL_WARMUP),
                "--iterations",
                str(FULL_ITERS),
                "-o",
                str(out_json),
            ],
            label=f"perf full s{s}/{FULL_SESSIONS} ({FULL_ITERS} iters)",
            timeout=BENCH_TIMEOUT_S,
        )
        if rc == 0 and out_json.exists():
            try:
                data = json.loads(out_json.read_text())
                lat = data["latency_ms"]
                p50, std = lat["p50"], lat["std"]
                cv = std / p50 if p50 > 0 else 999.0
                print(f"     full s{s}: p50={p50:.2f}ms  std={std:.2f}ms  CV={cv:.3f}", flush=True)
                p50s.append(p50)
            except Exception as e:
                print(f"     [warn] full bench s{s} parse error: {e}", flush=True)
        else:
            print(f"     [warn] full bench s{s} failed", flush=True)
        if s < FULL_SESSIONS:
            print(f"     cool-down {COOL_DOWN_S}s…", flush=True)
            time.sleep(COOL_DOWN_S)
    return p50s


def run_eval(model_path: Path, model_id: str, task: str) -> float | None:
    """Run `winml eval` for accuracy. Returns accuracy or None."""
    out_json = model_path.parent / "eval_result.json"
    rc, _, _ = run_cmd(
        [
            WINML,
            "eval",
            "-m",
            str(model_path),
            "--model-id",
            model_id,
            "--task",
            task,
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--samples",
            str(EVAL_SAMPLES),
            "-o",
            str(out_json),
        ],
        label="winml eval (accuracy gate)",
        timeout=EVAL_TIMEOUT_S,
    )
    if rc != 0 or not out_json.exists():
        return None
    try:
        data = json.loads(out_json.read_text())
        metrics = data.get("metrics", data)
        acc = metrics.get("accuracy")
        if acc is not None:
            print(f"     eval accuracy: {acc:.4f}", flush=True)
        return float(acc) if acc is not None else None
    except Exception as e:
        print(f"     [warn] eval parse error: {e}", flush=True)
        return None


def _perf_result(onnx_path: Path, model_id: str, task: str, run_eval_flag: bool) -> dict:
    """Run Phase A + Phase B bench and optionally eval. Returns result dict."""
    result: dict = {"status": "PENDING", "screen": {}, "full": {}, "accuracy": None}

    p50_screen, cv_screen, stable = bench_screen(onnx_path)
    result["screen"] = {
        "p50_ms": p50_screen,
        "cv": round(cv_screen, 4),
        "stable": stable,
    }

    if p50_screen is None:
        # Hard failure (rc != 0 or missing output) — cannot proceed
        result["status"] = "SCREEN_FAIL"
        return result

    # QNN NPU note: always proceed to Phase B even if screen CV is high.
    # Phase B multi-session cool-down is the thermal / DVFS control.
    if not stable:
        result["screen"]["note"] = "DVFS noise — high CV expected on QNN NPU"

    full_p50s = bench_full(onnx_path)
    if not full_p50s:
        result["status"] = "BENCH_FAIL"
        return result

    median_p50 = float(sorted(full_p50s)[len(full_p50s) // 2])
    result["full"] = {
        "p50s_ms": [round(p, 3) for p in full_p50s],
        "median_p50_ms": round(median_p50, 3),
    }
    result["status"] = "OK" if stable else "OK_HIGH_CV"

    if run_eval_flag:
        acc = run_eval(onnx_path, model_id, task)
        result["accuracy"] = acc

    return result


# ── main sweep logic ──────────────────────────────────────────────────────────


def sweep_model(
    model_id: str,
    task: str,
    model_type: str,
    run_eval_on_baseline: bool,
    only_hyp_ids: "set[str] | None" = None,
    reuse_h0_config: bool = False,
) -> dict:
    """Run hypotheses for one model on QNN NPU. Returns results dict.

    Args:
        only_hyp_ids: If set, only run these hypothesis IDs (e.g. {'h6','h7'}).
        reuse_h0_config: If True, load base config from existing h0/build_config.json
                         instead of calling winml config again.
    """
    model_slug = model_id.replace("/", "--")
    model_dir = RESULTS_DIR / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    # When resuming from partial run, load existing results to preserve prior data
    results_path = model_dir / "results.json"
    if only_hyp_ids and results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            print(f"  [resume] loaded existing results from {results_path}", flush=True)
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
    results.setdefault("conv_pct", None)
    results.setdefault("npu006_risk", None)
    results.setdefault("npu006_regression", None)
    results.setdefault("hypotheses", {})
    results.setdefault("best_hypothesis", None)
    results.setdefault("baseline_p50_ms", None)
    results.setdefault("best_p50_ms", None)
    results.setdefault("best_gain_pct", None)
    results.setdefault("best_gain_reliable", None)
    results.setdefault("best_gain_verdict", None)
    results.setdefault("best_gain_noise_floor_pct", None)
    results.setdefault("best_gain_ranges_separated", None)
    results.setdefault("npu001_generalized", None)
    results.setdefault("npu001_ranges_non_overlapping", None)
    results.setdefault("feature_gaps", [])
    results.setdefault("errors", [])

    print(f"\n{'=' * 64}", flush=True)
    print(f"  SWEEP: {model_id}  [{task}]", flush=True)
    if only_hyp_ids:
        print(f"  (delta sweep — only: {sorted(only_hyp_ids)})", flush=True)
    print(f"{'=' * 64}", flush=True)

    model_start = time.time()

    # ── Step 1: generate base config (or reuse from existing h0) ──────────────
    print("\n[1/3] Generating base config (winml config)…", flush=True)
    base_config = None

    if reuse_h0_config:
        h0_cfg_path = model_dir / "h0" / "build_config.json"
        if h0_cfg_path.exists():
            try:
                base_config = json.loads(h0_cfg_path.read_text(encoding="utf-8"))
                print(f"  [reuse] loaded h0 config from {h0_cfg_path}", flush=True)
            except Exception as e:
                print(f"  [reuse] failed to load h0 config: {e} — regenerating", flush=True)

    if base_config is None:
        base_config = get_base_config(model_id, task, model_type)

    if base_config is None:
        results["errors"].append("base config generation failed — model may not be supported")
        results["feature_gaps"].append("winml config failed for this model (inspect winml output)")
        _save_results(results, model_dir)
        _emit_model_artifacts(results, model_dir)
        return results

    baseline_opset = (base_config.get("export") or {}).get("opset_version", "?")
    results["baseline_opset"] = baseline_opset
    base_quant = base_config.get("quant")
    print(
        f"  auto-config: opset={baseline_opset}  quant={'W8A16' if base_quant else 'NONE'}",
        flush=True,
    )
    if base_quant is None:
        results["feature_gaps"].append(
            "auto-config did not include quantization — possible model type not supported for W8A16"
        )
    optim_keys = list((base_config.get("optim") or {}).keys())
    print(f"  auto-config optim: {optim_keys}", flush=True)

    # ── Step 2: per-hypothesis loop ───────────────────────────────────────────
    print(f"\n[2/3] Running {len(HYPOTHESES)} hypotheses…", flush=True)

    # conv_pct is filled in after h0 succeeds (used to annotate npu-006 risk for h4/h5)
    conv_pct: float = 0.0
    npu006_risk: bool = False

    for hyp_id, label, opset_override, extra_optim in HYPOTHESES:
        # Hypothesis filter: skip if not in --only-hypotheses list
        if only_hyp_ids is not None and hyp_id not in only_hyp_ids:
            continue
        elapsed_total = time.time() - model_start
        if elapsed_total > MODEL_TIMEOUT_S:
            print(
                f"\n  ⏰ MODEL TIMEOUT ({elapsed_total:.0f}s > {MODEL_TIMEOUT_S}s) — stopping",
                flush=True,
            )
            results["hypotheses"][hyp_id] = {"status": "TIMEOUT", "label": label}
            results["errors"].append(f"Model timed out at {elapsed_total:.0f}s (before {hyp_id})")
            continue

        sep = "─" * 56
        print(f"\n{sep}", flush=True)
        print(f"  {hyp_id}: {label}", flush=True)
        print(f"{sep}", flush=True)

        # Build config for this hypothesis
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
            is_timeout = "TIMEOUT" in build_out
            status = "BUILD_TIMEOUT" if is_timeout else "BUILD_FAIL"
            error_snippet = build_out[-600:] if not is_timeout else "build timed out"
            results["hypotheses"][hyp_id] = {
                "status": status,
                "label": label,
                "opset": opset_used,
                "build_error": error_snippet,
            }
            results["errors"].append(f"{hyp_id}: {status}")
            # Try to extract feature gap info from the build output
            if any(
                kw in build_out.lower() for kw in ("unsupported", "not supported", "no handler")
            ):
                results["feature_gaps"].append(
                    f"{hyp_id} ({label}): EP/op unsupported — '{build_out[-200:]}'"
                )
            elif is_timeout:
                results["feature_gaps"].append(
                    f"{hyp_id} ({label}): build timeout — possible QNN compilation hang"
                )
            continue

        onnx_path = hyp_dir / "model.onnx"
        if not onnx_path.exists():
            # Check for EPContext model (compile might have happened anyway)
            ctx_candidates = list(hyp_dir.glob("*_ctx*.onnx")) + list(
                hyp_dir.glob("model_npu*.onnx")
            )
            if ctx_candidates:
                onnx_path = ctx_candidates[0]
                print(f"  [info] using compiled model: {onnx_path.name}", flush=True)
            else:
                results["hypotheses"][hyp_id] = {
                    "status": "NO_MODEL_ONNX",
                    "label": label,
                    "opset": opset_used,
                }
                results["errors"].append(f"{hyp_id}: build OK but model.onnx missing")
                results["feature_gaps"].append(
                    f"{hyp_id}: build completed but no model.onnx produced (unexpected pipeline behavior)"
                )
                continue

        # After h0: analyze Conv% to assess npu-006 risk for h4/h5
        if hyp_id == "h0" and onnx_path.exists():
            conv_pct, conv_count, total_count = _count_conv_pct(onnx_path)
            # Treat (0.0, 0, 0) as UNKNOWN (not safe) — onnx may be unavailable.
            conv_unknown = conv_pct == 0.0 and total_count == 0
            npu006_risk = conv_pct > NPU006_CONV_PCT_THRESHOLD or conv_unknown
            results["conv_pct"] = None if conv_unknown else conv_pct
            results["npu006_risk"] = npu006_risk
            if conv_unknown:
                print(
                    "  [npu-006] Conv% analysis returned UNKNOWN (onnx unavailable or file missing)"
                    " — treating h4/h5 as HIGH RISK to be safe",
                    flush=True,
                )
            elif npu006_risk:
                print(
                    f"  [npu-006] Conv%={conv_pct:.1f}% ({conv_count}/{total_count} ops)"
                    f" > {NPU006_CONV_PCT_THRESHOLD:.0f}% threshold",
                    flush=True,
                )
                print(
                    "  [npu-006] h4/h5 (conv fusions) EXPECTED to catastrophically regress"
                    " — FusedConv not supported by QNN EP -> CPU fallback",
                    flush=True,
                )
            else:
                print(
                    f"  [npu-006] Conv%={conv_pct:.1f}% ({conv_count}/{total_count} ops)"
                    f" <= {NPU006_CONV_PCT_THRESHOLD:.0f}% — h4/h5 low risk",
                    flush=True,
                )

        # Annotate h4/h5 with npu-006 risk BEFORE running bench
        if hyp_id in ("h4", "h5") and npu006_risk:
            print(
                f"  [npu-006] WARNING: {hyp_id} uses conv fusions on Conv-dense model"
                f" (Conv%={conv_pct:.1f}%) — expect catastrophic regression",
                flush=True,
            )

        # Only run eval for h0 (baseline) on image-classification models
        do_eval = run_eval_on_baseline and hyp_id == "h0" and task == "image-classification"

        bench = _perf_result(onnx_path, model_id, task, do_eval)
        bench["label"] = label
        bench["opset"] = opset_used
        bench["extra_optim"] = extra_optim or {}
        if hyp_id in ("h4", "h5"):
            bench["npu006_expected_regression"] = npu006_risk
        results["hypotheses"][hyp_id] = bench

        if bench["status"] == "UNSTABLE":
            results["errors"].append(f"{hyp_id}: bench UNSTABLE (CV too high)")

    # ── Step 3: compute summary stats ─────────────────────────────────────────
    print("\n[3/3] Computing summary stats…", flush=True)
    _compute_summary(results)

    # ── Step 3b: Phase C — confirm the best hypothesis with 2 extra sessions ──
    _run_confirmation_pass_npu(results, model_dir)

    _save_results(results, model_dir)
    _emit_model_artifacts(results, model_dir)
    return results


def _run_confirmation_pass_npu(results: dict, model_dir: Path) -> None:
    """Phase C: run CONFIRM_SESSIONS extra sessions on the best hypothesis.

    For NPU (high DVFS noise), uses range-non-overlap criterion:
    - All (FULL_SESSIONS + CONFIRM_SESSIONS) p50s < baseline_min → CONFIRMED
    - Otherwise → MARGINAL_UNCONFIRMED, best_gain_pct flagged as uncertain
    """
    best_h_id: str | None = results.get("best_hypothesis")
    baseline_p50: float | None = results.get("baseline_p50_ms")
    if not best_h_id or not baseline_p50:
        return

    best_hyp = results["hypotheses"].get(best_h_id, {})
    best_gain = results.get("best_gain_pct", 0.0)
    if best_gain < 5.0:
        return  # nothing worth confirming

    # Find ONNX
    hyp_dir = model_dir / best_h_id
    onnx_path: Path | None = None
    for candidate in (hyp_dir / "quantized.onnx", hyp_dir / "optimized.onnx"):
        if candidate.exists():
            onnx_path = candidate
            break
    if onnx_path is None:
        return

    print(
        f"\n  ── Phase C: confirming best hypothesis {best_h_id} ({CONFIRM_SESSIONS} extra sessions) ──",
        flush=True,
    )

    confirm_p50s: list[float] = []
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
                "-o",
                str(out_json),
            ],
            label=f"confirm s{s}/{CONFIRM_SESSIONS}",
            timeout=BENCH_TIMEOUT_S,
        )
        if rc == 0 and out_json.exists():
            try:
                data = json.loads(out_json.read_text())
                lat = data["latency_ms"]
                p50 = lat["p50"]
                print(f"     confirm s{s}: p50={p50:.2f}ms", flush=True)
                confirm_p50s.append(p50)
            except Exception as e:
                print(f"     [warn] confirm s{s} parse error: {e}", flush=True)
        if s < CONFIRM_SESSIONS:
            print(f"     cool-down {COOL_DOWN_S}s…", flush=True)
            time.sleep(COOL_DOWN_S)

    if not confirm_p50s:
        print(f"  [confirm] {best_h_id}: confirm bench failed, conclusion unchanged", flush=True)
        return

    # Get all p50s including prior FULL_SESSIONS runs
    prior_p50s: list[float] = best_hyp.get("full", {}).get("p50s_ms", [])
    all_p50s = prior_p50s + confirm_p50s

    # Baseline comparison: use h0/h1 p50s for range overlap test
    baseline_h = None
    for h_id in ("h0", "h1"):
        h = results["hypotheses"].get(h_id, {})
        if h.get("status") in ("OK", "OK_HIGH_CV"):
            baseline_h = h
            break
    baseline_p50s: list[float] = (
        baseline_h["full"].get("p50s_ms", [baseline_p50]) if baseline_h else [baseline_p50]
    )

    overall_median = float(sorted(all_p50s)[len(all_p50s) // 2])
    overall_gain = (baseline_p50 - overall_median) / baseline_p50 * 100
    # Strict: max of all best-hypothesis sessions must be < min of baseline sessions
    ranges_confirmed = max(all_p50s) < min(baseline_p50s) if baseline_p50s else False

    best_hyp["confirm_p50s_ms"] = [round(p, 3) for p in confirm_p50s]
    best_hyp["all_p50s_ms"] = [round(p, 3) for p in all_p50s]
    best_hyp["confirm_overall_median_ms"] = round(overall_median, 3)
    best_hyp["confirm_overall_gain_pct"] = round(overall_gain, 2)
    best_hyp["confirm_ranges_non_overlapping"] = ranges_confirmed

    if ranges_confirmed:
        best_hyp["confirm_verdict"] = "CONFIRMED"
        results["best_gain_pct"] = round(overall_gain, 2)
        print(
            f"  [CONFIRMED] {best_h_id}: all {len(all_p50s)} p50s < baseline min"
            f" — gain={overall_gain:+.1f}% (ranges non-overlapping)",
            flush=True,
        )
    else:
        best_hyp["confirm_verdict"] = "MARGINAL_UNCONFIRMED"
        print(
            f"  [MARGINAL_UNCONFIRMED] {best_h_id}: max={max(all_p50s):.1f}ms"
            f" ≥ baseline min={min(baseline_p50s):.1f}ms — DVFS noise, ranges overlap",
            flush=True,
        )


def _session_cv(p50s: "list[float]") -> float:
    """Session-to-session CV (std/mean) of per-session p50 values.

    This is the run-to-run noise floor used by the effect-size gate. Unlike the
    intra-session CV (screen.cv), this captures thermal/DVFS drift *between*
    sessions, which is exactly the noise that produces fake cross-config wins.
    Returns 0.0 for <2 samples (cannot estimate spread).
    """
    n = len(p50s)
    if n < 2:
        return 0.0
    mean = sum(p50s) / n
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in p50s) / (n - 1)
    return (var**0.5) / mean


def _compute_summary(results: dict) -> None:
    """Fill in baseline_p50, best_hypothesis, best_gain, npu001_generalized, npu006_regression."""
    hyps = results["hypotheses"]

    # Baseline p50: prefer h0, fall back to h1
    baseline_p50: float | None = None
    baseline_h: dict = {}
    for h_id in ("h0", "h1"):
        h = hyps.get(h_id, {})
        if h.get("status") in ("OK", "OK_HIGH_CV"):
            baseline_p50 = h.get("full", {}).get("median_p50_ms")
            if baseline_p50:
                baseline_h = h
                break
    results["baseline_p50_ms"] = baseline_p50

    # Best hypothesis (minimum median p50)
    best_p50: float | None = None
    best_h: str | None = None
    best_hyp: dict = {}
    for h_id, h in hyps.items():
        if h.get("status") in ("OK", "OK_HIGH_CV"):
            p50 = h.get("full", {}).get("median_p50_ms")
            if p50 is not None and (best_p50 is None or p50 < best_p50):
                best_p50 = p50
                best_h = h_id
                best_hyp = h
    results["best_hypothesis"] = best_h
    results["best_p50_ms"] = best_p50

    if baseline_p50 and best_p50:
        gain_pct = (baseline_p50 - best_p50) / baseline_p50 * 100
        results["best_gain_pct"] = round(gain_pct, 2)

        # ── effect-size gate (anti-DVFS-artifact) ───────────────────────────
        # A gain is only "reliable" if it clears BOTH:
        #   (a) effect size: gain_pct >= EFFECT_SIZE_CV_MULT * noise_floor_pct
        #       where noise_floor = max session-to-session CV of baseline & best
        #   (b) range separation: max(best_p50s) < min(baseline_p50s)
        # This stops a thermally inflated baseline (high CV, overlapping ranges)
        # from being recorded as a real optimization. See npu-001/MobileViT.
        base_p50s: list[float] = baseline_h.get("full", {}).get("p50s_ms", [])
        best_p50s: list[float] = best_hyp.get("full", {}).get("p50s_ms", [])
        noise_cv = max(_session_cv(base_p50s), _session_cv(best_p50s))
        noise_floor_pct = round(EFFECT_SIZE_CV_MULT * noise_cv * 100, 2)
        ranges_separated = bool(best_p50s and base_p50s and max(best_p50s) < min(base_p50s))
        effect_size_ok = gain_pct >= noise_floor_pct
        reliable = bool(effect_size_ok and ranges_separated and best_h != "h0")

        results["best_gain_noise_floor_pct"] = noise_floor_pct
        results["best_gain_ranges_separated"] = ranges_separated
        results["best_gain_reliable"] = reliable
        if best_h == "h0":
            results["best_gain_verdict"] = "BASELINE_IS_BEST"
        elif reliable:
            results["best_gain_verdict"] = "RELIABLE"
        elif not effect_size_ok:
            results["best_gain_verdict"] = "NEUTRAL_WITHIN_NOISE"
        else:
            results["best_gain_verdict"] = "UNRELIABLE_RANGES_OVERLAP"
        print(
            f"  [effect-size] best={best_h} gain={gain_pct:+.1f}% "
            f"noise_floor={noise_floor_pct:.1f}% ranges_sep={ranges_separated} "
            f"-> {results['best_gain_verdict']}",
            flush=True,
        )

    # ── npu-001: opset21 vs opset17 (h3 vs h1) ──────────────────────────────
    # Criterion 1 (median): h3 p50 < h1 p50 by >=5%
    # Criterion 2 (range-overlap, stricter): max(h3_p50s) < min(h1_p50s)
    # Both must agree for "True"; either failing gives "neutral"
    h1 = hyps.get("h1", {})
    h3 = hyps.get("h3", {})
    if h1.get("status") in ("OK", "OK_HIGH_CV") and h3.get("status") in ("OK", "OK_HIGH_CV"):
        p50_h1 = h1["full"].get("median_p50_ms", float("inf"))
        p50_h3 = h3["full"].get("median_p50_ms", float("inf"))
        h1_p50s: list[float] = h1["full"].get("p50s_ms", [p50_h1])
        h3_p50s: list[float] = h3["full"].get("p50s_ms", [p50_h3])

        # Median-based test (>=5% improvement)
        median_gain = p50_h3 < p50_h1 * 0.95
        median_loss = p50_h1 < p50_h3 * 0.95

        # Range-overlap test (non-overlapping = more reliable for DVFS-noisy NPU)
        ranges_non_overlapping = max(h3_p50s) < min(h1_p50s) if h3_p50s and h1_p50s else None
        results["npu001_ranges_non_overlapping"] = ranges_non_overlapping

        if median_gain and ranges_non_overlapping:
            results["npu001_generalized"] = True
            gain = (p50_h1 - p50_h3) / p50_h1 * 100
            print(
                f"  [npu-001] CONFIRMED: opset21={p50_h3:.1f}ms vs opset17={p50_h1:.1f}ms"
                f" (+{gain:.1f}%, ranges non-overlapping)",
                flush=True,
            )
        elif median_gain and not ranges_non_overlapping:
            results["npu001_generalized"] = "median_only"
            gain = (p50_h1 - p50_h3) / p50_h1 * 100
            print(
                f"  [npu-001] MARGINAL: opset21 median {gain:.1f}% faster but ranges OVERLAP"
                f" (h3 max={max(h3_p50s):.1f}ms > h1 min={min(h1_p50s):.1f}ms) -- DVFS noise",
                flush=True,
            )
        elif median_loss:
            results["npu001_generalized"] = False
            print(
                f"  [npu-001] NEGATIVE: opset17={p50_h1:.1f}ms < opset21={p50_h3:.1f}ms",
                flush=True,
            )
        else:
            results["npu001_generalized"] = "neutral"
            print(
                f"  [npu-001] NEUTRAL: opset17={p50_h1:.1f}ms ~ opset21={p50_h3:.1f}ms",
                flush=True,
            )
    else:
        missing = [
            h for h, d in [("h1", h1), ("h3", h3)] if d.get("status") not in ("OK", "OK_HIGH_CV")
        ]
        results["npu001_generalized"] = f"N/A ({', '.join(missing)} not OK)"
        results["npu001_ranges_non_overlapping"] = None

    # ── npu-006: detect catastrophic conv-fusion regression (h4/h5) ──────────
    # "Catastrophic" = h4 or h5 median p50 >= 5x baseline (CPU fallback signature)
    npu006_regression = False
    for h_id in ("h4", "h5"):
        h = hyps.get(h_id, {})
        if h.get("status") in ("OK", "OK_HIGH_CV") and baseline_p50:
            p50_fused = h["full"].get("median_p50_ms")
            if p50_fused and p50_fused >= baseline_p50 * 5.0:
                npu006_regression = True
                ratio = p50_fused / baseline_p50
                print(
                    f"  [npu-006] CATASTROPHIC REGRESSION confirmed on {h_id}:"
                    f" {p50_fused:.1f}ms vs baseline {baseline_p50:.1f}ms ({ratio:.0f}x slower)"
                    f" -- FusedConv CPU fallback",
                    flush=True,
                )
        elif h.get("status") == "BENCH_FAIL" and h.get("npu006_expected_regression"):
            # Bench failure on expected-regression hypothesis is also a signal
            print(
                f"  [npu-006] {h_id} bench FAILED on conv-dense model -- possible CPU fallback timeout",
                flush=True,
            )
    results["npu006_regression"] = npu006_regression


def _save_results(results: dict, model_dir: Path) -> None:
    """Write results.json."""
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
    """Write SUMMARY.md to RESULTS_DIR."""
    lines: list[str] = [
        "# QNN NPU Optimization Sweep — Catalog Models",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"EP: `{EP}` / device: `{DEVICE}`  ",
        f"Bench protocol: Phase-A {SCREEN_ITERS} iters (high CV expected on QNN NPU — DVFS),"
        f" Phase-B {FULL_ITERS}x{FULL_SESSIONS} sessions, 30s cool-down  ",
        "npu-001 criterion: median >=5% gain AND ranges non-overlapping  ",
        "npu-006 criterion: Conv% of ops; h4/h5 marked catastrophic if >=5x baseline  ",
        f"Effect-size gate: gain reliable only if gain% >= {EFFECT_SIZE_CV_MULT:.0f}×(session-CV) AND ranges separated  ",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
        "| Model | Conv% | Baseline p50 | Best p50 | Best config | Gain% | Reliable? | npu-001? | npu-006 regression? | Notes |",
        "|-------|-------|-------------|----------|-------------|-------|-----------|----------|---------------------|-------|",
    ]

    for r in all_results:
        model_id = r["model_id"]
        conv_pct = r.get("conv_pct")
        conv_str = f"{conv_pct:.0f}%" if conv_pct is not None else "N/A"
        if r.get("npu006_risk"):
            conv_str += " ⚠️"
        baseline = f"{r['baseline_p50_ms']:.1f} ms" if r.get("baseline_p50_ms") else "N/A"
        best = f"{r['best_p50_ms']:.1f} ms" if r.get("best_p50_ms") else "N/A"
        best_h = r.get("best_hypothesis") or "N/A"
        if best_h != "N/A":
            h_data = r.get("hypotheses", {}).get(best_h, {})
            best_label = h_data.get("label", "")
        else:
            best_label = ""
        gain = f"{r['best_gain_pct']:.1f}%" if r.get("best_gain_pct") is not None else "N/A"
        npu001 = r.get("npu001_generalized")
        non_overlap = r.get("npu001_ranges_non_overlapping")
        if npu001 is True:
            npu001_str = "CONFIRMED (ranges sep.)" if non_overlap else "YES (median)"
        elif npu001 is False:
            npu001_str = "NO"
        elif npu001 == "median_only":
            npu001_str = "MARGINAL (overlap)"
        elif npu001 == "neutral":
            npu001_str = "neutral"
        else:
            npu001_str = "N/A"
        npu006 = (
            "YES ⚠️" if r.get("npu006_regression") else ("risk" if r.get("npu006_risk") else "no")
        )
        verdict = r.get("best_gain_verdict")
        verdict_map = {
            "RELIABLE": "✅ reliable",
            "NEUTRAL_WITHIN_NOISE": "⚠️ within noise",
            "UNRELIABLE_RANGES_OVERLAP": "⚠️ ranges overlap",
            "BASELINE_IS_BEST": "baseline best",
        }
        reliable_str = verdict_map.get(verdict, "N/A")
        errors = "; ".join(r.get("errors", []))[:80] or "none"
        lines.append(
            f"| `{model_id}` | {conv_str} | {baseline} | {best} | {best_h} ({best_label}) | {gain} | {reliable_str} | {npu001_str} | {npu006} | {errors} |"
        )

    # Per-model hypothesis breakdown
    lines += [
        "",
        "## Hypothesis Breakdown per Model",
        "",
    ]
    for r in all_results:
        lines.append(f"### {r['model_id']}")
        lines.append("")
        lines.append(
            "| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |"
        )
        lines.append(
            "|------------|-------|-----------|-------------------|-----|--------|---------|"
        )
        for h_id, h_data in r.get("hypotheses", {}).items():
            lbl = h_data.get("label", "")
            opset = h_data.get("opset", "?")
            s_p50 = h_data.get("screen", {}).get("p50_ms")
            s_p50_str = f"{s_p50:.1f}" if s_p50 else "—"
            f_p50 = h_data.get("full", {}).get("median_p50_ms")
            f_p50_str = f"{f_p50:.1f}" if f_p50 else "—"
            cv = h_data.get("screen", {}).get("cv", "?")
            cv_str = f"{cv:.3f}" if isinstance(cv, float) else str(cv)
            status = h_data.get("status", "?")
            stable = h_data.get("screen", {}).get("stable", True)
            if not stable and status.startswith("OK"):
                status += " ⚡DVFS"
            acc = h_data.get("accuracy")
            acc_str = f"{acc:.3f}" if acc is not None else "—"
            lines.append(
                f"| {h_id} ({lbl}) | {opset} | {s_p50_str} | {f_p50_str} | {cv_str} | {status} | {acc_str} |"
            )
        lines.append("")

    # Cross-model patterns
    lines += [
        "---",
        "",
        "## Cross-Model Patterns",
        "",
        "### npu-001: Does opset 21 bypass help broadly?",
        "",
    ]

    npu001_map = {r["model_id"]: r.get("npu001_generalized") for r in all_results}
    yes_m = [m for m, v in npu001_map.items() if v is True]
    no_m = [m for m, v in npu001_map.items() if v is False]
    neut_m = [m for m, v in npu001_map.items() if v == "neutral"]
    na_m = [m for m, v in npu001_map.items() if v not in (True, False, "neutral")]

    lines += [
        f"- **Helps ({len(yes_m)} models):** {', '.join(f'`{m}`' for m in yes_m) or 'none'}",
        f"- **Hurts ({len(no_m)} models):** {', '.join(f'`{m}`' for m in no_m) or 'none'}",
        f"- **Neutral ({len(neut_m)} models):** {', '.join(f'`{m}`' for m in neut_m) or 'none'}",
        f"- **N/A ({len(na_m)} models):** {', '.join(f'`{m}`' for m in na_m) or 'none'}",
        "",
    ]

    total_tested = len(yes_m) + len(no_m) + len(neut_m)
    if total_tested > 0:
        if len(yes_m) > total_tested / 2:
            lines.append(
                f"> **Finding**: opset 21 bypass generalizes to {len(yes_m)}/{total_tested} tested models."
                " Consider upgrading npu-001 scope from ConvNext-only to broader architectures."
            )
        elif len(no_m) > total_tested / 2:
            lines.append(
                f"> **Finding**: opset 21 bypass does NOT broadly generalize ({len(no_m)}/{total_tested} hurt)."
                " npu-001 appears ConvNext-specific (residual connection topology dependency confirmed)."
            )
        else:
            lines.append(
                f"> **Finding**: Mixed results ({len(yes_m)} help, {len(no_m)} hurt, {len(neut_m)} neutral)."
                " Architecture-dependent. Confirm ORT `kMaxSupportedOpset` version before drawing conclusions."
            )
        lines.append("")

    lines += [
        "### Feature Gaps",
        "",
    ]
    all_gaps: list[str] = []
    for r in all_results:
        for gap in r.get("feature_gaps", []):
            all_gaps.append(f"- **`{r['model_id']}`**: {gap}")
    lines += all_gaps if all_gaps else ["- No feature gaps observed"]

    lines += [
        "",
        "### Build / Compatibility Issues",
        "",
    ]
    for r in all_results:
        errs = r.get("errors", [])
        if errs:
            lines.append(f"**`{r['model_id']}`**")
            for e in errs:
                lines.append(f"  - {e}")

    lines += [
        "",
        "---",
        "",
        "## Updated Recommendations for `ep_knowledge/qnn_npu.json`",
        "",
        "Based on this cross-architecture sweep:",
        "",
    ]

    # Auto-generate KB recommendations
    if total_tested > 0:
        if len(yes_m) >= 2:
            lines += [
                "- **npu-001**: Broaden scope beyond ConvNext. Architectures that benefit: "
                f"{', '.join(yes_m)}. Update `scope` field and set `gate1_statistical` confidence accordingly.",
                "- **search_space_rules.opset.recommended_order**: Retain `[21, 17]` as default order.",
            ]
        if len(no_m) >= 2:
            lines += [
                "- **npu-001**: Keep 'architecture-specific' caveat. Architectures where opset 21 hurts: "
                f"{', '.join(no_m)}. Add to `do_not_generalize_to` list.",
                "- **search_space_rules**: Add architecture check before applying opset 21 preference.",
            ]

    # Conv fusions analysis
    lines += [
        "",
        "### Conv Fusion Findings (h4 vs h1, h5 vs h3)",
        "",
    ]
    for r in all_results:
        h1_p50 = r.get("hypotheses", {}).get("h1", {}).get("full", {}).get("median_p50_ms")
        h4_p50 = r.get("hypotheses", {}).get("h4", {}).get("full", {}).get("median_p50_ms")
        h3_p50 = r.get("hypotheses", {}).get("h3", {}).get("full", {}).get("median_p50_ms")
        h5_p50 = r.get("hypotheses", {}).get("h5", {}).get("full", {}).get("median_p50_ms")
        parts = []
        if h1_p50 and h4_p50:
            delta = (h1_p50 - h4_p50) / h1_p50 * 100
            parts.append(f"conv-fusions on opset17: {delta:+.1f}% ({h1_p50:.1f}→{h4_p50:.1f}ms)")
        if h3_p50 and h5_p50:
            delta = (h3_p50 - h5_p50) / h3_p50 * 100
            parts.append(f"conv-fusions on opset21: {delta:+.1f}% ({h3_p50:.1f}→{h5_p50:.1f}ms)")
        if parts:
            lines.append(f"- **`{r['model_id']}`**: {'; '.join(parts)}")

    summary_path = RESULTS_DIR / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📄 Summary: {summary_path}", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QNN NPU optimization hypothesis sweep for winml catalog models"
    )
    parser.add_argument(
        "--model", default=None, help="Single HF model ID to sweep (default: all catalog models)"
    )
    parser.add_argument(
        "--task", default=None, help="Task override (required when --model is given)"
    )
    parser.add_argument(
        "--model-type", default="auto", help="Model type hint (e.g. resnet, vit). Default: auto"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip winml eval accuracy step even for image models",
    )
    parser.add_argument(
        "--only-hypotheses",
        default=None,
        help=(
            "Comma-separated list of hypothesis IDs to run, e.g. h6,h7,h8. "
            "Skips all others. Use with --reuse-h0-config to avoid regenerating base config."
        ),
    )
    parser.add_argument(
        "--reuse-h0-config",
        action="store_true",
        help=(
            "Reuse the base config from an existing h0/build_config.json instead of "
            "running winml config again. Requires a previous full sweep to have run."
        ),
    )
    args = parser.parse_args()

    # Parse hypothesis filter
    only_hyp_ids: set[str] | None = None
    if args.only_hypotheses:
        only_hyp_ids = {h.strip() for h in args.only_hypotheses.split(",")}
        print(f"  Running only: {sorted(only_hyp_ids)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confirm QNN EP is present
    print("=== Confirming QNN EP ===", flush=True)
    rc, out, _ = run_cmd([WINML, "sys", "--list-ep"], label="winml sys --list-ep", timeout=30)
    if "qnn" not in out.lower():
        print("❌ QNN EP not detected! Aborting.", flush=True)
        sys.exit(1)
    print("✓ QNN EP available\n", flush=True)

    # Determine model list
    if args.model:
        if not args.task:
            print("Error: --task is required when --model is specified", flush=True)
            sys.exit(1)
        models_to_run: list[tuple[str, str, str, bool]] = [
            (args.model, args.task, args.model_type, not args.skip_eval)
        ]
    else:
        models_to_run = ALL_MODELS  # type: ignore[assignment]

    all_results: list[dict] = []

    for model_id, task, model_type, do_eval in models_to_run:
        if args.skip_eval:
            do_eval = False
        try:
            result = sweep_model(
                model_id,
                task,
                model_type,
                do_eval,
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

        # Save rolling summary after each model
        write_summary(all_results)

    print("\n" + "=" * 64, flush=True)
    print("  SWEEP COMPLETE", flush=True)
    print("=" * 64, flush=True)
    write_summary(all_results)


if __name__ == "__main__":
    main()
