# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""catalog_cpu_sweep.py — WinML CPU EP optimization sweep across catalog + recipe models.

Sweeps graph-optimization flags for CPU EP to find improvement opportunities beyond
autoconf defaults. Based on patterns detected by analyze_insight.py (30+ fusion candidates).

Key CPU constraints from ep_knowledge/cpu.json:
  cpu-001: opset 19+ REGRESSES on CPU (3-4x slowdown, Transpose Optimizer bypass)
           → h3/h4 included deliberately to test on transformer models (cpu-001 was ConvNext only)
  cpu-002: matmul_add_fusion REGRESSES if model already has Gemm ops
           → guarded by Gemm check before applying
  cpu-003: transpose_optimizer is neutral on ConvNext (may help transformers)
  cpu-004: nchwc_transformer neutral on Gemm-heavy models
  cpu-005: baseline is optimal for ConvNext — transformers untested

Phase A: 200-iter screen, CV < 10% required (CPU is thermally stable).
Phase B: 3 sessions × 300 iters, 2s cool-down.
Phase C (confirmation): best hypothesis + 2 extra sessions.
  All 5 p50s < baseline_min → CONFIRMED.
KEEP criterion: median p50 >= 5% improvement.

Results: catalog-cpu-sweep/<model_slug>/results.json
Summary: catalog-cpu-sweep/SUMMARY.md
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
EP = "cpu"
DEVICE = "cpu"
RESULTS_DIR = BASE_DIR / "catalog-cpu-sweep"
CHAMPION_DIR = BASE_DIR / "champion-configs"

SCREEN_WARMUP = 10
SCREEN_ITERS = 200
SCREEN_CV_MAX = 0.10  # CPU is stable — stricter than QNN

FULL_WARMUP = 10
FULL_ITERS = 300
FULL_SESSIONS = 3
CONFIRM_SESSIONS = 2  # Phase C: extra sessions for best hypothesis
COOL_DOWN_S = 2  # CPU cools quickly

MIN_IMPROVEMENT_PCT = 5.0

BUILD_TIMEOUT_S = 10 * 60
BENCH_TIMEOUT_S = 8 * 60

# Gemm threshold: if model has Gemm ops, skip matmul_add_fusion (cpu-002)
GEMM_SAFE_MATMUL_ADD = False  # Conservative default; overridden per model

# Hypotheses: (id, label, opset_override, extra_optim, skip_if_gemm)
# skip_if_gemm=True → skip if model.onnx already contains Gemm nodes (cpu-002 guard)
HYPOTHESES: list[tuple[str, str, int | None, dict | None, bool]] = [
    # ── Opset variants ─────────────────────────────────────────────────────
    ("h0", "baseline (opset 17, autoconf defaults)", None, None, False),
    ("h1", "opset 17 explicit", 17, None, False),
    # cpu-001: opset 19/21 KNOWN to regress on ConvNext — included to test transformers
    ("h2", "opset 19 (cpu-001 risk — transformer test)", 19, None, False),
    ("h3", "opset 21 (cpu-001 risk — transformer test)", 21, None, False),
    # ── Transformer fusions (graph-analysis-driven) ────────────────────────
    ("h4", "opset 17 + attention_fusion", 17, {"attention_fusion": True}, False),
    ("h5", "opset 17 + skip_layer_norm_fusion", 17, {"skip_layer_norm_fusion": True}, False),
    ("h6", "opset 17 + layer_norm_fusion", 17, {"layer_norm_fusion": True}, False),
    ("h7", "opset 17 + bias_softmax_fusion", 17, {"bias_softmax_fusion": True}, False),
    # ── MatMul fusions ─────────────────────────────────────────────────────
    # matmul_add_fusion: skip if Gemm already present (cpu-002)
    ("h8", "opset 17 + matmul_add_fusion (cpu-002 guarded)", 17, {"matmul_add_fusion": True}, True),
    ("h9", "opset 17 + matmul_transpose_fusion", 17, {"matmul_transpose_fusion": True}, False),
    # ── Transformer bundle (best flags combined) ───────────────────────────
    (
        "h10",
        "opset 17 + attention + skip_layer_norm + layer_norm",
        17,
        {"attention_fusion": True, "skip_layer_norm_fusion": True, "layer_norm_fusion": True},
        False,
    ),
    # ── Conv / layout (vision models) ─────────────────────────────────────
    # nchwc_transformer: neutral on Gemm-heavy models (cpu-004), may help Conv-heavy
    (
        "h11",
        "opset 17 + nchwc_transformer (Conv-heavy models)",
        17,
        {"nchwc_transformer": True},
        False,
    ),
    # ── Misc ───────────────────────────────────────────────────────────────
    ("h12", "opset 17 + transpose_optimizer", 17, {"transpose_optimizer": True}, False),
    ("h13", "opset 17 + gelu_fusion explicit", 17, {"gelu_fusion": True}, False),
]

# Catalog + recipe models (task, model_type)
ALL_MODELS = [
    ("microsoft/resnet-18", "image-classification", "resnet"),
    ("apple/mobilevit-small", "image-classification", "mobilevit"),
    ("facebook/dinov2-small", "image-feature-extraction", "dinov2"),
    ("deepset/roberta-base-squad2", "question-answering", "roberta"),
    ("deepset/tinyroberta-squad2", "question-answering", "roberta"),
    ("BAAI/bge-small-en-v1.5", "sentence-similarity", "bert"),
    ("sentence-transformers/all-MiniLM-L6-v2", "sentence-similarity", "bert"),
    ("microsoft/rad-dino", "image-feature-extraction", "dinov2"),
]


# ── subprocess helpers ────────────────────────────────────────────────────────


def run_cmd(cmd: list[str], label: str = "", timeout: int = 300) -> tuple[int, str, float]:
    t0 = time.monotonic()
    print(f"  >> {label or ' '.join(cmd[:3])}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.monotonic() - t0
        ok = "ok" if result.returncode == 0 else f"rc={result.returncode}"
        print(f"     {elapsed:.0f}s [{ok}]", flush=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f"     stderr: {stderr[:200]}", flush=True)
        return result.returncode, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        print(f"     TIMEOUT after {timeout}s", flush=True)
        return -1, "TIMEOUT", timeout


def _get_p50(perf_json: Path) -> float | None:
    try:
        d = json.loads(perf_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", d)
        return float(lat.get("p50") or 0) or None
    except Exception:
        return None


def _get_cv(perf_json: Path) -> float | None:
    try:
        d = json.loads(perf_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", d)
        p50 = float(lat.get("p50") or 0)
        std = float(lat.get("std") or 0)
        return std / p50 if p50 > 0 else None
    except Exception:
        return None


# ── config helpers ─────────────────────────────────────────────────────────────


def _patch_for_cpu(cfg: dict) -> dict:
    """Remove quantization and compile from CPU config."""
    cfg = copy.deepcopy(cfg)
    cfg["quant"] = None
    cfg["compile"] = None
    return cfg


def get_base_config(model_id: str, task: str, model_type: str) -> dict | None:
    tmp_dir = RESULTS_DIR / "_tmp_config"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_out = tmp_dir / f"{model_id.replace('/', '--')}_cpu.json"

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
        label=f"winml config --ep {EP}",
        timeout=300,
    )
    if rc != 0 or not cfg_out.exists():
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    return _patch_for_cpu(json.loads(line))
                except Exception:
                    pass
        return None
    return _patch_for_cpu(json.loads(cfg_out.read_text(encoding="utf-8")))


def make_hypothesis_config(
    base_config: dict, opset_override: int | None, extra_optim: dict | None
) -> dict:
    cfg = copy.deepcopy(base_config)
    if opset_override is not None:
        cfg.setdefault("export", {})["opset_version"] = opset_override
    if extra_optim:
        existing = cfg.get("optim") or {}
        cfg["optim"] = {**existing, **extra_optim}
    return cfg


def _model_has_gemm(model_onnx: Path) -> bool:
    """Check if an optimized.onnx has Gemm nodes (cpu-002 guard)."""
    try:
        import onnx

        m = onnx.load(str(model_onnx))
        return any(n.op_type == "Gemm" for n in m.graph.node)
    except Exception:
        return False  # Assume safe if can't check


# ── build + bench ──────────────────────────────────────────────────────────────


def run_build(model_id: str, cfg_path: Path, out_dir: Path) -> tuple[bool, str]:
    """winml build --no-quant --no-compile --rebuild for CPU EP."""
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
    p50s = []
    for s in range(1, FULL_SESSIONS + 1):
        out_json = hyp_dir / f"full_s{s}.json"
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


# ── sweep logic ────────────────────────────────────────────────────────────────


def sweep_model(
    model_id: str,
    task: str,
    model_type: str,
    only_hyp_ids: "set[str] | None" = None,
    reuse_h0_config: bool = False,
) -> dict:
    model_slug = model_id.replace("/", "--")
    model_dir = RESULTS_DIR / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

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
    results.setdefault("hypotheses", {})
    results.setdefault("baseline_p50_ms", None)
    results.setdefault("best_p50_ms", None)
    results.setdefault("best_hypothesis", None)
    results.setdefault("best_gain_pct", None)
    results.setdefault("errors", [])

    print(f"\n{'=' * 64}", flush=True)
    print(f"  SWEEP [CPU]: {model_id}  [{task}]", flush=True)
    if only_hyp_ids:
        print(f"  (delta — only: {sorted(only_hyp_ids)})", flush=True)
    print(f"{'=' * 64}", flush=True)

    # Step 1: base config
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
    print(f"  baseline opset={baseline_opset}  quant=NONE (CPU EP)  compile=NONE", flush=True)

    # Step 2: hypothesis loop
    print(f"\n[2/3] Running {len(HYPOTHESES)} hypotheses…", flush=True)

    baseline_p50: float | None = results.get("baseline_p50_ms")
    model_has_gemm: bool | None = None  # lazy-loaded for cpu-002 guard

    for hyp_id, label, opset_override, extra_optim, skip_if_gemm in HYPOTHESES:
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

        # cpu-002 guard: skip matmul_add_fusion if model already has Gemm
        if skip_if_gemm:
            if model_has_gemm is None:
                opt_onnx = hyp_dir / "optimized.onnx"
                model_has_gemm = _model_has_gemm(opt_onnx) if opt_onnx.exists() else False
            if model_has_gemm:
                print(
                    f"  [cpu-002] SKIP {hyp_id}: model has Gemm nodes — matmul_add_fusion likely harmful",
                    flush=True,
                )
                results["hypotheses"][hyp_id] = {
                    "status": "SKIPPED_CPU002",
                    "label": label,
                    "opset": opset_used,
                    "reason": "cpu-002: model already has Gemm — matmul_add_fusion skipped",
                }
                continue

        # Annotate cpu-001 risk
        if opset_override is not None and opset_override >= 19:
            print(
                f"  [cpu-001] NOTE: opset={opset_override} may regress on Conv-heavy models"
                f" (cpu-001 validated on ConvNext only — testing transformer behavior)",
                flush=True,
            )

        # Phase A: screen
        screen_json = hyp_dir / "screen_perf.json"
        screen_p50, screen_cv = run_perf_screen(onnx_path, screen_json)

        if screen_p50 is None:
            results["hypotheses"][hyp_id] = {"status": "BENCH_FAIL", "label": label}
            results["errors"].append(f"{hyp_id}: screen bench failed")
            continue

        if screen_cv is not None and screen_cv > SCREEN_CV_MAX:
            print(f"  [warn] high CV={screen_cv:.3f} on CPU (unusual) — proceeding", flush=True)

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

        if hyp_id == "h0":
            baseline_p50 = median_p50
            results["baseline_p50_ms"] = baseline_p50
            print(f"  [baseline] p50={baseline_p50:.2f}ms", flush=True)

        if baseline_p50 and hyp_id != "h0":
            gain_pct = (baseline_p50 - median_p50) / baseline_p50 * 100
            hyp_data["gain_vs_baseline_pct"] = round(gain_pct, 2)
            verdict = (
                "KEEP"
                if gain_pct >= MIN_IMPROVEMENT_PCT
                else ("MARGINAL" if gain_pct > 0 else "DISCARD")
            )
            # cpu-001: flag known-regression hypotheses specially
            if opset_override is not None and opset_override >= 19 and gain_pct <= -50:
                verdict = "CPU001_REGRESSION"
            hyp_data["verdict"] = verdict
            print(
                f"  [{verdict}] gain={gain_pct:+.1f}% ({baseline_p50:.2f}ms → {median_p50:.2f}ms)",
                flush=True,
            )

            best_p50 = results.get("best_p50_ms")
            if best_p50 is None or median_p50 < best_p50:
                if gain_pct >= MIN_IMPROVEMENT_PCT:
                    results["best_p50_ms"] = median_p50
                    results["best_hypothesis"] = hyp_id
                    results["best_gain_pct"] = round(gain_pct, 2)
        else:
            hyp_data["verdict"] = "BASELINE"

        results["hypotheses"][hyp_id] = hyp_data

    # Step 2b: Phase C confirmation
    _run_confirmation_pass(results, model_dir, baseline_p50)

    # Step 3: finalise
    _post_process(results)
    _save_results(results, model_dir)
    _emit_model_artifacts(results, model_dir)
    return results


def _run_confirmation_pass(results: dict, model_dir: Path, baseline_p50: float | None) -> None:
    """Phase C: CONFIRM_SESSIONS extra sessions for best hypothesis."""
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
        for candidate in (hyp_dir / "model.onnx", hyp_dir / "optimized.onnx"):
            if candidate.exists():
                onnx_path = candidate
                break
        if onnx_path is None:
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
            continue

        all_p50s = hyp_data.get("full_p50s_ms", []) + extra_p50s
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
                f" overall={overall_gain:+.1f}%",
                flush=True,
            )
        else:
            hyp_data["verdict"] = "MARGINAL_UNCONFIRMED"
            print(
                f"  [MARGINAL_UNCONFIRMED] {hyp_id}: only {wins}/{len(all_p50s)} sessions above threshold",
                flush=True,
            )

        if hyp_data["verdict"] == "KEEP_CONFIRMED":
            best_p50 = results.get("best_p50_ms")
            if best_p50 is None or overall_median < best_p50:
                results["best_p50_ms"] = overall_median
                results["best_hypothesis"] = hyp_id
                results["best_gain_pct"] = round(overall_gain, 2)


def _post_process(results: dict) -> None:
    hyps = results.get("hypotheses", {})
    baseline_p50 = results.get("baseline_p50_ms")
    if not baseline_p50:
        return

    keeps = [(hid, h) for hid, h in hyps.items() if h.get("verdict") in ("KEEP", "KEEP_CONFIRMED")]
    unconfirmed = [
        (hid, h) for hid, h in hyps.items() if h.get("verdict") == "MARGINAL_UNCONFIRMED"
    ]
    regressions = [(hid, h) for hid, h in hyps.items() if h.get("verdict") == "CPU001_REGRESSION"]

    if keeps:
        print(f"\n  ✓ KEEP/KEEP_CONFIRMED: {[h[0] for h in keeps]}", flush=True)
    if unconfirmed:
        print(f"  ⚠ MARGINAL_UNCONFIRMED: {[h[0] for h in unconfirmed]}", flush=True)
    if regressions:
        print(f"  ✗ CPU001_REGRESSION: {[h[0] for h in regressions]}", flush=True)
    if not keeps and not unconfirmed and not regressions:
        print("\n  No improvements found above 5% threshold.", flush=True)

    # Cross-architecture cpu-001 check: does opset 19/21 regress on THIS model?
    for hid in ("h2", "h3"):
        h = hyps.get(hid, {})
        if h.get("status") == "OK" and baseline_p50:
            gain = h.get("gain_vs_baseline_pct", 0.0)
            if gain < -50:
                print(
                    f"  [cpu-001] CONFIRMED regression on {hid} for this architecture: {gain:.1f}%",
                    flush=True,
                )
            elif gain > -10:
                print(
                    f"  [cpu-001] NOT OBSERVED on {hid} for {results.get('model_type')} — "
                    f"gain={gain:+.1f}% (ConvNext-specific?)",
                    flush=True,
                )


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
        "# CPU EP Optimization Sweep — Catalog Models",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"EP: `{EP}` / device: `{DEVICE}`  ",
        f"Protocol: screen {SCREEN_ITERS} iters (CV<{SCREEN_CV_MAX * 100:.0f}%),"
        f" full {FULL_ITERS}×{FULL_SESSIONS} sessions"
        f" + {CONFIRM_SESSIONS} confirm sessions for KEEP  ",
        "Constraints: NO quant, NO compile  ",
        "",
        "---",
        "",
        "## cpu-001 Check: Does opset 19/21 Regress on Non-ConvNext Models?",
        "",
        "| Model | type | h2(opset19) gain% | h3(opset21) gain% | cpu-001 fires? |",
        "|-------|------|-------------------|-------------------|---------------|",
    ]

    for r in all_results:
        model_id = r.get("model_id", "?")
        mtype = r.get("model_type", "?")
        h2 = r.get("hypotheses", {}).get("h2", {})
        h3 = r.get("hypotheses", {}).get("h3", {})
        g2 = (
            f"{h2.get('gain_vs_baseline_pct', 'N/A'):+.1f}%"
            if h2.get("gain_vs_baseline_pct") is not None
            else h2.get("status", "N/A")
        )
        g3 = (
            f"{h3.get('gain_vs_baseline_pct', 'N/A'):+.1f}%"
            if h3.get("gain_vs_baseline_pct") is not None
            else h3.get("status", "N/A")
        )
        fires = (
            "YES ≤-50%"
            if any(
                r.get("hypotheses", {}).get(h, {}).get("gain_vs_baseline_pct", 0) <= -50
                for h in ("h2", "h3")
            )
            else "no"
        )
        lines.append(f"| `{model_id}` | {mtype} | {g2} | {g3} | {fires} |")

    lines += [
        "",
        "## Per-Model Results",
        "",
        "| Model | Baseline p50 | Best p50 | Best config | Gain% | Notes |",
        "|-------|-------------|----------|-------------|-------|-------|",
    ]

    for r in all_results:
        model_id = r.get("model_id", "?")
        baseline = f"{r['baseline_p50_ms']:.1f} ms" if r.get("baseline_p50_ms") else "N/A"
        best = f"{r['best_p50_ms']:.1f} ms" if r.get("best_p50_ms") else "N/A"
        best_h = r.get("best_hypothesis") or "N/A"
        best_label = ""
        if best_h != "N/A":
            best_label = r.get("hypotheses", {}).get(best_h, {}).get("label", "")
        gain = f"{r['best_gain_pct']:.1f}%" if r.get("best_gain_pct") is not None else "N/A"
        errors = "; ".join(r.get("errors", []))[:80] or "none"
        lines.append(
            f"| `{model_id}` | {baseline} | {best} | {best_h} ({best_label}) | {gain} | {errors} |"
        )

    summary_path = RESULTS_DIR / "SUMMARY.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n📄 Summary: {summary_path}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU EP sweep across catalog models")
    parser.add_argument("--model", help="Run a single model (HuggingFace model ID)")
    parser.add_argument("--task", help="Task for single model run")
    parser.add_argument("--model-type", dest="model_type", help="Model type for single model run")
    parser.add_argument(
        "--only-hypotheses",
        dest="only_hyp",
        help="Comma-separated list of hypothesis IDs to run (e.g. h4,h5,h10)",
    )
    parser.add_argument(
        "--reuse-h0-config",
        dest="reuse_h0",
        action="store_true",
        help="Load base config from existing h0/build_config.json",
    )
    args = parser.parse_args()

    only_hyp_ids = set(args.only_hyp.split(",")) if args.only_hyp else None

    all_results = []

    if args.model:
        if not args.task or not args.model_type:
            print("ERROR: --task and --model-type required with --model", file=sys.stderr)
            sys.exit(1)
        r = sweep_model(
            args.model,
            args.task,
            args.model_type,
            only_hyp_ids=only_hyp_ids,
            reuse_h0_config=args.reuse_h0,
        )
        all_results.append(r)
    else:
        for model_id, task, model_type in ALL_MODELS:
            r = sweep_model(
                model_id,
                task,
                model_type,
                only_hyp_ids=only_hyp_ids,
                reuse_h0_config=args.reuse_h0,
            )
            all_results.append(r)

    write_summary(all_results)
    print("\n================================================================", flush=True)
    print("  CPU SWEEP COMPLETE", flush=True)
    print("================================================================", flush=True)
    print(f"\n📄 Summary: {RESULTS_DIR / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
