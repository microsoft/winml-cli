#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
validation_sweep.py — Focused validation sweep for npu-001 and npu-006.

Tests:
  npu-001: opset17 vs opset21 speedup on Conv+attention hybrid vs pure ViT
  npu-006: conv fusions regression — confirm MobileViT/DINOv2 are unaffected

Hypotheses (subset of catalog_qnn_sweep.py):
  h0: baseline (auto-config, W8A16)
  h1: opset 17 explicit
  h3: opset 21  ← npu-001 test
  h4: opset 17 + conv fusions  ← npu-006 test

Models:
  facebook/dinov2-base      → expect npu-001 speedup (larger DINOv2)
  microsoft/rad-dino        → expect npu-001 speedup (DINOv2 variant)
  facebook/dino-vitb16      → expect NEUTRAL (pure DINO ViT, no Conv+residual)
  Intel/dpt-hybrid-midas    → expect npu-001 speedup; npu-006 regression (ResNet backbone)

Output: research/autoconfig/catalog-qnn-sweep/<model-slug>/results_v2.json
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent  # research/autoconfig/ → research/ → repo root
WINML = str(REPO_ROOT / ".venv" / "Scripts" / "winml.exe")
EP = "qnn"
DEVICE = "npu"
RESULTS_DIR = BASE_DIR / "catalog-qnn-sweep"

SCREEN_WARMUP = 20
SCREEN_ITERS = 200

FULL_WARMUP = 50
FULL_ITERS = 500
FULL_SESSIONS = 3
COOL_DOWN_S = 30

MODEL_TIMEOUT_S = (
    120 * 60
)  # 120 min per model (rad-dino/large models: 450s per bench session × 3 × 3)
BUILD_TIMEOUT_S = 15 * 60
BENCH_TIMEOUT_S = 15 * 60
EVAL_TIMEOUT_S = 6 * 60

# Focused hypothesis matrix
HYPOTHESES = [
    ("h0", "baseline (auto-config, W8A16)", None, None),
    ("h1", "opset 17 explicit", 17, None),
    ("h3", "opset 21 (tests npu-001)", 21, None),
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
]

# (model_id, task, model_type, run_h4_fusion_test)
VALIDATION_MODELS = [
    ("facebook/dinov2-base", "image-feature-extraction", "dinov2", True),
    ("microsoft/rad-dino", "image-feature-extraction", "dinov2", False),
    ("facebook/dino-vitb16", "image-feature-extraction", "vit", True),
    ("Intel/dpt-hybrid-midas", "depth-estimation", "dpt", True),
]


def run_cmd(cmd, label="", timeout=600):
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
            print(f"     stderr: {(result.stderr or result.stdout or '')[-400:]}", flush=True)
        return result.returncode, result.stdout + result.stderr, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"     TIMEOUT after {elapsed:.0f}s", flush=True)
        return -999, f"TIMEOUT after {timeout}s", elapsed


def get_base_config(model_id, task, model_type):
    tmp = RESULTS_DIR / "_tmp_val_cfg.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    def _try(extra):
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
            str(tmp),
        ] + extra
        rc, _, _ = run_cmd(cmd, "winml config", 600)
        if rc == 0 and tmp.exists():
            try:
                cfg = json.loads(tmp.read_text(encoding="utf-8"))
                tmp.unlink(missing_ok=True)
                return cfg
            except Exception:
                pass
        tmp.unlink(missing_ok=True)
        return None

    cfg = _try(["--model-type", model_type])
    if cfg is None:
        print("  [warn] retrying without --model-type", flush=True)
        cfg = _try([])
    return cfg


def make_hyp_config(base, opset_override, extra_optim):
    cfg = copy.deepcopy(base)
    if opset_override is not None and cfg.get("export"):
        cfg["export"]["opset_version"] = opset_override
    if extra_optim is not None:
        cfg["optim"] = {**(cfg.get("optim") or {}), **extra_optim}
    return cfg


def run_build(model_id, cfg_path, out_dir):
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
    rc, out, _ = run_cmd(cmd, f"winml build [{out_dir.name}]", BUILD_TIMEOUT_S)
    return rc == 0, out


def bench_screen(model_path):
    out_json = model_path.parent / "val_screen.json"
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
        f"perf screen ({SCREEN_ITERS} iters)",
        BENCH_TIMEOUT_S,
    )
    if rc != 0 or not out_json.exists():
        return None, 999.0, False
    try:
        d = json.loads(out_json.read_text(encoding="utf-8"))
        lat = d.get("latency_ms", {})
        p50 = lat.get("p50") if isinstance(lat, dict) else None
        std = lat.get("std", 0) if isinstance(lat, dict) else 0
        if not p50:
            return None, 999.0, False
        cv = std / p50
        stable = cv < 0.15
        return p50, cv, stable
    except Exception:
        return None, 999.0, False


def bench_full(model_path):
    p50s = []
    for s in range(FULL_SESSIONS):
        if s > 0:
            print(f"  [cool-down {COOL_DOWN_S}s]", flush=True)
            time.sleep(COOL_DOWN_S)
        out_json = model_path.parent / f"val_full_s{s}.json"
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
            f"perf full s{s} ({FULL_ITERS} iters)",
            BENCH_TIMEOUT_S,
        )
        if rc != 0 or not out_json.exists():
            continue
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            lat = d.get("latency_ms", {})
            p50 = lat.get("p50") if isinstance(lat, dict) else None
            if p50:
                p50s.append(round(p50, 3))
        except Exception:
            pass
    if not p50s:
        return None, None
    median = sorted(p50s)[len(p50s) // 2]
    return p50s, round(median, 3)


def run_model(model_id, task, model_type, run_h4):
    slug = model_id.replace("/", "--")
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Model: {model_id}", flush=True)
    print("  Hypotheses: h0, h1, h3" + (", h4" if run_h4 else ""), flush=True)
    print(f"{'=' * 60}", flush=True)

    out_dir = RESULTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model_id": model_id,
        "task": task,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ep": EP,
        "device": DEVICE,
        "validation_sweep": True,
        "hypotheses": {},
        "errors": [],
    }

    base_cfg = get_base_config(model_id, task, model_type)
    if base_cfg is None:
        result["errors"].append("FAILED: could not generate base config")
        (out_dir / "results_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    t0_model = time.time()

    active_hyps = [
        (hid, lbl, opset, optim)
        for hid, lbl, opset, optim in HYPOTHESES
        if hid in ("h0", "h1", "h3") or (run_h4 and hid == "h4")
    ]

    for hid, label, opset_override, extra_optim in active_hyps:
        elapsed_model = time.time() - t0_model
        if elapsed_model > MODEL_TIMEOUT_S:
            result["errors"].append(f"Model timed out at {elapsed_model:.0f}s (before {hid})")
            result["hypotheses"][hid] = {"status": "TIMEOUT", "label": label}
            continue

        print(f"\n  --- {hid}: {label} ---", flush=True)
        hyp_dir = out_dir / f"val_{hid}"
        hyp_dir.mkdir(parents=True, exist_ok=True)

        cfg = make_hyp_config(base_cfg, opset_override, extra_optim)
        cfg_path = hyp_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        # Reuse existing build output if already present (avoids re-downloading)
        # Require optimized.onnx or quantized.onnx as completion signal — export.onnx alone
        # means the build was truncated before optimization/quantization finished.
        complete_models = [
            f for f in hyp_dir.glob("*.onnx") if "optimized" in f.name or "quantized" in f.name
        ]
        if complete_models:
            print(f"  [reuse] existing build in {hyp_dir.name}", flush=True)
            ok = True
            build_out = "(reused)"
        else:
            ok, build_out = run_build(model_id, cfg_path, hyp_dir)
        if not ok:
            result["hypotheses"][hid] = {
                "status": "BUILD_FAIL",
                "label": label,
                "build_error": build_out[-300:],
            }
            result["errors"].append(f"{hid}: BUILD_FAIL")
            continue

        # find model file — prefer quantized > optimized > any
        model_files = list(hyp_dir.glob("*.onnx"))
        model_path = next((f for f in model_files if "quantized" in f.name), None)
        if model_path is None:
            model_path = next((f for f in model_files if "optimized" in f.name), None)
        if model_path is None and model_files:
            model_path = model_files[0]
        if model_path is None:
            result["hypotheses"][hid] = {
                "status": "BUILD_FAIL",
                "label": label,
                "build_error": "no .onnx found",
            }
            continue

        p50_screen, cv, stable = bench_screen(model_path)
        if p50_screen is None:
            result["hypotheses"][hid] = {
                "status": "BENCH_FAIL",
                "label": label,
                "opset": opset_override or "auto",
            }
            continue

        p50s, median = bench_full(model_path)
        status = "OK" if cv < 0.15 else "OK_HIGH_CV"
        result["hypotheses"][hid] = {
            "status": status,
            "screen": {
                "p50_ms": round(p50_screen, 3),
                "cv": round(cv, 4),
                "stable": stable,
                "note": "DVFS noise — high CV expected on QNN NPU" if not stable else None,
            },
            "full": {"p50s_ms": p50s, "median_p50_ms": median},
            "label": label,
            "opset": opset_override or "auto",
        }
        print(
            f"  [RESULT {hid}] screen p50={p50_screen:.2f}ms CV={cv:.3f}  full_median={median}ms  sessions={p50s}",
            flush=True,
        )

    # Compute npu-001 signal
    h1 = result["hypotheses"].get("h1", {})
    h3 = result["hypotheses"].get("h3", {})
    if h1.get("full") and h3.get("full"):
        m1 = h1["full"]["median_p50_ms"]
        m3 = h3["full"]["median_p50_ms"]
        if m1 and m3:
            gain = round((m1 - m3) / m1 * 100, 1)
            result["npu001_opset21_vs_17_gain_pct"] = gain
            result["npu001_note"] = f"opset21 median {m3}ms vs opset17 {m1}ms = {gain:+.1f}%"

    # Compute npu-006 signal
    h4 = result["hypotheses"].get("h4", {})
    if h1.get("full") and h4.get("full"):
        m1 = h1["full"]["median_p50_ms"]
        m4 = h4["full"]["median_p50_ms"]
        if m1 and m4:
            regression = round((m4 - m1) / m1 * 100, 1)
            result["npu006_conv_fusion_regression_pct"] = regression
            result["npu006_note"] = (
                f"conv fusions median {m4}ms vs no-fusion {m1}ms = {regression:+.1f}%"
            )

    out_path = out_dir / "results_v2.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [SAVED] {out_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Focused npu-001/npu-006 validation sweep")
    parser.add_argument("--model", help="Run single model by ID")
    parser.add_argument(
        "--no-h4", action="store_true", help="Skip h4 (conv fusions) for all models"
    )
    args = parser.parse_args()

    models = VALIDATION_MODELS
    if args.model:
        models = [
            (m, t, tp, h4)
            for m, t, tp, h4 in VALIDATION_MODELS
            if m == args.model or m.split("/")[-1] == args.model
        ]
        if not models:
            print(f"Model '{args.model}' not in validation list. Available:")
            for m, t, tp, h4 in VALIDATION_MODELS:
                print(f"  {m}  ({t}, {tp})")
            sys.exit(1)

    print(f"\nValidation sweep — {len(models)} model(s)", flush=True)
    print(
        f"EP: {EP} / {DEVICE}  Proto: {FULL_SESSIONS}×{FULL_ITERS} iters, {COOL_DOWN_S}s cool-down\n",
        flush=True,
    )

    all_results = []
    for model_id, task, model_type, run_h4 in models:
        if args.no_h4:
            run_h4 = False
        res = run_model(model_id, task, model_type, run_h4)
        all_results.append(res)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for r in all_results:
        mid = r["model_id"]
        npu001 = r.get("npu001_note", "n/a")
        npu006 = r.get("npu006_note", "")
        print(f"  {mid}")
        print(f"    npu-001: {npu001}")
        if npu006:
            print(f"    npu-006: {npu006}")
        if r.get("errors"):
            print(f"    errors: {r['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
