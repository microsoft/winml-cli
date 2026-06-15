#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""autoconfig.py — AutoResearch-style optimize-pass search for winml-cli
Demo: facebook/convnext-tiny-224, CPU EP, FP32

Loop: hypothesize → winml build → quick-screen bench (CV gate) →
      full bench (iter=1000×3) → eval → keep/discard → repeat

Key design principles (from GPU Optimizer V2 + ConvNext lessons):
  1. Two-phase bench: 200-iter CV screen FIRST, full bench only if CV < 10%
  2. Use winml perf (NOT winml eval) for latency — eval includes HF preprocessing
  3. Mandatory external-research after 5 consecutive DISCARDs in same dimension
  4. Load ep_knowledge/*.json (only "confirmed" entries) to prune search space
  5. Per-experiment structured output: hypothesis/impl/parity/perf/analysis/decision
  6. Stop condition: 30 consecutive DISCARDs (not 5)
"""

import copy
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── settings ─────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/convnext-tiny-224"
TASK = "image-classification"
EP = "cpu"
DEVICE = "cpu"
WINML = str(Path(__file__).parent / ".venv" / "Scripts" / "winml.exe")
WORK_DIR = Path(__file__).parent / "convnext-search"
RESULTS_TSV = WORK_DIR / "results.tsv"
KB_DIR = Path(__file__).parent / "ep_knowledge"

EVAL_SAMPLES = 50  # for accuracy gate
ACCURACY_FLOOR = 0.70  # cosine drop below this → discard
MIN_IMPROVEMENT = 0.01  # require ≥1% p50 improvement to KEEP

# Bench protocol (two-phase, from GPU Optimizer V2)
SCREEN_WARMUP = 20
SCREEN_ITERS = 200
SCREEN_CV_MAX = 0.10  # Coefficient of Variation = std/p50; reject if > 10%
FULL_WARMUP = 50
FULL_ITERS = 1000
FULL_SESSIONS = 3
COOL_DOWN_S = 60  # seconds between full-bench sessions

# Stop conditions
STOP_CONSECUTIVE_DISCARDS = 30  # plateau stop
EXTERNAL_RESEARCH_TRIGGER = 5  # trigger after this many DISCARDs in same dimension

# ── load ep_knowledge (confirmed entries only) ────────────────────────────────


def load_ep_knowledge(ep: str) -> dict:
    """Load confirmed KB entries for given EP. Only 'confirmed' status entries
    are used to prune search space. 'draft' entries are informational only.
    """
    kb_path = KB_DIR / f"{ep}.json"
    if not kb_path.exists():
        return {"skip_passes": [], "skip_quantization": False, "notes": []}

    kb = json.loads(kb_path.read_text(encoding="utf-8"))
    rules = kb.get("search_space_rules", {})
    skip_passes = []
    skip_quant = False
    notes = []

    # Only apply rules from confirmed findings
    confirmed_ids = {f["id"] for f in kb.get("findings", []) if f.get("mechanism_confirmed", False)}

    for finding in kb.get("findings", []):
        if finding["id"] not in confirmed_ids:
            notes.append(f"[DRAFT] {finding['id']}: {finding['title'][:60]}…")
            continue
        action = finding.get("action_for_autoconfig", "")
        if "skip" in action.lower() and "quantization" in action.lower():
            skip_quant = True
            notes.append(f"[KB confirmed] Skip quantization: {finding['id']}")
        if "skip" in action.lower() and "compile" in action.lower():
            notes.append(f"[KB confirmed] Skip compile: {finding['id']}")

    # Parse search_space_rules for passes to skip
    graph_passes = rules.get("graph_passes", {})
    for p in graph_passes.get("skip", []):
        skip_passes.append(p)
        notes.append(f"[KB confirmed] Skip pass: {p}")

    return {"skip_passes": skip_passes, "skip_quantization": skip_quant, "notes": notes}


# ── baseline config ───────────────────────────────────────────────────────────
BASELINE: dict = {
    "export": {
        "opset_version": 17,
        "batch_size": 1,
        "do_constant_folding": True,
        "dynamo": False,
        "input_tensors": [
            {
                "name": "pixel_values",
                "dtype": "float32",
                "shape": [1, 3, 224, 224],
                "value_range": [0, 1],
            }
        ],
        "output_tensors": [{"name": "logits"}],
    },
    "optim": {},
    "loader": {
        "task": TASK,
        "model_class": "AutoModelForImageClassification",
        "model_type": "convnext",
    },
    "eval": {
        "task": TASK,
        "dataset": {"path": "timm/mini-imagenet", "split": "test", "samples": EVAL_SAMPLES},
    },
}


# ── hypothesis sequence ───────────────────────────────────────────────────────
def h0_baseline(cfg: dict) -> dict:
    """FP32 export, no extra fusions — reference point"""
    cfg["optim"] = {}
    return cfg


def h1_conv_fusions(cfg: dict) -> dict:
    cfg["optim"] = {"conv-bn-fusion": True, "conv-add-fusion": True, "conv-activation-fusion": True}
    return cfg


def h2_gelu_fusion(cfg: dict) -> dict:
    cfg["optim"] = {**cfg["optim"], "gelu-fusion": True}
    return cfg


def h3_add_layernorm(cfg: dict) -> dict:
    cfg["optim"] = {**cfg["optim"], "layer-norm-fusion": True}
    return cfg


def h4_add_matmul(cfg: dict) -> dict:
    cfg["optim"] = {**cfg["optim"], "matmul-add-fusion": True}
    return cfg


def h5_transpose_opt(cfg: dict) -> dict:
    cfg["optim"] = {**cfg["optim"], "transpose-optimizer": True}
    return cfg


def h6_opset21(cfg: dict) -> dict:
    """Try opset 21 — may trigger kMaxSupportedOpset bypass on older ORT (see npu-001).
    NOTE: This is a research hypothesis, not a confirmed optimization. Gate 2 required.
    """
    cfg["export"]["opset_version"] = 21
    cfg["optim"] = {**cfg["optim"], "transpose-optimizer": True}
    return cfg


HYPOTHESES: list[tuple[str, object, str]] = [
    # (label, patch_fn, search_dimension)
    ("baseline: no fusions (FP32 reference)", h0_baseline, "baseline"),
    ("conv fusions: bn+add+activation", h1_conv_fusions, "graph_pass"),
    ("+ gelu-fusion", h2_gelu_fusion, "graph_pass"),
    ("+ layer-norm-fusion", h3_add_layernorm, "graph_pass"),
    ("+ matmul-add-fusion (MLP blocks)", h4_add_matmul, "graph_pass"),
    ("+ transpose-optimizer", h5_transpose_opt, "graph_pass"),
    ("opset=21 (kMaxSupportedOpset research)", h6_opset21, "opset"),
]

# ── helpers ───────────────────────────────────────────────────────────────────


def run(cmd: list[str], label: str = "") -> tuple[int, str, float]:
    t0 = time.time()
    print(f"  >> {label or cmd[1]}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    status = "ok" if result.returncode == 0 else f"rc={result.returncode}"
    print(f"     done in {elapsed:.0f}s  [{status}]")
    if result.returncode != 0:
        print(f"     stderr: {(result.stderr or result.stdout or '')[-400:]}")
    return result.returncode, result.stdout + result.stderr, elapsed


def build(cfg: dict, out_dir: Path) -> tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    rc, out, _ = run(
        [
            WINML,
            "build",
            "-c",
            str(cfg_path),
            "-m",
            MODEL_ID,
            "-o",
            str(out_dir),
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--no-quant",
            "--no-compile",
        ],
        label="winml build",
    )
    return rc == 0, out


def bench_phase_a(model_path: Path) -> tuple[float | None, float]:
    """Phase A quick screen: 200 iters, check CV < SCREEN_CV_MAX.
    Returns (p50_ms, cv). p50_ms=None means unstable (reject).
    """
    out_json = model_path.parent / "screen_perf.json"
    rc, _, _ = run(
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
        label=f"winml perf (screen, iter={SCREEN_ITERS})",
    )
    if rc != 0 or not out_json.exists():
        return None, 999.0
    try:
        data = json.loads(out_json.read_text())
        lat = data["latency_ms"]
        p50 = lat["p50"]
        std = lat["std"]
        cv = std / p50 if p50 > 0 else 999.0
        print(f"     screen: p50={p50:.1f}ms  std={std:.1f}ms  CV={cv:.2f}")
        if cv > SCREEN_CV_MAX:
            print(f"     ⚠️  CV={cv:.2f} > {SCREEN_CV_MAX} — UNSTABLE, rejecting candidate")
            return None, cv
        return p50, cv
    except Exception as e:
        print(f"     [warn] parse error: {e}")
        return None, 999.0


def bench_phase_b(model_path: Path, label: str) -> list[float]:
    """Phase B full bench: 3 independent sessions × 1000 iters with cool-down.
    Returns list of p50_ms values (one per session).
    """
    p50s = []
    for session in range(1, FULL_SESSIONS + 1):
        out_json = model_path.parent / f"full_perf_s{session}.json"
        rc, _, _ = run(
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
            label=f"winml perf (full s{session}/{FULL_SESSIONS}, iter={FULL_ITERS})",
        )
        if rc == 0 and out_json.exists():
            data = json.loads(out_json.read_text())
            p50 = data["latency_ms"]["p50"]
            std = data["latency_ms"]["std"]
            cv = std / p50 if p50 > 0 else 999.0
            print(f"     full s{session}: p50={p50:.1f}ms  std={std:.1f}ms  CV={cv:.2f}")
            p50s.append(p50)
        if session < FULL_SESSIONS:
            print(f"     cooling down {COOL_DOWN_S}s …")
            time.sleep(COOL_DOWN_S)
    return p50s


def eval_accuracy(out_dir: Path) -> float | None:
    """Run winml eval; return accuracy (top-1 or cosine). For latency: use bench_*."""
    model_path = out_dir / "model.onnx"
    if not model_path.exists():
        return None
    result_json = out_dir / "eval_result.json"
    rc, _, _ = run(
        [
            WINML,
            "eval",
            "-m",
            str(model_path),
            "--model-id",
            MODEL_ID,
            "--task",
            TASK,
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--samples",
            str(EVAL_SAMPLES),
            "-o",
            str(result_json),
        ],
        label="winml eval (accuracy gate)",
    )
    if rc != 0 or not result_json.exists():
        return None
    try:
        data = json.loads(result_json.read_text())
        metrics = data.get("metrics", data)
        acc = metrics.get("accuracy")
        return float(acc) if acc is not None else None
    except Exception as e:
        print(f"     [warn] parse error: {e}")
        return None


def write_experiment_doc(exp_dir: Path, info: dict) -> None:
    """Write per-experiment structured artifact (V2 pattern):
    Hypothesis / Implementation / Parity / Perf / Analysis / Decision
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    doc = f"""# Experiment {info["iter"]:02d}: {info["label"]}

## Hypothesis
{info.get("hypothesis", "(not recorded)")}

## Implementation
- Config flags: `{info.get("optim_flags", "")}`
- Opset: `{info.get("opset", 17)}`
- Search dimension: `{info.get("dimension", "")}`

## Parity (accuracy gate)
- Accuracy: `{info.get("accuracy", "N/A")}`
- Floor: `{ACCURACY_FLOOR}`
- Result: `{"PASS" if (info.get("accuracy") or 0) >= ACCURACY_FLOOR else "FAIL"}`

## Performance
### Phase A (quick screen, {SCREEN_ITERS} iters)
- p50: `{info.get("screen_p50", "N/A")}ms`
- CV: `{info.get("screen_cv", "N/A")}` (threshold: {SCREEN_CV_MAX})

### Phase B (full bench, {FULL_ITERS}×{FULL_SESSIONS} sessions)
- p50 per session: `{info.get("full_p50s", [])}`
- Median p50: `{info.get("median_p50", "N/A")}ms`
- Baseline p50: `{info.get("baseline_p50", "N/A")}ms`
- Delta: `{info.get("delta_pct", "N/A")}`

## Analysis
{info.get("analysis", "(auto-generated: no significant analysis)")}

## Decision
**{info.get("status", "UNKNOWN").upper()}**

Timestamp: {datetime.now().isoformat(timespec="seconds")}
"""
    (exp_dir / "experiment.md").write_text(doc, encoding="utf-8")


def log(row: dict) -> None:
    fields = [
        "iter",
        "label",
        "dimension",
        "optim_flags",
        "opset",
        "accuracy",
        "screen_p50_ms",
        "median_p50_ms",
        "baseline_p50_ms",
        "delta_pct",
        "cv",
        "status",
        "elapsed_s",
        "timestamp",
    ]
    is_new = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def optim_flags(cfg: dict) -> str:
    flags = [k for k, v in cfg.get("optim", {}).items() if v is True]
    return ",".join(flags) if flags else "(none)"


# ── main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load EP knowledge (confirmed entries only)
    kb = load_ep_knowledge(EP)
    print(f"\n=== KB loaded for EP={EP} ===")
    for note in kb["notes"]:
        print(f"  {note}")

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  autoconfig search  --  {MODEL_ID}")
    print(f"  EP: {EP}   eval_samples: {EVAL_SAMPLES}   hypotheses: {len(HYPOTHESES)}")
    print(
        f"  Bench: screen={SCREEN_ITERS} iters (CV<{SCREEN_CV_MAX}) → full={FULL_ITERS}×{FULL_SESSIONS}"
    )
    print(f"  Stop: {STOP_CONSECUTIVE_DISCARDS} consecutive DISCARDs OR budget")
    print(f"  External research trigger: after {EXTERNAL_RESEARCH_TRIGGER} DISCARDs same dimension")
    print(f"{sep}\n")

    baseline_p50: float | None = None
    best_p50 = float("inf")
    best_label = ""
    consecutive_discards = 0
    discard_by_dimension: dict[str, int] = {}

    for i, (label, patch_fn, dimension) in enumerate(HYPOTHESES):
        iter_start = time.time()
        print(f"\n{'--' * 32}")
        print(f"  iter {i}  |  {label}  [{dimension}]")
        print(f"{'--' * 32}")

        # Check KB skip_set (confirmed rules only)
        flags_preview = optim_flags(patch_fn(copy.deepcopy(BASELINE)))  # type: ignore[operator]
        skip_reason = next(
            (r for r in kb["skip_passes"] if any(f in flags_preview for f in r.split()[:2])), None
        )
        if skip_reason:
            print(f"  ⏭️  skipped by KB confirmed rule: {skip_reason}")
            continue

        cfg = patch_fn(copy.deepcopy(BASELINE))  # type: ignore[operator]
        flags = optim_flags(cfg)
        opset = cfg["export"]["opset_version"]
        print(f"  optim: {flags}")
        print(f"  opset: {opset}")

        out_dir = WORK_DIR / f"iter_{i:02d}"
        exp_dir = WORK_DIR / "experiments" / f"{i:02d}_{dimension}"
        ok, _ = build(cfg, out_dir)

        exp_info: dict = {
            "iter": i,
            "label": label,
            "dimension": dimension,
            "optim_flags": flags,
            "opset": opset,
            "hypothesis": label,
            "baseline_p50": f"{baseline_p50:.1f}" if baseline_p50 else "N/A",
        }

        if not ok:
            status = "crash"
            exp_info["analysis"] = "winml build failed — check build log"
        else:
            # Phase A: quick screen
            screen_p50, screen_cv = bench_phase_a(out_dir / "model.onnx")
            exp_info["screen_p50"] = f"{screen_p50:.1f}" if screen_p50 else "UNSTABLE"
            exp_info["screen_cv"] = f"{screen_cv:.3f}"

            if screen_p50 is None:
                status = "discard (unstable — CV too high)"
                exp_info["analysis"] = (
                    f"Phase A rejected: CV={screen_cv:.2f} > {SCREEN_CV_MAX}. Likely DVFS noise. Cool device and retry."
                )
            else:
                # Phase B: full bench
                full_p50s = bench_phase_b(out_dir / "model.onnx", label)
                if not full_p50s:
                    status = "crash (full bench failed)"
                    exp_info["analysis"] = "Phase B winml perf returned no data"
                else:
                    median_p50 = sorted(full_p50s)[len(full_p50s) // 2]
                    exp_info["full_p50s"] = [f"{p:.1f}" for p in full_p50s]
                    exp_info["median_p50"] = f"{median_p50:.1f}"

                    if baseline_p50 is None and i == 0:
                        baseline_p50 = median_p50
                        exp_info["baseline_p50"] = f"{baseline_p50:.1f}"

                    # Accuracy gate
                    accuracy = eval_accuracy(out_dir)
                    exp_info["accuracy"] = f"{accuracy:.4f}" if accuracy is not None else "N/A"

                    if accuracy is not None and accuracy < ACCURACY_FLOOR:
                        status = f"discard (accuracy {accuracy:.4f} < floor {ACCURACY_FLOOR})"
                        exp_info["analysis"] = "Accuracy regression below floor"
                    elif baseline_p50 is not None and median_p50 > baseline_p50 * (
                        1 - MIN_IMPROVEMENT
                    ):
                        delta_pct = (median_p50 - baseline_p50) / baseline_p50 * 100
                        status = f"discard (Δp50={delta_pct:+.1f}% < {MIN_IMPROVEMENT * 100:.0f}% threshold)"
                        exp_info["delta_pct"] = f"{delta_pct:+.1f}%"
                        exp_info["analysis"] = (
                            f"No meaningful improvement: {delta_pct:+.1f}% vs {MIN_IMPROVEMENT * 100:.0f}% threshold"
                        )
                    else:
                        delta_pct = (
                            (median_p50 - (baseline_p50 or median_p50))
                            / (baseline_p50 or median_p50)
                            * 100
                        )
                        status = "keep"
                        exp_info["delta_pct"] = f"{delta_pct:+.1f}%"
                        exp_info["analysis"] = (
                            f"Improvement confirmed: p50 {baseline_p50:.1f}ms → {median_p50:.1f}ms ({delta_pct:+.1f}%)"
                        )
                        if median_p50 < best_p50:
                            best_p50 = median_p50
                            best_label = label
                            status = "keep *** NEW BEST ***"

        # Write per-experiment doc (V2 pattern)
        exp_info["status"] = status
        write_experiment_doc(exp_dir, exp_info)

        # Track consecutive discards + external research trigger
        if "discard" in status or "crash" in status:
            consecutive_discards += 1
            discard_by_dimension[dimension] = discard_by_dimension.get(dimension, 0) + 1
            if discard_by_dimension[dimension] == EXTERNAL_RESEARCH_TRIGGER:
                print(
                    f"\n  ⚡ EXTERNAL RESEARCH TRIGGER: {EXTERNAL_RESEARCH_TRIGGER} consecutive DISCARDs in [{dimension}]"
                )
                print("     → Search ORT/QNN source code for mechanism before continuing")
                print(
                    "     → Check kMaxSupportedOpset for opset dimension, EP-specific rules for others"
                )
                print(f"     → File findings in ep_knowledge/{EP}.json as 'draft' entry")
        else:
            consecutive_discards = 0
            discard_by_dimension[dimension] = 0

        # Log to TSV
        log(
            {
                "iter": i,
                "label": label,
                "dimension": dimension,
                "optim_flags": flags,
                "opset": opset,
                "accuracy": exp_info.get("accuracy", "N/A"),
                "screen_p50_ms": exp_info.get("screen_p50", "N/A"),
                "median_p50_ms": exp_info.get("median_p50", "N/A"),
                "baseline_p50_ms": exp_info.get("baseline_p50", "N/A"),
                "delta_pct": exp_info.get("delta_pct", "N/A"),
                "cv": exp_info.get("screen_cv", "N/A"),
                "status": status,
                "elapsed_s": f"{time.time() - iter_start:.0f}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

        print(f"  → {status}")

        # Stop condition
        if consecutive_discards >= STOP_CONSECUTIVE_DISCARDS:
            print(
                f"\n  🛑 STOP: {STOP_CONSECUTIVE_DISCARDS} consecutive DISCARDs — plateau reached"
            )
            break

    print(f"\n{sep}")
    print("  SEARCH COMPLETE")
    print(f"  Best config: {best_label}")
    print(f"  Best p50: {best_p50:.1f}ms" if best_p50 < float("inf") else "  No improvement found")
    print(f"  Results: {RESULTS_TSV}")
    print(f"  Experiments: {WORK_DIR / 'experiments'}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()


import sys
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── settings ─────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/convnext-tiny-224"
TASK = "image-classification"
EP = "cpu"
DEVICE = "cpu"
WINML = str(Path(__file__).parent / ".venv" / "Scripts" / "winml.exe")
WORK_DIR = Path(__file__).parent / "convnext-search"
RESULTS_TSV = WORK_DIR / "results.tsv"

EVAL_SAMPLES = 50  # small for demo speed (~12s per eval)
ACCURACY_FLOOR = 0.70  # drop below this → discard (FP32 baseline ~78%)
LATENCY_FLOOR = 1.0  # seconds — more than this means regression

# ── baseline config ───────────────────────────────────────────────────────────
BASELINE: dict = {
    "export": {
        "opset_version": 17,
        "batch_size": 1,
        "do_constant_folding": True,
        "dynamo": False,
        "input_tensors": [
            {
                "name": "pixel_values",
                "dtype": "float32",
                "shape": [1, 3, 224, 224],
                "value_range": [0, 1],
            }
        ],
        "output_tensors": [{"name": "logits"}],
    },
    "optim": {},  # will be patched per hypothesis
    "loader": {
        "task": TASK,
        "model_class": "AutoModelForImageClassification",
        "model_type": "convnext",
    },
    "eval": {
        "task": TASK,
        "dataset": {"path": "timm/mini-imagenet", "split": "test", "samples": EVAL_SAMPLES},
    },
}

# ── hypothesis sequence ───────────────────────────────────────────────────────
# ConvNext-tiny architecture:
#   Stem: Conv 4x4 + LN → 4 stages of ConvNext blocks
#   Each block: DW-Conv → LN → Linear (=Gemm) → GELU → Linear
#   Skip connections: pointwise Add
#
# Relevant fusions:
#   conv-bn-fusion      — conv+BatchNorm folding (stem/downsample layers)
#   conv-add-fusion     — conv+bias add (ConvNext uses DepthwiseConv with bias)
#   gelu-fusion         — fuse decomposed GELU → com.microsoft/Gelu
#   layer-norm-fusion   — fuse LN subgraph (ConvNext uses LayerNorm heavily)
#   matmul-add-fusion   — fuse Gemm+bias (the inverted bottleneck MLPs)
#   transpose-optimizer — eliminate redundant transposes around reshape ops
#   constant-folding    — pre-fold constant subgraphs (on by default in export,
#                         but also at optim stage via ORT)


def h0_baseline(cfg: dict) -> dict:
    """FP32 export, no extra fusions — reference point"""
    cfg["optim"] = {}
    return cfg


def h1_conv_fusions(cfg: dict) -> dict:
    """Enable all conv fusions — ConvNext stem uses Conv+BN, blocks use DW-Conv+bias"""
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
    }
    return cfg


def h2_gelu_fusion(cfg: dict) -> dict:
    """Add GELU fusion — ConvNext MLP blocks use GELU activation"""
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
    }
    return cfg


def h3_add_layernorm(cfg: dict) -> dict:
    """Add LayerNorm fusion — ConvNext uses LN (not BN) in blocks"""
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
        "layer-norm-fusion": True,
    }
    return cfg


def h4_add_matmul(cfg: dict) -> dict:
    """Add MatMul+Add fusion — ConvNext MLP uses Gemm (collapsed MatMul+bias)"""
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
        "layer-norm-fusion": True,
        "matmul-add-fusion": True,
    }
    return cfg


def h5_transpose_opt(cfg: dict) -> dict:
    """Add transpose optimizer — ConvNext has many Transpose ops (NCHW reshapes)"""
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
        "layer-norm-fusion": True,
        "matmul-add-fusion": True,
        "transpose-optimizer": True,
    }
    return cfg


def h6_opset18(cfg: dict) -> dict:
    """Try opset 18 with all fusions — GroupNorm introduced in opset18"""
    cfg["export"]["opset_version"] = 18
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
        "layer-norm-fusion": True,
        "matmul-add-fusion": True,
        "transpose-optimizer": True,
    }
    return cfg


def h7_surgery(cfg: dict) -> dict:
    """Add clamp-constant-values — prevents -inf attention mask quant issues"""
    cfg["export"]["opset_version"] = 17
    cfg["optim"] = {
        "conv-bn-fusion": True,
        "conv-add-fusion": True,
        "conv-activation-fusion": True,
        "gelu-fusion": True,
        "layer-norm-fusion": True,
        "matmul-add-fusion": True,
        "transpose-optimizer": True,
        "clamp-constant-values": True,
    }
    return cfg


HYPOTHESES: list[tuple[str, object]] = [
    ("baseline: no fusions (FP32 reference)", h0_baseline),
    ("conv fusions: bn+add+activation", h1_conv_fusions),
    ("+ gelu-fusion", h2_gelu_fusion),
    ("+ layer-norm-fusion", h3_add_layernorm),
    ("+ matmul-add-fusion (MLP blocks)", h4_add_matmul),
    ("+ transpose-optimizer", h5_transpose_opt),
    ("opset=18 + all fusions", h6_opset18),
    ("back to opset=17 + surgery: clamp-constant-values", h7_surgery),
]

# ── helpers ───────────────────────────────────────────────────────────────────


def run(cmd: list[str], label: str = "") -> tuple[int, str, float]:
    t0 = time.time()
    print(f"  >> {label or cmd[1]}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    status = "ok" if result.returncode == 0 else f"rc={result.returncode}"
    print(f"     done in {elapsed:.0f}s  [{status}]")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-600:]
        print(f"     stderr: {tail}")
    return result.returncode, result.stdout + result.stderr, elapsed


def build(cfg: dict, out_dir: Path) -> tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    rc, out, _ = run(
        [
            WINML,
            "build",
            "-c",
            str(cfg_path),
            "-m",
            MODEL_ID,
            "-o",
            str(out_dir),
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--no-quant",
            "--no-compile",
        ],
        label="winml build",
    )
    return rc == 0, out


def eval_onnx(out_dir: Path) -> tuple[float | None, float | None]:
    """Eval model.onnx; return (accuracy, latency_s)."""
    model_path = out_dir / "model.onnx"
    if not model_path.exists():
        print("     [warn] model.onnx not found")
        return None, None

    result_json = out_dir / "eval_result.json"
    rc, _, _ = run(
        [
            WINML,
            "eval",
            "-m",
            str(model_path),
            "--model-id",
            MODEL_ID,
            "--task",
            TASK,
            "--ep",
            EP,
            "--device",
            DEVICE,
            "--samples",
            str(EVAL_SAMPLES),
            "-o",
            str(result_json),
        ],
        label="winml eval",
    )
    if rc != 0 or not result_json.exists():
        return None, None
    try:
        data = json.loads(result_json.read_text())
        metrics = data.get("metrics", data)
        accuracy = metrics.get("accuracy")
        latency = metrics.get("latency_in_seconds")
        return (
            float(accuracy) if accuracy is not None else None,
            float(latency) if latency is not None else None,
        )
    except Exception as e:
        print(f"     [warn] parse error: {e}")
        return None, None


def log(row: dict) -> None:
    fields = [
        "iter",
        "label",
        "optim_flags",
        "opset",
        "accuracy",
        "latency_ms",
        "delta_acc",
        "delta_lat_ms",
        "status",
        "elapsed_s",
        "timestamp",
    ]
    is_new = not RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def optim_flags(cfg: dict) -> str:
    flags = [k for k, v in cfg.get("optim", {}).items() if v is True]
    return ",".join(flags) if flags else "(none)"


# ── main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  autoconfig search  --  {MODEL_ID}")
    print(f"  EP: {EP}   eval_samples: {EVAL_SAMPLES}   hypotheses: {len(HYPOTHESES)}")
    print(f"  Objective: maximize accuracy  (floor={ACCURACY_FLOOR})")
    print("  Search space: WinMLOptimizationConfig capability flags")
    print(f"{sep}\n")

    baseline_acc: float | None = None
    baseline_lat: float | None = None
    best_acc = 0.0
    best_lat = float("inf")
    best_label = ""
    total_start = time.time()

    for i, (label, patch_fn) in enumerate(HYPOTHESES):
        iter_start = time.time()
        print(f"\n{'--' * 31}")
        print(f"  iter {i}  |  {label}")
        print(f"{'--' * 31}")

        cfg = patch_fn(copy.deepcopy(BASELINE))  # type: ignore[operator]
        flags = optim_flags(cfg)
        opset = cfg["export"]["opset_version"]
        print(f"  optim: {flags}")
        print(f"  opset: {opset}")

        out_dir = WORK_DIR / f"iter_{i:02d}"
        ok, _ = build(cfg, out_dir)
        if not ok:
            status = "crash"
            accuracy = latency = None
        else:
            accuracy, latency = eval_onnx(out_dir)
            if accuracy is None:
                status = "eval_error"
            elif accuracy < ACCURACY_FLOOR:
                status = "discard (accuracy < floor)"
            elif latency is not None and latency > LATENCY_FLOOR:
                status = "discard (latency regression)"
            else:
                status = "keep"
                if accuracy > best_acc or (accuracy == best_acc and (latency or 999) < best_lat):
                    best_acc = accuracy
                    best_lat = latency or float("inf")
                    best_label = label
                    status = "keep *** NEW BEST ***"

        # Print result
        if accuracy is not None:
            lat_ms = f"{(latency or 0) * 1000:.0f}ms" if latency else "N/A"
            print(f"  accuracy={accuracy:.4f}  latency={lat_ms}  -> {status}")
            if baseline_acc is None and i == 0:
                baseline_acc = accuracy
                baseline_lat = latency
            if baseline_acc is not None and i > 0:
                d_acc = accuracy - baseline_acc
                d_lat = ((latency or 0) - (baseline_lat or 0)) * 1000
                sign_acc = "+" if d_acc >= 0 else ""
                sign_lat = "+" if d_lat >= 0 else ""
                print(f"  vs baseline: acc {sign_acc}{d_acc:.4f}  lat {sign_lat}{d_lat:.0f}ms")
        else:
            print(f"  -> {status}")

        elapsed = time.time() - iter_start
        delta_acc = (
            f"{accuracy - baseline_acc:+.4f}"
            if (accuracy is not None and baseline_acc is not None)
            else "N/A"
        )
        delta_lat = (
            f"{((latency or 0) - (baseline_lat or 0)) * 1000:+.0f}"
            if (latency is not None and baseline_lat is not None)
            else "N/A"
        )
        log(
            {
                "iter": i,
                "label": label,
                "optim_flags": flags,
                "opset": opset,
                "accuracy": f"{accuracy:.4f}" if accuracy is not None else "N/A",
                "latency_ms": f"{(latency or 0) * 1000:.0f}" if latency is not None else "N/A",
                "delta_acc": delta_acc,
                "delta_lat_ms": delta_lat,
                "status": status,
                "elapsed_s": f"{elapsed:.0f}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    total = time.time() - total_start
    print(f"\n{sep}")
    print(f"  SEARCH COMPLETE  |  {total / 60:.1f} min total")
    print(f"  Best config: {best_label}")
    print(f"  Best accuracy: {best_acc:.4f}   latency: {best_lat * 1000:.0f}ms")
    print(f"  Results: {RESULTS_TSV}")
    print(f"{sep}\n")

    if RESULTS_TSV.exists():
        print(RESULTS_TSV.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
