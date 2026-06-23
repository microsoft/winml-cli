#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""autoconfig.py — AutoResearch-style optimize-pass search for winml-cli
Demo: facebook/convnext-tiny-224, CPU EP, FP32

Loop: hypothesize → winml build → quick-screen bench (CV gate) →
      full bench (3 sessions) → eval → keep/discard → repeat

Key design principles (from GPU Optimizer V2 + ConvNext lessons):
  1. Two-phase bench: 200-iter CV screen FIRST, full bench only if CV < threshold
     (CPU/GPU) — or unconditionally for QNN NPU (npu-007: DVFS makes CV unreliable)
  2. Use winml perf (NOT winml eval) for latency — eval includes HF preprocessing
  3. Mandatory external-research after 5 consecutive DISCARDs in same dimension
  4. Load ep_device_knowledge/*.json (only "confirmed" entries) to prune search space
  5. Per-experiment structured output: hypothesis/impl/parity/perf/analysis/decision
  6. Stop condition: 30 consecutive DISCARDs (not 5)

Hypothesis design — ISOLATED mode (each hypothesis is independent):
  Each hypothesis is applied to a fresh copy of BASELINE. The labels "+" prefix
  is cosmetic; no state is accumulated across hypotheses. This allows independent
  attribution: "does gelu-fusion alone help?" rather than "does gelu help on top
  of conv fusions?". To run a cumulative search, chain patch functions explicitly.
"""

import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Agent package bootstrap: make the autoconfig root (the dir holding ep_device_knowledge/)
# importable so sibling skills/lib packages resolve when run as a standalone script.
_AGENT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "ep_device_knowledge").is_dir()
)
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from lib.report_gen import generate_report  # noqa: E402
from skills.explorer.analyze_insight import build_insight  # noqa: E402
from skills.optimizer.bench_utils import (  # noqa: E402
    FULL_ITERS,
    FULL_SESSIONS,
    SCREEN_CV_MAX_STD,
    SCREEN_ITERS,
    SessionManager,
    ThroughputOnly,
    VerdictInput,
    bench_full,
    bench_screen,
    median_p50,
    run_cmd,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── settings ─────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/convnext-tiny-224"
TASK = "image-classification"
EP = "cpu"
DEVICE = "cpu"
WINML = str(_AGENT_ROOT / ".venv" / "Scripts" / "winml.exe")
WORK_DIR = _AGENT_ROOT / "convnext-search"
RESULTS_TSV = WORK_DIR / "results.tsv"
KB_DIR = _AGENT_ROOT / "ep_device_knowledge"

EVAL_SAMPLES = 50  # for accuracy gate
ACCURACY_FLOOR = 0.70  # cosine drop below this → discard
MIN_IMPROVEMENT = 0.01  # require ≥1% p50 improvement to KEEP

# Verdict policy: improvement must exceed max(MIN_IMPROVEMENT, STAT_BAR * screen_cv)
# Borrowed from AgenticGPUOptimizer V2 (avoids calling noise-level deltas "improvements")
STAT_BAR_MULTIPLIER = 2.0

# Screen early exit: skip 3x full-bench when screen already shows < this % improvement.
# Saves ~25-90 min per rejected hypothesis (3 sessions × FULL_ITERS iters).
SCREEN_PASS_MIN_IMPROVEMENT_PCT = 1.0

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
    kb_path = KB_DIR / f"{ep}_{DEVICE}.json"
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


# ── full search space — the unbiased, zero-experience OFAT grid ───────────────
# The orchestrator reference loop enumerates the COMPLETE one-factor-at-a-time
# grid: every supported opset crossed with {baseline, each single graph pass}.
# This is the full set of "all combinations" BEFORE any experience is applied.
# The Explorer then prunes/reorders it via confirmed-KB hard-blocks + the
# Insight Engine. The per-(ep, device) catalog_sweep matrices in
# ep_device_knowledge/<ep>_<device>.json ("hypotheses") are the experience-pruned
# and reordered subsets of this same grid (single source of truth lives here).
#
# Each patch_fn receives a FRESH copy of BASELINE (isolated mode): hypotheses are
# independent, no state is accumulated across them.

OPSET_RANGE: list[int] = [17, 18, 19, 20, 21]

# The full universe of single graph-optimization passes winml-cli can toggle.
# catalog_sweep KBs draw their per-EP hypothesis matrices from this same set.
OPTIM_PASSES: list[str] = [
    "conv_bn_fusion",
    "conv_add_fusion",
    "conv_activation_fusion",
    "gelu_fusion",
    "layer_norm_fusion",
    "skip_layer_norm_fusion",
    "matmul_add_fusion",
    "matmul_transpose_fusion",
    "attention_fusion",
    "bias_softmax_fusion",
    "transpose_optimizer",
    "nchwc_transformer",
    "highdimRTR_lowdimRTR",
]


def _make_patch(opset: int, pass_name: str | None):
    """Return a patch_fn setting one opset and at most one optim pass on a fresh
    BASELINE copy. pass_name=None => pure opset (no fusion flags)."""

    def patch(cfg: dict) -> dict:
        cfg["export"]["opset_version"] = opset
        cfg["optim"] = {pass_name: True} if pass_name else {}
        return cfg

    return patch


def build_search_space(
    opsets: list[int] = OPSET_RANGE, passes: list[str] = OPTIM_PASSES
) -> list[tuple[str, object, str]]:
    """Enumerate the full OFAT grid: opset x {baseline, each single pass}.

    Returns (label, patch_fn, search_dimension) triples. The lowest opset with no
    pass is the global 'baseline'; other no-pass entries form the 'opset'
    dimension; every single-pass entry is a 'graph_pass'.
    """
    base_opset = opsets[0]
    space: list[tuple[str, object, str]] = []
    # 1. pure-opset axis (no fusion flags): baseline + opset sweep
    for op in opsets:
        if op == base_opset:
            label, dim = f"baseline (opset {op}, no fusions)", "baseline"
        else:
            label, dim = f"opset={op}", "opset"
        space.append((label, _make_patch(op, None), dim))
    # 2. single graph-pass axis, crossed with every opset
    for op in opsets:
        for p in passes:
            space.append((f"opset={op} + {p}", _make_patch(op, p), "graph_pass"))
    return space


HYPOTHESES: list[tuple[str, object, str]] = build_search_space()

# ── helpers ───────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — Opt Loop subagents (autoconfig_diagram.html)
#
#  The experiment loop is split into three explicit subagents that mirror the
#  architecture diagram:
#
#    Explorer   — decides *what to try next*: loads the hypothesis pool, applies
#                 KB hard-blocks + Insight-Engine skip rules (skip_set), and ranks
#                 the survivors by Insight priority boost (priority_queue).
#    Optimizer  — *runs* one hypothesis: winml build -> Phase A screen (CV gate) ->
#                 Phase B full bench -> accuracy eval. Produces raw measurements
#                 only; it makes no keep/discard decision.
#    Reviewer   — *judges* the measurements: applies the ThroughputOnly verdict
#                 policy (threshold = max(min_improvement, stat_bar x CV)), emits
#                 KEEP / MARGINAL / DISCARD, and drafts KB entries for real wins.
#
#  The orchestrator (main) wires them together: Explorer yields a hypothesis ->
#  Optimizer benchmarks it -> Reviewer returns a verdict -> repeat.
# ═══════════════════════════════════════════════════════════════════════════


class Explorer:
    """Phase 2 Explorer — hypothesis pool -> skip_set pruning -> priority_queue.

    Owns search *order* only; it never builds or benchmarks. It fuses two pruning
    signals (confirmed KB hard-blocks and the Phase 1 Insight Engine skip_set) and
    ranks the remaining hypotheses by Insight priority boost.
    """

    def __init__(self, hypotheses: list[tuple], kb: dict, insight) -> None:
        self.kb = kb
        self.insight = insight
        # priority_queue: stable sort, highest Insight priority boost first.
        self.priority_queue = sorted(
            hypotheses, key=lambda item: -insight.priority_boosts.get(item[0], 0.0)
        )

    def __iter__(self):
        """Iterate hypotheses in priority order (pop next from priority_queue)."""
        return iter(self.priority_queue)

    def skip_reason(self, label: str, flags_preview: str) -> str | None:
        """Return why this hypothesis is pruned, or None to run it.

        Checks the confirmed-KB hard-block rules first, then the Insight Engine
        skip_set. Mirrors the diagram's "Apply KB hard blocks -> skip_set" step.
        """
        kb_rule = next(
            (r for r in self.kb["skip_passes"] if any(f in flags_preview for f in r.split()[:2])),
            None,
        )
        if kb_rule is not None:
            return f"KB confirmed rule: {kb_rule}"
        if label in self.insight.skip_set:
            return f"Insight Engine: {label}"
        return None


class Optimizer:
    """Phase 2 Optimizer — winml build -> Phase A screen -> Phase B full bench -> accuracy.

    Produces raw measurements for one hypothesis. Holds the winml binary path and
    the build target (model id / EP / device); thresholds stay as module constants.
    """

    def __init__(self, winml: str, model_id: str, ep: str, device: str) -> None:
        self.winml = winml
        self.model_id = model_id
        self.ep = ep
        self.device = device

    def build(self, cfg: dict, out_dir: Path) -> tuple[bool, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        rc, out, _ = run_cmd(
            [
                self.winml,
                "build",
                "-c",
                str(cfg_path),
                "-m",
                self.model_id,
                "-o",
                str(out_dir),
                "--ep",
                self.ep,
                "--device",
                self.device,
                "--no-quant",
                "--no-compile",
            ],
            label="winml build",
        )
        return rc == 0, out

    def screen(self, model_path: Path) -> tuple[float | None, float]:
        """Phase A: 200-iter screen with CV gate.

        For CPU EP, high CV means thermal/scheduling noise — reject and retry later.
        Returns (p50_ms, cv). p50_ms=None means unstable or command failed.
        """
        sr = bench_screen(winml=self.winml, model_path=model_path, ep=self.ep, device=self.device)
        if sr.hard_failed:
            return None, 999.0
        if sr.cv is not None and sr.cv > SCREEN_CV_MAX:
            print(
                f"     Phase A rejected: CV={sr.cv:.2f} > {SCREEN_CV_MAX}"
                f" (thermal/scheduling noise on {self.ep.upper()} — cool device and retry)"
            )
            return None, sr.cv
        return sr.p50_ms, sr.cv or 0.0

    def full_bench(self, model_path: Path) -> list[float]:
        """Phase B: 3 sessions × FULL_ITERS with cool-down. Returns p50 per session."""
        return bench_full(
            winml=self.winml,
            model_path=model_path,
            ep=self.ep,
            device=self.device,
            out_prefix="full",
            iters=FULL_ITERS,
            cool_down_s=COOL_DOWN_S,
        )

    def eval_accuracy(self, out_dir: Path) -> float | None:
        """Run winml eval; return accuracy (top-1 or cosine). For latency: use bench_*."""
        model_path = out_dir / "model.onnx"
        if not model_path.exists():
            return None
        result_json = out_dir / "eval_result.json"
        rc, _, _ = run_cmd(
            [
                self.winml,
                "eval",
                "-m",
                str(model_path),
                "--model-id",
                self.model_id,
                "--task",
                TASK,
                "--ep",
                self.ep,
                "--device",
                self.device,
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


class Reviewer:
    """Phase 2 Reviewer — ThroughputOnly verdict -> KEEP / MARGINAL / DISCARD.

    Turns Optimizer measurements (full-bench p50s + accuracy) into a verdict via
    the ThroughputOnly policy, promotes the first successful bench to baseline,
    and drafts a KB entry for notable confirmed wins.
    """

    def __init__(
        self, policy: ThroughputOnly, ep: str, model_id: str, accuracy_floor: float
    ) -> None:
        self.policy = policy
        self.ep = ep
        self.model_id = model_id
        self.accuracy_floor = accuracy_floor

    def review(
        self,
        label: str,
        exp_info: dict,
        screen_cv: float,
        baseline_p50: float | None,
        full_p50s: list[float],
        accuracy: float | None,
    ) -> tuple[str, dict]:
        """Judge one hypothesis from its measurements.

        Returns (status_str, updated exp_info). Does not update best_p50/best_label —
        the orchestrator owns champion tracking so it stays in one place.
        """
        med_p50 = median_p50(full_p50s)
        assert med_p50 is not None
        exp_info["full_p50s"] = [f"{p:.1f}" for p in full_p50s]
        exp_info["median_p50"] = f"{med_p50:.1f}"

        # Promote baseline from first successful full bench
        if baseline_p50 is None:
            baseline_p50 = med_p50
            exp_info["baseline_p50"] = f"{baseline_p50:.1f}"

        exp_info["accuracy"] = f"{accuracy:.4f}" if accuracy is not None else "N/A"

        improvement_pct = (baseline_p50 - med_p50) / baseline_p50 * 100
        delta_pct = -improvement_pct
        exp_info["delta_pct"] = f"{delta_pct:+.1f}%"

        correctness_pass = accuracy is None or accuracy >= self.accuracy_floor
        verdict = self.policy.evaluate(
            VerdictInput(
                improvement_pct=improvement_pct,
                cv_pct=screen_cv * 100.0,
                correctness_pass=correctness_pass,
            )
        )

        exp_info["analysis"] = verdict.reasoning
        if verdict.verdict in ("KEEP", "MARGINAL_KEEP"):
            status = "keep" + (" (marginal)" if verdict.marginal else "")
            exp_info["analysis"] = (
                f"Improvement confirmed: p50 {baseline_p50:.1f}ms -> {med_p50:.1f}ms "
                f"({delta_pct:+.1f}%). {verdict.reasoning}"
            )
            # Auto-write KB draft entry for notable improvements
            if not verdict.marginal:
                write_kb_draft(
                    ep=self.ep,
                    label=label,
                    improvement_pct=improvement_pct,
                    cv=screen_cv,
                    model_id=self.model_id,
                    dimension=exp_info.get("dimension", "unknown"),
                )
        elif verdict.verdict == "ACC_FAIL":
            status = f"discard (accuracy {accuracy:.4f} < floor {self.accuracy_floor})"
        else:
            status = f"discard ({verdict.reasoning})"

        return status, exp_info


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


def write_kb_draft(
    ep: str, label: str, improvement_pct: float, cv: float, model_id: str, dimension: str
) -> None:
    """Append a draft finding to ep_device_knowledge/<ep>_<device>.json when improvement > 10%.

    The entry gets status='draft' — a human must review and promote to 'confirmed'
    after Gate 2 validation (>=2 independent models, mechanism understood).
    """
    if improvement_pct < 10.0:
        return
    kb_path = KB_DIR / f"{ep}_{DEVICE}.json"
    if not kb_path.exists():
        return
    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:
        return

    findings = kb.setdefault("findings", [])
    # Auto-generate a draft ID: ep-draft-<timestamp>
    draft_id = f"{ep}-draft-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Don't duplicate if same label+model already drafted
    for f in findings:
        if (
            f.get("status") == "draft"
            and f.get("model_id") == model_id
            and f.get("title", "").startswith(label[:30])
        ):
            return

    draft = {
        "id": draft_id,
        "status": "draft",
        "title": f"[DRAFT] {label} — {improvement_pct:+.1f}% on {model_id}",
        "model_id": model_id,
        "dimension": dimension,
        "improvement_pct": round(improvement_pct, 2),
        "cv": round(cv, 3),
        "mechanism_confirmed": False,
        "note": "Auto-generated draft. Requires Gate 2: >=2 models, mechanism understood.",
        "action_for_autoconfig": "investigate",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    findings.append(draft)
    kb_path.write_text(json.dumps(kb, indent=2), encoding="utf-8")
    print(f"  [KB draft] Wrote draft entry {draft_id}: {label} ({improvement_pct:+.1f}%)")


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

    # Resume from prior session if interrupted
    session = SessionManager(WORK_DIR)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  autoconfig search  --  {MODEL_ID}")
    print(f"  EP: {EP}   eval_samples: {EVAL_SAMPLES}   hypotheses: {len(HYPOTHESES)}")
    print(
        f"  Bench: screen={SCREEN_ITERS} iters (CV<{SCREEN_CV_MAX}) -> full={FULL_ITERS}x{FULL_SESSIONS}"
    )
    print(f"  Stop: {STOP_CONSECUTIVE_DISCARDS} consecutive DISCARDs OR budget")
    print(f"  External research trigger: after {EXTERNAL_RESEARCH_TRIGGER} DISCARDs same dimension")
    print(
        f"  Verdict: improvement must exceed max({MIN_IMPROVEMENT * 100:.0f}%, {STAT_BAR_MULTIPLIER:.0f}x screen-CV)"
    )
    print(
        f"  Screen early exit: skip full bench if screen improvement < {SCREEN_PASS_MIN_IMPROVEMENT_PCT:.0f}%"
    )
    print(f"{sep}\n")

    # Restore state from prior session (if resuming)
    baseline_p50: float | None = session.baseline_p50
    best_p50 = session.best_p50
    best_label = session.best_label
    consecutive_discards = session.consecutive_discards
    discard_by_dimension: dict[str, int] = session.discard_by_dimension

    policy = ThroughputOnly(
        min_improvement_pct=MIN_IMPROVEMENT * 100,
        stat_bar_multiplier=STAT_BAR_MULTIPLIER,
    )

    # Phase 2 subagents: Optimizer runs hypotheses, Reviewer judges them.
    # Explorer is constructed after Phase 1 (it needs the Insight Engine output).
    optimizer = Optimizer(WINML, MODEL_ID, EP, DEVICE)
    reviewer = Reviewer(policy, EP, MODEL_ID, ACCURACY_FLOOR)

    # ── Phase 1: Insight Engine ────────────────────────────────────────────────
    # Run AFTER baseline build so we have a real ONNX to analyse.
    # The baseline ONNX is expected at WORK_DIR/iter_00/model.onnx once h0 has run.
    # On first run the baseline may not exist yet — insight falls back gracefully.
    baseline_onnx = WORK_DIR / "iter_00" / "model.onnx"
    insight = build_insight(
        onnx_path=baseline_onnx,
        winml=WINML,
        ep=EP,
        device=DEVICE,
        hypotheses=HYPOTHESES,
        kb=kb,
    )

    # Explorer (Phase 2 "what to try next"): owns the priority_queue + skip rules.
    explorer = Explorer(HYPOTHESES, kb, insight)

    for i, (label, patch_fn, dimension) in enumerate(explorer):
        # Skip iters completed in a prior run
        if i in session.completed_iters:
            print(f"  [resume] skipping iter {i} ({label}) — already done")
            continue

        iter_start = time.time()
        print(f"\n{'--' * 32}")
        print(f"  iter {i}  |  {label}  [{dimension}]")
        print(f"{'--' * 32}")

        # Explorer decides whether to prune this hypothesis (KB hard-block or Insight skip_set)
        flags_preview = optim_flags(patch_fn(copy.deepcopy(BASELINE)))  # type: ignore[operator]
        skip_reason = explorer.skip_reason(label, flags_preview)
        if skip_reason:
            print(f"  skipped by {skip_reason}")
            continue

        cfg = patch_fn(copy.deepcopy(BASELINE))  # type: ignore[operator]
        flags = optim_flags(cfg)
        opset = cfg["export"]["opset_version"]
        print(f"  optim: {flags}")
        print(f"  opset: {opset}")

        out_dir = WORK_DIR / f"iter_{i:02d}"
        exp_dir = WORK_DIR / "experiments" / f"{i:02d}_{dimension}"
        ok, _ = optimizer.build(cfg, out_dir)

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
            # Optimizer Phase A: quick screen
            screen_p50, screen_cv = optimizer.screen(out_dir / "model.onnx")
            exp_info["screen_p50"] = f"{screen_p50:.1f}" if screen_p50 else "UNSTABLE"
            exp_info["screen_cv"] = f"{screen_cv:.3f}"

            screen_improvement_pct = (
                (baseline_p50 - screen_p50) / baseline_p50 * 100
                if (screen_p50 is not None and baseline_p50 is not None)
                else None
            )

            if screen_p50 is None:
                status = "discard (unstable — CV too high)"
                exp_info["analysis"] = (
                    f"Phase A rejected: CV={screen_cv:.2f} > {SCREEN_CV_MAX}. "
                    f"Thermal or scheduling noise on {EP.upper()} EP. Cool device and retry."
                )
            elif (
                screen_improvement_pct is not None
                and screen_improvement_pct < SCREEN_PASS_MIN_IMPROVEMENT_PCT
            ):
                # Screen early exit: skip full bench when screen shows negligible gain.
                # Saves 3x full-bench time for clearly non-improving configs.
                status = (
                    f"discard (screen early exit: improvement {screen_improvement_pct:+.1f}%"
                    f" < {SCREEN_PASS_MIN_IMPROVEMENT_PCT:.0f}% — full bench skipped)"
                )
                exp_info["analysis"] = (
                    f"Phase A early exit: screen p50={screen_p50:.1f}ms vs baseline "
                    f"{baseline_p50:.1f}ms ({screen_improvement_pct:+.1f}% improvement) is "
                    f"below {SCREEN_PASS_MIN_IMPROVEMENT_PCT:.0f}% threshold. "
                    f"Full bench skipped — not worth 3x{FULL_ITERS} iters."
                )
                exp_info["delta_pct"] = f"{-screen_improvement_pct:+.1f}% (screen estimate)"
            else:
                # Optimizer Phase B: full bench + accuracy, then Reviewer verdict.
                full_p50s = optimizer.full_bench(out_dir / "model.onnx")
                if not full_p50s:
                    status = "crash (full bench failed)"
                    exp_info["analysis"] = "Phase B winml perf returned no data"
                else:
                    accuracy = optimizer.eval_accuracy(out_dir)
                    status, exp_info = reviewer.review(
                        label=label,
                        exp_info=exp_info,
                        screen_cv=screen_cv,
                        baseline_p50=baseline_p50,
                        full_p50s=full_p50s,
                        accuracy=accuracy,
                    )
                    if status.startswith("keep"):
                        # Orchestrator owns champion tracking
                        new_p50 = float(exp_info.get("median_p50", best_p50))
                        if new_p50 < best_p50:
                            best_p50 = new_p50
                            best_label = label
                            status = "keep *** NEW BEST ***"

        # Extract baseline from first successful full bench
        if baseline_p50 is None and "median_p50" in exp_info:
            try:
                baseline_p50 = float(exp_info["median_p50"])
                exp_info["baseline_p50"] = f"{baseline_p50:.1f}"
            except (ValueError, TypeError):
                pass

        # Write per-experiment doc (V2 pattern)
        exp_info["status"] = status
        write_experiment_doc(exp_dir, exp_info)

        # Track consecutive discards + external research trigger
        if "discard" in status or "crash" in status:
            consecutive_discards += 1
            discard_by_dimension[dimension] = discard_by_dimension.get(dimension, 0) + 1
            if discard_by_dimension[dimension] == EXTERNAL_RESEARCH_TRIGGER:
                print(
                    f"\n  EXTERNAL RESEARCH TRIGGER: {EXTERNAL_RESEARCH_TRIGGER} consecutive DISCARDs in [{dimension}]"
                )
                print("     -> Search ORT/QNN source code for mechanism before continuing")
                print(
                    "     -> Check kMaxSupportedOpset for opset dimension, EP-specific rules for others"
                )
                print(
                    f"     -> File findings in ep_device_knowledge/{EP}_{DEVICE}.json as 'draft' entry"
                )
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

        print(f"  -> {status}")

        # Persist state for crash-resume
        session.save(
            iter_idx=i,
            verdict=status,
            baseline_p50=baseline_p50,
            best_p50=best_p50,
            best_label=best_label,
            consecutive_discards=consecutive_discards,
            discard_by_dimension=discard_by_dimension,
        )

        # Stop condition
        if consecutive_discards >= STOP_CONSECUTIVE_DISCARDS:
            print(f"\n  STOP: {STOP_CONSECUTIVE_DISCARDS} consecutive DISCARDs — plateau reached")
            break

    print(f"\n{sep}")
    print("  SEARCH COMPLETE")
    print(f"  Best config: {best_label}")
    print(f"  Best p50: {best_p50:.1f}ms" if best_p50 < float("inf") else "  No improvement found")
    print(f"  Results: {RESULTS_TSV}")
    print(f"  Experiments: {WORK_DIR / 'experiments'}")

    # ── Phase 3: Generate HTML report ─────────────────────────────────────────
    try:
        report_path = generate_report(
            results_tsv=RESULTS_TSV,
            work_dir=WORK_DIR,
            model_id=MODEL_ID,
            ep=EP,
            insight_notes=insight.notes,
        )
        print(f"  Report:    {report_path}")
    except Exception as e:
        print(f"  [warn] Report generation failed: {e}")

    print(f"{sep}\n")


if __name__ == "__main__":
    main()
