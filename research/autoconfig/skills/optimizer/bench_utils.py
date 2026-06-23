#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""bench_utils.py — Shared benchmarking helpers for QNN NPU sweeps.

Bench protocol (npu-007):
  Phase A: 200-iter screen. For QNN NPU, high CV (0.15-1.2) is NORMAL due to
    DVFS/Hexagon HTP thermal throttling. Phase A result is informational only;
    it never gates Phase B on NPU. Only use CV gate for CPU/GPU EPs.
  Phase B: 3 independent sessions x 500 iters with 30s cool-down.
    KEEP criterion: all p50s below baseline; for NPU, ranges must not overlap.

winml config + build helpers are also centralized here to avoid duplication
between catalog_qnn_sweep.py and validation_sweep.py.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ── Protocol constants (overridable by callers via module-level reassignment) ─
SCREEN_WARMUP: int = 20
SCREEN_ITERS: int = 200
SCREEN_CV_MAX_NPU: float = 999.0  # never gate on CV for QNN NPU (npu-007)
SCREEN_CV_MAX_STD: float = 0.10  # CPU / GPU: reject if CV > 10%

FULL_WARMUP: int = 50
FULL_ITERS: int = 500
FULL_SESSIONS: int = 3
COOL_DOWN_S: int = 30  # seconds between full-bench sessions (NPU)

BUILD_TIMEOUT_S: int = 8 * 60
BENCH_TIMEOUT_S: int = 8 * 60
CONFIG_TIMEOUT_S: int = 120

# ── Paired A/B + adaptive sampling (self-evolution-design Fix #1 / Fix #2) ─────
MIN_PAIRS: int = 3  # never conclude on fewer than this many A/B pairs
MAX_PAIRS: int = 8  # force-stop (MARGINAL) after this many pairs
KEEP_GAIN_PCT: float = 5.0  # CI lower bound must exceed this to KEEP_CONFIRMED
DISCARD_GAIN_PCT: float = -2.0  # CI upper bound below this -> DISCARD

# ── Thermal reference classification (self-evolution-design Fix #5) ────────────
THERMAL_COOL_MULT: float = 1.05  # <= 1.05x cold reference -> proceed
THERMAL_HOT_MULT: float = 1.30  # >= 1.30x cold reference -> HOT_RUN


# ── subprocess wrapper ────────────────────────────────────────────────────────


def run_cmd(cmd: list[str], label: str = "", timeout: int = 600) -> tuple[int, str, float]:
    """Run a subprocess command. Returns (returncode, combined_output, elapsed_s)."""
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


# ── winml wrappers ────────────────────────────────────────────────────────────


def get_base_config(
    winml: str,
    model_id: str,
    task: str,
    model_type: str,
    ep: str,
    device: str,
    out_path: Path,
) -> dict | None:
    """Generate a config via `winml config`. Returns parsed dict or None on failure.

    Tries with --model-type first, then falls back without it.
    """

    def _try(extra_args: list[str]) -> dict | None:
        cmd = [
            winml,
            "config",
            "-m",
            model_id,
            "-t",
            task,
            "--device",
            device,
            "--ep",
            ep,
            "--no-compile",
            "-o",
            str(out_path),
        ] + extra_args
        rc, _, _ = run_cmd(cmd, label="winml config", timeout=CONFIG_TIMEOUT_S)
        if rc == 0 and out_path.exists():
            try:
                cfg = json.loads(out_path.read_text(encoding="utf-8"))
                out_path.unlink(missing_ok=True)
                return cfg
            except Exception as e:
                print(f"  [warn] config parse error: {e}", flush=True)
        out_path.unlink(missing_ok=True)
        return None

    cfg = _try(["--model-type", model_type])
    if cfg is None:
        print("  [warn] config with --model-type failed, retrying without...", flush=True)
        cfg = _try([])
    return cfg


def run_build(
    winml: str,
    model_id: str,
    cfg_path: Path,
    out_dir: Path,
    ep: str,
    device: str,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Run `winml build`. Returns (success, combined_output)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        winml,
        "build",
        "-c",
        str(cfg_path),
        "-m",
        model_id,
        "-o",
        str(out_dir),
        "--ep",
        ep,
        "--device",
        device,
        "--no-compile",
        "--rebuild",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    rc, out, _ = run_cmd(cmd, label=f"winml build [{out_dir.name}]", timeout=BUILD_TIMEOUT_S)
    return rc == 0, out


def make_hypothesis_config(
    base: dict, opset_override: int | None, extra_optim: dict | None
) -> dict:
    """Return a modified deep copy of base config for one hypothesis."""
    cfg = copy.deepcopy(base)
    if opset_override is not None and cfg.get("export"):
        cfg["export"]["opset_version"] = opset_override
    if extra_optim is not None:
        cfg["optim"] = {**(cfg.get("optim") or {}), **extra_optim}
    return cfg


def find_model_onnx(hyp_dir: Path) -> Path | None:
    """Locate the best ONNX artifact in a build output dir.

    Priority: quantized > optimized > any .onnx.
    Returns None if no .onnx file exists.
    """
    model_files = list(hyp_dir.glob("*.onnx"))
    if not model_files:
        return None
    for preference in ("quantized", "optimized"):
        match = next((f for f in model_files if preference in f.name), None)
        if match:
            return match
    return model_files[0]


def is_build_complete(hyp_dir: Path) -> bool:
    """Return True if the hyp_dir contains a complete build artifact.

    'Complete' means optimized.onnx or quantized.onnx is present.
    export.onnx alone means the pipeline was truncated before optimization.
    """
    return any(
        f.name for f in hyp_dir.glob("*.onnx") if "optimized" in f.name or "quantized" in f.name
    )


# ── benchmark helpers ─────────────────────────────────────────────────────────


class ScreenResult:
    """Result from Phase A quick screen."""

    __slots__ = ("p50_ms", "cv", "rc_failed")

    def __init__(self, p50_ms: float | None, cv: float, rc_failed: bool = False) -> None:
        self.p50_ms = p50_ms
        self.cv = cv
        self.rc_failed = rc_failed  # True only on subprocess failure; never on high CV

    @property
    def hard_failed(self) -> bool:
        """True if the bench command itself failed (rc != 0 or no output file)."""
        return self.rc_failed

    def to_dict(self, ep: str = "cpu") -> dict:
        note = None
        if ep in ("qnn", "npu") and self.cv > 0.10:
            note = "DVFS noise — high CV expected on QNN NPU (npu-007)"
        return {
            "p50_ms": round(self.p50_ms, 3) if self.p50_ms is not None else None,
            "cv": round(self.cv, 4),
            "note": note,
        }


def bench_screen(
    winml: str,
    model_path: Path,
    ep: str,
    device: str,
    out_json: Path | None = None,
) -> ScreenResult:
    """Phase A: 200-iter screen.

    For QNN NPU: high CV is NORMAL (npu-007). Never treat high CV as failure.
    Only hard-fail on subprocess rc != 0 or missing output file.
    For CPU/GPU: high CV (> SCREEN_CV_MAX_STD) indicates measurement instability.
    """
    if out_json is None:
        out_json = model_path.parent / "screen_perf.json"
    rc, _, _ = run_cmd(
        [
            winml,
            "perf",
            "-m",
            str(model_path),
            "--ep",
            ep,
            "--device",
            device,
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
        return ScreenResult(None, 999.0, rc_failed=True)
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        lat = data.get("latency_ms", data)
        p50 = lat.get("p50") if isinstance(lat, dict) else None
        std = lat.get("std", 0.0) if isinstance(lat, dict) else 0.0
        if not p50:
            return ScreenResult(None, 999.0, rc_failed=True)
        cv = std / p50 if p50 > 0 else 999.0
        ep_tag = "NPU" if ep in ("qnn",) and device in ("npu",) else ep.upper()
        print(
            f"     screen: p50={p50:.2f}ms  cv={cv:.3f}"
            + (" [DVFS-normal]" if ep_tag == "NPU" and cv > 0.10 else ""),
            flush=True,
        )
        return ScreenResult(p50, cv)
    except Exception as e:
        print(f"     [warn] screen parse error: {e}", flush=True)
        return ScreenResult(None, 999.0, rc_failed=True)


def bench_full(
    winml: str,
    model_path: Path,
    ep: str,
    device: str,
    out_prefix: str = "full_perf",
    warmup: int | None = None,
    iters: int | None = None,
    cool_down_s: int | None = None,
) -> list[float]:
    """Phase B: 3 × FULL_ITERS-iter full bench with cool-down.

    Returns list of per-session p50_ms values. Empty list = all sessions failed.
    Session files are written as {out_prefix}_s{n}.json in model_path.parent.

    warmup/iters/cool_down_s override module-level defaults when provided.
    """
    _warmup = warmup if warmup is not None else FULL_WARMUP
    _iters = iters if iters is not None else FULL_ITERS
    _cool_down = cool_down_s if cool_down_s is not None else COOL_DOWN_S
    p50s: list[float] = []
    for s in range(1, FULL_SESSIONS + 1):
        out_json = model_path.parent / f"{out_prefix}_s{s}.json"
        rc, _, _ = run_cmd(
            [
                winml,
                "perf",
                "-m",
                str(model_path),
                "--ep",
                ep,
                "--device",
                device,
                "--warmup",
                str(_warmup),
                "--iterations",
                str(_iters),
                "-o",
                str(out_json),
            ],
            label=f"perf full s{s}/{FULL_SESSIONS} ({_iters} iters)",
            timeout=BENCH_TIMEOUT_S,
        )
        if rc == 0 and out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                lat = data.get("latency_ms", data)
                p50 = lat.get("p50") if isinstance(lat, dict) else None
                std = lat.get("std", 0.0) if isinstance(lat, dict) else 0.0
                if p50:
                    cv = std / p50 if p50 > 0 else 999.0
                    print(
                        f"     full s{s}: p50={p50:.2f}ms  std={std:.2f}ms  cv={cv:.3f}",
                        flush=True,
                    )
                    p50s.append(round(p50, 3))
            except Exception as e:
                print(f"     [warn] full bench s{s} parse error: {e}", flush=True)
        else:
            print(f"     [warn] full bench s{s} failed", flush=True)
        if s < FULL_SESSIONS:
            print(f"     cool-down {_cool_down}s...", flush=True)
            time.sleep(_cool_down)
    return p50s


def median_p50(p50s: list[float]) -> float | None:
    """Return the median of a list of p50 values, or None if empty."""
    if not p50s:
        return None
    return sorted(p50s)[len(p50s) // 2]


def ranges_non_overlapping(a: list[float], b: list[float]) -> bool | None:
    """Return True if max(a) < min(b) (a is strictly faster than b).

    Returns None if either list is empty (can't determine).
    """
    if not a or not b:
        return None
    return max(a) < min(b)


def session_cv(p50s: list[float]) -> float:
    """Session-to-session coefficient of variation (sample stddev / mean).

    This is the run-to-run noise floor used by the effect-size gate. Unlike the
    intra-session CV (screen.cv), it captures thermal / DVFS drift *between*
    sessions — the noise that produces fake cross-config wins. Returns 0.0 for
    fewer than 2 samples (spread cannot be estimated).
    """
    n = len(p50s)
    if n < 2:
        return 0.0
    mean = sum(p50s) / n
    if mean <= 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in p50s) / (n - 1)
    return (var**0.5) / mean


# ── Paired A/B bench protocol (Fix #1) ─────────────────────────────────────────


def run_perf_session(
    winml: str,
    model_path: Path,
    ep: str,
    device: str,
    iters: int | None = None,
    warmup: int | None = None,
    out_json: Path | None = None,
) -> float | None:
    """Run a single `winml perf` session. Returns p50_ms, or None on failure.

    This is the atomic measurement primitive shared by full-bench and paired A/B.
    """
    _iters = iters if iters is not None else FULL_ITERS
    _warmup = warmup if warmup is not None else FULL_WARMUP
    if out_json is None:
        out_json = model_path.parent / "ab_perf.json"
    rc, _, _ = run_cmd(
        [
            winml,
            "perf",
            "-m",
            str(model_path),
            "--ep",
            ep,
            "--device",
            device,
            "--warmup",
            str(_warmup),
            "--iterations",
            str(_iters),
            "-o",
            str(out_json),
        ],
        label=f"perf session ({_iters} iters)",
        timeout=BENCH_TIMEOUT_S,
    )
    if rc != 0 or not out_json.exists():
        return None
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        lat = data.get("latency_ms", data)
        p50 = lat.get("p50") if isinstance(lat, dict) else None
        return round(float(p50), 3) if p50 else None
    except Exception as e:
        print(f"     [warn] perf session parse error: {e}", flush=True)
        return None


def _ci_half_95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval of the mean (1.96 * SE).

    Returns a large sentinel (999.0) for fewer than 2 samples (CI undefined).
    """
    if len(values) < 2:
        return 999.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _verdict_from_gains(gains: list[float]) -> dict:
    """Summarise within-pair gain percentages into mean / 95% CI / verdict.

    verdict:
      KEEP_CONFIRMED — CI lower bound > KEEP_GAIN_PCT (real, robust speedup)
      DISCARD        — CI upper bound < DISCARD_GAIN_PCT (real regression)
      MARGINAL       — CI straddles the indifference band (need more pairs)
      BENCH_FAIL     — no usable pairs
    """
    if not gains:
        return {
            "gains_pct": [],
            "mean_gain_pct": None,
            "ci_half_95": None,
            "n_pairs": 0,
            "verdict": "BENCH_FAIL",
        }
    mean = statistics.mean(gains)
    ci = _ci_half_95(gains)
    if mean - ci > KEEP_GAIN_PCT:
        verdict = "KEEP_CONFIRMED"
    elif mean + ci < DISCARD_GAIN_PCT:
        verdict = "DISCARD"
    else:
        verdict = "MARGINAL"
    return {
        "gains_pct": [round(g, 2) for g in gains],
        "mean_gain_pct": round(mean, 2),
        "ci_half_95": round(ci, 2) if ci < 999 else None,
        "n_pairs": len(gains),
        "verdict": verdict,
    }


def paired_ab_bench(
    run_session: Callable[[Path], float | None],
    baseline_path: Path,
    hyp_path: Path,
    n_pairs: int = MIN_PAIRS,
    cool_down_s: int | None = None,
) -> dict:
    """Interleaved A/B bench (Fix #1): baseline then hypothesis in one thermal window.

    Each pair measures ``baseline`` immediately followed by ``hyp`` so DVFS / thermal
    drift appears in BOTH legs and cancels in the within-pair ratio. The mean of the
    per-pair gains (with a 95% CI) is far more reliable than comparing a cold baseline
    against a warm hypothesis across separate sweep phases.

    ``run_session`` is an injectable callable ``(model_path) -> p50_ms | None`` so the
    statistics can be unit-tested without hardware. Use :func:`run_perf_session` (via a
    lambda binding winml/ep/device) for the real measurement.
    """
    _cool = cool_down_s if cool_down_s is not None else COOL_DOWN_S
    gains: list[float] = []
    for i in range(max(1, n_pairs)):
        b = run_session(baseline_path)
        h = run_session(hyp_path)
        if b and h and b > 0:
            gains.append((b - h) / b * 100)
        if i < n_pairs - 1:
            print(f"     cool-down {_cool}s...", flush=True)
            time.sleep(_cool)
    return _verdict_from_gains(gains)


def adaptive_paired_ab_bench(
    run_session: Callable[[Path], float | None],
    baseline_path: Path,
    hyp_path: Path,
    min_pairs: int = MIN_PAIRS,
    max_pairs: int = MAX_PAIRS,
    cool_down_s: int | None = None,
) -> dict:
    """Adaptive paired A/B (Fix #2): keep sampling until the 95% CI is decisive.

    Stops early once the CI clears the KEEP or DISCARD band (after at least
    ``min_pairs`` pairs); otherwise force-stops at ``max_pairs`` and returns MARGINAL.
    Stable models finish in ``min_pairs``; noisy ones automatically get more pairs.
    """
    _cool = cool_down_s if cool_down_s is not None else COOL_DOWN_S
    gains: list[float] = []
    for i in range(max(1, max_pairs)):
        b = run_session(baseline_path)
        h = run_session(hyp_path)
        if b and h and b > 0:
            gains.append((b - h) / b * 100)
        if len(gains) >= min_pairs:
            mean = statistics.mean(gains)
            ci = _ci_half_95(gains)
            if mean - ci > KEEP_GAIN_PCT or mean + ci < DISCARD_GAIN_PCT:
                break
        if i < max_pairs - 1:
            print(f"     cool-down {_cool}s...", flush=True)
            time.sleep(_cool)
    return _verdict_from_gains(gains)


def thermal_classify(
    ref_p50_ms: float,
    cold_ref_p50_ms: float,
    cool_mult: float = THERMAL_COOL_MULT,
    hot_mult: float = THERMAL_HOT_MULT,
) -> str:
    """Classify device thermal state from a reference-model latency (Fix #5).

    ``cold_ref_p50_ms`` is the reference latency captured when the device is cold.
    Returns ``COOL`` (proceed), ``WARM`` (borderline), ``HOT_RUN`` (throttled —
    exclude from L2 promotion), or ``UNKNOWN`` if no valid cold reference.
    """
    if cold_ref_p50_ms <= 0 or ref_p50_ms <= 0:
        return "UNKNOWN"
    ratio = ref_p50_ms / cold_ref_p50_ms
    if ratio <= cool_mult:
        return "COOL"
    if ratio >= hot_mult:
        return "HOT_RUN"
    return "WARM"


# ── ONNX analysis helpers ─────────────────────────────────────────────────────


# ── Verdict policies ─────────────────────────────────────────────────────────


@dataclass
class VerdictInput:
    """Inputs to a verdict policy.

    improvement_pct: positive = latency improvement
        = (baseline_p50 - new_p50) / baseline_p50 * 100
    cv_pct: screen coefficient of variation as percent (e.g., 5.0 for 5%)
    correctness_pass: True if accuracy/parity check passed
    build_ok: True if build succeeded
    """

    improvement_pct: float
    cv_pct: float
    correctness_pass: bool
    build_ok: bool = True


@dataclass
class VerdictOutput:
    """Output from a verdict policy."""

    verdict: str  # KEEP | MARGINAL_KEEP | DISCARD | ACC_FAIL | BUILD_FAIL
    reasoning: str
    marginal: bool = False
    threshold_pct: float = 0.0


class VerdictPolicy(ABC):
    """Abstract base for verdict policies."""

    def __init__(self, min_improvement_pct: float = 1.0, stat_bar_multiplier: float = 2.0) -> None:
        self.min_improvement_pct = min_improvement_pct
        self.stat_bar_multiplier = stat_bar_multiplier

    @abstractmethod
    def evaluate(self, inp: VerdictInput) -> VerdictOutput: ...


class ThroughputOnly(VerdictPolicy):
    """KEEP iff improvement > max(min_improvement_pct, stat_bar * cv_pct).

    Parameterized statistical significance: forces improvements to exceed
    measurement noise before being declared real (borrowed from
    AgenticGPUOptimizer V2). Marks verdicts as 'marginal' when improvement is
    between 1x and 1.5x the threshold.
    """

    def evaluate(self, inp: VerdictInput) -> VerdictOutput:
        if not inp.build_ok:
            return VerdictOutput("BUILD_FAIL", "Build step failed.")
        if not inp.correctness_pass:
            return VerdictOutput("ACC_FAIL", "Accuracy check failed.")

        threshold = max(self.min_improvement_pct, self.stat_bar_multiplier * inp.cv_pct)

        if inp.improvement_pct < threshold:
            return VerdictOutput(
                "DISCARD",
                f"Improvement +{inp.improvement_pct:.1f}% < threshold {threshold:.1f}% "
                f"(max({self.min_improvement_pct:.0f}% floor, "
                f"{self.stat_bar_multiplier:.0f}x CV={inp.cv_pct:.1f}%))",
                threshold_pct=threshold,
            )

        marginal = inp.improvement_pct < threshold * 1.5
        return VerdictOutput(
            "MARGINAL_KEEP" if marginal else "KEEP",
            f"Improvement +{inp.improvement_pct:.1f}% > threshold {threshold:.1f}%",
            marginal=marginal,
            threshold_pct=threshold,
        )


# ── Session manager ───────────────────────────────────────────────────────────


class SessionManager:
    """Crash-resume state manager backed by session.json.

    Writes session state atomically (temp-file + rename) after each experiment
    so an interrupted run can be resumed from where it left off.

    Usage::
        sm = SessionManager(WORK_DIR)
        if sm.has_state:
            print(f"Resuming: {len(sm.completed_iters)} completed iters")
        # In the hypothesis loop:
        if i in sm.completed_iters:
            continue
        # ... run experiment ...
        sm.save(iter_idx=i, verdict=status, baseline_p50=..., ...)
    """

    def __init__(self, work_dir: Path) -> None:
        self.path = work_dir / "session.json"
        self._state: dict = {}
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text(encoding="utf-8"))
                n = len(self.completed_iters)
                if n > 0:
                    print(
                        f"  [session] Resuming: {n} completed iter(s) loaded from {self.path.name}",
                        flush=True,
                    )
            except Exception as e:
                print(f"  [session] Warning: could not load {self.path.name}: {e}", flush=True)

    @property
    def has_state(self) -> bool:
        return bool(self._state)

    @property
    def completed_iters(self) -> set[int]:
        return set(self._state.get("completed_iters", []))

    @property
    def baseline_p50(self) -> float | None:
        return self._state.get("baseline_p50")

    @property
    def best_p50(self) -> float:
        v = self._state.get("best_p50")
        return float(v) if v is not None else float("inf")

    @property
    def best_label(self) -> str:
        return self._state.get("best_label", "")

    @property
    def consecutive_discards(self) -> int:
        return int(self._state.get("consecutive_discards", 0))

    @property
    def discard_by_dimension(self) -> dict[str, int]:
        return dict(self._state.get("discard_by_dimension", {}))

    def save(
        self,
        *,
        iter_idx: int,
        verdict: str,
        baseline_p50: float | None,
        best_p50: float,
        best_label: str,
        consecutive_discards: int,
        discard_by_dimension: dict[str, int],
    ) -> None:
        """Save current state to session.json atomically."""
        completed = list(self.completed_iters | {iter_idx})
        self._state.update(
            {
                "completed_iters": completed,
                "last_verdict": verdict,
                "baseline_p50": baseline_p50,
                "best_p50": best_p50 if best_p50 < float("inf") else None,
                "best_label": best_label,
                "consecutive_discards": consecutive_discards,
                "discard_by_dimension": discard_by_dimension,
                "last_iter": iter_idx,
            }
        )
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:
            print(f"  [session] Warning: could not save session state: {e}", flush=True)


def count_conv_pct(model_onnx: Path) -> tuple[float, int, int]:
    """Count Conv ops as a percentage of all graph nodes.

    Returns (conv_pct, conv_count, total_count).
    Used to assess npu-006 risk: Conv% > 20% means conv fusions will likely
    produce FusedConv ops that QNN EP cannot dispatch (-> CPU fallback).

    Returns (0.0, 0, 0) if onnx is not installed or file is missing.
    The caller must treat (0.0, 0, 0) as 'unknown', not as 'safe'.
    """
    if not model_onnx.exists():
        return 0.0, 0, 0
    try:
        import onnx  # noqa: PLC0415

        model = onnx.load(str(model_onnx))
        ops = [n.op_type for n in model.graph.node]
        total = len(ops)
        conv_count = sum(1 for o in ops if o == "Conv")
        pct = conv_count / total * 100 if total > 0 else 0.0
        return round(pct, 1), conv_count, total
    except Exception as e:
        print(f"  [warn] Conv% analysis failed (onnx not installed?): {e}", flush=True)
        return 0.0, 0, 0
