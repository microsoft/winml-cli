# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Accuracy evaluation data structures, threshold logic, and summary generation.

Mirrors the design of reporter.py (Signal 1):
- eval_result.json["accuracy"] stores only facts (raw metrics + deltas)
- verdict is a DERIVED value computed on-the-fly from stored facts
- Updating thresholds reruns verdict derivation without re-evaluating models
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Verdict enum and thresholds
# ---------------------------------------------------------------------------


class AccuracyVerdict(str, Enum):
    """Verdict of a single model's accuracy evaluation."""

    ACCURACY_PASS = "ACCURACY_PASS"  # noqa: S105  # |relative_delta| < 5%
    ACCURACY_AT_RISK = "ACCURACY_AT_RISK"  # 5% ≤ |relative_delta| < 10%
    ACCURACY_REGRESSION = "ACCURACY_REGRESSION"  # |relative_delta| ≥ 10%
    EVAL_ERROR = "EVAL_ERROR"  # winml eval or baseline subprocess failed
    SKIPPED = "SKIPPED"  # perf_failed
    DATASET_CONFIG_MISSING = "DATASET_CONFIG_MISSING"  # no dataset_config in registry


# Per-metric comparison strategy:
# (delta_key, thresh_pass, thresh_at_risk, higher_is_better)
# higher_is_better: True  = larger value is better (accuracy, f1, spearman)
#                   False = smaller value is better (WER, loss)
METRIC_COMPARE_STRATEGY: dict[str, tuple[str, float, float, bool]] = {
    "cosine_spearman": ("delta_absolute", 2.0, 4.0, True),
    # WinML-vs-baseline delta is small — pick a tighter threshold than default.
    "knn_top1_accuracy": ("delta_relative", 0.02, 0.05, True),
    "pseudo_perplexity": ("delta_relative", 0.05, 0.10, False),
    # CER (OCR error rate): lower is better.
    "cer": ("delta_relative", 0.05, 0.10, False),
    # AbsRel (depth-estimation error): lower is better.
    "abs_rel": ("delta_relative", 0.05, 0.10, False),
    "default": ("delta_relative", 0.05, 0.10, True), # 5% and 10%
}


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_delta(
    winml_metric: dict | None,
    baseline_metric: dict | None,
) -> tuple[float | None, float | None]:
    """Return (delta_absolute, delta_relative) from metric dicts.

    Both dicts must have a ``"value"`` key.
    Returns (None, None) if either is missing or baseline value is zero.

    Note: For error-rate metrics (WER — lower is better) a positive delta
    means the WinML pipeline is *worse*.  derive_verdict() normalizes the
    sign using higher_is_better from METRIC_COMPARE_STRATEGY.
    """
    if winml_metric is None or baseline_metric is None:
        return None, None
    winml_val = winml_metric.get("value")
    base_val = baseline_metric.get("value")
    if winml_val is None or base_val is None:
        return None, None
    delta_abs = winml_val - base_val
    if base_val == 0:
        return round(delta_abs, 6), None
    return round(delta_abs, 6), round(delta_abs / base_val, 6)


def format_delta(accuracy: dict) -> str:
    """Format the comparison delta as a display string using the metric strategy.

    Returns e.g. ``"-6.0%"`` for relative metrics or ``"-1.27"`` for absolute.
    Returns ``""`` if delta is unavailable.
    """
    metric_name = (accuracy.get("dataset_config") or {}).get("metric")
    delta_key = METRIC_COMPARE_STRATEGY.get(
        metric_name, METRIC_COMPARE_STRATEGY["default"]
    )[0]
    d = accuracy.get(delta_key)
    if d is None:
        return ""
    if delta_key == "delta_absolute":
        return f"{d:+.2f}"
    return f"{d:.1%}"


# ---------------------------------------------------------------------------
# Verdict derivation (always from stored facts, never from a stored verdict)
# ---------------------------------------------------------------------------


def derive_verdict(accuracy: dict | None) -> AccuracyVerdict:
    """Derive AccuracyVerdict from the accuracy sub-section of eval_result.json.

    Takes ``result["accuracy"]`` (the nested dict), not the top-level result.
    Returns EVAL_ERROR if accuracy is None (not run).

    Never reads a stored ``verdict`` field so that reruns with updated
    thresholds always produce fresh verdicts.
    """
    if accuracy is None:
        return AccuracyVerdict.EVAL_ERROR

    if accuracy.get("skipped"):
        if accuracy.get("skip_reason") == "no_dataset_config":
            return AccuracyVerdict.DATASET_CONFIG_MISSING
        return AccuracyVerdict.SKIPPED

    winml_ok = accuracy.get("winml_eval_status") == "PASS"
    base_ok = accuracy.get("pytorch_baseline_status") == "PASS"
    if not winml_ok or not base_ok:
        return AccuracyVerdict.EVAL_ERROR

    # Look up compare strategy by metric name
    metric_name = (accuracy.get("dataset_config") or {}).get("metric")
    delta_key, thresh_pass, thresh_at_risk, higher_is_better = METRIC_COMPARE_STRATEGY.get(
        metric_name, METRIC_COMPARE_STRATEGY["default"]
    )
    delta = accuracy.get(delta_key)
    if delta is None:
        return AccuracyVerdict.EVAL_ERROR

    # Normalize so negative always means regression.
    delta = delta if higher_is_better else -delta
    if delta >= 0:
        return AccuracyVerdict.ACCURACY_PASS
    if abs(delta) < thresh_pass:
        return AccuracyVerdict.ACCURACY_PASS
    if abs(delta) < thresh_at_risk:
        return AccuracyVerdict.ACCURACY_AT_RISK
    return AccuracyVerdict.ACCURACY_REGRESSION


def derive_verdicts(results: list[dict]) -> None:
    """Add ``verdict`` to each result's accuracy sub-section in-place.

    Only modifies results that have a non-None accuracy section.
    """
    for r in results:
        acc = r.get("accuracy")
        if acc is not None:
            acc["verdict"] = derive_verdict(acc).value


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_accuracy_summary(results: list[dict]) -> dict:
    """Aggregate accuracy results from unified eval_result dicts.

    Expects each result's accuracy section to already have a ``verdict``
    key (call ``derive_verdicts()`` first).
    """
    acc_results = [r for r in results if r.get("accuracy") is not None]
    total = len(acc_results)
    skipped_no_dataset = sum(
        1 for r in acc_results if (r["accuracy"] or {}).get("skip_reason") == "no_dataset_config"
    )
    skipped_perf_failed = sum(
        1 for r in acc_results if (r["accuracy"] or {}).get("skip_reason") == "perf_failed"
    )
    evaluated = sum(1 for r in acc_results if not (r["accuracy"] or {}).get("skipped"))

    counts: dict[str, int] = {}
    for r in acc_results:
        v = (r["accuracy"] or {}).get("verdict", AccuracyVerdict.EVAL_ERROR.value)
        counts[v] = counts.get(v, 0) + 1

    pass_count = counts.get(AccuracyVerdict.ACCURACY_PASS.value, 0)
    pass_rate = round(pass_count / evaluated, 3) if evaluated > 0 else 0.0

    # Per-task breakdown
    by_task: dict[str, dict] = {}
    for r in acc_results:
        task = r.get("task") or "unknown"
        if task not in by_task:
            by_task[task] = {
                "total": 0,
                "pass": 0,
                "at_risk": 0,
                "regression": 0,
                "error": 0,
                "skipped": 0,
            }
        entry = by_task[task]
        entry["total"] += 1
        v = (r["accuracy"] or {}).get("verdict", "")
        if v == AccuracyVerdict.ACCURACY_PASS.value:
            entry["pass"] += 1
        elif v == AccuracyVerdict.ACCURACY_AT_RISK.value:
            entry["at_risk"] += 1
        elif v == AccuracyVerdict.ACCURACY_REGRESSION.value:
            entry["regression"] += 1
        elif v in (AccuracyVerdict.SKIPPED.value, AccuracyVerdict.DATASET_CONFIG_MISSING.value):
            entry["skipped"] += 1
        else:
            entry["error"] += 1

    return {
        "total_candidates": total,
        "skipped_perf_failed": skipped_perf_failed,
        "skipped_no_dataset": skipped_no_dataset,
        "evaluated": evaluated,
        "accuracy_pass": counts.get(AccuracyVerdict.ACCURACY_PASS.value, 0),
        "accuracy_at_risk": counts.get(AccuracyVerdict.ACCURACY_AT_RISK.value, 0),
        "accuracy_regression": counts.get(AccuracyVerdict.ACCURACY_REGRESSION.value, 0),
        "eval_error": counts.get(AccuracyVerdict.EVAL_ERROR.value, 0),
        "pass_rate": pass_rate,
        "by_task": by_task,
    }


def _build_accuracy_md_lines(results: list[dict], summary: dict) -> list[str]:
    """Build accuracy markdown section as a list of lines."""
    s = summary

    def _val(acc: dict, key: str) -> str:
        v = (acc.get(key) or {}).get("value")
        return f"{v:.4f}" if isinstance(v, float) else str(v) if v is not None else "N/A"

    lines = [
        "# Accuracy Evaluation Summary",
        "",
        f"**Date**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total candidates | {s['total_candidates']} |",
        f"| Evaluated | {s['evaluated']} |",
        f"| Accuracy PASS | {s['accuracy_pass']} |",
        f"| Accuracy AT_RISK | {s['accuracy_at_risk']} |",
        f"| Accuracy REGRESSION | {s['accuracy_regression']} |",
        f"| Eval errors | {s['eval_error']} |",
        f"| Skipped (perf failed) | {s['skipped_perf_failed']} |",
        f"| Skipped (no dataset config) | {s['skipped_no_dataset']} |",
        f"| **Pass rate** | **{s['pass_rate']:.1%}** |",
        "",
        "## By Task",
        "",
        "| Task | Total | Pass | At Risk | Regression | Error | Skipped |",
        "|------|-------|------|---------|------------|-------|---------|",
    ]
    for task, t in sorted(s.get("by_task", {}).items()):
        lines.append(
            f"| {task} | {t['total']} | {t['pass']} | {t['at_risk']} "
            f"| {t['regression']} | {t['error']} | {t['skipped']} |"
        )

    regressions = [
        r
        for r in results
        if (r.get("accuracy") or {}).get("verdict") == AccuracyVerdict.ACCURACY_REGRESSION.value
    ]
    lines += ["", "## Accuracy Regressions", ""]
    if regressions:
        lines += [
            "| Model | Task | WinML | Baseline | Delta |",
            "|-------|------|-------|----------|-------|",
        ]
        for r in regressions:
            acc = r["accuracy"]
            lines.append(
                f"| {r['model']} | {r.get('task', '')} "
                f"| {_val(acc, 'winml_metric')} | {_val(acc, 'pytorch_baseline_metric')} "
                f"| {format_delta(acc)} |"
            )
    else:
        lines.append("_No regressions._")

    at_risk = [
        r
        for r in results
        if (r.get("accuracy") or {}).get("verdict") == AccuracyVerdict.ACCURACY_AT_RISK.value
    ]
    lines += ["", "## At-Risk Models", ""]
    if at_risk:
        lines += [
            "| Model | Task | WinML | Baseline | Delta |",
            "|-------|------|-------|----------|-------|",
        ]
        for r in at_risk:
            acc = r["accuracy"]
            lines.append(
                f"| {r['model']} | {r.get('task', '')} "
                f"| {_val(acc, 'winml_metric')} | {_val(acc, 'pytorch_baseline_metric')} "
                f"| {format_delta(acc)} |"
            )
    else:
        lines.append("_No at-risk models._")

    no_cfg = [
        r
        for r in results
        if (r.get("accuracy") or {}).get("verdict") == AccuracyVerdict.DATASET_CONFIG_MISSING.value
    ]
    if no_cfg:
        lines += ["", "## Skipped — No Dataset Config", ""]
        tasks_missing = sorted({r.get("task", "unknown") for r in no_cfg})
        lines.append(
            f"_Tasks not yet configured ({len(no_cfg)} models): {', '.join(tasks_missing)}_"
        )

    return lines


def write_accuracy_summary_md(results: list[dict], summary: dict, path: Path) -> None:
    """Write human-readable accuracy section to a markdown file."""
    lines = _build_accuracy_md_lines(results, summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
