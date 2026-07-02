# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Result construction and per-model result-file IO for the eval runner.

Works with the unified eval_result.json format (facts only, no derived report
fields — report rendering lives in the ModelKitArtifacts-site scripts):
  result["perf"]     — perf phase facts (always present when perf ran)
  result["accuracy"] — accuracy phase facts (present when accuracy ran, else None)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .classifier import classify_failure


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .registry import ModelEntry


# ---------------------------------------------------------------------------
# Result dict construction
# ---------------------------------------------------------------------------


def build_eval_result(
    entry: ModelEntry,
    perf_proc: dict | None,
    device: str,
    eval_types_run: list[str],
    accuracy_result: dict | None = None,
    ep: str | None = None,
    onnx_size_bytes: int | None = None,
    sanitize_fn: Callable[[str], str] | None = None,
    precision: str | None = None,
) -> dict:
    """Build a unified eval_result dict (facts only, no derived fields).

    perf_proc is the raw subprocess result from run_model(), or None when
    eval_types_run is ["accuracy"] (accuracy-only mode, perf phase skipped).
    accuracy_result is the accuracy sub-section dict (or None if not run).
    ep is the explicit execution provider (e.g., "qnn", "dml"), or None when
    not specified (device-to-provider mapping was used).
    onnx_size_bytes is the combined size of the exported ONNX + .data files.
    sanitize_fn, when provided, is applied to stdout/stderr to remove noise.
    """
    perf_section: dict | None = None
    if perf_proc is not None:
        passed = perf_proc["exit_code"] == 0
        raw_stdout = perf_proc["stdout"]
        raw_stderr = perf_proc["stderr"]
        if sanitize_fn is not None:
            stdout = sanitize_fn(raw_stdout)
            stderr = sanitize_fn(raw_stderr)
        else:
            stdout = raw_stdout
            stderr = raw_stderr
        perf_section = {
            "passed": passed,
            "elapsed": perf_proc["elapsed"],
            "exit_code": perf_proc["exit_code"],
            "stdout_output": stdout,
            "stderr_output": stderr,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
            "timeout": perf_proc["timeout"],
            "command": perf_proc["command"],
            "error": perf_proc.get("error_summary", ""),
        }

    result = {
        "model": entry.hf_id,
        "task": entry.task,
        "device": device,
        "model_type": entry.model_type,
        "group": entry.group,
        "priority": entry.priority,
        "eval_types_run": eval_types_run,
        "run_timestamp": (
            perf_proc.get("timestamp") if perf_proc else datetime.now(timezone.utc).isoformat()
        ),
        "perf": perf_section,
        "accuracy": accuracy_result,
    }
    # Optional fields: only include when explicitly provided by the user.
    if onnx_size_bytes is not None:
        result["onnx_size_bytes"] = onnx_size_bytes
    if ep is not None:
        result["ep"] = ep
    if precision is not None:
        result["precision"] = precision
    return result


# ---------------------------------------------------------------------------
# Perf failure classification (derived from perf sub-section facts)
# ---------------------------------------------------------------------------


def classify_result(result: dict) -> str | None:
    """Derive failure_classification from result["perf"] facts. Returns None if passed."""
    perf = result.get("perf")
    if perf is None or perf.get("passed"):
        return None
    if perf.get("timeout"):
        return "TIMEOUT"
    combined = perf.get("stdout_output", "") + perf.get("stderr_output", "")
    exit_code = perf.get("exit_code", -1)
    return classify_failure(combined, exit_code).value


def classify_results(results: list[dict]) -> None:
    """Add failure_classification to each result's perf sub-section in-place."""
    for r in results:
        perf = r.get("perf")
        if perf is not None:
            perf["failure_classification"] = classify_result(r)


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------


def write_result_json(result: dict, path: Path) -> None:
    """Write a single model eval_result.json (facts only, no derived fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def load_result_json(path: Path) -> dict:
    """Load a single model eval_result.json."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Accuracy status (coarse, baseline-free)
# ---------------------------------------------------------------------------


def accuracy_status(accuracy: dict | None) -> str:
    """Return a coarse accuracy status from the recorded facts.

    The runner does not grade against a PyTorch baseline (that is the report
    site's job), so this only reflects whether winml eval produced metrics:

      * ``NOT_RUN``  — accuracy phase did not run (accuracy is None)
      * ``SKIPPED``  — accuracy was skipped (e.g. perf failed)
      * ``PASS``     — winml eval succeeded and produced metrics
      * ``FAIL``     — winml eval ran but did not produce usable metrics
    """
    if accuracy is None:
        return "NOT_RUN"
    if accuracy.get("skipped"):
        return "SKIPPED"
    return "PASS" if accuracy.get("winml_eval_status") == "PASS" else "FAIL"
