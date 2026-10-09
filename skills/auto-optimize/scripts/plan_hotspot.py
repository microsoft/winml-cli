# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deterministic planner for dominant-hotspot fast-lane routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ALLOWED_OUTCOME_VALUES = {None, "KEEP", "DISCARD", "INCONCLUSIVE"}
STEP_ORDER = ("representation", "qdq-boundary")
STEP_INSTRUCTIONS = {
    "representation": (
        "On the dominant region only, test one semantics-preserving "
        "representation change supported by its topology. Hold quantization "
        "parameters fixed. Apply normal correctness and paired-screen gates. "
        "Record KEEP, DISCARD, or INCONCLUSIVE."
    ),
    "qdq-boundary": (
        "Starting from the representation probe winner (or the original "
        "representation if that probe was discarded), hold representation and "
        "every quantization parameter fixed. Vary only complete-region versus "
        "branch-local QDQ placement around the same dominant region. Apply "
        "normal correctness and paired-screen gates. Record KEEP, DISCARD, or "
        "INCONCLUSIVE."
    ),
}
EXIT_INSTRUCTION = (
    "Record both outcomes, then invoke the normal hypothesis loop in a later planning step."
)


class HotspotPlanError(ValueError):
    """Raised when hotspot planning evidence is malformed."""


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HotspotPlanError(f"{field_name} must be an object")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise HotspotPlanError(f"{field_name} must be a boolean")
    return value


def _require_percentage(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HotspotPlanError("dominant_accelerator_time_pct must be a number in [0, 100]")
    percentage = float(value)
    if not math.isfinite(percentage) or percentage < 0 or percentage > 100:
        raise HotspotPlanError("dominant_accelerator_time_pct must be a number in [0, 100]")
    return percentage


def _validate_outcomes(value: Any) -> dict[str, str | None]:
    outcomes = _require_mapping(value, "outcomes")
    unknown_keys = sorted(set(outcomes) - set(STEP_ORDER))
    if unknown_keys:
        raise HotspotPlanError(f"outcomes contains unsupported step ids: {', '.join(unknown_keys)}")

    normalized: dict[str, str | None] = {}
    for step_id in STEP_ORDER:
        outcome = outcomes.get(step_id)
        if outcome not in ALLOWED_OUTCOME_VALUES:
            raise HotspotPlanError("outcome values must be null, KEEP, DISCARD, or INCONCLUSIVE")
        normalized[step_id] = outcome
    return normalized


def _normal_plan(reason: str) -> dict[str, Any]:
    return {
        "mode": "normal-hypothesis-loop",
        "reason": reason,
        "steps": [],
    }


def _dominant_plan(step_ids: list[str]) -> dict[str, Any]:
    return {
        "exit": EXIT_INSTRUCTION,
        "mode": "dominant-hotspot-fast-lane",
        "steps": [
            {"id": step_id, "instruction": STEP_INSTRUCTIONS[step_id]} for step_id in step_ids
        ],
    }


def plan_hotspot(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic hotspot plan for the supplied evidence."""
    data = _require_mapping(evidence, "evidence")
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise HotspotPlanError("schema_version must be 1")

    provider_attribution = data.get("provider_attribution")
    if provider_attribution not in {"valid", "invalid"}:
        raise HotspotPlanError("provider_attribution must be 'valid' or 'invalid'")

    dominant_accelerator_time_pct = _require_percentage(data.get("dominant_accelerator_time_pct"))
    fallback_is_larger_explanation = _require_bool(
        data.get("fallback_is_larger_explanation"),
        "fallback_is_larger_explanation",
    )
    partitioning_is_larger_explanation = _require_bool(
        data.get("partitioning_is_larger_explanation"),
        "partitioning_is_larger_explanation",
    )
    transfers_are_larger_explanation = _require_bool(
        data.get("transfers_are_larger_explanation"),
        "transfers_are_larger_explanation",
    )
    quantized = _require_bool(data.get("quantized"), "quantized")
    outcomes = _validate_outcomes(data.get("outcomes"))

    required_steps = ["representation", "qdq-boundary"] if quantized else ["representation"]
    pending_steps = [step_id for step_id in required_steps if outcomes[step_id] is None]
    if not pending_steps:
        return _normal_plan("all required fast-lane outcomes are already recorded")
    if provider_attribution != "valid":
        return _normal_plan("provider attribution is not valid")
    if dominant_accelerator_time_pct < 70:
        return _normal_plan("dominant accelerator time is below 70 percent")
    if fallback_is_larger_explanation:
        return _normal_plan("fallback is a larger explanation")
    if partitioning_is_larger_explanation:
        return _normal_plan("partitioning is a larger explanation")
    if transfers_are_larger_explanation:
        return _normal_plan("transfers are a larger explanation")
    return _dominant_plan(pending_steps)


def _render_plan(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Build and optionally persist a hotspot plan from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        evidence = json.loads(args.input.read_text(encoding="utf-8"))
        plan = plan_hotspot(evidence)
        rendered = _render_plan(plan)
        if args.output is not None:
            args.output.write_bytes(rendered.encode("utf-8"))
        sys.stdout.buffer.write(rendered.encode("utf-8"))
        return 0
    except (HotspotPlanError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
