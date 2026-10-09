# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the deterministic dominant-hotspot planner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SKILL_ROOT / "scripts" / "plan_hotspot.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_hotspot", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_evidence(*, quantized: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider_attribution": "valid",
        "dominant_accelerator_time_pct": 92,
        "fallback_is_larger_explanation": False,
        "partitioning_is_larger_explanation": False,
        "transfers_are_larger_explanation": False,
        "quantized": quantized,
        "outcomes": {
            "representation": None,
            "qdq-boundary": None,
        },
    }


def test_quantized_dominant_hotspot_returns_fast_lane_plan() -> None:
    module = _load_module()

    plan = module.plan_hotspot(_base_evidence())

    assert plan["mode"] == "dominant-hotspot-fast-lane"
    assert [step["id"] for step in plan["steps"]] == [
        "representation",
        "qdq-boundary",
    ]
    assert (
        plan["exit"]
        == "Record both outcomes, then invoke the normal hypothesis loop in a later planning step."
    )


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        (
            {"dominant_accelerator_time_pct": 69},
            "dominant accelerator time is below 70 percent",
        ),
        ({"provider_attribution": "invalid"}, "provider attribution is not valid"),
        ({"fallback_is_larger_explanation": True}, "fallback is a larger explanation"),
        (
            {"partitioning_is_larger_explanation": True},
            "partitioning is a larger explanation",
        ),
        (
            {"transfers_are_larger_explanation": True},
            "transfers are a larger explanation",
        ),
    ],
)
def test_non_qualifying_evidence_returns_normal_hypothesis_loop(
    patch: dict[str, Any], reason: str
) -> None:
    module = _load_module()
    evidence = _base_evidence()
    evidence.update(patch)

    plan = module.plan_hotspot(evidence)

    assert plan == {
        "mode": "normal-hypothesis-loop",
        "reason": reason,
        "steps": [],
    }


def test_unquantized_dominant_hotspot_returns_representation_only() -> None:
    module = _load_module()

    plan = module.plan_hotspot(_base_evidence(quantized=False))

    assert plan["mode"] == "dominant-hotspot-fast-lane"
    assert [step["id"] for step in plan["steps"]] == ["representation"]


def test_all_required_outcomes_recorded_returns_normal_hypothesis_loop() -> None:
    module = _load_module()
    evidence = _base_evidence()
    evidence["outcomes"] = {
        "representation": "KEEP",
        "qdq-boundary": "DISCARD",
    }

    plan = module.plan_hotspot(evidence)

    assert plan == {
        "mode": "normal-hypothesis-loop",
        "reason": "all required fast-lane outcomes are already recorded",
        "steps": [],
    }


@pytest.mark.parametrize(
    "evidence",
    [
        {
            **_base_evidence(),
            "quantized": "true",
        },
        {
            **_base_evidence(),
            "schema_version": 2,
        },
        {
            **_base_evidence(),
            "dominant_accelerator_time_pct": -1,
        },
        {
            **_base_evidence(),
            "dominant_accelerator_time_pct": 101,
        },
        {
            **_base_evidence(),
            "outcomes": {
                "representation": "MAYBE",
                "qdq-boundary": None,
            },
        },
    ],
)
def test_invalid_evidence_raises_hotspot_plan_error(evidence: dict[str, Any]) -> None:
    module = _load_module()

    with pytest.raises(module.HotspotPlanError):
        module.plan_hotspot(evidence)


@pytest.mark.parametrize(
    "percentage",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_non_finite_percentage_raises_hotspot_plan_error(percentage: float) -> None:
    module = _load_module()
    evidence = _base_evidence()
    evidence["dominant_accelerator_time_pct"] = percentage

    with pytest.raises(
        module.HotspotPlanError,
        match=r"dominant_accelerator_time_pct must be a number in \[0, 100\]",
    ):
        module.plan_hotspot(evidence)


def test_cli_stdout_and_output_are_byte_stable(tmp_path: Path) -> None:
    input_path = tmp_path / "hotspot_evidence.json"
    output_path = tmp_path / "hotspot_plan.json"
    input_path.write_text(json.dumps(_base_evidence()), encoding="utf-8")

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        str(input_path),
        "--output",
        str(output_path),
    ]
    run_process = subprocess.run
    result = run_process(
        command,
        capture_output=True,
        text=False,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8")
    expected = (
        json.dumps(
            {
                "exit": (
                    "Record both outcomes, then invoke the normal hypothesis loop "
                    "in a later planning step."
                ),
                "mode": "dominant-hotspot-fast-lane",
                "steps": [
                    {
                        "id": "representation",
                        "instruction": (
                            "On the dominant region only, test one semantics-preserving "
                            "representation change supported by its topology. Hold "
                            "quantization parameters fixed. Apply normal correctness and "
                            "paired-screen gates. Record KEEP, DISCARD, or INCONCLUSIVE."
                        ),
                    },
                    {
                        "id": "qdq-boundary",
                        "instruction": (
                            "Starting from the representation probe winner (or the original "
                            "representation if that probe was discarded), hold representation "
                            "and every quantization parameter fixed. Vary only complete-region "
                            "versus branch-local QDQ placement around the same dominant region. "
                            "Apply normal correctness and paired-screen gates. Record KEEP, "
                            "DISCARD, or INCONCLUSIVE."
                        ),
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    assert result.stdout == expected
    assert output_path.read_bytes() == expected


def test_cli_prints_error_and_exits_one_for_invalid_evidence(tmp_path: Path) -> None:
    input_path = tmp_path / "hotspot_evidence.json"
    input_path.write_text(
        json.dumps({**_base_evidence(), "quantized": "true"}),
        encoding="utf-8",
    )

    command = [sys.executable, str(SCRIPT_PATH), str(input_path)]
    run_process = subprocess.run
    result = run_process(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("ERROR: ")


def test_cli_prints_error_and_exits_one_for_nan_percentage(tmp_path: Path) -> None:
    input_path = tmp_path / "hotspot_evidence.json"
    input_path.write_text(
        json.dumps(
            {
                **_base_evidence(),
                "dominant_accelerator_time_pct": float("nan"),
            }
        ),
        encoding="utf-8",
    )

    command = [sys.executable, str(SCRIPT_PATH), str(input_path)]
    run_process = subprocess.run
    result = run_process(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == ("ERROR: dominant_accelerator_time_pct must be a number in [0, 100]\n")
