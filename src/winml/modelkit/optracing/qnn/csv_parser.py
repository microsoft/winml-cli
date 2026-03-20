# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parse QNN basic-mode profiling CSV files.

The CSV emitted by QNN EP in basic profiling mode has seven columns::

    Msg Timestamp, Message, Time, Unit of Measurement,
    Timing Source, Event Level, Event Identifier

ROOT rows carry aggregate metadata (HVX thread count, accelerator
execute time in cycles/US).  NODE SUB-EVENT rows carry per-operator
cycle counts.  UNKNOWN SUB-EVENT rows (compile stages) are ignored.

Multiple inference samples are separated by ROOT
"Accelerator (execute) time (cycles)" boundaries.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


# Regex for extracting operator name and OpId from the Event Identifier.
# Examples:
#   "Input OpId_2 (cycles)"
#   "pixel_values_QuantizeLinear_3:OpId_16 (cycles)"
#   "/resnet/embedder/embedder/convolution/Conv_token_1_2:OpId_24 (cycles)"
_OP_PATTERN = re.compile(r"(.+?)(?:\s+|:)OpId_(\d+)\s*\(cycles\)")

# Regex for stripping _token_\d+ suffixes injected by the QNN compiler.
_TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")


def parse_qnn_profiling_csv(csv_path: str | Path) -> dict[str, Any]:
    """Parse a QNN basic-mode profiling CSV into a structured dict.

    Returns:
    -------
    dict with keys:
        metadata : dict  -- hvx_threads, accel_execute_cycles, num_samples
        operators : list[dict]  -- aggregated ops sorted by cycles desc
        samples : list[list[dict]]  -- per-sample operator lists
    """
    rows = _read_csv(csv_path)
    metadata = _extract_metadata(rows)
    samples = _extract_samples(rows)
    metadata["num_samples"] = len(samples)
    operators = _aggregate_operators(samples)
    return {
        "metadata": metadata,
        "operators": operators,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    """Read the CSV file and return rows as list of dicts."""
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _extract_metadata(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Extract ROOT-level metadata from the CSV rows.

    Captures the *first* occurrence of each metric so the result
    reflects the initial inference sample.
    """
    hvx_threads: int | None = None
    accel_execute_cycles: int | None = None
    accel_execute_us: int | None = None

    for row in rows:
        event_level = row.get("Event Level", "").strip()
        event_id = row.get("Event Identifier", "").strip()
        time_val = row.get("Time", "").strip()
        unit = row.get("Unit of Measurement", "").strip()

        if event_level != "ROOT":
            continue

        if (
            event_id == "Number of HVX threads used"
            and unit == "COUNT"
            and hvx_threads is None
        ):
            hvx_threads = int(time_val)

        if (
            event_id == "Accelerator (execute) time (cycles)"
            and unit == "CYCLES"
            and accel_execute_cycles is None
        ):
            accel_execute_cycles = int(time_val)

        if (
            event_id == "Accelerator (execute) time"
            and unit == "US"
            and accel_execute_us is None
        ):
            accel_execute_us = int(time_val)

    return {
        "hvx_threads": hvx_threads or 0,
        "accel_execute_cycles": accel_execute_cycles or 0,
        "accel_execute_us": accel_execute_us or 0,
    }


def _extract_samples(rows: list[dict[str, str]]) -> list[list[dict[str, Any]]]:
    """Parse NODE SUB-EVENT rows into per-sample operator lists.

    Each sample begins at a ROOT row with
    ``Accelerator (execute) time (cycles)`` and ends before the
    next such row (or end-of-file).
    """
    samples: list[list[dict[str, Any]]] = []
    current_sample: list[dict[str, Any]] | None = None

    for row in rows:
        event_level = row.get("Event Level", "").strip()
        event_id = row.get("Event Identifier", "").strip()
        message = row.get("Message", "").strip()
        time_val = row.get("Time", "").strip()
        unit = row.get("Unit of Measurement", "").strip()

        # Detect sample boundary.
        if (
            event_level == "ROOT"
            and event_id == "Accelerator (execute) time (cycles)"
            and unit == "CYCLES"
        ):
            # Close any previous sample before starting a new one.
            if current_sample is not None:
                samples.append(current_sample)
            current_sample = []
            continue

        # Only collect NODE SUB-EVENT rows with CYCLES unit.
        if (
            current_sample is not None
            and message == "NODE"
            and event_level == "SUB-EVENT"
            and unit == "CYCLES"
        ):
            parsed = _parse_node_event(event_id, time_val)
            if parsed is not None:
                current_sample.append(parsed)

    # Flush the last sample.
    if current_sample is not None and len(current_sample) > 0:
        samples.append(current_sample)

    return samples


def _parse_node_event(
    event_id: str, time_val: str
) -> dict[str, Any] | None:
    """Parse a single NODE SUB-EVENT identifier into name/op_id/cycles."""
    m = _OP_PATTERN.match(event_id)
    if m is None:
        return None

    raw_name = m.group(1).strip()
    op_id = int(m.group(2))
    cycles = int(time_val)

    # Strip _token_\d+ suffixes inserted by the QNN compiler.
    name = _TOKEN_SUFFIX.sub("", raw_name)

    return {"name": name, "op_id": op_id, "cycles": cycles}


def _aggregate_operators(
    samples: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average operator cycles across samples and sort by cycles desc.

    Operators are keyed by ``op_id`` so identically-named ops in
    different positions are kept separate.
    """
    if not samples:
        return []

    # Accumulate totals keyed by op_id.
    totals: dict[int, dict[str, Any]] = {}
    counts: dict[int, int] = {}

    for sample in samples:
        for op in sample:
            oid = op["op_id"]
            if oid not in totals:
                totals[oid] = {
                    "name": op["name"],
                    "op_id": oid,
                    "cycles": 0,
                }
                counts[oid] = 0
            totals[oid]["cycles"] += op["cycles"]
            counts[oid] += 1

    # Average.
    aggregated: list[dict[str, Any]] = []
    for oid, entry in totals.items():
        avg_cycles = entry["cycles"] / counts[oid]
        aggregated.append(
            {
                "name": entry["name"],
                "op_id": entry["op_id"],
                "cycles": avg_cycles,
            }
        )

    # Sort descending by cycles.
    aggregated.sort(key=lambda op: op["cycles"], reverse=True)
    return aggregated
