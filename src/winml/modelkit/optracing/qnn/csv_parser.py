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

Multiple inference samples are delimited by the ROOT
"Number of HVX threads used" marker (the first ROOT metric of each
inference); every sample carries its own ROOT metadata.
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
        metadata : dict  -- aggregate hvx_threads, accel_execute_cycles,
            accel_execute_us, num_samples (representative of all samples)
        operators : list[dict]  -- ops averaged across samples, sorted by
            cycles desc
        samples : list[dict]  -- one entry per inference sample, each
            ``{"metadata": {...}, "samples": [op, ...]}`` carrying that
            sample's own ROOT metadata and per-operator cycle counts
    """
    rows = _read_csv(csv_path)
    samples = _extract_samples(rows)
    metadata = _aggregate_metadata(samples)
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


def _extract_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Split the CSV rows into per-sample records.

    A sample begins at the ROOT ``Number of HVX threads used`` marker — the
    first ROOT metric QNN emits for each inference — and runs until the next
    such marker (or end-of-file). This groups every ROOT metric (HVX threads,
    accelerator execute cycles/US) with the NODE rows of the same inference,
    so each sample carries its *own* metadata rather than sharing a single
    first-occurrence snapshot.

    Returns a list of ``{"metadata": {...}, "samples": [op, ...]}`` dicts;
    samples that produced no operator rows are dropped.
    """
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in rows:
        event_level = row.get("Event Level", "").strip()
        event_id = row.get("Event Identifier", "").strip()
        message = row.get("Message", "").strip()
        time_val = row.get("Time", "").strip()
        unit = row.get("Unit of Measurement", "").strip()

        # Sample boundary: a new HVX-threads marker starts a fresh sample.
        if event_level == "ROOT" and event_id == "Number of HVX threads used" and unit == "COUNT":
            if current is not None and current["samples"]:
                samples.append(current)
            current = {
                "metadata": {
                    "hvx_threads": int(time_val),
                    "accel_execute_cycles": 0,
                    "accel_execute_us": 0,
                },
                "samples": [],
            }
            continue

        if current is None:
            # Rows before the first HVX marker are compile/finalize noise.
            continue

        meta = current["metadata"]

        if (
            event_level == "ROOT"
            and event_id == "Accelerator (execute) time (cycles)"
            and unit == "CYCLES"
        ):
            meta["accel_execute_cycles"] = int(time_val)
            continue

        if event_level == "ROOT" and event_id == "Accelerator (execute) time" and unit == "US":
            meta["accel_execute_us"] = int(time_val)
            continue

        # Collect NODE SUB-EVENT rows with CYCLES unit.
        if message == "NODE" and event_level == "SUB-EVENT" and unit == "CYCLES":
            parsed = _parse_node_event(event_id, time_val)
            if parsed is not None:
                current["samples"].append(parsed)

    # Flush the last sample.
    if current is not None and current["samples"]:
        samples.append(current)

    return samples


def _aggregate_metadata(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a representative metadata dict spanning all samples.

    ``hvx_threads`` is constant across samples, so the first sample's value is
    used. Accelerator cycles/US are averaged so the headline figures reflect
    the whole run rather than a single inference.
    """
    if not samples:
        return {"hvx_threads": 0, "accel_execute_cycles": 0, "accel_execute_us": 0}

    n = len(samples)
    metas = [s["metadata"] for s in samples]
    return {
        "hvx_threads": metas[0]["hvx_threads"],
        "accel_execute_cycles": round(sum(m["accel_execute_cycles"] for m in metas) / n),
        "accel_execute_us": round(sum(m["accel_execute_us"] for m in metas) / n),
    }


def _parse_node_event(event_id: str, time_val: str) -> dict[str, Any] | None:
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
    samples: list[dict[str, Any]],
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
        for op in sample["samples"]:
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
