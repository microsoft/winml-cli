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


def parse_qnn_profiling_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Parse a QNN basic-mode profiling CSV into a list of per-sample records.

    Returns one entry per inference sample::

        [
            {
                "metadata": {hvx_threads, accel_execute_cycles, accel_execute_us},
                "samples": [{name, op_id, cycles}, ...],
            },
            ...
        ]

    Each sample carries its *own* ROOT metadata so per-operator durations can
    be derived against the accelerator cycle counts of the same inference
    (the cycle->US factor varies slightly between samples). Operator
    aggregation across samples is left to the caller.
    """
    rows = _read_csv(csv_path)
    return _extract_samples(rows)


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
