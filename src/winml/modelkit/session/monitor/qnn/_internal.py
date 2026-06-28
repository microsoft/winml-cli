# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Private QNN parsing internals. Only qnn_monitor.py imports from here.

Consolidates the previously public ``qnn/csv_parser.py`` and
``qnn/qhas_parser.py`` modules into a single private submodule per the
v2.4 simplification (spec §3.2 / coreloop §4.3 / OQ-1 resolution (b)).

The CSV side parses QNN basic-mode profiling output: a seven-column CSV
with ROOT-level metadata (HVX threads, accelerator execute cycles/us)
and NODE SUB-EVENT rows carrying per-operator cycle counts.  Multiple
inference samples are separated by ROOT
``Accelerator (execute) time (cycles)`` boundaries.

The QHAS side parses QNN Hardware Acceleration Summary JSON: per-operator
hardware metrics including cycles, DRAM/VTCM traffic, and dominant-path
information.  QHAS is authoritative for ``qnn_op_type`` (e.g. ``"Conv2d"``)
and feeds the L2 layer of the v2.4 fallback chain
(``QNNMonitor._resolve_op_type``).
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
# Imported by qnn_monitor.QNNMonitor._heuristic_op_type to share strip
# semantics with the CSV path.
_TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")


# ---------------------------------------------------------------------------
# CSV parser (basic-mode profiling)
# ---------------------------------------------------------------------------


def _split_op_event_id(event_id: str) -> tuple[str, str]:
    """Heuristic split for the CSV-only path (no authoritative op-type column).

    The QNN basic-mode profiling CSV does not carry an op-type column, so
    we recover one from the event ID.  QNN compiler emits event IDs as
    either bare op types (e.g. ``"Gelu"``) or hierarchical framework
    paths (e.g. ``"/encoder/layer/Conv"``).  The leaf segment of a
    slash-delimited event ID is *usually* the ONNX op symbol; this is a
    best-effort fallback used by :func:`_parse_node_event` only.

    .. warning::

       The leaf segment is the **ONNX op symbol** (``"Conv"``, ``"Add"``,
       ``"LayerNormalization"``), NOT the **QNN op type**
       (``"Conv2d"``, ``"ElementWiseAdd"``, ``"PoolMax2d"``).  When an
       authoritative QNN op type is available — as it is in QHAS detail
       mode via ``qnn_op_type`` — use it directly and do NOT call this
       helper.

    Edge-case behavior:
    - Strips leading/trailing whitespace before splitting.
    - Empty or whitespace-only input returns ``("", "")``.
    - Trailing-slash input (``"/encoder/"``) preserves the input for
      ``op_path`` and falls back to the input for ``op_type`` so callers
      never receive an empty string they didn't supply.

    Returns:
    -------
    tuple[str, str]
        ``(op_type, op_path)``.  For bare event IDs both elements are
        the trimmed input.  For path-style event IDs ``op_path`` is the
        trimmed input verbatim and ``op_type`` is the trailing
        ``/``-delimited segment (also trimmed).
    """
    s = event_id.strip()
    if "/" not in s:
        return s, s
    leaf = s.rsplit("/", 1)[1].strip()
    return (leaf or s), s


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

        if event_id == "Number of HVX threads used" and unit == "COUNT" and hvx_threads is None:
            hvx_threads = int(time_val)

        if (
            event_id == "Accelerator (execute) time (cycles)"
            and unit == "CYCLES"
            and accel_execute_cycles is None
        ):
            accel_execute_cycles = int(time_val)

        if event_id == "Accelerator (execute) time" and unit == "US" and accel_execute_us is None:
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


def _parse_node_event(event_id: str, time_val: str) -> dict[str, Any] | None:
    """Parse a single NODE SUB-EVENT identifier into name/op_path/op_id/cycles.

    The captured event id may be a bare op type (``"Gelu"``) or a
    hierarchical framework path (``"/encoder/layer/Conv"``).  The split
    rule (see :func:`_split_op_event_id`) maps these into the dataclass
    contract documented at :class:`OperatorMetrics`: ``name`` is the QNN
    op type (leaf segment), ``op_path`` is the framework path (full
    string).
    """
    m = _OP_PATTERN.match(event_id)
    if m is None:
        return None

    raw_name = m.group(1).strip()
    op_id = int(m.group(2))
    cycles = int(time_val)

    # Strip _token_\d+ suffixes inserted by the QNN compiler.
    cleaned = _TOKEN_SUFFIX.sub("", raw_name)
    name, op_path = _split_op_event_id(cleaned)

    return {"name": name, "op_path": op_path, "op_id": op_id, "cycles": cycles}


def _aggregate_operators(
    samples: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average operator cycles across samples and sort by cycles desc.

    Operators are keyed by ``op_id`` so identically-named ops in
    different positions are kept separate.

    Each returned dict carries an additional ``samples_cycles`` field:
    a list of per-sample cycle counts in input order, one entry per
    sample where the op appeared.  Downstream layers (e.g.
    :class:`QNNMonitor`) convert these to microseconds and surface
    them as :attr:`OperatorMetrics.samples_us` for p90 / total / count
    derivation.  Cycle->microsecond conversion is deliberately left to
    the caller because the ``cycle_to_us`` ratio lives in ROOT-level
    metadata, not in the per-sample rows.
    """
    if not samples:
        return []

    # Accumulate totals + per-sample lists keyed by op_id.
    totals: dict[int, dict[str, Any]] = {}
    counts: dict[int, int] = {}
    samples_cycles: dict[int, list[int]] = {}

    for sample in samples:
        for op in sample:
            oid = op["op_id"]
            if oid not in totals:
                totals[oid] = {
                    "name": op["name"],
                    "op_path": op["op_path"],
                    "op_id": oid,
                    "cycles": 0,
                }
                counts[oid] = 0
                samples_cycles[oid] = []
            totals[oid]["cycles"] += op["cycles"]
            counts[oid] += 1
            samples_cycles[oid].append(op["cycles"])

    # Average and attach per-sample list.
    aggregated: list[dict[str, Any]] = []
    for oid, entry in totals.items():
        avg_cycles = entry["cycles"] / counts[oid]
        aggregated.append(
            {
                "name": entry["name"],
                "op_path": entry["op_path"],
                "op_id": entry["op_id"],
                "cycles": avg_cycles,
                "samples_cycles": samples_cycles[oid],
            }
        )

    # Sort descending by cycles.
    aggregated.sort(key=lambda op: op["cycles"], reverse=True)
    return aggregated


# ---------------------------------------------------------------------------
# QHAS parser (detail-mode roofline + DMA traffic)
# ---------------------------------------------------------------------------


def _require(d: dict, key: str, context: str) -> Any:
    """Return ``d[key]``, raising a named :exc:`KeyError` if absent.

    The plain ``d[key]`` form raises an opaque ``KeyError: 'key'`` that
    the outer ``_try_qhas`` ``except Exception`` catches and logs as
    ``"basic_fallback"`` *without* recording which key was missing.
    Surfacing the key name in the exception message lets the outer handler
    log it verbatim, making SDK schema drift diagnosable from the log.
    """
    if key not in d:
        raise KeyError(f"Required QHAS field {key!r} is missing in {context}")
    return d[key]


def parse_qhas(qhas_data: dict) -> dict:
    """Parse a QHAS JSON structure into normalised summary + operator list.

    Parameters
    ----------
    qhas_data:
        Deserialised QHAS JSON (must contain ``data.htp_overall_summary``
        and ``data.qnn_op_instances_nodes``).

    Returns:
    -------
    dict
        ``{"summary": {...}, "operators": [...]}``.
    """
    data = _require(qhas_data, "data", "QHAS root")
    summary = _extract_summary(data)

    # Derive a cycle-to-microsecond factor from the summary.
    timeline_cycles = summary["timeline_cycles"]
    cycle_to_us = summary["inference_us"] / timeline_cycles if timeline_cycles else 0.0

    raw_ops = data.get("qnn_op_instances_nodes", {}).get("data", [])
    operators = [_transform_op(op, cycle_to_us) for op in raw_ops]

    return {"summary": summary, "operators": operators}


def _extract_summary(data: dict) -> dict:
    """Extract the HTP overall summary into a flat dict.

    Keys are renamed from the raw QHAS source-of-truth
    (``time_us``, ``graph_execute_us``, ``total_dram_read``, ...) to
    the user-facing renderer vocabulary
    (``inference_us``, ``execute_us``, ``dram_read_bytes``, ...) so
    :func:`winml.modelkit.session.monitor.report._display_detail_report`
    can read them directly.  The renderer is the source of truth for
    the user-facing names because they make units explicit
    (``_us``, ``_bytes``) and the ``_bytes`` suffix indicates an
    aggregate (was implicit in the raw ``total_*`` prefix).
    """
    rows = data.get("htp_overall_summary", {}).get("data", [])
    if not rows:
        return {}
    raw = rows[0]
    ctx = "htp_overall_summary row"
    return {
        "inference_us": _require(raw, "time_us", ctx),
        "execute_us": _require(raw, "graph_execute_us", ctx),
        "inf_per_s": _require(raw, "inf_per_s", ctx),
        "timeline_cycles": _require(raw, "timeline_cycles", ctx),
        "utilization_pct": _require(raw, "percent_utilization", ctx),
        "dram_read_bytes": _require(raw, "total_dram_read", ctx),
        "dram_write_bytes": _require(raw, "total_dram_write", ctx),
        "vtcm_read_bytes": _require(raw, "total_vtcm_read", ctx),
        "vtcm_write_bytes": _require(raw, "total_vtcm_write", ctx),
        "vtcm_peak_bytes": _require(raw, "peak_vtcm_alloc", ctx),
        "qnn_nodes": _require(raw, "qnn_nodes", ctx),
        "htp_nodes": _require(raw, "htp_nodes", ctx),
        "unique_qnn_ops": _require(raw, "unique_qnn_ops", ctx),
        "unique_htp_ops": _require(raw, "unique_htp_ops", ctx),
    }


def _transform_op(op: dict, cycle_to_us: float) -> dict:
    """Transform a single ``qnn_op_instances_nodes`` entry.

    Converts raw cycle counts to microseconds and computes derived
    metrics such as VTCM hit ratio and dominant-path duration.
    """
    ctx = "qnn_op_instances_nodes entry"
    cycles = _require(op, "cycles", ctx)
    duration_us = cycles * cycle_to_us

    dp_cycles = op.get("num_dominant_path_cycles_htp_0")
    dominant_path_us = dp_cycles * cycle_to_us if dp_cycles else None

    vtcm_read = op.get("vtcm_read", 0)
    dram_read = op.get("dram_read", 0)

    # QHAS provides the authoritative QNN op type via ``qnn_op_type``
    # (e.g. ``"Conv2d"``, ``"ElementWiseAdd"``, ``"PoolMax2d"``).  Use it
    # directly for ``name`` — do NOT leaf-split ``qnn_op``: that path
    # carries the framework-level node id and its trailing segment is the
    # ONNX op symbol (``"Conv"``, ``"Add"``, ``"MaxPool"``), a *different
    # vocabulary* from the QNN op type.  Mixing the two across basic and
    # detail modes produced inconsistent Type-column rendering and would
    # invite a hardcoded translation table (Cardinal Rule #1 violation)
    # to reconcile.
    #
    # In v2.4 the resolver in :class:`QNNMonitor` may *override* this
    # ``name`` with the ONNX node.op_type when an op-type map is injected
    # (L1 of the fallback chain) — that's the FR-14 intent of giving the
    # ONNX graph the last word.  When no map is present, the QHAS
    # ``qnn_op_type`` carried in the ``"name"`` field below remains the
    # L2 authoritative source.
    #
    # ``op_path`` is stripped of the QNN compiler's ``_token_\d+(?:_\d+)?``
    # suffix so QHAS path keys match the CSV path's strip semantics
    # (see :func:`_parse_node_event`) and, more critically, match the
    # clean ONNX ``node.name`` keys produced by
    # :py:meth:`WinMLSession._build_op_type_map`.  Without this strip,
    # the FR-14 L1 ONNX-primary lookup is silently inert in detail mode:
    # production map keys are clean (``/encoder/conv1/Conv``) but QHAS
    # ``qnn_op`` carries the suffix (``/encoder/conv1/Conv_token_1_2``),
    # so L1 always misses and L2 (``qnn_op_type``) wrongly wins.
    # The strip is idempotent on already-clean strings.
    return {
        "name": _require(op, "qnn_op_type", ctx),
        "op_path": _TOKEN_SUFFIX.sub("", _require(op, "qnn_op", ctx)),
        "cycles": cycles,
        "duration_us": duration_us,
        "percent_of_total": _require(op, "percent_active_cycles", ctx),
        "dominant_path_us": dominant_path_us,
        "num_htp_ops": op.get("num_htp_ops", 0),
        "dram_read_bytes": dram_read,
        "dram_write_bytes": op.get("dram_write", 0),
        "vtcm_read_bytes": vtcm_read,
        "vtcm_write_bytes": op.get("vtcm_write", 0),
        "vtcm_hit_ratio": _vtcm_ratio(op),
    }


def _vtcm_ratio(op: dict) -> float | None:
    """Compute VTCM hit ratio: vtcm_read / (vtcm_read + dram_read).

    Returns ``None`` when both values are zero (no read traffic).
    """
    vtcm_read = op.get("vtcm_read", 0)
    dram_read = op.get("dram_read", 0)
    total = vtcm_read + dram_read
    if total == 0:
        return None
    return vtcm_read / total
