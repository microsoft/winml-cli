# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parse ORT's built-in profiling JSON into per-operator aggregates.

When ``SessionOptions.enable_profiling`` is set, ORT writes a Chrome-tracing
style JSON file (``onnxruntime_profile_*.json``). Each operator execution is a
``cat == "Node"`` event whose ``name`` ends with ``_kernel_time`` and whose
``dur`` is the kernel duration in microseconds. One such event is emitted per
inference run, so durations are accumulated per node across the warmup +
measured runs and then aggregated.

The event ``name`` is the node's *output* name and may carry CPU-EP suffixes
(``_nchwc`` for NCHWc-reordered kernels, ``_token_N`` for QDQ-duplicated
nodes). :func:`resolve_node_name` maps it back to the ONNX graph node so the
operator identity is stable regardless of those transformations -- the same
approach ORT's own ``onnxruntime_test`` tooling uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


# Suffix ORT appends to every per-kernel timing event.
_KERNEL_TIME_SUFFIX = "_kernel_time"
# CPU EP suffix for NCHWc-reordered (channel-blocked) kernels.
_NCHWC_SUFFIX = "_nchwc"
# QDQ tooling duplicates a node and appends a "_token_<n>" suffix.
_TOKEN_SUFFIX = re.compile(r"_token_\d+$")


def resolve_node_name(
    name_to_type: dict[str, str],
    output_to_name: dict[str, str],
    name: str,
) -> str:
    """Resolve a profile event name back to its ONNX graph node name.

    Tries, in order: an exact node-name match, the ``_nchwc``-stripped output
    name, and the ``_token_<n>``-stripped name. Falls back to the output->node
    map (profile names are output names), then to the name unchanged when the
    node is not present in the model (e.g. an EP-inserted reorder kernel).
    """
    if name in name_to_type:
        return name
    if name.endswith(_NCHWC_SUFFIX):
        trimmed = name[: -len(_NCHWC_SUFFIX)]
        if trimmed in output_to_name:
            return output_to_name[trimmed]
    trimmed = _TOKEN_SUFFIX.sub("", name)
    if trimmed in name_to_type:
        return trimmed
    if name in output_to_name:
        return output_to_name[name]
    return name


def _resolve_node_type(name_to_type: dict[str, str], name: str, line: dict) -> str:
    """Resolve a node's op type, preferring the profile's own ``op_name`` arg."""
    args = line.get("args") or {}
    op_name = args.get("op_name")
    if op_name:
        return op_name
    return name_to_type.get(name, "")


def build_node_maps(onnx_path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Build ``node.name -> op_type`` and ``output -> node.name`` maps.

    External weights are not needed for the graph topology, so they are left
    unloaded to keep this cheap for large models.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    name_to_type: dict[str, str] = {}
    output_to_name: dict[str, str] = {}
    for node in model.graph.node:
        name_to_type[node.name] = node.op_type
        for output in node.output:
            output_to_name[output] = node.name
    return name_to_type, output_to_name


def parse_ort_profile(
    profile_path: str | Path,
    name_to_type: dict[str, str],
    output_to_name: dict[str, str],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Parse an ORT profiling JSON into aggregated per-operator metrics.

    Per-node kernel durations are accumulated across every inference run, the
    leading ``warmup`` samples are dropped, and the remainder is averaged.
    Operators are returned sorted by average duration descending.

    Parameters
    ----------
    profile_path:
        Path to the ORT profiling JSON file.
    name_to_type / output_to_name:
        Graph maps from :func:`build_node_maps`.
    warmup:
        Number of leading (un-measured) runs to drop per operator.
    iterations:
        Number of measured runs (used to bound the kept-sample window).

    Returns:
    -------
    dict with keys:
        ``operators``: list of ``{name, op_path, duration_us, percent_of_total}``
        ``num_samples``: number of measured samples retained.
    """
    with Path(profile_path).open(encoding="utf-8") as f:
        profile = json.load(f)

    # Preserve first-seen (model execution) order while accumulating samples.
    order: list[str] = []
    results: dict[str, list[float]] = {}
    op_types: dict[str, str] = {}

    for line in profile:
        if line.get("cat") != "Node":
            continue
        raw_name = line.get("name", "")
        if not raw_name.endswith(_KERNEL_TIME_SUFFIX):
            continue
        event_name = raw_name[: -len(_KERNEL_TIME_SUFFIX)]
        op_path = resolve_node_name(name_to_type, output_to_name, event_name)
        if op_path not in results:
            results[op_path] = []
            op_types[op_path] = _resolve_node_type(name_to_type, op_path, line)
            order.append(op_path)
        results[op_path].append(float(line.get("dur", 0)))

    # Drop warmup samples per operator. Keeping the trailing samples is robust
    # to the exact count varying (it always reflects the measured runs).
    kept = max(1, iterations)
    averages: dict[str, float] = {}
    for op_path in order:
        samples = results[op_path]
        if len(samples) > warmup:
            samples = samples[warmup:]
        samples = samples[-kept:]
        averages[op_path] = float(np.mean(samples)) if samples else 0.0

    total = sum(averages.values())
    operators = [
        {
            "name": op_types[op_path],
            "op_path": op_path,
            "duration_us": averages[op_path],
            "percent_of_total": (averages[op_path] / total * 100) if total > 0 else 0.0,
        }
        for op_path in order
    ]
    operators.sort(key=lambda op: op["duration_us"], reverse=True)

    return {"operators": operators, "num_samples": kept}
