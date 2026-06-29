# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Compare benchmark counters against compile-time schedule trace per layer.

Compares real/cpu time (CSV) against DPU/DMA time (JSON trace) on a per-layer
basis.

Inputs
------
- benchmark_average_counters_report.csv : ';'-delimited, has `layerName`,
  `realTime (ms)`, `cpuTime (ms)`.
- compileTimeScheduleTrace.json : Chrome-trace events; entries with
  `"cat": "Layer"` carry `args["DPU time"]` and `args["DMA time"]` formatted
  like "49us 736ns".

For every layer present in both files we compute the symmetric relative
difference between each (csv-metric, json-metric) pair and decide which of two
candidate mappings aligns better:

    Mapping A : real <-> DPU , cpu <-> DMA
    Mapping B : real <-> DMA , cpu <-> DPU

A mapping is "more similar" when the average of its two relative differences
(over all matched layers) is smaller. The four pairwise comparisons
(real-DPU, real-DMA, cpu-DPU, cpu-DMA) are all reported per layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent

# value + unit, e.g. "49us", "736ns", "1.2ms"
_TIME_TOKEN = re.compile(r"([\d.]+)\s*(ms|us|ns|s)", re.IGNORECASE)
_UNIT_TO_MS = {"s": 1000.0, "ms": 1.0, "us": 1e-3, "ns": 1e-6}


def parse_time_ms(text: str | None) -> float | None:
    """Convert a trace time string like '49us 736ns' into milliseconds.

    Returns None when no recognizable time token is present.
    """
    if not text:
        return None
    tokens = _TIME_TOKEN.findall(text)
    if not tokens:
        return None
    return sum(float(value) * _UNIT_TO_MS[unit.lower()] for value, unit in tokens)


def rel_diff(a: float | None, b: float | None) -> float | None:
    """Symmetric relative difference: |a-b| / ((|a|+|b|)/2).

    Returns None if either value is missing; 0.0 when both are exactly 0.
    """
    if a is None or b is None:
        return None
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return abs(a - b) / denom


def ratio(num: float | None, den: float | None) -> float | None:
    """Ratio num/den, with num (DPU/DMA) on top so a 0 numerator is safe.

    Returns None if den is missing or 0.
    """
    if num is None or not den:
        return None
    return num / den


def load_csv(path: Path) -> dict[str, dict[str, float]]:
    """Map layerName -> {'real': ms, 'cpu': ms}."""
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row["layerName"]
            try:
                real = float(row["realTime (ms)"])
                cpu = float(row["cpuTime (ms)"])
            except (KeyError, ValueError):
                continue
            out[name] = {"real": real, "cpu": cpu}
    return out


def load_trace(path: Path) -> dict[str, dict[str, float]]:
    """Map layer name -> {'dpu': ms, 'dma': ms}.

    Multiple trace entries sharing a name are summed (a layer can be split
    across clusters/threads).
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for event in data.get("traceEvents", []):
        if event.get("cat") != "Layer":
            continue
        name = event.get("name")
        args = event.get("args", {})
        dpu = parse_time_ms(args.get("DPU time"))
        dma = parse_time_ms(args.get("DMA time"))
        if dpu is None and dma is None:
            continue
        agg = out.setdefault(name, {"dpu": 0.0, "dma": 0.0})
        if dpu is not None:
            agg["dpu"] += dpu
        if dma is not None:
            agg["dma"] += dma
    return out


def main() -> None:
    """Run the per-layer comparison and write the report CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "benchmark_average_counters_report.csv")
    parser.add_argument("--trace", type=Path, default=HERE / "compileTimeScheduleTrace.json")
    parser.add_argument("--out", type=Path, default=HERE / "mapping_comparison_report.csv")
    args = parser.parse_args()

    csv_data = load_csv(args.csv)
    trace_data = load_trace(args.trace)

    matched = sorted(set(csv_data) & set(trace_data))
    only_csv = sorted(set(csv_data) - set(trace_data))
    only_trace = sorted(set(trace_data) - set(csv_data))

    rows = []
    # accumulators for the two candidate mappings
    sum_a, sum_b = 0.0, 0.0  # A: real-DPU + cpu-DMA ; B: real-DMA + cpu-DPU
    n_a, n_b = 0, 0

    for name in matched:
        real = csv_data[name]["real"]
        cpu = csv_data[name]["cpu"]
        dpu = trace_data[name]["dpu"]
        dma = trace_data[name]["dma"]

        rd_real_dpu = rel_diff(real, dpu)
        rd_real_dma = rel_diff(real, dma)
        rd_cpu_dpu = rel_diff(cpu, dpu)
        rd_cpu_dma = rel_diff(cpu, dma)

        # Mapping A score = avg(real-DPU, cpu-DMA)
        a_parts = [v for v in (rd_real_dpu, rd_cpu_dma) if v is not None]
        b_parts = [v for v in (rd_real_dma, rd_cpu_dpu) if v is not None]
        map_a = sum(a_parts) / len(a_parts) if a_parts else None
        map_b = sum(b_parts) / len(b_parts) if b_parts else None
        if map_a is not None:
            sum_a += map_a
            n_a += 1
        if map_b is not None:
            sum_b += map_b
            n_b += 1

        better = None
        if map_a is not None and map_b is not None:
            better = "A (real~DPU,cpu~DMA)" if map_a <= map_b else "B (real~DMA,cpu~DPU)"

        rows.append(
            {
                "layerName": name,
                "real_ms": f"{real:.6f}",
                "cpu_ms": f"{cpu:.6f}",
                "dpu_ms": f"{dpu:.6f}",
                "dma_ms": f"{dma:.6f}",
                "reldiff_real_dpu": _fmt(rd_real_dpu),
                "reldiff_real_dma": _fmt(rd_real_dma),
                "reldiff_cpu_dpu": _fmt(rd_cpu_dpu),
                "reldiff_cpu_dma": _fmt(rd_cpu_dma),
                "ratio_dpu_real": _fmt(ratio(dpu, real)),
                "ratio_dpu_cpu": _fmt(ratio(dpu, cpu)),
                "ratio_dma_real": _fmt(ratio(dma, real)),
                "ratio_dma_cpu": _fmt(ratio(dma, cpu)),
                "mapA_score": _fmt(map_a),
                "mapB_score": _fmt(map_b),
                "better_mapping": better or "",
            }
        )

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_a = sum_a / n_a if n_a else float("nan")
    avg_b = sum_b / n_b if n_b else float("nan")

    print(f"Matched layers : {len(matched)}")
    print(f"Only in CSV    : {len(only_csv)}")
    print(f"Only in trace  : {len(only_trace)}")
    print()
    print("Average relative difference over all matched layers:")
    print(f"  Mapping A (real~DPU, cpu~DMA): {avg_a:.4f}  ({avg_a * 100:.2f}%)")
    print(f"  Mapping B (real~DMA, cpu~DPU): {avg_b:.4f}  ({avg_b * 100:.2f}%)")
    print()
    if avg_a < avg_b:
        print(f">>> Mapping A is more similar (lower avg rel diff by {avg_b - avg_a:.4f}).")
    elif avg_b < avg_a:
        print(f">>> Mapping B is more similar (lower avg rel diff by {avg_a - avg_b:.4f}).")
    else:
        print(">>> Both mappings are equally similar.")
    print()
    print(f"Per-layer report written to: {args.out}")


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
