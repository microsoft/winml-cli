#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Print a matrix of example-eval verdicts across every EP/device folder.

Rows: ``<model>  <task>``
Cols: each ``examples/<ep>/<device>/`` folder.
Cell: best verdict across precisions for that (model, task, ep, device),
colored P (green), R (yellow), F (red), N (cyan, N/A), - (plain).

Verdict semantics:
  PASS       eval_result.json exists, metric value present, passes threshold vs baseline
  REGRESSION eval_result.json exists, but metric missing/None OR fails threshold vs baseline
  FAIL       eval_result.json does not exist (the evaluation itself failed)
  N/A        eval_result.json exists but no baseline is available to compare against
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Reuse parsing/grouping helpers from the test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_example_tests import (
    KNOWN_PRECISIONS,
    REPO_ROOT,
    build_grouped_configs,
    has_eval_section,
    infer_group_task,
    infer_hf_id,
    split_task_precision,
)


sys.path.insert(0, str(REPO_ROOT / "scripts" / "e2e_eval"))
from utils.accuracy import (  # type: ignore[import-not-found]
    METRIC_COMPARE_STRATEGY,
    compute_delta,
)


DEVICE_NAMES = {"cpu", "gpu", "npu"}

# Higher rank = better verdict; used to pick the BEST across precisions.
# Order: PASS > REGRESSION > FAIL > N/A > no data.
_VERDICT_RANK = {"PASS": 4, "REGRESSION": 3, "FAIL": 2, "N/A": 1, None: 0}
_VERDICT_CHAR = {"PASS": "P", "REGRESSION": "R", "N/A": "N", "FAIL": "F", None: "-"}
_ANSI = {"P": "\x1b[32m", "R": "\x1b[33m", "F": "\x1b[31m", "N": "\x1b[36m"}
_RESET = "\x1b[0m"


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Side effect: enables ANSI VT processing in legacy cmd.exe.
        os.system("")
    return True


def _best_verdict(a: str | None, b: str | None) -> str | None:
    return a if _VERDICT_RANK.get(a, 0) >= _VERDICT_RANK.get(b, 0) else b


def _load_registry_map() -> dict[tuple[str, str], dict]:
    reg_path = REPO_ROOT / "scripts/e2e_eval/testsets/models_with_acc.json"
    reg_map: dict[tuple[str, str], dict] = {}
    for e in json.loads(reg_path.read_text(encoding="utf-8")):
        ds = e.get("dataset_config")
        if not isinstance(ds, dict):
            continue
        cfg = {**ds}
        if "path" in cfg:
            cfg["dataset"] = cfg.pop("path")
        if "name" in cfg:
            cfg["dataset_config"] = cfg.pop("name")
        if "samples" in cfg:
            cfg["num_samples"] = cfg.pop("samples")
        reg_map[(e["hf_id"], e["task"])] = cfg
    return reg_map


def _load_baseline_cache() -> dict:
    cache_path = REPO_ROOT / "scripts/e2e_eval/cache/baseline_cache.json"
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _grade_group(
    model_dir: Path,
    group_stem: str,
    hf_id: str,
    task: str,
    reg_map: dict[tuple[str, str], dict],
    cache: dict,
) -> str | None:
    """Return verdict ('PASS' | 'REGRESSION' | 'FAIL' | 'N/A' | None).

    PASS       => eval_result.json exists, metric present, passes threshold.
    REGRESSION => eval_result.json exists, but metric missing/None OR fails
                  the threshold vs baseline.
    N/A        => eval_result.json exists but no baseline is available.
    FAIL       => no eval_result.json, but a build/eval error or timeout marker
                  exists on disk (the attempt failed).
    None       => no result and no error marker -> never evaluated (no data).
    """
    result_json = model_dir / f"{group_stem}_eval_result.json"
    if not result_json.exists():
        # Distinguish an attempted-but-failed run (error/timeout marker present
        # -> FAIL) from a pair that was simply never evaluated (no markers ->
        # no data).
        failure_markers = (
            model_dir / f"{group_stem}_eval_result.error.txt",
            model_dir / f"{group_stem}_eval_result.timeout",
            model_dir / f"{group_stem}_build_result.error.txt",
            model_dir / f"{group_stem}_build_result.timeout",
        )
        if any(marker.exists() for marker in failure_markers):
            return "FAIL"
        return None

    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "REGRESSION"

    reg_ds = reg_map.get((hf_id, task))
    if reg_ds is None:
        return "N/A"

    ck = "|".join(
        [
            hf_id,
            task,
            reg_ds.get("dataset", ""),
            reg_ds.get("dataset_config", ""),
            reg_ds.get("split", ""),
            str(reg_ds.get("num_samples", 1000)),
        ]
    )
    cached = cache.get(ck)
    bv = (cached or {}).get("metric", {}).get("value") if isinstance(cached, dict) else None
    if not (
        isinstance(cached, dict)
        and cached.get("status") == "PASS"
        and isinstance(bv, (int, float))
    ):
        return "N/A"

    metric_name = reg_ds.get("metric")
    winml_key = reg_ds.get("winml_metric_key") or metric_name
    raw = (result.get("metrics") or {}).get(winml_key)
    value = raw.get("value") if isinstance(raw, dict) else raw
    if not isinstance(value, (int, float)):
        return "REGRESSION"

    delta_abs, delta_rel = compute_delta({"value": float(value)}, {"value": float(bv)})
    delta_key, _thresh_pass, thresh_at_risk, higher = METRIC_COMPARE_STRATEGY.get(
        metric_name, METRIC_COMPARE_STRATEGY["default"]
    )
    chosen = delta_abs if delta_key == "delta_absolute" else delta_rel
    if chosen is None:
        return "REGRESSION"
    signed = chosen if higher else -chosen
    # No 'AT_RISK' verdict here: anything strictly inside the at-risk threshold
    # counts as PASS; only deltas at/beyond it become REGRESSION.
    return "PASS" if signed >= 0 or abs(signed) < thresh_at_risk else "REGRESSION"


def _format_cell(ch: str, width: int, use_color: bool) -> str:
    cell = f"{ch:^{width}}"
    if use_color and ch in _ANSI:
        return f"{_ANSI[ch]}{cell}{_RESET}"
    return cell


def print_summary_table(
    models_filter: str | None = None,
    eps_filter: str | None = None,
    devices_filter: str | None = None,
    precision_filter: str | None = None,
) -> None:
    reg_map = _load_registry_map()
    cache = _load_baseline_cache()
    use_color = _color_enabled()
    allowed = set(models_filter.split(",")) if models_filter else None
    allowed_eps = {e.strip().lower() for e in eps_filter.split(",")} if eps_filter else None
    allowed_devices = (
        {d.strip().lower() for d in devices_filter.split(",")} if devices_filter else None
    )

    columns: list[str] = []
    matrix: dict[tuple[str, str], dict[str, str | None]] = {}

    for ep_dir in sorted((REPO_ROOT / "examples").iterdir()):
        if not ep_dir.is_dir():
            continue
        if allowed_eps and ep_dir.name.lower() not in allowed_eps:
            continue
        for device_dir in sorted(ep_dir.iterdir()):
            if not device_dir.is_dir() or device_dir.name not in DEVICE_NAMES:
                continue
            if allowed_devices and device_dir.name.lower() not in allowed_devices:
                continue
            col = f"{ep_dir.name}/{device_dir.name}"
            columns.append(col)
            model_dirs = sorted(d for d in device_dir.iterdir() if d.is_dir())
            if allowed:
                model_dirs = [d for d in model_dirs if d.name in allowed]
            for model_dir in model_dirs:
                # Gather every group (all precisions, including ones without an
                # eval section) so row membership reflects the full eval set
                # while per-precision grading can still report N/A.
                prec_map: dict[str, dict[str | None, tuple[str, list[Path]]]] = {}
                for _md, group_stem, group_paths in build_grouped_configs([model_dir]):
                    task_name, group_precision = split_task_precision(group_stem)
                    prec_map.setdefault(task_name, {})[group_precision] = (
                        group_stem,
                        group_paths,
                    )

                for task_name, groups_by_prec in prec_map.items():
                    eval_groups = {
                        prec: (stem, paths)
                        for prec, (stem, paths) in groups_by_prec.items()
                        if any(has_eval_section(p) for p in paths)
                    }
                    if not eval_groups:
                        # This (model, task) is not part of the eval set.
                        continue
                    if precision_filter is not None and precision_filter not in eval_groups:
                        # When filtering by precision, only pairs that have an
                        # eval-section config for that precision are rows.
                        continue

                    hf_id = task = None
                    for stem, paths in eval_groups.values():
                        hf_id = next((infer_hf_id(p) for p in paths if infer_hf_id(p)), None)
                        task = infer_group_task(stem, paths)
                        if hf_id and task:
                            break
                    if not (hf_id and task):
                        continue

                    row = matrix.setdefault((hf_id, task), {})
                    # Best verdict across the precisions in range. --precision
                    # only narrows the range to a single precision; the
                    # best-verdict logic is identical either way. A precision
                    # with no result and no error marker contributes no data.
                    if precision_filter is None:
                        range_groups = list(eval_groups.values())
                    else:
                        target = eval_groups.get(precision_filter)
                        range_groups = [target] if target is not None else []
                    verdict: str | None = None
                    for stem, _paths in range_groups:
                        verdict = _best_verdict(
                            verdict,
                            _grade_group(model_dir, stem, hf_id, task, reg_map, cache),
                        )
                    row[col] = _best_verdict(row.get(col), verdict)

    if not matrix:
        print("No example results found.")
        return

    rows = sorted(matrix.keys())
    model_w = max(len("model"), max(len(hf) for hf, _ in rows))
    task_w = max(len("task"), max(len(t) for _, t in rows))
    col_w = {c: max(len(c), 1) for c in columns}

    header = f"{'model':<{model_w}}  {'task':<{task_w}}  " + "  ".join(
        f"{c:^{col_w[c]}}" for c in columns
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    tally = {"P": 0, "R": 0, "F": 0, "N": 0, "-": 0}
    for hf_id, task in rows:
        row = matrix[(hf_id, task)]
        cells = []
        for c in columns:
            ch = _VERDICT_CHAR[row.get(c)] if c in row else "-"
            tally[ch] += 1
            cells.append(_format_cell(ch, col_w[c], use_color))
        print(f"{hf_id:<{model_w}}  {task:<{task_w}}  " + "  ".join(cells))
    print(sep)
    legend = (
        f"Legend: {_format_cell('P', 1, use_color)}=PASS  "
        f"{_format_cell('R', 1, use_color)}=REGRESSION  "
        f"{_format_cell('F', 1, use_color)}=FAIL  "
        f"{_format_cell('N', 1, use_color)}=N/A (no baseline)  -=no data"
    )
    print(
        f"{legend}   |   P={tally['P']}  R={tally['R']}  F={tally['F']}  "
        f"N={tally['N']}  -={tally['-']}  total={sum(tally.values())}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a matrix of example-eval verdicts (model+task x ep/device)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model slugs to restrict the rows.",
    )
    parser.add_argument(
        "--ep",
        "--eps",
        dest="ep",
        type=str,
        default=None,
        help="Comma-separated EP folder names (e.g. 'qnn,openvino') to restrict the columns.",
    )
    parser.add_argument(
        "--device",
        "--devices",
        dest="device",
        type=str,
        default=None,
        help="Comma-separated device names (cpu,gpu,npu) to restrict the columns.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=KNOWN_PRECISIONS,
        help=(
            "Restrict each cell to a single precision (e.g. 'fp16'). "
            "Without it, cells show the best verdict across precisions."
        ),
    )
    args = parser.parse_args()
    print_summary_table(args.models, args.ep, args.device, args.precision)


if __name__ == "__main__":
    main()
