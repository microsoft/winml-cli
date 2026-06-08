#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Print a matrix of example-eval verdicts across every EP/device folder.

Rows: ``<model>  <task>``
Cols: each ``examples/<ep>/<device>/`` folder.
Cell: worst verdict across precisions for that (model, task, ep, device),
colored P (green), R (yellow), F (red), - (plain).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Reuse parsing/grouping helpers from the test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_example_tests import (  # noqa: E402
    REPO_ROOT,
    build_grouped_configs,
    has_eval_section,
    infer_group_task,
    infer_hf_id,
)

sys.path.insert(0, str(REPO_ROOT / "scripts" / "e2e_eval"))
from utils.accuracy import METRIC_COMPARE_STRATEGY, compute_delta  # type: ignore[import-not-found]  # noqa: E402


DEVICE_NAMES = {"cpu", "gpu", "npu"}

_VERDICT_RANK = {"REGRESSION": 3, "FAIL": 2, "PASS": 1, None: 0}
_VERDICT_CHAR = {"REGRESSION": "R", "FAIL": "F", "PASS": "P", None: "-"}
_ANSI = {"P": "\x1b[32m", "R": "\x1b[33m", "F": "\x1b[31m"}
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


def _worst_verdict(a: str | None, b: str | None) -> str | None:
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
    """Return verdict ('PASS' | 'REGRESSION' | 'FAIL') or None when ungradable."""
    result_json = model_dir / f"{group_stem}_eval_result.json"
    if result_json.with_suffix(".error.txt").exists():
        return "FAIL"
    if result_json.with_suffix(".timeout").exists():
        return "FAIL"
    if not result_json.exists():
        return None

    result = json.loads(result_json.read_text(encoding="utf-8"))
    reg_ds = reg_map.get((hf_id, task))
    if reg_ds is None:
        return None

    metric_name = reg_ds.get("metric")
    winml_key = reg_ds.get("winml_metric_key") or metric_name
    raw = (result.get("metrics") or {}).get(winml_key)
    value = raw.get("value") if isinstance(raw, dict) else raw
    if not isinstance(value, (int, float)):
        return "FAIL"

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
    if not (isinstance(cached, dict) and cached.get("status") == "PASS" and isinstance(bv, (int, float))):
        return None

    delta_abs, delta_rel = compute_delta({"value": float(value)}, {"value": float(bv)})
    delta_key, _thresh_pass, thresh_at_risk, higher = METRIC_COMPARE_STRATEGY.get(
        metric_name, METRIC_COMPARE_STRATEGY["default"]
    )
    chosen = delta_abs if delta_key == "delta_absolute" else delta_rel
    if chosen is None:
        return "FAIL"
    signed = chosen if higher else -chosen
    return "PASS" if signed >= 0 or abs(signed) < thresh_at_risk else "REGRESSION"


def _format_cell(ch: str, width: int, use_color: bool) -> str:
    cell = f"{ch:^{width}}"
    if use_color and ch in _ANSI:
        return f"{_ANSI[ch]}{cell}{_RESET}"
    return cell


def print_summary_table(models_filter: str | None = None) -> None:
    reg_map = _load_registry_map()
    cache = _load_baseline_cache()
    use_color = _color_enabled()
    allowed = set(models_filter.split(",")) if models_filter else None

    columns: list[str] = []
    matrix: dict[tuple[str, str], dict[str, str | None]] = {}

    for ep_dir in sorted((REPO_ROOT / "examples").iterdir()):
        if not ep_dir.is_dir():
            continue
        for device_dir in sorted(ep_dir.iterdir()):
            if not device_dir.is_dir() or device_dir.name not in DEVICE_NAMES:
                continue
            col = f"{ep_dir.name}/{device_dir.name}"
            columns.append(col)
            model_dirs = sorted(d for d in device_dir.iterdir() if d.is_dir())
            if allowed:
                model_dirs = [d for d in model_dirs if d.name in allowed]
            for model_dir, group_stem, group_paths in build_grouped_configs(model_dirs):
                if not any(has_eval_section(p) for p in group_paths):
                    continue
                hf_id = next((infer_hf_id(p) for p in group_paths if infer_hf_id(p)), None)
                task = infer_group_task(group_stem, group_paths)
                if not (hf_id and task):
                    continue
                verdict = _grade_group(model_dir, group_stem, hf_id, task, reg_map, cache)
                row = matrix.setdefault((hf_id, task), {})
                row[col] = _worst_verdict(row.get(col), verdict)

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
    tally = {"P": 0, "R": 0, "F": 0, "-": 0}
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
        f"{_format_cell('F', 1, use_color)}=FAIL  -=no data/no result"
    )
    print(
        f"{legend}   |   P={tally['P']}  R={tally['R']}  F={tally['F']}  -={tally['-']}"
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
    args = parser.parse_args()
    print_summary_table(args.models)


if __name__ == "__main__":
    main()
