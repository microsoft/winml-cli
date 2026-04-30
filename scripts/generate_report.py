#!/usr/bin/env python3
"""Generate summary reports for example config test results.

Creates:
- examples/<ep>/REPORT.md for each EP
- examples/SUMMARY.md with overall pass rates
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

EPS = [
    ("amd", "AMD (VitisAI)"),
    ("qnn", "QNN (Qualcomm)"),
    ("ov", "OpenVINO (Intel)"),
]

PRECISIONS = ["w8a8", "w8a16", "fp16"]


def get_status(model_dir: Path, task: str, precision: str, kind: str) -> str:
    base = f"{task}_{precision}_{kind}"
    if (model_dir / f"{base}.json").exists():
        return "PASS"
    if (model_dir / f"{base}.timeout").exists():
        return "TIMEOUT"
    if (model_dir / f"{base}.error.txt").exists():
        return "FAIL"
    return "-"


def get_metric_value(model_dir: Path, task: str, precision: str, kind: str) -> str:
    path = model_dir / f"{task}_{precision}_{kind}.json"
    if not path.exists():
        return "-"
    try:
        data = json.loads(path.read_text())
        if kind == "perf":
            lat = data.get("latency_ms", {})
            tp = data.get("throughput", {})
            mean = lat.get("mean")
            sps = tp.get("samples_per_sec")
            if mean is not None and sps is not None:
                return f"{mean:.2f}ms / {sps:.1f} sps"
            if mean is not None:
                return f"{mean:.2f}ms"
            return "PASS"
        else:
            metrics = data.get("metrics", {})
            # Collect all non-timing metrics
            skip = {"total_time_in_seconds", "samples_per_second", "latency_in_seconds"}
            parts = []
            for k, v in metrics.items():
                if k in skip:
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                elif isinstance(v, (int, str)):
                    parts.append(f"{k}={v}")
                # Skip dicts (per-class metrics) for brevity
            return " ".join(parts) if parts else "PASS"
    except Exception:
        return "PASS"


def slug_to_hf_id(slug: str) -> str:
    return slug.replace("_", "/", 1)


def make_rel_link(model_dir: Path, task: str, precision: str, kind: str, ep_dir: Path) -> str:
    base = f"{task}_{precision}_{kind}.json"
    path = model_dir / base
    if not path.exists():
        # Check timeout/error
        if (model_dir / f"{task}_{precision}_{kind}.timeout").exists():
            return "TIMEOUT"
        if (model_dir / f"{task}_{precision}_{kind}.error.txt").exists():
            return "FAIL"
        return "-"
    rel = path.relative_to(ep_dir).as_posix()
    value = get_metric_value(model_dir, task, precision, kind)
    return f"[{value}]({rel})"


def generate_ep_report(ep_folder: str, ep_name: str) -> tuple[int, int, int]:
    ep_dir = EXAMPLES_DIR / ep_folder
    if not ep_dir.exists():
        return 0, 0, 0

    model_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())

    rows = []
    for model_dir in model_dirs:
        slug = model_dir.name
        hf_id = slug_to_hf_id(slug)
        configs = sorted(model_dir.glob("*_config.json"))
        if not configs:
            continue

        tasks = set()
        for cfg in configs:
            stem = cfg.stem.replace("_config", "")
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                tasks.add(parts[0])

        for task in sorted(tasks):
            row = {"hf_id": hf_id, "slug": slug, "task": task, "results": {}}
            for prec in PRECISIONS:
                perf_link = make_rel_link(model_dir, task, prec, "perf", ep_dir)
                eval_link = make_rel_link(model_dir, task, prec, "eval", ep_dir)
                config_link = f"[config]({slug}/{task}_{prec}_config.json)"
                row["results"][prec] = {
                    "perf_status": get_status(model_dir, task, prec, "perf"),
                    "eval_status": get_status(model_dir, task, prec, "eval"),
                    "perf_link": perf_link,
                    "eval_link": eval_link,
                    "config_link": config_link,
                }
            rows.append(row)

    perf_pass = sum(1 for r in rows if any(v["perf_status"] == "PASS" for v in r["results"].values()))
    eval_pass = sum(1 for r in rows if any(v["eval_status"] == "PASS" for v in r["results"].values()))
    total = len(rows)

    has_perf = any(r["perf_status"] != "-" for row in rows for r in row["results"].values())
    has_eval = any(r["eval_status"] != "-" for row in rows for r in row["results"].values())

    lines = [
        f"# {ep_name} Test Report\n",
        f"## Summary\n",
        f"- **Models tested**: {total}",
    ]
    if has_perf:
        lines.append(f"- **Perf pass rate**: {perf_pass}/{total} ({perf_pass*100//max(total,1)}%)")
    if has_eval:
        lines.append(f"- **Eval pass rate**: {eval_pass}/{total} ({eval_pass*100//max(total,1)}%)")
    lines.extend(["", "## Results\n"])

    # Build header
    cols = ["Model", "Task", "Precision", "Config"]
    if has_perf:
        cols.append("Perf")
    if has_eval:
        cols.append("Eval")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["------"] * len(cols)) + "|")

    for row in rows:
        for i, prec in enumerate(PRECISIONS):
            r = row["results"][prec]
            model_col = row["hf_id"] if i == 0 else ""
            task_col = row["task"] if i == 0 else ""
            cells = [model_col, task_col, prec, r["config_link"]]
            if has_perf:
                cells.append(r["perf_link"])
            if has_eval:
                cells.append(r["eval_link"])
            lines.append("| " + " | ".join(cells) + " |")

    (ep_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"  Written {ep_dir / 'REPORT.md'}")
    return total, perf_pass, eval_pass


def generate_summary(results: list[tuple[str, str, int, int, int]]) -> None:
    lines = [
        "# Example Configs Test Summary\n",
        "## Overview\n",
        "| EP | Models | Perf Pass | Eval Pass | Report |",
        "|----|--------|-----------|-----------|--------|",
    ]
    for ep_folder, ep_name, total, perf_pass, eval_pass in results:
        perf_str = f"{perf_pass}/{total} ({perf_pass*100//max(total,1)}%)"
        eval_str = f"{eval_pass}/{total} ({eval_pass*100//max(total,1)}%)"
        lines.append(f"| {ep_name} | {total} | {perf_str} | {eval_str} | [Report]({ep_folder}/REPORT.md) |")

    (EXAMPLES_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"  Written {EXAMPLES_DIR / 'SUMMARY.md'}")


def main() -> None:
    results = []
    for ep_folder, ep_name in EPS:
        print(f"Generating report for {ep_name}...")
        total, perf_pass, eval_pass = generate_ep_report(ep_folder, ep_name)
        results.append((ep_folder, ep_name, total, perf_pass, eval_pass))

    print("Generating summary...")
    generate_summary(results)
    print("Done!")


if __name__ == "__main__":
    main()
