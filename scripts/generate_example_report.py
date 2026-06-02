#!/usr/bin/env python3
"""Generate a REPORT.md for an example EP/hardware folder.

Walks ``examples/<ep>/<hw>/`` and produces a Markdown table summarizing
each ``*_config.json`` with relative links to the corresponding
``*_eval_result.json`` / ``*.error.txt`` / ``*.timeout`` artifacts.

Usage:
    python scripts/generate_example_report.py --ep openvino --hardware npu --title "OpenVINO (Intel, NPU)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PRECISIONS = ("fp16", "w8a16", "w8a8")


def fmt_eval(eval_path: Path, link: str) -> str:
    try:
        data = json.loads(eval_path.read_text())
        metrics = data.get("metrics") or {}
        skip_keys = {"total_time_in_seconds", "samples_per_second", "latency_in_seconds"}
        parts: list[str] = []
        for k, v in metrics.items():
            if k in skip_keys:
                continue
            if isinstance(v, bool):
                parts.append(f"{k}={v}")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, int):
                parts.append(f"{k}={v}")
        if not parts:
            return f"PASS ([metric]({link}))"
        return f"{', '.join(parts)} ([metric]({link}))"
    except Exception:
        return f"PARSE_ERROR ([metric]({link}))"


def status_cell(model_dir: Path, stem: str, kind: str) -> str:
    ok = model_dir / f"{stem}_{kind}_result.json"
    err = model_dir / f"{stem}_{kind}_result.error.txt"
    timeout = model_dir / f"{stem}_{kind}_result.timeout"
    slug = model_dir.name
    if ok.exists():
        link = f"./{slug}/{ok.name}"
        return fmt_eval(ok, link)
    if err.exists():
        return "FAIL"
    if timeout.exists():
        return "TIMEOUT"
    return "\u2014"


def extract_precision(stem: str) -> str:
    for p in PRECISIONS:
        if stem.endswith(f"_{p}"):
            return p
    return ""


def extract_task(stem: str) -> str:
    for p in PRECISIONS:
        if stem.endswith(f"_{p}"):
            return stem[: -(len(p) + 1)]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REPORT.md for an example EP folder")
    parser.add_argument("--ep", required=True, help="EP folder, e.g. openvino")
    parser.add_argument("--hardware", required=True, help="Hardware sub-folder, e.g. npu")
    parser.add_argument("--title", required=True, help="Display name for report heading")
    args = parser.parse_args()

    ep_dir = REPO_ROOT / "examples" / args.ep / args.hardware
    if not ep_dir.exists():
        raise SystemExit(f"Directory not found: {ep_dir}")

    model_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())
    configs = sorted(cfg for d in model_dirs for cfg in d.glob("*_config.json"))

    def _has(c: Path, kind: str, ext: str) -> bool:
        stem = c.stem.replace("_config", "")
        return (c.parent / f"{stem}_{kind}_result.{ext}").exists()

    # Overall stats
    eval_pass = sum(1 for c in configs if _has(c, "eval", "json"))
    eval_fail = sum(1 for c in configs if _has(c, "eval", "error.txt"))
    eval_timeout = sum(1 for c in configs if _has(c, "eval", "timeout"))
    total = max(len(configs), 1)

    # Per-precision stats
    prec_stats: dict[str, dict] = {}
    for p in PRECISIONS:
        p_configs = [c for c in configs if c.stem.replace("_config", "").endswith(f"_{p}")]
        if not p_configs:
            continue
        p_models = len({c.parent.name for c in p_configs})
        p_total = len(p_configs)
        p_eval = sum(1 for c in p_configs if _has(c, "eval", "json"))
        p_fail = sum(1 for c in p_configs if _has(c, "eval", "error.txt"))
        p_timeout = sum(1 for c in p_configs if _has(c, "eval", "timeout"))
        prec_stats[p] = {
            "models": p_models,
            "configs": p_total,
            "eval_pass": p_eval,
            "eval_fail": p_fail,
            "eval_timeout": p_timeout,
        }

    lines: list[str] = [
        f"# {args.title} Report",
        "",
        "## Summary",
        "",
        f"- Models: {len(model_dirs)}",
        f"- Configs: {len(configs)}",
        f"- Eval Pass: {eval_pass}/{len(configs)} ({100 * eval_pass / total:.0f}%)",
        f"- Eval Fail: {eval_fail}/{len(configs)} ({100 * eval_fail / total:.0f}%)",
        f"- Eval Timeout: {eval_timeout}/{len(configs)} ({100 * eval_timeout / total:.0f}%)",
    ]

    if prec_stats:
        lines += [
            "",
            "### Per-precision breakdown",
            "",
            "| Precision | Models | Configs | Eval Pass | Eval Fail | Eval Timeout |",
            "|---|---|---|---|---|---|",
        ]
        for p, s in prec_stats.items():
            pt = max(s["configs"], 1)
            lines.append(
                f"| {p} | {s['models']} | {s['configs']} "
                f"| {s['eval_pass']}/{s['configs']} ({100 * s['eval_pass'] / pt:.0f}%) "
                f"| {s['eval_fail']}/{s['configs']} ({100 * s['eval_fail'] / pt:.0f}%) "
                f"| {s['eval_timeout']}/{s['configs']} ({100 * s['eval_timeout'] / pt:.0f}%) |"
            )

    lines += [
        "",
        "## Results",
        "",
        "| Model | Task | Precision | Config | Eval |",
        "|---|---|---|---|---|",
    ]

    prev_slug = None
    for cfg in configs:
        model_dir = cfg.parent
        slug = model_dir.name
        stem = cfg.stem.replace("_config", "")
        task = extract_task(stem)
        precision = extract_precision(stem)

        hf_id = slug.replace("_", "/", 1)

        model_cell = hf_id if slug != prev_slug else ""
        prev_slug = slug

        cfg_link = f"[config](./{slug}/{cfg.name})"
        eval_cell = status_cell(model_dir, stem, "eval")
        lines.append(f"| {model_cell} | {task} | {precision} | {cfg_link} | {eval_cell} |")

    out = ep_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
