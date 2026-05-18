#!/usr/bin/env python3
"""Generate a REPORT.md for an example EP/hardware folder.

Walks ``examples/<ep>/<hw>/`` and produces a Markdown table summarizing
each ``*_config.json`` with relative links to the corresponding
``*_perf.json`` / ``*_eval.json`` / ``*.error.txt`` / ``*.timeout`` artifacts.

Usage:
    python scripts/generate_example_report.py --ep mlas --hardware cpu --title "MLAS (CPU)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def fmt_perf(perf_path: Path, link: str) -> str:
    try:
        data = json.loads(perf_path.read_text())
        lat = data.get("latency_ms", {}).get("mean")
        tput = data.get("throughput", {}).get("samples_per_sec")
        if lat is None or tput is None:
            return f"[link]({link})"
        return f"[{lat:.2f}ms, {tput:.1f}sps]({link})"
    except Exception:
        return f"[link]({link})"


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
            # Skip non-scalars (lists, dicts) to keep table compact
        if not parts:
            return f"[PASS]({link})"
        return f"[{' '.join(parts)}]({link})"
    except Exception:
        return f"[PARSE_ERROR]({link})"


def status_cell(model_dir: Path, stem: str, kind: str) -> str:
    ok = model_dir / f"{stem}_{kind}.json"
    err = model_dir / f"{stem}_{kind}.error.txt"
    timeout = model_dir / f"{stem}_{kind}.timeout"
    slug = model_dir.name
    if ok.exists():
        link = f"{slug}/{ok.name}"
        return fmt_perf(ok, link) if kind == "perf" else fmt_eval(ok, link)
    if err.exists():
        return "FAIL"
    if timeout.exists():
        return "TIMEOUT"
    return "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REPORT.md for an example EP folder")
    parser.add_argument("--ep", required=True, help="EP folder, e.g. mlas")
    parser.add_argument("--hardware", required=True, help="Hardware sub-folder, e.g. cpu")
    parser.add_argument("--title", required=True, help="Display name for report heading")
    args = parser.parse_args()

    ep_dir = REPO_ROOT / "examples" / args.ep / args.hardware
    if not ep_dir.exists():
        raise SystemExit(f"Directory not found: {ep_dir}")

    model_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())
    configs = sorted(cfg for d in model_dirs for cfg in d.glob("*_config.json"))

    def _has(c: Path, kind: str, ext: str) -> bool:
        return (c.parent / f"{c.stem.replace('_config', '')}_{kind}.{ext}").exists()

    perf_pass = sum(1 for c in configs if _has(c, "perf", "json"))
    eval_pass = sum(1 for c in configs if _has(c, "eval", "json"))
    eval_err = sum(1 for c in configs if _has(c, "eval", "error.txt"))
    eval_to = sum(1 for c in configs if _has(c, "eval", "timeout"))

    total = max(len(configs), 1)
    lines: list[str] = [
        f"# {args.title} Test Report",
        "",
        "## Summary",
        "",
        f"- **Models tested**: {len(model_dirs)}",
        f"- **Configs tested**: {len(configs)}",
        f"- **Perf pass rate**: {perf_pass}/{len(configs)} ({100 * perf_pass / total:.0f}%)",
        f"- **Eval pass rate**: {eval_pass}/{len(configs)} ({100 * eval_pass / total:.0f}%)",
        f"- **Non-pass results**: {eval_err} errors, {eval_to} timeouts",
        "",
        "## Results",
        "",
        "| Model | Task | Config | Perf | Eval |",
        "|------|------|------|------|------|",
    ]

    prev_slug = None
    for cfg in configs:
        model_dir = cfg.parent
        slug = model_dir.name
        stem = cfg.stem.replace("_config", "")
        task = stem.rsplit("_", 1)[0] if stem.endswith(("_w8a8", "_w8a16", "_fp16")) else stem
        try:
            data = json.loads(cfg.read_text())
            hf_id = (data.get("quant") or {}).get("model_name") or slug.replace("_", "/", 1)
        except Exception:
            hf_id = slug.replace("_", "/", 1)

        model_cell = hf_id if slug != prev_slug else ""
        prev_slug = slug

        cfg_link = f"[config]({slug}/{cfg.name})"
        perf_cell = status_cell(model_dir, stem, "perf")
        eval_cell = status_cell(model_dir, stem, "eval")
        lines.append(f"| {model_cell} | {task} | {cfg_link} | {perf_cell} | {eval_cell} |")

    out = ep_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
