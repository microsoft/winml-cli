#!/usr/bin/env python3
"""Generate a REPORT.md for an example EP/hardware folder.

Walks ``examples/<ep>/<hw>/`` and produces a Markdown table summarizing
each canonical (model, task) config with relative links to the
corresponding ``*_eval_result.json`` / ``*_perf_result.json`` /
``*.error.txt`` / ``*.timeout`` artifacts.

Counting basis:
- Only (model_slug, task) pairs in ``scripts/e2e_eval/testsets/models_57.txt``
  are counted.
- Composite models emit multiple split configs sharing one stem; they are
  counted as a single config group (one eval / one perf entry).

Usage:
    python scripts/generate_example_report.py --ep openvino --hardware npu --title "OpenVINO (Intel, NPU)"
    python scripts/generate_example_report.py --all   # regenerate every EP/hw row
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
MODELS_57 = REPO_ROOT / "scripts" / "e2e_eval" / "testsets" / "models_57.txt"

PRECISIONS = ("fp16", "w8a16", "w8a8")
_NPU_PRECISION_RE = re.compile(r"_(fp16|w8a16|w8a8)$")
_CONFIG_NAME_RE = re.compile(r"^(?P<stem>.+?)_config(?:_(?P<role>.+))?\.json$")

# (title, ep, hardware) for --all
ROWS: list[tuple[str, str, str]] = [
    ("AMD (VitisAI, NPU)", "vitisai", "npu"),
    ("QNN (Qualcomm, NPU)", "qnn", "npu"),
    ("OpenVINO (Intel, NPU)", "openvino", "npu"),
    ("QNN (Qualcomm, GPU)", "qnn", "gpu"),
    ("OpenVINO (Intel, CPU)", "openvino", "cpu"),
    ("OpenVINO (Intel, GPU)", "openvino", "gpu"),
    ("DML (GPU)", "dml", "gpu"),
    ("MLAS (CPU)", "mlas", "cpu"),
    ("NVIDIA TensorRT RTX (GPU)", "nv_tensorrt_rtx", "gpu"),
]


def load_target_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for line in MODELS_57.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hf_id, task = s.split("|", 1)
        pairs.add((hf_id.strip().replace("/", "_"), task.strip()))
    return pairs


def extract_precision(stem: str) -> str:
    m = _NPU_PRECISION_RE.search(stem)
    return m.group(1) if m else ""


def extract_task(stem: str, hardware: str) -> str:
    if hardware == "npu":
        m = _NPU_PRECISION_RE.search(stem)
        if m:
            return stem[: m.start()]
        return stem
    return stem.removesuffix("_fp16")


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



def status_cell(model_dir: Path, stem: str, kind: str, fmt) -> str:
    ok = model_dir / f"{stem}_{kind}_result.json"
    err = model_dir / f"{stem}_{kind}_result.error.txt"
    timeout = model_dir / f"{stem}_{kind}_result.timeout"
    slug = model_dir.name
    if ok.exists():
        return fmt(ok, f"./{slug}/{ok.name}")
    if err.exists():
        return "FAIL"
    if timeout.exists():
        return "TIMEOUT"
    return "\u2014"


def generate(ep: str, hardware: str, title: str) -> None:
    ep_dir = EXAMPLES / ep / hardware
    if not ep_dir.exists():
        raise SystemExit(f"Directory not found: {ep_dir}")

    target_pairs = load_target_pairs()

    groups: list[tuple[Path, str]] = []
    for model_dir in sorted(d for d in ep_dir.iterdir() if d.is_dir()):
        seen: set[str] = set()
        for cfg in sorted(model_dir.glob("*_config*.json")):
            m = _CONFIG_NAME_RE.match(cfg.name)
            if not m:
                continue
            stem = m.group("stem")
            if stem in seen:
                continue
            seen.add(stem)
            task = extract_task(stem, hardware)
            if (model_dir.name, task) not in target_pairs:
                continue
            groups.append((model_dir, stem))

    def has(model_dir: Path, stem: str, kind: str, ext: str) -> bool:
        return (model_dir / f"{stem}_{kind}_result.{ext}").exists()

    total = max(len(groups), 1)
    eval_pass = sum(1 for d, s in groups if has(d, s, "eval", "json"))
    distinct_pairs = {(d.name, extract_task(s, hardware)) for d, s in groups}

    prec_stats: dict[str, dict] = {}
    if hardware == "npu":
        for p in PRECISIONS:
            p_groups = [(d, s) for d, s in groups if extract_precision(s) == p]
            if not p_groups:
                continue
            prec_stats[p] = {
                "pairs": len({(d.name, extract_task(s, hardware)) for d, s in p_groups}),
                "configs": len(p_groups),
                "eval_pass": sum(1 for d, s in p_groups if has(d, s, "eval", "json")),
            }

    lines: list[str] = [
        f"# {title} Report",
        "",
        "## Summary",
        "",
        "Counts canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/models_57.txt`.",
        "",
        f"- (Model, Task): {len(distinct_pairs)}",
        f"- Configs: {len(groups)}",
        f"- Eval Pass: {eval_pass}/{len(groups)} ({100 * eval_pass / total:.0f}%)",
    ]

    if prec_stats:
        lines += [
            "",
            "### Per-precision breakdown",
            "",
            "| Precision | (Model, Task) | Configs | Eval Pass |",
            "|---|---|---|---|",
        ]
        for p, s in prec_stats.items():
            pt = max(s["configs"], 1)
            lines.append(
                f"| {p} | {s['pairs']} | {s['configs']} "
                f"| {s['eval_pass']}/{s['configs']} ({100 * s['eval_pass'] / pt:.0f}%) |"
            )

    lines += [
        "",
        "## Results",
        "",
        "| Model | Task | Precision | Config | Eval |",
        "|---|---|---|---|---|",
    ]

    prev_slug = None
    for model_dir, stem in groups:
        slug = model_dir.name
        task = extract_task(stem, hardware)
        precision = extract_precision(stem)
        hf_id = slug.replace("_", "/", 1)

        model_cell = hf_id if slug != prev_slug else ""
        prev_slug = slug

        primary_cfg = model_dir / f"{stem}_config.json"
        if not primary_cfg.exists():
            split = sorted(model_dir.glob(f"{stem}_config_*.json"))
            if split:
                primary_cfg = split[0]
        cfg_link = f"[config](./{slug}/{primary_cfg.name})"
        eval_cell = status_cell(model_dir, stem, "eval", fmt_eval)
        lines.append(
            f"| {model_cell} | {task} | {precision} | {cfg_link} | {eval_cell} |"
        )

    out = ep_dir / "REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {out}  ((model,task)={len(distinct_pairs)}, configs={len(groups)}, "
        f"eval={eval_pass})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REPORT.md for an example EP folder")
    parser.add_argument("--ep", help="EP folder, e.g. openvino")
    parser.add_argument("--hardware", help="Hardware sub-folder, e.g. npu")
    parser.add_argument("--title", help="Display name for report heading")
    parser.add_argument("--all", action="store_true", help="Regenerate all known EP/hw reports")
    args = parser.parse_args()

    if args.all:
        for title, ep, hw in ROWS:
            generate(ep, hw, title)
        return

    if not (args.ep and args.hardware and args.title):
        parser.error("--ep, --hardware, --title required unless --all is used")
    generate(args.ep, args.hardware, args.title)


if __name__ == "__main__":
    main()
