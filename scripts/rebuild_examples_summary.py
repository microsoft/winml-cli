#!/usr/bin/env python3
"""Rebuild examples/summary.md from real config/result files (eval-only).

Counts per row:
- Models       : distinct model slugs that have at least one config
- Configs      : config files in the row's bucket
- Eval Pass    : sibling *_eval_result.json exists
- Eval Fail    : sibling *_eval_result.error.txt exists
- Eval Timeout : sibling *_eval_result.timeout exists

Buckets:
- For NPU folders, rows are split by precision (fp16 / w8a16 / w8a8)
- For CPU/GPU folders, single row
"""

from __future__ import annotations

import re
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MODELS_57 = Path(__file__).resolve().parents[1] / "scripts" / "e2e_eval" / "testsets" / "models_57.txt"

ROWS: list[tuple[str, str, str, str | None, str]] = [
    ("AMD (VitisAI, NPU) - fp16",        "vitisai",         "npu", "fp16",  "vitisai/npu/REPORT.md"),
    ("AMD (VitisAI, NPU) - w8a16",       "vitisai",         "npu", "w8a16", "vitisai/npu/REPORT.md"),
    ("AMD (VitisAI, NPU) - w8a8",        "vitisai",         "npu", "w8a8",  "vitisai/npu/REPORT.md"),
    ("QNN (Qualcomm, NPU) - fp16",       "qnn",             "npu", "fp16",  "qnn/npu/REPORT.md"),
    ("QNN (Qualcomm, NPU) - w8a16",      "qnn",             "npu", "w8a16", "qnn/npu/REPORT.md"),
    ("QNN (Qualcomm, NPU) - w8a8",       "qnn",             "npu", "w8a8",  "qnn/npu/REPORT.md"),
    ("OpenVINO (Intel, NPU) - fp16",     "openvino",        "npu", "fp16",  "openvino/npu/REPORT.md"),
    ("OpenVINO (Intel, NPU) - w8a16",    "openvino",        "npu", "w8a16", "openvino/npu/REPORT.md"),
    ("OpenVINO (Intel, NPU) - w8a8",     "openvino",        "npu", "w8a8",  "openvino/npu/REPORT.md"),
    ("QNN (Qualcomm, GPU)",              "qnn",             "gpu", None,    "qnn/gpu/REPORT.md"),
    ("OpenVINO (Intel, CPU)",            "openvino",        "cpu", None,    "openvino/cpu/REPORT.md"),
    ("OpenVINO (Intel, GPU)",            "openvino",        "gpu", None,    "openvino/gpu/REPORT.md"),
    ("DML (GPU)",                        "dml",             "gpu", None,    "dml/gpu/REPORT.md"),
    ("MLAS (CPU)",                       "mlas",            "cpu", None,    "mlas/cpu/REPORT.md"),
    ("NVIDIA TensorRT RTX (GPU)",        "nv_tensorrt_rtx", "gpu", None,    "nv_tensorrt_rtx/gpu/REPORT.md"),
]

_NPU_PRECISION_RE = re.compile(r"_(fp16|w8a16|w8a8)$")


def load_target_pairs() -> set[tuple[str, str]]:
    """Load canonical target set as (model_slug, task)."""
    pairs: set[tuple[str, str]] = set()
    for line in MODELS_57.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hf_id, task = s.split("|", 1)
        pairs.add((hf_id.strip().replace("/", "_"), task.strip()))
    return pairs


def collect(
    folder: Path,
    hardware: str,
    precision_filter: str | None,
    target_pairs: set[tuple[str, str]],
) -> tuple[int, int, int, int, int]:
    """Return ((model,task), configs, eval_pass, eval_fail, eval_timeout)."""
    model_tasks: set[tuple[str, str]] = set()
    configs = 0
    eval_pass = 0
    eval_fail = 0
    eval_timeout = 0

    if not folder.is_dir():
        return 0, 0, 0, 0, 0

    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        for cfg in model_dir.glob("*_config.json"):
            stem = cfg.name[: -len("_config.json")]
            task = stem
            if hardware == "npu":
                m = _NPU_PRECISION_RE.search(stem)
                if m:
                    task = stem[: m.start()]
            else:
                task = stem.removesuffix("_fp16")

            if (model_dir.name, task) not in target_pairs:
                continue

            if hardware == "npu" and precision_filter:
                m = _NPU_PRECISION_RE.search(stem)
                if not m or m.group(1) != precision_filter:
                    continue
            configs += 1
            model_tasks.add((model_dir.name, task))
            if (model_dir / f"{stem}_eval_result.json").exists():
                eval_pass += 1
            elif (model_dir / f"{stem}_eval_result.error.txt").exists():
                eval_fail += 1
            elif (model_dir / f"{stem}_eval_result.timeout").exists():
                eval_timeout += 1

    return len(model_tasks), configs, eval_pass, eval_fail, eval_timeout


def main() -> int:
    target_pairs = load_target_pairs()

    out_lines: list[str] = [
        "# Example Configs Test Summary",
        "",
        "## Overview",
        "",
        "Count basis is canonical `(model, task)` pairs from `scripts/e2e_eval/testsets/models_57.txt`.",
        "",
        "| EP | (Model, Task) | Configs | Eval Pass | Eval Fail | Eval Timeout | Report |",
        "|----|---------------|---------|-----------|-----------|--------------|--------|",
    ]

    for label, ep, hw, prec, report in ROWS:
        model_tasks, configs, ok, fail, tmo = collect(EXAMPLES / ep / hw, hw, prec, target_pairs)

        def pct(x: int, tot: int) -> str:
            return f"{x}/{tot} ({100 * x / tot:.0f}%)" if tot else f"{x}/0 (0%)"

        out_lines.append(
            f"| {label} | {model_tasks} | {configs} | {pct(ok, configs)} | {pct(fail, configs)} | {pct(tmo, configs)} | [Report]({report}) |"
        )

    out_lines.append("")
    out_path = EXAMPLES / "summary.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
