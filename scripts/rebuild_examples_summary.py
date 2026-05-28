"""Rebuild examples/summary.md from real config/result files (no fabrication).

Counts per row (matching scripts/generate_example_report.py semantics):
  - Models       : distinct model slugs that have at least one config
  - Configs      : config files in the row's bucket
  - Perf Pass    : sibling *_perf_result.json existence
  - Eval Pass    : sibling *_eval_result.json existence

Buckets:
  - For NPU folders, rows are split by precision (fp16 / w8a16 / w8a8).
  - For CPU/GPU folders, single row.

Also emits a top-level "Builtin Models" section listing (model_id, task) tuples
that:
  1. Have at least one config in every one of the 9 (ep, hardware) buckets,
  2. Every existing config (across all 9 buckets, any precision) has a
     sibling *_perf_result.json,
  3. At least one config has a sibling *_eval_result.json.
"""

from __future__ import annotations

import re
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Display row label, (ep folder, hardware), optional precision (None = all),
# and report path used in the table. Order matches the previous summary.md.
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

# The 9 (ep folder, hardware) buckets each model must cover for "builtin" status.
ALL_EP_BUCKETS: list[tuple[str, str]] = [
    ("dml", "gpu"),
    ("mlas", "cpu"),
    ("nv_tensorrt_rtx", "gpu"),
    ("openvino", "cpu"),
    ("openvino", "gpu"),
    ("openvino", "npu"),
    ("qnn", "gpu"),
    ("qnn", "npu"),
    ("vitisai", "npu"),
]

_NPU_PRECISION_RE = re.compile(r"_(fp16|w8a16|w8a8)$")


def collect(folder: Path, hardware: str, precision_filter: str | None) -> tuple[int, int, int, int]:
    """Return (models, configs, perf_pass, eval_pass).

    Matches the semantics of scripts/generate_example_report.py:
    - All ``*_config.json`` files count as configs (CPU/GPU rows include any
      precision suffix; only the NPU rows are filtered by precision).
    - Perf/Eval pass = sibling ``*_perf_result.json`` / ``*_eval_result.json``
      file exists for the same stem.
    """
    models: set[str] = set()
    configs = 0
    perf_pass = 0
    eval_pass = 0

    if not folder.is_dir():
        return 0, 0, 0, 0

    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        for cfg in model_dir.glob("*_config.json"):
            stem = cfg.name[: -len("_config.json")]  # e.g. "image-classification" or "..._fp16"
            if hardware == "npu" and precision_filter:
                m = _NPU_PRECISION_RE.search(stem)
                if not m or m.group(1) != precision_filter:
                    continue
            configs += 1
            models.add(model_dir.name)
            if (model_dir / f"{stem}_perf_result.json").exists():
                perf_pass += 1
            if (model_dir / f"{stem}_eval_result.json").exists():
                eval_pass += 1

    return len(models), configs, perf_pass, eval_pass


def _split_stem(stem: str) -> tuple[str, str | None]:
    m = _NPU_PRECISION_RE.search(stem)
    if m:
        return stem[: m.start()], m.group(1)
    return stem, None


def _builtin_models() -> list[tuple[str, str, dict[tuple[str, str], list[str]]]]:
    """Return list of (model_slug, task, bucket_to_precisions) for tuples that
    qualify as builtin models per the criteria documented at the top of this file.

    A (model, task) qualifies when, for every one of the 9 (ep, hardware)
    buckets:
      - at least one config exists,
      - every existing config has a sibling *_perf_result.json,
      - at least one existing config has a sibling *_eval_result.json.
    """
    # (slug, task) -> { (ep, hw): { "precisions": [..], "perf_all": bool, "eval_any": bool } }
    by_key: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}

    for ep, hw in ALL_EP_BUCKETS:
        folder = EXAMPLES / ep / hw
        if not folder.is_dir():
            continue
        for model_dir in folder.iterdir():
            if not model_dir.is_dir():
                continue
            for cfg in model_dir.glob("*_config.json"):
                stem = cfg.name[: -len("_config.json")]
                task, precision = _split_stem(stem)
                key = (model_dir.name, task)
                bucket = by_key.setdefault(key, {}).setdefault(
                    (ep, hw), {"precisions": [], "perf_all": True, "eval_any": False}
                )
                bucket["precisions"].append(precision or "")
                if not (model_dir / f"{stem}_perf_result.json").exists():
                    bucket["perf_all"] = False
                if (model_dir / f"{stem}_eval_result.json").exists():
                    bucket["eval_any"] = True

    qualified: list[tuple[str, str, dict[tuple[str, str], list[str]]]] = []
    required = set(ALL_EP_BUCKETS)
    for key, buckets in sorted(by_key.items()):
        if set(buckets.keys()) != required:
            continue
        if not all(b["perf_all"] and b["eval_any"] for b in buckets.values()):
            continue
        qualified.append((key[0], key[1], {k: v["precisions"] for k, v in buckets.items()}))
    return qualified


def main() -> int:
    out_lines: list[str] = [
        "# Example Configs Test Summary",
        "",
        "## Overview",
        "",
        "| EP | Models | Configs | Perf Pass | Eval Pass | Report |",
        "|----|--------|---------|-----------|-----------|--------|",
    ]
    for label, ep, hw, prec, report in ROWS:
        models, configs, p, e = collect(EXAMPLES / ep / hw, hw, prec)
        pct = lambda x, tot: f"{x}/{tot} ({100 * x / tot:.0f}%)" if tot else f"{x}/0 (0%)"
        out_lines.append(
            f"| {label} | {models} | {configs} | {pct(p, configs)} | {pct(e, configs)} | [Report]({report}) |"
        )

    out_lines.append("")
    out_path = EXAMPLES / "summary.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

_NPU_PRECISION_RE = re.compile(r"_(fp16|w8a16|w8a8)$")


def collect(folder: Path, hardware: str, precision_filter: str | None) -> tuple[int, int, int, int]:
    """Return (models, configs, perf_pass, eval_pass).

    Matches the semantics of scripts/generate_example_report.py:
    - All ``*_config.json`` files count as configs (CPU/GPU rows include any
      precision suffix; only the NPU rows are filtered by precision).
    - Perf/Eval pass = sibling ``*_perf_result.json`` / ``*_eval_result.json``
      file exists for the same stem.
    """
    models: set[str] = set()
    configs = 0
    perf_pass = 0
    eval_pass = 0

    if not folder.is_dir():
        return 0, 0, 0, 0

    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        for cfg in model_dir.glob("*_config.json"):
            stem = cfg.name[: -len("_config.json")]  # e.g. "image-classification" or "..._fp16"
            if hardware == "npu" and precision_filter:
                m = _NPU_PRECISION_RE.search(stem)
                if not m or m.group(1) != precision_filter:
                    continue
            configs += 1
            models.add(model_dir.name)
            if (model_dir / f"{stem}_perf_result.json").exists():
                perf_pass += 1
            if (model_dir / f"{stem}_eval_result.json").exists():
                eval_pass += 1

    return len(models), configs, perf_pass, eval_pass


def main() -> int:
    out_lines = [
        "# Example Configs Test Summary",
        "",
        "## Overview",
        "",
        "| EP | Models | Configs | Perf Pass | Eval Pass | Report |",
        "|----|--------|---------|-----------|-----------|--------|",
    ]
    for label, ep, hw, prec, report in ROWS:
        models, configs, p, e = collect(EXAMPLES / ep / hw, hw, prec)
        pct = lambda x, tot: f"{x}/{tot} ({100 * x / tot:.0f}%)" if tot else f"{x}/0 (0%)"
        out_lines.append(
            f"| {label} | {models} | {configs} | {pct(p, configs)} | {pct(e, configs)} | [Report]({report}) |"
        )

    out_lines.append("")
    out_path = EXAMPLES / "summary.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
