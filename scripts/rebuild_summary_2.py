"""Regenerate examples/summary_2.md with 5 sections built from real eval results.

Sections:
  1. Target Builtin Models (the 42 eval-supported pairs from the external 57
     perf list joined with models_with_acc.json)
  2. fp16 eval pass on ALL 9 EPs
  3. fp16 eval pass on AT LEAST ONE EP
  4. w8a8 eval pass on ALL 3 NPU EPs
  5. w8a16 eval pass on ALL 3 NPU EPs
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
ACC = REPO / "scripts/e2e_eval/testsets/models_with_acc.json"
LIST57 = REPO / "temp/list57.txt"
OUT = EX / "summary_2.md"

EPS_ALL: list[tuple[str, str]] = [
    ("dml", "gpu"),
    ("mlas", "cpu"),
    ("openvino", "cpu"),
    ("openvino", "gpu"),
    ("openvino", "npu"),
    ("qnn", "gpu"),
    ("qnn", "npu"),
    ("vitisai", "npu"),
    ("nv_tensorrt_rtx", "gpu"),
]
NPU_EPS: list[tuple[str, str]] = [("openvino", "npu"), ("qnn", "npu"), ("vitisai", "npu")]


def load_target_pairs() -> list[tuple[str, str]]:
    acc = json.loads(ACC.read_text(encoding="utf-8"))
    acc_pairs = {(e["hf_id"], e["task"]) for e in acc}
    ext = set()
    for line in LIST57.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        m, t = line.split("|")
        ext.add((m, t))
    return sorted(ext & acc_pairs)


def eval_pass_on(model_dir: Path, task: str, ep: str, hw: str, precision: str) -> bool:
    """Return True if an eval_result.json exists for this bucket at the given precision."""
    if hw == "npu":
        return (model_dir / f"{task}_{precision}_eval_result.json").exists()
    # CPU/GPU: only fp16 makes sense. Stem may be plain task or task_fp16.
    if precision != "fp16":
        return False
    return (
        (model_dir / f"{task}_eval_result.json").exists()
        or (model_dir / f"{task}_fp16_eval_result.json").exists()
    )


def fp16_pass_counts(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    counts = {}
    for hf, task in pairs:
        slug = hf.replace("/", "_", 1)
        n = 0
        for ep, hw in EPS_ALL:
            if eval_pass_on(EX / ep / hw / slug, task, ep, hw, "fp16"):
                n += 1
        counts[(hf, task)] = n
    return counts


def npu_precision_pass_all(pairs: list[tuple[str, str]], precision: str) -> list[tuple[str, str]]:
    out = []
    for hf, task in pairs:
        slug = hf.replace("/", "_", 1)
        if all(eval_pass_on(EX / ep / hw / slug, task, ep, hw, precision) for ep, hw in NPU_EPS):
            out.append((hf, task))
    return out


def md_table(pairs: list[tuple[str, str]], extra_col: tuple[str, list[str]] | None = None) -> list[str]:
    if extra_col:
        header = f"| Model | Task | {extra_col[0]} |"
        sep = "|---|---|---|"
        rows = [f"| {hf} | {task} | {extra_col[1][i]} |" for i, (hf, task) in enumerate(pairs)]
    else:
        header = "| Model | Task |"
        sep = "|---|---|"
        rows = [f"| {hf} | {task} |" for hf, task in pairs]
    return [header, sep, *rows]


def main() -> int:
    target = load_target_pairs()
    fp16_counts = fp16_pass_counts(target)
    fp16_all = [p for p in target if fp16_counts[p] == len(EPS_ALL)]
    fp16_any = [p for p in target if fp16_counts[p] >= 1]
    w8a8_all = npu_precision_pass_all(target, "w8a8")
    w8a16_all = npu_precision_pass_all(target, "w8a16")

    fp16_any_with_counts = [(hf, task) for hf, task in fp16_any]
    pass_col = [f"{fp16_counts[(hf, task)]}/{len(EPS_ALL)}" for hf, task in fp16_any]

    lines: list[str] = [
        "# Builtin Model Coverage",
        "",
        "Five views over the eval-supported model set.",
        "",
        "---",
        "",
        "## 1. Target Builtin Models",
        "",
        "Models that:",
        "1. Appear in the external 57 (model, task) perf list, AND",
        "2. Are eval-supported (present in `scripts/e2e_eval/testsets/models_with_acc.json`).",
        "",
        f"Total: **{len(target)}** (model, task) tuples.",
        "",
        *md_table(target),
        "",
        "---",
        "",
        "## 2. fp16 eval pass on ALL 9 EPs",
        "",
        "Subset of the target list where fp16 eval pass on every one of the 9 (EP, device) buckets "
        "(CPU/GPU rows use plain `<task>_eval_result.json` or `<task>_fp16_eval_result.json`; "
        "NPU rows use `<task>_fp16_eval_result.json`).",
        "",
        f"Total: **{len(fp16_all)}** (model, task) tuples.",
        "",
        *md_table(fp16_all),
        "",
        "---",
        "",
        "## 3. fp16 eval pass on AT LEAST ONE EP",
        "",
        "Subset of the target list where fp16 eval pass on at least one of the 9 EPs.",
        "",
        f"Total: **{len(fp16_any)}** (model, task) tuples.",
        "",
        *md_table(fp16_any_with_counts, extra_col=("EPs Passed", pass_col)),
        "",
        "---",
        "",
        "## 4. w8a8 eval pass on ALL 3 NPU EPs",
        "",
        "Subset of the target list where `*_w8a8_eval_result.json` exists in **every** NPU EP "
        "(QNN, OpenVINO, VitisAI).",
        "",
        f"Total: **{len(w8a8_all)}** (model, task) tuples.",
        "",
        *md_table(w8a8_all),
        "",
        "---",
        "",
        "## 5. w8a16 eval pass on ALL 3 NPU EPs",
        "",
        "Subset of the target list where `*_w8a16_eval_result.json` exists in **every** NPU EP "
        "(QNN, OpenVINO, VitisAI).",
        "",
        f"Total: **{len(w8a16_all)}** (model, task) tuples.",
        "",
        *md_table(w8a16_all),
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  1. Target: {len(target)}")
    print(f"  2. fp16 all 9 EPs: {len(fp16_all)}")
    print(f"  3. fp16 any EP: {len(fp16_any)}")
    print(f"  4. w8a8 all 3 NPUs: {len(w8a8_all)}")
    print(f"  5. w8a16 all 3 NPUs: {len(w8a16_all)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
