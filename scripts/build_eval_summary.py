"""Build examples/eval_summary.md.

Rows: (model, task, precision) for the canonical model-task list.
Precisions = fp16, w8a16, w8a8.

Columns: 9 EPs. Cell content:
  - PASS  : ✓ with link to *_eval_result.json
  - FAIL  : ✗ with link to *_eval_result.error.txt
  - TIMO  : ⏱ with link to *_eval_result.timeout
  - N/A   : — (no config exists for that bucket/precision)

Failure summary per EP: top categories from Error: line of error files.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
MODEL_TASK_LIST = REPO / "scripts" / "e2e_eval" / "testsets" / "example_model_tasks.txt"
OUT = EX / "eval_summary.md"

# 9 EP columns (label, ep_folder, hardware)
EPS: list[tuple[str, str, str]] = [
    ("DML GPU",     "dml",             "gpu"),
    ("MLAS CPU",    "mlas",            "cpu"),
    ("OV CPU",      "openvino",        "cpu"),
    ("OV GPU",      "openvino",        "gpu"),
    ("OV NPU",      "openvino",        "npu"),
    ("QNN GPU",     "qnn",             "gpu"),
    ("QNN NPU",     "qnn",             "npu"),
    ("VitisAI NPU", "vitisai",         "npu"),
    ("TRTRTX GPU",  "nv_tensorrt_rtx", "gpu"),
]

PRECISIONS = ["fp16", "w8a16", "w8a8"]


def load_pairs() -> list[tuple[str, str]]:
    """Read canonical (model, task) pairs from the shared list file."""
    pairs: list[tuple[str, str]] = []
    for line in MODEL_TASK_LIST.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        hf_id, task = s.split("|", 1)
        pairs.append((hf_id.strip(), task.strip()))
    return pairs


def _stems_for(model_dir: Path, task: str, precision: str, hardware: str) -> list[str]:
    """Return possible config stems for this (task, precision, hardware)."""
    if hardware == "npu":
        return [f"{task}_{precision}"]
    # CPU/GPU: precision only meaningful for fp16 row; others -> N/A
    if precision != "fp16":
        return []
    return [task, f"{task}_fp16"]


def cell(model_dir: Path, stems: list[str]) -> str:
    """Return markdown cell for a (model, task, precision, ep) combo."""
    if not stems:
        return "—"
    cfg_found = False
    for stem in stems:
        # Single config file OR composite split configs (`_config_<role>.json`).
        cfg = model_dir / f"{stem}_config.json"
        has_cfg = cfg.exists() or any(model_dir.glob(f"{stem}_config_*.json"))
        if not has_cfg:
            continue
        cfg_found = True
        ok = model_dir / f"{stem}_eval_result.json"
        err = model_dir / f"{stem}_eval_result.error.txt"
        tmo = model_dir / f"{stem}_eval_result.timeout"
        # Workspace-relative POSIX path
        if ok.exists():
            rel = ok.relative_to(EX).as_posix()
            return f"[✓]({rel})"
        if err.exists():
            rel = err.relative_to(EX).as_posix()
            return f"[✗]({rel})"
        if tmo.exists():
            rel = tmo.relative_to(EX).as_posix()
            return f"[⏱]({rel})"
    return "—" if not cfg_found else "?"


_ERROR_RE = re.compile(r"^(Error:\s*.+)$", re.MULTILINE)
_PY_EXC_RE = re.compile(r"^([A-Z][A-Za-z]*Error|[A-Z][A-Za-z]*Exception)\s*:\s*(.+)$", re.MULTILINE)


def classify_error(text: str) -> str:
    """Return a short category for the error text (full stderr tail)."""
    # 1. Explicit Error: lines (winml CLI emits these)
    err_lines = [m.group(1).strip() for m in _ERROR_RE.finditer(text)]
    candidate = err_lines[-1] if err_lines else ""
    # 2. Python exception lines as fallback
    if not candidate:
        exc = _PY_EXC_RE.findall(text)
        if exc:
            etype, emsg = exc[-1]
            candidate = f"{etype}: {emsg.strip()}"

    t_full = text.lower()
    t = candidate.lower()

    # Pattern matching — most specific first
    if "nodearg name already exists" in t_full:
        return "ORT graph: NodeArg name conflict"
    if "unexcepted exception" in t_full or "unexpected exception" in t_full:
        return "ORT runtime exception (pass_main)"
    if "out of memory" in t_full or "cuda oom" in t_full or "cudnn_status_out_of_memory" in t_full:
        return "Out of memory"
    # Specific eval-pipeline causes (winml CLI wraps these in "Evaluation failed:")
    if "no samples remain after label filtering" in t_full or "labels have no overlap" in t_full:
        return "Label mismatch (no overlap with dataset)"
    if "is not supported. supported tasks" in t_full:
        return "Unsupported task"
    if "couldn't find any data file" in t_full or "failed to load dataset" in t_full:
        return "Dataset missing / not built"
    if "failed to load model" in t:
        return "Model load failure"
    if "failed to load" in t_full and "tokeniz" in t_full:
        return "Tokenizer load failure"
    if "no module named" in t_full:
        return "Missing Python module"
    if "huggingface_hub" in t_full and ("404" in t_full or "not found" in t_full or "repository not found" in t_full):
        return "HF model not found"
    if "connection" in t_full or ("timed out" in t_full and "connection" in t_full):
        return "Network/connection error"
    if "dataset" in t_full and ("not found" in t_full or "download" in t_full or "load_dataset" in t_full):
        return "Dataset/download error"
    if "metric" in t and ("threshold" in t or "below" in t or "regression" in t):
        return "Accuracy below threshold"
    if "shape" in t_full and "mismatch" in t_full:
        return "Shape mismatch"
    if "dtype" in t or "data type" in t:
        return "Dtype/data type error"
    if "unsupported" in t and ("op" in t or "operator" in t):
        return "Unsupported op"
    if "session" in t and ("create" in t or "init" in t):
        return "Session creation failure"
    if "compile" in t or "compilation" in t:
        return "Compilation failure"
    if "tokeniz" in t:
        return "Tokenizer error"
    if "evaluation failed" in t:
        # winml CLI generic wrapper; try to peek above for specifics
        if "tokeniz" in t_full:
            return "Tokenizer error"
        if "ep" in t_full and "registration" in t_full:
            return "EP registration error"
        return "Evaluation failed (generic)"
    if candidate:
        return f"Other: {candidate[:80]}"
    # Heuristic for task-specific silent completion failures
    if "pppl" in t_full and "100%" in t_full:
        return "PPPL completed but no metric (likely NaN/overflow)"
    if "100%" in t_full:
        return "Eval ran to completion but no metric file written"
    return "Unknown (no error line in stderr tail)"


def build_failure_summary(pairs: list[tuple[str, str]]) -> str:
    """Per-EP top failure categories from the .error.txt files within the target pairs."""
    by_ep: dict[str, Counter] = defaultdict(Counter)
    examples_by_cat: dict[tuple[str, str], list[Path]] = defaultdict(list)
    timeouts_by_ep: dict[str, int] = defaultdict(int)

    pair_slugs = {(hf.replace("/", "_", 1), task) for hf, task in pairs}

    for label, ep, hw in EPS:
        ep_folder = EX / ep / hw
        if not ep_folder.is_dir():
            continue
        for model_dir in ep_folder.iterdir():
            if not model_dir.is_dir():
                continue
            for err in model_dir.glob("*_eval_result.error.txt"):
                stem = err.name[: -len("_eval_result.error.txt")]
                # task = stem with optional _precision suffix stripped
                base = re.sub(r"_(fp16|w8a16|w8a8)$", "", stem)
                if (model_dir.name, base) not in pair_slugs:
                    continue
                try:
                    text = err.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                cat = classify_error(text)
                by_ep[label][cat] += 1
                if len(examples_by_cat[(label, cat)]) < 2:
                    examples_by_cat[(label, cat)].append(err)
            for tmo in model_dir.glob("*_eval_result.timeout"):
                stem = tmo.name[: -len("_eval_result.timeout")]
                base = re.sub(r"_(fp16|w8a16|w8a8)$", "", stem)
                if (model_dir.name, base) not in pair_slugs:
                    continue
                by_ep[label]["Timeout"] += 1
                timeouts_by_ep[label] += 1

    lines = ["## Failure Summary by EP", ""]
    for label, _, _ in EPS:
        counts = by_ep.get(label)
        if not counts:
            lines += [f"### {label}", "", "_No eval failures recorded for the target pairs._", ""]
            continue
        total = sum(counts.values())
        lines += [f"### {label}", "", f"Total failures: **{total}**", "",
                  "| Category | Count | Example |", "|---|---|---|"]
        for cat, n in counts.most_common():
            examples = examples_by_cat.get((label, cat), [])
            ex_link = ""
            if examples:
                rel = examples[0].relative_to(EX).as_posix()
                ex_link = f"[{examples[0].parent.name}/{examples[0].name}]({rel})"
            lines.append(f"| {cat} | {n} | {ex_link} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    pairs = load_pairs()
    if not pairs:
        raise SystemExit(f"No model-task pairs found in {MODEL_TASK_LIST}")

    header = ["# Eval Result Summary",
              "",
              ("Rows = (model, task, precision) for the canonical model-task list "
               "in `scripts/e2e_eval/testsets/example_model_tasks.txt`. Columns = 9 EPs."),
              "",
              ("Cells: ✓ = eval pass (links to *_eval_result.json), "
               "✗ = eval failure (links to *_eval_result.error.txt), "
               "⏱ = timeout (links to *_eval_result.timeout), "
               "? = config exists but eval not yet attempted, "
               "— = no config exists for this bucket/precision."),
              "",
              "## Matrix",
              "",
              "| Model | Task | Precision | " + " | ".join(label for label, _, _ in EPS) + " |",
              "|---|---|---|" + "---|" * len(EPS)]

    rows: list[str] = []
    for hf, task in pairs:
        slug = hf.replace("/", "_", 1)
        for prec in PRECISIONS:
            cells = []
            any_real = False
            for _, ep, hw in EPS:
                model_dir = EX / ep / hw / slug
                stems = _stems_for(model_dir, task, prec, hw)
                c = cell(model_dir, stems)
                cells.append(c)
                if c not in ("—",):
                    any_real = True
            if not any_real:
                continue  # skip a precision row with no data at all
            rows.append(f"| {hf} | {task} | {prec} | " + " | ".join(cells) + " |")

    body = "\n".join(header + rows + ["", build_failure_summary(pairs)])
    OUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
