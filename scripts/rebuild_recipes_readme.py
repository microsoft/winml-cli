"""Regenerate `examples/recipes/README.md` by walking `examples/<ep>/<device>/`.

Discovery is **filesystem-driven**: no static model/task list is consulted.

For every `examples/<ep>/<device>/<slug>/` folder we scan for fp16-passing
eval results:

  - NPU buckets: `<task>_fp16_eval_result.json`
  - CPU/GPU buckets: `<task>_eval_result.json` (EP default precision) or
    `<task>_fp16_eval_result.json`

A `(slug, task)` pair is **Built-in** iff every one of the 10 (EP, device)
buckets contains a passing fp16 eval result.

The README's prose (everything before `## Models`) is preserved verbatim;
only the Models table and a `Total` line are rewritten.
"""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
README = EX / "recipes" / "README.md"
MARKER = "## Models"

EPS_ALL: list[tuple[str, str]] = [
    ("dml", "gpu"),
    ("mlas", "cpu"),
    ("migraphx", "gpu"),
    ("nv_tensorrt_rtx", "gpu"),
    ("openvino", "cpu"),
    ("openvino", "gpu"),
    ("openvino", "npu"),
    ("qnn", "gpu"),
    ("qnn", "npu"),
    ("vitisai", "npu"),
]
NPU_EPS: list[tuple[str, str]] = [("openvino", "npu"), ("qnn", "npu"), ("vitisai", "npu")]
EVAL_SUFFIX = "_eval_result.json"


def slug_to_hf_id(slug: str) -> str:
    """Convert a folder slug back to a HuggingFace id.

    The slug is `<owner>_<name>` where the first `_` replaces the owner/name
    slash. HF owners do not contain `_` (they use `-`), so splitting on the
    first `_` is unambiguous.
    """
    return slug.replace("_", "/", 1)


def fp16_tasks_in_bucket(bucket_dir: Path, hw: str) -> set[tuple[str, str]]:
    """Return {(slug, task)} that have a passing fp16 eval in this bucket.

    NPU layout: `<task>_<precision>_eval_result.json`; only `fp16` counts here.
    CPU/GPU layout: `<task>_eval_result.json` (EP default precision).
    """
    out: set[tuple[str, str]] = set()
    if not bucket_dir.is_dir():
        return out
    for slug_dir in bucket_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        for ev in slug_dir.glob(f"*{EVAL_SUFFIX}"):
            stem = ev.name[: -len(EVAL_SUFFIX)]  # strip trailing "_eval_result.json"
            if hw == "npu":
                if "_" not in stem:
                    continue
                task, precision = stem.rsplit("_", 1)
                if precision == "fp16":
                    out.add((slug_dir.name, task))
            else:
                out.add((slug_dir.name, stem))
    return out


def npu_quant_passes(slug: str, task: str, precision: str) -> bool:
    """True iff `<task>_<precision>_eval_result.json` exists on any NPU EP."""
    for ep, hw in NPU_EPS:
        if (EX / ep / hw / slug / f"{task}_{precision}{EVAL_SUFFIX}").exists():
            return True
    return False


def discover_builtin_pairs() -> list[tuple[str, str]]:
    """Walk examples/ and return Built-in (slug, task) pairs."""
    per_bucket = {
        (ep, hw): fp16_tasks_in_bucket(EX / ep / hw, hw) for ep, hw in EPS_ALL
    }
    all_pairs = set().union(*per_bucket.values())
    builtin = [
        pair for pair in sorted(all_pairs)
        if all(pair in per_bucket[k] for k in per_bucket)
    ]
    return builtin


def render_models_section(pairs: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"| {slug_to_hf_id(slug)} | {task} |" for slug, task in pairs)
    return (
        f"{MARKER}\n"
        f"\n"
        f"Total: **{len(pairs)}** (model, task) tuples that pass fp16 eval on "
        f"all {len(EPS_ALL)} (EP, device) buckets.\n"
        f"\n"
        f"| Model | Task |\n"
        f"|---|---|\n"
        f"{rows}\n"
    )


def main() -> int:
    pairs = discover_builtin_pairs()
    text = README.read_text(encoding="utf-8")
    idx = text.find(MARKER)
    if idx == -1:
        new = text.rstrip() + "\n\n" + render_models_section(pairs)
    else:
        new = text[:idx] + render_models_section(pairs)
    README.write_text(new, encoding="utf-8")
    print(f"Wrote {README}")
    print(f"  Built-in (slug, task) tuples: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
