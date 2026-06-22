"""Pick recipes for Built-in Models into examples/recipes/.

A `(slug, task)` pair is **Built-in** iff fp16 eval passes on every one of
the 10 (EP, device) buckets — see `rebuild_recipes_readme.discover_builtin_pairs`.

For each Built-in pair this script copies recipe config files into
`examples/recipes/<slug>/`:

  - `<task>_fp16_config*.json` — always picked (sourced from an NPU bucket
    whose fp16 eval passed, or a CPU/GPU bucket as a defensive fallback).
  - `<task>_w8a8_config*.json` — picked iff w8a8 eval passes on **at least
    one** NPU EP (sourced from that EP).
  - `<task>_w8a16_config*.json` — picked iff w8a16 eval passes on at least
    one NPU EP (sourced from that EP).

Composite tasks (e.g. CLIP zero-shot-image-classification) produce multiple
config files matching `<task>_<precision>_config*.json`; all matching files
are copied.

`examples/recipes/README.md` is **not** modified here — run
`scripts/rebuild_recipes_readme.py` separately.

Run with `--dry-run` to preview without writing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_recipes_readme import (  # noqa: E402
    EPS_ALL,
    EVAL_SUFFIX,
    NPU_EPS,
    discover_builtin_pairs,
)


REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
RECIPES = EX / "recipes"

QUANT_PRECISIONS = ("w8a8", "w8a16")


def find_source_dir(slug: str, task: str, precision: str) -> Path | None:
    """Return the bucket whose `<task>_<precision>_eval_result.json` exists.

    NPU EPs are preferred (they carry precision in the filename). fp16 falls
    back to any CPU/GPU bucket as a defensive measure.
    """
    for ep, hw in NPU_EPS:
        d = EX / ep / hw / slug
        if (d / f"{task}_{precision}{EVAL_SUFFIX}").exists():
            return d
    if precision == "fp16":
        for ep, hw in EPS_ALL:
            if (ep, hw) in NPU_EPS:
                continue
            d = EX / ep / hw / slug
            if (
                (d / f"{task}{EVAL_SUFFIX}").exists()
                or (d / f"{task}_fp16{EVAL_SUFFIX}").exists()
            ):
                return d
    return None


def source_config_files(src_dir: Path, task: str, precision: str) -> list[Path]:
    """Return matching `<task>_<precision>_config*.json` (NPU) or
    `<task>_config*.json` (CPU/GPU) files in src_dir."""
    matches = sorted(src_dir.glob(f"{task}_{precision}_config*.json"))
    if matches:
        return matches
    return sorted(src_dir.glob(f"{task}_config*.json"))


def recipe_target_name(src_name: str, task: str, precision: str) -> str:
    """Map a source config filename to its recipe filename.

    Examples:
      `image-classification_config.json`                 -> `image-classification_fp16_config.json`
      `image-classification_fp16_config.json`            -> `image-classification_fp16_config.json`
      `zero-shot-image-classification_config_text-encoder.json`
                                                         -> `zero-shot-image-classification_fp16_config_text-encoder.json`
    """
    npu_prefix = f"{task}_{precision}_config"
    if src_name.startswith(npu_prefix):
        return src_name
    cpu_prefix = f"{task}_config"
    assert src_name.startswith(cpu_prefix), src_name
    suffix = src_name[len(cpu_prefix):]
    return f"{npu_prefix}{suffix}"


def copy_recipe(
    slug: str,
    task: str,
    precision: str,
    src_dir: Path,
    dry_run: bool,
) -> list[str]:
    sources = source_config_files(src_dir, task, precision)
    if not sources:
        return []
    dest_dir = RECIPES / slug
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in sources:
        dest_name = recipe_target_name(src.name, task, precision)
        dest = dest_dir / dest_name
        if not dry_run:
            shutil.copy2(src, dest)
        written.append(dest_name)
    return written


def clean_existing_recipes(slugs_to_keep: set[str], dry_run: bool) -> list[str]:
    removed: list[str] = []
    for child in sorted(RECIPES.iterdir()):
        if not child.is_dir():
            continue
        if child.name not in slugs_to_keep:
            if not dry_run:
                shutil.rmtree(child)
            removed.append(child.name)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove existing recipe folders that are no longer Built-in.",
    )
    args = parser.parse_args()

    pairs = discover_builtin_pairs()
    print(f"Built-in (slug, task) pairs: {len(pairs)}")

    slug_set: set[str] = set()
    total_files = 0
    for slug, task in pairs:
        slug_set.add(slug)
        fp16_src = find_source_dir(slug, task, "fp16")
        if fp16_src is None:
            print(f"  ! SKIP {slug} | {task} (no fp16 source found)")
            continue
        files = copy_recipe(slug, task, "fp16", fp16_src, args.dry_run)
        total_files += len(files)
        print(
            f"  + {slug} {task} fp16 <- "
            f"{fp16_src.relative_to(REPO)} ({len(files)} files)"
        )
        for precision in QUANT_PRECISIONS:
            qsrc = find_source_dir(slug, task, precision)
            if qsrc is None:
                continue
            qfiles = copy_recipe(slug, task, precision, qsrc, args.dry_run)
            total_files += len(qfiles)
            print(
                f"  + {slug} {task} {precision} <- "
                f"{qsrc.relative_to(REPO)} ({len(qfiles)} files)"
            )

    if args.prune:
        removed = clean_existing_recipes(slug_set, args.dry_run)
        for slug in removed:
            print(f"  - removed recipes/{slug}")

    print(
        f"\nWrote {total_files} recipe file(s) across {len(slug_set)} model folder(s)."
    )
    if args.dry_run:
        print("(dry-run: no files were modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
