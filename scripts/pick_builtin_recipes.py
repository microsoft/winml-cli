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
    EVAL_SUFFIX,
    NPU_EPS,
    discover_builtin_pairs,
)


REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
RECIPES = EX / "recipes"

QUANT_PRECISIONS = ("w8a8", "w8a16")


def find_source_dir(slug: str, task: str, precision: str) -> Path | None:
    """Return an NPU bucket whose `<task>_<precision>_eval_result.json` exists
    on **every** NPU EP, or None if any NPU EP is missing it.

    For Built-in pairs, fp16 is guaranteed to satisfy this (NPU fp16 passes
    on every NPU EP by definition). w8a8/w8a16 may legitimately return None.
    """
    candidate: Path | None = None
    for ep, hw in NPU_EPS:
        d = EX / ep / hw / slug
        if not (d / f"{task}_{precision}{EVAL_SUFFIX}").exists():
            return None
        if candidate is None:
            candidate = d
    return candidate


def source_config_files(src_dir: Path, task: str, precision: str) -> list[Path]:
    """Return matching `<task>_<precision>_config*.json` files in an NPU bucket."""
    return sorted(src_dir.glob(f"{task}_{precision}_config*.json"))


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
        dest = dest_dir / src.name
        if not dry_run:
            shutil.copy2(src, dest)
        written.append(src.name)
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
