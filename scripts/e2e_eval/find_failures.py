# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Find failed models matching a pattern in E2E evaluation results.

Searches model_type first (exact), then falls back to full-text search
across failure_classification, stderr, and stdout.

Usage:
    python scripts/e2e_eval/find_failures.py roberta
    python scripts/e2e_eval/find_failures.py roberta eval_results/npu_run/
    python scripts/e2e_eval/find_failures.py "index out of range"
    python scripts/e2e_eval/find_failures.py EXPORT_FAIL
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def safe_print(text: str) -> None:
    """Print text with fallback for Unicode errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def find_latest_results_dir() -> Path | None:
    """Find the most recent eval_results directory."""
    eval_root = Path("eval_results")
    if not eval_root.exists():
        return None
    candidates = sorted(
        [d for d in eval_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_failures(pattern: str, results_dir: Path) -> list[dict]:
    """Find failed models matching pattern. Exact model_type first, then full-text."""
    models_dir = results_dir / "models"
    if not models_dir.exists():
        safe_print(f"No models/ directory in {results_dir}")
        sys.exit(1)

    all_failed = []
    for f in sorted(models_dir.glob("*/result.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if r.get("passed"):
            continue
        all_failed.append(r)

    # Phase 1: exact model_type match
    exact = [r for r in all_failed if r.get("model_type", "") == pattern]
    if exact:
        return exact

    # Phase 2: regex match across model_type + failure_classification
    meta_matches = []
    for r in all_failed:
        searchable = f"{r.get('model_type', '')} {r.get('failure_classification', '')}"
        if re.search(pattern, searchable, re.IGNORECASE):
            meta_matches.append(r)
    if meta_matches:
        return meta_matches

    # Phase 3: full-text search including stderr/stdout
    full_matches = []
    for r in all_failed:
        searchable = " ".join(
            [
                r.get("model_type", ""),
                r.get("failure_classification", ""),
                r.get("stderr_output", "") or "",
                r.get("stdout_output", "") or "",
            ]
        )
        if re.search(pattern, searchable, re.IGNORECASE):
            full_matches.append(r)
    return full_matches


def check_registry_no_result(pattern: str, results_dir: Path) -> list[dict]:
    """Check models.json for models matching pattern that have NO eval results."""
    registry_path = Path("scripts/e2e_eval/testsets/models_all.json")
    if not registry_path.exists():
        return []

    with registry_path.open() as f:
        registry = json.load(f)

    models_dir = results_dir / "models"
    no_result = []
    for m in registry:
        if not re.search(pattern, m.get("model_type", ""), re.IGNORECASE):
            continue
        slug = m["hf_id"].replace("/", "__")
        rpath = models_dir / slug / "result.json"
        if not rpath.exists():
            no_result.append(m)

    return no_result


def main() -> None:
    """Find failed models matching a pattern."""
    parser = argparse.ArgumentParser(description="Find failed models matching a pattern")
    parser.add_argument(
        "pattern", help="Pattern to search (model_type, error message, or classification)"
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path, default=None, help="Eval results directory"
    )
    args = parser.parse_args()

    results_dir = args.results_dir or find_latest_results_dir()

    # Search eval results (if they exist)
    matches = []
    if results_dir and (results_dir / "models").exists():
        matches = find_failures(args.pattern, results_dir)

    # Always check registry for never-evaluated models
    no_result = check_registry_no_result(args.pattern, results_dir) if results_dir else []
    if not results_dir:
        # No eval results at all — check registry only
        registry_path = Path("scripts/e2e_eval/testsets/models_all.json")
        if registry_path.exists():
            with registry_path.open() as f:
                registry = json.load(f)
            no_result = [
                m
                for m in registry
                if re.search(args.pattern, m.get("model_type", ""), re.IGNORECASE)
            ]

    safe_print(f"Pattern: {args.pattern!r}")
    safe_print(f"Results dir: {results_dir or '(none)'}")
    safe_print(f"Matched failed: {len(matches)}")
    safe_print(f"Never evaluated: {len(no_result)}")

    if matches:
        mt = Counter(r.get("model_type", "?") for r in matches)
        fc = Counter(r.get("failure_classification", "?") for r in matches)
        safe_print(f"\nBy model_type: {dict(mt.most_common())}")
        safe_print(f"By failure_class: {dict(fc.most_common())}")
        safe_print("\nFailed models:")
        for r in matches:
            safe_print(f"  {r['model']:50s} {r.get('model_type', '?'):15s} {r.get('task', '')}")

    if no_result:
        safe_print("\nNever evaluated (in registry but no result.json):")
        for m in no_result:
            safe_print(f"  {m['hf_id']:50s} {m['model_type']:15s} {m.get('task', '')}")


if __name__ == "__main__":
    main()
