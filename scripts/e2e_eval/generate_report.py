# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Regenerate reports from existing eval_result.json files.

Usage:
    python scripts/e2e_eval/generate_report.py --input-dir ./eval_results/2026-02-22/
    python scripts/e2e_eval/generate_report.py --input-dir eval_results/xxx --format markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from utils.accuracy import derive_verdicts
from utils.reporter import (
    classify_results,
    format_text_summary,
    generate_html_report,
    generate_summary,
    load_results_from_dir,
    write_summary_json,
    write_summary_md,
)


def safe_print(text: str) -> None:
    """Cross-platform safe print (handles Windows Unicode issues)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate reports from existing eval_result.json files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with models/*/eval_result.json",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "html", "text", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent / "testsets" / "models_all.json",
        help="Model registry JSON for enrichment (default: testsets/models_all.json)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    results = load_results_from_dir(input_dir)
    if not results:
        safe_print(f"No eval_result.json files found in {input_dir}/models/")
        sys.exit(1)

    safe_print(f"Loaded {len(results)} results from {input_dir}")

    # Derive classifications and verdicts from stored facts (always uses latest rules)
    classify_results(results)
    derive_verdicts(results)

    summary = generate_summary(results, 0.0)

    fmt = args.format
    if fmt in ("json", "all"):
        out = input_dir / "summary.json"
        write_summary_json(summary, out)
        safe_print(f"  Written: {out}")

    if fmt in ("markdown", "all"):
        out = input_dir / "summary.md"
        write_summary_md(results, summary, out)
        safe_print(f"  Written: {out}")

    if fmt in ("html", "all"):
        out = input_dir / "eval_report.html"
        generate_html_report(summary, out, args.registry)
        safe_print(f"  Written: {out}")

    if fmt in ("text", "all"):
        text = format_text_summary(results)
        safe_print(text)

    ps = summary["perf_summary"]
    total = ps["total"]
    rate = (ps["passed"] / total * 100) if total else 0
    safe_print(f"\nPerf pass rate: {ps['passed']}/{total} ({rate:.1f}%)")
    acc_s = summary.get("accuracy_summary")
    if acc_s:
        evaluated = acc_s.get("evaluated", 0)
        acc_pass = acc_s.get("accuracy_pass", 0)
        acc_rate = acc_s.get("pass_rate", 0)
        safe_print(
            f"Accuracy pass rate: {acc_pass}/{evaluated} ({acc_rate:.1%})  "
            f"[at-risk={acc_s.get('accuracy_at_risk', 0)} "
            f"regression={acc_s.get('accuracy_regression', 0)} "
            f"error={acc_s.get('eval_error', 0)}]"
        )


if __name__ == "__main__":
    main()
