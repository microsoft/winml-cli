"""Analyze E2E evaluation results — summary, failure distribution, and run comparison.

Usage:
    python scripts/e2e_eval/analyze_results.py eval_results/npu_run/
    python scripts/e2e_eval/analyze_results.py eval_results/npu_run/ --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def safe_print(text: str) -> None:
    """Cross-platform safe print (handles Windows Unicode issues)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def load_results(results_dir: Path) -> list[dict]:
    """Load all result.json files from a results directory."""
    models_dir = results_dir / "models"
    if not models_dir.exists():
        safe_print(f"No models/ directory in {results_dir}")
        sys.exit(1)

    results = []
    for f in sorted(models_dir.glob("*/result.json")):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            pass

    if not results:
        safe_print("No results found")
        sys.exit(1)

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary, failure distribution, by-task breakdown, and failed models."""
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if not r.get("passed")]
    total = len(results)
    rate = len(passed) / total * 100 if total else 0

    safe_print("=== SUMMARY ===")
    safe_print(f"Total: {total} | Passed: {len(passed)} | Failed: {len(failed)} | Rate: {rate:.1f}%")

    if failed:
        fc = Counter(r.get("failure_classification", "UNKNOWN") or "UNKNOWN" for r in failed)
        safe_print("\n=== FAILURE DISTRIBUTION ===")
        for cls, count in fc.most_common():
            pct = count / len(failed) * 100
            safe_print(f"  {cls:<20} {count:>3} ({pct:.0f}%)")

    safe_print("\n=== BY TASK ===")
    by_task: dict = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        task = r.get("task", "(none)")
        by_task[task]["total"] += 1
        if r.get("passed"):
            by_task[task]["passed"] += 1
    for task, s in sorted(by_task.items(), key=lambda x: x[1]["passed"] / max(x[1]["total"], 1)):
        t_rate = s["passed"] / s["total"] * 100 if s["total"] else 0
        safe_print(f"  {task:<42} {s['passed']}/{s['total']} ({t_rate:.0f}%)")

    if failed:
        safe_print("\n=== FAILED MODELS ===")
        for r in sorted(failed, key=lambda x: x.get("failure_classification", "")):
            fc_val = r.get("failure_classification", "UNKNOWN")
            safe_print(f"  [{fc_val:<15}] {r['model']} / {r.get('task', '')}")


def compare_runs(current_dir: Path) -> None:
    """Compare current run with the most recent previous run."""
    eval_root = current_dir.parent
    all_runs = sorted(
        [d for d in eval_root.iterdir() if d.is_dir() and d != current_dir],
        key=lambda d: d.name,
        reverse=True,
    )
    if not all_runs:
        safe_print("No previous run found for comparison.")
        return

    prev_dir = all_runs[0]
    safe_print(f"Comparing: {current_dir.name} vs {prev_dir.name} (previous)")

    def _load(d: Path) -> dict:
        results = {}
        md = d / "models"
        if not md.exists():
            return results
        for f in md.glob("*/result.json"):
            try:
                r = json.loads(f.read_text(encoding="utf-8", errors="replace"))
                key = f"{r['model']} / {r.get('task', '')}"
                results[key] = r
            except Exception:
                pass
        return results

    old, new = _load(prev_dir), _load(current_dir)
    common = set(old) & set(new)
    if not common:
        safe_print("No common models to compare.")
        return

    improved, regressed = [], []
    for k in sorted(common):
        o, n = old[k].get("passed", False), new[k].get("passed", False)
        if not o and n:
            improved.append(k)
        elif o and not n:
            regressed.append((k, new[k]))

    safe_print(f"Common models: {len(common)}")
    safe_print(f"Improved (FAIL->PASS): {len(improved)}")
    safe_print(f"Regressed (PASS->FAIL): {len(regressed)}")

    if improved:
        safe_print("\n=== FIXED (was FAIL, now PASS) ===")
        for k in improved:
            safe_print(f"  + {k}")

    if regressed:
        safe_print("\n=== REGRESSED (was PASS, now FAIL) ===")
        for k, r in regressed:
            stdout = r.get("stdout_output", "") or ""
            stderr = r.get("stderr_output", "") or ""
            combined = stdout + stderr
            if "No such file" in combined and "safetensors" in combined:
                cause = "RACE_CONDITION (safetensors deleted by parallel process)"
            elif "timeout" in combined.lower():
                cause = "TIMEOUT"
            elif "MemoryError" in combined or "out of memory" in combined.lower():
                cause = "OOM"
            elif "torch.onnx" in combined or "tracing error" in combined.lower():
                cause = "EXPORT_FAIL"
            elif "CompilationError" in combined or "quantization" in combined.lower():
                cause = "COMPILE_FAIL"
            else:
                err_lines = [
                    line for line in combined.splitlines()
                    if "error" in line.lower() or "Error" in line
                ]
                cause = err_lines[-1].strip()[:80] if err_lines else "UNKNOWN"
            safe_print(f"  - {k}")
            safe_print(f"    Cause: {cause}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E2E evaluation results")
    parser.add_argument("results_dir", type=Path, help="Path to eval results directory")
    parser.add_argument("--compare", action="store_true", help="Compare with previous run")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if args.compare:
        compare_runs(args.results_dir)
        safe_print("")

    print_summary(results)


if __name__ == "__main__":
    main()
