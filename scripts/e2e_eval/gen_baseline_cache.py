# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generate a PyTorch baseline cache at a fixed sample cap (default 100).

Runs ONLY the PyTorch CPU baseline for each model in the accuracy registry.
No winml/ONNX build or eval is performed.

The output is keyed exactly like ``run_eval.py``'s baseline cache
(``hf_id|task|dataset|dataset_config|split|num_samples``), so the 100-sample
cache lives in its own file and never collides with the canonical
``baseline_cache.json`` (whose keys carry the original sample counts).

Usage:
    # Generate baseline_cache_100.json for all accuracy-eligible models
    uv run python scripts/e2e_eval/gen_baseline_cache.py

    # Only P0 image-classification models, 50 samples
    uv run python scripts/e2e_eval/gen_baseline_cache.py \
        --priority P0 --task image-classification --samples 50

    # Single model
    uv run python scripts/e2e_eval/gen_baseline_cache.py \
        --hf-model microsoft/resnet-50

    # Preview the work without running
    uv run python scripts/e2e_eval/gen_baseline_cache.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure sibling modules (run_eval, utils) are importable when invoked directly.
sys.path.insert(0, str(Path(__file__).parent))

import run_eval  # noqa: E402  (re-uses helpers + cache-key scheme)
from utils.dataset_config import get_dataset_config, register_from_registry  # noqa: E402
from utils.registry import filter_registry, load_registry, make_adhoc_entry  # noqa: E402


DEFAULT_REGISTRY = Path(__file__).parent / "testsets" / "models_with_acc.json"
DEFAULT_CACHE = Path(__file__).parent / "cache" / "baseline_cache_100.json"
DEFAULT_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a PyTorch-only baseline cache at a fixed sample cap."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"Accuracy registry JSON (default: {DEFAULT_REGISTRY.name})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Output cache file (default: {DEFAULT_CACHE.name})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=(
            f"Sample cap (default: {DEFAULT_SAMPLES}). Each model uses "
            "min(registry_samples, cap), so models already below the cap keep "
            "their smaller count."
        ),
    )
    parser.add_argument("--hf-model", help="Single model (overrides registry filters)")
    parser.add_argument("--task", help="Filter by HF task")
    parser.add_argument(
        "--priority",
        nargs="+",
        choices=["P0", "P1", "P2", "P3"],
        metavar="{P0,P1,P2,P3}",
        help="Filter by priority (default: all).",
    )
    parser.add_argument("--model-type", help="Filter by model_type")
    parser.add_argument("--group", help="Filter by group")
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-model timeout in seconds (default: 7200; raise for slow fill-mask models).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run and overwrite even if a PASS entry already exists in the cache.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the models that would be processed (with effective sample count) and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Point run_eval's cache helpers at our separate file.
    run_eval.BASELINE_CACHE_PATH = args.cache
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    registry_entries = load_registry(args.registry)
    register_from_registry(registry_entries)

    if args.hf_model:
        entries = [
            e
            for e in registry_entries
            if e.hf_id == args.hf_model and (not args.task or e.task == args.task)
        ]
        if not entries:
            entries = [make_adhoc_entry(args.hf_model, args.task)]
    else:
        entries = filter_registry(
            registry_entries,
            task=args.task,
            priority=args.priority,
            model_type=args.model_type,
            group=args.group,
        )

    # Keep only accuracy-eligible models (those with a dataset configured).
    eligible: list = []
    skipped_no_dataset: list[str] = []
    for entry in entries:
        ds = get_dataset_config(entry.hf_id, entry.task)
        if not ds or not ds.get("dataset"):
            skipped_no_dataset.append(f"{entry.hf_id} [{entry.task}]")
            continue
        # Cap samples in place so both _run_pytorch_baseline and the cache key
        # observe the same value.
        original = ds.get("num_samples", run_eval._DEFAULT_SAMPLES)
        ds["num_samples"] = min(original, args.samples)
        eligible.append(entry)

    if args.list:
        run_eval.safe_print(
            f"Registry: {args.registry.name} | cache: {args.cache.name} | cap: {args.samples}"
        )
        for entry in eligible:
            ds = get_dataset_config(entry.hf_id, entry.task)
            run_eval.safe_print(
                f"  {entry.hf_id:60s} {entry.task:28s} samples={ds['num_samples']}"
            )
        if skipped_no_dataset:
            run_eval.safe_print(f"\nSkipped (no dataset config): {len(skipped_no_dataset)}")
            for name in skipped_no_dataset:
                run_eval.safe_print(f"  {name}")
        run_eval.safe_print(f"\nTotal to process: {len(eligible)}")
        return

    run_eval.safe_print(
        f"Generating {args.cache.name} | {len(eligible)} models | cap {args.samples} samples"
    )

    passed = failed = cached = 0
    for i, entry in enumerate(eligible, 1):
        ds = get_dataset_config(entry.hf_id, entry.task)
        key = run_eval._baseline_cache_key(entry.hf_id, entry.task, ds)
        run_eval.safe_print(
            f"\n[{i}/{len(eligible)}] {entry.hf_id} [{entry.task}] (n={ds['num_samples']})"
        )

        existing = run_eval._load_baseline_cache().get(key)
        if not args.force and existing and existing.get("status") == "PASS":
            cached += 1
            run_eval.safe_print(f"    cached ({existing['metric']})")
            continue

        # Build the local dataset if the config declares a build_script.
        run_eval._build_dataset(ds, args.timeout)

        result = run_eval._run_pytorch_baseline(entry, "cpu", args.timeout)
        if result["status"] == "PASS":
            passed += 1
            run_eval._store_baseline_cache(entry.hf_id, entry.task, ds, result)
            run_eval.safe_print(f"    PASS {result['metric']} ({result['elapsed']:.1f}s)")
        else:
            failed += 1
            run_eval.safe_print(f"    FAIL (exit {result.get('exit_code')})")
            for line in (result.get("stderr") or "").strip().splitlines()[-5:]:
                run_eval.safe_print(f"      {line}")

    run_eval.safe_print(
        f"\nDone. passed={passed} cached={cached} failed={failed} "
        f"-> {args.cache}"
    )


if __name__ == "__main__":
    main()
