#!/usr/bin/env python3
"""Run eval tests for all example configs under a given EP.

Usage:
    python scripts/run_example_tests.py --ep qnn --hardware npu --device npu
    python scripts/run_example_tests.py --ep qnn --hardware npu --device npu --timeout 600
    python scripts/run_example_tests.py --ep openvino --hardware cpu --device cpu --eval-only
    python scripts/run_example_tests.py --ep vitisai --hardware npu --device npu \
        --models microsoft_resnet-50,BAAI_bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_eval(
    hf_id: str,
    config_path: Path,
    output_path: Path,
    device: str,
    timeout: int,
    trust_remote_code: bool = False,
) -> str:
    """Run winml eval and return 'PASS', 'FAIL', or 'TIMEOUT'."""
    cmd = [
        sys.executable, "-m", "winml.modelkit", "eval",
        "-m", hf_id,
        "--device", device,
        "-c", str(config_path),
        "-o", str(output_path),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)
        )
        if result.returncode == 0 and output_path.exists():
            return "PASS"
        # Save error
        err_path = output_path.with_suffix(".error.txt")
        err_path.write_text(result.stderr[-2000:] if result.stderr else "Unknown error")
        return "FAIL"
    except subprocess.TimeoutExpired:
        timeout_path = output_path.with_suffix(".timeout")
        timeout_path.write_text("timeout")
        return "TIMEOUT"


def run_perf(
    hf_id: str,
    config_path: Path,
    output_path: Path,
    device: str,
    timeout: int,
) -> str:
    """Run winml perf and return 'PASS', 'FAIL', or 'TIMEOUT'."""
    cmd = [
        sys.executable, "-m", "winml.modelkit", "perf",
        "-m", hf_id,
        "--device", device,
        "-c", str(config_path),
        "-o", str(output_path),
    ]

    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)
        )
        if result.returncode == 0 and output_path.exists():
            return "PASS"
        err_path = output_path.with_suffix(".error.txt")
        err_path.write_text(result.stderr[-2000:] if result.stderr else "Unknown error")
        return "FAIL"
    except subprocess.TimeoutExpired:
        timeout_path = output_path.with_suffix(".timeout")
        timeout_path.write_text("timeout")
        return "TIMEOUT"


def infer_hf_id(config_path: Path) -> str | None:
    """Extract HF model ID from config's quant.model_name or loader."""
    try:
        cfg = json.loads(config_path.read_text())
        # Try quant.model_name first
        model_name = (cfg.get("quant") or {}).get("model_name")
        if model_name:
            return model_name
        # Fallback: derive from folder name
        slug = config_path.parent.name
        return slug.replace("_", "/", 1)
    except Exception:
        return None


def needs_trust_remote_code(config_path: Path) -> bool:
    """Check if config has dataset_script requiring --trust-remote-code."""
    try:
        cfg = json.loads(config_path.read_text())
        dataset = (cfg.get("eval") or {}).get("dataset") or {}
        return bool(dataset.get("build_script"))
    except Exception:
        return False


def clean_caches() -> None:
    """Clean HF and winml caches to free disk space."""
    for cache_dir in [
        Path.home() / ".cache" / "winml",
        Path.home() / ".cache" / "huggingface",
    ]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


def main() -> None:
    """Entrypoint for running perf/eval on example configs."""
    parser = argparse.ArgumentParser(description="Run eval tests for example configs")
    parser.add_argument(
        "--ep",
        required=True,
        choices=["qnn", "openvino", "vitisai", "nv_tensorrt_rtx", "mlas", "dml"],
        help="EP folder under examples/",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        choices=["npu", "gpu", "cpu"],
        help="Hardware sub-folder under examples/<ep>/",
    )
    parser.add_argument("--device", default="npu", help="Device (default: npu)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout per eval (default: 1200s)",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip perf, only eval")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model slugs")
    args = parser.parse_args()

    ep_dir = REPO_ROOT / "examples" / args.ep / args.hardware
    if not ep_dir.exists():
        print(f"EP directory not found: {ep_dir}")
        sys.exit(1)

    # Collect all model dirs
    model_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())
    if args.models:
        allowed = set(args.models.split(","))
        model_dirs = [d for d in model_dirs if d.name in allowed]

    # Collect all configs
    configs = sorted(
        cfg_file
        for model_dir in model_dirs
        for cfg_file in model_dir.glob("*_config.json")
    )

    print(f"EP: {args.ep}, Hardware: {args.hardware}, Device: {args.device}")
    print(f"Models: {len(model_dirs)}, Configs: {len(configs)}")
    print()

    results = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0, "SKIP": 0}
    prev_model = None

    for i, cfg_path in enumerate(configs, 1):
        stem = cfg_path.stem.replace("_config", "")
        model_slug = cfg_path.parent.name
        hf_id = infer_hf_id(cfg_path)
        if not hf_id:
            print(f"[{i}/{len(configs)}] {model_slug}/{stem} ... SKIP (no model ID)")
            results["SKIP"] += 1
            continue

        eval_output = cfg_path.parent / f"{stem}_eval.json"
        perf_output = cfg_path.parent / f"{stem}_perf.json"

        # Clean caches between different models
        if model_slug != prev_model and prev_model is not None:
            clean_caches()
        prev_model = model_slug

        trust = needs_trust_remote_code(cfg_path)

        # Run perf first if missing, even when eval was already generated.
        if not args.eval_only and not perf_output.exists():
            print(f"[{i}/{len(configs)}] {hf_id} / {stem} perf ...", end=" ", flush=True)
            perf_status = run_perf(hf_id, cfg_path, perf_output, args.device, args.timeout)
            print(perf_status)

        # Skip eval if already done or already failed/timed out.
        if eval_output.exists():
            results["SKIP"] += 1
            continue
        if eval_output.with_suffix(".error.txt").exists():
            results["SKIP"] += 1
            continue
        if eval_output.with_suffix(".timeout").exists():
            results["SKIP"] += 1
            continue

        # Run eval
        print(f"[{i}/{len(configs)}] {hf_id} / {stem} eval ...", end=" ", flush=True)
        status = run_eval(hf_id, cfg_path, eval_output, args.device, args.timeout, trust)
        results[status] += 1
        print(status)

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
