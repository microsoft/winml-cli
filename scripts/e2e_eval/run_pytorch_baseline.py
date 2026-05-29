# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""PyTorch baseline inference for accuracy evaluation (Signal 2).

Performs native PyTorch inference on a HuggingFace model using the same
dataset configuration as ``winml eval``, so both sides are always evaluated on
identical inputs.

Dataset config is read from ``utils/dataset_config.py`` — the authoritative
source shared with run_eval.py.  When ``winml eval`` is implemented inside
WinML CLI, it should import from the same location.

Output: prints a single JSON object as the last line on stdout:
    {"metric": "<name>", "value": <float>, "num_samples": <int>}

Exit codes:
    0  — success
    1  — task not configured / model loading error / evaluation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Ensure utils/ and modelkit package are importable when invoked as a subprocess
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.dataset_config import get_dataset_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _out(msg: str) -> None:
    """Print to stderr so it doesn't pollute the JSON stdout line."""
    print(msg, file=sys.stderr)


def _emit_result(metric: str, value: float, num_samples: int) -> None:
    """Print the metric JSON as the last stdout line."""
    print(json.dumps({"metric": metric, "value": round(value, 6), "num_samples": num_samples}))


def _emit_latency(latency: dict) -> None:
    """Print latency JSON to stdout (emitted before ``_emit_result``)."""
    print(json.dumps(latency))


def _extract_pipeline_input(sample: dict, columns_mapping: dict) -> Any:
    """Pick a single raw input from a dataset sample to feed the HF pipeline.

    Looks up common column-role keys first, then falls back to common column
    names. Returns the value verbatim — PIL.Image for vision tasks, ``str``
    for text tasks, etc.
    """
    for key in ("image_column", "text_column", "input_column", "question_column"):
        col = columns_mapping.get(key)
        if col and col in sample:
            return sample[col]
    for col in ("image", "text", "input", "question", "sentence"):
        if col in sample:
            return sample[col]
    return None


def _measure_pytorch_latency(task_evaluator: Any, warmup: int, iterations: int) -> dict:
    """Time HF pipeline calls on one dataset sample and return summary stats.

    Mirrors ``winml perf``'s ``latency_ms`` structure so the two outputs can
    be compared directly. Includes preprocess + forward + postprocess in the
    measurement (full user-perceived call).
    """
    import time

    if len(task_evaluator.data) == 0:
        raise RuntimeError("Dataset is empty; cannot measure pytorch latency")

    sample = task_evaluator.data[0]
    columns_mapping = task_evaluator.config.dataset.columns_mapping or {}
    raw_input = _extract_pipeline_input(sample, columns_mapping)
    if raw_input is None:
        raise RuntimeError(
            "Could not determine pipeline input column for latency measurement; "
            "looked for columns_mapping keys (image_column/text_column/"
            "input_column/question_column) and defaults (image/text/input/"
            "question/sentence)."
        )

    pipe = task_evaluator.pipe
    needs_cuda_sync = False
    try:
        import torch

        model_device = next(pipe.model.parameters()).device
        needs_cuda_sync = model_device.type == "cuda"
    except Exception:
        torch = None  # type: ignore[assignment]

    _out(f"PyTorch latency: warming up ({warmup} iter)...")
    for _ in range(warmup):
        pipe(raw_input)
        if needs_cuda_sync:
            torch.cuda.synchronize()

    _out(f"PyTorch latency: measuring ({iterations} iter)...")
    samples_ms: list[float] = []
    for _ in range(iterations):
        if needs_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe(raw_input)
        if needs_cuda_sync:
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    n = len(samples_ms)
    mean_ms = sum(samples_ms) / n
    p50 = samples_ms[n // 2]
    p90 = samples_ms[min(int(n * 0.9), n - 1)]
    p95 = samples_ms[min(int(n * 0.95), n - 1)]
    p99 = samples_ms[min(int(n * 0.99), n - 1)]
    return {
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(samples_ms[0], 3),
        "max_ms": round(samples_ms[-1], 3),
        "p50_ms": round(p50, 3),
        "p90_ms": round(p90, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "warmup": warmup,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Model and dataset helpers
# ---------------------------------------------------------------------------


def _load_pytorch_model(model_id: str, task: str, device_str: str):
    """Load a native PyTorch model with the task-appropriate AutoModel class."""
    import torch
    from transformers import AutoConfig

    from winml.modelkit.loader.task import resolve_task_and_model_class

    config = AutoConfig.from_pretrained(model_id)
    _, cls = resolve_task_and_model_class(config, task=task)
    _out(f"Loading {cls.__name__} for {model_id} on {device_str}")
    device = torch.device(
        device_str if device_str != "cuda" or torch.cuda.is_available() else "cpu"
    )
    return cls.from_pretrained(model_id).to(device).eval()


def _build_dataset_config(ds_dict: dict, num_samples: int):
    """Convert registry config dict to DatasetConfig.

    The registry uses a "dataset" key (normalised from "path" by
    dataset_config.py).  DatasetConfig uses "path".
    """
    from winml.modelkit.eval.config import DatasetConfig

    columns_mapping = ds_dict.get("columns_mapping", {})
    if isinstance(columns_mapping, str):
        try:
            columns_mapping = json.loads(columns_mapping)
        except json.JSONDecodeError:
            columns_mapping = {}

    # Load label mapping from file if specified
    label_mapping = None
    mapping_file = ds_dict.get("label_mapping_file")
    if mapping_file and Path(mapping_file).exists():
        label_mapping = json.loads(Path(mapping_file).read_text(encoding="utf-8"))

    return DatasetConfig(
        path=ds_dict.get("dataset"),
        name=ds_dict.get("dataset_config"),
        split=ds_dict.get("split", "validation"),
        samples=num_samples,
        columns_mapping=columns_mapping,
        label_mapping=label_mapping,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch baseline inference for accuracy evaluation (Signal 2)"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--task", help="HF task (auto-detected if omitted)")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of evaluation samples from dataset config",
    )
    # Per-model dataset config overrides (passed by run_eval.py from registry)
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset path")
    parser.add_argument("--split", default=None, help="Dataset split (e.g. test, validation)")
    parser.add_argument(
        "--dataset-config", default=None, help="HuggingFace dataset config/subset name"
    )
    parser.add_argument(
        "--columns-mapping",
        default=None,
        help="JSON object mapping column roles to dataset column names",
    )
    parser.add_argument(
        "--label-mapping-file",
        default=None,
        help="Path to JSON label mapping file (dataset label -> model ID)",
    )
    parser.add_argument(
        "--winml-metric-key",
        required=True,
        help="Lookup key for the primary metric in the evaluator output dict. "
        "Used as both the lookup key and the emitted label. Mirrors registry's "
        "``dataset_config.winml_metric_key`` (or ``dataset_config.metric`` when "
        "the former is absent).",
    )
    parser.add_argument(
        "--perf-iterations",
        type=int,
        default=0,
        help="Number of timed iterations for pytorch latency measurement. "
        "When >0, runs the pytorch model on one dataset sample repeatedly and "
        "emits a latency JSON line before the metric line. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--perf-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations excluded from latency statistics "
        "(only used when --perf-iterations > 0). Default: 10.",
    )
    return parser.parse_args()


def main() -> None:
    """Run PyTorch baseline inference for accuracy evaluation."""
    args = parse_args()
    model_id = args.model

    # Resolve task
    task = args.task
    if not task:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_id)
            task = getattr(cfg, "problem_type", None) or ""
        except Exception:
            task = ""

    if not task:
        _out(f"ERROR: --task not provided and could not be auto-detected for {model_id}")
        sys.exit(1)

    # Build dataset config dict from CLI args or registry
    if args.dataset:
        columns_mapping: dict = {}
        if args.columns_mapping:
            try:
                columns_mapping = json.loads(args.columns_mapping)
            except json.JSONDecodeError:
                _out(
                    "WARNING: --columns-mapping is not valid JSON, "
                    f"ignoring: {args.columns_mapping}"
                )
        ds_config_dict: dict | None = {
            "dataset": args.dataset,
            "split": args.split or "validation",
            **({"dataset_config": args.dataset_config} if args.dataset_config else {}),
            **({"columns_mapping": columns_mapping} if columns_mapping else {}),
            **({"label_mapping_file": args.label_mapping_file} if args.label_mapping_file else {}),
            "winml_metric_key": args.winml_metric_key,
        }
    else:
        ds_config_dict = get_dataset_config(args.model, task)

    if ds_config_dict is None:
        _out(
            f"ERROR: no dataset config for '{args.model}' (task: {task})"
            " and --dataset was not provided"
        )
        sys.exit(1)

    num_samples = args.num_samples or ds_config_dict.get("num_samples") or 100
    winml_metric_key = (
        ds_config_dict.get("winml_metric_key")
        or ds_config_dict.get("metric")
        or args.winml_metric_key
    )

    _out(f"Task: {task} | Model: {model_id} | Device: {args.device} | Samples: {num_samples}")
    ds_name = ds_config_dict.get("dataset")
    ds_cfg = ds_config_dict.get("dataset_config", "")
    ds_split = ds_config_dict.get("split", "validation")
    _out(f"Dataset: {ds_name} / {ds_cfg} [{ds_split}]")

    try:
        from winml.modelkit.eval.config import WinMLEvaluationConfig

        pytorch_model = _load_pytorch_model(model_id, task, args.device)
        dataset_config = _build_dataset_config(ds_config_dict, num_samples)

        eval_config = WinMLEvaluationConfig(
            model_id=model_id,
            task=task,
            device=args.device,
            dataset=dataset_config,
        )

        from winml.modelkit.eval.evaluate import get_evaluator_class

        evaluator_cls = get_evaluator_class(task)
        task_evaluator = evaluator_cls(eval_config, pytorch_model)

        metrics = task_evaluator.compute()

        if args.perf_iterations > 0:
            latency = _measure_pytorch_latency(
                task_evaluator,
                warmup=args.perf_warmup,
                iterations=args.perf_iterations,
            )
            _out(
                f"PyTorch latency: mean={latency['mean_ms']}ms "
                f"p50={latency['p50_ms']}ms p90={latency['p90_ms']}ms"
            )
            _emit_latency(latency)

        value = float(metrics[winml_metric_key])
        # Emit result as last stdout line (parsed by run_eval.py accuracy phase)
        _emit_result(winml_metric_key, value, num_samples)
    except Exception as exc:
        _out(f"ERROR: evaluation failed: {exc}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
