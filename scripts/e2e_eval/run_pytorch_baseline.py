"""PyTorch baseline inference for accuracy evaluation (Signal 2).

Performs native PyTorch inference on a HuggingFace model using the same
dataset configuration as ``wmk eval``, so both sides are always evaluated on
identical inputs.

Dataset config is read from ``utils/dataset_config.py`` — the authoritative
source shared with run_eval.py.  When ``wmk eval`` is implemented inside
ModelKit, it should import from the same location.

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


# ---------------------------------------------------------------------------
# Task → HuggingFace auto model class
# ---------------------------------------------------------------------------

_TASK_AUTO_MODEL_CLS: dict[str, str] = {
    "image-classification": "AutoModelForImageClassification",
    "text-classification": "AutoModelForSequenceClassification",
    "sequence-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
    "object-detection": "AutoModelForObjectDetection",
    "image-segmentation": "AutoModelForSemanticSegmentation",
}

# Primary metric key returned by the HuggingFace ``evaluate`` library per task.
_TASK_HF_METRIC_KEY: dict[str, str] = {
    "image-classification": "accuracy",
    "text-classification": "accuracy",
    "sequence-classification": "accuracy",
    "automatic-speech-recognition": "wer",
    "token-classification": "overall_f1",
    "question-answering": "f1",
    "object-detection": "map",
    "image-segmentation": "mean_iou",
    "feature-extraction": "cosine_spearman",
    "sentence-similarity": "cosine_spearman",
}


# ---------------------------------------------------------------------------
# Model and dataset helpers
# ---------------------------------------------------------------------------


def _load_pytorch_model(model_id: str, task: str, device_str: str):
    """Load a native PyTorch model with the task-appropriate AutoModel class."""
    import importlib

    import torch

    from transformers import AutoConfig
    from winml.modelkit.loader.task import _get_custom_model_class
    model_type = AutoConfig.from_pretrained(model_id).model_type
    cls = _get_custom_model_class(model_type, task)
    if cls is None:
        cls_name = _TASK_AUTO_MODEL_CLS.get(task, "AutoModel")
        transformers = importlib.import_module("transformers")
        cls = getattr(transformers, cls_name)
    _out(f"Loading {cls.__name__} for {model_id} on {device_str}")
    device = torch.device(device_str if device_str != "cuda" or torch.cuda.is_available() else "cpu")
    return cls.from_pretrained(model_id).to(device).eval()


def _build_dataset_config(ds_dict: dict, num_samples: int):
    """Convert registry config dict to DatasetConfig.

    The registry uses a "dataset" key (normalised from "path" by
    dataset_config.py).  DatasetConfig uses "path".
    """
    from modelkit.datasets.config import DatasetConfig

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


def _extract_metric(metrics: dict, task: str, metric_label: str) -> tuple[str, float]:
    """Return (label, value) for the primary metric of this task.

    Uses ``_TASK_HF_METRIC_KEY`` to locate the value inside the HF evaluator
    dict, then emits it under the label from the registry config.
    """
    hf_key = _TASK_HF_METRIC_KEY.get(task, "accuracy")
    if hf_key in metrics:
        return metric_label, float(metrics[hf_key])
    # Fallback: first score-range value (0–1), skipping timing/throughput fields
    # that HF evaluator injects (total_time_in_seconds, samples_per_second, etc.).
    _SKIP_KEYS = {"total_time_in_seconds", "samples_per_second", "latency_in_seconds"}
    for k, v in metrics.items():
        if k not in _SKIP_KEYS and isinstance(v, (int, float)) and 0.0 <= v <= 1.0:
            return metric_label, float(v)
    raise ValueError(
        f"No score metric found in evaluator output for task '{task}': {metrics}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
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
                _out(f"WARNING: --columns-mapping is not valid JSON, ignoring: {args.columns_mapping}")
        ds_config_dict: dict | None = {
            "dataset": args.dataset,
            "split": args.split or "validation",
            **({"dataset_config": args.dataset_config} if args.dataset_config else {}),
            **({"columns_mapping": columns_mapping} if columns_mapping else {}),
            **({"label_mapping_file": args.label_mapping_file} if args.label_mapping_file else {}),
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
    metric_label = ds_config_dict.get("metric", _TASK_HF_METRIC_KEY.get(task, "accuracy"))

    _out(f"Task: {task} | Model: {model_id} | Device: {args.device} | Samples: {num_samples}")
    _out(f"Dataset: {ds_config_dict.get('dataset')} / {ds_config_dict.get('dataset_config', '')} [{ds_config_dict.get('split', 'validation')}]")

    try:
        from modelkit.eval.base_evaluator import WinMLEvaluator
        from modelkit.eval.config import WinMLEvaluationConfig

        pytorch_model = _load_pytorch_model(model_id, task, args.device)
        dataset_config = _build_dataset_config(ds_config_dict, num_samples)

        eval_config = WinMLEvaluationConfig(
            model_id=model_id,
            task=task,
            device=args.device,
            dataset=dataset_config,
        )

        from modelkit.eval.evaluate import _EVALUATOR_REGISTRY
        evaluator_cls = _EVALUATOR_REGISTRY.get(task, WinMLEvaluator)
        task_evaluator = evaluator_cls(eval_config, pytorch_model)

        metrics = task_evaluator.compute()

        metric_name, value = _extract_metric(metrics, task, metric_label)
        # Emit result as last stdout line (parsed by run_eval.py accuracy phase)
        _emit_result(metric_name, value, num_samples)
    except Exception as exc:
        _out(f"ERROR: evaluation failed: {exc}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
