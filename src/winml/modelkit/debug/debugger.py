# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Per-op quantization error measurement.

Runs a float ONNX model and its quantized counterpart over the same inputs and
reports, per intermediate activation and per weight, the local and cumulative
SQNR (dB) using ``onnxruntime.quantization.qdq_loss_debug``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console


logger = logging.getLogger(__name__)


def _graph_output_names(model_path: str | Path) -> list[str]:
    """Return the model's graph output tensor names, in graph order."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return [o.name for o in model.graph.output]


def _sqnr_db(x: Any, y: Any) -> float:
    """SQNR (dB) wrapper that tolerates scalar (0-d) tensors.

    ORT's ``compute_signal_to_quantization_noice_ratio`` calls ``len()`` on its
    inputs, which fails for a weight that dequantizes to a numpy scalar. Coercing
    with ``atleast_1d`` keeps such tensors as length-1 arrays.
    """
    import numpy as np
    from onnxruntime.quantization.qdq_loss_debug import (
        compute_signal_to_quantization_noice_ratio,
    )

    return compute_signal_to_quantization_noice_ratio(np.atleast_1d(x), np.atleast_1d(y))


def _summarize(values: Any) -> dict:
    """Return count/mean/std/min/max for a sequence of SQNR values.

    ``None`` and non-finite (``nan``/``inf``) entries are skipped — the latter
    arise when ORT's SQNR hits an overflow or a zero-difference tensor.
    ``mean``/``std``/``min``/``max`` are ``None`` when no finite values remain.
    """
    import math
    import statistics

    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def debug_quantization(
    float_model_path: str | Path,
    quant_model_path: str | Path,
    *,
    samples: int = 8,
    model_id: str | None = None,
    task: str | None = None,
) -> dict:
    """Measure per-activation and per-weight SQNR between two ONNX models.

    Returns a dict with three lists:

    - ``activations``: ``{tensor_name, local_sqnr_db, cumulative_sqnr_db}`` per
      intermediate tensor. ``cumulative_sqnr_db`` is ``None`` when the float
      reference is unavailable.
    - ``weights``: ``{weight_name, weight_sqnr_db}`` per quantized weight.
    - ``model_outputs``: ``{output_name, cumulative_sqnr_db}`` per graph output.
    - ``summary``: per-category ``{count, mean, std, min, max}`` over the
      ``local``, ``cumulative``, and ``weight`` SQNR values.

    Calibration inputs come from ``DatasetCalibrationReader`` (task-aware when
    ``model_id``/``task`` are given, random otherwise). Both models run on the
    CPU execution provider, matching ORT's quantization debugging guidance.
    """
    from onnxruntime.quantization.qdq_loss_debug import (
        collect_activations,
        compute_activation_error,
        compute_weight_error,
        create_activation_matching,
        create_weight_matching,
        modify_model_output_intermediate_tensors,
    )

    from ..datasets import DatasetCalibrationReader

    console = Console()

    float_model_path = Path(float_model_path)
    quant_model_path = Path(quant_model_path)

    reader = DatasetCalibrationReader(
        model_name=model_id or "random",
        task=task or "random",
        max_samples=samples,
        model_path=float_model_path,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)
        aug_float = work_path / "augmented_float.onnx"
        aug_quant = work_path / "augmented_quant.onnx"

        console.print("[bold]Augmenting models...[/bold]")
        modify_model_output_intermediate_tensors(
            str(float_model_path), str(aug_float), save_as_external_data=True
        )
        modify_model_output_intermediate_tensors(
            str(quant_model_path), str(aug_quant), save_as_external_data=True
        )

        # Both passes must replay the same samples, so rewind between them.
        console.print("[bold]Collecting activations...[/bold]")
        float_acts = collect_activations(str(aug_float), reader)
        reader.rewind()
        qdq_acts = collect_activations(str(aug_quant), reader)

        console.print("[bold]Matching activations and weights...[/bold]")
        matched = create_activation_matching(qdq_acts, float_acts)
        act_err = compute_activation_error(matched)

        weight_match = create_weight_matching(str(float_model_path), str(quant_model_path))
        weight_err = compute_weight_error(weight_match, err_func=_sqnr_db)

    activations = [
        {
            "tensor_name": name,
            "local_sqnr_db": float(err["qdq_err"]),
            "cumulative_sqnr_db": (
                float(err["xmodel_err"]) if "xmodel_err" in err else None
            ),
        }
        for name, err in act_err.items()
    ]

    weights = [
        {"weight_name": name, "weight_sqnr_db": float(sqnr)}
        for name, sqnr in weight_err.items()
    ]

    cumulative_output = {
        name: err["xmodel_err"] for name, err in act_err.items() if "xmodel_err" in err
    }
    model_outputs = [
        {
            "output_name": name,
            "cumulative_sqnr_db": (
                float(cumulative_output[name]) if name in cumulative_output else None
            ),
        }
        for name in _graph_output_names(float_model_path)
    ]

    summary = {
        "local": _summarize(a["local_sqnr_db"] for a in activations),
        "cumulative": _summarize(a["cumulative_sqnr_db"] for a in activations),
        "weight": _summarize(w["weight_sqnr_db"] for w in weights),
    }

    return {
        "activations": activations,
        "weights": weights,
        "model_outputs": model_outputs,
        "summary": summary,
    }
