# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Binary tabular classification evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from .config import WinMLEvaluationConfig


class WinMLTabularClassificationEvaluator(WinMLEvaluator):
    """Evaluate numeric feature vectors against binary labels."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "tabular-classification"
        self._input_col = mapping.get("input_column", get_default(task, "input_column"))
        self._label_col = mapping.get("label_column", get_default(task, "label_column"))
        super().__init__(config, model)

    def prepare_pipeline(self) -> Any:
        """Use the tensor-native model directly; no HF processor is required."""
        return self.model

    def compute(self) -> dict[str, Any]:
        """Compute accuracy and macro-F1 using a zero-logit binary threshold."""
        import numpy as np
        import torch

        from .metrics import ClassificationMetric

        predictions: list[str] = []
        references: list[str] = []
        for sample in self.data:
            features = np.asarray(sample[self._input_col], dtype=np.float32)
            if features.ndim != 1:
                raise ValueError(
                    f"Tabular features must be a 1D numeric vector, got shape {features.shape}."
                )
            label = int(sample[self._label_col])
            if label not in (0, 1):
                raise ValueError(f"Binary tabular label must be 0 or 1, got {label}.")

            output = self.model(features=torch.from_numpy(features).unsqueeze(0))
            logits = getattr(output, "logits", None)
            if logits is None and isinstance(output, dict):
                logits = output.get("logits")
            if logits is None:
                raise ValueError("Tabular model output does not contain logits.")
            values = np.asarray(logits.detach().cpu(), dtype=np.float32).reshape(-1)
            if values.size == 1:
                prediction = int(values[0] >= 0.0)
            elif values.size == 2:
                prediction = int(np.argmax(values))
            else:
                raise ValueError(
                    "Binary tabular classifier must emit one logit or two class logits per "
                    f"sample, got {values.size}."
                )
            predictions.append(str(prediction))
            references.append(str(label))

        metric = ClassificationMetric()
        result = metric.compute(predictions, references, labels=["0", "1"])
        result["num_samples"] = len(references)
        return result
