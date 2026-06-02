# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Monocular depth estimation evaluator.

HF ``evaluate`` has no depth-estimation evaluator, so we run the metric
loop manually against ``DepthMetric`` (AbsRel, RMSE, delta1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    import numpy as np
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLDepthEstimationEvaluator(WinMLEvaluator):
    """Evaluator for monocular depth estimation."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "depth-estimation"
        self._input_col = mapping.get("input_column", get_default(task, "input_column"))
        self._depth_col = mapping.get("depth_column", get_default(task, "depth_column"))
        self._align = mapping.get("align", get_default(task, "align"))
        self._depth_kind = mapping.get("depth_kind", get_default(task, "depth_kind"))
        self._min_depth = float(mapping.get("min_depth", get_default(task, "min_depth")))
        max_depth_raw = mapping.get("max_depth", get_default(task, "max_depth"))
        self._max_depth: float | None
        if isinstance(max_depth_raw, str) and max_depth_raw.lower() == "none":
            self._max_depth = None
        else:
            self._max_depth = float(max_depth_raw)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and match image processor size to ONNX input shape.

        Image processors for depth and detection models often default to
        aspect-preserving resize and/or padding (e.g. Depth-Anything sets
        ``keep_aspect_ratio=True`` with ``ensure_multiple_of=14``), which
        produces a per-image output shape that does not match the static
        ONNX input shape. We override these flags so the processor produces
        exactly the target ``(h, w)`` for every input.

        Models without these attributes are unaffected.
        """
        pipe = super().prepare_pipeline()

        io_config = getattr(self.model, "io_config", None) or {}
        input_shapes = io_config.get("input_shapes", [])
        if input_shapes and len(input_shapes[0]) == 4:
            _, _, h, w = input_shapes[0]
            pipe.image_processor.size = {"height": h, "width": w}
            if hasattr(pipe.image_processor, "keep_aspect_ratio"):
                pipe.image_processor.keep_aspect_ratio = False
            if hasattr(pipe.image_processor, "do_pad"):
                pipe.image_processor.do_pad = False

        return pipe

    def compute(self) -> dict[str, Any]:
        """Run depth evaluation over all samples."""
        import numpy as np
        from tqdm import tqdm

        from .metrics import DepthMetric

        metric = DepthMetric(
            align=self._align,
            depth_kind=self._depth_kind,
            min_depth=self._min_depth,
            max_depth=self._max_depth,
        )

        skipped = 0
        for sample in tqdm(self.data, desc="Evaluating depth"):
            image = sample.get(self._input_col)
            depth = sample.get(self._depth_col)
            if image is None or depth is None:
                skipped += 1
                continue

            self._validate_image_input(image)
            result = self.pipe(image)
            pred = self._extract_predicted_depth(result)
            gt = np.asarray(depth, dtype=np.float32)
            metric.update(pred, gt)

        if skipped:
            logger.warning("Skipped %d samples with missing image or depth.", skipped)

        return metric.compute()

    def _validate_image_input(self, image: Any) -> None:
        """Raise a clear error for tensor-formatted image columns."""
        import numpy as np
        import torch
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return

        if isinstance(image, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Depth-estimation input column {self._input_col!r} must yield PIL "
                f"images; got {type(image).__name__}. Use a datasets.Image column "
                "or remove tensor/NumPy formatting before evaluation.",
            )

    @staticmethod
    def _extract_predicted_depth(result: Any) -> np.ndarray:
        """Pull the numeric depth tensor out of an HF pipeline result."""
        import numpy as np
        import torch

        if not isinstance(result, dict):
            raise TypeError(
                f"Unexpected pipeline output type: {type(result).__name__}; expected dict.",
            )

        predicted = result.get("predicted_depth")
        if predicted is None:
            raise ValueError(
                f"Pipeline output missing 'predicted_depth'; got keys {list(result)}.",
            )
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        return np.asarray(predicted, dtype=np.float32).squeeze()
