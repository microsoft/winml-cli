# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tensor-similarity evaluator.

Runs an ONNX candidate and an HF PyTorch reference on identical random
inputs (drawn from :class:`RandomDataset` over the candidate's ONNX I/O)
and reports per-output tensor-parity metrics (SQNR, PSNR, cosine, MSE,
max absolute diff) via :class:`TensorSimilarityMetric`.

No labeled dataset, no HF pipeline, no preprocessor — any divergence
reflects the build pipeline (optimize / quantize / compile) only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig


logger = logging.getLogger(__name__)


class TensorSimilarityEvaluator:
    """Per-output tensor parity between an ONNX candidate and an HF reference."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..models.winml.composite_model import WinMLCompositeModel

        # WinMLCompositeModel and WinMLPreTrainedModel are siblings under
        # PreTrainedModel; mypy proves this branch unreachable but the
        # runtime check still guards against composite models reaching here.
        if isinstance(model, WinMLCompositeModel):  # type: ignore[unreachable]
            sub_tasks = list(  # type: ignore[unreachable]
                getattr(type(model), "_SUB_MODEL_CONFIG", {}).values()
            )
            raise TypeError(
                "--mode compare does not support composite models directly. "
                f"Run compare per sub-component instead (sub-tasks: {sub_tasks}). "
                "Example: winml eval --mode compare --task <sub_task> "
                f"--model <sub_onnx_path> --model-id {config.model_id}"
            )
        self.config = config
        self.model = model
        self.reference_model = self._load_reference_model()
        self.data = self.prepare_data()

    def _load_reference_model(self) -> Any:
        """Load the HF PyTorch reference model on CPU/fp32 in eval mode.

        Resolves the appropriate ``AutoModelFor*`` class via
        :func:`resolve_task_and_model_class` so no task-specific mapping is
        needed here.
        """
        import torch
        from transformers import AutoConfig

        from ..loader.task import resolve_task_and_model_class

        if self.config.model_id is None:
            raise ValueError(
                "model_id is required to load the HF reference model."
            )

        hf_config = AutoConfig.from_pretrained(self.config.model_id)
        _, cls = resolve_task_and_model_class(hf_config, task=self.config.task)
        logger.info("Loading HF reference %s on CPU/fp32", cls.__name__)
        # cls is a HF model class which exposes from_pretrained; not in `type`.
        return cls.from_pretrained(  # type: ignore[attr-defined]
            self.config.model_id, dtype=torch.float32
        ).eval()

    def prepare_data(self) -> Any:
        """Build a RandomDataset over the candidate ONNX's I/O spec."""
        from ..datasets.random_dataset import RandomDataset

        ds = self.config.dataset
        return RandomDataset(
            model_path=str(self.model.onnx_path),
            max_samples=int(ds.samples if ds.samples is not None else 100),
            seed=int(ds.seed if ds.seed is not None else 42),
        )

    def compute(self) -> dict[str, dict[str, float]]:
        """Run paired inference and return display-ready per-metric per-output values.

        Returns ``{f"{metric}_{stat}": {output_name: float}}`` — the flat shape
        the generic eval report renderer prints as one row per ``{metric}_{stat}``
        with ``output_name=value`` cells joined across outputs.
        """
        import torch
        from tqdm import tqdm

        from .metrics.tensor_similarity import TensorSimilarityMetric

        input_names = list(self.model.io_config["input_names"])
        metrics: dict[str, TensorSimilarityMetric] = {}
        common_keys: list[str] | None = None
        ort_keys: set[str] = set()
        hf_keys: set[str] = set()

        with torch.no_grad():
            for i in tqdm(range(len(self.data)), desc="compare", unit="sample"):
                row = self.data[i]
                sample = {name: row[name] for name in input_names}

                ort_out = self._inference_model(self.model, sample)
                hf_out = self._inference_model(self.reference_model, sample)

                if common_keys is None:
                    ort_keys, hf_keys = set(ort_out), set(hf_out)
                    common_keys = [name for name in hf_out if name in ort_keys & hf_keys]
                    if not common_keys:
                        raise ValueError(
                            f"ONNX and HF reference output names do not overlap. "
                            f"ONNX: {sorted(ort_keys)}, HF: {sorted(hf_keys)}."
                        )

                for name in common_keys:
                    metrics.setdefault(name, TensorSimilarityMetric()).update(
                        ort_out[name], hf_out[name],
                    )

        if ort_keys != hf_keys:
            logger.warning(
                "ONNX and HF reference output names differ. "
                "ONNX: %s, HF: %s.",
                sorted(ort_keys),
                sorted(hf_keys),
            )

        # Pivot per-output flat dicts -> {stat_key: {output: value}}.
        pivoted: dict[str, dict[str, float]] = {}
        for output_name, metric in metrics.items():
            for stat_key, value in metric.compute().items():
                pivoted.setdefault(stat_key, {})[output_name] = value
        return pivoted

    @staticmethod
    def _inference_model(
        model: Any, sample: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one sample through a model and return its named tensor outputs.

        Uniform for both backends: HF embeddings require int64 indices, so
        any narrower integer tensor is upcast here. WinMLSession down-casts
        to the ORT graph's declared dtype on its side, so the same dict
        feeds both ``WinMLPreTrainedModel`` and an HF reference model.
        """
        import torch

        inputs = {
            k: (
                v.to(torch.int64)
                if v.dtype in (torch.int8, torch.int16, torch.int32)
                else v
            )
            for k, v in sample.items()
        }
        output = model(**inputs)
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in output.items()
            if isinstance(tensor, torch.Tensor)
        }
