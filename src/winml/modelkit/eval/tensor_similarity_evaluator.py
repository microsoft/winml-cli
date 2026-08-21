# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tensor-similarity evaluator.

Runs a candidate and a reference on identical inputs (random by default,
drawn from :class:`RandomDataset` over the candidate's model I/O metadata)
and reports per-output tensor-parity metrics (SQNR, PSNR, cosine, MSE,
max absolute diff) via :class:`TensorSimilarityMetric`.

The evaluator receives the candidate model and loads an HF PyTorch reference
by default or a WinML reference when ``config.reference_path`` is set.

When ``config.input_data`` is set, both sides run on real tensors from a
``.npz`` archive instead of random inputs.

No labeled dataset, no HF pipeline, no preprocessor — any divergence
reflects the build pipeline (optimize / quantize / compile) only.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel
    from .config import WinMLEvaluationConfig


logger = logging.getLogger(__name__)


def _make_reference_config(config: WinMLEvaluationConfig) -> WinMLEvaluationConfig:
    """Derive an ordinary model-loading config for the compare reference."""
    if config.reference_path is not None:
        return replace(
            config,
            model_id=None,
            model_path=config.reference_path,
            reference_path=None,
            runtime="winml",
            device=config.reference_device,
            ep=config.reference_ep,
            precision="auto",
            mode="onnx",
            skip_build=True,
        )

    if config.model_id is None:
        raise ValueError("model_id is required to load the Hugging Face reference model.")

    return replace(
        config,
        model_path=None,
        reference_path=None,
        runtime="pytorch",
        device="cpu",
        ep=None,
        precision="auto",
        mode="onnx",
    )


class TensorSimilarityEvaluator:
    """Per-output tensor parity between a candidate and a reference model."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel | WinMLCompositeModel,
    ) -> None:
        from ..models.winml.composite_model import WinMLCompositeModel

        self.config = config

        # Composite models must be split into their sub-components before
        # tensor-similarity comparison — the union param keeps this runtime
        # guard live for type checkers.
        if isinstance(model, WinMLCompositeModel):
            sub_tasks = list(getattr(type(model), "_SUB_MODEL_CONFIG", {}).values())
            raise TypeError(
                "--mode compare does not support composite models directly. "
                f"Run compare per sub-component instead (sub-tasks: {sub_tasks}). "
                "Example: winml eval --mode compare --task <sub_task> "
                f"--model <sub_onnx_path> --model-id {config.model_id}"
            )
        import torch

        from .evaluate import load_model

        self.model = model
        self.reference_model = load_model(
            _make_reference_config(config),
            torch_dtype=torch.float32,
        )
        self.data = self.prepare_data()

    def prepare_data(self) -> Any:
        """Build the compare dataset over the candidate model's I/O spec.

        Uses real tensors from ``config.input_data`` (wrapped as a multi-sample
        :class:`InputDataDataset` whose leading axis is the sample axis and
        validated against the candidate's inputs) when provided, otherwise a
        :class:`RandomDataset` of synthetic inputs sized by ``config.dataset``.

        The real sample count is reported via ``EvalResult.num_samples`` (set by
        :func:`evaluate`), not by mutating ``config`` here.
        """
        if self.config.input_data is not None:
            from ..datasets.input_data import InputDataDataset

            return InputDataDataset(self.config.input_data, self.model.io_config)

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
        # Only the default path has an HF PyTorch reference.
        reference_label = "reference ONNX" if self.config.reference_path else "HF reference"

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
                            f"Candidate ONNX and {reference_label} output names do not "
                            f"overlap. candidate: {sorted(ort_keys)}, "
                            f"reference: {sorted(hf_keys)}."
                        )

                for name in common_keys:
                    metrics.setdefault(name, TensorSimilarityMetric()).update(
                        ort_out[name],
                        hf_out[name],
                    )

        if ort_keys != hf_keys:
            logger.warning(
                "Candidate ONNX and %s output names differ. candidate: %s, reference: %s.",
                reference_label,
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
    def _inference_model(model: Any, sample: dict[str, Any]) -> dict[str, Any]:
        """Run one sample through a model and return its named tensor outputs.

        Uniform for both backends: HF embeddings require int64 indices, so
        any narrower integer tensor is upcast here. WinMLSession down-casts
        to the ORT graph's declared dtype on its side, so the same dict
        feeds both ``WinMLPreTrainedModel`` and an HF reference model.
        """
        import torch

        inputs = {
            k: (v.to(torch.int64) if v.dtype in (torch.int8, torch.int16, torch.int32) else v)
            for k, v in sample.items()
        }
        output = model(**inputs)
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in output.items()
            if isinstance(tensor, torch.Tensor)
        }
