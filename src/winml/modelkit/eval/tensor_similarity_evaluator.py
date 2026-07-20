# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tensor-similarity evaluator.

Runs an ONNX candidate and a reference on identical inputs (random by
default, drawn from :class:`RandomDataset` over the candidate's ONNX I/O)
and reports per-output tensor-parity metrics (SQNR, PSNR, cosine, MSE, max
absolute diff) via :class:`TensorSimilarityMetric`.

The reference is an HF PyTorch model resolved from ``model_id`` by default.
When ``config.reference_path`` is set, the reference is instead a second
ONNX file and both sides run as raw ORT sessions (no HF config / task).
When ``config.input_data`` is set, both sides run on real tensors from a
``.npz`` archive instead of random inputs.

No labeled dataset, no HF pipeline, no preprocessor — any divergence
reflects the build pipeline (optimize / quantize / compile) only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel
    from .config import WinMLEvaluationConfig


logger = logging.getLogger(__name__)


class _ONNXSessionModel:
    """Minimal raw-ORT wrapper for two-ONNX ``--mode compare``.

    Exposes just the slice of the :class:`WinMLPreTrainedModel` surface that
    the tensor-similarity loop needs — ``onnx_path``, ``io_config`` and a
    callable returning named ``torch`` tensors — without any HF config or
    task-specific output renaming, so both sides compare on their raw ONNX
    output names.
    """

    def __init__(self, onnx_path: str, device: str = "auto", ep: Any | None = None) -> None:
        from pathlib import Path

        from ..session.session import WinMLSession

        self.onnx_path = Path(onnx_path)
        self._session = WinMLSession(onnx_path=self.onnx_path, device=device, ep=ep)

    @property
    def io_config(self) -> dict:
        """ONNX I/O metadata (delegated to the session)."""
        return self._session.io_config

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        """Run one sample and return raw outputs as named ``torch`` tensors."""
        import torch

        outputs = self._session.run(inputs)
        return {name: torch.from_numpy(arr) for name, arr in outputs.items()}


class TensorSimilarityEvaluator:
    """Per-output tensor parity between an ONNX candidate and an HF reference."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel | WinMLCompositeModel,
    ) -> None:
        from ..models.winml.composite_model import WinMLCompositeModel

        self.config = config

        # Two-ONNX compare: build both raw ORT sessions directly, bypassing the
        # HF PyTorch reference. ``model`` is None here (see evaluate._load_model).
        if config.reference_path is not None:
            self.model = _ONNXSessionModel(
                str(config.model_path), device=config.device, ep=config.ep
            )
            self.reference_model = _ONNXSessionModel(
                str(config.reference_path), device=config.device, ep=config.ep
            )
            self.data = self.prepare_data()
            return

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
        self.model = model
        self.reference_model = self._load_reference_model()
        self.data = self.prepare_data()

    def _load_reference_model(self) -> Any:
        """Load the HF PyTorch reference model on CPU/fp32 in eval mode.

        Resolves the appropriate ``AutoModelFor*`` class via
        :func:`resolve_task` so no task-specific mapping is
        needed here.
        """
        import torch
        from transformers import AutoConfig

        from ..loader.resolution import resolve_task

        if self.config.model_id is None:
            raise ValueError("model_id is required to load the HF reference model.")

        hf_config = AutoConfig.from_pretrained(self.config.model_id)
        cls = resolve_task(hf_config, task=self.config.task).model_class
        logger.info("Loading HF reference %s on CPU/fp32", cls.__name__)
        # cls is a HF model class which exposes from_pretrained; not in `type`.
        return cls.from_pretrained(  # type: ignore[attr-defined]
            self.config.model_id, dtype=torch.float32
        ).eval()

    def prepare_data(self) -> Any:
        """Build the compare dataset over the candidate ONNX's I/O spec.

        Uses real tensors from ``config.input_data`` (wrapped as a
        single-sample :class:`InputDataDataset`, validated against the
        candidate's inputs) when provided, otherwise a :class:`RandomDataset`
        of synthetic inputs sized by ``config.dataset``.
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
                        ort_out[name],
                        hf_out[name],
                    )

        if ort_keys != hf_keys:
            logger.warning(
                "ONNX and HF reference output names differ. ONNX: %s, HF: %s.",
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
