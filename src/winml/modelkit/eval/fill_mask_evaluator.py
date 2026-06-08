# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Fill-mask evaluator using pseudo-perplexity (Salazar et al. 2020).

For each real (non-special, non-pad) token, builds an input where that one
position is replaced by ``[MASK]``, runs the model, and records
``log P(original_token | context)``. The aggregate is
``PPPL = exp(-mean log P)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig


class WinMLFillMaskEvaluator(WinMLEvaluator):
    """Evaluate MLMs via pseudo-perplexity."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from transformers import AutoTokenizer

        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        self._input_col = mapping.get("input_column", get_default("fill-mask", "input_column"))
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Bypass the HF pipeline pattern used by other evaluators.

        Pseudo-perplexity requires a per-position masking protocol (mask one token
        at a time and score the original token from the logits), which doesn't map
        to the HF ``fill-mask`` pipeline's top-k prediction output. ``compute()`` is
        fully overridden and calls the model directly, so ``self.pipe`` is unused.
        """
        return None  # type: ignore[return-value]

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No class labels for fill-mask."""
        return dataset

    def _max_length(self) -> int | None:
        """Fixed seq_len from the ONNX model, else None (HF PyTorch / dynamic)."""
        io_config = getattr(self.model, "io_config", None) or {}
        shapes = io_config.get("input_shapes") or [[]]
        if len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
            return shapes[0][1]
        return None

    def _logits(self, outputs: Any) -> torch.Tensor:
        if not isinstance(outputs, dict):
            return outputs.logits
        if "logits" not in outputs:
            raise KeyError(f"Model output dict has no 'logits' key; got keys {list(outputs)}.")
        return outputs["logits"]

    def _score(
        self,
        encoding: dict[str, torch.Tensor],
        positions: list[int],
    ) -> torch.Tensor:
        """Return log P(original | context) at each position (one forward per position)."""
        import torch
        import torch.nn.functional as F

        input_ids = encoding["input_ids"]
        mask_id = self._tokenizer.mask_token_id
        log_probs: list[torch.Tensor] = []

        for pos in positions:
            original = int(input_ids[0, pos])
            input_ids[0, pos] = mask_id
            with torch.no_grad():
                logits = self._logits(self.model(**encoding))
            log_probs.append(F.log_softmax(logits[0, pos], dim=-1)[original])
            input_ids[0, pos] = original  # restore for next iteration

        return torch.stack(log_probs) if log_probs else torch.empty(0)

    def compute(self) -> dict[str, Any]:
        """Run pseudo-perplexity evaluation over the dataset."""
        import torch

        from .metrics import PseudoPerplexityMetric

        tok = self._tokenizer
        if tok.mask_token_id is None:
            raise RuntimeError(f"Tokenizer for {self.config.model_id} has no mask token.")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.mask_token

        max_length = self._max_length()
        tok_kwargs: dict[str, Any] = {"truncation": True, "return_tensors": "pt"}
        if max_length is not None:
            tok_kwargs["padding"] = "max_length"
            tok_kwargs["max_length"] = max_length

        metric = PseudoPerplexityMetric()

        for sample in tqdm(self.data, desc="Evaluating fill-mask (PPPL)"):
            text = sample[self._input_col]
            if not text or not text.strip():
                continue

            encoding = {
                k: v for k, v in tok(text, **tok_kwargs).items()
                if isinstance(v, torch.Tensor)
            }
            ids = encoding["input_ids"][0].tolist()
            specials = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)
            positions = [
                i for i, (t, s) in enumerate(zip(ids, specials, strict=True))
                if not s and t != tok.pad_token_id
            ]
            if not positions:
                continue

            metric.update(self._score(encoding, positions))

        return metric.compute()
