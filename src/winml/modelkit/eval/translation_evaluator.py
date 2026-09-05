# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evaluator for text-to-text machine translation models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

    from ..models.winml.composite_model import WinMLCompositeModel
    from .config import DatasetConfig, WinMLEvaluationConfig


logger = logging.getLogger(__name__)


def _positive_int(mapping: Mapping[str, str], name: str, default: int) -> int:
    from ..utils.eval_utils import DatasetValidationError

    raw_value = mapping.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(f"{name} must be a positive integer") from error
    if value < 1:
        raise DatasetValidationError(f"{name} must be a positive integer")
    return value


class WinMLTranslationEvaluator(WinMLEvaluator):
    """Evaluate translations with bounded generation and corpus metrics."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLCompositeModel,
    ) -> None:
        from ..utils.eval_utils import DatasetValidationError, get_default

        mapping = config.dataset.columns_mapping
        self._source_col = mapping.get(
            "source_column",
            get_default("translation", "source_column") or "translation",
        )
        self._reference_col = mapping.get(
            "reference_column",
            get_default("translation", "reference_column") or "translation",
        )
        self._source_lang = mapping.get("source_lang")
        self._target_lang = mapping.get("target_lang")
        self._tokenizer_source_lang = mapping.get("tokenizer_source_lang", self._source_lang)
        self._tokenizer_target_lang = mapping.get("tokenizer_target_lang", self._target_lang)
        self._source_prefix = mapping.get("source_prefix", "")
        max_source_tokens = _positive_int(mapping, "max_source_tokens", 128)
        max_new_tokens = _positive_int(mapping, "max_new_tokens", 64)
        num_beams = _positive_int(mapping, "num_beams", 1)
        num_return_sequences = _positive_int(mapping, "num_return_sequences", 1)
        if num_beams != 1 or num_return_sequences != 1:
            raise DatasetValidationError(
                "translation evaluation requires num_beams=1 and num_return_sequences=1 "
                "for the static batch-one generation contract"
            )

        super().__init__(config, model)
        self._pipeline_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "num_return_sequences": 1,
            "truncation": True,
        }
        if self._tokenizer_source_lang:
            self._pipeline_kwargs["src_lang"] = self._tokenizer_source_lang
        if self._tokenizer_target_lang:
            self._pipeline_kwargs["tgt_lang"] = self._tokenizer_target_lang

        max_encoder_length = getattr(model, "max_encoder_length", None)
        if isinstance(max_encoder_length, int) and max_encoder_length > 0:
            max_source_tokens = min(max_source_tokens, max_encoder_length)
        if self.pipe.tokenizer is not None:
            self.pipe.tokenizer.model_max_length = max_source_tokens

        max_decode_length = getattr(model, "max_decode_length", None)
        if isinstance(max_decode_length, int):
            if max_decode_length <= 1:
                raise DatasetValidationError(
                    "decoder static cache must leave capacity for at least one generated token"
                )
            self._pipeline_kwargs["max_new_tokens"] = min(
                max_new_tokens,
                max_decode_length - 1,
            )
            self.pipe.generation_config.max_length = max_decode_length
            self.pipe.generation_config.max_new_tokens = None
            self.pipe.generation_config.num_beams = 1
            self.pipe.generation_config.num_return_sequences = 1
            self.pipe.generation_config.do_sample = False

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Return free-text references unchanged."""
        return dataset

    @staticmethod
    def _extract_text(value: Any, language: str | None, field: str) -> str:
        """Extract one text value from a flat string or language-keyed mapping."""
        from ..utils.eval_utils import DatasetValidationError

        if isinstance(value, Mapping):
            if not language:
                language_option = "source_lang" if field == "source" else "target_lang"
                raise DatasetValidationError(
                    f"{field} contains a translation dict; provide --column "
                    f"{language_option}=<language key>"
                )
            if language not in value:
                raise DatasetValidationError(
                    f"{field} translation dict has no language key '{language}'; "
                    f"available keys: {sorted(str(key) for key in value)}"
                )
            value = value[language]
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"{field} must resolve to a non-empty string")
        return value.strip()

    def _extract_references(self, value: Any) -> str | list[str]:
        """Extract one or more reference translations."""
        if isinstance(value, list):
            if not value:
                from ..utils.eval_utils import DatasetValidationError

                raise DatasetValidationError("reference must contain at least one translation")
            return [self._extract_text(item, self._target_lang, "reference") for item in value]
        return self._extract_text(value, self._target_lang, "reference")

    @staticmethod
    def _prediction_text(output: Any) -> str:
        """Normalize Hugging Face translation pipeline output shapes."""
        from ..utils.eval_utils import DatasetValidationError

        if isinstance(output, list):
            if not output:
                raise DatasetValidationError("pipeline returned no translations")
            output = output[0]
        if isinstance(output, Mapping):
            prediction = output.get("translation_text", output.get("generated_text", ""))
            if isinstance(prediction, str) and prediction.strip():
                return prediction.strip()
            raise DatasetValidationError("pipeline returned an empty translation")
        if isinstance(output, str) and output.strip():
            return output.strip()
        raise DatasetValidationError("pipeline returned an unsupported translation result")

    def _extract_sample(self, sample: Any) -> tuple[str, str | list[str]]:
        """Extract and validate source/reference values from one dataset row."""
        from ..utils.eval_utils import DatasetValidationError

        if not isinstance(sample, Mapping):
            raise DatasetValidationError(
                f"dataset row must be a mapping, got {type(sample).__name__}"
            )
        source = self._extract_text(sample.get(self._source_col), self._source_lang, "source")
        references = self._extract_references(sample.get(self._reference_col))
        return f"{self._source_prefix}{source}", references

    def compute(self) -> dict[str, Any]:
        """Translate each valid row and compute corpus-level metrics."""
        from tqdm.auto import tqdm

        from ..utils.eval_utils import DatasetValidationError
        from .metrics.translation import TranslationMetric

        metric = TranslationMetric()
        skipped = 0
        first_error: str | None = None
        attempted = 0

        for sample in tqdm(self.data, desc="Evaluating", unit="sample"):
            attempted += 1
            try:
                source, references = self._extract_sample(sample)
            except DatasetValidationError as error:
                logger.warning("Translation sample rejected: %s", error)
                first_error = first_error or str(error)
                skipped += 1
                continue

            output = self.pipe(source, **self._pipeline_kwargs)
            try:
                prediction = self._prediction_text(output)
                metric.update(prediction, references)
            except (DatasetValidationError, ValueError) as error:
                logger.warning("Translation output rejected: %s", error)
                first_error = first_error or str(error)
                skipped += 1

        result = metric.compute()
        if result["n_samples"] == 0:
            detail = f": {first_error}" if first_error else ""
            raise DatasetValidationError(f"No valid translation samples were evaluated{detail}")
        result["attempted"] = attempted
        result["evaluated"] = result["n_samples"]
        result["skipped"] = skipped
        return result
