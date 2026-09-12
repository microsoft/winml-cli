# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Automatic speech recognition evaluation for CTC and seq2seq models."""

from __future__ import annotations

import io
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..utils.eval_utils import DatasetValidationError, get_default
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from .config import DatasetConfig, WinMLEvaluationConfig


ASRMode = Literal["ctc", "seq2seq"]


def _word_error_counts(prediction: str, reference: str) -> tuple[int, int]:
    """Return word-level Levenshtein distance and reference word count."""
    predicted_words = prediction.lower().split()
    reference_words = reference.lower().split()
    if not reference_words:
        return (0 if not predicted_words else len(predicted_words), 0)

    previous = list(range(len(predicted_words) + 1))
    for ref_index, ref_word in enumerate(reference_words, start=1):
        current = [ref_index]
        for pred_index, pred_word in enumerate(predicted_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + (ref_word != pred_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def _word_error_rate(prediction: str, reference: str) -> float:
    """Return word-level Levenshtein distance divided by reference words."""
    errors, reference_words = _word_error_counts(prediction, reference)
    if reference_words == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_words


def _asr_mode(model: Any) -> ASRMode:
    """Select ASR decoding from the model's architecture contract."""
    config = getattr(model, "config", None)
    return "seq2seq" if bool(getattr(config, "is_encoder_decoder", False)) else "ctc"


class WinMLAutomaticSpeechRecognitionEvaluator(WinMLEvaluator):
    """Evaluate raw speech with explicit CTC or encoder-decoder decoding."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        if config.model_id is None:
            raise ValueError("ASR evaluation requires model_id to load its processor.")

        columns = config.dataset.columns_mapping
        self._audio_column = columns.get(
            "input_column",
            cast("str", get_default("automatic-speech-recognition", "input_column")),
        )
        self._label_column = columns.get(
            "label_column",
            cast("str", get_default("automatic-speech-recognition", "label_column")),
        )
        self._max_audio_seconds = min(
            30.0,
            float(columns.get("max_audio_seconds", "30")),
        )
        self._max_new_tokens = min(32, int(columns.get("max_new_tokens", "32")))
        self._language = columns.get("language", "english")
        self._generation_task = columns.get("generation_task", "transcribe")
        if self._max_audio_seconds <= 0 or self._max_new_tokens <= 0:
            raise ValueError("ASR audio and generation caps must be positive.")

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        self.mode = _asr_mode(model)
        super().__init__(config, model)

    def prepare_data(self) -> Dataset:
        """Load rows with audio decoding disabled so bytes/path handling is explicit."""
        data = super().prepare_data()
        try:
            from datasets import Audio

            return cast("Dataset", data.cast_column(self._audio_column, Audio(decode=False)))
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"could not expose raw audio column {self._audio_column!r}: {error}"
            ) from error

    def prepare_pipeline(self) -> Pipeline | None:  # type: ignore[override]
        """ASR uses the processor and model directly to keep decoding paths explicit."""
        return None

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Free-text transcripts need no class-label alignment."""
        return dataset

    def _decode_audio(self, value: Any) -> np.ndarray:
        """Decode an Audio row, mix to mono, resample to 16 kHz, and cap duration."""
        sampling_rate: int | None = None
        waveform: np.ndarray | None = None

        if isinstance(value, dict) and value.get("array") is not None:
            waveform = np.asarray(value["array"], dtype=np.float32)
            sampling_rate = int(value.get("sampling_rate") or 0)
        else:
            source: Any = value
            if isinstance(value, dict):
                if value.get("bytes") is not None:
                    source = io.BytesIO(value["bytes"])
                elif value.get("path"):
                    source = Path(value["path"])
            try:
                import soundfile as sf
            except ImportError as error:
                raise DatasetValidationError(
                    "Raw ASR audio decoding requires the 'audio' extra: "
                    "pip install winml-cli[audio]"
                ) from error
            try:
                decoded, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
            except Exception as error:
                raise DatasetValidationError(f"failed to decode audio: {error}") from error
            waveform = np.asarray(decoded, dtype=np.float32)

        if sampling_rate is None or sampling_rate <= 0 or waveform.size == 0:
            raise DatasetValidationError("audio must contain samples and a positive sampling rate")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1, dtype=np.float32)
        if waveform.ndim != 1:
            raise DatasetValidationError(
                f"audio must be mono or channel-last, got {waveform.shape}"
            )

        target_rate = 16_000
        if sampling_rate != target_rate:
            from scipy.signal import resample_poly

            divisor = math.gcd(sampling_rate, target_rate)
            waveform = resample_poly(
                waveform,
                target_rate // divisor,
                sampling_rate // divisor,
            ).astype(np.float32, copy=False)
        max_samples = int(target_rate * self._max_audio_seconds)
        return np.ascontiguousarray(waveform[:max_samples], dtype=np.float32)

    def _predict_ctc(self, waveform: np.ndarray) -> str:
        import torch

        encoded = self.processor(waveform, sampling_rate=16_000, return_tensors="pt")
        outputs = self.model(**dict(encoded))
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        if logits is None:
            raise ValueError("CTC ASR model must return frame logits.")
        token_ids = torch.as_tensor(logits).argmax(dim=-1)
        decoded = self.processor.batch_decode(token_ids)
        return str(decoded[0]).strip() if decoded else ""

    def _predict_seq2seq(self, waveform: np.ndarray) -> str:
        encoded = self.processor(waveform, sampling_rate=16_000, return_tensors="pt")
        generation_config = deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = self._max_new_tokens
        generation_config.num_beams = 1
        generation_config.do_sample = False
        if hasattr(self.processor, "get_decoder_prompt_ids"):
            generation_config.language = self._language
            generation_config.task = self._generation_task
            generation_config.return_timestamps = False
        generated = self.model.generate(**dict(encoded), generation_config=generation_config)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""

    def compute(self) -> dict[str, Any]:
        """Transcribe every selected row and report WER plus exact accounting."""
        predictions: list[str] = []
        references: list[str] = []
        for row_index, sample in enumerate(self.data):
            audio = sample.get(self._audio_column)
            reference = sample.get(self._label_column)
            if audio is None or not isinstance(reference, str) or not reference.strip():
                raise DatasetValidationError(
                    f"row {row_index} must contain audio and a non-empty transcript"
                )
            waveform = self._decode_audio(audio)
            prediction = (
                self._predict_seq2seq(waveform)
                if self.mode == "seq2seq"
                else self._predict_ctc(waveform)
            )
            if not prediction:
                raise ValueError(f"ASR produced an empty transcript for row {row_index}")
            predictions.append(prediction)
            references.append(reference.strip())

        if not predictions:
            raise DatasetValidationError("ASR evaluation selected no usable rows")
        word_counts = (
            _word_error_counts(prediction, reference)
            for prediction, reference in zip(predictions, references, strict=True)
        )
        error_count = 0
        reference_word_count = 0
        for errors, words in word_counts:
            error_count += errors
            reference_word_count += words
        if reference_word_count == 0:
            raise DatasetValidationError("ASR references must contain at least one word")
        return {
            "wer": error_count / reference_word_count,
            "asr_mode": self.mode,
            "requested_samples": self.config.dataset.samples,
            "processed_samples": len(predictions),
            "skipped_samples": 0,
            "max_audio_seconds": self._max_audio_seconds,
            "max_new_tokens": self._max_new_tokens if self.mode == "seq2seq" else 0,
            "num_beams": 1 if self.mode == "seq2seq" else 0,
            "return_sequences": 1,
        }


__all__ = ["WinMLAutomaticSpeechRecognitionEvaluator"]
