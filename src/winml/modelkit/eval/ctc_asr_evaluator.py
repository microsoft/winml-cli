# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Metadata-driven CTC automatic speech recognition evaluation."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.eval_utils import DatasetValidationError, validate_dataset_columns
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from .config import WinMLEvaluationConfig

_MAX_WINDOWS_PER_UTTERANCE = 64


class _RejectedSampleError(ValueError):
    """A dataset row that cannot be scored under the ASR contract."""


def _normalize_transcript(value: Any) -> str:
    """Normalize Unicode and collapse whitespace without changing text semantics."""
    if not isinstance(value, str):
        raise _RejectedSampleError("transcription is not a string")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _edit_distance(reference: list[str], prediction: list[str]) -> int:
    """Return Levenshtein distance with memory bounded by the shorter sequence."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def _corpus_error_rate(references: list[str], predictions: list[str], *, words: bool) -> float:
    """Compute corpus word or character error rate."""
    if len(references) != len(predictions) or not references:
        raise ValueError("WER/CER require equally sized, non-empty reference and prediction lists.")
    reference_units = [text.split() if words else list(text) for text in references]
    prediction_units = [text.split() if words else list(text) for text in predictions]
    denominator = sum(len(units) for units in reference_units)
    if denominator == 0:
        raise ValueError("WER/CER cannot score empty normalized references.")
    errors = sum(
        _edit_distance(reference, prediction)
        for reference, prediction in zip(reference_units, prediction_units, strict=True)
    )
    return errors / denominator


def _decode_audio(value: Any) -> tuple[np.ndarray, int]:
    """Decode a datasets ``Audio(decode=False)`` bytes/path value with SoundFile."""
    if not isinstance(value, dict):
        raise _RejectedSampleError("audio must be a decode=False bytes/path mapping")
    source: BytesIO | str
    audio_bytes = value.get("bytes")
    audio_path = value.get("path")
    if isinstance(audio_bytes, bytes):
        source = BytesIO(audio_bytes)
    elif isinstance(audio_path, str) and audio_path:
        source = audio_path
    else:
        raise _RejectedSampleError("audio has neither bytes nor path")

    try:
        import soundfile as sf

        waveform, sampling_rate = sf.read(source, dtype="float32", always_2d=False)
    except Exception as error:
        raise _RejectedSampleError(f"SoundFile decode failed: {error}") from error
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise _RejectedSampleError(f"decoded audio has invalid shape {waveform.shape}")
    return waveform, int(sampling_rate)


def _resample_audio(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample mono audio only when processor metadata requests another rate."""
    if source_rate == target_rate:
        return waveform
    if source_rate <= 0 or target_rate <= 0:
        raise _RejectedSampleError("audio sampling rates must be positive")
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(waveform, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _is_ctc_config(config: Any) -> bool:
    architectures = getattr(config, "architectures", None) or []
    return any(
        isinstance(architecture, str)
        and (architecture.endswith("ForCTC") or architecture == "AutoModelForCTC")
        for architecture in architectures
    )


def _load_ctc_processor(model_id: str, *, trust_remote_code: bool) -> Any:
    """Load the published processor, falling back only from an unavailable optional LM."""
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except ImportError as error:
        message = str(error)
        if "Wav2Vec2ProcessorWithLM" not in message or "pyctcdecode" not in message:
            raise

    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    except Exception as error:
        raise ValueError(
            "The checkpoint declares Wav2Vec2ProcessorWithLM, but optional pyctcdecode is "
            "unavailable and its feature extractor/tokenizer cannot form a greedy CTC processor."
        ) from error


def _configure_processor_language(processor: Any, model_config: Any) -> str | None:
    """Select and validate a tokenizer language using checkpoint metadata only."""
    tokenizer = getattr(processor, "tokenizer", processor)
    configured = getattr(model_config, "target_lang", None) or getattr(
        model_config,
        "adapter_lang",
        None,
    )
    active = getattr(tokenizer, "target_lang", None)
    adapter_attn_dim = getattr(model_config, "adapter_attn_dim", None)
    adapter_capable = (
        isinstance(adapter_attn_dim, int) and adapter_attn_dim > 0
    ) or active is not None
    if not adapter_capable:
        if configured is not None:
            raise ValueError(
                f"Checkpoint requests target language {configured!r}, but its metadata does not "
                "expose language-adapter support."
            )
        return None
    if configured is not None and configured != active:
        setter = getattr(tokenizer, "set_target_lang", None)
        if not callable(setter):
            raise ValueError(
                f"Checkpoint requests target language {configured!r}, but its tokenizer "
                "cannot select a language."
            )
        try:
            setter(configured)
        except Exception as error:
            raise ValueError(
                f"Checkpoint tokenizer cannot select requested adapter language {configured!r}."
            ) from error
        active = getattr(tokenizer, "target_lang", configured)
        if active != configured:
            raise ValueError(
                f"Checkpoint tokenizer did not activate requested adapter language {configured!r}."
            )
    if not active:
        raise ValueError("The checkpoint supports language adapters but no language is active.")
    return str(active) if active is not None else None


class WinMLCTCASREvaluator(WinMLEvaluator):
    """Evaluate metadata-resolved CTC ASR models with bounded full utterances."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        mapping = config.dataset.columns_mapping
        self._audio_column = mapping.get("input_column", "audio")
        self._transcription_column = mapping.get("label_column", "transcription")
        model_config = getattr(model, "config", None)
        if model_config is None or not _is_ctc_config(model_config):
            raise ValueError(
                "automatic-speech-recognition evaluation currently supports only "
                "metadata-resolved *ForCTC checkpoints."
            )
        if not config.model_id:
            raise ValueError("CTC ASR evaluation requires model_id to load its processor.")

        self.processor = _load_ctc_processor(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        self.target_lang = _configure_processor_language(self.processor, model_config)
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if not isinstance(sampling_rate, int) or sampling_rate <= 0:
            raise ValueError(
                "The checkpoint processor has no valid feature-extractor sampling rate."
            )
        self.sampling_rate = sampling_rate
        self.blank_token_id = self._resolve_blank_token_id(model_config)
        self.tokenizer_vocab_size = self._resolve_tokenizer_vocab_size()
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline | None:  # type: ignore[override]
        """Skip the HF pipeline because CTC decoding drives the model directly."""
        return None

    def prepare_data(self) -> Dataset:
        """Load deterministic rows and force explicit decode=False audio values."""
        from datasets import Audio, Dataset, load_dataset, load_from_disk

        ds = self.config.dataset
        try:
            ds_path = Path(ds.path).expanduser() if ds.path else None
            if ds_path and ds_path.is_dir():
                dataset = load_from_disk(str(ds_path))
            else:
                dataset = load_dataset(
                    ds.path,
                    name=ds.name,
                    split=ds.split,
                    streaming=ds.streaming,
                    revision=ds.revision,
                )
            dataset = dataset.cast_column(self._audio_column, Audio(decode=False))
        except Exception as error:
            raise DatasetValidationError(
                f"Failed to load ASR dataset '{ds.path}' "
                f"(name={ds.name!r}, split='{ds.split}'): {error}"
            ) from error

        validate_dataset_columns(dataset, "automatic-speech-recognition", ds.columns_mapping)
        if ds.streaming:
            rows = list(dataset.take(ds.samples))
            if rows and "id" in rows[0]:
                rows.sort(key=lambda row: row["id"])
            return Dataset.from_list(rows)
        if "id" in dataset.column_names:
            dataset = dataset.sort("id")
        return dataset.select(range(min(ds.samples, len(dataset))))

    def compute(self) -> dict[str, Any]:
        """Decode bounded rows and report corpus WER/CER plus accounting."""
        predictions: list[str] = []
        references: list[str] = []
        rejection_reasons: Counter[str] = Counter()

        for row in self.data:
            try:
                reference = _normalize_transcript(row.get(self._transcription_column))
                if not reference:
                    raise _RejectedSampleError("normalized transcription is empty")
                prediction = self._transcribe(row.get(self._audio_column))
            except _RejectedSampleError as error:
                rejection_reasons[str(error)] += 1
                continue
            references.append(reference)
            predictions.append(prediction)

        rejected = sum(rejection_reasons.values())
        if not predictions:
            raise DatasetValidationError(
                "No usable ASR samples remain after validation; "
                f"processed=0, rejected={rejected}, reasons={dict(rejection_reasons)}"
            )
        return {
            "wer": _corpus_error_rate(references, predictions, words=True),
            "cer": _corpus_error_rate(references, predictions, words=False),
            "predictions": predictions,
            "references": references,
            "processed_samples": len(predictions),
            "skipped_samples": 0,
            "rejected_samples": rejected,
            "rejection_reasons": dict(rejection_reasons),
            "target_lang": self.target_lang,
        }

    def _transcribe(self, audio_value: Any) -> str:
        waveform, source_rate = _decode_audio(audio_value)
        waveform = _resample_audio(waveform, source_rate, self.sampling_rate)
        encoded = self.processor(
            waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
        )
        arrays = {name: np.asarray(value) for name, value in encoded.items()}
        if "input_values" not in arrays or arrays["input_values"].ndim != 2:
            raise _RejectedSampleError("processor did not produce rank-2 input_values")

        input_names = list((getattr(self.model, "io_config", None) or {}).get("input_names", []))
        if not input_names:
            input_names = ["input_values"]
        window_size = self._fixed_waveform_size(input_names)
        sample_count = arrays["input_values"].shape[1]
        window_count = 1 if window_size is None else math.ceil(sample_count / window_size)
        if window_count > _MAX_WINDOWS_PER_UTTERANCE:
            raise _RejectedSampleError(
                f"utterance requires {window_count} windows; cap is {_MAX_WINDOWS_PER_UTTERANCE}"
            )

        predicted_ids: list[int] = []
        for window_index in range(window_count):
            inputs = self._window_inputs(
                arrays,
                input_names,
                window_index=window_index,
                window_size=window_size,
            )
            outputs = self.model(**inputs)
            logits = self._extract_logits(outputs)
            if logits.ndim != 3 or logits.shape[0] != 1:
                raise ValueError(
                    f"CTC logits must have shape [1, frames, vocab], got {logits.shape}."
                )
            if logits.shape[-1] != self.tokenizer_vocab_size:
                raise ValueError(
                    f"CTC output vocabulary {logits.shape[-1]} does not match active tokenizer "
                    f"vocabulary {self.tokenizer_vocab_size}."
                )
            if predicted_ids:
                predicted_ids.append(self.blank_token_id)
            predicted_ids.extend(np.argmax(logits, axis=-1)[0].astype(int).tolist())

        decoded = self.processor.batch_decode([predicted_ids])
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise ValueError("CTC processor.batch_decode must return one transcript per utterance.")
        return _normalize_transcript(decoded[0])

    def _fixed_waveform_size(self, input_names: list[str]) -> int | None:
        io_config = getattr(self.model, "io_config", None) or {}
        shapes = io_config.get("input_shapes", [])
        try:
            input_index = input_names.index("input_values")
            shape = shapes[input_index]
        except (ValueError, IndexError):
            return None
        return shape[1] if len(shape) == 2 and isinstance(shape[1], int) else None

    @staticmethod
    def _window_inputs(
        arrays: dict[str, np.ndarray],
        input_names: list[str],
        *,
        window_index: int,
        window_size: int | None,
    ) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        for name in input_names:
            if name not in arrays:
                raise ValueError(f"Processor did not produce required ONNX input {name!r}.")
            value = arrays[name]
            if window_size is None:
                inputs[name] = value
                continue
            start = window_index * window_size
            chunk = value[:, start : start + window_size]
            if chunk.shape[1] < window_size:
                chunk = np.pad(chunk, ((0, 0), (0, window_size - chunk.shape[1])))
            inputs[name] = chunk
        return inputs

    @staticmethod
    def _extract_logits(outputs: Any) -> np.ndarray:
        logits = (
            outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
        )
        if logits is None:
            raise ValueError("CTC model output does not contain logits.")
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        return np.asarray(logits)

    def _resolve_blank_token_id(self, model_config: Any) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        blank_token_id = getattr(tokenizer, "pad_token_id", None)
        if blank_token_id is None:
            blank_token_id = getattr(model_config, "pad_token_id", None)
        if not isinstance(blank_token_id, int):
            raise TypeError("The active CTC tokenizer has no integer blank/pad token ID.")
        return blank_token_id

    def _resolve_tokenizer_vocab_size(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        get_vocab = getattr(tokenizer, "get_vocab", None)
        vocabulary = get_vocab() if callable(get_vocab) else getattr(tokenizer, "vocab", None)
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise TypeError("The active CTC tokenizer has no vocabulary metadata.")
        return len(vocabulary)


__all__ = ["WinMLCTCASREvaluator"]
