# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Audio classification evaluator with fixed-window ONNX inference.

The evaluator deliberately has no built-in dataset. Audio-classification
labels can represent languages, speakers, emotions, intents, or arbitrary
acoustic events, so callers must provide a labeled dataset whose semantics
match the checkpoint.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.signal import resample_poly

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SelectedAudioSample:
    model_id: int
    row: dict[str, Any] | None = None
    index: int | None = None


class WinMLAudioClassificationEvaluator(WinMLEvaluator):
    """Evaluate utterance-level audio classifiers using accuracy and macro-F1."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "audio-classification"
        self._audio_col = mapping.get("input_column", get_default(task, "input_column"))
        self._label_col = mapping.get("label_column", get_default(task, "label_column"))
        self._eligible_count = 0
        self._selected_count = 0
        self._dataset_id_to_model_id: dict[int, int] = {}
        self._eligible_model_labels: list[str] = []
        self._indexed_dataset: Any = None
        super().__init__(config, model)

    def prepare_pipeline(self) -> Any:
        """Load the checkpoint's audio feature extractor.

        A Transformers audio pipeline processes one fixed input at a time and
        would silently truncate long utterances for a static ONNX graph. This
        evaluator instead uses only its feature extractor and owns deterministic
        windowing and utterance-level logit aggregation.
        """
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(self.config.model_id)

    def prepare_data(self) -> list[_SelectedAudioSample]:
        """Load, label-filter, then select a seeded stratified sample.

        Filtering happens before sampling so unsupported classes cannot consume
        the requested sample budget. Shuffled streaming datasets use bounded
        per-class reservoir sampling over the complete stream. Deterministic
        streams stop once every authoritative overlapping class has supplied
        its balanced quota.
        """
        from datasets import load_dataset, load_from_disk

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
        except Exception as error:
            raise DatasetValidationError(
                f"Failed to load dataset '{ds.path}' "
                f"(name={ds.name!r}, split='{ds.split}'): {error}",
            ) from error

        self._validate_and_resolve_labels(dataset, ds)
        dataset = self._disable_backend_audio_decoding(dataset)
        if ds.samples <= 0:
            raise DatasetValidationError("samples must be greater than zero.")

        if ds.streaming:
            rows_by_label = self._streaming_reservoirs(
                dataset,
                ds.samples,
                ds.seed,
                ds.shuffle,
            )
        else:
            self._indexed_dataset = dataset
            rows_by_label = self._indexed_rows(dataset, ds.seed, ds.shuffle)

        selected = self._balanced_take(rows_by_label, ds.samples)
        self._selected_count = len(selected)
        if not selected:
            raise DatasetValidationError(
                "No samples remain after label filtering. "
                "Dataset and model labels have no overlap.",
            )
        return selected

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Keep base-class construction compatible; alignment is done before sampling."""
        return dataset

    def _validate_and_resolve_labels(self, dataset: Any, ds: DatasetConfig) -> None:
        """Validate required columns and resolve exact dataset-name to model-ID mapping."""
        from datasets import ClassLabel

        columns = set(dataset.column_names)
        missing = [col for col in (self._audio_col, self._label_col) if col not in columns]
        if missing:
            raise DatasetValidationError(
                f"missing required column(s) {missing}; dataset has {sorted(columns)}",
            )

        label_feature = dataset.features[self._label_col]
        if not isinstance(label_feature, ClassLabel):
            raise DatasetValidationError(
                f"Column '{self._label_col}' must be a ClassLabel so label semantics "
                "can be aligned explicitly.",
            )

        model_label2id = getattr(self.model.config, "label2id", None) or {}
        model_id2label = getattr(self.model.config, "id2label", None) or {}
        if not model_id2label:
            raise DatasetValidationError("model.config.id2label is required for evaluation.")
        if not model_label2id:
            model_label2id = {str(name): int(model_id) for model_id, name in model_id2label.items()}

        # A user mapping is authoritative. Without one, only exact label-name
        # identity is accepted; no locale, case, punctuation, or region inference.
        label_mapping = ds.label_mapping
        if label_mapping is None:
            label_mapping = {
                name: int(model_label2id[name])
                for name in label_feature.names
                if name in model_label2id
            }

        name_to_dataset_id = {name: index for index, name in enumerate(label_feature.names)}
        valid_model_ids = {int(key) for key in model_id2label}
        resolved: dict[int, int] = {}
        for dataset_name, model_id in label_mapping.items():
            if dataset_name not in name_to_dataset_id:
                continue
            target_id = int(model_id)
            if target_id not in valid_model_ids:
                raise DatasetValidationError(
                    f"Label mapping target {target_id} for '{dataset_name}' is not present "
                    "in model.config.id2label.",
                )
            resolved[name_to_dataset_id[dataset_name]] = target_id

        if not resolved:
            raise DatasetValidationError(
                "No samples remain after label filtering. Dataset and model labels have "
                "no exact overlap; provide --label-mapping with authoritative semantics.",
            )

        self._dataset_id_to_model_id = resolved
        self._eligible_model_labels = sorted(
            {self._decode_model_label(model_id) for model_id in resolved.values()},
        )

    def _disable_backend_audio_decoding(self, dataset: Any) -> Any:
        """Keep dataset audio as bytes/path so decoding does not require TorchCodec."""
        from datasets import Audio

        audio_feature = dataset.features[self._audio_col]
        if isinstance(audio_feature, Audio) and audio_feature.decode:
            return dataset.cast_column(
                self._audio_col,
                Audio(sampling_rate=audio_feature.sampling_rate, decode=False),
            )
        return dataset

    def _indexed_rows(
        self,
        dataset: Any,
        seed: int,
        shuffle: bool,
    ) -> dict[int, list[_SelectedAudioSample]]:
        """Collect shuffled eligible row indices without decoding the audio column."""
        indices: dict[int, list[int]] = defaultdict(list)
        for index, raw_label in enumerate(dataset[self._label_col]):
            dataset_id = int(raw_label)
            if dataset_id in self._dataset_id_to_model_id:
                indices[self._dataset_id_to_model_id[dataset_id]].append(index)
        self._eligible_count = sum(len(items) for items in indices.values())

        rows: dict[int, list[_SelectedAudioSample]] = {}
        for model_id, label_indices in sorted(indices.items()):
            if shuffle:
                random.Random(seed + model_id).shuffle(label_indices)
            selected_indices = label_indices[: self.config.dataset.samples]
            rows[model_id] = [
                _SelectedAudioSample(model_id=model_id, index=index) for index in selected_indices
            ]
        return rows

    def _streaming_reservoirs(
        self,
        dataset: Any,
        limit: int,
        seed: int,
        shuffle: bool,
    ) -> dict[int, list[_SelectedAudioSample]]:
        """Select streaming rows without weakening requested shuffle semantics."""
        if not shuffle:
            return self._streaming_balanced_prefix(dataset, limit)

        # True reservoir sampling requires visiting the complete stream. Keep
        # this path separate from deterministic bounded selection so --shuffle
        # never presents a prefix as a random sample.
        reservoirs: dict[int, list[_SelectedAudioSample]] = defaultdict(list)
        seen: dict[int, int] = defaultdict(int)
        rng = random.Random(seed)
        for row in dataset:
            dataset_id = int(row[self._label_col])
            model_id = self._dataset_id_to_model_id.get(dataset_id)
            if model_id is None:
                continue
            self._eligible_count += 1
            seen[model_id] += 1
            bucket = reservoirs[model_id]
            selected = _SelectedAudioSample(model_id=model_id, row=row)
            if len(bucket) < limit:
                bucket.append(selected)
            else:
                replacement = rng.randrange(seen[model_id])
                if replacement < limit:
                    bucket[replacement] = selected
        return dict(reservoirs)

    def _streaming_balanced_prefix(
        self,
        dataset: Any,
        limit: int,
    ) -> dict[int, list[_SelectedAudioSample]]:
        """Collect a deterministic balanced prefix and stop when quotas are full.

        Quotas are based on the unique model classes resolved from the
        authoritative exact label mapping, not on labels encountered so far.
        Consequently an absent or short class forces stream exhaustion and a
        short selection instead of silently redistributing its quota.
        """
        quotas = self._balanced_quotas(self._dataset_id_to_model_id.values(), limit)
        rows: dict[int, list[_SelectedAudioSample]] = defaultdict(list)
        pending = sum(quota > 0 for quota in quotas.values())

        for row in dataset:
            dataset_id = int(row[self._label_col])
            model_id = self._dataset_id_to_model_id.get(dataset_id)
            if model_id is None:
                continue
            self._eligible_count += 1
            bucket = rows[model_id]
            quota = quotas[model_id]
            if len(bucket) >= quota:
                continue
            bucket.append(_SelectedAudioSample(model_id=model_id, row=row))
            if len(bucket) == quota:
                pending -= 1
                if pending == 0:
                    break
        return dict(rows)

    @staticmethod
    def _balanced_quotas(model_ids: Any, limit: int) -> dict[int, int]:
        """Assign deterministic per-class quotas whose sum is ``limit``."""
        labels = sorted({int(model_id) for model_id in model_ids})
        if not labels:
            return {}
        base, remainder = divmod(limit, len(labels))
        return {label: base + (index < remainder) for index, label in enumerate(labels)}

    @staticmethod
    def _balanced_take(
        rows_by_label: dict[int, list[_SelectedAudioSample]],
        limit: int,
    ) -> list[_SelectedAudioSample]:
        """Round-robin classes to form a balanced sample and redistribute shortages."""
        selected: list[_SelectedAudioSample] = []
        offsets = dict.fromkeys(rows_by_label, 0)
        labels = sorted(rows_by_label)
        while len(selected) < limit:
            added = False
            for label in labels:
                offset = offsets[label]
                if offset < len(rows_by_label[label]):
                    selected.append(rows_by_label[label][offset])
                    offsets[label] += 1
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
        return selected

    def compute(self) -> dict[str, Any]:
        """Run all windows and report utterance accuracy, macro-F1, and accounting."""
        from .metrics import ClassificationMetric

        predictions: list[str] = []
        references: list[str] = []
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        processed_by_label: dict[str, int] = defaultdict(int)
        selected_by_label: dict[str, int] = defaultdict(int)
        rejected_by_reason: dict[str, int] = defaultdict(int)

        for selected in self.data:
            reference = self._decode_model_label(selected.model_id)
            selected_by_label[reference] += 1
            try:
                if selected.row is not None:
                    sample = selected.row
                elif selected.index is not None:
                    sample = self._indexed_dataset[selected.index]
                else:
                    raise ValueError("selected audio sample has no row or dataset index")
                windows = self._preprocess_audio(sample[self._audio_col])
            except (TypeError, ValueError, RuntimeError) as error:
                rejected_by_reason[type(error).__name__] += 1
                logger.warning("Skipping audio sample: %s", error)
                continue

            window_logits = [self._infer_window(window) for window in windows]
            utterance_logits = np.mean(np.stack(window_logits), axis=0)
            prediction_id = int(np.argmax(utterance_logits))
            prediction = self._decode_model_label(prediction_id)
            predictions.append(prediction)
            references.append(reference)
            confusion[reference][prediction] += 1
            processed_by_label[reference] += 1

        if not references:
            raise DatasetValidationError("No audio samples were successfully processed.")

        insufficient = {
            label: (processed_by_label[label], min(5, selected_count))
            for label, selected_count in selected_by_label.items()
            if processed_by_label[label] < min(5, selected_count)
        }
        if insufficient:
            details = ", ".join(
                f"{label}={actual}/{required} required"
                for label, (actual, required) in sorted(insufficient.items())
            )
            raise DatasetValidationError(
                f"Too few usable samples after audio decoding/preprocessing: {details}.",
            )

        metric = ClassificationMetric().compute(
            predictions,
            references,
            self._eligible_model_labels,
        )
        processed = len(references)
        return {
            "accuracy": metric["accuracy"],
            "macro_f1": metric["f1"],
            "requested_samples": self.config.dataset.samples,
            "eligible_samples": self._eligible_count,
            "selected_samples": self._selected_count,
            "processed_samples": processed,
            "rejected_samples": self._selected_count - processed,
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "per_label_processed": dict(sorted(processed_by_label.items())),
            "confusion_matrix": {
                reference: dict(sorted(predictions.items()))
                for reference, predictions in sorted(confusion.items())
            },
        }

    def _preprocess_audio(self, audio: Any) -> np.ndarray:
        """Convert raw audio or pre-normalized waveform input to model-sized windows."""
        input_name, input_shape = self._model_input_contract()
        if isinstance(audio, (list, tuple, np.ndarray)):
            if len(input_shape) != 2:
                raise ValueError(
                    "pre-normalized waveform input requires a rank-2 ONNX input",
                )
            input_values = np.asarray(audio, dtype=np.float32)
            if input_values.ndim != 1:
                raise ValueError(
                    f"pre-normalized waveform input must be 1D, got shape {input_values.shape}",
                )
            return self._window_waveform(
                input_values,
                int(input_shape[1]),
                float(getattr(self.pipe, "padding_value", 0.0)),
            )

        waveform, sampling_rate = self._decode_audio(audio)
        target_rate = int(getattr(self.pipe, "sampling_rate", sampling_rate))
        waveform = self._to_mono(waveform)
        if waveform.size == 0:
            raise ValueError("audio waveform is empty")
        if sampling_rate <= 0 or target_rate <= 0:
            raise ValueError("audio sampling rate must be positive")
        if sampling_rate != target_rate:
            divisor = math.gcd(sampling_rate, target_rate)
            waveform = resample_poly(
                waveform,
                target_rate // divisor,
                sampling_rate // divisor,
            ).astype(np.float32)

        encoded = self.pipe(
            waveform,
            sampling_rate=target_rate,
            return_tensors="np",
        )
        if input_name not in encoded:
            raise ValueError(
                f"audio feature extractor output must contain {input_name!r}; "
                f"got {sorted(encoded)}",
            )
        input_values = np.asarray(encoded[input_name], dtype=np.float32)
        if len(input_shape) == 2:
            return self._window_waveform(
                input_values.reshape(-1),
                int(input_shape[1]),
                float(getattr(self.pipe, "padding_value", 0.0)),
            )
        if input_values.ndim != len(input_shape) or input_values.shape[0] != 1:
            raise ValueError(
                f"feature extractor produced shape {input_values.shape}; "
                f"expected one batch matching {input_shape}",
            )
        mismatches = [
            (index, actual, expected)
            for index, (actual, expected) in enumerate(
                zip(input_values.shape[1:], input_shape[1:], strict=True),
                start=1,
            )
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                f"feature extractor produced shape {input_values.shape}; "
                f"expected {input_shape}",
            )
        return input_values

    @staticmethod
    def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
        """Decode common datasets Audio values without assuming one backend version."""
        if isinstance(audio, dict):
            if audio.get("array") is not None and audio.get("sampling_rate") is not None:
                return np.asarray(audio["array"], dtype=np.float32), int(
                    audio["sampling_rate"]
                )

            encoded_bytes = audio.get("bytes")
            encoded_path = audio.get("path")
            if encoded_bytes is not None or encoded_path:
                try:
                    import soundfile as sf
                    from datasets.download.streaming_download_manager import xopen
                except ImportError as error:
                    raise RuntimeError(
                        "Decoding encoded audio requires the 'audio' extra "
                        "(install winml-cli[audio])."
                    ) from error
                try:
                    if encoded_bytes is not None:
                        waveform, sampling_rate = sf.read(
                            BytesIO(encoded_bytes),
                            dtype="float32",
                            always_2d=False,
                        )
                    else:
                        with xopen(str(encoded_path), "rb") as source:
                            waveform, sampling_rate = sf.read(
                                source,
                                dtype="float32",
                                always_2d=False,
                            )
                except (OSError, RuntimeError) as error:
                    raise ValueError(f"failed to decode audio: {error}") from error
                # SoundFile returns [frames, channels]. Normalize its known
                # layout explicitly instead of guessing the channel axis for
                # very short clips later in preprocessing.
                if waveform.ndim == 2:
                    waveform = waveform.T
                return np.asarray(waveform, dtype=np.float32), int(sampling_rate)

            raise ValueError(
                "audio dict requires array and sampling_rate, or encoded bytes/path"
            )

        get_all_samples = getattr(audio, "get_all_samples", None)
        if callable(get_all_samples):
            decoded = get_all_samples()
            data = getattr(decoded, "data", getattr(decoded, "samples", None))
            rate = getattr(decoded, "sample_rate", getattr(decoded, "sampling_rate", None))
            if data is None or rate is None:
                raise ValueError("decoded audio does not expose samples and sampling rate")
            if hasattr(data, "detach"):
                data = data.detach().cpu().numpy()
            return np.asarray(data, dtype=np.float32), int(rate)

        array = getattr(audio, "array", None)
        rate = getattr(audio, "sampling_rate", None)
        if array is not None and rate is not None:
            return np.asarray(array, dtype=np.float32), int(rate)
        raise TypeError(f"Unsupported audio value: {type(audio).__name__}")

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> np.ndarray:
        """Return one float32 channel from mono or common stereo layouts."""
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            return waveform
        if waveform.ndim != 2:
            raise ValueError(f"audio must be 1D or 2D, got shape {waveform.shape}")
        if waveform.shape[0] <= 8:
            return np.asarray(waveform.mean(axis=0), dtype=np.float32)
        if waveform.shape[1] <= 8:
            return np.asarray(waveform.mean(axis=1), dtype=np.float32)
        raise ValueError(f"cannot determine channel axis for audio shape {waveform.shape}")

    @staticmethod
    def _window_waveform(
        waveform: np.ndarray,
        window_length: int,
        padding_value: float = 0.0,
    ) -> np.ndarray:
        """Split contiguous windows and right-pad only the final short window."""
        if window_length <= 0:
            raise ValueError("window length must be positive")
        if waveform.size == 0:
            raise ValueError("audio waveform is empty")
        count = math.ceil(waveform.size / window_length)
        windows = np.full((count, window_length), padding_value, dtype=np.float32)
        for index in range(count):
            chunk = waveform[index * window_length : (index + 1) * window_length]
            windows[index, : chunk.size] = chunk
        return windows

    def _model_input_contract(self) -> tuple[str, list[int]]:
        """Return the single static ONNX input name and shape."""
        io_config = getattr(self.model, "io_config", None) or {}
        input_names = io_config.get("input_names") or []
        input_shapes = io_config.get("input_shapes") or []
        if len(input_names) != 1 or len(input_shapes) != 1:
            raise ValueError("audio-classification evaluation requires one ONNX input")
        shape = input_shapes[0]
        if len(shape) < 2 or any(not isinstance(dimension, int) for dimension in shape[1:]):
            raise ValueError(
                "audio-classification evaluation requires a static non-batch input shape",
            )
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("audio-classification evaluation requires batch size 1")
        return str(input_names[0]), [1, *(int(dimension) for dimension in shape[1:])]

    def _infer_window(self, window: np.ndarray) -> np.ndarray:
        """Run one waveform window or feature tensor and return its 1D logits."""
        model = cast("WinMLPreTrainedModel", self.model)
        input_name = model.io_config["input_names"][0]
        output = model(**{input_name: torch.from_numpy(window[None, :])})
        if isinstance(output, dict):
            logits = output.get("logits")
            if logits is None and len(output) == 1:
                logits = next(iter(output.values()))
        else:
            logits = getattr(output, "logits", None)
        if logits is None:
            raise ValueError("audio-classification model output does not contain logits")
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        array = np.asarray(logits, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"expected logits shape [1, classes], got {array.shape}")
        return np.asarray(array[0], dtype=np.float32)

    def _decode_model_label(self, model_id: int) -> str:
        """Decode a class ID through checkpoint id2label."""
        model = cast("WinMLPreTrainedModel", self.model)
        config = model.config
        if config is None:
            return str(model_id)
        id2label = cast("dict[Any, Any]", config.id2label or {})
        label = id2label.get(model_id, id2label.get(str(model_id)))
        return str(label) if label is not None else str(model_id)
