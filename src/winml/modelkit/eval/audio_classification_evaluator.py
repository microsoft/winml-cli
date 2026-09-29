# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generalized evaluation for utterance-level audio classifiers."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.signal import resample_poly

from ..utils.eval_utils import DatasetValidationError, get_default
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SelectedAudioSample(Mapping[str, Any]):
    model_id: int
    row: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.row[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.row)

    def __len__(self) -> int:
        return len(self.row)


class _AudioModelAdapter:
    """Normalize one utterance and run native HF or WinML inference."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        from transformers import AutoFeatureExtractor

        if not config.model_id:
            raise ValueError("model_id is required to load the audio feature extractor.")
        self.config = config
        self.model = model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        self.input_contracts = self._resolve_input_contracts()
        self._validate_output_contract()
        self.last_was_truncated = False
        self.last_window_count = 0

    def __call__(self, raw_audio: Any) -> NDArray[np.float32]:
        if isinstance(raw_audio, (list, tuple, np.ndarray)):
            waveform = np.asarray(raw_audio, dtype=np.float32)
            sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 0))
            if waveform.ndim != 1:
                raise ValueError(
                    f"pre-normalized waveform input must be 1D, got shape {waveform.shape}"
                )
        else:
            waveform, sampling_rate = WinMLAudioClassificationEvaluator._decode_audio(raw_audio)

        waveform = WinMLAudioClassificationEvaluator._to_mono(waveform)
        if waveform.size == 0:
            raise ValueError("audio waveform is empty")
        target_rate = int(getattr(self.feature_extractor, "sampling_rate", sampling_rate))
        if sampling_rate <= 0 or target_rate <= 0:
            raise ValueError("audio sampling rate must be positive")
        if sampling_rate != target_rate:
            divisor = math.gcd(sampling_rate, target_rate)
            waveform = resample_poly(
                waveform,
                target_rate // divisor,
                sampling_rate // divisor,
            ).astype(np.float32)

        self.last_was_truncated = False
        max_duration = self.config.dataset.max_duration_seconds
        if max_duration is not None:
            max_samples = max(1, int(max_duration * target_rate))
            if waveform.size > max_samples:
                waveform = waveform[:max_samples]
                self.last_was_truncated = True

        contract = self.input_contracts.get("input_values")
        if contract is not None and len(contract) == 2:
            window_length = contract[1]
            windows = [
                waveform[offset : offset + window_length]
                for offset in range(0, waveform.size, window_length)
            ]
        else:
            windows = [waveform]
        self.last_window_count = len(windows)
        return cast(
            "NDArray[np.float32]",
            np.mean(np.stack([self._run_window(window) for window in windows]), axis=0),
        )

    def _run_window(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        target_rate = int(getattr(self.feature_extractor, "sampling_rate", 0))
        kwargs: dict[str, Any] = {"sampling_rate": target_rate, "return_tensors": "pt"}
        contract = self.input_contracts.get("input_values")
        if contract is not None and len(contract) == 2:
            kwargs.update(padding="max_length", truncation=True, max_length=contract[1])
        encoded = self.feature_extractor(waveform, **kwargs)
        model_inputs = self._select_model_inputs(encoded)
        device = self.config.pipeline_device if self.config.runtime == "pytorch" else "cpu"
        model_inputs = {
            name: value.to(device)
            if hasattr(value, "to")
            else torch.as_tensor(value, device=device)
            for name, value in model_inputs.items()
        }
        output = self.model(**model_inputs)
        logits = (
            output.get("logits")
            if isinstance(output, dict)
            else getattr(output, "logits", None)
        )
        if logits is None:
            raise ValueError("audio-classification model output does not contain logits")
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        array = np.asarray(logits, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"expected logits shape [1, classes], got {array.shape}")
        return cast("NDArray[np.float32]", array[0])

    def _resolve_input_contracts(self) -> dict[str, list[int]]:
        io_config = getattr(self.model, "io_config", None) or {}
        names = io_config.get("input_names") or []
        shapes = io_config.get("input_shapes") or []
        if not names and not shapes:
            return {}
        if len(names) != len(shapes) or len(names) != 1:
            raise ValueError("audio-classification evaluation requires exactly one model input")
        shape = list(shapes[0])
        if len(shape) < 2 or any(not isinstance(value, int) for value in shape[1:]):
            raise ValueError(
                "audio-classification evaluation requires static non-batch input shapes"
            )
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("audio-classification evaluation requires batch size 1")
        return {str(names[0]): [1, *(int(value) for value in shape[1:])]}

    def _validate_output_contract(self) -> None:
        io_config = getattr(self.model, "io_config", None) or {}
        names = io_config.get("output_names") or []
        shapes = io_config.get("output_shapes") or []
        if not names and not shapes:
            return
        if names != ["logits"] or len(shapes) != 1:
            raise ValueError("audio-classification evaluation requires exactly one 'logits' output")
        shape = list(shapes[0])
        if (
            len(shape) != 2
            or (isinstance(shape[0], int) and shape[0] != 1)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError("audio-classification evaluation requires logits shape [1, classes]")

    def _select_model_inputs(self, encoded: Any) -> dict[str, Any]:
        values = dict(encoded)
        if not self.input_contracts:
            if not values:
                raise ValueError("audio feature extractor produced no model inputs")
            return values
        selected: dict[str, Any] = {}
        for name, expected_shape in self.input_contracts.items():
            if name not in values:
                raise ValueError(
                    f"audio feature extractor output must contain {name!r}; got {sorted(values)}"
                )
            actual_shape = list(getattr(values[name], "shape", ()))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"audio feature extractor produced {name} shape {actual_shape}; "
                    f"expected {expected_shape}"
                )
            selected[name] = values[name]
        return selected


class WinMLAudioClassificationEvaluator(WinMLEvaluator):
    """Evaluate single-label audio classifiers using accuracy and macro-F1."""

    def __init__(self, config: WinMLEvaluationConfig, model: WinMLPreTrainedModel) -> None:
        mapping = config.dataset.columns_mapping
        self.audio_column = mapping.get(
            "input_column", get_default("audio-classification", "input_column")
        )
        self.label_column = mapping.get(
            "label_column", get_default("audio-classification", "label_column")
        )
        if self.audio_column is None or self.label_column is None:
            raise DatasetValidationError(
                "audio-classification requires input_column and label_column defaults"
            )
        self.eligible_count = 0
        self.selected_count = 0
        self.dataset_label_to_model_id: dict[int | str, int] = {}
        self.scalar_string_target = False
        self.model_id2label, self.model_label2id = self._model_labels(model)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Any:
        """Create the shared native-HF/WinML audio adapter."""
        return _AudioModelAdapter(self.config, self.model)

    def prepare_data(self) -> list[_SelectedAudioSample]:
        """Load raw media, resolve exact labels, and select balanced rows."""
        from datasets import load_dataset, load_from_disk

        ds = self.config.dataset
        try:
            ds_path = Path(ds.path).expanduser() if ds.path else None
            dataset = (
                load_from_disk(str(ds_path))
                if ds_path and ds_path.is_dir()
                else load_dataset(
                    ds.path,
                    name=ds.name,
                    split=ds.split,
                    streaming=ds.streaming,
                    revision=ds.revision,
                )
            )
            if hasattr(dataset, "keys") and ds.split in dataset:
                dataset = dataset[ds.split]
            elif hasattr(dataset, "keys"):
                raise DatasetValidationError(
                    f"Dataset split '{ds.split}' was not found; "
                    f"available splits: {sorted(str(key) for key in dataset)}"
                )
        except Exception as error:
            if isinstance(error, DatasetValidationError):
                raise
            raise DatasetValidationError(
                f"Failed to load dataset '{ds.path}' (name={ds.name!r}, split='{ds.split}'): "
                f"{error}"
            ) from error

        self._resolve_labels(dataset, ds)
        dataset = self._disable_backend_audio_decoding(dataset)
        if ds.samples <= 0:
            raise DatasetValidationError("samples must be greater than zero.")

        rows_by_label: dict[int, list[_SelectedAudioSample]] = defaultdict(list)
        rows = list(dataset) if ds.streaming else [dataset[index] for index in range(len(dataset))]
        for row in rows:
            raw_label = row[self.label_column]
            key: int | str = str(raw_label) if self.scalar_string_target else int(raw_label)
            model_id = self.dataset_label_to_model_id.get(key)
            if model_id is None:
                continue
            self.eligible_count += 1
            rows_by_label[model_id].append(_SelectedAudioSample(model_id, row))

        if ds.shuffle:
            for model_id, bucket in rows_by_label.items():
                random.Random(ds.seed + model_id).shuffle(bucket)
        selected: list[_SelectedAudioSample] = []
        offsets = dict.fromkeys(rows_by_label, 0)
        while len(selected) < ds.samples:
            added = False
            for model_id in sorted(rows_by_label):
                offset = offsets[model_id]
                if offset < len(rows_by_label[model_id]):
                    selected.append(rows_by_label[model_id][offset])
                    offsets[model_id] += 1
                    added = True
                    if len(selected) == ds.samples:
                        break
            if not added:
                break
        self.selected_count = len(selected)
        if not selected:
            raise DatasetValidationError(
                "No samples remain after label filtering. Dataset and model labels have no overlap."
            )
        return selected

    def align_labels(self, dataset: Any, ds_config: DatasetConfig) -> Any:
        """Preserve raw labels because alignment happens before sampling."""
        return dataset

    @staticmethod
    def _model_labels(model: Any) -> tuple[dict[int, str], dict[str, int]]:
        config = getattr(model, "config", None)
        raw_id2label = getattr(config, "id2label", None) or {}
        id2label = {int(index): str(label) for index, label in raw_id2label.items()}
        if not id2label or sorted(id2label) != list(range(len(id2label))):
            raise DatasetValidationError(
                "model.config.id2label must define contiguous class IDs starting at zero."
            )
        label2id = {label: index for index, label in id2label.items()}
        if len(label2id) != len(id2label):
            raise DatasetValidationError("model.config.id2label contains duplicate label names.")
        return id2label, label2id

    def _resolve_labels(self, dataset: Any, ds: DatasetConfig) -> None:
        from datasets import ClassLabel, IterableDataset, Value

        columns = set(dataset.column_names)
        missing = [
            column
            for column in (self.audio_column, self.label_column)
            if column not in columns
        ]
        if missing:
            raise DatasetValidationError(
                f"missing required column(s) {missing}; dataset has {sorted(columns)}"
            )
        feature = dataset.features[self.label_column]
        if isinstance(feature, ClassLabel):
            names = list(feature.names)
            name_to_raw: dict[str, int | str] = {
                name: index for index, name in enumerate(names)
            }
        elif isinstance(feature, Value) and feature.dtype == "string":
            self.scalar_string_target = True
            names = (
                sorted((ds.label_mapping or self.model_label2id).keys())
                if isinstance(dataset, IterableDataset)
                else sorted({str(value) for value in dataset[self.label_column]})
            )
            name_to_raw = {name: name for name in names}
        else:
            raise DatasetValidationError(
                f"Column '{self.label_column}' must be a ClassLabel or string Value; "
                f"got {feature!r}."
            )

        mapping = ds.label_mapping
        if mapping is None:
            mapping = {
                name: self.model_label2id[name]
                for name in names
                if name in self.model_label2id
            }
        valid_ids = set(self.model_id2label)
        for name, model_id in mapping.items():
            if name not in name_to_raw:
                continue
            target = int(model_id)
            if target not in valid_ids:
                raise DatasetValidationError(
                    f"Label mapping target {target} for '{name}' is not present in "
                    "model.config.id2label."
                )
            self.dataset_label_to_model_id[name_to_raw[name]] = target
        if not self.dataset_label_to_model_id:
            raise DatasetValidationError(
                "No samples remain after label filtering. Dataset and model labels have no "
                "exact overlap; provide --label-mapping with authoritative semantics."
            )

    def _disable_backend_audio_decoding(self, dataset: Any) -> Any:
        from datasets import Audio, IterableDataset, Value

        feature = dataset.features[self.audio_column]
        if not isinstance(feature, Audio):
            return dataset
        if isinstance(dataset, IterableDataset):
            dataset = copy(dataset)
            dataset._info = dataset._info.copy()
            dataset._info.features = None
            return dataset
        return dataset.cast_column(
            self.audio_column,
            {"bytes": Value("binary"), "path": Value("string")},
        )

    def compute(self) -> dict[str, Any]:
        """Run one utterance prediction and report metrics plus accounting."""
        from .metrics.classification import ClassificationMetric

        predictions: list[str] = []
        references: list[str] = []
        selected_by_label: dict[str, int] = defaultdict(int)
        processed_by_label: dict[str, int] = defaultdict(int)
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        rejected_by_reason: dict[str, int] = defaultdict(int)
        inference_windows = 0
        truncated_samples = 0

        adapter = cast("_AudioModelAdapter", self.pipe)
        for selected in self.data:
            reference = self.model_id2label[selected.model_id]
            selected_by_label[reference] += 1
            try:
                logits = adapter(selected[self.audio_column])
                prediction = self.model_id2label[int(np.argmax(logits))]
            except (TypeError, ValueError, RuntimeError) as error:
                rejected_by_reason[type(error).__name__] += 1
                logger.warning("Skipping audio sample: %s", error)
                continue
            predictions.append(prediction)
            references.append(reference)
            inference_windows += adapter.last_window_count
            truncated_samples += int(adapter.last_was_truncated)
            processed_by_label[reference] += 1
            confusion[reference][prediction] += 1

        if not predictions:
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
                f"Too few usable samples after audio decoding/preprocessing: {details}."
            )

        represented = sorted(set(references))
        result = ClassificationMetric().compute(predictions, references, represented)
        processed = len(predictions)
        return {
            "accuracy": result["accuracy"],
            "macro_f1": result["f1"],
            "represented_classes": len(represented),
            "total_classes": len(self.model_id2label),
            "class_coverage": len(represented) / len(self.model_id2label),
            "requested_samples": self.config.dataset.samples,
            "eligible_samples": self.eligible_count,
            "selected_samples": self.selected_count,
            "processed_samples": processed,
            "inference_windows": inference_windows,
            "truncated_samples": truncated_samples,
            "rejected_samples": self.selected_count - processed,
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "per_label_processed": dict(sorted(processed_by_label.items())),
            "confusion_matrix": {
                reference: dict(sorted(values.items()))
                for reference, values in sorted(confusion.items())
            },
        }

    @staticmethod
    def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
        encoded_bytes = None
        encoded_path = None
        if isinstance(audio, dict):
            if audio.get("array") is not None and audio.get("sampling_rate") is not None:
                return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
            encoded_bytes = audio.get("bytes")
            encoded_path = audio.get("path")
            if encoded_bytes is None and not encoded_path:
                raise ValueError(
                    "audio dict requires array and sampling_rate, or encoded bytes/path"
                )
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            encoded_bytes = bytes(audio)
        elif isinstance(audio, (str, Path)):
            encoded_path = str(audio)

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
                        BytesIO(encoded_bytes), dtype="float32", always_2d=False
                    )
                else:
                    with xopen(str(encoded_path), "rb") as source:
                        waveform, sampling_rate = sf.read(
                            source, dtype="float32", always_2d=False
                        )
            except (OSError, RuntimeError) as error:
                raise ValueError(f"failed to decode audio: {error}") from error
            if waveform.ndim == 2:
                waveform = waveform.T
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)

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
        raise TypeError(f"Unsupported audio value: {type(audio).__name__}")

    @staticmethod
    def _to_mono(waveform: np.ndarray) -> NDArray[np.float32]:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            return cast("NDArray[np.float32]", waveform)
        if waveform.ndim != 2:
            raise ValueError(f"audio must be 1D or 2D, got shape {waveform.shape}")
        if waveform.shape[0] <= 8:
            return cast("NDArray[np.float32]", waveform.mean(axis=0).astype(np.float32))
        if waveform.shape[1] <= 8:
            return cast("NDArray[np.float32]", waveform.mean(axis=1).astype(np.float32))
        raise ValueError(f"cannot determine channel axis for audio shape {waveform.shape}")
