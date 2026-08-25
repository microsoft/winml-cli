# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Audio classification evaluation for scalar and multi-label targets.

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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.signal import resample_poly

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
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
    """Decode, preprocess, and run one audio row for native HF or WinML models."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        from transformers import AutoFeatureExtractor

        if not config.model_id:
            raise ValueError("model_id is required to load the audio feature extractor.")
        self._config = config
        self.model = model
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model_id,
            trust_remote_code=getattr(config, "trust_remote_code", False),
        )
        self._input_contracts = self._resolve_input_contracts()

    def __call__(self, raw_audio: Any) -> np.ndarray:
        """Return one row of logits after exactly one model forward."""
        if isinstance(raw_audio, (list, tuple, np.ndarray)):
            waveform = np.asarray(raw_audio, dtype=np.float32)
            sampling_rate = int(getattr(self._feature_extractor, "sampling_rate", 0))
            if waveform.ndim != 1:
                raise ValueError(
                    f"pre-normalized waveform input must be 1D, got shape {waveform.shape}",
                )
        else:
            waveform, sampling_rate = WinMLAudioClassificationEvaluator._decode_audio(raw_audio)

        waveform = WinMLAudioClassificationEvaluator._to_mono(waveform)
        if waveform.size == 0:
            raise ValueError("audio waveform is empty")
        target_rate = int(
            getattr(self._feature_extractor, "sampling_rate", sampling_rate),
        )
        if sampling_rate <= 0 or target_rate <= 0:
            raise ValueError("audio sampling rate must be positive")
        if sampling_rate != target_rate:
            divisor = math.gcd(sampling_rate, target_rate)
            waveform = resample_poly(
                waveform,
                target_rate // divisor,
                sampling_rate // divisor,
            ).astype(np.float32)

        extractor_kwargs: dict[str, Any] = {
            "sampling_rate": target_rate,
            "return_tensors": "pt",
        }
        waveform_contract = self._input_contracts.get("input_values")
        if waveform_contract is not None and len(waveform_contract) == 2:
            extractor_kwargs.update(
                padding="max_length",
                truncation=True,
                max_length=waveform_contract[1],
            )
        encoded = self._feature_extractor(waveform, **extractor_kwargs)
        model_inputs = self._select_model_inputs(encoded)
        device = (
            getattr(self._config, "pipeline_device", "cpu")
            if getattr(self._config, "runtime", None) == "pytorch"
            else "cpu"
        )
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
        return cast("NDArray[np.float32]", np.asarray(array[0], dtype=np.float32))

    def _resolve_input_contracts(self) -> dict[str, list[int]]:
        io_config = getattr(self.model, "io_config", None) or {}
        names = io_config.get("input_names") or []
        shapes = io_config.get("input_shapes") or []
        if not names and not shapes:
            return {}
        if len(names) != len(shapes) or not names:
            raise ValueError(
                "audio-classification input names and shapes must have equal non-zero lengths",
            )
        contracts: dict[str, list[int]] = {}
        for name, raw_shape in zip(names, shapes, strict=True):
            shape = list(raw_shape)
            if len(shape) < 2 or any(not isinstance(value, int) for value in shape[1:]):
                raise ValueError(
                    "audio-classification evaluation requires static non-batch input shapes",
                )
            if isinstance(shape[0], int) and shape[0] != 1:
                raise ValueError("audio-classification evaluation requires batch size 1")
            contracts[str(name)] = [1, *(int(value) for value in shape[1:])]
        return contracts

    def _select_model_inputs(self, encoded: Any) -> dict[str, Any]:
        values = dict(encoded)
        if not self._input_contracts:
            if not values:
                raise ValueError("audio feature extractor produced no model inputs")
            return values
        selected: dict[str, Any] = {}
        for name, expected_shape in self._input_contracts.items():
            if name not in values:
                raise ValueError(
                    f"audio feature extractor output must contain {name!r}; got {sorted(values)}",
                )
            actual_shape = list(getattr(values[name], "shape", ()))
            if actual_shape != expected_shape:
                raise ValueError(
                    f"audio feature extractor produced {name} shape {actual_shape}; "
                    f"expected {expected_shape}",
                )
            selected[name] = values[name]
        return selected


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
        audio_col = mapping.get("input_column", get_default(task, "input_column"))
        label_col = mapping.get("label_column", get_default(task, "label_column"))
        if audio_col is None or label_col is None:
            raise DatasetValidationError(
                "audio-classification requires input_column and label_column defaults",
            )
        self._audio_col = audio_col
        self._label_col = label_col
        self._eligible_count = 0
        self._selected_count = 0
        self._dataset_id_to_model_id: dict[int, int] = {}
        self._eligible_model_labels: list[str] = []
        self._indexed_dataset: Any = None
        self._target_kind = ""
        self._label_feature: Any = None
        self._label_name_col = mapping.get("label_name_column", "human_labels")
        self._model_id2label, self._model_label2id = self._model_labels(model)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Any:
        """Create the shared native-HF/WinML callable audio adapter."""
        return _AudioModelAdapter(self.config, self.model)

    def prepare_data(self) -> list[_SelectedAudioSample] | list[dict[str, Any]]:
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
            if hasattr(dataset, "keys") and ds.split in dataset:
                dataset = dataset[ds.split]
            elif hasattr(dataset, "keys"):
                available = sorted(str(split) for split in dataset)
                raise DatasetValidationError(
                    f"Dataset split '{ds.split}' was not found; available splits: {available}",
                )
        except Exception as error:
            if isinstance(error, DatasetValidationError):
                raise
            raise DatasetValidationError(
                f"Failed to load dataset '{ds.path}' "
                f"(name={ds.name!r}, split='{ds.split}'): {error}",
            ) from error

        self._validate_target_schema(dataset, ds)
        dataset = self._disable_backend_audio_decoding(dataset)
        if ds.samples <= 0:
            raise DatasetValidationError("samples must be greater than zero.")

        if self._target_kind == "multi-label":
            if ds.streaming:
                if ds.shuffle:
                    dataset = dataset.shuffle(seed=ds.seed)
                selected_rows = list(islice(iter(dataset), ds.samples))
                self._eligible_count = len(selected_rows)
            else:
                if ds.shuffle:
                    dataset = dataset.shuffle(seed=ds.seed)
                count = min(ds.samples, len(dataset))
                selected_rows = [dataset[index] for index in range(count)]
                self._eligible_count = len(dataset)
            self._selected_count = len(selected_rows)
            if not selected_rows:
                raise DatasetValidationError("No samples were selected for evaluation.")
            return selected_rows

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

    def _validate_target_schema(self, dataset: Any, ds: DatasetConfig) -> None:
        """Validate scalar or sequence label semantics before selecting rows."""
        from datasets import ClassLabel, Sequence, Value

        columns = set(dataset.column_names)
        missing = [col for col in (self._audio_col, self._label_col) if col not in columns]
        if missing:
            raise DatasetValidationError(
                f"missing required column(s) {missing}; dataset has {sorted(columns)}",
            )
        feature = dataset.features[self._label_col]
        self._label_feature = feature
        if isinstance(feature, ClassLabel):
            self._target_kind = "single-label"
            self._validate_and_resolve_labels(dataset, ds)
        elif isinstance(feature, Sequence) and isinstance(feature.feature, (ClassLabel, Value)):
            self._target_kind = "multi-label"
        else:
            raise DatasetValidationError(
                f"Column '{self._label_col}' must be ClassLabel or a sequence of "
                f"ClassLabel/string values; got {feature!r}.",
            )

    @staticmethod
    def _model_labels(model: Any) -> tuple[dict[int, str], dict[str, int]]:
        config = getattr(model, "config", None)
        raw_id2label = getattr(config, "id2label", None) or {}
        id2label = {int(index): str(label) for index, label in raw_id2label.items()}
        if not id2label or sorted(id2label) != list(range(len(id2label))):
            raise DatasetValidationError(
                "model.config.id2label must define contiguous class IDs starting at zero.",
            )
        label2id = {label: index for index, label in id2label.items()}
        if len(label2id) != len(id2label):
            raise DatasetValidationError("model.config.id2label contains duplicate label names.")
        return id2label, label2id

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
                _SelectedAudioSample(model_id=model_id, row=dataset[index])
                for index in selected_indices
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
        """Run exactly one forward per selected row and report task metrics."""
        logits: list[np.ndarray] = []
        targets: list[Any] = []
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        processed_by_label: dict[str, int] = defaultdict(int)
        selected_by_label: dict[str, int] = defaultdict(int)
        rejected_by_reason: dict[str, int] = defaultdict(int)

        for selected in self.data:
            if isinstance(selected, _SelectedAudioSample):
                reference_id = selected.model_id
                reference = self._decode_model_label(reference_id)
                selected_by_label[reference] += 1
            else:
                reference_id = None
                reference = None
            try:
                if isinstance(selected, _SelectedAudioSample) and selected.row is not None:
                    sample = selected.row
                elif isinstance(selected, dict):
                    sample = selected
                else:
                    raise ValueError("selected audio sample has no row or dataset index")
                target = (
                    reference_id
                    if reference_id is not None
                    else self._target_for_row(sample)
                )
                prediction_logits = self.pipe(sample[self._audio_col])
            except (TypeError, ValueError, RuntimeError) as error:
                rejected_by_reason[type(error).__name__] += 1
                logger.warning("Skipping audio sample: %s", error)
                continue
            logits.append(prediction_logits)
            targets.append(target)
            if reference is not None:
                prediction = self._decode_model_label(int(np.argmax(prediction_logits)))
                confusion[reference][prediction] += 1
                processed_by_label[reference] += 1

        if not logits:
            raise DatasetValidationError("No audio samples were successfully processed.")

        if self._target_kind == "single-label":
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

        scores = np.stack(logits)
        if scores.shape[1] != len(self._model_id2label):
            raise DatasetValidationError(
                f"model returned {scores.shape[1]} classes but config defines "
                f"{len(self._model_id2label)} labels.",
            )
        metrics: dict[str, Any] = (
            self._multi_label_metrics(scores, targets)
            if self._target_kind == "multi-label"
            else self._single_label_metrics(scores, targets)
        )
        processed = len(targets)
        metrics.update({
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
        })
        return metrics

    def _target_for_row(self, row: dict[str, Any]) -> int | list[int]:
        from datasets import ClassLabel

        raw_target = row[self._label_col]
        if self._target_kind == "single-label":
            return self._resolve_label(self._label_feature.int2str(int(raw_target)))

        values = list(raw_target)
        feature = self._label_feature.feature
        decoded = (
            [feature.int2str(int(value)) for value in values]
            if isinstance(feature, ClassLabel)
            else [str(value) for value in values]
        )
        parallel_names = row.get(self._label_name_col)
        if parallel_names is not None and len(parallel_names) != len(decoded):
            raise DatasetValidationError(
                f"Columns '{self._label_col}' and '{self._label_name_col}' must contain "
                "the same number of labels.",
            )
        resolved = [
            self._resolve_label(
                value,
                fallback_name=parallel_names[index] if parallel_names is not None else None,
            )
            for index, value in enumerate(decoded)
        ]
        if not resolved:
            raise DatasetValidationError("Multi-label targets must contain at least one label.")
        return sorted(set(resolved))

    def _resolve_label(self, value: str, *, fallback_name: Any = None) -> int:
        mapping = self.config.dataset.label_mapping or {}
        if value in mapping:
            model_id = int(mapping[value])
        elif value in self._model_label2id:
            model_id = self._model_label2id[value]
        elif fallback_name is not None and str(fallback_name) in self._model_label2id:
            model_id = self._model_label2id[str(fallback_name)]
        else:
            raise DatasetValidationError(
                f"Dataset label {value!r} has no exact model-label match; provide an "
                "authoritative label mapping or parallel exact label-name column.",
            )
        if model_id not in self._model_id2label:
            raise DatasetValidationError(
                f"Label mapping target {model_id} is absent from model.config.id2label.",
            )
        return model_id

    def _single_label_metrics(
        self,
        logits: np.ndarray,
        targets: list[int],
    ) -> dict[str, Any]:
        from .metrics.classification import ClassificationMetric

        predictions = [self._model_id2label[int(index)] for index in np.argmax(logits, axis=1)]
        references = [self._model_id2label[int(index)] for index in targets]
        represented = sorted(set(references))
        result = ClassificationMetric().compute(predictions, references, represented)
        return {
            "accuracy": result["accuracy"],
            "macro_f1": result["f1"],
            "represented_classes": len(represented),
            "total_classes": len(self._model_id2label),
            "class_coverage": len(represented) / len(self._model_id2label),
        }

    @staticmethod
    def _multi_label_metrics(
        logits: np.ndarray,
        targets: list[list[int]],
    ) -> dict[str, float]:
        from sklearn.metrics import average_precision_score

        references = np.zeros_like(logits, dtype=np.int8)
        for row_index, model_ids in enumerate(targets):
            references[row_index, model_ids] = 1
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        sample_ap = float(average_precision_score(references, probabilities, average="samples"))
        micro_ap = float(average_precision_score(references, probabilities, average="micro"))
        if not np.isfinite(sample_ap) or not np.isfinite(micro_ap):
            raise DatasetValidationError("Multi-label average precision must be finite.")
        return {
            "sample_average_precision": sample_ap,
            "micro_average_precision": micro_ap,
        }

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

    def _decode_model_label(self, model_id: int) -> str:
        """Decode a class ID through checkpoint id2label."""
        model = cast("WinMLPreTrainedModel", self.model)
        config = model.config
        if config is None:
            return str(model_id)
        id2label = cast("dict[Any, Any]", config.id2label or {})
        label = id2label.get(model_id, id2label.get(str(model_id)))
        return str(label) if label is not None else str(model_id)
