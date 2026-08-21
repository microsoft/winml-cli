# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Audio classification evaluation for scalar and multi-label targets."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from numpy.typing import NDArray
    from transformers.pipelines.base import Pipeline

    from .config import DatasetConfig, WinMLEvaluationConfig


class _AudioModelAdapter:
    """One preprocessing and forward contract for native-HF and WinML models."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        from transformers import AutoFeatureExtractor

        if not config.model_id:
            raise ValueError("model_id is required to load the audio feature extractor.")
        self._config = config
        self._model = model
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        self._input_contract = self._resolve_input_contract()

    def predict_logits(self, audio: Any) -> NDArray[np.float32]:
        """Decode and process one audio row, then perform exactly one forward."""
        import torch

        waveform, sampling_rate = self._decode_audio(audio)
        waveform = self._downmix(waveform)
        target_rate = int(getattr(self._feature_extractor, "sampling_rate", sampling_rate))
        if sampling_rate != target_rate:
            waveform = self._resample(waveform, sampling_rate, target_rate)

        processor_kwargs: dict[str, Any] = {
            "sampling_rate": target_rate,
            "return_tensors": "pt",
        }
        if self._input_contract is not None and len(self._input_contract[1]) == 2:
            sequence_length = self._input_contract[1][1]
            processor_kwargs.update(
                padding="max_length",
                truncation=True,
                max_length=sequence_length,
            )
        encoded = self._feature_extractor(waveform, **processor_kwargs)
        model_inputs = self._select_model_inputs(encoded)
        device = self._config.pipeline_device if self._config.runtime == "pytorch" else "cpu"
        model_inputs = {
            name: (
                value.to(device)
                if hasattr(value, "to")
                else torch.as_tensor(value, device=device)
            )
            for name, value in model_inputs.items()
        }
        outputs = self._model(**model_inputs)
        logits = self._extract_logits(outputs)
        array = logits.detach().float().cpu().numpy() if hasattr(logits, "detach") else logits
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"expected logits shape [1, classes], got {array.shape}")
        return cast("NDArray[np.float32]", array[0])

    def _resolve_input_contract(self) -> tuple[str, list[int]] | None:
        io_config = getattr(self._model, "io_config", None) or {}
        names = io_config.get("input_names") or []
        shapes = io_config.get("input_shapes") or []
        if not names and not shapes:
            return None
        if len(names) != 1 or len(shapes) != 1:
            raise ValueError(
                "audio-classification adapter supports one model input; "
                f"got names={names}, shapes={shapes}"
            )
        shape = list(shapes[0])
        if len(shape) < 2 or any(not isinstance(value, int) for value in shape[1:]):
            raise ValueError(
                "audio-classification adapter requires static non-batch input dimensions; "
                f"got {shape}"
            )
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError("audio-classification adapter requires batch size 1.")
        return str(names[0]), [1, *(int(value) for value in shape[1:])]

    def _select_model_inputs(self, encoded: Any) -> dict[str, Any]:
        values = dict(encoded)
        if self._input_contract is None:
            if not values:
                raise ValueError("audio feature extractor produced no model inputs.")
            return values

        input_name, expected_shape = self._input_contract
        if input_name not in values:
            raise ValueError(
                f"audio feature extractor output must contain {input_name!r}; "
                f"got {sorted(values)}"
            )
        tensor = values[input_name]
        actual_shape = list(getattr(tensor, "shape", ()))
        if actual_shape != expected_shape:
            raise ValueError(
                f"audio feature extractor produced {input_name} shape {actual_shape}; "
                f"expected {expected_shape}"
            )
        return {input_name: tensor}

    @staticmethod
    def _extract_logits(outputs: Any) -> Any:
        logits = (
            outputs.get("logits")
            if isinstance(outputs, dict)
            else getattr(outputs, "logits", None)
        )
        if logits is None:
            raise ValueError("audio-classification model output does not contain logits.")
        return logits

    @staticmethod
    def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(audio, dict):
            if audio.get("array") is not None and audio.get("sampling_rate") is not None:
                return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
            encoded_bytes = audio.get("bytes")
            encoded_path = audio.get("path")
            if encoded_bytes is not None or encoded_path:
                try:
                    import soundfile as sf
                except ImportError as error:
                    raise RuntimeError(
                        "Encoded audio decoding requires the optional audio dependencies."
                    ) from error
                source = BytesIO(encoded_bytes) if encoded_bytes is not None else str(encoded_path)
                try:
                    waveform, sampling_rate = sf.read(
                        source,
                        dtype="float32",
                        always_2d=False,
                    )
                except (OSError, RuntimeError) as error:
                    raise ValueError(f"failed to decode audio: {error}") from error
                return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        raise TypeError(
            "audio value must contain array and sampling_rate, or encoded bytes/path"
        )

    @staticmethod
    def _downmix(waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            mono = waveform
        elif waveform.ndim == 2:
            mono = waveform.mean(axis=1)
        else:
            raise ValueError(f"audio must be mono or frames-by-channels, got {waveform.shape}")
        if mono.size == 0:
            raise ValueError("audio waveform is empty.")
        return np.asarray(mono, dtype=np.float32)

    @staticmethod
    def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        from scipy.signal import resample_poly

        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("audio sampling rates must be positive.")
        divisor = math.gcd(source_rate, target_rate)
        return np.asarray(
            resample_poly(waveform, target_rate // divisor, source_rate // divisor),
            dtype=np.float32,
        )


class WinMLAudioClassificationEvaluator(WinMLEvaluator):
    """Evaluate audio classifiers with schema-driven target semantics."""

    def __init__(self, config: WinMLEvaluationConfig, model: Any) -> None:
        mapping = config.dataset.columns_mapping
        self._audio_column = mapping.get("input_column", "audio")
        self._label_column = mapping.get("label_column", "label")
        self._label_name_column = mapping.get("label_name_column", "human_labels")
        self._target_kind = ""
        self._label_feature: Any = None
        self._model_id2label, self._model_label2id = self._model_labels(model)
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create the shared native-HF/WinML audio adapter."""
        return cast("Pipeline", _AudioModelAdapter(self.config, self.model))

    def prepare_data(self) -> Dataset:
        """Load, validate, explicitly disable media decoding, and select rows."""
        from datasets import (
            Audio,
            Dataset,
            DatasetDict,
            IterableDataset,
            load_dataset,
            load_from_disk,
        )

        ds = self.config.dataset
        try:
            ds_path = Path(ds.path).expanduser() if ds.path else None
            if ds_path and ds_path.is_dir():
                loaded = load_from_disk(str(ds_path))
                if isinstance(loaded, DatasetDict):
                    if ds.split not in loaded:
                        raise DatasetValidationError(
                            f"saved dataset has no split {ds.split!r}; available: {sorted(loaded)}"
                        )
                    dataset = loaded[ds.split]
                else:
                    dataset = loaded
            else:
                dataset = load_dataset(
                    ds.path,
                    name=ds.name,
                    split=ds.split,
                    streaming=ds.streaming,
                    revision=ds.revision,
                )
        except DatasetValidationError:
            raise
        except Exception as error:
            raise DatasetValidationError(
                f"Failed to load dataset {ds.path!r} (name={ds.name!r}, split={ds.split!r}): "
                f"{error}"
            ) from error

        self._validate_target_schema(dataset)
        audio_feature = dataset.features[self._audio_column]
        if isinstance(audio_feature, Audio) and audio_feature.decode:
            dataset = dataset.cast_column(
                self._audio_column,
                Audio(sampling_rate=audio_feature.sampling_rate, decode=False),
            )
        if ds.shuffle:
            dataset = dataset.shuffle(seed=ds.seed)
        if isinstance(dataset, IterableDataset):
            dataset = Dataset.from_list(list(dataset.take(ds.samples)))
        else:
            dataset = dataset.select(range(min(ds.samples, len(dataset))))
        return dataset

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Leave target alignment to the scalar/multi-label decoder."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run one forward per selected row and compute task-appropriate metrics."""
        logits: list[np.ndarray] = []
        targets: list[Any] = []
        rejected_by_reason: dict[str, int] = {}
        adapter = cast("_AudioModelAdapter", self.pipe)
        for row in self.data:
            target = self._target_for_row(row)
            try:
                prediction = adapter.predict_logits(row[self._audio_column])
            except (TypeError, ValueError, RuntimeError) as error:
                reason = type(error).__name__
                rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
                continue
            logits.append(prediction)
            targets.append(target)
        if not logits:
            raise DatasetValidationError("No selected audio samples were successfully processed.")
        scores = np.stack(logits)
        if scores.shape[1] != len(self._model_id2label):
            raise DatasetValidationError(
                f"model returned {scores.shape[1]} classes but config defines "
                f"{len(self._model_id2label)} labels."
            )
        if self._target_kind == "multi-label":
            metrics = self._multi_label_metrics(scores, targets)
        else:
            metrics = self._single_label_metrics(scores, targets)
        selected = len(self.data)
        metrics.update(
            requested_samples=self.config.dataset.samples,
            selected_samples=selected,
            processed_samples=len(targets),
            rejected_samples=selected - len(targets),
            rejected_by_reason=dict(sorted(rejected_by_reason.items())),
        )
        return metrics

    def _validate_target_schema(self, dataset: Any) -> None:
        from datasets import ClassLabel, Sequence, Value

        missing = [
            name
            for name in (self._audio_column, self._label_column)
            if name not in dataset.column_names
        ]
        if missing:
            raise DatasetValidationError(
                f"missing required column(s) {missing}; dataset has {sorted(dataset.column_names)}"
            )
        feature = dataset.features[self._label_column]
        if isinstance(feature, ClassLabel):
            self._target_kind = "single-label"
        elif isinstance(feature, Sequence) and isinstance(feature.feature, (ClassLabel, Value)):
            self._target_kind = "multi-label"
        else:
            raise DatasetValidationError(
                f"Column {self._label_column!r} must be ClassLabel or a sequence of "
                f"ClassLabel/string values; got {feature!r}."
            )
        self._label_feature = feature

    def _target_for_row(self, row: dict[str, Any]) -> Any:
        from datasets import ClassLabel

        raw_target = row[self._label_column]
        if self._target_kind == "single-label":
            assert isinstance(self._label_feature, ClassLabel)
            return self._resolve_label(self._label_feature.int2str(int(raw_target)))

        values = list(raw_target)
        feature = self._label_feature.feature
        decoded = (
            [feature.int2str(int(value)) for value in values]
            if isinstance(feature, ClassLabel)
            else values
        )
        parallel_names = row.get(self._label_name_column)
        if parallel_names is not None and len(parallel_names) != len(decoded):
            raise DatasetValidationError(
                f"Columns {self._label_column!r} and {self._label_name_column!r} "
                "must contain the same number of labels."
            )
        resolved: list[int] = []
        for index, value in enumerate(decoded):
            fallback_name = parallel_names[index] if parallel_names is not None else None
            resolved.append(self._resolve_label(str(value), fallback_name=fallback_name))
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
                "authoritative label mapping or parallel exact label-name column."
            )
        if model_id not in self._model_id2label:
            raise DatasetValidationError(
                f"Label mapping target {model_id} is absent from model.config.id2label."
            )
        return model_id

    def _single_label_metrics(
        self,
        logits: np.ndarray,
        targets: list[int],
    ) -> dict[str, Any]:
        from .metrics import ClassificationMetric

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

    def _multi_label_metrics(
        self,
        logits: np.ndarray,
        targets: list[list[int]],
    ) -> dict[str, Any]:
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
