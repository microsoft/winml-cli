# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import soundfile as sf
import torch
from click.testing import CliRunner
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from onnx import TensorProto, helper
from transformers import Wav2Vec2Config

from winml.modelkit.commands.eval import eval as eval_command
from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.audio_classification_evaluator import (
    WinMLAudioClassificationEvaluator,
)
from winml.modelkit.utils.eval_utils import DatasetValidationError


class _IdentityFeatureExtractor:
    sampling_rate = 16_000
    padding_value = 0.0

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "np"
        return {"input_values": np.asarray(waveform, dtype=np.float32)[None, :]}


class _SignClassifier:
    io_config: ClassVar = {"input_names": ["input_values"], "input_shapes": [[1, 8]]}
    config = SimpleNamespace(
        label2id={"cat": 0, "dog": 1, "other": 2},
        id2label={0: "cat", 1: "dog", 2: "other"},
    )

    def __call__(self, **kwargs):
        values = kwargs["input_values"].numpy()
        score = float(values.mean())
        return {"logits": torch.tensor([[score, -score, -10.0]])}


class _SpectrogramFeatureExtractor:
    sampling_rate = 16_000

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert waveform.size > 0
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "np"
        return {"input_values": np.ones((1, 4, 3), dtype=np.float32)}


class _SpectrogramClassifier:
    io_config: ClassVar = {"input_names": ["input_values"], "input_shapes": [[1, 4, 3]]}


class _CountingStreamingDataset:
    def __init__(self, rows):
        self.column_names = ["audio", "label"]
        self.features = {
            "audio": Value("string"),
            "label": ClassLabel(names=["cat", "dog", "en_us"]),
        }
        self._rows = rows
        self.rows_yielded = 0

    def __iter__(self):
        for row in self._rows:
            self.rows_yielded += 1
            yield row


def _audio_dataset(rows):
    features = Features(
        {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "label": ClassLabel(names=["cat", "dog", "en_us"]),
        }
    )
    return Dataset.from_list(rows, features=features)


def _config(samples=4, label_mapping=None, *, streaming=False, shuffle=True):
    return WinMLEvaluationConfig(
        model_id="example/audio-classifier",
        task="audio-classification",
        dataset=DatasetConfig(
            path="example/audio-dataset",
            split="test",
            samples=samples,
            shuffle=shuffle,
            seed=42,
            label_mapping=label_mapping,
            streaming=streaming,
        ),
    )


def _save_sign_classifier(path):
    input_info = helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, 8])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input_values"], ["score"], axes=[1], keepdims=1),
            helper.make_node("Neg", ["score"], ["negative_score"]),
            helper.make_node("Concat", ["score", "negative_score"], ["logits"], axis=1),
        ],
        "sign_classifier",
        [input_info],
        [output_info],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.save(model, path)


class TestAudioPreprocessing:
    def test_decodes_encoded_audio_bytes_without_torchcodec(self):
        encoded = BytesIO()
        sf.write(encoded, np.array([0.25, -0.5], dtype=np.float32), 16_000, format="WAV")

        waveform, sampling_rate = WinMLAudioClassificationEvaluator._decode_audio(
            {"bytes": encoded.getvalue(), "path": None}
        )

        assert sampling_rate == 16_000
        np.testing.assert_allclose(waveform, [0.25, -0.5], atol=4e-5)

    def test_decodes_path_audio_through_dataset_opener(self, tmp_path):
        audio_path = tmp_path / "clip.wav"
        frames_first = np.array([[0.25, 0.5], [-0.25, -0.5]], dtype=np.float32)
        sf.write(audio_path, frames_first, 16_000)

        waveform, sampling_rate = WinMLAudioClassificationEvaluator._decode_audio(
            {"bytes": None, "path": str(audio_path)}
        )

        assert sampling_rate == 16_000
        assert waveform.shape == (2, 2)
        np.testing.assert_allclose(waveform, frames_first.T, atol=4e-5)

    def test_mono_and_both_stereo_layouts(self):
        mono = np.array([1.0, 3.0], dtype=np.float32)
        channels_first = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
        channels_last = channels_first.T

        np.testing.assert_array_equal(
            WinMLAudioClassificationEvaluator._to_mono(mono),
            mono,
        )
        np.testing.assert_array_equal(
            WinMLAudioClassificationEvaluator._to_mono(channels_first),
            np.array([2.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            WinMLAudioClassificationEvaluator._to_mono(channels_last),
            np.array([2.0, 4.0], dtype=np.float32),
        )

    def test_exact_short_and_multi_window_padding(self):
        exact = WinMLAudioClassificationEvaluator._window_waveform(
            np.arange(4, dtype=np.float32), 4
        )
        short = WinMLAudioClassificationEvaluator._window_waveform(
            np.array([1.0, 2.0], dtype=np.float32), 4, -1.0
        )
        multi = WinMLAudioClassificationEvaluator._window_waveform(
            np.arange(10, dtype=np.float32), 4
        )

        assert exact.shape == (1, 4)
        np.testing.assert_array_equal(short, [[1.0, 2.0, -1.0, -1.0]])
        assert multi.shape == (3, 4)
        np.testing.assert_array_equal(multi[-1], [8.0, 9.0, 0.0, 0.0])

    def test_empty_audio_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            WinMLAudioClassificationEvaluator._window_waveform(np.array([], dtype=np.float32), 8)

    def test_missing_feature_extractor_input_is_rejected_clearly(self):
        evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
        evaluator.pipe = lambda *_args, **_kwargs: {"input_features": np.ones((1, 8))}
        evaluator.model = _SignClassifier()

        with pytest.raises(ValueError, match="must contain 'input_values'"):
            evaluator._preprocess_audio(
                {"array": np.ones(8, dtype=np.float32), "sampling_rate": 16_000}
            )

    def test_preserves_rank_three_spectrogram_features(self):
        evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
        evaluator.pipe = _SpectrogramFeatureExtractor()
        evaluator.model = _SpectrogramClassifier()

        features = evaluator._preprocess_audio(
            {"array": np.ones(16, dtype=np.float32), "sampling_rate": 16_000}
        )

        assert features.shape == (1, 4, 3)
        np.testing.assert_array_equal(features, np.ones((1, 4, 3), dtype=np.float32))

    def test_rejects_spectrogram_shape_mismatch(self):
        evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
        evaluator.pipe = _SpectrogramFeatureExtractor()
        evaluator.model = _SpectrogramClassifier()
        evaluator.model.io_config = {
            "input_names": ["input_values"],
            "input_shapes": [[1, 5, 3]],
        }

        with pytest.raises(ValueError, match="expected"):
            evaluator._preprocess_audio(
                {"array": np.ones(16, dtype=np.float32), "sampling_rate": 16_000}
            )

    def test_resamples_to_feature_extractor_rate_and_fixed_model_length(self):
        evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
        evaluator.pipe = _IdentityFeatureExtractor()
        evaluator.model = _SignClassifier()
        waveform = np.sin(np.linspace(0, 4 * np.pi, 4_000, dtype=np.float32))

        windows = evaluator._preprocess_audio({"array": waveform, "sampling_rate": 8_000})

        # 0.5 seconds at 8 kHz becomes 0.5 seconds at 16 kHz, then 1000
        # static 8-sample windows.
        assert windows.shape == (1_000, 8)
        assert windows.dtype == np.float32


class TestAudioLabelAlignmentAndSampling:
    def test_filters_authoritative_labels_before_stratified_sampling(self):
        rows = [
            {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [2.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 1},
            {"audio": {"array": [-2.0] * 8, "sampling_rate": 16_000}, "label": 1},
            *[
                {
                    "audio": {"array": [9.0] * 8, "sampling_rate": 16_000},
                    "label": 2,
                }
                for _ in range(20)
            ],
        ]
        dataset = _audio_dataset(rows)
        config = _config(samples=4, label_mapping={"cat": 0, "dog": 1})

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())

        assert evaluator._eligible_count == 4
        assert [sample.model_id for sample in evaluator.data].count(0) == 2
        assert [sample.model_id for sample in evaluator.data].count(1) == 2
        assert all(sample.model_id != 2 for sample in evaluator.data)

    def test_does_not_infer_cross_region_or_other_near_match(self):
        dataset = _audio_dataset(
            [
                {
                    "audio": {"array": [1.0] * 8, "sampling_rate": 16_000},
                    "label": 2,
                }
            ]
        )
        model = _SignClassifier()
        model.config = SimpleNamespace(
            label2id={"en-IN": 0},
            id2label={0: "en-IN"},
        )

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
            pytest.raises(DatasetValidationError, match="no exact overlap"),
        ):
            WinMLAudioClassificationEvaluator(_config(samples=1), model)

    def test_zero_overlap_mapping_fails_closed(self):
        dataset = _audio_dataset(
            [
                {
                    "audio": {"array": [1.0] * 8, "sampling_rate": 16_000},
                    "label": 0,
                }
            ]
        )
        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
            pytest.raises(DatasetValidationError, match="no exact overlap"),
        ):
            WinMLAudioClassificationEvaluator(
                _config(samples=1, label_mapping={"missing": 0}),
                _SignClassifier(),
            )

    def test_no_shuffle_streaming_stops_after_balanced_quota(self):
        rows = [{"audio": f"cat-{index}", "label": 0} for index in range(3)] + [
            {"audio": f"dog-{index}", "label": 1} for index in range(20)
        ]
        dataset = _CountingStreamingDataset(rows)
        config = _config(
            samples=6,
            label_mapping={"cat": 0, "dog": 1},
            streaming=True,
            shuffle=False,
        )

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())

        assert dataset.rows_yielded == 6
        assert dataset.rows_yielded < len(rows)
        assert [sample.model_id for sample in evaluator.data] == [0, 1, 0, 1, 0, 1]
        assert [sample.row["audio"] for sample in evaluator.data] == [
            "cat-0",
            "dog-0",
            "cat-1",
            "dog-1",
            "cat-2",
            "dog-2",
        ]

    @pytest.mark.parametrize(
        ("labels", "expected_counts"),
        [
            ([0, 0, 0, 1, 1, 1], {0: 2, 1: 2}),
            ([0, 1, 1, 2, 2, 2], {0: 1, 1: 2, 2: 2}),
        ],
        ids=["absent-authoritative-class", "uneven-class"],
    )
    def test_no_shuffle_streaming_exhausts_and_reports_short_quota(
        self,
        labels,
        expected_counts,
    ):
        rows = [{"audio": str(index), "label": label} for index, label in enumerate(labels)]
        dataset = _CountingStreamingDataset(rows)
        config = _config(
            samples=6,
            label_mapping={"cat": 0, "dog": 1, "en_us": 2},
            streaming=True,
            shuffle=False,
        )

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())

        actual_counts = {
            model_id: [sample.model_id for sample in evaluator.data].count(model_id)
            for model_id in expected_counts
        }
        assert dataset.rows_yielded == len(rows)
        assert actual_counts == expected_counts
        assert evaluator._selected_count == sum(expected_counts.values())
        assert evaluator._selected_count < config.dataset.samples

    def test_shuffle_streaming_still_scans_complete_stream(self):
        rows = [{"audio": f"cat-{index}", "label": 0} for index in range(20)] + [
            {"audio": f"dog-{index}", "label": 1} for index in range(20)
        ]
        dataset = _CountingStreamingDataset(rows)
        config = _config(
            samples=6,
            label_mapping={"cat": 0, "dog": 1},
            streaming=True,
            shuffle=True,
        )

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())

        assert dataset.rows_yielded == len(rows)
        assert evaluator._eligible_count == len(rows)
        assert [sample.model_id for sample in evaluator.data].count(0) == 3
        assert [sample.model_id for sample in evaluator.data].count(1) == 3


class TestAudioPredictionAndMetrics:
    def test_mean_logit_aggregation_produces_one_utterance_prediction(self):
        class _QueuedModel(_SignClassifier):
            def __init__(self):
                self.outputs = [
                    torch.tensor([[10.0, 0.0, -1.0]]),
                    torch.tensor([[0.0, 12.0, -1.0]]),
                ]

            def __call__(self, **_kwargs):
                return {"logits": self.outputs.pop(0)}

        evaluator = WinMLAudioClassificationEvaluator.__new__(WinMLAudioClassificationEvaluator)
        evaluator.model = _QueuedModel()
        evaluator.pipe = _IdentityFeatureExtractor()
        evaluator.config = _config(samples=1, label_mapping={"dog": 1})
        evaluator._audio_col = "audio"
        evaluator._label_col = "label"
        evaluator.data = [
            SimpleNamespace(
                model_id=1,
                row={
                    "audio": {"array": [1.0] * 16, "sampling_rate": 16_000},
                    "label": 1,
                },
                index=None,
            )
        ]
        evaluator._dataset_id_to_model_id = {1: 1}
        evaluator._eligible_model_labels = ["dog"]
        evaluator._eligible_count = 1
        evaluator._selected_count = 1

        metrics = evaluator.compute()

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["processed_samples"] == 1

    def test_end_to_end_fixed_shape_evaluator_reports_accounting(self):
        rows = [
            {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [2.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 1},
            {"audio": {"array": [-2.0] * 8, "sampling_rate": 16_000}, "label": 1},
        ]
        dataset = _audio_dataset(rows)
        config = _config(samples=4, label_mapping={"cat": 0, "dog": 1})

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            metrics = WinMLAudioClassificationEvaluator(config, _SignClassifier()).compute()

        assert metrics == {
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "requested_samples": 4,
            "eligible_samples": 4,
            "selected_samples": 4,
            "processed_samples": 4,
            "rejected_samples": 0,
            "rejected_by_reason": {},
            "per_label_processed": {"cat": 2, "dog": 2},
            "confusion_matrix": {"cat": {"cat": 2}, "dog": {"dog": 2}},
        }

    def test_rejected_audio_is_accounted_after_selection(self):
        rows = [
            {"audio": {"array": [], "sampling_rate": 16_000}, "label": 0},
            *[
                {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0}
                for _ in range(5)
            ],
            *[
                {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 1}
                for _ in range(6)
            ],
        ]
        dataset = _audio_dataset(rows)
        config = _config(samples=12, label_mapping={"cat": 0, "dog": 1})

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            metrics = WinMLAudioClassificationEvaluator(config, _SignClassifier()).compute()

        assert metrics["eligible_samples"] == 12
        assert metrics["selected_samples"] == 12
        assert metrics["processed_samples"] == 11
        assert metrics["rejected_samples"] == 1
        assert metrics["rejected_by_reason"] == {"ValueError": 1}
        assert metrics["per_label_processed"] == {"cat": 5, "dog": 6}

    def test_prediction_outside_reference_labels_is_an_accuracy_error(self):
        class _OtherClassifier(_SignClassifier):
            def __call__(self, **_kwargs):
                return {"logits": torch.tensor([[0.0, 0.0, 10.0]])}

        dataset = _audio_dataset(
            [
                {
                    "audio": {"array": [1.0] * 8, "sampling_rate": 16_000},
                    "label": 0,
                }
            ]
        )
        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            metrics = WinMLAudioClassificationEvaluator(
                _config(samples=1, label_mapping={"cat": 0}),
                _OtherClassifier(),
            ).compute()

        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0
        assert metrics["confusion_matrix"] == {"cat": {"other": 1}}

    def test_cli_with_saved_dataset_and_fixed_shape_onnx_reports_metrics(self, tmp_path):
        dataset_path = tmp_path / "dataset"
        model_path = tmp_path / "classifier.onnx"
        output_path = tmp_path / "result.json"
        label_mapping_path = tmp_path / "labels.json"
        rows = [
            {"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [2.0] * 8, "sampling_rate": 16_000}, "label": 0},
            {"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 1},
            {"audio": {"array": [-2.0] * 8, "sampling_rate": 16_000}, "label": 1},
        ]
        _audio_dataset(rows).save_to_disk(dataset_path)
        _save_sign_classifier(model_path)
        label_mapping_path.write_text(json.dumps({"cat": 0, "dog": 1}), encoding="utf-8")
        hf_config = Wav2Vec2Config(
            id2label={0: "cat", 1: "dog"},
            label2id={"cat": 0, "dog": 1},
        )

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=hf_config),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            result = CliRunner().invoke(
                eval_command,
                [
                    "-m",
                    str(model_path),
                    "--model-id",
                    "example/audio-classifier",
                    "--task",
                    "audio-classification",
                    "--dataset",
                    str(dataset_path),
                    "--split",
                    "train",
                    "--samples",
                    "4",
                    "--device",
                    "cpu",
                    "--label-mapping",
                    str(label_mapping_path),
                    "--output",
                    str(output_path),
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        metrics = json.loads(output_path.read_text(encoding="utf-8"))["metrics"]
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["requested_samples"] == 4
        assert metrics["eligible_samples"] == 4
        assert metrics["processed_samples"] == 4
        assert metrics["rejected_samples"] == 0


class TestAudioRegistryCompatibility:
    def test_audio_schema_and_evaluator_registered_without_changing_sibling(self):
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS, get_evaluator_class
        from winml.modelkit.eval.text_classification_evaluator import (
            WinMLTextClassificationEvaluator,
        )
        from winml.modelkit.utils.eval_utils import TASK_SCHEMAS

        audio_schema = TASK_SCHEMAS["audio-classification"]
        assert [item.default for item in audio_schema.columns] == ["audio", "label"]
        assert "audio-classification" not in _DEFAULT_DATASETS
        assert get_evaluator_class(_config()).__name__ == "WinMLAudioClassificationEvaluator"
        assert (
            get_evaluator_class(WinMLEvaluationConfig(task="text-classification"))
            is WinMLTextClassificationEvaluator
        )

        schema_result = CliRunner().invoke(
            eval_command,
            ["--schema", "--task", "audio-classification"],
            obj={},
        )
        assert schema_result.exit_code == 0
        assert "default: audio" in schema_result.output
        assert "default: label" in schema_result.output
