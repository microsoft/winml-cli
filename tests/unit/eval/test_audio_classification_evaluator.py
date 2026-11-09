# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
import runpy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
from click.testing import CliRunner
from datasets import (
    Audio,
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    IterableDataset,
    Sequence,
    Value,
)
from onnx import TensorProto, helper, save
from transformers import Wav2Vec2Config

from winml.modelkit.commands.eval import eval as eval_command
from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.audio_classification_evaluator import (
    WinMLAudioClassificationEvaluator,
    _AudioModelAdapter,
)
from winml.modelkit.utils.eval_utils import DatasetValidationError


class _IdentityFeatureExtractor:
    sampling_rate = 16_000
    padding_value = 0.0

    def __call__(self, waveform, *, sampling_rate, return_tensors, **kwargs):
        assert sampling_rate == self.sampling_rate
        assert return_tensors in {"np", "pt"}
        values = np.asarray(waveform, dtype=np.float32)
        if kwargs.get("padding") == "max_length":
            max_length = kwargs["max_length"]
            values = values[:max_length]
            values = np.pad(values, (0, max_length - values.size))
        tensor = values[None, :]
        return {
            "input_values": torch.from_numpy(tensor) if return_tensors == "pt" else tensor,
        }


class _SignClassifier:
    io_config: ClassVar = {"input_names": ["input_values"], "input_shapes": [[1, 8]]}
    config = SimpleNamespace(
        label2id={"cat": 0, "dog": 1, "other": 2},
        id2label={0: "cat", 1: "dog", 2: "other"},
    )

    def __call__(self, **kwargs):
        values = kwargs["input_values"].cpu().numpy()
        score = float(values.mean())
        return {"logits": torch.tensor([[score, -score, -10.0]])}


class _TwoInputFeatureExtractor:
    sampling_rate = 16_000

    def __init__(self):
        self.kwargs = None

    def __call__(self, waveform, *, sampling_rate, return_tensors, **kwargs):
        self.kwargs = kwargs
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        length = kwargs.get("max_length", len(waveform))
        values = np.asarray(waveform, dtype=np.float32)[:length]
        values = np.pad(values, (0, length - values.size))
        return {
            "input_values": torch.from_numpy(values[None, :]),
            "attention_mask": torch.ones((1, length), dtype=torch.int64),
            "extra": torch.zeros((1, length), dtype=torch.int64),
        }


class _TwoInputClassifier:
    io_config: ClassVar = {
        "input_names": ["input_values", "attention_mask"],
        "input_shapes": [[1, 8], [1, 8]],
    }
    config = _SignClassifier.config

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled()
        return {"logits": torch.tensor([[5.0, 0.0, -5.0]])}


class _SpectrogramFeatureExtractor:
    sampling_rate = 16_000

    def __call__(self, waveform, *, sampling_rate, return_tensors):
        assert waveform.size > 0
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        return {"input_values": torch.ones((1, 4, 3), dtype=torch.float32)}


class _SpectrogramClassifier:
    io_config: ClassVar = {"input_names": ["input_values"], "input_shapes": [[1, 4, 3]]}
    config = _SignClassifier.config

    def __call__(self, **kwargs):
        assert kwargs["input_values"].shape == (1, 4, 3)
        return {"logits": torch.tensor([[1.0, 0.0, -1.0]])}


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


class _EncodingMustNotRunAudio(Audio):
    def encode_example(self, value):
        raise AssertionError("bounded selection must not re-encode raw audio")


class _RawAudioIterableDataset(IterableDataset):
    def __init__(self, rows, features):
        self._rows = rows
        self._raw_features = features

    @property
    def features(self):
        return self._raw_features

    @property
    def column_names(self):
        return list(self._raw_features)

    def __iter__(self):
        return iter(self._rows)


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
    save(model, path)


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
        one_frame_channels_first = np.array([[1.0], [0.5]], dtype=np.float32)

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
        np.testing.assert_array_equal(
            WinMLAudioClassificationEvaluator._to_mono(one_frame_channels_first),
            np.array([0.75], dtype=np.float32),
        )

class TestAudioLabelAlignmentAndSampling:
    def test_saved_dataset_dict_selects_requested_split(self, tmp_path):
        train = _audio_dataset(
            [{"audio": {"array": [-1.0] * 8, "sampling_rate": 16_000}, "label": 1}]
        )
        test = _audio_dataset(
            [{"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0}]
        )
        dataset_path = tmp_path / "audio-dataset"
        DatasetDict({"train": train, "test": test}).save_to_disk(dataset_path)
        config = _config(samples=1, label_mapping={"cat": 0})
        config.dataset.path = str(dataset_path)

        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityFeatureExtractor(),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())

        assert evaluator._eligible_count == 1
        assert [sample.model_id for sample in evaluator.data] == [0]

    def test_saved_dataset_dict_missing_split_is_clear(self, tmp_path):
        dataset_path = tmp_path / "audio-dataset"
        DatasetDict(
            {
                "train": _audio_dataset(
                    [
                        {
                            "audio": {"array": [1.0] * 8, "sampling_rate": 16_000},
                            "label": 0,
                        }
                    ]
                )
            }
        ).save_to_disk(dataset_path)
        config = _config(samples=1)
        config.dataset.path = str(dataset_path)

        with pytest.raises(
            DatasetValidationError,
            match=r"Dataset split 'test' was not found; available splits: \['train'\]",
        ):
            WinMLAudioClassificationEvaluator(config, _SignClassifier())

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
    def test_baseline_builds_explicit_shared_audio_length_contract(self):
        baseline = runpy.run_path(
            str(Path(__file__).parents[3] / "scripts/e2e_eval/run_pytorch_baseline.py")
        )

        config = baseline["_build_dataset_config"](
            {"dataset": "example/audio", "columns_mapping": {"input_column": "audio"}},
            10,
            audio_input_length=16_000,
        )

        assert config.audio_input_length == 16_000
        assert config.columns_mapping == {"input_column": "audio"}

    def test_pytorch_baseline_latency_uses_selected_raw_audio_and_shared_pipe(self):
        baseline = runpy.run_path(
            str(Path(__file__).parents[3] / "scripts/e2e_eval/run_pytorch_baseline.py")
        )
        measure_latency = baseline["_measure_pytorch_latency"]

        dataset = _audio_dataset(
            [{"audio": {"array": [1.0] * 8, "sampling_rate": 16_000}, "label": 0}]
        )
        config = _config(samples=1, label_mapping={"cat": 0}, shuffle=False)
        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_IdentityFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _SignClassifier())
            latency = measure_latency(evaluator, warmup=0, iterations=1)

        assert evaluator.pipe.model is evaluator.model
        assert latency["iterations"] == 1

    def test_pre_normalized_waveform_must_be_one_dimensional(self):
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityFeatureExtractor(),
        ):
            adapter = _AudioModelAdapter(_config(samples=1), _SignClassifier())
            with pytest.raises(ValueError, match="must be 1D"):
                adapter(np.ones((1, 8), dtype=np.float32))

    def test_rank_three_spectrogram_runs_through_adapter(self):
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_SpectrogramFeatureExtractor(),
        ):
            logits = _AudioModelAdapter(_config(samples=1), _SpectrogramClassifier())(
                {"array": np.ones(16, dtype=np.float32), "sampling_rate": 16_000}
            )

        np.testing.assert_array_equal(logits, [1.0, 0.0, -1.0])

    def test_overlength_waveform_uses_one_padded_truncated_forward(self):
        model = _TwoInputClassifier()
        extractor = _TwoInputFeatureExtractor()
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=extractor,
        ):
            adapter = _AudioModelAdapter(_config(samples=1), model)
            logits = adapter([1.0] * 24)

        assert adapter.model is model
        assert len(model.calls) == 1
        assert set(model.calls[0]) == {"input_values", "attention_mask"}
        assert model.calls[0]["input_values"].shape == (1, 8)
        assert model.calls[0]["attention_mask"].shape == (1, 8)
        assert extractor.kwargs == {
            "padding": "max_length",
            "truncation": True,
            "max_length": 8,
        }
        np.testing.assert_array_equal(logits, [5.0, 0.0, -5.0])

    def test_explicit_input_length_applies_to_native_model(self):
        model = _TwoInputClassifier()
        model.io_config = {}
        extractor = _TwoInputFeatureExtractor()
        config = _config(samples=1)
        config.runtime = "pytorch"
        config.dataset.audio_input_length = 8
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=extractor,
        ):
            logits = _AudioModelAdapter(config, model)([1.0] * 24)

        assert model.calls[0]["input_values"].shape == (1, 8)
        assert extractor.kwargs["max_length"] == 8
        np.testing.assert_array_equal(logits, [5.0, 0.0, -5.0])

    def test_dynamic_waveform_dimension_preserves_full_input(self):
        model = _TwoInputClassifier()
        model.io_config = {
            "input_names": ["input_values", "attention_mask"],
            "input_shapes": [[1, "samples"], [1, "samples"]],
        }
        extractor = _TwoInputFeatureExtractor()
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=extractor,
        ):
            _AudioModelAdapter(_config(samples=1), model)([1.0] * 24)

        assert model.calls[0]["input_values"].shape == (1, 24)
        assert extractor.kwargs == {}

    def test_single_custom_onnx_input_name_uses_semantic_waveform(self):
        class CustomInputModel:
            io_config: ClassVar = {"input_names": ["waveform"], "input_shapes": [[1, 8]]}
            config = _SignClassifier.config

            def __init__(self):
                self.inputs = None

            def __call__(self, **kwargs):
                self.inputs = kwargs
                return {"scores": torch.tensor([[1.0, 0.0, -1.0]])}

        model = CustomInputModel()
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=_IdentityFeatureExtractor(),
        ):
            logits = _AudioModelAdapter(_config(samples=1), model)([1.0] * 8)

        assert set(model.inputs) == {"waveform"}
        np.testing.assert_array_equal(logits, [1.0, 0.0, -1.0])

    def test_native_model_without_io_config_receives_all_extractor_outputs(self):
        class _NativeModel:
            config = _SignClassifier.config

            def __init__(self):
                self.inputs = None

            def __call__(self, **kwargs):
                self.inputs = kwargs
                return SimpleNamespace(logits=torch.tensor([[1.0, 0.0, -1.0]]))

        model = _NativeModel()
        extractor = _TwoInputFeatureExtractor()
        with patch(
            "transformers.AutoFeatureExtractor.from_pretrained",
            return_value=extractor,
        ):
            logits = _AudioModelAdapter(_config(samples=1), model)([1.0] * 8)

        assert set(model.inputs) == {"input_values", "attention_mask", "extra"}
        np.testing.assert_array_equal(logits, [1.0, 0.0, -1.0])

    @pytest.mark.parametrize(
        ("label_feature", "labels", "names", "label_mapping"),
        [
            (
                Sequence(ClassLabel(names=["cat", "dog", "other"])),
                [[0, 1], [1, 2]],
                None,
                None,
            ),
            (
                Sequence(Value("string")),
                [["/cat", "/dog"], ["/dog", "/other"]],
                [["cat", "dog"], ["dog", "other"]],
                None,
            ),
        ],
        ids=["sequence-classlabel", "sequence-value-with-label-names"],
    )
    def test_multi_label_targets_have_finite_sigmoid_average_precision(
        self,
        label_feature,
        labels,
        names,
        label_mapping,
    ):
        feature_map = {
            "audio": {
                "array": Sequence(Value("float32")),
                "sampling_rate": Value("int32"),
            },
            "labels": label_feature,
        }
        rows = [
            {
                "audio": {"array": [1.0] * 8, "sampling_rate": 16_000},
                "labels": labels[0],
            },
            {
                "audio": {"array": [-1.0] * 8, "sampling_rate": 16_000},
                "labels": labels[1],
            },
        ]
        columns_mapping = {"label_column": "labels"}
        if names is not None:
            feature_map["names"] = Sequence(Value("string"))
            rows[0]["names"] = names[0]
            rows[1]["names"] = names[1]
            columns_mapping["label_name_column"] = "names"
        dataset = Dataset.from_list(rows, features=Features(feature_map))
        config = WinMLEvaluationConfig(
            model_id="example/audio-classifier",
            task="audio-classification",
            dataset=DatasetConfig(
                path="example/audio-dataset",
                split="test",
                samples=2,
                shuffle=False,
                columns_mapping=columns_mapping,
                label_mapping=label_mapping,
            ),
        )
        model = _TwoInputClassifier()

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_TwoInputFeatureExtractor(),
            ),
        ):
            metrics = WinMLAudioClassificationEvaluator(config, model).compute()

        assert len(model.calls) == 2
        assert np.isfinite(metrics["sample_average_precision"])
        assert np.isfinite(metrics["micro_average_precision"])

    def test_streaming_selection_preserves_raw_audio_without_reencoding(self):
        encoded = BytesIO()
        sf.write(encoded, np.ones(8, dtype=np.float32), 16_000, format="WAV")
        raw_audio = {"bytes": encoded.getvalue(), "path": None}
        dataset = _RawAudioIterableDataset(
            [{"audio": raw_audio, "labels": ["cat"]}],
            Features(
                {
                    "audio": _EncodingMustNotRunAudio(decode=False),
                    "labels": Sequence(Value("string")),
                }
            ),
        )
        config = WinMLEvaluationConfig(
            model_id="example/audio-classifier",
            task="audio-classification",
            dataset=DatasetConfig(
                path="example/audio-dataset",
                split="test",
                samples=1,
                shuffle=False,
                streaming=True,
                columns_mapping={"label_column": "labels"},
            ),
        )

        with (
            patch("datasets.load_dataset", return_value=dataset),
            patch(
                "transformers.AutoFeatureExtractor.from_pretrained",
                return_value=_TwoInputFeatureExtractor(),
            ),
        ):
            evaluator = WinMLAudioClassificationEvaluator(config, _TwoInputClassifier())
            assert evaluator.data[0]["audio"]["bytes"] == raw_audio["bytes"]
            metrics = evaluator.compute()

        assert metrics["processed_samples"] == 1
        assert np.isfinite(metrics["sample_average_precision"])

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
            "represented_classes": 2,
            "total_classes": 3,
            "class_coverage": 2 / 3,
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
        assert metrics["represented_classes"] == 1
        assert metrics["total_classes"] == 3
        assert metrics["class_coverage"] == pytest.approx(1 / 3)
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
            patch("winml.modelkit.loader.load_hf_config", return_value=hf_config),
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
