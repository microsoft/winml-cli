# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from winml.modelkit.eval.ctc_asr_evaluator import (
    WinMLCTCASREvaluator,
    _configure_processor_language,
    _corpus_error_rate,
    _decode_audio,
    _is_ctc_config,
    _load_ctc_processor,
    _normalize_transcript,
    _RejectedSampleError,
    _resample_audio,
)
from winml.modelkit.utils.eval_utils import TASK_SCHEMAS, DatasetValidationError


class _Tokenizer:
    pad_token_id = 0

    def __init__(self, target_lang: str | None = None, vocab_size: int = 3) -> None:
        self.target_lang = target_lang
        self._vocab = {str(index): index for index in range(vocab_size)}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def set_target_lang(self, target_lang: str) -> None:
        self.target_lang = target_lang


class _Processor:
    def __init__(self, *, target_lang: str | None = None, vocab_size: int = 3) -> None:
        self.tokenizer = _Tokenizer(target_lang, vocab_size)
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.decoded_ids: list[int] = []

    def __call__(self, waveform, **_kwargs):
        return {"input_values": np.asarray(waveform, dtype=np.float32)[None, :]}

    def batch_decode(self, sequences):
        self.decoded_ids = list(sequences[0])
        collapsed: list[int] = []
        previous = None
        for token_id in self.decoded_ids:
            if token_id != 0 and token_id != previous:
                collapsed.append(token_id)
            previous = token_id
        return [" ".join(str(token_id) for token_id in collapsed)]


def _wav_bytes(samples: np.ndarray, sampling_rate: int = 16_000) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, samples, sampling_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _evaluator(*, input_shape: list[object], processor: _Processor | None = None):
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    evaluator = WinMLCTCASREvaluator.__new__(WinMLCTCASREvaluator)
    evaluator.processor = processor or _Processor()
    evaluator.sampling_rate = 16_000
    evaluator.blank_token_id = 0
    evaluator.tokenizer_vocab_size = 3
    evaluator.target_lang = evaluator.processor.tokenizer.target_lang
    evaluator._audio_column = "audio"
    evaluator._transcription_column = "transcription"
    evaluator.config = WinMLEvaluationConfig(
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset", samples=2, shuffle=False),
    )
    evaluator.model = MagicMock()
    evaluator.model.io_config = {
        "input_names": ["input_values"],
        "input_shapes": [input_shape],
    }
    return evaluator


def test_asr_registry_schema_and_generic_model_route() -> None:
    from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class
    from winml.modelkit.inference import TASK_REGISTRY
    from winml.modelkit.models.winml import get_winml_class
    from winml.modelkit.models.winml.base import WinMLModelForGenericTask

    config = WinMLEvaluationConfig(task="automatic-speech-recognition")
    assert get_evaluator_class(config) is WinMLCTCASREvaluator
    assert "automatic-speech-recognition" in TASK_SCHEMAS
    assert get_winml_class("wav2vec2", "automatic-speech-recognition") is WinMLModelForGenericTask
    assert TASK_REGISTRY["automatic-speech-recognition"].user_inputs[0].type == "audio"


def test_asr_default_dataset_is_pinned_and_bounded() -> None:
    from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

    default = _DEFAULT_DATASETS["automatic-speech-recognition"]
    assert default["path"] == "google/fleurs"
    assert default["name"] == "en_us"
    assert default["revision"] == "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
    assert default["samples"] == 2
    assert default["shuffle"] is False


@pytest.mark.parametrize("architecture", ["Wav2Vec2ForCTC", "HubertForCTC", "AutoModelForCTC"])
def test_ctc_architectures_are_metadata_driven(architecture: str) -> None:
    assert _is_ctc_config(SimpleNamespace(architectures=[architecture]))


def test_non_ctc_asr_fails_closed() -> None:
    assert not _is_ctc_config(SimpleNamespace(architectures=["WhisperForConditionalGeneration"]))

    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = MagicMock()
    model.config = SimpleNamespace(architectures=["WhisperForConditionalGeneration"])
    config = WinMLEvaluationConfig(
        model_id="org/whisper-model",
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset"),
    )
    with pytest.raises(ValueError, match=r"only metadata-resolved \*ForCTC"):
        WinMLCTCASREvaluator(config, model)


def test_constructor_initializes_base_evaluator_state() -> None:
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    processor = _Processor(target_lang="eng")
    model = MagicMock()
    model.config = SimpleNamespace(architectures=["Wav2Vec2ForCTC"], pad_token_id=0)
    config = WinMLEvaluationConfig(
        model_id="org/ctc-model",
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset"),
    )
    prepared_data = object()

    with (
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch.object(WinMLCTCASREvaluator, "prepare_data", return_value=prepared_data),
    ):
        evaluator = WinMLCTCASREvaluator(config, model)

    assert evaluator.model is model
    assert evaluator.config is config
    assert evaluator.data is prepared_data
    assert evaluator.pipe is None


@pytest.mark.parametrize("streaming", [False, True])
def test_prepare_data_preserves_source_order_with_duplicate_ids(streaming: bool) -> None:
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    rows = [
        {"id": 1525, "audio": {"path": "first.wav"}, "transcription": "first"},
        {"id": 1657, "audio": {"path": "second.wav"}, "transcription": "second"},
        {"id": 1525, "audio": {"path": "third.wav"}, "transcription": "third"},
    ]
    dataset = MagicMock(column_names=["id", "audio", "transcription"])
    dataset.cast_column.return_value = dataset
    dataset.__len__.return_value = len(rows)
    dataset.__getitem__.side_effect = rows.__getitem__
    dataset.take.return_value = rows
    evaluator = WinMLCTCASREvaluator.__new__(WinMLCTCASREvaluator)
    evaluator._audio_column = "audio"
    evaluator.config = WinMLEvaluationConfig(
        task="automatic-speech-recognition",
        dataset=DatasetConfig(path="dataset", samples=3, shuffle=False, streaming=streaming),
    )

    with patch("datasets.load_dataset", return_value=dataset):
        selected = evaluator.prepare_data()

    assert selected["id"] == [1525, 1657, 1525]
    assert selected["_winml_source_index"] == [0, 1, 2]
    assert selected["transcription"] == ["first", "second", "third"]
    dataset.sort.assert_not_called()


def test_soundfile_decode_mixes_stereo_to_mono_float32() -> None:
    stereo = np.column_stack(
        [np.linspace(-1, 1, 32, dtype=np.float32), np.linspace(1, -1, 32, dtype=np.float32)]
    )
    waveform, sampling_rate = _decode_audio({"bytes": _wav_bytes(stereo), "path": None})
    assert sampling_rate == 16_000
    assert waveform.shape == (32,)
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform, 0.0, atol=1e-6)


def test_soundfile_decode_rejects_invalid_schema() -> None:
    with pytest.raises(_RejectedSampleError, match="neither bytes nor path"):
        _decode_audio({"bytes": None, "path": None})


def test_resample_uses_processor_rate_only_when_needed() -> None:
    waveform = np.arange(80, dtype=np.float32)
    assert _resample_audio(waveform, 16_000, 16_000) is waveform
    assert _resample_audio(waveform, 8_000, 16_000).shape == (160,)


def test_optional_lm_dependency_falls_back_to_plain_wav2vec2_processor() -> None:
    feature_extractor = object()
    tokenizer = object()
    processor = object()
    missing_lm = ImportError(
        "Wav2Vec2ProcessorWithLM requires the pyctcdecode library but it was not found"
    )

    with (
        patch("transformers.AutoProcessor.from_pretrained", side_effect=missing_lm),
        patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=feature_extractor),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.Wav2Vec2Processor", return_value=processor) as processor_class,
    ):
        assert _load_ctc_processor("org/ctc-model", trust_remote_code=False) is processor

    processor_class.assert_called_once_with(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def test_available_lm_processor_is_preserved() -> None:
    processor = object()
    with patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
        assert _load_ctc_processor("org/lm-model", trust_remote_code=False) is processor


def test_unrelated_processor_import_error_fails_closed() -> None:
    with (
        patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=ImportError("unsupported custom processor dependency"),
        ),
        pytest.raises(ImportError, match="unsupported custom processor dependency"),
    ):
        _load_ctc_processor("org/unsupported-model", trust_remote_code=False)


def test_incompatible_greedy_processor_components_fail_clearly() -> None:
    missing_lm = ImportError(
        "Wav2Vec2ProcessorWithLM requires the pyctcdecode library but it was not found"
    )
    with (
        patch("transformers.AutoProcessor.from_pretrained", side_effect=missing_lm),
        patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=object()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
        patch("transformers.Wav2Vec2Processor", side_effect=TypeError("incompatible components")),
        pytest.raises(ValueError, match="cannot form a greedy CTC processor"),
    ):
        _load_ctc_processor("org/unsupported-model", trust_remote_code=False)


def test_ordinary_wav2vec2_tokenizer_accepts_null_target_language() -> None:
    processor = _Processor(target_lang=None)
    assert _configure_processor_language(processor, SimpleNamespace(adapter_attn_dim=None)) is None


def test_mms_language_selection_uses_adapter_metadata() -> None:
    processor = _Processor(target_lang="eng")
    active = _configure_processor_language(
        processor,
        SimpleNamespace(target_lang="deu", adapter_attn_dim=16),
    )
    assert active == "deu"
    assert processor.tokenizer.target_lang == "deu"


def test_mms_published_language_is_preserved_without_config_override() -> None:
    processor = _Processor(target_lang="eng")
    assert _configure_processor_language(processor, SimpleNamespace(adapter_attn_dim=16)) == "eng"


def test_invalid_mms_adapter_fails_closed() -> None:
    processor = _Processor(target_lang="eng")
    processor.tokenizer.set_target_lang = MagicMock(side_effect=ValueError("invalid adapter"))
    with pytest.raises(ValueError, match="cannot select requested adapter language 'invalid'"):
        _configure_processor_language(
            processor,
            SimpleNamespace(target_lang="invalid", adapter_attn_dim=16),
        )


def test_mms_adapter_metadata_requires_an_active_language() -> None:
    processor = _Processor(target_lang=None)
    with pytest.raises(ValueError, match="supports language adapters but no language is active"):
        _configure_processor_language(processor, SimpleNamespace(adapter_attn_dim=16))


def test_static_audio_windows_pad_and_insert_blank_for_processor_ctc_decode() -> None:
    processor = _Processor()
    evaluator = _evaluator(input_shape=[1, 4], processor=processor)
    logits = np.array([[[0, 5, 0], [0, 5, 0]]], dtype=np.float32)
    evaluator.model.side_effect = [
        {"logits": logits},
        {"logits": logits},
    ]
    audio = {"bytes": _wav_bytes(np.arange(6, dtype=np.float32)), "path": None}

    assert evaluator._transcribe(audio) == "1 1"
    assert len(evaluator.model.call_args_list) == 2
    final_input = evaluator.model.call_args_list[1].kwargs["input_values"]
    assert final_input.shape == (1, 4)
    np.testing.assert_array_equal(final_input[0, 2:], np.zeros(2))
    assert processor.decoded_ids == [1, 1, 0, 1, 1]


def test_dynamic_audio_runs_full_utterance_once() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.model.return_value = {
        "logits": np.array([[[0, 5, 0], [5, 0, 0], [0, 0, 5]]], dtype=np.float32)
    }
    audio = {"bytes": _wav_bytes(np.arange(9, dtype=np.float32)), "path": None}

    assert evaluator._transcribe(audio) == "1 2"
    assert evaluator.model.call_count == 1
    assert evaluator.model.call_args.kwargs["input_values"].shape == (1, 9)


def test_static_audio_rejects_more_than_64_windows_before_inference() -> None:
    evaluator = _evaluator(input_shape=[1, 1])
    audio = {"bytes": _wav_bytes(np.arange(65, dtype=np.float32)), "path": None}
    with pytest.raises(_RejectedSampleError, match="cap is 64"):
        evaluator._transcribe(audio)
    evaluator.model.assert_not_called()


def test_vocab_mismatch_fails_closed() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.model.return_value = {"logits": np.zeros((1, 2, 4), dtype=np.float32)}
    audio = {"bytes": _wav_bytes(np.ones(4, dtype=np.float32)), "path": None}
    with pytest.raises(ValueError, match="does not match active tokenizer"):
        evaluator._transcribe(audio)


def test_transcript_normalization_and_exact_wer_cer() -> None:
    assert _normalize_transcript("  hello\t  world  ") == "hello world"
    assert _corpus_error_rate(["hello world"], ["hello there"], words=True) == 0.5
    assert _corpus_error_rate(["abc"], ["adc"], words=False) == pytest.approx(1 / 3)


def test_compute_preserves_accounting_and_rejection_reasons() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {
            "_winml_source_index": 0,
            "id": 1525,
            "audio": {"path": r"C:\scratch\10018492969996036091.wav", "key": "first"},
            "transcription": " hello  world ",
        },
        {
            "_winml_source_index": 1,
            "id": 1657,
            "audio": {"path": r"C:\scratch\10288018704489549018.wav", "key": "second"},
            "transcription": "ignored",
        },
    ]

    def transcribe(value: dict[str, str]) -> str:
        if value["key"] == "second":
            raise _RejectedSampleError("bad audio")
        return "hello world"

    evaluator._transcribe = transcribe
    result = evaluator.compute()
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0
    assert result["requested_samples"] == 2
    assert result["selected_samples"] == 2
    assert result["selected_source_ids"] == [1525, 1657]
    assert result["selected_source_indices"] == [0, 1]
    assert result["selected_audio_paths"] == [
        "10018492969996036091.wav",
        "10288018704489549018.wav",
    ]
    assert result["selected_audio_keys"] == ["first", "second"]
    assert result["selected_rows"] == [
        {
            "source_index": 0,
            "dataset_id": 1525,
            "audio_path": "10018492969996036091.wav",
            "audio_key": "first",
        },
        {
            "source_index": 1,
            "dataset_id": 1657,
            "audio_path": "10288018704489549018.wav",
            "audio_key": "second",
        },
    ]
    assert result["processed_samples"] == 1
    assert result["skipped_samples"] == 0
    assert result["rejected_samples"] == 1
    assert result["rejection_reasons"] == {"bad audio": 1}
    assert (
        result["processed_samples"] + result["rejected_samples"] + result["skipped_samples"]
        == result["selected_samples"]
    )


def test_compute_rejects_duplicate_selected_source_indices_before_inference() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"_winml_source_index": 0, "audio": "first", "transcription": "first"},
        {"_winml_source_index": 0, "audio": "second", "transcription": "second"},
    ]
    evaluator._transcribe = MagicMock()

    with pytest.raises(DatasetValidationError, match="duplicate source indices"):
        evaluator.compute()

    evaluator._transcribe.assert_not_called()


def test_compute_rejects_missing_selected_source_index_before_inference() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [{"audio": "value", "transcription": "reference"}]
    evaluator._transcribe = MagicMock()

    with pytest.raises(DatasetValidationError, match="no valid source index provenance"):
        evaluator.compute()

    evaluator._transcribe.assert_not_called()


def test_compute_scores_successful_empty_decode_as_deletions() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"_winml_source_index": 0, "audio": "empty", "transcription": "hello world"},
        {"_winml_source_index": 1, "audio": "decoded", "transcription": "good day"},
    ]
    evaluator._transcribe = lambda value: "" if value == "empty" else "good day"

    result = evaluator.compute()

    assert result["wer"] == 0.5
    assert result["cer"] == pytest.approx(11 / 19)
    assert result["predictions"] == ["", "good day"]
    assert result["processed_samples"] == 2
    assert result["skipped_samples"] == 0
    assert result["rejected_samples"] == 0
    assert result["rejection_reasons"] == {}


def test_compute_scores_all_empty_hypotheses_as_deletions() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"_winml_source_index": 0, "audio": "first", "transcription": "hello world"},
        {"_winml_source_index": 1, "audio": "second", "transcription": "good day"},
    ]
    evaluator._transcribe = lambda _value: ""

    result = evaluator.compute()

    assert result["wer"] == 1.0
    assert result["cer"] == 1.0
    assert result["predictions"] == ["", ""]
    assert result["processed_samples"] == 2
    assert result["skipped_samples"] == 0
    assert result["rejected_samples"] == 0
    assert result["rejection_reasons"] == {}


def test_compute_rejects_empty_reference_but_keeps_empty_hypothesis() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"_winml_source_index": 0, "audio": "invalid-reference", "transcription": " \t "},
        {"_winml_source_index": 1, "audio": "empty-hypothesis", "transcription": "valid"},
    ]
    evaluator._transcribe = lambda _value: ""

    result = evaluator.compute()

    assert result["wer"] == 1.0
    assert result["cer"] == 1.0
    assert result["predictions"] == [""]
    assert result["references"] == ["valid"]
    assert result["processed_samples"] == 1
    assert result["skipped_samples"] == 0
    assert result["rejected_samples"] == 1
    assert result["rejection_reasons"] == {"normalized transcription is empty": 1}


def test_compute_distinguishes_decode_failure_from_empty_decode() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [
        {"_winml_source_index": 0, "audio": "failure", "transcription": "rejected"},
        {"_winml_source_index": 1, "audio": "empty", "transcription": "kept"},
    ]

    def transcribe(value: str) -> str:
        if value == "failure":
            raise _RejectedSampleError("decode failed")
        return ""

    evaluator._transcribe = transcribe
    result = evaluator.compute()

    assert result["wer"] == 1.0
    assert result["cer"] == 1.0
    assert result["predictions"] == [""]
    assert result["processed_samples"] == 1
    assert result["skipped_samples"] == 0
    assert result["rejected_samples"] == 1
    assert result["rejection_reasons"] == {"decode failed": 1}


def test_compute_fails_closed_when_all_rows_are_rejected() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [{"_winml_source_index": 0, "audio": None, "transcription": ""}]
    with pytest.raises(DatasetValidationError, match="processed=0, rejected=1"):
        evaluator.compute()


def test_unexpected_inference_error_propagates() -> None:
    evaluator = _evaluator(input_shape=[1, "samples"])
    evaluator.data = [{"_winml_source_index": 0, "audio": "value", "transcription": "reference"}]
    evaluator._transcribe = MagicMock(side_effect=RuntimeError("session failed"))
    with pytest.raises(RuntimeError, match="session failed"):
        evaluator.compute()
