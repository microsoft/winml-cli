# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared HF Pipeline factory for WinML models.

Used by both ``winml serve`` (InferenceEngine) and ``winml eval`` (WinMLEvaluator)
to create a ``transformers.pipeline`` backed by a WinMLPreTrainedModel.

The pipeline handles all preprocessing and postprocessing; the WinML model
only provides the ONNX Runtime inference session.

ONNX models have fixed input shapes. This module adapts the pipeline's
tokenizer/image_processor to match those shapes so inputs are correctly
padded/resized before hitting the ONNX runtime.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any


logger = logging.getLogger(__name__)

# Tasks that WinML recognises but HF ``transformers.pipeline`` does not.
# Mapped to their HF pipeline equivalent before calling ``pipeline()``.
_HF_PIPELINE_TASK_MAP: dict[str, str] = {
    "image-to-text": "image-text-to-text",
    "next-sentence-prediction": "text-classification",
    "sequence-classification": "text-classification",
    "sentence-similarity": "feature-extraction",
}

def _is_transformers4() -> bool:
    """Return whether the active Transformers runtime is version 4.x."""
    from transformers import __version__

    return __version__.startswith("4.")


def _resolve_hf_pipeline_task(task: str) -> str:
    """Resolve task aliases while preserving Transformers 4 native pipelines."""
    if task == "image-to-text" and _is_transformers4():
        return task
    return _HF_PIPELINE_TASK_MAP.get(task, task)


_PIPELINE_COMPONENT_FLAGS = (
    ("tokenizer", "_load_tokenizer"),
    ("feature_extractor", "_load_feature_extractor"),
    ("image_processor", "_load_image_processor"),
    ("processor", "_load_processor"),
)


def _model_static_batch_size(
    io_config: Mapping[str, Any],
    model_input_names: set[str],
) -> int:
    """Return the model's fixed batch size, or one for dynamic batches."""
    static_batch_sizes = {
        shape[0]
        for name, shape in zip(
            io_config.get("input_names", []),
            io_config.get("input_shapes", []),
            strict=False,
        )
        if name in model_input_names and shape and isinstance(shape[0], int) and shape[0] > 0
    }
    if len(static_batch_sizes) > 1:
        raise ValueError("Question-answering model inputs declare inconsistent batch sizes.")
    return next(iter(static_batch_sizes), 1)


class _ExtractiveQuestionAnsweringPipeline:
    """Transformers 5-compatible extractive question-answering pipeline."""

    task = "question-answering"

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._preprocess_params: dict[str, Any] = {}

    def preprocess(self, inputs: Any, **kwargs: Any) -> Any:
        """Expose a tokenizer-compatible signature for fixed-shape adaptation."""
        return inputs

    def _sanitize_parameters(
        self,
        handle_impossible_answer: bool = False,
        max_answer_len: int | None = None,
        doc_stride: int | None = None,
        topk: int | None = None,
        top_k: int | None = None,
        align_to_words: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Expose supported inference parameters for serving metadata."""
        postprocess_params: dict[str, Any] = {}
        if topk is not None and top_k is None:
            warnings.warn(
                "topk parameter is deprecated, use top_k instead",
                UserWarning,
                stacklevel=2,
            )
            top_k = topk
        if top_k is not None:
            self._validate_top_k(top_k)
            postprocess_params["top_k"] = top_k
        if max_answer_len is not None:
            self._validate_max_answer_len(max_answer_len)
            postprocess_params["max_answer_len"] = max_answer_len
        if handle_impossible_answer is not None:
            postprocess_params["handle_impossible_answer"] = handle_impossible_answer
        if align_to_words is not None:
            postprocess_params["align_to_words"] = align_to_words
        return (
            {"doc_stride": doc_stride} if doc_stride is not None else {},
            {},
            postprocess_params,
        )

    def __call__(
        self,
        *args: Any,
        question: Any = None,
        context: Any = None,
        handle_impossible_answer: bool = False,
        max_answer_len: int | None = None,
        doc_stride: int | None = None,
        topk: int | None = None,
        top_k: int | None = None,
        align_to_words: bool | None = None,
    ) -> (
        dict[str, Any]
        | list[dict[str, Any]]
        | list[dict[str, Any] | list[dict[str, Any]]]
        | Iterator[
            dict[str, Any] | list[dict[str, Any]] | list[dict[str, Any] | list[dict[str, Any]]]
        ]
    ):
        """Extract answer spans for one or more question/context pairs."""
        inputs = None
        if args:
            warnings.warn(
                "Passing a list of SQuAD examples to the pipeline is deprecated and will be "
                "removed in v5. Inputs should be passed using the `question` and `context` "
                "keyword arguments instead.",
                FutureWarning,
                stacklevel=2,
            )
            if len(args) == 1:
                inputs = args[0]
            elif len(args) == 2 and all(isinstance(item, str) for item in args):
                question, context = args
            else:
                inputs = list(args)
        if topk is not None and top_k is None:
            warnings.warn(
                "topk parameter is deprecated, use top_k instead",
                UserWarning,
                stacklevel=2,
            )
            top_k = topk
        if top_k is None:
            top_k = 1
        self._validate_top_k(top_k)
        if max_answer_len is None:
            max_answer_len = 15
        self._validate_max_answer_len(max_answer_len)
        if align_to_words is None:
            align_to_words = True
        if isinstance(inputs, Iterator):
            if question is not None or context is not None:
                raise ValueError(
                    "Pass question/context either as inputs or keyword arguments, not both."
                )
            return self._answer_iterator(
                inputs,
                handle_impossible_answer=handle_impossible_answer,
                max_answer_len=max_answer_len,
                doc_stride=doc_stride,
                top_k=top_k,
                align_to_words=align_to_words,
            )
        questions, contexts, single_input = self._normalize_inputs(inputs, question, context)
        answers = [
            self._answer(
                item_question,
                item_context,
                handle_impossible_answer=handle_impossible_answer,
                max_answer_len=max_answer_len,
                doc_stride=doc_stride,
                top_k=top_k,
                align_to_words=align_to_words,
            )
            for item_question, item_context in zip(questions, contexts, strict=True)
        ]
        return answers[0] if single_input else answers

    def _answer_iterator(
        self,
        inputs: Iterator[Any],
        *,
        handle_impossible_answer: bool,
        max_answer_len: int,
        doc_stride: int | None,
        top_k: int,
        align_to_words: bool,
    ) -> Iterator[
        dict[str, Any] | list[dict[str, Any]] | list[dict[str, Any] | list[dict[str, Any]]]
    ]:
        for item in inputs:
            questions, contexts, single_input = self._normalize_inputs(item, None, None)
            answers = [
                self._answer(
                    item_question,
                    item_context,
                    handle_impossible_answer=handle_impossible_answer,
                    max_answer_len=max_answer_len,
                    doc_stride=doc_stride,
                    top_k=top_k,
                    align_to_words=align_to_words,
                )
                for item_question, item_context in zip(questions, contexts, strict=True)
            ]
            yield answers[0] if single_input else answers

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k < 1:
            raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")

    @staticmethod
    def _validate_max_answer_len(max_answer_len: int) -> None:
        if max_answer_len < 1:
            raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")

    @staticmethod
    def _normalize_inputs(
        inputs: Any,
        question: Any,
        context: Any,
    ) -> tuple[list[str], list[str], bool]:
        if inputs is not None:
            if question is not None or context is not None:
                raise ValueError(
                    "Pass question/context either as inputs or keyword arguments, not both."
                )
            if isinstance(inputs, Mapping):
                question = inputs.get("question")
                context = inputs.get("context")
            elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
                if not all(isinstance(item, Mapping) for item in inputs):
                    raise TypeError("Question-answering batch inputs must be mappings.")
                question = [item.get("question") for item in inputs]
                context = [item.get("context") for item in inputs]
            else:
                raise TypeError(
                    "Question-answering inputs must be a mapping or sequence of mappings."
                )

        single_input = isinstance(question, str) and isinstance(context, str)
        questions = (
            [question]
            if isinstance(question, str)
            else list(question)
            if isinstance(question, Iterable)
            else [question]
            if question is not None
            else []
        )
        if isinstance(context, str):
            contexts = [context] * len(questions) if not isinstance(question, str) else [context]
        else:
            contexts = (
                list(context)
                if isinstance(context, Iterable)
                else [context]
                if context is not None
                else []
            )
        if not questions or len(questions) != len(contexts):
            raise ValueError("question and context must contain the same non-zero number of items.")
        if not all(isinstance(item, str) for item in (*questions, *contexts)):
            raise TypeError("question and context values must be strings.")
        for item in questions:
            if not item:
                raise ValueError("`question` cannot be empty")
        for item in contexts:
            if not item:
                raise ValueError("`context` cannot be empty")
        return questions, contexts, single_input

    def _answer(
        self,
        question: str,
        context: str,
        *,
        handle_impossible_answer: bool,
        max_answer_len: int,
        doc_stride: int | None,
        top_k: int,
        align_to_words: bool,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        import torch
        from transformers.tokenization_utils_base import LARGE_INTEGER

        max_length = self._preprocess_params.get("max_length")
        if max_length is None:
            tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < LARGE_INTEGER:
                max_length = tokenizer_max_length

        question_first = getattr(self.tokenizer, "padding_side", "right") == "right"
        context_sequence_id = 1 if question_first else 0
        first_text = question if question_first else context
        second_text = context if question_first else question
        tokenizer_kwargs: dict[str, Any] = {
            "truncation": "only_second" if question_first else "only_first",
            "return_offsets_mapping": True,
            "return_overflowing_tokens": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            question_length = len(self.tokenizer.encode(question, add_special_tokens=False))
            context_window = (
                max_length - question_length - self.tokenizer.num_special_tokens_to_add(pair=True)
            )
            if context_window < 1:
                raise ValueError(
                    "The tokenized question leaves no room for context in the model input."
                )
            resolved_stride = (
                min(max_length // 2, 128, context_window - 1) if doc_stride is None else doc_stride
            )
            if resolved_stride < 0 or resolved_stride >= context_window:
                raise ValueError(
                    f"doc_stride ({resolved_stride}) must be between 0 and "
                    f"{context_window - 1} for this question."
                )
            tokenizer_kwargs.update(
                {
                    "max_length": max_length,
                    "padding": "max_length",
                    "stride": resolved_stride,
                }
            )
        elif doc_stride is not None:
            tokenizer_kwargs["stride"] = doc_stride

        encoding = self.tokenizer(first_text, second_text, **tokenizer_kwargs)
        offsets = encoding["offset_mapping"]
        io_config = getattr(self.model, "io_config", None) or {}
        model_input_names = set(io_config.get("input_names", []))
        if not model_input_names:
            model_input_names = set(getattr(self.tokenizer, "model_input_names", []))
        model_batch_size = _model_static_batch_size(io_config, model_input_names)
        cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
        primary_input_name = next(
            (name for name in getattr(self.tokenizer, "model_input_names", []) if name in encoding),
            None,
        )

        answers_by_span: dict[tuple[int, int], dict[str, Any]] = {}
        best_no_answer_score: float | None = None
        feature_count = len(offsets)
        for batch_start in range(0, feature_count, model_batch_size):
            batch_end = min(feature_count, batch_start + model_batch_size)
            real_batch_size = batch_end - batch_start
            model_inputs: dict[str, Any] = {}
            for name, value in encoding.items():
                if name not in model_input_names or not isinstance(value, torch.Tensor):
                    continue
                batch = value[batch_start:batch_end]
                if real_batch_size < model_batch_size:
                    repeats = (model_batch_size - real_batch_size,) + (1,) * (batch.ndim - 1)
                    batch = torch.cat((batch, batch[-1:].repeat(repeats)), dim=0)
                model_inputs[name] = batch
            if not model_inputs:
                raise ValueError("Tokenizer outputs do not match the model's declared inputs.")
            model_inputs = {name: tensor.to(self.device) for name, tensor in model_inputs.items()}
            outputs = self.model(**model_inputs)
            start_logits = getattr(outputs, "start_logits", None)
            end_logits = getattr(outputs, "end_logits", None)
            if start_logits is None or end_logits is None:
                raise ValueError(
                    "Question-answering model must return start_logits and end_logits."
                )

            if start_logits.shape[0] < real_batch_size or end_logits.shape[0] < real_batch_size:
                raise ValueError(
                    "Question-answering model returned fewer rows than the input batch."
                )

            for batch_index in range(real_batch_size):
                feature_index = batch_start + batch_index
                sequence_ids = encoding.sequence_ids(feature_index)
                feature_offsets = offsets[feature_index].tolist()
                feature_encoding = encoding.encodings[feature_index]
                start_scores = start_logits[batch_index].float()
                end_scores = end_logits[batch_index].float()
                valid_positions = torch.tensor(
                    [sequence_id == context_sequence_id for sequence_id in sequence_ids],
                    dtype=torch.bool,
                    device=start_scores.device,
                )

                cls_index: int | None = None
                if cls_token_id is not None and primary_input_name is not None:
                    cls_positions = (
                        encoding[primary_input_name][feature_index] == cls_token_id
                    ).nonzero(as_tuple=False)
                    if cls_positions.numel():
                        cls_index = int(cls_positions[0].item())
                        valid_positions[cls_index] = True

                if (
                    valid_positions.numel() != start_scores.numel()
                    or valid_positions.numel() != end_scores.numel()
                ):
                    raise ValueError(
                        "Question-answering logits do not match the tokenized feature length."
                    )
                start_probabilities = torch.softmax(
                    start_scores.masked_fill(~valid_positions, -torch.inf),
                    dim=-1,
                )
                end_probabilities = torch.softmax(
                    end_scores.masked_fill(~valid_positions, -torch.inf),
                    dim=-1,
                )

                if cls_index is not None:
                    no_answer_score = float(
                        start_probabilities[cls_index] * end_probabilities[cls_index]
                    )
                    if best_no_answer_score is None or no_answer_score < best_no_answer_score:
                        best_no_answer_score = no_answer_score

                for start_index, sequence_id in enumerate(sequence_ids):
                    if sequence_id != context_sequence_id:
                        continue
                    max_end = min(len(sequence_ids), start_index + max_answer_len)
                    for end_index in range(start_index, max_end):
                        if sequence_ids[end_index] != context_sequence_id:
                            continue
                        char_start = int(feature_offsets[start_index][0])
                        char_end = int(feature_offsets[end_index][1])
                        if align_to_words:
                            start_word = feature_encoding.token_to_word(start_index)
                            end_word = feature_encoding.token_to_word(end_index)
                            if start_word is not None and end_word is not None:
                                start_chars = feature_encoding.word_to_chars(
                                    start_word,
                                    sequence_index=context_sequence_id,
                                )
                                end_chars = feature_encoding.word_to_chars(
                                    end_word,
                                    sequence_index=context_sequence_id,
                                )
                                if start_chars is not None and end_chars is not None:
                                    char_start = start_chars[0]
                                    char_end = end_chars[1]
                        if char_end <= char_start:
                            continue
                        score = float(
                            start_probabilities[start_index] * end_probabilities[end_index]
                        )
                        answer_text = context[char_start:char_end]
                        answer_key = (char_start, char_end)
                        existing_answer = answers_by_span.get(answer_key)
                        if existing_answer is None or score > existing_answer["score"]:
                            answers_by_span[answer_key] = {
                                "score": score,
                                "start": char_start,
                                "end": char_end,
                                "answer": answer_text,
                            }

        no_answer_score = best_no_answer_score if best_no_answer_score is not None else 0.0
        answers = list(answers_by_span.values())
        if handle_impossible_answer:
            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
        if not answers:
            return {"score": no_answer_score, "start": 0, "end": 0, "answer": ""}
        answers.sort(key=lambda answer: answer["score"], reverse=True)
        answers = answers[:top_k]
        return answers[0] if len(answers) == 1 else answers


def _create_extractive_question_answering_pipeline(
    model: Any,
    model_id: str | None,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> _ExtractiveQuestionAnsweringPipeline:
    """Create the extractive QA pipeline removed from Transformers 5."""
    from transformers import AutoTokenizer

    tokenizer_source = model_id or getattr(getattr(model, "config", None), "_name_or_path", None)
    if not tokenizer_source:
        raise ValueError("model_id is required to load a question-answering tokenizer.")
    tokenizer_kwargs = {"use_fast": True}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Extractive question answering requires a fast tokenizer with offset mappings. "
            "Use a model that provides a fast tokenizer implementation."
        )
    return _ExtractiveQuestionAnsweringPipeline(model, tokenizer, device=device)


_COMPAT_PIPELINE_FACTORIES = {
    "question-answering": _create_extractive_question_answering_pipeline,
    "document-question-answering": _create_extractive_question_answering_pipeline,
}


def _pipeline_component_kwargs(task: str, model_id: str | None) -> dict[str, str]:
    """Select model components from the resolved pipeline's capabilities."""
    if model_id is None:
        return {}

    from transformers.pipelines import check_task

    _, targeted_task, _ = check_task(task)
    pipeline_class = targeted_task["impl"]
    return {
        argument: model_id
        for argument, flag in _PIPELINE_COMPONENT_FLAGS
        if getattr(pipeline_class, flag, None) is not False
    }


def create_pipeline(
    task: str,
    model: Any,
    model_id: str | None = None,
    *,
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> Any:
    """Create an HF pipeline for a WinML model.

    Automatically adapts tokenizer padding and image processor size
    to match the ONNX model's fixed input shapes.

    Args:
        task: HF task name (e.g. "image-classification")
        model: Loaded WinML or native Hugging Face model instance.
        model_id: HF model ID for loading processors (tokenizer, image processor).
                  If None, pipeline will attempt auto-detection.
        device: Device used by the Transformers pipeline for input tensors.
        trust_remote_code: Whether custom Hugging Face component code may execute.

    Returns:
        A configured task callable ready for inference.
    """
    from transformers import pipeline

    hf_task = _resolve_hf_pipeline_task(task)
    compatibility_factory = _COMPAT_PIPELINE_FACTORIES.get(hf_task)
    if compatibility_factory is not None and not _is_transformers4():
        pipe = compatibility_factory(
            model,
            model_id,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    else:
        kwargs: dict[str, Any] = {
            # "device" is for HF pipeline tensor placement, not ORT EP.
            # WinMLSession handles device delegation internally.
            "device": device,
            **_pipeline_component_kwargs(hf_task, model_id),
        }
        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        # transformers.pipeline has 60+ Literal overloads — runtime task strings can't
        # be statically matched. The string-task fallback handles unknown tasks safely.
        pipe = pipeline(hf_task, model=model, **kwargs)  # type: ignore[call-overload]

    # Adapt pipeline to fixed ONNX input shapes
    _adapt_tokenizer_padding(pipe, task, model)
    _adapt_image_processor_size(pipe, task, model)

    logger.info("Created HF pipeline: task=%s model=%s", task, model_id)
    return pipe


def _adapt_tokenizer_padding(pipe: Any, task: str, model: Any) -> None:
    """Pad tokenizer output to match ONNX fixed sequence length.

    ONNX models are exported with a fixed sequence_length dimension.
    Without padding, the tokenizer produces variable-length tensors
    that cause INVALID_ARGUMENT errors at inference time.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has a tokenizer AND the
    model's first input shape is 2-D with a fixed integer second
    dimension (batch, sequence_length).  4-D shapes (N, C, H, W) are
    image tensors and are explicitly skipped.
    """
    if pipe.tokenizer is None:
        return

    io_config = getattr(model, "io_config", None) or {}
    shapes = io_config.get("input_shapes", [[]])
    # Find the first 2-D shape (batch, seq_len) — multi-modal models like CLIP
    # have both 2-D text inputs and 4-D image inputs; scanning all shapes ensures
    # tokenizer padding is applied regardless of input ordering.
    max_length = None
    for shape in shapes:
        if len(shape) == 2 and isinstance(shape[1], int):
            max_length = shape[1]
            break
    if max_length is None:
        return

    # HF pipeline classes consume tokenizer settings in three patterns:
    #
    # A) Direct **kwargs → tokenizer (TextClassification, FeatureExtraction)
    #    e.g. self.tokenizer(text, **tokenizer_kwargs)
    #    → set top-level padding/max_length/truncation in _preprocess_params
    #
    # B) Nested tokenizer dict (TokenClassification, FillMask)
    #    e.g. tok_params = preprocess_params.pop("tokenizer_params", {})
    #         self.tokenizer(text, truncation=truncation, **tok_params)
    #    or:  self.tokenizer(text, **tokenizer_kwargs)  [named param]
    #    → set padding/max_length inside a dict param
    #
    # C) Explicit named params only (QuestionAnswering: max_seq_len)
    #    No **kwargs — only accepts specific named params
    #    → set only params that appear in the signature

    preprocess_sig = inspect.signature(type(pipe).preprocess)
    sig_params = preprocess_sig.parameters

    tok_dict_key = _detect_tokenizer_dict_param(pipe, sig_params)
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())

    if tok_dict_key:
        # Pattern B: tokenizer settings go in a nested dict
        pipe._preprocess_params.setdefault(tok_dict_key, {})
        tok = pipe._preprocess_params[tok_dict_key]
        tok.setdefault("padding", "max_length")
        tok.setdefault("max_length", max_length)
        # TokenClassification pops "truncation" separately from **kwargs
        if tok_dict_key == "tokenizer_params":
            pipe._preprocess_params.setdefault("truncation", True)
        else:
            tok.setdefault("truncation", True)
    elif has_varkw:
        # Pattern A: **kwargs forwarded directly to tokenizer
        pipe._preprocess_params.setdefault("padding", "max_length")
        pipe._preprocess_params.setdefault("max_length", max_length)
        pipe._preprocess_params.setdefault("truncation", True)
    else:
        # Pattern C: no **kwargs — only set params the signature accepts
        if "max_seq_len" in sig_params:
            pipe._preprocess_params.setdefault("max_seq_len", max_length)
        elif "max_length" in sig_params:
            pipe._preprocess_params.setdefault("max_length", max_length)
        if "padding" in sig_params:
            pipe._preprocess_params.setdefault("padding", "max_length")
        if "truncation" in sig_params:
            pipe._preprocess_params.setdefault("truncation", True)

    pipe.tokenizer.model_max_length = max_length


def _detect_tokenizer_dict_param(
    pipe: Any, sig_params: Mapping[str, inspect.Parameter]
) -> str | None:
    """Detect if preprocess() consumes tokenizer settings via a nested dict.

    Returns the dict key name (e.g. "tokenizer_kwargs", "tokenizer_params"),
    or None if the pipeline uses direct **kwargs or explicit named params.
    """
    # Check for a named (non-**kwargs) parameter like tokenizer_kwargs=None
    # (e.g. FillMaskPipeline)
    for name, param in sig_params.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if name != "self" and ("tokenizer" in name or "tokenize" in name):
            return name

    # Check if preprocess() pops "tokenizer_params" from **kwargs
    # (e.g. TokenClassificationPipeline).  Source inspection is fragile —
    # it fails for compiled (.pyc-only) code or C extensions — but there
    # is no runtime API to detect dict-style consumption of **kwargs.
    # The except clause degrades gracefully to "no nested dict detected".
    try:
        src = inspect.getsource(type(pipe).preprocess)
    except (OSError, TypeError):
        return None
    if "tokenizer_params" in src:
        return "tokenizer_params"

    return None


def _adapt_image_processor_size(pipe: Any, task: str, model: Any) -> None:
    """Match image processor size to ONNX fixed input shape (NCHW).

    Models with 4D input shapes have fixed spatial dimensions.
    The image processor must resize to exactly those dimensions.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has an image_processor AND
    the model's first input shape is 4D (N, C, H, W).

    Size dict format varies by processor class:
      - ``{"height": h, "width": w}`` — direct resize (ViT, DETR, …)
      - ``{"shortest_edge": n}`` — aspect-preserving resize, usually
        followed by a center crop (ResNet, ConvNeXt, …)
    We preserve the processor's original format to avoid validation errors.
    """
    if not hasattr(pipe, "image_processor"):
        return

    io_config = getattr(model, "io_config", None) or {}
    input_shapes = io_config.get("input_shapes", [])
    # Find the first 4-D shape (N, C, H, W) — multi-modal models may have
    # both 2-D text and 4-D image inputs in any order.
    image_shape = None
    for shape in input_shapes:
        if len(shape) == 4:
            image_shape = shape
            break
    if image_shape is None:
        return

    _, _, h, w = image_shape
    proc = pipe.image_processor
    original_size = getattr(proc, "size", {}) or {}

    if "shortest_edge" in original_size and "longest_edge" not in original_size:
        # Processor only accepts shortest_edge format (e.g. ConvNeXt).
        # These processors use crop_pct internally to resize then
        # center-crop to (shortest_edge, shortest_edge), so setting
        # shortest_edge = min(h, w) produces the correct output for
        # square ONNX shapes.  Forcing {"height", "width"} would raise
        # a validation error in their resize() method.
        proc.size = {"shortest_edge": min(h, w)}
    else:
        # Processors with height/width (ViT) or shortest_edge+longest_edge
        # (DETR) all accept explicit height/width for exact dimensions.
        proc.size = {"height": h, "width": w}

    if hasattr(proc, "do_pad"):
        proc.do_pad = False
