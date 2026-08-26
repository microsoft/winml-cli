# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Document-aware extractive question answering evaluator."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _bounded_int(mapping: dict[str, str], key: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(mapping.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def _polygon_to_box(
    polygon: Sequence[int | float], width: int, height: int
) -> list[int]:
    """Scale an OCR pixel polygon to an ordered positive-area LayoutLM box."""
    if len(polygon) != 8:
        raise ValueError("Each OCR word bounding_box must contain eight polygon coordinates.")
    if width <= 0 or height <= 0:
        raise ValueError("OCR page width and height must be positive.")
    x_values = [float(value) for value in polygon[0::2]]
    y_values = [float(value) for value in polygon[1::2]]
    left = max(0, min(1000, round(min(x_values) * 1000 / width)))
    top = max(0, min(1000, round(min(y_values) * 1000 / height)))
    right = max(0, min(1000, round(max(x_values) * 1000 / width)))
    bottom = max(0, min(1000, round(max(y_values) * 1000 / height)))
    if left >= right or top >= bottom:
        raise ValueError("OCR word bounding boxes must have positive area after normalization.")
    return [left, top, right, bottom]


def _extract_ocr_words_and_boxes(ocr_results: object) -> tuple[list[str], list[list[int]]]:
    """Flatten the pinned Pixparse nested OCR row in source order."""
    if isinstance(ocr_results, list):
        if len(ocr_results) != 1:
            raise ValueError("Document QA requires exactly one OCR page per sample.")
        ocr_results = ocr_results[0]
    if not isinstance(ocr_results, dict):
        raise TypeError("ocr_results must be a page object.")
    width = int(ocr_results.get("width", 0))
    height = int(ocr_results.get("height", 0))
    lines = ocr_results.get("lines")
    if not isinstance(lines, list):
        raise TypeError("ocr_results.lines must be a list.")

    words: list[str] = []
    boxes: list[list[int]] = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("words"), list):
            raise TypeError("Each OCR line must contain a words list.")
        for word in line["words"]:
            if not isinstance(word, dict):
                raise TypeError("Each OCR word must be an object.")
            text = word.get("text")
            polygon = word.get("bounding_box")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Each OCR word must contain non-empty text.")
            if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
                raise TypeError("Each OCR word must contain a bounding_box polygon.")
            words.append(text.strip())
            boxes.append(_polygon_to_box(polygon, width, height))
    if not words or len(words) != len(boxes):
        raise ValueError("Document OCR must contain aligned non-empty words and boxes.")
    return words, boxes


def _align_token_boxes(
    encoding: Any,
    feature_index: int,
    word_boxes: Sequence[Sequence[int]],
    sep_token_id: int | None,
) -> list[list[int]]:
    """Propagate document word boxes to subwords in one overflow window."""
    input_ids = encoding["input_ids"][feature_index].tolist()
    sequence_ids = encoding.sequence_ids(feature_index)
    word_ids = encoding.word_ids(feature_index)
    aligned: list[list[int]] = []
    for token_id, sequence_id, word_id in zip(input_ids, sequence_ids, word_ids, strict=True):
        if sequence_id == 1 and word_id is not None:
            if word_id >= len(word_boxes):
                raise ValueError("Tokenizer word alignment exceeds the available OCR boxes.")
            aligned.append(list(word_boxes[word_id]))
        elif sep_token_id is not None and token_id == sep_token_id:
            aligned.append([1000, 1000, 1000, 1000])
        else:
            aligned.append([0, 0, 0, 0])
    return aligned


def _decode_document_span(
    start_logits: Any,
    end_logits: Any,
    sequence_ids: Sequence[int | None],
    word_ids: Sequence[int | None],
    words: Sequence[str],
    max_answer_words: int,
) -> tuple[float, str]:
    """Decode the highest-scoring valid contiguous OCR-word span."""
    import numpy as np

    starts = np.asarray(start_logits).reshape(-1)
    ends = np.asarray(end_logits).reshape(-1)
    if len(starts) != len(sequence_ids) or len(ends) != len(sequence_ids):
        raise ValueError("Question-answering logits do not match the tokenized window length.")
    best_score = float("-inf")
    best_answer = ""
    for start_index, (sequence_id, start_word) in enumerate(
        zip(sequence_ids, word_ids, strict=True)
    ):
        if sequence_id != 1 or start_word is None:
            continue
        for end_index in range(start_index, len(sequence_ids)):
            end_word = word_ids[end_index]
            if sequence_ids[end_index] != 1 or end_word is None:
                break
            word_count = end_word - start_word + 1
            if word_count < 1:
                continue
            if word_count > max_answer_words:
                break
            if end_word >= len(words):
                raise ValueError("Tokenizer word alignment exceeds the available OCR words.")
            score = float(starts[start_index] + ends[end_index])
            if score > best_score:
                best_score = score
                best_answer = " ".join(words[start_word : end_word + 1])
    return best_score, best_answer


class WinMLDocumentQuestionAnsweringEvaluator(WinMLEvaluator):
    """Evaluate contiguous LayoutLM-style document answer spans with ANLS."""

    def prepare_pipeline(self) -> Any:
        """Load the tokenizer without constructing the text-only QA pipeline."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Document question answering requires a fast tokenizer.")
        return SimpleNamespace(tokenizer=tokenizer, device=self.config.pipeline_device)

    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return cast("PreTrainedTokenizerBase", self.pipe.tokenizer)

    def _model_input_names(self) -> set[str]:
        """Return inputs declared by ONNX metadata or a native model signature."""
        io_config = getattr(self.model, "io_config", None) or {}
        input_names = set(io_config.get("input_names") or [])
        if input_names:
            return input_names
        forward = getattr(self.model, "forward", None)
        try:
            parameters = inspect.signature(forward).parameters if forward is not None else {}
        except (TypeError, ValueError):
            return set()
        return {
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

    def compute(self) -> dict[str, Any]:
        """Run bounded document preprocessing, inference, decoding, and ANLS."""
        import torch

        from .metrics import ANLSMetric

        input_names = self._model_input_names()
        if "bbox" not in input_names:
            raise ValueError("Document question answering requires a declared bbox model input.")
        mapping = self.config.dataset.columns_mapping
        max_windows = _bounded_int(mapping, "max_windows", 1)
        doc_stride = _bounded_int(mapping, "doc_stride", 128, minimum=0)
        max_answer_words = _bounded_int(mapping, "max_answer_words", 64)
        if _bounded_int(mapping, "top_k", 1) != 1:
            raise ValueError("Document question answering currently requires top_k=1.")
        tokenizer = self._tokenizer()
        max_length = self._fixed_seq_length() or tokenizer.model_max_length
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("Document QA requires a finite sequence length.")

        metric = ANLSMetric()
        windows_processed = 0
        processed_samples = 0
        question_column = mapping.get("question_column", "question")
        label_column = mapping.get("label_column", "answers")
        ocr_column = mapping.get("ocr_column", "ocr_results")
        for row in self.data:
            words, boxes = _extract_ocr_words_and_boxes(row[ocr_column])
            encoding = tokenizer(
                str(row[question_column]).split(),
                words,
                is_split_into_words=True,
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                padding="max_length",
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            feature_count = min(len(encoding["input_ids"]), max_windows)
            best_score = float("-inf")
            prediction = ""
            for feature_index in range(feature_count):
                aligned_boxes = _align_token_boxes(
                    encoding, feature_index, boxes, tokenizer.sep_token_id
                )
                model_inputs: dict[str, Any] = {}
                for input_name in input_names:
                    if input_name == "bbox":
                        model_inputs[input_name] = torch.tensor([aligned_boxes], dtype=torch.long)
                    elif input_name == "token_type_ids" and input_name not in encoding:
                        model_inputs[input_name] = torch.zeros_like(
                            encoding["input_ids"][feature_index : feature_index + 1]
                        )
                    elif input_name in encoding and isinstance(encoding[input_name], torch.Tensor):
                        model_inputs[input_name] = encoding[input_name][
                            feature_index : feature_index + 1
                        ]
                    else:
                        raise ValueError(
                            f"Tokenizer did not produce declared model input '{input_name}'."
                        )
                model_inputs = {
                    name: tensor.to(self.config.pipeline_device)
                    for name, tensor in model_inputs.items()
                }
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                score, answer = _decode_document_span(
                    outputs.start_logits[0].detach().cpu().numpy(),
                    outputs.end_logits[0].detach().cpu().numpy(),
                    encoding.sequence_ids(feature_index),
                    encoding.word_ids(feature_index),
                    words,
                    max_answer_words,
                )
                if score > best_score:
                    best_score, prediction = score, answer
                windows_processed += 1
            metric.update(prediction, row[label_column])
            processed_samples += 1
        return {
            **metric.compute(),
            "requested_samples": self.config.dataset.samples,
            "processed_samples": processed_samples,
            "windows_processed": windows_processed,
        }
