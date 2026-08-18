# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Question answering evaluator with tokenizer padding."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from .base_evaluator import WinMLEvaluator, _ensure_evaluate_transformers_compat


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

_DOCUMENT_COLUMN_KEYS = {"words_column", "boxes_column", "image_column"}


def _is_true(value: str | None) -> bool:
    return value is not None and value.casefold() in {"1", "true", "yes", "on"}


def _bounded_int(mapping: dict[str, str], key: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(mapping.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def _normalize_boxes(
    boxes: Sequence[Sequence[int | float]],
    *,
    image_size: tuple[int, int] | None,
    coordinate_system: str,
) -> list[list[int]]:
    """Normalize xyxy boxes to integer LayoutLM coordinates in [0, 1000]."""
    if coordinate_system not in {"auto", "absolute", "normalized"}:
        raise ValueError("box_coords must be one of: auto, absolute, normalized.")
    parsed: list[tuple[float, float, float, float]] = []
    for box in boxes:
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError("Each document box must contain four xyxy coordinates.")
        x0, y0, x1, y1 = (float(value) for value in box)
        parsed.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    use_absolute = coordinate_system == "absolute" or (
        coordinate_system == "auto"
        and image_size is not None
        and any(value > 1000 for box in parsed for value in box)
    )
    if use_absolute and image_size is None:
        raise ValueError("Absolute document boxes require an image with width and height.")
    width, height = image_size or (1000, 1000)
    if width <= 0 or height <= 0:
        raise ValueError("Document image dimensions must be positive.")

    normalized: list[list[int]] = []
    for x0, y0, x1, y1 in parsed:
        if use_absolute:
            x0, x1 = x0 * 1000 / width, x1 * 1000 / width
            y0, y1 = y0 * 1000 / height, y1 * 1000 / height
        normalized.append(
            [round(max(0, min(1000, value))) for value in (x0, y0, x1, y1)]
        )
    return normalized


def _align_token_boxes(
    encoding: Any,
    feature_index: int,
    word_boxes: Sequence[Sequence[int]],
    sep_token_id: int | None,
) -> list[list[int]]:
    """Propagate document word boxes to subwords in one overflow feature."""
    input_ids = encoding["input_ids"][feature_index].tolist()
    sequence_ids = encoding.sequence_ids(feature_index)
    word_ids = encoding.word_ids(feature_index)
    aligned: list[list[int]] = []
    for token_id, sequence_id, word_id in zip(input_ids, sequence_ids, word_ids, strict=True):
        if sequence_id == 1 and word_id is not None:
            if word_id >= len(word_boxes):
                raise ValueError("Tokenizer word alignment exceeds the available document boxes.")
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
    """Decode the best valid contiguous document-word span."""
    import numpy as np

    starts = np.asarray(start_logits).reshape(-1)
    ends = np.asarray(end_logits).reshape(-1)
    if len(starts) != len(sequence_ids) or len(ends) != len(sequence_ids):
        raise ValueError("Question-answering logits do not match the tokenized feature length.")

    best_score = float("-inf")
    best_answer = ""
    for start_index, (start_sequence, start_word) in enumerate(
        zip(sequence_ids, word_ids, strict=True)
    ):
        if start_sequence != 1 or start_word is None:
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
                raise ValueError("Tokenizer word alignment exceeds the available document words.")
            score = float(starts[start_index] + ends[end_index])
            if score > best_score:
                best_score = score
                best_answer = " ".join(words[start_word : end_word + 1])
    return best_score, best_answer


class WinMLQuestionAnsweringEvaluator(WinMLEvaluator):
    """Evaluator for extractive question answering tasks.

    Uses HF QuestionAnsweringEvaluator with SQuAD metrics (exact_match, f1).
    Configures tokenizer max length to match the ONNX model's fixed sequence length.

    Dataset schema requires: question, context, id, answers.
    Additional model inputs like ``token_type_ids`` are not part of the dataset
    schema — they are generated by the tokenizer and auto-filtered by
    :class:`WinMLModelForQuestionAnswering` based on the ONNX model's declared
    inputs (e.g., DeBERTa-v3 omits token_type_ids).

    The evaluator auto-detects SQuAD v2 (unanswerable questions) and switches
    the metric accordingly.  The default dataset (``rajpurkar/squad``) is v1;
    passing a v2 dataset works transparently.
    """

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and set tokenizer padding for fixed-shape ONNX.

        Extracts sequence length from ``io_config["input_shapes"][0][1]``
        (shape of the first input's second dimension, typically seq_len).
        If io_config is unavailable or the shape cannot be resolved, the
        tokenizer keeps its own default max length and a warning is logged.
        """
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                max_length = shapes[0][1]
                pipe.tokenizer.model_max_length = max_length
                pipe._preprocess_params.setdefault("padding", "max_length")
                pipe._preprocess_params.setdefault("max_seq_len", max_length)
            else:
                logger.warning(
                    "Could not determine sequence length from io_config input_shapes. "
                    "Tokenizer will use its own default max length."
                )

        return pipe

    def _is_document_mode(self) -> bool:
        mapping = self.config.dataset.columns_mapping
        if _is_true(mapping.get("document_mode")):
            return True
        input_names = set((getattr(self.model, "io_config", None) or {}).get("input_names", []))
        return "bbox" in input_names and bool(_DOCUMENT_COLUMN_KEYS & mapping.keys())

    def prepare_data(self) -> Dataset:
        """Load ordinary QA samples or one explicitly selected document row."""
        mapping = self.config.dataset.columns_mapping
        if not self._is_document_mode() or "sample_index" not in mapping:
            return super().prepare_data()

        sample_index = _bounded_int(mapping, "sample_index", 0, minimum=0)
        dataset_config = self.config.dataset
        original_samples = dataset_config.samples
        original_shuffle = dataset_config.shuffle
        dataset_config.samples = sample_index + 1
        dataset_config.shuffle = False
        try:
            dataset = super().prepare_data()
        finally:
            dataset_config.samples = original_samples
            dataset_config.shuffle = original_shuffle
        if sample_index >= len(dataset):
            raise ValueError(
                f"sample_index {sample_index} is outside the loaded dataset of {len(dataset)} rows."
            )
        return dataset.select([sample_index])

    def validate_data(self, dataset: Dataset) -> None:
        """Validate either the existing text schema or document QA columns."""
        if not self._is_document_mode():
            super().validate_data(dataset)
            return

        from ..utils.eval_utils import DatasetValidationError

        mapping = self.config.dataset.columns_mapping
        columns = set(dataset.column_names)
        question_column = mapping.get("question_column", "question")
        label_column = mapping.get("label_column", "answers")
        words_column = mapping.get("words_column", "words")
        boxes_column = mapping.get("boxes_column", "boxes")
        image_column = mapping.get("image_column", "image")
        missing = [name for name in (question_column, label_column) if name not in columns]
        has_precomputed = words_column in columns and boxes_column in columns
        if missing or (not has_precomputed and image_column not in columns):
            details = ", ".join(missing) if missing else "words+boxes or image"
            raise DatasetValidationError(
                f"document question answering is missing required data: {details}; "
                f"dataset has {sorted(columns)}"
            )

    def _document_words_and_boxes(
        self,
        row: dict[str, Any],
    ) -> tuple[list[str], list[list[int]], str]:
        mapping = self.config.dataset.columns_mapping
        words_column = mapping.get("words_column", "words")
        boxes_column = mapping.get("boxes_column", "boxes")
        image_column = mapping.get("image_column", "image")
        image = row.get(image_column)
        image_size = getattr(image, "size", None)

        if words_column in row and boxes_column in row:
            words = [str(word) for word in row[words_column]]
            boxes = _normalize_boxes(
                row[boxes_column],
                image_size=image_size,
                coordinate_system=mapping.get("box_coords", "auto").casefold(),
            )
            source = "precomputed"
        else:
            if mapping.get("ocr_engine", "tesseract").casefold() != "tesseract":
                raise ValueError("ocr_engine currently supports only 'tesseract'.")
            try:
                import pytesseract  # type: ignore[import-untyped]
            except ImportError as error:
                raise RuntimeError(
                    "Image-only document QA requires the optional OCR dependency. "
                    "Install winml-cli[document] and native Tesseract, or provide words and boxes."
                ) from error
            if image is None or image_size is None:
                raise ValueError(
                    "Image-only document QA requires a PIL image with width and height."
                )
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            except pytesseract.TesseractNotFoundError as error:
                raise RuntimeError(
                    "Native Tesseract was not found. Install Tesseract or provide precomputed "
                    "words and boxes."
                ) from error
            words = []
            absolute_boxes = []
            for text, left, top, width, height in zip(
                data["text"],
                data["left"],
                data["top"],
                data["width"],
                data["height"],
                strict=True,
            ):
                text = str(text).strip()
                if not text:
                    continue
                words.append(text)
                absolute_boxes.append([left, top, left + width, top + height])
            boxes = _normalize_boxes(
                absolute_boxes,
                image_size=image_size,
                coordinate_system="absolute",
            )
            source = "tesseract"

        if not words or len(words) != len(boxes):
            raise ValueError("Document OCR must produce equal non-zero word and box counts.")
        return words, boxes, source

    @staticmethod
    def _references(value: Any) -> list[str]:
        if isinstance(value, dict):
            value = value.get("text", [])
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(reference) for reference in value]
        return []

    def _compute_document(self) -> dict[str, Any]:
        import torch

        from .metrics import ANLSMetric

        input_names = set((getattr(self.model, "io_config", None) or {}).get("input_names", []))
        if "bbox" not in input_names:
            raise ValueError(
                "Document question answering requires a model with a declared bbox input."
            )

        mapping = self.config.dataset.columns_mapping
        max_windows = _bounded_int(mapping, "max_windows", 1)
        doc_stride = _bounded_int(mapping, "doc_stride", 128, minimum=0)
        max_answer_words = _bounded_int(mapping, "max_answer_words", 64)
        top_k = _bounded_int(mapping, "top_k", 1)
        if top_k != 1:
            raise ValueError("Document QA evaluation currently requires top_k=1.")
        tokenizer = self.pipe.tokenizer
        if tokenizer is None:
            raise ValueError("Document question answering requires a tokenizer.")
        max_length = self._fixed_seq_length() or tokenizer.model_max_length
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("Document QA requires a finite tokenizer or model sequence length.")

        metric = ANLSMetric()
        windows_processed = 0
        ocr_samples = 0
        precomputed_samples = 0
        for row in self.data:
            words, boxes, source = self._document_words_and_boxes(row)
            ocr_samples += source == "tesseract"
            precomputed_samples += source == "precomputed"
            question = str(row[mapping.get("question_column", "question")])
            encoding = tokenizer(
                question.split(),
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
                    encoding,
                    feature_index,
                    boxes,
                    tokenizer.sep_token_id,
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
                device = getattr(self.pipe, "device", "cpu")
                model_inputs = {name: value.to(device) for name, value in model_inputs.items()}
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
            metric.update(prediction, self._references(row[mapping.get("label_column", "answers")]))

        return {
            **metric.compute(),
            "windows_processed": windows_processed,
            "ocr_samples": ocr_samples,
            "precomputed_samples": precomputed_samples,
        }

    def compute(self) -> dict[str, Any]:
        """Run QA evaluation with automatic SQuAD v2 detection.

        Detects whether the dataset has unanswerable questions (SQuAD v2
        format) and passes the correct metric and format flag to the HF
        QuestionAnsweringEvaluator.  Works with both SQuAD v1 (100 samples
        default) and v2 datasets — metric selection is automatic.
        """
        if self._is_document_mode():
            return self._compute_document()

        _ensure_evaluate_transformers_compat()
        from evaluate.evaluator.question_answering import QuestionAnsweringEvaluator

        logger.info("Running evaluation...")
        task_evaluator = QuestionAnsweringEvaluator(task=self.config.task)

        from ..utils.eval_utils import get_default

        mapping = self.config.dataset.columns_mapping
        label_col = mapping.get("label_column", get_default("question-answering", "label_column"))

        try:
            squad_v2 = task_evaluator.is_squad_v2_format(self.data, label_column=label_col)
        except Exception:
            logger.warning(
                "Could not detect SQuAD v2 format for column '%s'; defaulting to v1.",
                label_col,
            )
            squad_v2 = False

        metric_name = "squad_v2" if squad_v2 else "squad"
        logger.info("Using metric: %s (squad_v2_format=%s)", metric_name, squad_v2)

        kwargs: dict[str, Any] = {
            "model_or_pipeline": self.pipe,
            "data": self.data,
            "metric": metric_name,
            "squad_v2_format": squad_v2,
            **self.config.dataset.columns_mapping,
        }

        return cast("dict[str, Any]", task_evaluator.compute(**kwargs))
