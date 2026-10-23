# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Evaluator for fixed-vocabulary visual question answering."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

    from .config import DatasetConfig

logger = logging.getLogger(__name__)

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cannot": "can't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'll": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "im": "i'm",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shed've": "she'd've",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldve": "should've",
    "somebodyd": "somebody'd",
    "somebodyd've": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wed've": "we'd've",
    "well": "we'll",
    "werent": "weren't",
    "weve": "we've",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "yall'd've": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = r";/[\]\"{}()=+\_-><@`,?!"
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA = re.compile(r"(?<=\d),(?=\d)")


def normalize_vqa_answer(answer: str) -> str:
    """Apply the normalization used by the official VQA accuracy metric."""
    normalized = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    normalized = _COMMA.sub("", normalized)
    for punctuation in _PUNCTUATION:
        surrounded_by_space = (
            f"{punctuation} " in normalized or f" {punctuation}" in normalized
        )
        replacement = "" if surrounded_by_space else " "
        normalized = normalized.replace(punctuation, replacement)
    normalized = _PERIOD.sub("", normalized)
    words = []
    for word in normalized.split():
        word = _NUMBER_WORDS.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_soft_accuracy(prediction: str, answers: list[str]) -> float:
    """Return VQAv2 soft accuracy for one prediction and its annotator answers."""
    normalized_prediction = normalize_vqa_answer(prediction)
    matches = sum(normalize_vqa_answer(answer) == normalized_prediction for answer in answers)
    return min(matches / 3.0, 1.0)


def _answer_strings(value: Any) -> list[str]:
    """Extract answer strings from common VQAv2 row representations."""
    if isinstance(value, dict):
        value = value.get("answer", value.get("answers"))
    if not isinstance(value, list):
        return []
    answers: list[str] = []
    for item in value:
        answer = item.get("answer") if isinstance(item, dict) else item
        if isinstance(answer, str):
            answers.append(answer)
    return answers


class WinMLVisualQuestionAnsweringEvaluator(WinMLEvaluator):
    """Compute official soft accuracy for classification VQA predictions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from ..utils.eval_utils import get_default

        config = args[0] if args else kwargs["config"]
        mapping = config.dataset.columns_mapping
        self._image_col = mapping.get(
            "input_column", get_default("visual-question-answering", "input_column")
        )
        self._question_col = mapping.get(
            "question_column", get_default("visual-question-answering", "question_column")
        )
        self._label_col = mapping.get(
            "label_column", get_default("visual-question-answering", "label_column")
        )
        super().__init__(*args, **kwargs)

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Keep free-text answer annotations unchanged."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run one top-answer prediction per usable row and aggregate soft accuracy."""
        from tqdm.auto import tqdm

        rows: list[dict[str, Any]] = []
        skipped = 0
        for sample_index, sample in enumerate(tqdm(self.data, desc="Evaluating", unit="sample")):
            image = sample.get(self._image_col)
            question = sample.get(self._question_col)
            answers = _answer_strings(sample.get(self._label_col))
            if image is None or not isinstance(question, str) or not answers:
                skipped += 1
                continue
            try:
                output = self.pipe(image, question=question)
                if not isinstance(output, list) or not output or not isinstance(output[0], dict):
                    raise ValueError("Classification VQA pipeline must return ranked answer dicts.")
                prediction = output[0].get("answer")
                if not isinstance(prediction, str):
                    raise TypeError("Classification VQA pipeline returned no answer string.")
            except Exception as error:
                logger.warning("VQA pipeline call failed (skipping): %s", error)
                skipped += 1
                continue
            rows.append(
                {
                    "sample_index": sample_index,
                    "prediction": prediction,
                    "score": vqa_soft_accuracy(prediction, answers),
                }
            )

        if not rows:
            raise ValueError("No usable visual-question-answering samples were evaluated.")
        return {
            "vqa_accuracy": sum(row["score"] for row in rows) / len(rows),
            "n_samples": len(rows),
            "skipped": skipped,
            "predictions": rows,
        }


__all__ = [
    "WinMLVisualQuestionAnsweringEvaluator",
    "normalize_vqa_answer",
    "vqa_soft_accuracy",
]
