# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""In-house implementation of the CIDEr image-captioning metric.

Reference:
    Vedantam, Zitnick, Parikh. "CIDEr: Consensus-based Image Description
    Evaluation." CVPR 2015. arXiv:1411.5726.

Exposes a :class:`Cider` class whose :meth:`Cider.compute_score`
accepts the standard ``(gts, res)`` dict-of-lists corpus format used
throughout the image-captioning literature.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


# Paper defaults. Tests pin these — do not change without a test update.
_NGRAM_ORDER = 4
_LENGTH_SIGMA = 6.0
_SCORE_SCALE = 10.0


# Internal type aliases for readability only.
_Ngram = tuple[str, ...]
_NgramCounts = Counter[_Ngram]
_TfidfVector = dict[_Ngram, float]


def _tokenize(sentence: str) -> list[str]:
    """Whitespace tokenization.

    Callers handle any required upstream normalization (case folding,
    punctuation stripping). Keeping tokenization trivial here makes the
    metric deterministic and free of external tokenizer dependencies.
    """
    return sentence.split()


def _collect_ngrams(tokens: Sequence[str], order_max: int) -> _NgramCounts:
    """Return n-gram counts for orders 1..``order_max`` over ``tokens``."""
    counts: _NgramCounts = Counter()
    n_tokens = len(tokens)
    for order in range(1, order_max + 1):
        last_start = n_tokens - order + 1
        for start in range(last_start):
            counts[tuple(tokens[start : start + order])] += 1
    return counts


def _build_doc_frequency(
    ref_ngrams_per_image: Sequence[Sequence[_NgramCounts]],
) -> Counter[_Ngram]:
    """Document frequency where each *image* (reference set) is one document.

    Per Vedantam et al., the document unit is an image, not an individual
    caption. An n-gram appearing in any of an image's references contributes
    exactly 1 to that image's document-frequency tally.
    """
    df: Counter[_Ngram] = Counter()
    for refs in ref_ngrams_per_image:
        seen_in_image: set[_Ngram] = set()
        for ref in refs:
            seen_in_image.update(ref.keys())
        for ngram in seen_in_image:
            df[ngram] += 1
    return df


def _tfidf_split_by_order(
    sentence_ngrams: _NgramCounts,
    doc_freq: Counter[_Ngram],
    log_total_docs: float,
    order_max: int,
) -> tuple[list[_TfidfVector], list[float], int]:
    """Split a sentence's TF-IDF weights by n-gram order.

    Returns:
        - ``vectors``: ``order_max`` dicts; ``vectors[k]`` maps each
          (k+1)-gram to its TF-IDF weight in the sentence.
        - ``norms``: L2 norm of each per-order vector.
        - ``token_length``: total unigram count (used for the length
          penalty in :func:`_pair_similarity`).
    """
    vectors: list[_TfidfVector] = [{} for _ in range(order_max)]
    sq_norms = [0.0] * order_max
    token_length = 0

    for ngram, term_freq in sentence_ngrams.items():
        order_idx = len(ngram) - 1
        df = doc_freq.get(ngram, 0)
        # Floor df at 1 so the idf factor stays defined and non-negative.
        if df < 1:
            df = 1
        idf = log_total_docs - math.log(df)
        weight = term_freq * idf
        vectors[order_idx][ngram] = weight
        sq_norms[order_idx] += weight * weight
        if order_idx == 0:
            token_length += term_freq

    norms = [math.sqrt(s) for s in sq_norms]
    return vectors, norms, token_length


def _pair_similarity(
    cand_vectors: list[_TfidfVector],
    ref_vectors: list[_TfidfVector],
    cand_norms: list[float],
    ref_norms: list[float],
    cand_length: int,
    ref_length: int,
    order_max: int,
    sigma: float,
) -> list[float]:
    """Per-order CIDEr similarity for one (candidate, reference) pair.

    For each n-gram order:

      1. **Clipped cosine similarity.** Each shared n-gram's contribution
         is ``min(cand_weight, ref_weight) * ref_weight``. The clip prevents
         a candidate from over-rewarding itself by repeating a single
         high-IDF n-gram.

      2. **Gaussian length penalty**
         ``exp(-(|cand| - |ref|)^2 / (2 * sigma^2))``. Multiplicative.
         This is the "-D" variant's defense against length-based gaming.
    """
    delta = cand_length - ref_length
    length_penalty = math.exp(-(delta * delta) / (2.0 * sigma * sigma))

    similarities: list[float] = []
    for k in range(order_max):
        cand_vec = cand_vectors[k]
        ref_vec = ref_vectors[k]

        # Walk the smaller vector to compute the dot product over shared
        # n-grams only; n-grams not in the other side contribute zero.
        if len(cand_vec) <= len(ref_vec):
            walk, lookup = cand_vec, ref_vec
            walk_is_cand = True
        else:
            walk, lookup = ref_vec, cand_vec
            walk_is_cand = False

        dot_product = 0.0
        for ngram, walk_weight in walk.items():
            other_weight = lookup.get(ngram)
            if other_weight is None:
                continue
            if walk_is_cand:
                cand_weight, ref_weight = walk_weight, other_weight
            else:
                cand_weight, ref_weight = other_weight, walk_weight
            dot_product += min(cand_weight, ref_weight) * ref_weight

        denom = cand_norms[k] * ref_norms[k]
        cosine_sim = (dot_product / denom) if denom > 0.0 else 0.0
        similarities.append(cosine_sim * length_penalty)

    return similarities


class Cider:
    """Compute the CIDEr image-captioning metric.

    Example::

        cider = Cider()
        score, per_sample = cider.compute_score(gts, res)

    Args:
        n: Maximum n-gram order. Paper default is 4.
        sigma: Standard deviation of the length-penalty Gaussian. Paper
            default is 6.0.
    """

    def __init__(self, n: int = _NGRAM_ORDER, sigma: float = _LENGTH_SIGMA) -> None:
        self._order_max = n
        self._sigma = sigma

    def compute_score(
        self,
        gts: dict[Hashable, list[str]],
        res: dict[Hashable, list[str]],
    ) -> tuple[float, np.ndarray]:
        """Score a corpus of (candidate, references) pairs.

        Args:
            gts: Ground-truth references, keyed by sample id. Each value
                is a list of one or more reference captions.
            res: Candidate captions, keyed by sample id. Each value is a
                list holding exactly one candidate caption.

        Returns:
            ``(corpus_score, per_sample_scores)`` where ``corpus_score``
            is the unweighted mean of the per-sample scores and
            ``per_sample_scores`` is a 1-D ``numpy`` array aligned with
            ``sorted(gts.keys())``-style iteration of ``gts``.

        Raises:
            ValueError: If ``gts`` and ``res`` don't share the same keys,
                if any ``res`` value lacks exactly one candidate, or if
                the corpus has fewer than 2 samples (IDF is undefined for
                a single-document corpus).
        """
        if gts.keys() != res.keys():
            missing_in_res = set(gts) - set(res)
            missing_in_gts = set(res) - set(gts)
            raise ValueError(
                "gts and res must share identical keys. "
                f"missing in res: {sorted(map(str, missing_in_res))}; "
                f"missing in gts: {sorted(map(str, missing_in_gts))}."
            )

        ids = list(gts.keys())
        n_images = len(ids)
        if n_images < 2:
            raise ValueError(
                "CIDEr requires at least 2 samples for a well-defined IDF; "
                f"received {n_images}."
            )

        # Unpack predictions and references in a stable order.
        predictions: list[str] = []
        references: list[list[str]] = []
        for sample_id in ids:
            cand_list = res[sample_id]
            if len(cand_list) != 1:
                raise ValueError(
                    f"res[{sample_id!r}] must hold exactly one candidate; "
                    f"got {len(cand_list)}."
                )
            predictions.append(cand_list[0])
            references.append(list(gts[sample_id]))

        return self._score(predictions, references)

    # --- internals -----------------------------------------------------

    def _score(
        self,
        predictions: Sequence[str],
        references: Sequence[Sequence[str]],
    ) -> tuple[float, np.ndarray]:
        order_max = self._order_max
        sigma = self._sigma

        # Stage 1 — tokenize and count n-grams once per sentence.
        cand_ngrams = [
            _collect_ngrams(_tokenize(pred), order_max) for pred in predictions
        ]
        ref_ngrams = [
            [_collect_ngrams(_tokenize(r), order_max) for r in refs]
            for refs in references
        ]

        # Stage 2 — document frequency over reference images.
        doc_freq = _build_doc_frequency(ref_ngrams)
        log_total_docs = math.log(float(len(predictions)))

        # Stage 3 — per-sample scores.
        per_sample: list[float] = []
        for cand_counts, ref_counts_list in zip(cand_ngrams, ref_ngrams, strict=True):
            n_refs = len(ref_counts_list)
            if n_refs == 0:
                per_sample.append(0.0)
                continue

            cand_vecs, cand_norms, cand_len = _tfidf_split_by_order(
                cand_counts, doc_freq, log_total_docs, order_max
            )

            per_order_sum = [0.0] * order_max
            for ref_counts in ref_counts_list:
                ref_vecs, ref_norms, ref_len = _tfidf_split_by_order(
                    ref_counts, doc_freq, log_total_docs, order_max
                )
                sims = _pair_similarity(
                    cand_vecs,
                    ref_vecs,
                    cand_norms,
                    ref_norms,
                    cand_len,
                    ref_len,
                    order_max,
                    sigma,
                )
                for k in range(order_max):
                    per_order_sum[k] += sims[k]

            score = sum(per_order_sum) / order_max / n_refs * _SCORE_SCALE
            per_sample.append(score)

        corpus_score = sum(per_sample) / len(per_sample)
        return corpus_score, np.asarray(per_sample, dtype=float)

    @property
    def n(self) -> int:
        """Maximum n-gram order."""
        return self._order_max

    def method(self) -> str:
        """Metric label."""
        return "CIDEr"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={self._order_max}, "
            f"sigma={self._sigma})"
        )


__all__ = ["Cider"]
