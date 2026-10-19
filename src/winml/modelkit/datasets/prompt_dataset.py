# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Prompt-corpus dataset for evaluators that iterate over text prompts.

``PromptDataset`` is a task-agnostic data source: unlike the calibration-
oriented ``TextDataset`` (which yields tokenized tensors bound to a
specific model) or the image-primary ``MaskGenerationDataset`` (which
carries a prompt column alongside images), ``PromptDataset``'s primary
yield is a bare prompt record. It is intended for evaluators that consume
prompts to produce something and score the output against a reference
(e.g. zero-shot classification with textual labels, VLM caption/QA
evaluation, retrieval, future generative-image workflows).

Records
-------

Each item yielded by ``PromptDataset`` is a plain ``dict`` with these keys::

    {
        "prompt":            str,                    # required
        "negative_prompt":   str | None,             # optional
        "reference_text":    str | None,             # optional
        "reference_image":   str | None,             # optional (path or URL)
        "metadata":          dict[str, Any],         # optional, arbitrary
    }

The dict shape (rather than a dataclass) matches the convention used by
sibling datasets in this package. A parallel :class:`PromptRecord`
dataclass is provided for callers that prefer a typed representation
during construction.

Loading
-------

Three constructors are supported::

    PromptDataset.from_list([{"prompt": "..."}, ...])
    PromptDataset.from_jsonl("path/to/prompts.jsonl")
    PromptDataset.from_hf("dataset/repo", split="test", prompt_col="prompt")

``from_hf`` also accepts a pre-loaded ``datasets.Dataset`` object in
place of the name, which makes it straightforward to test without hitting
the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from .base import BaseTaskDataset


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptRecord:
    """Typed record for a single prompt-corpus entry.

    Callers may construct :class:`PromptDataset` from a list of
    :class:`PromptRecord` instances (via :meth:`PromptDataset.from_list`)
    or from plain dicts \u2014 they are coerced to the same internal shape.
    """

    prompt: str
    negative_prompt: str | None = None
    reference_text: str | None = None
    reference_image: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a dict with all keys present."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "reference_text": self.reference_text,
            "reference_image": self.reference_image,
            "metadata": dict(self.metadata),
        }


_ALLOWED_KEYS = frozenset(
    {"prompt", "negative_prompt", "reference_text", "reference_image", "metadata"}
)


def _coerce_record(raw: Any) -> dict[str, Any]:
    """Coerce a raw input (dict or PromptRecord) into the canonical dict shape.

    Raises:
        ValueError: If ``prompt`` is missing / empty, or if unknown keys
            are present at the top level.
        TypeError: If ``raw`` is not a dict or PromptRecord.
    """
    if isinstance(raw, PromptRecord):
        return raw.to_dict()
    if not isinstance(raw, dict):
        raise TypeError(
            f"PromptDataset record must be a dict or PromptRecord, got {type(raw).__name__}"
        )

    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"PromptDataset record has unknown keys {sorted(unknown)}; "
            f"allowed keys are {sorted(_ALLOWED_KEYS)}",
        )

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            "PromptDataset record requires a non-empty string 'prompt' field",
        )

    metadata = raw.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        raise TypeError(
            f"PromptDataset record 'metadata' must be a dict, got {type(metadata).__name__}"
        )

    return {
        "prompt": prompt,
        "negative_prompt": raw.get("negative_prompt"),
        "reference_text": raw.get("reference_text"),
        "reference_image": raw.get("reference_image"),
        "metadata": dict(metadata),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PromptDataset(BaseTaskDataset):
    """Prompt-corpus dataset for evaluators.

    Not intended as a calibration dataset \u2014 use :class:`TextDataset` for
    that. Not registered in ``TASK_DATASET_MAPPING``.

    Typical usage::

        ds = PromptDataset.from_jsonl("prompts/drawbench.jsonl")
        for sample in ds:
            image = pipeline(sample["prompt"])
            score = clip_score(image, sample["reference_text"])
    """

    def __init__(
        self,
        records: Iterable[PromptRecord | dict[str, Any]],
        *,
        model_name: str | None = None,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize from an iterable of records.

        Args:
            records: Iterable of :class:`PromptRecord` or plain dicts.
                Each record is validated and coerced to the canonical shape.
            model_name: Optional. PromptDataset is task-agnostic and does
                not require a model. Accepted for API consistency with the
                base class.
            dataset_name: Optional label describing the prompt corpus
                (e.g. ``"drawbench"``); surfaced by ``dataset_name``.
            max_samples: Truncate the record list to at most this many
                entries. Applied after validation.
            data_split: Optional split label, purely informational.
            **kwargs: Forwarded to the base class.

        Raises:
            ValueError: If any record fails validation or if ``records``
                is empty.
        """
        # Coerce and validate before the base class initialises so that
        # _initialize can just assign the list.
        self._raw_records = list(records)

        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            **kwargs,
        )

    def _initialize(self) -> None:
        coerced = [_coerce_record(r) for r in self._raw_records]
        if not coerced:
            raise ValueError("PromptDataset requires at least one record")
        if self._max_samples is not None and self._max_samples < len(coerced):
            coerced = coerced[: self._max_samples]
        self._dataset = coerced
        self._metadata = {
            "num_records": len(coerced),
            "source": self._dataset_name,
        }
        # Release the pre-init buffer.
        del self._raw_records

    # ---------------------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        records: Iterable[PromptRecord | dict[str, Any]],
        **kwargs: Any,
    ) -> PromptDataset:
        """Construct from an in-memory iterable of records."""
        return cls(records, **kwargs)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> PromptDataset:
        """Construct from a JSONL file (one JSON object per line)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PromptDataset JSONL not found: {path}")
        logger.info("Loading prompt corpus from JSONL: %s", path)
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"PromptDataset JSONL parse error at {path}:{line_no}: {e.msg}",
                    ) from e
        kwargs.setdefault("dataset_name", str(path))
        return cls(records, **kwargs)

    @classmethod
    def from_hf(
        cls,
        dataset_or_name: Any,
        *,
        split: str = "train",
        prompt_col: str = "prompt",
        negative_prompt_col: str | None = None,
        reference_text_col: str | None = None,
        reference_image_col: str | None = None,
        metadata_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> PromptDataset:
        """Construct from a HuggingFace dataset repo name or a pre-loaded dataset.

        Accepts either a HF dataset repo id (calls ``load_dataset``) or an
        already-loaded ``datasets.Dataset`` object. The object path lets
        callers pre-load or mock without hitting the network (useful in tests).

        Args:
            dataset_or_name: Either a HF dataset repo id (``str``) or a
                pre-loaded ``datasets.Dataset``. Accepting the object
                directly lets callers pre-load or mock without invoking
                ``load_dataset`` (useful in tests).
            split: Split name (used only when passed a name string).
            prompt_col: Source column mapped to ``prompt``.
            negative_prompt_col, reference_text_col, reference_image_col:
                Optional source columns for the corresponding record fields.
            metadata_cols: Extra source columns to collect into ``metadata``.
            **kwargs: Forwarded to :class:`PromptDataset`.

        Raises:
            KeyError: If ``prompt_col`` (or any mapped column) is missing
                from the dataset schema.
        """
        if isinstance(dataset_or_name, str):
            from datasets import load_dataset

            logger.info(
                "Loading prompt corpus from HF dataset: %s (split=%s, prompt_col=%s)",
                dataset_or_name,
                split,
                prompt_col,
            )
            hf_dataset = load_dataset(dataset_or_name, split=split)
            kwargs.setdefault("dataset_name", dataset_or_name)
            kwargs.setdefault("data_split", split)
        else:
            hf_dataset = dataset_or_name

        col_names = set(getattr(hf_dataset, "column_names", []) or [])
        required_cols = [prompt_col]
        optional_col_map = {
            "negative_prompt": negative_prompt_col,
            "reference_text": reference_text_col,
            "reference_image": reference_image_col,
        }
        required_cols.extend(src for src in optional_col_map.values() if src is not None)
        required_cols.extend(metadata_cols or [])

        if col_names:
            missing = [c for c in required_cols if c not in col_names]
            if missing:
                raise KeyError(
                    f"PromptDataset.from_hf: missing columns {missing} "
                    f"in dataset with columns {sorted(col_names)}",
                )

        records: list[dict[str, Any]] = []
        for row in hf_dataset:
            record: dict[str, Any] = {"prompt": row[prompt_col]}
            for target, src in optional_col_map.items():
                if src is not None:
                    value = row.get(src) if hasattr(row, "get") else row[src]
                    record[target] = value
            if metadata_cols:
                record["metadata"] = {c: row[c] for c in metadata_cols}
            records.append(record)

        return cls(records, **kwargs)

    # ---------------------------------------------------------------------
    # BaseTaskDataset interface
    # ---------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the record at ``idx`` as a dict."""
        return dict(self._dataset[idx])

    @property
    def label_col(self) -> str:
        """PromptDataset is task-agnostic and has no fixed label column.

        Returns an empty string to satisfy the abstract-property contract.
        Callers should inspect ``reference_text`` / ``reference_image``
        on each record when a reference is available.
        """
        return ""

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the coerced record dicts."""
        return iter(self._dataset)

    # ---------------------------------------------------------------------
    # Derived-dataset operations (filter / sample)
    #
    # Design notes for future contributors
    # ------------------------------------
    # * These operations follow a common shape: transform the record list,
    #   then build a new dataset instance from the derived records.  The
    #   :meth:`_derive` hook is the single extension point -- override it in
    #   a subclass to change the resulting type, inject extra metadata, or
    #   customise how kwargs cascade.  ``filter`` and ``sample`` delegate to
    #   ``_derive`` so any new derived-dataset operation you add stays
    #   consistent with those two.
    # * All operations return a *new* dataset (never mutate ``self``) --
    #   this keeps ``PromptDataset`` safely reusable across evaluations and
    #   supports chaining (``ds.filter(...).sample(...)``).
    # * Adding a new derived operation (e.g. ``deduplicate``, ``group_by``)
    #   is a matter of one method that transforms ``self._dataset`` into a
    #   new record list and calls ``self._derive(new_records)``.
    # ---------------------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        **kwargs: Any,
    ) -> PromptDataset:
        """Return a new dataset keeping only records where ``predicate`` is true.

        Args:
            predicate: Callable receiving a record ``dict`` (a shallow copy;
                see note) and returning ``True`` to keep the record.
            **kwargs: Forwarded to the derived dataset's constructor.  By
                default the derived dataset inherits ``model_name``,
                ``dataset_name`` and ``data_split`` from ``self``; pass
                them explicitly here to override.

        Raises:
            ValueError: If the predicate rejects every record
                (:class:`PromptDataset` requires at least one).

        Note:
            The predicate receives a shallow copy of each record (matching
            ``__getitem__`` semantics), so mutating top-level fields is
            safe.  Mutating nested containers such as ``metadata`` still
            affects the source -- predicates should be read-only.
        """
        kept = [record for record in self._dataset if predicate(dict(record))]
        return self._derive(kept, **kwargs)

    def sample(
        self,
        n: int,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> PromptDataset:
        """Return a new dataset containing ``n`` randomly-drawn records.

        Args:
            n: Number of records to draw.  Must satisfy
                ``1 <= n <= len(self)`` -- oversampling is rejected because
                ``PromptDataset`` records are unique instances; use
                ``max_samples`` at construction time to enforce a size
                cap.
            seed: Optional integer for a reproducible draw.  ``None`` uses
                a fresh, non-deterministic RNG (matches
                :func:`random.Random` semantics).
            **kwargs: Forwarded to the derived dataset's constructor.

        Raises:
            ValueError: If ``n`` is outside ``[1, len(self)]``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n > len(self):
            raise ValueError(
                f"Cannot sample {n} records from a dataset of size {len(self)}; "
                "use max_samples at construction to cap size.",
            )
        rng = Random(seed)
        sampled = rng.sample(list(self._dataset), n)
        return self._derive(sampled, **kwargs)

    def _derive(
        self,
        records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> PromptDataset:
        """Construct a derived dataset from ``records`` inheriting metadata.

        Subclasses may override this to return their own type, inject
        additional metadata, or customise the cascade of default kwargs.
        The default preserves ``model_name`` / ``dataset_name`` /
        ``data_split`` from ``self`` unless the caller has passed an
        explicit override in ``kwargs``.

        Uses ``type(self)`` (not ``PromptDataset``) as the constructor so
        subclasses receive an instance of themselves.
        """
        kwargs.setdefault("model_name", self._model_name)
        kwargs.setdefault("dataset_name", self._dataset_name)
        kwargs.setdefault("data_split", self._data_split)
        return type(self)(records, **kwargs)
