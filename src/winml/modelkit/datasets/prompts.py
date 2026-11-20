# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Text generation prompts read from a dataset.

:func:`load_prompts` loads a split with :func:`~.loading.load_split`, selects
rows with :func:`~.loading.select_indices`, and turns each row into a
:class:`GenerationPrompt`. The columns it reads are described by
:data:`PROMPT_SCHEMA` and remapped with ``--column``:

* ``input_column`` (default ``prompt``): a string, a list of turns (the first
  turn is sent as the user message), or a list of ``{role, content}`` messages
  (sent as they are);
* ``id_column`` (default ``id``): the prompt id recorded per request; the row
  index when the dataset has no such column;
* ``cohort_column`` (no default): comma-separated columns whose values group
  prompts into cohorts, each column a separate grouping.

Prompts keep their text: the runtime that generates from them applies the
chat template and tokenizes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils.eval_utils import DatasetValidationError, SchemaItem, TaskSchema
from .loading import load_split, select_indices


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..eval import DatasetConfig  # noqa: TC004


PROMPT_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "prompt: a string, a list of turns (the first turn is sent), or a list of "
            "{role, content} messages",
            default="prompt",
            remap_hint="<your_prompt_column>",
        ),
        SchemaItem(
            "id_column",
            "prompt id recorded per request; the row index when the column is absent",
            default="id",
            remap_hint="<your_id_column>",
        ),
        SchemaItem(
            "cohort_column",
            "comma-separated columns whose values group prompts into cohorts",
            default=None,
            remap_hint="<column>[,<column>...]",
        ),
    )
)


@dataclass(frozen=True)
class GenerationPrompt:
    """One selected row, ready to generate from."""

    row_index: int
    prompt_id: str
    messages: list[dict[str, Any]]
    cohorts: dict[str, str]


@dataclass
class PromptSet:
    """The selected prompts of a dataset and how they were chosen."""

    config: DatasetConfig
    source: str
    display_path: str
    split: str | None
    row_count: int
    input_column: str
    id_column: str | None
    cohort_columns: list[str]
    prompts: list[GenerationPrompt]

    @property
    def prompts_sha256(self) -> str:
        """SHA-256 over the selected prompt ids and messages, in order."""
        payload = [{"id": prompt.prompt_id, "messages": prompt.messages} for prompt in self.prompts]
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for a report; a local path is reduced to its last component."""
        cfg = self.config
        return {
            "source": self.source,
            "path": self.display_path,
            "name": cfg.name,
            "revision": cfg.revision,
            "split": self.split,
            "samples": cfg.samples,
            "shuffle": cfg.shuffle,
            "seed": cfg.seed,
            "column_mapping": {
                "input_column": self.input_column,
                "id_column": self.id_column,
                "cohort_column": list(self.cohort_columns),
            },
            "row_count": self.row_count,
            "selected_row_count": len(self.prompts),
            "cohort_columns": list(self.cohort_columns),
            "prompts_sha256": self.prompts_sha256,
        }


# =============================================================================
# Columns and prompts
# =============================================================================


def _validate_column_keys(columns_mapping: Mapping[str, str]) -> None:
    """Reject ``--column`` keys the prompt schema does not define."""
    known = [item.name for item in PROMPT_SCHEMA.columns]
    unknown = sorted(set(columns_mapping) - set(known))
    if unknown:
        raise DatasetValidationError(
            f"Unknown --column key(s): {', '.join(unknown)}. Valid keys: {', '.join(known)}."
        )


def _default(name: str) -> str | None:
    return next(item.default for item in PROMPT_SCHEMA.columns if item.name == name)


def _resolve_columns(
    columns_mapping: Mapping[str, str], column_names: Sequence[str]
) -> tuple[str, str | None, list[str]]:
    """Return the prompt column, the id column (or ``None``) and the cohort columns.

    Raises:
        DatasetValidationError: A key is unknown or a named column is missing.
    """
    _validate_column_keys(columns_mapping)
    available = ", ".join(column_names) or "none"

    def missing(column: str, key: str) -> DatasetValidationError:
        return DatasetValidationError(
            f"The dataset has no column '{column}' ({key}). Available columns: {available}. "
            f"Set it with --column {key}=<name>."
        )

    input_column = columns_mapping.get("input_column") or _default("input_column") or ""
    if input_column not in column_names:
        raise missing(input_column, "input_column")
    id_column: str | None = columns_mapping.get("id_column") or _default("id_column")
    if id_column not in column_names:
        if "id_column" in columns_mapping:
            raise missing(str(id_column), "id_column")
        id_column = None
    cohort_columns: list[str] = []
    for column in (columns_mapping.get("cohort_column") or "").split(","):
        column = column.strip()
        if not column or column in cohort_columns:
            continue
        if column not in column_names:
            raise missing(column, "cohort_column")
        cohort_columns.append(column)
    if "cohort_column" in columns_mapping and not cohort_columns:
        raise DatasetValidationError("--column cohort_column needs at least one column name.")
    return input_column, id_column, cohort_columns


def _normalize_prompt(value: object, row_index: int) -> list[dict[str, Any]]:
    """Turn a prompt cell into chat messages.

    A string becomes one user message; a list of strings (turns) sends its
    first turn as the user message; a list of ``{role, content}`` objects is
    sent as it is.

    Raises:
        DatasetValidationError: The prompt is empty or has another shape.
    """

    def fail(reason: str) -> DatasetValidationError:
        return DatasetValidationError(f"Row {row_index}: the prompt {reason}.")

    if isinstance(value, str):
        if not value.strip():
            raise fail("is empty")
        return [{"role": "user", "content": value}]
    if isinstance(value, list | tuple):
        if not value:
            raise fail("is an empty list")
        if all(isinstance(turn, str) for turn in value):
            return _normalize_prompt(value[0], row_index)
        if all(isinstance(message, dict) for message in value):
            messages = [dict(message) for message in value]
            for message in messages:
                if not isinstance(message.get("role"), str) or message.get("content") in (
                    None,
                    "",
                ):
                    raise fail("has a message without a 'role' string and 'content'")
            return messages
        raise fail("mixes turns and messages")
    if value is None:
        raise fail("is missing")
    raise fail(f"is a {type(value).__name__}, not a string, a list of turns or messages")


def _cell_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "none"
    if isinstance(value, bool | int | float):
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _build_prompts(
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    *,
    input_column: str,
    id_column: str | None,
    cohort_columns: Sequence[str],
) -> list[GenerationPrompt]:
    """Turn selected rows into prompts; ``rows[i]`` is row ``row_indices[i]``.

    Raises:
        DatasetValidationError: A prompt is invalid, an id is missing, or ids repeat.
    """
    prompts: list[GenerationPrompt] = []
    seen: dict[str, int] = {}
    for row_index, row in zip(row_indices, rows, strict=True):
        messages = _normalize_prompt(row.get(input_column), row_index)
        if id_column is None:
            prompt_id = str(row_index)
        else:
            raw_id = row.get(id_column)
            if raw_id is None or raw_id == "":
                raise DatasetValidationError(
                    f"Row {row_index}: the id column '{id_column}' is empty."
                )
            prompt_id = _cell_text(raw_id)
        if prompt_id in seen:
            raise DatasetValidationError(
                f"Rows {seen[prompt_id]} and {row_index} share the prompt id '{prompt_id}'. "
                "Choose a unique column with --column id_column=<name>."
            )
        seen[prompt_id] = row_index
        cohorts = {column: _cell_text(row.get(column)) for column in cohort_columns}
        prompts.append(GenerationPrompt(row_index, prompt_id, messages, cohorts))
    return prompts


# =============================================================================
# Loading
# =============================================================================


def load_prompts(config: DatasetConfig) -> PromptSet:
    """Load the dataset, map its columns and select the prompts.

    ``config.columns_mapping`` holds the :data:`PROMPT_SCHEMA` overrides.

    Raises:
        DatasetValidationError: The dataset could not be loaded or does not
            fit the prompt schema.
    """
    _validate_column_keys(config.columns_mapping)
    import datasets

    # Callers show their own progress; keep the library's bars out of reports.
    bars_enabled = datasets.is_progress_bar_enabled()
    datasets.disable_progress_bars()
    try:
        return _load_prompts(config)
    finally:
        if bars_enabled:
            datasets.enable_progress_bars()


def _load_prompts(config: DatasetConfig) -> PromptSet:
    loaded = load_split(config)
    dataset = loaded.dataset
    column_names = list(dataset.column_names or [])
    input_column, id_column, cohort_columns = _resolve_columns(config.columns_mapping, column_names)
    row_count = len(dataset)
    if row_count == 0:
        raise DatasetValidationError("The dataset split has no rows.")
    indices = select_indices(
        row_count, samples=config.samples, shuffle=config.shuffle, seed=config.seed
    )
    wanted = [input_column, *([id_column] if id_column else []), *cohort_columns]
    rows = dataset.select_columns(list(dict.fromkeys(wanted))).select(indices)
    prompts = _build_prompts(
        list(rows),
        indices,
        input_column=input_column,
        id_column=id_column,
        cohort_columns=cohort_columns,
    )
    return PromptSet(
        config=config,
        source=loaded.source,
        display_path=loaded.display_path,
        split=loaded.split,
        row_count=row_count,
        input_column=input_column,
        id_column=id_column,
        cohort_columns=cohort_columns,
        prompts=prompts,
    )
