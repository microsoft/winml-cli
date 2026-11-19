# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Prompt datasets for ``winml perf --endpoint --dataset``.

A prompt dataset is any table ``datasets`` can load: a Hugging Face Hub
dataset, a folder written by ``Dataset.save_to_disk``, or a local folder or
file of ``.jsonl``, ``.json``, ``.parquet`` or ``.csv`` rows. The options and
the loading path follow ``winml eval``; the columns ``winml perf`` reads are
described by :data:`PROMPT_DATASET_SCHEMA` and remapped with ``--column``:

* ``input_column`` (default ``prompt``): a string, a list of turns (the first
  turn is sent as the user message), or a list of ``{role, content}`` messages
  (sent as they are);
* ``id_column`` (default ``id``): the prompt id recorded per request; the row
  index when the dataset has no such column;
* ``cohort_column`` (no default): comma-separated columns whose values group
  prompts into cohorts, each column a separate grouping.

``datasets`` is imported only when a dataset is loaded.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.eval_utils import SchemaItem, TaskSchema


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# Recorded in the report so a shuffled selection can be reproduced.
SHUFFLE_SEED = 42

PROMPT_DATASET_SCHEMA = TaskSchema(
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

_FILE_BUILDERS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".csv": "csv"}
# Files written by Dataset.save_to_disk / DatasetDict.save_to_disk.
_SAVED_DATASET_MARKERS = ("state.json", "dataset_dict.json")


class PromptDatasetError(ValueError):
    """The prompt dataset could not be loaded or does not fit the schema."""


@dataclass
class PromptDatasetConfig:
    """Which dataset rows to send, named like ``winml eval``'s ``DatasetConfig``.

    Attributes:
        path: A Hugging Face dataset id, or a local folder or file.
        name: Config name for multi-config Hub datasets.
        revision: Git revision of a Hub dataset.
        split: Split to read; ``None`` uses the only split.
        samples: Rows to send; ``None`` sends every row.
        shuffle: Shuffle the rows with ``seed`` before taking ``samples``.
        seed: Shuffle seed.
        columns_mapping: ``--column`` overrides of :data:`PROMPT_DATASET_SCHEMA`.
    """

    path: str
    name: str | None = None
    revision: str | None = None
    split: str | None = None
    samples: int | None = None
    shuffle: bool = False
    seed: int = SHUFFLE_SEED
    columns_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetPrompt:
    """One selected row, ready to send."""

    row_index: int
    prompt_id: str
    messages: list[dict[str, Any]]
    cohorts: dict[str, str]


@dataclass
class PromptDataset:
    """The selected prompts of a loaded dataset and how they were chosen."""

    config: PromptDatasetConfig
    source: str
    display_path: str
    split: str | None
    row_count: int
    input_column: str
    id_column: str | None
    cohort_columns: list[str]
    prompts: list[DatasetPrompt]

    @property
    def prompts_sha256(self) -> str:
        """SHA-256 over the selected prompt ids and messages, in order."""
        payload = [{"id": prompt.prompt_id, "messages": prompt.messages} for prompt in self.prompts]
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the report; a local path is reduced to its last component."""
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


def validate_column_keys(columns_mapping: Mapping[str, str]) -> None:
    """Reject ``--column`` keys the prompt schema does not define."""
    known = [item.name for item in PROMPT_DATASET_SCHEMA.columns]
    unknown = sorted(set(columns_mapping) - set(known))
    if unknown:
        raise PromptDatasetError(
            f"Unknown --column key(s): {', '.join(unknown)}. Valid keys: {', '.join(known)}."
        )


def _default(name: str) -> str | None:
    return next(item.default for item in PROMPT_DATASET_SCHEMA.columns if item.name == name)


def resolve_columns(
    columns_mapping: Mapping[str, str], column_names: Sequence[str]
) -> tuple[str, str | None, list[str]]:
    """Return the prompt column, the id column (or ``None``) and the cohort columns.

    Raises:
        PromptDatasetError: A key is unknown or a named column is missing.
    """
    validate_column_keys(columns_mapping)
    available = ", ".join(column_names) or "none"

    def missing(column: str, key: str) -> PromptDatasetError:
        return PromptDatasetError(
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
        raise PromptDatasetError("--column cohort_column needs at least one column name.")
    return input_column, id_column, cohort_columns


def normalize_prompt(value: object, row_index: int) -> list[dict[str, Any]]:
    """Turn a prompt cell into chat messages.

    A string becomes one user message; a list of strings (turns) sends its
    first turn as the user message; a list of ``{role, content}`` objects is
    sent as it is.

    Raises:
        PromptDatasetError: The prompt is empty or has another shape.
    """

    def fail(reason: str) -> PromptDatasetError:
        return PromptDatasetError(f"Row {row_index}: the prompt {reason}.")

    if isinstance(value, str):
        if not value.strip():
            raise fail("is empty")
        return [{"role": "user", "content": value}]
    if isinstance(value, list | tuple):
        if not value:
            raise fail("is an empty list")
        if all(isinstance(turn, str) for turn in value):
            return normalize_prompt(value[0], row_index)
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


def select_rows(row_count: int, *, samples: int | None, shuffle: bool, seed: int) -> list[int]:
    """Return the row indices to send, in sending order."""
    indices = list(range(row_count))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return indices if samples is None else indices[:samples]


def build_prompts(
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    *,
    input_column: str,
    id_column: str | None,
    cohort_columns: Sequence[str],
) -> list[DatasetPrompt]:
    """Turn selected rows into prompts; ``rows[i]`` is row ``row_indices[i]``.

    Raises:
        PromptDatasetError: A prompt is invalid, an id is missing, or ids repeat.
    """
    prompts: list[DatasetPrompt] = []
    seen: dict[str, int] = {}
    for row_index, row in zip(row_indices, rows, strict=True):
        messages = normalize_prompt(row.get(input_column), row_index)
        if id_column is None:
            prompt_id = str(row_index)
        else:
            raw_id = row.get(id_column)
            if raw_id is None or raw_id == "":
                raise PromptDatasetError(f"Row {row_index}: the id column '{id_column}' is empty.")
            prompt_id = _cell_text(raw_id)
        if prompt_id in seen:
            raise PromptDatasetError(
                f"Rows {seen[prompt_id]} and {row_index} share the prompt id '{prompt_id}'. "
                "Choose a unique column with --column id_column=<name>."
            )
        seen[prompt_id] = row_index
        cohorts = {column: _cell_text(row.get(column)) for column in cohort_columns}
        prompts.append(DatasetPrompt(row_index, prompt_id, messages, cohorts))
    return prompts


# =============================================================================
# Loading
# =============================================================================


def _pick_split(loaded: Any, split: str | None) -> tuple[Any, str | None]:
    """Return one split of a ``Dataset`` or ``DatasetDict`` and its name."""
    if not hasattr(loaded, "keys"):  # a single Dataset
        actual = getattr(loaded, "split", None)
        return loaded, split or (str(actual) if actual else None)
    names = list(loaded.keys())
    if split is not None:
        if split not in names:
            raise PromptDatasetError(
                f"The dataset has no split '{split}'. Available splits: {', '.join(names)}."
            )
        return loaded[split], split
    if len(names) == 1:
        return loaded[names[0]], names[0]
    raise PromptDatasetError(
        f"The dataset has several splits; choose one with --split: {', '.join(names)}."
    )


def _load_local(path: Path, config: PromptDatasetConfig) -> Any:
    import datasets

    if config.revision is not None:
        raise PromptDatasetError("--dataset-revision applies only to Hugging Face Hub datasets.")
    if path.is_dir() and any((path / marker).exists() for marker in _SAVED_DATASET_MARKERS):
        if config.name is not None:
            raise PromptDatasetError("--dataset-name does not apply to a saved dataset folder.")
        return datasets.load_from_disk(str(path))
    if path.is_file():
        builder = _FILE_BUILDERS.get(path.suffix.lower())
        if builder is None:
            raise PromptDatasetError(
                f"'{path.name}' is not a {', '.join(sorted(_FILE_BUILDERS))} file."
            )
        return datasets.load_dataset(builder, data_files=str(path))
    return datasets.load_dataset(str(path), name=config.name)


def load_prompt_dataset(config: PromptDatasetConfig) -> PromptDataset:
    """Load the dataset, map its columns and select the prompts to send.

    Raises:
        PromptDatasetError: The dataset could not be loaded or does not fit
            the prompt schema.
    """
    validate_column_keys(config.columns_mapping)
    import datasets

    # The CLI shows its own progress; keep the library's bars out of the report.
    bars_enabled = datasets.is_progress_bar_enabled()
    datasets.disable_progress_bars()
    try:
        return _load_prompt_dataset(config)
    finally:
        if bars_enabled:
            datasets.enable_progress_bars()


def _load_prompt_dataset(config: PromptDatasetConfig) -> PromptDataset:
    local = Path(config.path).expanduser()
    try:
        if local.exists():
            loaded = _load_local(local, config)
            source, display_path = "local", local.resolve().name or config.path
        else:
            import datasets

            loaded = datasets.load_dataset(
                config.path, name=config.name, revision=config.revision, split=config.split
            )
            source, display_path = "hub", config.path
    except PromptDatasetError:
        raise
    except Exception as exc:  # datasets raises many types for unreadable data
        raise PromptDatasetError(f"Could not load the dataset '{config.path}': {exc}") from exc
    dataset, split = _pick_split(loaded, config.split)

    column_names = list(dataset.column_names or [])
    input_column, id_column, cohort_columns = resolve_columns(config.columns_mapping, column_names)
    row_count = len(dataset)
    if row_count == 0:
        raise PromptDatasetError("The dataset split has no rows.")
    indices = select_rows(
        row_count, samples=config.samples, shuffle=config.shuffle, seed=config.seed
    )
    wanted = [input_column, *([id_column] if id_column else []), *cohort_columns]
    rows = dataset.select_columns(list(dict.fromkeys(wanted))).select(indices)
    prompts = build_prompts(
        list(rows),
        indices,
        input_column=input_column,
        id_column=id_column,
        cohort_columns=cohort_columns,
    )
    return PromptDataset(
        config=config,
        source=source,
        display_path=display_path,
        split=split,
        row_count=row_count,
        input_column=input_column,
        id_column=id_column,
        cohort_columns=cohort_columns,
        prompts=prompts,
    )
