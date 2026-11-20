# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Load one split of a dataset described by a ``DatasetConfig``.

:func:`load_split` reads a Hugging Face Hub dataset, a folder written by
``Dataset.save_to_disk``, or a local folder or file of ``.jsonl``, ``.json``,
``.parquet`` or ``.csv`` rows. :func:`select_indices` picks the rows to use the
way ``winml eval`` does, so a seed selects the same rows in every command.

``datasets`` is imported only when a dataset is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..utils.eval_utils import DatasetValidationError


if TYPE_CHECKING:
    from ..eval import DatasetConfig

_FILE_BUILDERS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".csv": "csv"}
# Files written by Dataset.save_to_disk / DatasetDict.save_to_disk.
_SAVED_DATASET_MARKERS = ("state.json", "dataset_dict.json")


@dataclass
class LoadedSplit:
    """One loaded split and where it came from.

    Attributes:
        dataset: The ``datasets.Dataset`` of the split.
        split: The split name, when known.
        source: ``hub`` for a Hugging Face Hub dataset, ``local`` for a path.
        display_path: The Hub dataset id, or only the last component of a
            local path, so reports carry no local folders.
    """

    dataset: Any
    split: str | None
    source: Literal["hub", "local"]
    display_path: str


def select_indices(row_count: int, *, samples: int | None, shuffle: bool, seed: int) -> list[int]:
    """Return the indices of the rows to use, in order.

    With ``shuffle`` the rows are permuted as ``Dataset.shuffle(seed=seed)``
    permutes them, then the first ``samples`` are taken, matching the
    ``shuffle(seed).select(range(samples))`` selection of ``winml eval``.
    """
    indices = (
        [int(index) for index in np.random.default_rng(seed).permutation(row_count)]
        if shuffle
        else list(range(row_count))
    )
    return indices if samples is None else indices[:samples]


def _pick_split(loaded: Any, split: str | None) -> tuple[Any, str | None]:
    """Return one split of a ``Dataset`` or ``DatasetDict`` and its name."""
    if not hasattr(loaded, "keys"):  # a single Dataset
        actual = getattr(loaded, "split", None)
        return loaded, split or (str(actual) if actual else None)
    names = list(loaded.keys())
    if split is not None:
        if split not in names:
            raise DatasetValidationError(
                f"The dataset has no split '{split}'. Available splits: {', '.join(names)}."
            )
        return loaded[split], split
    if len(names) == 1:
        return loaded[names[0]], names[0]
    raise DatasetValidationError(
        f"The dataset has several splits; choose one with --split: {', '.join(names)}."
    )


def _load_local(path: Path, config: DatasetConfig) -> Any:
    import datasets

    if config.revision is not None:
        raise DatasetValidationError(
            "--dataset-revision applies only to Hugging Face Hub datasets."
        )
    if path.is_dir() and any((path / marker).exists() for marker in _SAVED_DATASET_MARKERS):
        if config.name is not None:
            raise DatasetValidationError("--dataset-name does not apply to a saved dataset folder.")
        return datasets.load_from_disk(str(path))
    if path.is_file():
        builder = _FILE_BUILDERS.get(path.suffix.lower())
        if builder is None:
            raise DatasetValidationError(
                f"'{path.name}' is not a {', '.join(sorted(_FILE_BUILDERS))} file."
            )
        return datasets.load_dataset(builder, data_files=str(path))
    return datasets.load_dataset(str(path), name=config.name)


def load_split(config: DatasetConfig) -> LoadedSplit:
    """Load the split ``config`` names.

    A local path is a ``save_to_disk`` folder (detected by its metadata files),
    a data file, or a folder of data files; anything else is a Hub dataset id.
    With ``config.split`` of ``None`` the dataset must have exactly one split.

    Raises:
        DatasetValidationError: The dataset could not be loaded, or the split
            is missing or ambiguous.
    """
    if not config.path:
        raise DatasetValidationError("No dataset path was given.")
    local = Path(config.path).expanduser()
    try:
        if local.exists():
            loaded = _load_local(local, config)
            source: Literal["hub", "local"] = "local"
            display_path = local.resolve().name or config.path
        else:
            import datasets

            loaded = datasets.load_dataset(
                config.path, name=config.name, revision=config.revision, split=config.split
            )
            source, display_path = "hub", config.path
    except DatasetValidationError:
        raise
    except Exception as exc:  # datasets raises many types for unreadable data
        raise DatasetValidationError(f"Could not load the dataset '{config.path}': {exc}") from exc
    dataset, split = _pick_split(loaded, config.split)
    return LoadedSplit(dataset=dataset, split=split, source=source, display_path=display_path)
