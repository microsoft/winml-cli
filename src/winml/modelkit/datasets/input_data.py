# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Real input tensors loaded from a ``.npz`` archive.

Shared by ``winml perf`` (benchmark on real tensors instead of random ones)
and ``winml eval --mode compare`` (compare a candidate and reference on the
same real inputs). :func:`load_input_data` validates and dtype-casts the
archive against a model's I/O config; :class:`InputDataDataset` wraps the
loaded archive as a torch dataset (leading axis = sample axis) the compare
loop can iterate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np


if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def load_input_data(
    path: Path,
    io_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load model inputs from a ``.npz`` file, validated against the model.

    Lets ``winml perf`` / ``winml eval`` run with real input tensors instead
    of randomly generated ones. Only ``.npz`` (a named-array archive) is
    supported today; a single-array ``.npy`` carries no input names to bind
    against and is rejected with guidance to repackage as ``.npz``.

    Validation:

    * the archive's keys must exactly match the model's input names -- any
      missing or unexpected key is an error (an unexpected key is usually a
      typo that would otherwise leave a required input silently unset);
    * an array whose dtype differs from the model's expected input dtype is
      cast to the expected dtype with a warning, matching the silent casting
      ``WinMLSession._prepare_inputs`` does on a normal run (e.g. numpy's
      default int64 literals binding to an int32 input).

    Shapes are taken from the arrays as-is; correctness beyond dtype (e.g. a
    static dimension the data violates) surfaces as a runtime error from the
    inference session.

    Args:
        path: Path to the ``.npz`` file.
        io_config: Model I/O configuration (``input_names``, ``input_types``).

    Returns:
        Dictionary of ``input_name -> numpy array``.

    Raises:
        click.UsageError: On a non-``.npz`` file or a key mismatch.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raise click.UsageError(
            f"--input-data does not support .npy files ({path.name}). A single "
            f"array carries no input names; save your inputs as a named .npz "
            f"archive instead (e.g. np.savez('inputs.npz', input_ids=..., "
            f"attention_mask=...))."
        )
    if path.suffix.lower() != ".npz":
        raise click.UsageError(
            f"--input-data must be a .npz file, got '{path.suffix or path.name}'."
        )

    try:
        with np.load(path, allow_pickle=False) as archive:
            provided = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise click.UsageError(f"Could not read --input-data file {path}: {exc}") from exc

    expected_names = list(io_config["input_names"])
    expected_types = list(io_config["input_types"])

    missing = [name for name in expected_names if name not in provided]
    unexpected = [name for name in provided if name not in expected_names]
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unexpected:
            parts.append(f"unexpected {unexpected}")
        raise click.UsageError(
            f"--input-data keys do not match the model inputs ({', '.join(parts)}). "
            f"Expected exactly: {expected_names}."
        )

    # Cast dtype mismatches instead of failing, mirroring the session's
    # _prepare_inputs, so inputs that would run fine on a normal invocation
    # (e.g. int64 literals against an int32 input) don't hard-error here.
    for name, expected_dtype in zip(expected_names, expected_types, strict=True):
        want = np.dtype(expected_dtype)
        got = provided[name].dtype
        if got != want:
            logger.warning(
                "--input-data dtype for '%s' is %s; casting to the model's expected %s.",
                name,
                got,
                want,
            )
            provided[name] = provided[name].astype(want)

    return provided


class InputDataDataset:
    """Multi-sample dataset backed by a validated ``.npz`` of real tensors.

    Loads the archive once via :func:`load_input_data` (keys and dtypes
    validated/cast against ``io_config``), then treats the **leading axis of
    each array as the sample axis**: an archive whose arrays have shape
    ``(N, ...)`` yields ``N`` samples, so ``--mode compare`` can run the
    candidate and reference on many real inputs and report a real
    distribution (mean/std/min/max) instead of a single point.

    Every input must share the same leading length ``N`` (a clear error is
    raised otherwise). Each sample keeps a leading batch dim of 1
    (``arr[i:i+1]``), so any dynamic-batch model accepts it and no assumption
    is made about output layout — each run is compared independently, exactly
    like :class:`RandomDataset`'s per-sample flow.

    Args:
        path: Path to the ``.npz`` file of real input tensors.
        io_config: Candidate model I/O config (``input_names``, ``input_types``).
    """

    TASK_TYPE = "input_data"

    def __init__(self, path: str | Path, io_config: dict[str, Any]) -> None:
        import torch

        arrays = load_input_data(Path(path), io_config)

        # Leading axis = sample axis. Reject scalars (no sample axis) and any
        # disagreement on N so a silent mis-pairing can't produce bogus metrics.
        leading: dict[str, int] = {}
        for name, arr in arrays.items():
            if arr.ndim == 0:
                raise click.UsageError(
                    f"--input-data array '{name}' is a scalar (0-d); the leading "
                    "axis is the sample axis, so each input needs at least one dim."
                )
            leading[name] = int(arr.shape[0])

        distinct = set(leading.values())
        if len(distinct) != 1:
            detail = ", ".join(f"{name}={leading[name]}" for name in arrays)
            raise click.UsageError(
                "--input-data arrays must share the same leading (sample) axis "
                f"length; got {detail}."
            )

        self._num_samples = distinct.pop()
        if self._num_samples == 0:
            raise click.UsageError("--input-data arrays are empty (sample axis length 0).")

        # np.load arrays are owned/writable; ascontiguousarray avoids the
        # non-contiguous from_numpy warning without an extra copy when possible.
        self._arrays: dict[str, torch.Tensor] = {
            name: torch.from_numpy(np.ascontiguousarray(arr)) for name, arr in arrays.items()
        }

    def __len__(self) -> int:
        """Number of samples (the shared leading-axis length of the inputs)."""
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return sample ``idx`` with each input sliced to a batch of 1."""
        if not 0 <= idx < self._num_samples:
            raise IndexError(
                f"InputDataDataset index {idx} out of range for {self._num_samples} samples."
            )
        return {name: tensor[idx : idx + 1] for name, tensor in self._arrays.items()}
