# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Real input tensors loaded from a ``.npz`` archive.

Shared by ``winml perf`` (benchmark on real tensors instead of random ones)
and ``winml eval --mode compare`` (compare a candidate and reference on the
same real inputs). :func:`load_input_data` validates and dtype-casts the
archive against a model's I/O config; :class:`InputDataDataset` wraps the
loaded archive as a torch dataset the compare loop can iterate.

The two commands interpret the *same* ``.npz`` differently, so an archive is
not always reusable across them:

* ``winml perf`` runs the whole archive as a **single batch** (its leading
  axis is the batch dimension of one benchmarked call).
* ``winml eval --mode compare`` treats the leading axis as the **sample
  axis** and runs each sample independently, chunked to the candidate's batch
  size (a dynamic batch collapses to one row per run), so the similarity table
  reflects a distribution across samples rather than a single call.
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


def _model_batch_size(io_config: dict[str, Any]) -> int:
    """The candidate's expected batch size from its ONNX input shapes.

    Reads the leading (batch) axis of each input: a positive static dim is the
    required batch, while a dynamic dim (``None`` / symbolic / ``<= 0``) accepts
    any batch and collapses to ``1`` -- the same "static preserved, dynamic ->
    1" convention :class:`RandomDataset` uses when sizing synthetic inputs, so
    a statically-batched model behaves consistently on the random and
    real-input paths. Inputs that declare conflicting static batch sizes are
    rejected (one archive cannot satisfy both). Returns ``1`` when no shape
    metadata is available.
    """
    shapes = io_config.get("input_shapes")
    if not shapes:
        return 1
    statics: set[int] = set()
    for shape in shapes:
        if not shape:
            continue
        lead = shape[0]
        if isinstance(lead, (int, np.integer)) and not isinstance(lead, bool) and int(lead) > 0:
            statics.add(int(lead))
    if not statics:
        return 1
    if len(statics) > 1:
        raise click.UsageError(
            "--input-data: the model's inputs declare conflicting static batch "
            f"sizes {sorted(statics)}; cannot batch the provided tensors "
            "unambiguously."
        )
    return statics.pop()


class InputDataDataset:
    """Multi-sample dataset backed by a validated ``.npz`` of real tensors.

    Loads the archive once via :func:`load_input_data` (keys and dtypes
    validated/cast against ``io_config``), then treats the **leading axis of
    each array as the sample axis**: an archive whose arrays have shape
    ``(N, ...)`` yields samples over ``N``, so ``--mode compare`` can run the
    candidate and reference on many real inputs and report a real
    distribution (mean/std/min/max) instead of a single point.

    Each run is shaped to the candidate's **batch size** (see
    :func:`_model_batch_size`): a dynamic batch dim runs one row per sample
    (``arr[i:i+1]``), while a static batch dim ``B`` chunks the leading axis
    into groups of ``B`` (``arr[i*B:(i+1)*B]``), yielding ``N // B`` samples.
    When ``N`` is not a multiple of ``B`` the trailing rows are dropped with a
    warning; ``N`` smaller than ``B`` is an error. No assumption is made about
    output layout -- each run is compared independently, like
    :class:`RandomDataset`'s per-sample flow.

    Every input must share the same leading length ``N`` (a clear error is
    raised otherwise).

    Args:
        path: Path to the ``.npz`` file of real input tensors.
        io_config: Candidate model I/O config (``input_names``, ``input_types``,
            and optionally ``input_shapes`` to honor a static batch dim).
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

        total_rows = distinct.pop()
        if total_rows == 0:
            raise click.UsageError("--input-data arrays are empty (sample axis length 0).")

        # Honor the candidate's batch dim: a statically-batched model (e.g. a
        # fixed batch of 4) needs each run shaped to that batch, so chunk the
        # leading axis into groups of ``batch`` rows. A dynamic batch dim
        # collapses to 1 (one row per run), matching RandomDataset.
        self._batch = _model_batch_size(io_config)
        if total_rows < self._batch:
            raise click.UsageError(
                f"--input-data has {total_rows} row(s) on the sample axis but the "
                f"model needs a batch of {self._batch}; provide at least "
                f"{self._batch} rows."
            )
        self._num_samples = total_rows // self._batch
        remainder = total_rows % self._batch
        if remainder:
            logger.warning(
                "--input-data has %d rows, not a multiple of the model's batch "
                "size %d; dropping the trailing %d row(s).",
                total_rows,
                self._batch,
                remainder,
            )

        # np.load arrays are owned/writable; ascontiguousarray avoids the
        # non-contiguous from_numpy warning without an extra copy when possible.
        self._arrays: dict[str, torch.Tensor] = {
            name: torch.from_numpy(np.ascontiguousarray(arr)) for name, arr in arrays.items()
        }

    def __len__(self) -> int:
        """Number of samples run (leading-axis length // the model's batch size)."""
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return sample ``idx`` as a batch of the model's batch size."""
        if not 0 <= idx < self._num_samples:
            raise IndexError(
                f"InputDataDataset index {idx} out of range for {self._num_samples} samples."
            )
        start = idx * self._batch
        stop = start + self._batch
        return {name: tensor[start:stop] for name, tensor in self._arrays.items()}
