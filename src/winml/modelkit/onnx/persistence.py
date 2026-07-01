# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX model persistence utilities.

Load, save, and clean up ONNX models with external data support.
Designed as the canonical persistence API for WinML CLI ONNX workflows.

See also: docs/design/onnx/persistence.md (if available)
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
from typing import NoReturn

import onnx
from onnx.external_data_helper import _get_all_tensors, uses_external_data

from .utils import EXTERNAL_DATA_THRESHOLD, get_model_size


logger = logging.getLogger(__name__)


# Windows ERROR_DISK_FULL. Python usually maps this to errno.ENOSPC via the CRT,
# but we check the raw winerror too so a disk-full write is always recognised.
_WINDOWS_ERROR_DISK_FULL = 112


class ONNXSaveError(OSError):
    """Raised when an ONNX model cannot be written to disk.

    Subclasses :class:`OSError` so existing ``except OSError`` handlers keep
    working and the original ``errno`` is preserved (see ``errno_code``), while
    surfacing a clear, actionable message. This matters most for disk-full
    conditions: without it, a failed write leaves a truncated/zero-byte
    ``.onnx`` behind and the real cause only shows up much later as an opaque
    opset-parsing error in a downstream stage.

    Note:
        ``OSError.__init__`` only populates ``errno`` from a 2-argument
        ``(errno, strerror)`` call, which would also rewrite ``str(self)`` as
        ``"[Errno N] <message>"``. To keep the clean message *and* preserve
        ``errno`` for ``except OSError`` callers that inspect ``e.errno``, we
        construct with the single message and set ``errno`` explicitly.

    Attributes:
        path: Destination path that could not be written.
        disk_full: ``True`` when the failure was caused by insufficient disk
            space (``errno.ENOSPC`` / Windows ``ERROR_DISK_FULL``).
        errno: The originating OS error code, when known (inherited from
            :class:`OSError`).
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | Path | None = None,
        disk_full: bool = False,
        errno_code: int | None = None,
    ) -> None:
        super().__init__(message)
        # super().__init__(message) leaves self.errno = None; set it explicitly
        # so callers catching this as OSError can still inspect e.errno.
        if errno_code is not None:
            self.errno = errno_code
        self.path = path
        self.disk_full = disk_full


def _is_disk_full_error(error: OSError) -> bool:
    """Return ``True`` when *error* represents an out-of-disk-space condition."""
    return (
        error.errno == errno.ENOSPC
        or getattr(error, "winerror", None) == _WINDOWS_ERROR_DISK_FULL
    )


def _cleanup_partial_save(*paths: Path | None) -> None:
    """Best-effort removal of partial artifacts left by a failed write.

    A failed ``onnx.save_model`` / copy can leave a zero-byte or truncated
    ``.onnx`` file (and ``.data`` sidecar) behind. Removing them prevents a
    later stage from loading a corrupt model and reporting a misleading error.
    """
    for partial in paths:
        if partial is None:
            continue
        try:
            Path(partial).unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove partial artifact: %s", partial, exc_info=True)


def _raise_save_error(error: OSError, path: Path) -> NoReturn:
    """Translate a write ``OSError`` into a clear :class:`ONNXSaveError`."""
    disk_full = _is_disk_full_error(error)
    if disk_full:
        message = (
            f"Insufficient disk space — unable to write ONNX model to {path}. "
            "Free up disk space and try again."
        )
    else:
        message = f"Failed to write ONNX model to {path}: {error}"
    raise ONNXSaveError(
        message, path=path, disk_full=disk_full, errno_code=error.errno
    ) from error


def load_onnx(
    path: str | Path,
    *,
    load_weights: bool = True,
    validate: bool = True,
) -> onnx.ModelProto:
    """Load an ONNX model from disk.

    Args:
        path: Path to the ``.onnx`` file.
        load_weights: If ``True`` (default), load external weight data.
            Set to ``False`` to load only the graph structure.
        validate: If ``True`` (default), run ``onnx.checker.check_model``
            against the file path (safe for models of any size).

    Returns:
        The loaded ``onnx.ModelProto``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        onnx.checker.ValidationError: If *validate* is True and the model
            fails validation.
    """
    path = Path(path)
    if not path.exists():
        msg = f"ONNX model not found: {path}"
        raise FileNotFoundError(msg)

    logger.debug("Loading ONNX model from %s (weights=%s)", path, load_weights)

    if validate:
        logger.debug("Validating ONNX model at %s", path)
        onnx.checker.check_model(str(path))

    model = onnx.load(str(path), load_external_data=load_weights)
    logger.debug(
        "Loaded ONNX model: %d nodes, %d initializers",
        len(model.graph.node),
        len(model.graph.initializer),
    )
    return model


def save_onnx(
    model: onnx.ModelProto,
    path: str | Path,
    *,
    use_external_data: bool = True,
    threshold_size: int = EXTERNAL_DATA_THRESHOLD,
    location: str | None = None,
) -> None:
    """Save an ONNX model to disk.

    Automatically decides whether to write weights inline or as an external
    ``.data`` sidecar file based on model size and options.

    Args:
        model: The ``onnx.ModelProto`` to save.
        path: Destination file path.
        use_external_data: If ``False``, force inline saving (no sidecar).
            Ignored when the model already contains external data markers.
        threshold_size: Byte threshold above which external data is used.
            Set to ``0`` to always use external data.  Defaults to 100 MiB.
        location: Custom filename for the external data sidecar.
            Defaults to ``{filename}.data``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    has_existing_external = any(uses_external_data(t) for t in _get_all_tensors(model))

    if has_existing_external:
        save_external = True
    elif not use_external_data:
        save_external = False
    elif threshold_size <= 0:
        save_external = True
    else:
        save_external = get_model_size(model) >= threshold_size

    if save_external:
        ext_location = location or f"{path.name}.data"
        # Delete any pre-existing sidecar so onnx.save_model doesn't raise
        # FileExistsError (e.g. when ORT quantize() already wrote the file).
        ext_path = path.parent / ext_location
        if ext_path.exists():
            ext_path.unlink()
            logger.debug("Removed existing external data sidecar: %s", ext_path)
        logger.debug(
            "Saving ONNX model with external data to %s (location=%s)",
            path,
            ext_location,
        )
        # Temporarily change CWD to the output directory so that the ONNX
        # library's CWD-relative existence check (external_data_helper.py)
        # resolves against the correct output directory rather than the
        # process CWD.  This avoids a false-positive FileExistsError when a
        # stale .data sidecar exists in the process CWD from a previous build
        # but the actual output directory is clean.
        # path.parent is guaranteed to exist: mkdir() was called above.
        original_cwd = Path.cwd()
        try:
            try:
                os.chdir(path.parent)
                onnx.save_model(
                    model,
                    path.name,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=ext_location,
                    size_threshold=1024,
                )
            finally:
                os.chdir(original_cwd)
        except OSError as e:
            # A failed external-data write can leave a truncated .onnx and/or
            # .data sidecar behind; remove them so a later stage never loads a
            # corrupt model and reports a misleading error.
            _cleanup_partial_save(path, ext_path)
            _raise_save_error(e, path)
    else:
        logger.debug("Saving ONNX model inline to %s", path)
        try:
            onnx.save_model(model, str(path))
        except OSError as e:
            _cleanup_partial_save(path)
            _raise_save_error(e, path)


def cleanup_onnx(path: str | Path) -> list[Path]:
    """Delete an ONNX model and its external data files.

    Args:
        path: Path to the ``.onnx`` file.

    Returns:
        List of :class:`~pathlib.Path` objects that were actually deleted.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"ONNX model not found: {path}"
        raise FileNotFoundError(msg)

    deleted: list[Path] = []

    # Load graph only (no weights) to discover external data locations
    model = onnx.load(str(path), load_external_data=False)
    data_locations: set[str] = set()
    for tensor in _get_all_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location":
                data_locations.add(entry.value)

    # Delete external data files
    for loc in sorted(data_locations):
        data_path = path.parent / loc
        if data_path.exists():
            data_path.unlink()
            deleted.append(data_path)
            logger.debug("Deleted external data: %s", data_path)
        else:
            logger.debug("External data file not found (skipping): %s", data_path)

    # Delete the .onnx file itself
    path.unlink()
    deleted.append(path)
    logger.debug("Deleted ONNX model: %s", path)

    return deleted
