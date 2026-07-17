# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Early warning filter configuration for WinML CLI.

This module configures warning filters ON IMPORT. It MUST have no dependencies
on modelkit subpackages to avoid triggering the import chain that loads
optimum/diffusers before filters are applied.

Usage:
    from . import _warnings  # Filters are configured automatically

Environment Variables:
    WINMLCLI_SHOW_ALL_WARNINGS: Set to "1", "true", "yes", or "on" to disable
        warning suppression.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

from ._env import env_flag_enabled


def _configure() -> None:
    """Configure warning filters and environment variables."""
    # Environment variable to reduce tokenizers noise
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Suppress huggingface_hub tqdm download progress bars by default.
    # These are written directly to stderr by tqdm and cannot be routed
    # through Python logging.  Users can override with
    # HF_HUB_DISABLE_PROGRESS_BARS=0 to restore them.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Allow users to see all warnings if they want
    if env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS"):
        return

    # =========================================================================
    # Logging filters (for logging.warning() calls)
    # =========================================================================

    class _DiffusersDistributionFilter(logging.Filter):
        """Filter 'Multiple distributions found' from diffusers.

        Caused by optimum and optimum-onnx both claiming the 'optimum' package.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            return "Multiple distributions found" not in record.getMessage()

    logging.getLogger("diffusers.utils.import_utils").addFilter(_DiffusersDistributionFilter())

    class _PipelineNoiseFilter(logging.Filter):
        """Filter noisy HF Pipeline warnings.

        - 'The model X is not supported for Y' — WinML models are duck-type
          compatible but not in HF's supported list.
        - 'Device set to use cpu' — HF Pipeline forces CPU, we handle device.
        - 'Using a slow image processor' — cosmetic deprecation notice.
        """

        _SUPPRESSED = (
            "is not supported for",
            "Device set to use cpu",
            "Using a slow image processor",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(s in msg for s in self._SUPPRESSED)

    for logger_name in (
        "transformers.pipelines.base",
        "transformers.models.auto.image_processing_auto",
    ):
        logging.getLogger(logger_name).addFilter(_PipelineNoiseFilter())

    # =========================================================================
    # Warning filters (for warnings.warn() calls)
    # =========================================================================
    # Transformers: suppress cosmetic warnings (not RuntimeWarning/ResourceWarning)
    for _cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=_cat, module=r"transformers\..*")

    # PyTorch: suppress cosmetic warnings (not RuntimeWarning/ResourceWarning)
    for _cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=_cat, module=r"torch\..*")

    # TracerWarning (from torch.jit, inherits Warning not UserWarning)
    # fires during ONNX export tracing — safe to suppress in both torch and
    # transformers. Only register the filter if torch has ALREADY been
    # imported; otherwise loading torch here would add ~2s to every
    # lightweight command (``winml sys`` etc.) that never touches ONNX
    # export. The export path re-triggers this by calling
    # :func:`install_torch_tracer_filter` after loading torch.
    if "torch" in sys.modules:
        install_torch_tracer_filter()

    # Diffusers
    warnings.filterwarnings(
        "ignore", message=r".*CUDA.*", category=UserWarning, module=r"diffusers.*"
    )

    # =========================================================================
    # huggingface_hub: suppress the Windows "symlinks not supported" notice
    # =========================================================================
    # On Windows without Developer Mode, huggingface_hub emits a UserWarning that
    # its cache will use file copies instead of symlinks. This is cosmetic — the
    # cache still works, just without deduplication. Drop it at the Python
    # warnings layer so it is hidden in every verbosity mode; this also stops it
    # before captureWarnings(True) (activated in build.py) could route it to the
    # py.warnings logger.
    warnings.filterwarnings(
        "ignore",
        message=r".*huggingface_hub.*cache-system.*symlinks.*",
        category=UserWarning,
    )

    # NOTE: optimum's informational WARNINGs (e.g. "TasksManager returned ...",
    # "No model type passed for the task ...") are gated by the verbosity-
    # conditional ERROR floor on the `optimum` logger in utils/logging.py
    # (configure_logging): hidden by default, shown at -v/-vv. A demote-to-INFO
    # filter here would only relabel a record after the root-level gate has
    # already passed it through — it would not suppress anything — so none is used.

    class _TransformersWeightsFilter(logging.Filter):
        """Suppress the transformers "weights not used" notice.

        When loading a checkpoint, transformers warns about pooler or other
        weights that are intentionally absent in the target architecture.
        This is expected (e.g. RobertaForSequenceClassification drops the
        pooler from a base checkpoint) and is purely cosmetic noise.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            return "were not used when initializing" not in record.getMessage()

    logging.getLogger("transformers.modeling_utils").addFilter(_TransformersWeightsFilter())


def install_torch_tracer_filter() -> None:
    """Register the ``TracerWarning`` filter — call after ``import torch``.

    Idempotent — ``warnings.filterwarnings`` de-duplicates identical entries.
    """
    try:
        # torch.jit exposes TracerWarning at runtime but its stubs don't export it.
        from torch.jit import TracerWarning  # type: ignore[attr-defined]
    except ImportError:
        return  # torch not installed
    warnings.filterwarnings("ignore", category=TracerWarning)


# Auto-configure on import
_configure()
