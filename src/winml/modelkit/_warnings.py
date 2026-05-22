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
    WINMLCLI_SHOW_ALL_WARNINGS: Set to "1" or "true" to disable warning suppression
"""

from __future__ import annotations

import logging
import os
import warnings


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
    if os.environ.get("WINMLCLI_SHOW_ALL_WARNINGS", "").lower() in ("1", "true", "yes"):
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

    # NOTE: TracerWarning (from torch.jit) is intentionally NOT filtered here.
    # Importing torch.jit at startup would pull all of torch (~1.7s) into
    # `winml --help` and violate the CLI import budget (tests/cli/test_import_time.py).
    # During ONNX export, export_pytorch() wraps torch.onnx.export in
    # `warnings.catch_warnings()` + `filterwarnings("ignore")`, which is strictly
    # broader than a TracerWarning-only filter.

    # Diffusers
    warnings.filterwarnings(
        "ignore", message=r".*CUDA.*", category=UserWarning, module=r"diffusers.*"
    )

    # =========================================================================
    # py.warnings logger filters (for warnings routed via logging.captureWarnings)
    # =========================================================================

    class _HFSymlinksInfoFilter(logging.Filter):
        r"""Downgrade the huggingface_hub symlinks UserWarning from WARNING to INFO.

        On Windows without Developer Mode, huggingface_hub warns that symlinks
        are unsupported and the cache will use copies instead. This is cosmetic —
        the cache still works, just without deduplication. WARNING is misleading
        here; INFO is the appropriate level.

        When warnings are routed via logging.captureWarnings(True), Python's
        warnings.formatwarning() embeds the source filename in the log message
        body ("path/to/huggingface_hub/file_download.py:1: UserWarning: ..."),
        so we match against getMessage() rather than record.pathname (which
        is always warnings.py in that path).

        Before (WARNING level, always visible):
            [09:12:34] WARNING  C:\\...\\huggingface_hub\\file_download.py:1:
                                UserWarning: `huggingface_hub` cache-system uses
                                symlinks by default to efficiently store
                                duplicated files but your machine does not
                                support them

        After (DEBUG level, only visible with --verbose):
            [09:12:34] DEBUG    C:\\...\\huggingface_hub\\file_download.py:1:
                                UserWarning: `huggingface_hub` cache-system uses
                                symlinks by default to efficiently store
                                duplicated files but your machine does not
                                support them
        """

        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "symlinks" in msg and "huggingface_hub" in msg:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
            return True

    logging.getLogger("py.warnings").addFilter(_HFSymlinksInfoFilter())

    class _TasksManagerFilter(logging.Filter):
        """Downgrade optimum TasksManager architecture-mismatch notice to DEBUG.

        optimum logs a WARNING when TasksManager selects a different Auto class
        than the one in config.architectures (e.g. AutoModelForSequenceClassification
        vs RobertaForSequenceClassification).  This is expected behaviour for
        WinML models and is purely informational.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            if "TasksManager returned" in record.getMessage():
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
            return True

    logging.getLogger("optimum.exporters.tasks").addFilter(_TasksManagerFilter())


# Auto-configure on import
_configure()
