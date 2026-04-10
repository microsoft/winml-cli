# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Early warning filter configuration for ModelKit.

This module configures warning filters ON IMPORT. It MUST have no dependencies
on modelkit subpackages to avoid triggering the import chain that loads
optimum/diffusers before filters are applied.

Usage:
    from . import _warnings  # Filters are configured automatically

Environment Variables:
    MODELKIT_SHOW_ALL_WARNINGS: Set to "1" or "true" to disable warning suppression
"""

from __future__ import annotations

import logging
import os
import warnings


def _configure() -> None:
    """Configure warning filters and environment variables."""
    # Environment variable to reduce tokenizers noise
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Allow users to see all warnings if they want
    if os.environ.get("MODELKIT_SHOW_ALL_WARNINGS", "").lower() in ("1", "true", "yes"):
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

    logging.getLogger("transformers.pipelines.base").addFilter(_PipelineNoiseFilter())

    # =========================================================================
    # Warning filters (for warnings.warn() calls)
    # =========================================================================
    # Transformers: suppress cosmetic warnings (not RuntimeWarning/ResourceWarning)
    for _cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=_cat, module=r"transformers\..*")

    # PyTorch: suppress cosmetic warnings (not RuntimeWarning/ResourceWarning)
    for _cat in (FutureWarning, DeprecationWarning, UserWarning):
        warnings.filterwarnings("ignore", category=_cat, module=r"torch\..*")

    # Diffusers
    warnings.filterwarnings(
        "ignore", message=r".*CUDA.*", category=UserWarning, module=r"diffusers.*"
    )


# Auto-configure on import
_configure()
