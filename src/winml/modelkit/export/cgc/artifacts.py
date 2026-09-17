# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGC artifact path conventions."""

from pathlib import Path


def cgc_metadata_path(model_path: Path) -> Path:
    """Return the CLI metadata sidecar path for a CGC model."""
    return model_path.with_name(f"{model_path.stem}_metadata.json")
