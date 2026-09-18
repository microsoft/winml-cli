# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Lazy import of the optional Windows ML Runtime native dependency."""

from typing import Any

import click


def import_runtime() -> Any:
    """Import ``windowsml.runtime`` or raise an actionable ClickException."""
    try:
        import windowsml.runtime as wr  # type: ignore[import-not-found, unused-ignore]
    except ImportError as e:
        raise click.ClickException(
            "--runtime winml-runtime requires the preview 'windowsml' package with the "
            "Runtime API. Install it with the ORT backend, e.g. "
            "`pip install windowsml[with-ort]`."
        ) from e
    except FileNotFoundError as e:  # missing WinMLRuntimeCore.dll payload
        raise click.ClickException(
            "--runtime winml-runtime: the installed 'windowsml' package does not ship the "
            "Runtime native library (WinMLRuntimeCore.dll). Install a preview build that "
            "includes the Runtime API."
        ) from e
    return wr
