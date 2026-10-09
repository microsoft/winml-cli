# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Locate FoundryToolbox and create the wheel's compiler."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from _foundry import Compiler


class FoundryToolboxUnavailableError(RuntimeError):
    """Raised when the installed wheel does not provide FoundryToolbox."""


def find_foundry_toolbox() -> Path:
    """Resolve FoundryToolbox from the installed windowsml wheel."""
    try:
        distribution = metadata.distribution("windowsml")
    except metadata.PackageNotFoundError as e:
        raise FoundryToolboxUnavailableError(
            "CGC export requires a windowsml wheel containing FoundryToolbox.dll."
        ) from e

    candidate = Path(
        str(distribution.locate_file(Path("windowsml") / "lib" / "FoundryToolbox.dll"))
    )
    if not candidate.is_file():
        raise FoundryToolboxUnavailableError(
            "The installed windowsml wheel "
            f"({distribution.version}) does not contain FoundryToolbox.dll."
        )
    return candidate.resolve()


def create_foundry_compiler() -> Compiler:
    """Create the upstream compiler with the installed wheel's DLL."""
    from _foundry import Compiler

    return Compiler(library_path=find_foundry_toolbox())
