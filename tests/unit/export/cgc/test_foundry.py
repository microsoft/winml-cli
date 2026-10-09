# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for locating FoundryToolbox and creating the upstream compiler."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from unittest.mock import Mock, patch

import _foundry
import pytest

from winml.modelkit.export.cgc import (
    FoundryToolboxUnavailableError,
    create_foundry_compiler,
    find_foundry_toolbox,
)


def test_find_foundry_toolbox_in_windowsml_wheel(tmp_path: Path) -> None:
    dll_path = tmp_path / "windowsml" / "lib" / "FoundryToolbox.dll"
    dll_path.parent.mkdir(parents=True)
    dll_path.touch()
    distribution = Mock()
    distribution.locate_file.return_value = dll_path

    with patch(
        "winml.modelkit.export.cgc.foundry.metadata.distribution",
        return_value=distribution,
    ):
        assert find_foundry_toolbox() == dll_path.resolve()
    distribution.locate_file.assert_called_once_with(
        Path("windowsml") / "lib" / "FoundryToolbox.dll"
    )


def test_find_foundry_toolbox_requires_windowsml_distribution() -> None:
    with (
        patch(
            "winml.modelkit.export.cgc.foundry.metadata.distribution",
            side_effect=metadata.PackageNotFoundError,
        ),
        pytest.raises(
            FoundryToolboxUnavailableError,
            match="requires a windowsml wheel",
        ),
    ):
        find_foundry_toolbox()


def test_find_foundry_toolbox_requires_dll_in_wheel(tmp_path: Path) -> None:
    distribution = Mock(version=metadata.version("windowsml"))
    distribution.locate_file.return_value = (
        tmp_path / "windowsml" / "lib" / "FoundryToolbox.dll"
    )

    with (
        patch(
            "winml.modelkit.export.cgc.foundry.metadata.distribution",
            return_value=distribution,
        ),
        pytest.raises(FoundryToolboxUnavailableError) as captured,
    ):
        find_foundry_toolbox()
    assert distribution.version in str(captured.value)
    assert "does not contain FoundryToolbox.dll" in str(captured.value)


def test_create_foundry_compiler_returns_upstream_instance(tmp_path: Path) -> None:
    dll_path = tmp_path / "FoundryToolbox.dll"
    with (
        patch(
            "winml.modelkit.export.cgc.foundry.find_foundry_toolbox",
            return_value=dll_path,
        ),
        patch("_foundry.Compiler") as constructor,
    ):
        assert create_foundry_compiler() is constructor.return_value
    constructor.assert_called_once_with(library_path=dll_path)


@pytest.mark.parametrize(
    "error",
    [
        _foundry.FoundryLibraryNotFoundError(_foundry.__file__),
        _foundry.CompilerError(_foundry.Result.ERROR_INVALID_ARGUMENT),
    ],
)
def test_create_foundry_compiler_preserves_upstream_errors(
    tmp_path: Path, error: Exception,
) -> None:
    with (
        patch(
            "winml.modelkit.export.cgc.foundry.find_foundry_toolbox",
            return_value=tmp_path / "FoundryToolbox.dll",
        ),
        patch("_foundry.Compiler", side_effect=error),
        pytest.raises(type(error)) as captured,
    ):
        create_foundry_compiler()
    assert captured.value is error
