# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for _warnings._configure() filter behaviour."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


def _get_hf_symlinks_filter() -> logging.Filter:
    """Return the _HFSymlinksInfoFilter installed on the py.warnings logger."""
    logger = logging.getLogger("py.warnings")
    matches = [f for f in logger.filters if type(f).__name__ == "_HFSymlinksInfoFilter"]
    assert matches, "_HFSymlinksInfoFilter not found on py.warnings logger"
    return matches[0]


def _emit_warning(message: str, filename: str) -> None:
    """Emit a UserWarning through the real logging.captureWarnings path."""
    logging.captureWarnings(True)
    try:
        warnings.warn_explicit(
            message=message,
            category=UserWarning,
            filename=filename,
            lineno=1,
            registry={},  # fresh registry — always route to showwarning
        )
    finally:
        logging.captureWarnings(False)


class TestHFSymlinksInfoFilter:
    """_HFSymlinksInfoFilter downgrades huggingface_hub symlinks warnings to DEBUG."""

    def test_filter_is_installed(self) -> None:
        """_configure() installs _HFSymlinksInfoFilter on the py.warnings logger."""
        import winml.modelkit._warnings  # noqa: F401 — triggers _configure()

        _get_hf_symlinks_filter()  # asserts filter exists

    def test_downgrade_to_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning from huggingface_hub containing 'symlinks' is downgraded to DEBUG."""
        import winml.modelkit._warnings  # noqa: F401

        with caplog.at_level(logging.DEBUG, logger="py.warnings"):
            _emit_warning(
                message=(
                    "`huggingface_hub` cache-system uses symlinks by default to"
                    " efficiently store duplicated files but your machine does not"
                    " support them"
                ),
                filename="C:/fake/huggingface_hub/file_download.py",
            )

        records = [r for r in caplog.records if "symlinks" in r.getMessage()]
        assert records, "No matching record captured"
        assert records[0].levelno == logging.DEBUG
        assert records[0].levelname == "DEBUG"

    def test_unrelated_warning_unchanged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Records without 'symlinks' in message are not modified."""
        import winml.modelkit._warnings  # noqa: F401

        with caplog.at_level(logging.DEBUG, logger="py.warnings"):
            _emit_warning(
                message="Some other huggingface_hub warning without the keyword",
                filename="C:/fake/huggingface_hub/file_download.py",
            )

        records = [r for r in caplog.records if "huggingface_hub" in r.getMessage()]
        assert records, "No matching record captured"
        assert records[0].levelno == logging.WARNING

    def test_symlinks_from_other_module_unchanged(self, caplog: pytest.LogCaptureFixture) -> None:
        """A 'symlinks' warning not from huggingface_hub is not modified."""
        import winml.modelkit._warnings  # noqa: F401

        with caplog.at_level(logging.DEBUG, logger="py.warnings"):
            _emit_warning(
                message="symlinks are not supported on this platform",
                filename="C:/fake/other_library/utils.py",
            )

        records = [r for r in caplog.records if "symlinks" in r.getMessage()]
        assert records, "No matching record captured"
        assert records[0].levelno == logging.WARNING
