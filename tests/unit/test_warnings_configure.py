# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for _warnings._configure() filter behaviour."""

from __future__ import annotations

import logging


def _make_record(
    message: str,
    pathname: str = "",
    level: int = logging.WARNING,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="py.warnings",
        level=level,
        pathname=pathname,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def _get_hf_symlinks_filter() -> logging.Filter:
    """Return the _HFSymlinksInfoFilter installed on the py.warnings logger."""
    logger = logging.getLogger("py.warnings")
    matches = [f for f in logger.filters if type(f).__name__ == "_HFSymlinksInfoFilter"]
    assert matches, "_HFSymlinksInfoFilter not found on py.warnings logger"
    return matches[0]


class TestHFSymlinksInfoFilter:
    """_HFSymlinksInfoFilter downgrades huggingface_hub symlinks warnings to INFO."""

    def test_filter_is_installed(self) -> None:
        """_configure() installs _HFSymlinksInfoFilter on the py.warnings logger."""
        import winml.modelkit._warnings  # noqa: F401 — triggers _configure()

        _get_hf_symlinks_filter()  # asserts filter exists

    def test_downgrade_to_info(self) -> None:
        """Matching record is mutated to INFO and allowed through (returns True)."""
        import winml.modelkit._warnings  # noqa: F401

        f = _get_hf_symlinks_filter()
        record = _make_record(
            message=(
                "`huggingface_hub` cache-system uses symlinks by default to efficiently "
                "store duplicated files but your machine does not support them"
            ),
            pathname=r"C:\some\path\huggingface_hub\file_download.py",
        )

        result = f.filter(record)

        assert result is True
        assert record.levelno == logging.INFO
        assert record.levelname == "INFO"

    def test_unrelated_warning_unchanged(self) -> None:
        """Records without 'symlinks' in message are not modified."""
        import winml.modelkit._warnings  # noqa: F401

        f = _get_hf_symlinks_filter()
        record = _make_record(
            message="Some other huggingface_hub warning",
            pathname=r"C:\some\path\huggingface_hub\file_download.py",
        )

        result = f.filter(record)

        assert result is True
        assert record.levelno == logging.WARNING

    def test_symlinks_from_other_module_unchanged(self) -> None:
        """A 'symlinks' message from a non-huggingface_hub path is not modified."""
        import winml.modelkit._warnings  # noqa: F401

        f = _get_hf_symlinks_filter()
        record = _make_record(
            message="symlinks are not supported",
            pathname=r"C:\some\other\library\utils.py",
        )

        result = f.filter(record)

        assert result is True
        assert record.levelno == logging.WARNING
