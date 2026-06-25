# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for _warnings._configure() suppression behaviour."""

from __future__ import annotations

import logging
import warnings

import pytest

from winml.modelkit._warnings import _configure


_HF_SYMLINKS_MESSAGE = (
    "`huggingface_hub` cache-system uses symlinks by default to efficiently"
    " store duplicated files but your machine does not support them"
)

# Loggers that _configure() attaches noise-suppression filters to. Calling
# _configure() repeatedly (as these tests do) would otherwise accumulate
# duplicate filter instances in global logging state.
_FILTERED_LOGGERS = (
    "diffusers.utils.import_utils",
    "transformers.pipelines.base",
    "transformers.models.auto.image_processing_auto",
    "transformers.modeling_utils",
)


@pytest.fixture(autouse=True)
def _restore_logging_filters():
    """Snapshot and restore the filter lists of the loggers _configure() mutates."""
    saved = {name: list(logging.getLogger(name).filters) for name in _FILTERED_LOGGERS}
    yield
    for name, filters in saved.items():
        logging.getLogger(name).filters[:] = filters


class TestHFSymlinksSuppression:
    """_configure() drops the huggingface_hub symlinks UserWarning at the warnings layer.

    The filter is a hard ``filterwarnings("ignore")`` rather than a demote-to-INFO
    logging filter, so the warning never reaches the ``py.warnings`` logger and is
    hidden in every verbosity mode.
    """

    def test_symlinks_warning_is_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The huggingface_hub symlinks UserWarning is ignored, not surfaced."""
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            _configure()
            warnings.warn(_HF_SYMLINKS_MESSAGE, UserWarning, stacklevel=2)

        assert not [w for w in caught if "symlinks" in str(w.message)]

    def test_unrelated_symlinks_warning_not_suppressed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ignore filter targets the HF message only; other warnings pass through."""
        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            _configure()
            warnings.warn("symlinks are unsupported on this platform", UserWarning, stacklevel=2)

        assert [w for w in caught if "symlinks" in str(w.message)]
