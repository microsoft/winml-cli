# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for configure_logging — third-party logger noise control."""

import logging

import pytest

from winml.modelkit.utils.logging import configure_logging


@pytest.fixture(autouse=True)
def _restore_logger_levels():
    """configure_logging mutates global logger state (root + the noisy library loggers);
    restore both after each test so verbosity changes don't leak across tests."""
    root = logging.getLogger()
    optimum = logging.getLogger("optimum")
    root_before, optimum_before = root.level, optimum.level
    yield
    root.setLevel(root_before)
    optimum.setLevel(optimum_before)


def test_library_loggers_floored_at_error_in_normal_mode():
    # Default verbosity: noisy library loggers (optimum) must not leak below ERROR,
    # so their informational notices never reach normal CLI output.
    configure_logging(verbosity=0)
    assert logging.getLogger("optimum").level == logging.ERROR


def test_quiet_keeps_library_loggers_at_error():
    configure_logging(quiet=True)
    assert logging.getLogger("optimum").level == logging.ERROR


@pytest.mark.parametrize("verbosity,expected", [(1, logging.INFO), (2, logging.DEBUG)])
def test_library_loggers_follow_cli_level_when_verbose(verbosity, expected):
    # With -v/-vv the library loggers follow the CLI level so the detail is on demand.
    configure_logging(verbosity=verbosity)
    assert logging.getLogger("optimum").level == expected
