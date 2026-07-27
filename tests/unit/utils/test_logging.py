# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for configure_logging — third-party logger noise control."""

import logging

import pytest

from winml.modelkit.utils.logging import _NOISY_LIBRARY_LOGGERS, configure_logging


@pytest.fixture(autouse=True)
def _restore_logger_levels():
    """configure_logging mutates global logger state (root + the noisy library loggers);
    restore all of them after each test so verbosity changes don't leak across tests."""
    saved = [(logging.getLogger(), logging.getLogger().level)]
    for name in _NOISY_LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
    yield
    for logger, level in saved:
        logger.setLevel(level)


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


def test_optimum_child_logger_gated_by_parent_floor():
    # The optimum "TasksManager returned ..." notice originates on the child logger
    # optimum.exporters.tasks. With no demote filter, the parent ERROR floor must hide
    # it by default and reveal it at -v (the floor follows the CLI level). This is what
    # replaces the removed _TasksManagerFilter demote-to-INFO filter.
    child = logging.getLogger("optimum.exporters.tasks")

    configure_logging(verbosity=0)
    assert not child.isEnabledFor(logging.WARNING)

    configure_logging(verbosity=1)
    assert child.isEnabledFor(logging.WARNING)


def test_onnxscript_version_converter_floored_at_error_in_normal_mode():
    # The onnxscript version-converter fallback WARNING carries a full call stack when
    # the dynamo exporter cannot down-convert to the requested opset. winml surfaces
    # its own concise opset warning, so the raw traceback is floored out by default.
    configure_logging(verbosity=0)
    assert logging.getLogger("onnxscript.version_converter").level == logging.ERROR


@pytest.mark.parametrize("verbosity,expected", [(1, logging.INFO), (2, logging.DEBUG)])
def test_onnxscript_version_converter_revealed_when_verbose(verbosity, expected):
    # -v/-vv opts into the detail: the converter logger follows the CLI level so the
    # call stack becomes visible on demand.
    logger = logging.getLogger("onnxscript.version_converter")

    configure_logging(verbosity=0)
    assert not logger.isEnabledFor(logging.WARNING)

    configure_logging(verbosity=verbosity)
    assert logger.level == expected
    assert logger.isEnabledFor(logging.WARNING)


def test_torch_compat_opset_notice_floored_at_error_in_normal_mode():
    # torch's exporter emits a one-line "Setting ONNX exporter to use operator set
    # version 18 ..." WARNING when it cannot honor a lower requested opset. winml
    # surfaces its own concise opset warning, so torch's notice is floored by default.
    configure_logging(verbosity=0)
    logger = logging.getLogger("torch.onnx._internal.exporter._compat")
    assert logger.level == logging.ERROR
    assert not logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("verbosity,expected", [(1, logging.INFO), (2, logging.DEBUG)])
def test_torch_compat_opset_notice_revealed_when_verbose(verbosity, expected):
    # -v/-vv opts into the detail: the torch logger follows the CLI level.
    logger = logging.getLogger("torch.onnx._internal.exporter._compat")

    configure_logging(verbosity=0)
    assert not logger.isEnabledFor(logging.WARNING)

    configure_logging(verbosity=verbosity)
    assert logger.level == expected
    assert logger.isEnabledFor(logging.WARNING)
