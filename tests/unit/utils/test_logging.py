# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for configure_logging — third-party logger noise control."""

import logging
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from winml.modelkit.utils.logging import (
    _HUGGINGFACE_WARNING_LOGGERS,
    _NOISY_LIBRARY_LOGGERS,
    configure_logging,
    suppress_huggingface_warning_logs,
    suppress_third_party_progress,
)


_MISSING = object()
_HUGGINGFACE_VERBOSITY_ENVS = ("TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY")
_PROGRESS_ENVS = ("HF_DATASETS_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_PROGRESS_BARS")


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Restore global logger/env state mutated by configure_logging."""
    saved = [(logging.getLogger(), logging.getLogger().level)]
    for name in (*_NOISY_LIBRARY_LOGGERS, *_HUGGINGFACE_WARNING_LOGGERS):
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
    saved_env = {
        name: os.environ.get(name, _MISSING)
        for name in (*_HUGGINGFACE_VERBOSITY_ENVS, *_PROGRESS_ENVS)
    }
    yield
    for logger, level in saved:
        logger.setLevel(level)
    for name, value in saved_env.items():
        if value is _MISSING:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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


@pytest.mark.parametrize("logger_name", _HUGGINGFACE_WARNING_LOGGERS)
def test_huggingface_warning_loggers_not_floored_by_default(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.NOTSET)

    configure_logging(verbosity=0)

    assert logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("logger_name", _HUGGINGFACE_WARNING_LOGGERS)
def test_huggingface_warning_loggers_floored_when_requested(logger_name):
    # Transformers emits cosmetic image-processor deprecation notices and
    # huggingface_hub emits unauthenticated-download warnings at WARNING. They
    # should not interleave with inspect's normal progress output.
    with suppress_huggingface_warning_logs(verbosity=0):
        assert not logging.getLogger(logger_name).isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("logger_name", _HUGGINGFACE_WARNING_LOGGERS)
@pytest.mark.parametrize("verbosity,expected", [(1, logging.INFO), (2, logging.DEBUG)])
def test_huggingface_warning_loggers_revealed_when_verbose(logger_name, verbosity, expected):
    logger = logging.getLogger(logger_name)

    with suppress_huggingface_warning_logs(verbosity=0):
        assert not logger.isEnabledFor(logging.WARNING)

    with suppress_huggingface_warning_logs(verbosity=verbosity):
        assert logger.level == expected
        assert logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("logger_name", _HUGGINGFACE_WARNING_LOGGERS)
def test_show_all_warnings_env_reveals_huggingface_loggers(monkeypatch, logger_name):
    monkeypatch.setenv("WINMLCLI_SHOW_ALL_WARNINGS", "1")

    with suppress_huggingface_warning_logs(verbosity=0):
        logger = logging.getLogger(logger_name)
        assert logger.level == logging.WARNING
        assert logger.isEnabledFor(logging.WARNING)


@pytest.mark.parametrize("env_name", ["TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"])
def test_default_logging_does_not_mutate_huggingface_library_verbosity(monkeypatch, env_name):
    monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
    monkeypatch.delenv(env_name, raising=False)

    configure_logging(verbosity=0)

    assert env_name not in os.environ


@pytest.mark.parametrize("env_name", ["TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"])
def test_requested_suppression_sets_huggingface_library_verbosity_to_error(monkeypatch, env_name):
    monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
    monkeypatch.delenv(env_name, raising=False)

    with suppress_huggingface_warning_logs(verbosity=0):
        assert os.environ[env_name] == "error"

    assert env_name not in os.environ


@pytest.mark.parametrize(
    ("verbosity", "expected"),
    [
        (1, "info"),
        (2, "debug"),
    ],
)
@pytest.mark.parametrize("env_name", ["TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"])
def test_verbose_logging_sets_huggingface_library_verbosity(
    monkeypatch, env_name, verbosity, expected
):
    monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
    monkeypatch.delenv(env_name, raising=False)

    with suppress_huggingface_warning_logs(verbosity=verbosity):
        assert os.environ[env_name] == expected

    assert env_name not in os.environ


@pytest.mark.parametrize("env_name", ["TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"])
def test_show_all_warnings_sets_huggingface_library_verbosity_to_warning(monkeypatch, env_name):
    monkeypatch.setenv("WINMLCLI_SHOW_ALL_WARNINGS", "1")
    monkeypatch.delenv(env_name, raising=False)

    with suppress_huggingface_warning_logs(verbosity=0):
        assert os.environ[env_name] == "warning"

    assert env_name not in os.environ


def test_huggingface_warning_context_restores_logger_levels():
    baseline = {
        "huggingface_hub": logging.WARNING,
        "transformers": logging.NOTSET,
    }
    for name, level in baseline.items():
        logging.getLogger(name).setLevel(level)

    with suppress_huggingface_warning_logs(verbosity=0):
        assert logging.getLogger("huggingface_hub").level == logging.ERROR
        assert logging.getLogger("transformers").level == logging.ERROR

    for name, level in baseline.items():
        assert logging.getLogger(name).level == level


@pytest.mark.parametrize("env_name", ["TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"])
def test_huggingface_warning_context_restores_existing_env_values(monkeypatch, env_name):
    monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
    monkeypatch.setenv(env_name, "warning")

    with suppress_huggingface_warning_logs(verbosity=0):
        assert os.environ[env_name] == "error"

    assert os.environ[env_name] == "warning"


def test_huggingface_warning_context_restores_imported_library_verbosity(monkeypatch):
    transformers_state = _install_fake_huggingface_logging(
        monkeypatch, "transformers", logging.WARNING
    )
    hub_state = _install_fake_huggingface_logging(monkeypatch, "huggingface_hub", logging.INFO)

    with suppress_huggingface_warning_logs(verbosity=0):
        assert transformers_state.level == logging.ERROR
        assert hub_state.level == logging.ERROR

    assert transformers_state.level == logging.WARNING
    assert hub_state.level == logging.INFO


def test_third_party_progress_suppression_disables_datasets_progress(monkeypatch):
    datasets_state = _install_fake_datasets_progress(monkeypatch, enabled=True)

    with suppress_third_party_progress(verbosity=0):
        assert not datasets_state.enabled

    assert datasets_state.enabled


def test_third_party_progress_suppression_disables_huggingface_hub_progress(monkeypatch):
    hub_state = _install_fake_huggingface_hub_progress(monkeypatch, disabled=False)

    with suppress_third_party_progress(verbosity=0):
        assert hub_state.disabled

    assert not hub_state.disabled


def test_third_party_progress_suppression_ignores_datasets_api_failures(monkeypatch):
    package = ModuleType("datasets")
    package.is_progress_bar_enabled = lambda: True

    def fail_disable() -> None:
        raise OSError(1, "Incorrect function")

    package.disable_progress_bars = fail_disable
    monkeypatch.setitem(sys.modules, "datasets", package)

    with suppress_third_party_progress(verbosity=0):
        assert os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"


def test_third_party_progress_suppression_restores_disabled_datasets_progress(monkeypatch):
    datasets_state = _install_fake_datasets_progress(monkeypatch, enabled=False)

    with suppress_third_party_progress(verbosity=0):
        assert not datasets_state.enabled

    assert not datasets_state.enabled


def test_third_party_progress_suppression_sets_official_progress_envs(monkeypatch):
    monkeypatch.delenv("HF_DATASETS_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    with suppress_third_party_progress(verbosity=0):
        assert os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

    assert "HF_DATASETS_DISABLE_PROGRESS_BARS" not in os.environ
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


def test_third_party_progress_suppression_restores_existing_progress_envs(monkeypatch):
    monkeypatch.setenv("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    with suppress_third_party_progress(verbosity=0):
        assert os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"

    assert os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "0"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"


@pytest.mark.parametrize("verbosity", [1, 2])
def test_third_party_progress_suppression_preserves_progress_when_verbose(monkeypatch, verbosity):
    datasets_state = _install_fake_datasets_progress(monkeypatch, enabled=True)
    hub_state = _install_fake_huggingface_hub_progress(monkeypatch, disabled=False)
    monkeypatch.delenv("HF_DATASETS_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    with suppress_third_party_progress(verbosity=verbosity):
        assert datasets_state.enabled
        assert not hub_state.disabled
        assert "HF_DATASETS_DISABLE_PROGRESS_BARS" not in os.environ
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    assert datasets_state.enabled
    assert not hub_state.disabled


def test_show_all_warnings_env_preserves_third_party_progress(monkeypatch):
    datasets_state = _install_fake_datasets_progress(monkeypatch, enabled=True)
    hub_state = _install_fake_huggingface_hub_progress(monkeypatch, disabled=False)
    monkeypatch.setenv("WINMLCLI_SHOW_ALL_WARNINGS", "1")
    monkeypatch.delenv("HF_DATASETS_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

    with suppress_third_party_progress(verbosity=0):
        assert datasets_state.enabled
        assert not hub_state.disabled
        assert "HF_DATASETS_DISABLE_PROGRESS_BARS" not in os.environ
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    assert datasets_state.enabled
    assert not hub_state.disabled


def _install_fake_huggingface_logging(
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    initial_level: int,
) -> SimpleNamespace:
    state = SimpleNamespace(level=initial_level)
    logging_api = SimpleNamespace(
        get_verbosity=lambda: state.level,
        set_verbosity=lambda level: setattr(state, "level", level),
    )
    package = ModuleType(package_name)
    utils = ModuleType(f"{package_name}.utils")
    utils.logging = logging_api
    package.utils = utils

    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.utils", utils)
    return state


def _install_fake_datasets_progress(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> SimpleNamespace:
    state = SimpleNamespace(enabled=enabled)
    package = ModuleType("datasets")
    package.is_progress_bar_enabled = lambda: state.enabled
    package.disable_progress_bars = lambda: setattr(state, "enabled", False)
    package.enable_progress_bars = lambda: setattr(state, "enabled", True)

    monkeypatch.setitem(sys.modules, "datasets", package)
    return state


def _install_fake_huggingface_hub_progress(
    monkeypatch: pytest.MonkeyPatch, *, disabled: bool
) -> SimpleNamespace:
    state = SimpleNamespace(disabled=disabled)
    package = ModuleType("huggingface_hub")
    utils = ModuleType("huggingface_hub.utils")
    utils.are_progress_bars_disabled = lambda: state.disabled
    utils.disable_progress_bars = lambda: setattr(state, "disabled", True)
    utils.enable_progress_bars = lambda: setattr(state, "disabled", False)
    package.utils = utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return state
