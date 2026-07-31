# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Logging utilities for WinML CLI.

Verbosity Convention (adopted from pip, ansible, pytest):
=========================================================

    Flag        Level       Value   Use case
    ----        -----       -----   --------
    -q          ERROR       40      Errors only (quiet / scripting)
    (default)   WARNING     30      Warnings + errors (production default)
    -v          INFO        20      Operational progress messages
    -vv         DEBUG       10      Developer-level tracing
    --debug     DEBUG       10      Alias for -vv (backward compat)

    Formula: level = WARNING - (verbosity * 10)  ->  30, 20, 10
    Quiet:   level = ERROR (40)

All log output goes to stderr so stdout stays clean for structured data
(JSON, compact output, piped commands). Format:

    [%(asctime)s %(levelname)-7s %(name)s] %(message)s

Sample line: ``[14:32:11 INFO    winml.modelkit.export] Loaded config.json``
"""

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING

from .._env import env_flag_enabled


if TYPE_CHECKING:
    from collections.abc import Iterator


_HANDLER_MARKER = "_winml_cli_handler"
_LOG_FORMAT = "[%(asctime)s %(levelname)-7s %(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"
logger = logging.getLogger(__name__)

# Third-party loggers whose INFO/WARNING chatter is noise for CLI users and can
# interleave with rich progress output. Examples: optimum's "No model type passed
# for the task ..." notice when a task maps to several loader classes; onnxscript's
# version-converter fallback WARNING (with a full call stack) that fires when the
# dynamo exporter cannot down-convert a model to the requested opset; and torch's
# own one-line "Setting ONNX exporter to use operator set version 18 ..." notice for
# the same down-convert case -- winml already surfaces a concise opset warning, so
# both are redundant. They are floored at ERROR in normal output and only follow the
# CLI level once the user passes -v/-vv.
_NOISY_LIBRARY_LOGGERS = (
    "optimum",
    "onnxscript.version_converter",
    "torch.onnx._internal.exporter._compat",
)

_HUGGINGFACE_WARNING_LOGGERS = (
    "huggingface_hub",
    "transformers",
)
_HUGGINGFACE_VERBOSITY_ENVS = ("TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY")
_HUGGINGFACE_WARNING_MODULE_RE = r"(huggingface_hub|transformers)(\.|$).*"
_PROGRESS_ENV_OVERRIDES = {
    "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}


def configure_logging(
    verbosity: int = 0,
    quiet: bool = False,
    *,
    # Backward-compat: accept old bool signature
    verbose: bool = False,
) -> None:
    """Configure root logger based on verbosity level.

    Idempotent: subcommands re-call this after merging top-level + subcommand
    ``-v``/``-q``. The first call installs the WinML stderr handler; later
    calls only adjust the level. Existing non-WinML handlers (notably pytest's
    ``caplog`` propagate-handler) are preserved.

    Args:
        verbosity: Number of ``-v`` flags (0=WARNING, 1=INFO, 2+=DEBUG).
        quiet: If True, override to ERROR level regardless of verbosity.
        verbose: **Deprecated bool compat** — treated as verbosity=1 when
                 True and verbosity is 0. Existing callers that pass
                 ``verbose=True`` keep working without changes.
    """
    verbosity = _normalize_verbosity(verbosity, verbose)
    log_level = _cli_log_level(verbosity, quiet)

    root = logging.getLogger()
    # Drop any prior WinML handler and install a fresh one bound to the
    # *current* ``sys.stderr``. Click's ``CliRunner.invoke()`` swaps the
    # process stderr for each test, so a cached handler from an earlier
    # invocation would write to a stream the test no longer captures.
    # We leave non-WinML handlers (notably pytest's caplog handler) alone.
    for h in list(root.handlers):
        if getattr(h, _HANDLER_MARKER, False):
            root.removeHandler(h)
    own_handler = logging.StreamHandler(sys.stderr)
    own_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    setattr(own_handler, _HANDLER_MARKER, True)
    root.addHandler(own_handler)
    # The root level is the sole gate: it already filters every record before
    # it reaches any handler, so the handler is left at NOTSET (passes through)
    # to avoid a redundant double-filter at the same threshold. This mirrors
    # the prior ``logging.basicConfig`` behavior, which never set a handler level.
    root.setLevel(log_level)

    # Keep noisy third-party library chatter out of normal output. Their loggers float
    # up to the CLI level only when the user opts into verbosity (-v/-vv) or explicitly
    # asks for all warnings; otherwise they are pinned at ERROR so library notices never
    # leak into / interleave with output. (verbose=True is folded into verbosity above,
    # so verbosity > 0 covers it.)
    show_all_warnings = env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS")
    library_level = log_level if verbosity > 0 or show_all_warnings else logging.ERROR
    for name in _NOISY_LIBRARY_LOGGERS:
        logging.getLogger(name).setLevel(library_level)


@contextmanager
def suppress_huggingface_warning_logs(
    verbosity: int = 0,
    quiet: bool = False,
    *,
    verbose: bool = False,
) -> "Iterator[None]":
    """Temporarily hide Hugging Face warning chatter for model loading."""
    verbosity = _normalize_verbosity(verbosity, verbose)
    log_level = _cli_log_level(verbosity, quiet)
    show_all_warnings = env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS")
    huggingface_level = log_level if verbosity > 0 or show_all_warnings else logging.ERROR
    hide_python_warnings = verbosity == 0 and not show_all_warnings

    saved_logger_levels = {
        name: logging.getLogger(name).level for name in _HUGGINGFACE_WARNING_LOGGERS
    }
    saved_env: dict[str, str | None] = {
        name: os.environ.get(name) for name in _HUGGINGFACE_VERBOSITY_ENVS
    }
    saved_library_verbosity = _get_imported_huggingface_verbosity()

    try:
        for name in _HUGGINGFACE_WARNING_LOGGERS:
            logging.getLogger(name).setLevel(huggingface_level)

        library_verbosity = _library_verbosity_name(huggingface_level)
        for env_name in _HUGGINGFACE_VERBOSITY_ENVS:
            os.environ[env_name] = library_verbosity
        _sync_imported_huggingface_verbosity(huggingface_level)
        if hide_python_warnings:
            with warnings.catch_warnings():
                for category in (FutureWarning, DeprecationWarning, UserWarning):
                    warnings.filterwarnings(
                        "ignore",
                        category=category,
                        module=_HUGGINGFACE_WARNING_MODULE_RE,
                    )
                yield
        else:
            yield
    finally:
        for env_name, saved_value in saved_env.items():
            if saved_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = saved_value
        _restore_imported_huggingface_verbosity(saved_library_verbosity)
        for name, level in saved_logger_levels.items():
            logging.getLogger(name).setLevel(level)


@contextmanager
def suppress_third_party_progress(
    verbosity: int = 0,
    quiet: bool = False,
    *,
    verbose: bool = False,
) -> "Iterator[None]":
    """Temporarily hide third-party tqdm/datasets progress in normal output."""
    verbosity = _normalize_verbosity(verbosity, verbose)
    show_all_warnings = env_flag_enabled("WINMLCLI_SHOW_ALL_WARNINGS")
    if verbosity > 0 or show_all_warnings:
        yield
        return

    saved_env: dict[str, str | None] = {
        name: os.environ.get(name) for name in _PROGRESS_ENV_OVERRIDES
    }
    saved_hub_progress: bool | None = None
    saved_datasets_progress: bool | None = None

    try:
        for name, env_value in _PROGRESS_ENV_OVERRIDES.items():
            os.environ[name] = env_value
        saved_hub_progress = _disable_imported_huggingface_hub_progress()
        saved_datasets_progress = _disable_imported_datasets_progress()
        yield
    finally:
        _restore_imported_datasets_progress(saved_datasets_progress)
        _restore_imported_huggingface_hub_progress(saved_hub_progress)
        for name, saved_value in saved_env.items():
            if saved_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved_value


def _normalize_verbosity(verbosity: int, verbose: bool) -> int:
    # Backward compat: bool verbose -> int, also handles count passthrough.
    if verbose and verbosity == 0:
        return int(verbose)
    return verbosity


def _cli_log_level(verbosity: int, quiet: bool) -> int:
    # Clamp between DEBUG (10) and WARNING (30); quiet overrides to ERROR.
    return logging.ERROR if quiet else max(logging.DEBUG, logging.WARNING - verbosity * 10)


def _library_verbosity_name(level: int) -> str:
    if level <= logging.DEBUG:
        return "debug"
    if level <= logging.INFO:
        return "info"
    if level <= logging.WARNING:
        return "warning"
    if level <= logging.ERROR:
        return "error"
    return "critical"


def _disable_imported_datasets_progress() -> bool | None:
    datasets = sys.modules.get("datasets")
    if datasets is None:
        return None

    is_enabled = getattr(datasets, "is_progress_bar_enabled", None)
    try:
        saved_enabled = is_enabled() if callable(is_enabled) else None
    except Exception:
        logger.debug("Could not read datasets progress-bar state", exc_info=True)
        saved_enabled = None
    disable = getattr(datasets, "disable_progress_bars", None)
    if callable(disable):
        try:
            disable()
        except Exception:
            logger.debug("Could not disable datasets progress bars", exc_info=True)
    return saved_enabled


def _restore_imported_datasets_progress(saved_enabled: bool | None) -> None:
    if saved_enabled is None:
        return
    datasets = sys.modules.get("datasets")
    if datasets is None:
        return

    method_name = "enable_progress_bars" if saved_enabled else "disable_progress_bars"
    restore = getattr(datasets, method_name, None)
    if callable(restore):
        try:
            restore()
        except Exception:
            logger.debug("Could not restore datasets progress-bar state", exc_info=True)


def _disable_imported_huggingface_hub_progress() -> bool | None:
    try:
        hub_utils = import_module("huggingface_hub.utils")
    except ImportError:
        return None
    are_progress_bars_disabled = getattr(hub_utils, "are_progress_bars_disabled", None)
    disable_progress_bars = getattr(hub_utils, "disable_progress_bars", None)
    if not callable(are_progress_bars_disabled) or not callable(disable_progress_bars):
        return None

    try:
        saved_disabled = bool(are_progress_bars_disabled())
    except Exception:
        logger.debug("Could not read Hugging Face Hub progress-bar state", exc_info=True)
        saved_disabled = None
    try:
        disable_progress_bars()
    except Exception:
        logger.debug("Could not disable Hugging Face Hub progress bars", exc_info=True)
    return saved_disabled


def _restore_imported_huggingface_hub_progress(saved_disabled: bool | None) -> None:
    if saved_disabled is None:
        return
    try:
        hub_utils = import_module("huggingface_hub.utils")
    except ImportError:
        return

    method_name = "disable_progress_bars" if saved_disabled else "enable_progress_bars"
    restore = getattr(hub_utils, method_name, None)
    if not callable(restore):
        return
    try:
        restore()
    except Exception:
        logger.debug("Could not restore Hugging Face Hub progress-bar state", exc_info=True)


def _sync_imported_huggingface_verbosity(verbosity: int) -> None:
    if sys.modules.get("transformers") is not None:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity(verbosity)

    if sys.modules.get("huggingface_hub") is not None:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity(verbosity)


def _get_imported_huggingface_verbosity() -> dict[str, int]:
    saved: dict[str, int] = {}
    if sys.modules.get("transformers") is not None:
        from transformers.utils import logging as transformers_logging

        saved["transformers"] = transformers_logging.get_verbosity()

    if sys.modules.get("huggingface_hub") is not None:
        from huggingface_hub.utils import logging as hub_logging

        saved["huggingface_hub"] = hub_logging.get_verbosity()

    return saved


def _restore_imported_huggingface_verbosity(saved: dict[str, int]) -> None:
    if "transformers" in saved and sys.modules.get("transformers") is not None:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity(saved["transformers"])

    if "huggingface_hub" in saved and sys.modules.get("huggingface_hub") is not None:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity(saved["huggingface_hub"])


def flush_ort_startup_logs() -> None:
    """No-op kept for backward compatibility.

    ORT startup stderr is now discarded to devnull (not captured), so there
    is nothing to replay.
    """
