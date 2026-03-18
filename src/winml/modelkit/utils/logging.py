"""Logging utilities for ModelKit."""

import logging
import sys


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level based on verbosity flags.

    Args:
        verbose: Enable verbose logging (DEBUG level)
        quiet: Enable quiet mode (ERROR level only)

    Default level is INFO when both flags are False.
    """
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
