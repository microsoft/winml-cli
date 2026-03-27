# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Python environment utilities for managing isolated Python environments using uv."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)


def ensure_uv() -> Path:
    """Ensure uv is installed and return path to executable.

    Returns:
        Path to uv executable

    Raises:
        EnvironmentError: If uv is not installed
    """
    uv_path = shutil.which("uv")
    if uv_path:
        return Path(uv_path)

    raise OSError(
        "uv is not installed.\n\n"
        "Please install uv:\n"
        "  PowerShell: irm https://astral.sh/uv/install.ps1 | iex\n"
        "  Or: pip install uv\n\n"
        "More info: https://docs.astral.sh/uv/getting-started/installation/"
    )


def ensure_venv(
    root_path: str | Path,
    venv_name: str,
    python_version: str,
    requirements: list[str] | None = None,
) -> Path:
    """Ensure a virtual environment exists, creating it if necessary.

    Args:
        root_path: Parent directory for the virtual environment
        venv_name: Name of the virtual environment directory
        python_version: Python version to use (e.g., "3.10")
        requirements: Optional list of package specs to install (e.g., ["numpy>=1.0", "onnx"])

    Returns:
        Path to the Python executable in the virtual environment

    Raises:
        EnvironmentError: If uv is not installed
        RuntimeError: If environment creation fails
    """
    venv_path = Path(root_path) / venv_name
    venv_python = venv_path / "Scripts" / "python.exe"

    # Check if valid environment exists
    if venv_python.exists():
        try:
            result = subprocess.run(  # noqa: S603
                [str(venv_python), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and python_version in result.stdout:
                logger.info(f"Using existing virtual environment at {venv_path}")
                return venv_python
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Invalid environment, ask user before removing
        response = input(
            f"Existing environment at {venv_path} has wrong Python version.\n"
            f"Remove and recreate with Python {python_version}? [y/N]: "
        )
        if response.lower() != "y":
            raise RuntimeError("Cannot use existing environment with wrong Python version")
        # uv --clear will handle the deletion

    # Create virtual environment using uv
    uv = ensure_uv()
    logger.info(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.run(  # noqa: S603
            [str(uv), "venv", str(venv_path), "--python", python_version, "--clear"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create virtual environment: {e.stderr.decode()}") from e

    # Install requirements if provided
    if requirements:
        logger.info(f"Installing {len(requirements)} packages...")
        try:
            subprocess.run(  # noqa: S603
                [str(uv), "pip", "install", "--python", str(venv_python), *requirements],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install requirements: {e.stderr.decode()}") from e

    return venv_python
