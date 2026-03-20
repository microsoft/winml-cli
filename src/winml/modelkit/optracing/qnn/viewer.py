# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Wrapper for qnn-profile-viewer.exe post-processing tool.

The QNN profile viewer converts raw profiling logs into human-readable
CSV and QHAS (QNN Hardware Acceleration Summary) JSON artifacts.  Two
modes are supported:

- **basic**: runs the viewer with ``--input_log`` only, producing a CSV
  summarising per-operator cycle counts.
- **detail** (optrace): additionally feeds a schematic binary and an
  optrace-reader config to produce full QHAS JSON with roofline, DMA
  traffic, and memory information.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Default QHAS post-processing features.
_DEFAULT_CONFIG: dict[str, Any] = {
    "features": {
        "qhas_json": True,
        "qhas_schema": True,
        "htp_json": True,
        "runtrace": True,
        "memory_info": True,
        "traceback": True,
        "enable_input_output_flow_events": True,
        "enable_sequencer_flow_events": True,
    }
}

# Common SDK installation directories (Windows).
_COMMON_SDK_PATHS: list[str] = [
    r"D:\QC",
    r"C:\Qualcomm\AIStack\qairt",
]


def find_qnn_sdk() -> Path | None:
    """Auto-detect QNN SDK installation.

    Resolution order:
    1. ``QNN_SDK_ROOT`` environment variable.
    2. Common installation directories on Windows.

    Returns the SDK root ``Path`` or ``None`` when not found.
    """
    env_root = os.environ.get("QNN_SDK_ROOT")
    if env_root:
        root = Path(env_root)
        if root.is_dir():
            logger.debug("QNN SDK found via QNN_SDK_ROOT: %s", root)
            return root

    for base in _COMMON_SDK_PATHS:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        # Look for a versioned subdirectory containing bin/
        for child in sorted(base_path.iterdir(), reverse=True):
            if child.is_dir() and (child / "bin").is_dir():
                logger.debug("QNN SDK found at: %s", child)
                return child

    logger.debug("QNN SDK not found")
    return None


def _find_viewer_exe(sdk_root: Path | None = None) -> Path | None:
    """Locate ``qnn-profile-viewer.exe`` within the SDK."""
    if sdk_root is None:
        sdk_root = find_qnn_sdk()
    if sdk_root is None:
        return None

    # Expected location: <sdk_root>/bin/<arch>/qnn-profile-viewer.exe
    bin_dir = sdk_root / "bin"
    if not bin_dir.is_dir():
        return None

    for arch_dir in bin_dir.iterdir():
        candidate = arch_dir / "qnn-profile-viewer.exe"
        if candidate.is_file():
            return candidate

    # Fallback: direct child of bin/
    candidate = bin_dir / "qnn-profile-viewer.exe"
    if candidate.is_file():
        return candidate

    return None


def run_basic_viewer(
    qnn_log: Path,
    output: Path,
    *,
    sdk_root: Path | None = None,
) -> Path | None:
    """Run qnn-profile-viewer for basic CSV output.

    Parameters
    ----------
    qnn_log:
        Path to the ``*_qnn.log`` file produced by QNN EP profiling.
    output:
        Path for the resulting CSV file.
    sdk_root:
        Override SDK root (auto-detected when ``None``).

    Returns:
    -------
    Path to the generated CSV, or ``None`` on failure.
    """
    viewer = _find_viewer_exe(sdk_root)
    if viewer is None:
        logger.warning("qnn-profile-viewer not found; skipping basic viewer")
        return None

    cmd = [
        str(viewer),
        "--input_log",
        str(qnn_log),
        "--output",
        str(output),
    ]
    logger.info("Running basic viewer: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        logger.error("Basic viewer failed: %s", exc.stderr)
        return None
    except FileNotFoundError:
        logger.error("qnn-profile-viewer executable not found at %s", viewer)
        return None

    if output.is_file():
        return output
    return None


def run_qhas_viewer(
    qnn_log: Path,
    schematic: Path,
    output: Path,
    config: dict[str, Any] | None = None,
    *,
    sdk_root: Path | None = None,
) -> Path | None:
    """Run qnn-profile-viewer with optrace reader for QHAS output.

    Parameters
    ----------
    qnn_log:
        Path to the ``*_qnn.log`` file.
    schematic:
        Path to the ``*_schematic.bin`` file.
    output:
        Path for the resulting QHAS JSON file.
    config:
        Post-processing features config.  Uses default if ``None``.
    sdk_root:
        Override SDK root (auto-detected when ``None``).

    Returns:
    -------
    Path to the generated QHAS JSON, or ``None`` on failure.
    """
    viewer = _find_viewer_exe(sdk_root)
    if viewer is None:
        logger.warning("qnn-profile-viewer not found; skipping QHAS viewer")
        return None

    if not schematic.is_file():
        logger.warning("Schematic file not found: %s", schematic)
        return None

    # Write config to a temporary JSON next to the output.
    cfg = config if config is not None else _DEFAULT_CONFIG
    config_path = output.parent / "optrace_config.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cmd = [
        str(viewer),
        "--input_log",
        str(qnn_log),
        "--output",
        str(output),
        "--reader",
        "optrace",
        "--schematic",
        str(schematic),
        "--config",
        str(config_path),
    ]
    logger.info("Running QHAS viewer: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        logger.error("QHAS viewer failed: %s", exc.stderr)
        return None
    except FileNotFoundError:
        logger.error("qnn-profile-viewer executable not found at %s", viewer)
        return None

    if output.is_file():
        return output
    return None
