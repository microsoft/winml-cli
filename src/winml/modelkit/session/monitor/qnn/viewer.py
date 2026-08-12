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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..op_metrics import TraceFallbackReason


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

# Documented common installation directories (Windows).
_COMMON_SDK_PATHS: list[Path] = [
    Path(r"D:\QC"),
    Path(r"C:\Qualcomm\AIStack\qairt"),
]
_OPTRACE_READER_NAME = "QnnHtpOptraceProfilingReader.dll"
_QHAS_SUMMARY_SUFFIX = "_qnn_htp_analysis_summary.json"


@dataclass(frozen=True)
class QHASViewerResult:
    """Detailed outcome of QHAS viewer preparation and execution."""

    path: Path | None
    failure_reason: TraceFallbackReason | None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.failure_reason is None):
            raise ValueError("QHAS viewer result requires exactly one outcome")


def find_qnn_sdk() -> Path | None:
    """Auto-detect a QNN SDK from the environment or documented common roots.

    ``QNN_SDK_ROOT`` wins when it identifies a valid directory. An unset or
    invalid environment value falls through to versioned SDK directories under
    the documented QAIRT/QNN installation roots.
    """
    env_root = os.environ.get("QNN_SDK_ROOT")
    if env_root:
        root = Path(env_root)
        if root.is_dir():
            return root

    for base_path in _COMMON_SDK_PATHS:
        if not base_path.is_dir():
            continue
        for child in sorted(base_path.iterdir(), reverse=True):
            if child.is_dir() and (child / "bin").is_dir():
                return child
        if (base_path / "bin").is_dir():
            return base_path

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


def _find_optrace_reader(viewer: Path) -> Path | None:
    """Locate the optrace reader matching the selected viewer architecture."""
    bin_dir = viewer.parent
    if bin_dir.parent.name == "bin":
        # Architecture-specific layout: bin/<arch> mirrors lib/<arch>.
        sdk_root = bin_dir.parent.parent
        candidate = sdk_root / "lib" / bin_dir.name / _OPTRACE_READER_NAME
    else:
        # Flat layout: bin/qnn-profile-viewer.exe pairs with lib/<reader>.
        sdk_root = bin_dir.parent
        candidate = sdk_root / "lib" / _OPTRACE_READER_NAME
    return candidate if candidate.is_file() else None


def run_basic_viewer(
    qnn_log: Path,
    output: Path,
    *,
    sdk_root: Path | None = None,
) -> Path | None:
    """Run qnn-profile-viewer for basic CSV output."""
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

    return output if output.is_file() else None


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
        Output prefix passed to the viewer. The optrace reader appends its QHAS
        artifact suffixes to this path.
    config:
        Post-processing features config.  Uses default if ``None``.
    sdk_root:
        Override SDK root (auto-detected when ``None``).

    Returns:
    -------
    Path to the generated QNN HTP analysis summary JSON, or ``None`` on failure.
    """
    return run_qhas_viewer_result(
        qnn_log,
        schematic,
        output,
        config,
        sdk_root=sdk_root,
    ).path


def run_qhas_viewer_result(
    qnn_log: Path,
    schematic: Path,
    output: Path,
    config: dict[str, Any] | None = None,
    *,
    sdk_root: Path | None = None,
) -> QHASViewerResult:
    """Run QHAS viewer and distinguish execution from missing-output failures."""
    try:
        viewer = _find_viewer_exe(sdk_root)
        if viewer is None:
            logger.warning(
                "qnn-profile-viewer not found; set QNN_SDK_ROOT to enable detail mode "
                "(falling back to basic CSV)"
            )
            return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.VIEWER_FAILED)
        reader = _find_optrace_reader(viewer)
        if reader is None:
            logger.warning(
                "%s not found for qnn-profile-viewer at %s; falling back to basic CSV",
                _OPTRACE_READER_NAME,
                viewer,
            )
            return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.VIEWER_FAILED)

        if not schematic.is_file():
            logger.warning("Schematic file not found: %s", schematic)
            return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.VIEWER_FAILED)

        cfg = config if config is not None else _DEFAULT_CONFIG
        config_path = output.with_name(f"{output.stem}_optrace_config.json")
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cmd = [
            str(viewer),
            "--input_log",
            str(qnn_log),
            "--output",
            str(output),
            "--reader",
            str(reader),
            "--schematic",
            str(schematic),
            "--config",
            str(config_path),
        ]
        logger.info("Running QHAS viewer: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        logger.error("QHAS viewer failed: %s", exc.stderr)
        return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.VIEWER_FAILED)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("QHAS viewer preparation or execution failed: %s", exc)
        return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.VIEWER_FAILED)

    summary_output = output.with_name(f"{output.stem}{_QHAS_SUMMARY_SUFFIX}")
    try:
        if summary_output.is_file():
            return QHASViewerResult(path=summary_output, failure_reason=None)
    except OSError as exc:
        logger.warning("Could not inspect QHAS analysis summary %s: %s", summary_output, exc)
    logger.warning("QHAS viewer did not produce analysis summary: %s", summary_output)
    return QHASViewerResult(path=None, failure_reason=TraceFallbackReason.QHAS_OUTPUT_MISSING)
