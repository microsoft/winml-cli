# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Failure classification via stdout+stderr pattern matching."""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """Failure taxonomy — ordered by pipeline stage."""

    EXPORT_FAIL = "EXPORT_FAIL"
    ANALYZER_BLOCK = "ANALYZER_BLOCK"
    OPT_FAIL = "OPT_FAIL"
    COMPILE_FAIL = "COMPILE_FAIL"
    RUNTIME_FAIL = "RUNTIME_FAIL"
    ENVIRONMENT = "ENVIRONMENT"  # disk/network/resource — retryable
    TIMEOUT = "TIMEOUT"  # exceeded per-model time limit
    UNKNOWN = "UNKNOWN"


# Ordered by pipeline stage — first match wins.
# All patterns are lowercase (matching is case-insensitive).
# Pattern sources:
#   modelkit/build/hf.py          ("compilation failed", "quantization failed")
#   modelkit/build/common.py      ("black nodes persist")
#   modelkit/session/session.py   (compilationerror, inferenceerror)
CLASSIFICATION_RULES: list[tuple[FailureType, list[str]]] = [
    (
        FailureType.EXPORT_FAIL,
        [
            "torch.onnx",
            "onnx export",
            "torch.jit",
            "opset",
            "export_onnx",
            "exporting to onnx",
            "onnx_program",
        ],
    ),
    (
        FailureType.ANALYZER_BLOCK,
        [
            "black nodes persist",
            "static_analyzer",
            "supportlevel.black",
        ],
    ),
    (
        FailureType.OPT_FAIL,
        [
            "optimize_onnx",
            "optimizing onnx",
            "shape_infer",
            "shapeinferenceerror",
            "graph_optimization",
        ],
    ),
    (
        FailureType.COMPILE_FAIL,
        [
            "compilation failed",
            "compilationerror",
            "quantization failed",
            "compile_onnx",
        ],
    ),
    (
        FailureType.RUNTIME_FAIL,
        [
            "inferenceerror",
            "inference failed",
            "out of memory",
            "memoryerror",
            "cuda error",
        ],
    ),
    (
        FailureType.ENVIRONMENT,
        [
            "no space left on device",
            "errno 28",
            "disk full",
            "connectionerror",
            "httpsconnectionpool",
            "requests.exceptions",
            "socket.timeout",
            "readtimeouterror",
        ],
    ),
]


def classify_failure(combined_output: str, exit_code: int) -> FailureType:
    """Classify failure from combined stdout+stderr output.

    Uses case-insensitive substring matching, ordered by pipeline stage.
    First match wins. Exit code 3 + no match defaults to EXPORT_FAIL.
    """
    lower = combined_output.lower()
    for failure_type, patterns in CLASSIFICATION_RULES:
        for pattern in patterns:
            if pattern in lower:
                return failure_type
    # Exit code 3 = file not found (model loading is pre-export)
    return FailureType.EXPORT_FAIL if exit_code == 3 else FailureType.UNKNOWN
