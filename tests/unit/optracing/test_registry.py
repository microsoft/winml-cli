# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test OpTracer registry: registration, lookup, and EP pattern matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winml.modelkit.optracing import OpTracer, OpTraceResult, get_tracer, register_tracer
from winml.modelkit.optracing.registry import _TRACERS  # Testing internal implementation


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _MockTracer(OpTracer):
    """Concrete OpTracer for testing."""

    def run(
        self,
        onnx_path: Path,
        *,
        iterations: int = 5,
        warmup: int = 2,
        output_dir: Path | None = None,
    ) -> OpTraceResult:
        return OpTraceResult(
            model=onnx_path.name,
            device="mock",
            tracing_level="basic",
        )

    def is_available(self) -> bool:
        return True


class _AnotherMockTracer(OpTracer):
    """A second mock tracer for multi-level tests."""

    def run(
        self,
        onnx_path: Path,
        *,
        iterations: int = 5,
        warmup: int = 2,
        output_dir: Path | None = None,
    ) -> OpTraceResult:
        return OpTraceResult(
            model=onnx_path.name,
            device="mock2",
            tracing_level="detail",
        )

    def is_available(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the registry around each test."""
    snapshot = {k: dict(v) for k, v in _TRACERS.items()}
    yield
    _TRACERS.clear()
    _TRACERS.update(snapshot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_get_tracer():
    """Register a mock tracer and retrieve it."""
    register_tracer("MOCK", "basic", _MockTracer)
    cls = get_tracer("MOCK", "basic")
    assert cls is _MockTracer


def test_get_tracer_not_found():
    """Return None for unregistered EP/level."""
    assert get_tracer("NonExistent", "basic") is None
    assert get_tracer("MOCK", "unknown_level") is None


def test_ep_pattern_matching():
    """'QNN' pattern matches 'QNNExecutionProvider'."""
    register_tracer("QNN", "basic", _MockTracer)
    cls = get_tracer("QNNExecutionProvider", "basic")
    assert cls is _MockTracer


def test_register_multiple_levels():
    """Same EP can have different tracers for basic/detail."""
    register_tracer("MOCK", "basic", _MockTracer)
    register_tracer("MOCK", "detail", _AnotherMockTracer)

    assert get_tracer("MOCK", "basic") is _MockTracer
    assert get_tracer("MOCK", "detail") is _AnotherMockTracer


def test_default_qnn_tracers_registered():
    """The auto-registered QNN tracers should be present."""
    from winml.modelkit.optracing.qnn.profiler import QNNProfiler

    basic_cls = get_tracer("QNN", "basic")
    detail_cls = get_tracer("QNN", "detail")

    assert basic_cls is QNNProfiler
    assert detail_cls is QNNProfiler


def test_default_cpu_tracer_registered():
    """The auto-registered CPU basic tracer should be present."""
    from winml.modelkit.optracing.cpu.profiler import CPUProfiler

    assert get_tracer("CPUExecutionProvider", "basic") is CPUProfiler


def test_pattern_substring_not_exact():
    """Pattern matching uses substring, not exact match."""
    register_tracer("Custom", "basic", _MockTracer)

    # "Custom" is a substring of "CustomExecutionProvider"
    assert get_tracer("CustomExecutionProvider", "basic") is _MockTracer
    # But "CustomOther" should NOT match "Custom" if "Custom" is in "CustomOther"
    # Actually substring: "Custom" IS in "CustomOther", so it should match.
    assert get_tracer("CustomOther", "basic") is _MockTracer
    # "Cust" should NOT match pattern "Custom"
    assert get_tracer("Cust", "basic") is None
