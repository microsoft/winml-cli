# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the relocated OpTraceResult + new status/error fields."""

from __future__ import annotations

import json

from winml.modelkit.session.monitor.op_metrics import (
    OperatorMetrics,
    OpTraceResult,
)


def test_model_field_accepts_none():
    """model: str | None — passing None must not raise."""
    r = OpTraceResult(model=None, device="npu", tracing_level="basic")
    assert r.model is None


def test_status_default_is_ok():
    """New status field defaults to 'ok' for backward compat with existing construction."""
    r = OpTraceResult(model="x", device="npu", tracing_level="basic")
    assert r.status == "ok"
    assert r.error is None


def test_status_can_be_set():
    r = OpTraceResult(
        model="x",
        device="npu",
        tracing_level="basic",
        status="parse_failed",
        error="corrupt CSV",
    )
    assert r.status == "parse_failed"
    assert r.error == "corrupt CSV"


def test_to_dict_preserves_nested_schema():
    """Existing nested schema must be preserved.

    The ``metadata`` block must include ``num_samples`` and ``timestamp`` —
    A2-I1 in PR review: a regression that drops either field would silently
    pass an "only check the easy keys" assertion.
    """
    r = OpTraceResult(
        model="m.onnx",
        device="npu",
        tracing_level="basic",
        ep="QNN",
        num_samples=42,
    )
    d = r.to_dict()
    assert "metadata" in d
    assert d["metadata"]["model"] == "m.onnx"
    assert d["metadata"]["device"] == "npu"
    assert d["metadata"]["tracing_level"] == "basic"
    assert d["metadata"]["ep"] == "QNN"
    assert "summary" in d
    assert "operators" in d
    assert "statistics" in d
    assert "artifacts" in d
    # A2-I1: num_samples + timestamp must be in nested metadata.
    assert "num_samples" in d["metadata"]
    assert "timestamp" in d["metadata"]
    assert d["metadata"]["num_samples"] == r.num_samples == 42
    # The timestamp default is an ISO-8601 string from datetime.isoformat().
    assert isinstance(d["metadata"]["timestamp"], str)
    assert d["metadata"]["timestamp"] == r.timestamp
    # Sanity-check it parses as ISO-8601 (drops the 'Z'/offset gracefully).
    from datetime import datetime

    datetime.fromisoformat(d["metadata"]["timestamp"])


def test_to_dict_adds_status_and_error_at_top_level():
    """New fields are additive top-level keys."""
    r = OpTraceResult(
        model="x",
        device="npu",
        tracing_level="basic",
        status="no_data",
        error=None,
    )
    d = r.to_dict()
    assert d["status"] == "no_data"
    assert d["error"] is None


def test_to_json_round_trip():
    r = OpTraceResult(model="x", device="npu", tracing_level="basic", status="ok")
    parsed = json.loads(r.to_json())
    assert parsed["metadata"]["model"] == "x"
    assert parsed["status"] == "ok"


def test_operator_metrics_to_dict_preserved():
    op = OperatorMetrics(name="Conv", op_path="/conv_1", duration_us=12.5, percent_of_total=5.0)
    d = op.to_dict()
    assert d["name"] == "Conv"
    assert d["duration_us"] == 12.5


def test_to_dict_status_only_accepts_known_values_per_typing() -> None:
    """status is a Literal — assert each declared value round-trips through to_dict.

    Python does not enforce ``Literal`` at runtime, so this test verifies *the
    declared values are accepted and serialize correctly*. Static enforcement
    is delegated to mypy / ruff.
    """
    for status in ("ok", "no_data", "parse_failed", "basic_fallback", "not_run"):
        r = OpTraceResult(model=None, device="npu", tracing_level="basic", status=status)
        assert r.to_dict()["status"] == status


def test_trace_status_alias_importable() -> None:
    """``TraceStatus`` must be importable as a public symbol from op_metrics."""
    from winml.modelkit.session.monitor.op_metrics import TraceStatus  # noqa: F401
