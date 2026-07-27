# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``_monitor_to_json_dict`` dispatch helper (v2.4).

Bundle B addition: pin error containment on the JSON-serialization path.
A regression in any monitor's serializer must not crash ``wmk perf``
mid-output after the benchmark already ran — the dispatcher logs WARNING
and surfaces a sentinel ``{"error": "monitor_serialization_failed: ..."}``
dict so the JSON report still serialises successfully.
"""

from __future__ import annotations

from typing import Any


class _ExplodingMonitor:
    """Fake monitor whose ``to_dict()`` always raises.

    Models a transitional proof-of-execution monitor (VitisAI / OpenVINO
    style) where ``monitor.result`` is ``None`` and the dispatcher falls
    through to ``hasattr(monitor, "to_dict")`` — and that ``to_dict()``
    has regressed.
    """

    def __init__(self) -> None:
        self.result: Any = None

    def to_dict(self) -> dict[str, Any]:
        raise RuntimeError("simulated serializer regression")


def test_monitor_to_json_dict_swallows_to_dict_exception():
    """When ``monitor.to_dict()`` raises, dispatcher returns error dict, not crash."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    out = _monitor_to_json_dict(_ExplodingMonitor())

    assert isinstance(out, dict)
    assert "error" in out
    assert "monitor_serialization_failed" in out["error"]


class _ExplodingResultMonitor:
    """Fake monitor whose ``result.to_dict()`` raises.

    Models the op-tracing path: ``monitor.result`` is non-``None`` so the
    dispatcher takes the L1 branch and calls ``result.to_dict()`` — which
    has regressed.
    """

    class _BadResult:
        def to_dict(self) -> dict[str, Any]:
            raise ValueError("bad result")

    def __init__(self) -> None:
        self.result: Any = self._BadResult()


def test_monitor_to_json_dict_swallows_result_to_dict_exception():
    """When ``result.to_dict()`` raises, dispatcher returns error dict, not crash."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    out = _monitor_to_json_dict(_ExplodingResultMonitor())

    assert isinstance(out, dict)
    assert "error" in out
    assert "monitor_serialization_failed" in out["error"]


class _NullishMonitor:
    """Monitor with no result and no to_dict — should return {}."""

    def __init__(self) -> None:
        self.result: Any = None


def test_monitor_to_json_dict_empty_when_no_data_and_no_to_dict():
    """Sanity check: NullEPMonitor-style returns ``{}`` (no error)."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    out = _monitor_to_json_dict(_NullishMonitor())

    assert out == {}


# =============================================================================
# Bundle D — happy-path coverage for the three dispatch branches (CRIT-6A).
#
# Bundle B (above) pinned the failure-mode behaviour: any serializer regression
# is contained as a sentinel ``error`` dict.  Bundle D adds the matching
# happy-path lock so a future reorder of the dispatch precedence (e.g.
# ``hasattr(monitor, "to_dict")`` checked before ``monitor.result``) gets
# caught immediately — and the failing test name tells us *which* branch
# regressed.
# =============================================================================


# ---- Branch 1: op-tracing monitor (result is not None) ----


class _OpTracingMonitor:
    """Fake monitor whose ``result`` is a real :class:`OpTraceResult`.

    Deliberately omits ``to_dict()`` so this fake also verifies that the
    dispatcher takes the ``result`` branch on its own merit — not because
    it falls through to a legacy ``to_dict()``.
    """

    def __init__(self) -> None:
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        self.result: Any = OpTraceResult(
            model="test/model",
            device="NPU",
            tracing_level="basic",
            operators=[],
            ep="QNN",
            tracing_backend="qnn",
            num_samples=1,
            summary={"hvx_threads": 4},
        )


def test_monitor_to_json_dict_returns_op_trace_result_dict():
    """Branch 1: ``monitor.result`` set → dispatch returns ``result.to_dict()``."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    monitor = _OpTracingMonitor()
    out = _monitor_to_json_dict(monitor)

    assert isinstance(out, dict)
    # OpTraceResult.to_dict() carries a "metadata" wrapper around model/
    # device/ep/etc.
    assert "metadata" in out
    assert out["metadata"]["device"] == "NPU"
    assert out["metadata"]["ep"] == "QNN"
    assert "summary" in out
    assert out["summary"]["hvx_threads"] == 4
    # Success path: ``OpTraceResult.to_dict()`` emits ``status="ok"`` and
    # ``error=None`` additively.  Distinguish this from the Bundle B
    # containment sentinel (``{"error": "monitor_serialization_failed: ..."}``)
    # by asserting on both fields.
    assert out["status"] == "ok"
    assert out["error"] is None


# ---- Branch 2: proof-of-execution monitor (result is None, has to_dict()) ----


class _ProofMonitor:
    """Fake VitisAI/OpenVINO-style monitor: no ``result``, has ``to_dict()``."""

    def __init__(self) -> None:
        self.result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"ep": "VitisAI", "npu_proven": True}


def test_monitor_to_json_dict_falls_through_to_legacy_to_dict():
    """Branch 2: ``result`` is None but ``to_dict()`` exists → returns ``monitor.to_dict()``."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    monitor = _ProofMonitor()
    out = _monitor_to_json_dict(monitor)

    assert out == {"ep": "VitisAI", "npu_proven": True}


# ---- Branch 3: null monitor (result is None, no to_dict()) ----


class _BareMonitor:
    """Fake :class:`NullEPMonitor`-style monitor: no ``result``, no ``to_dict()``."""

    def __init__(self) -> None:
        self.result: Any = None


def test_monitor_to_json_dict_returns_empty_for_null_monitor():
    """Branch 3: ``result`` is None and no ``to_dict()`` → dispatch returns ``{}``."""
    from winml.modelkit.commands.perf import _monitor_to_json_dict

    monitor = _BareMonitor()
    out = _monitor_to_json_dict(monitor)

    assert out == {}
