# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""v2.4 EPMonitor extensions: ``set_onnx_op_types`` and ``result`` property.

Phase 1 additive contract. Verifies that:

* :class:`NullEPMonitor` inherits the concrete no-op default for
  :meth:`set_onnx_op_types` and the ``None`` default for the ``result``
  property — i.e. non-op-tracing monitors silently ignore the map and
  expose no result.
* Subclasses can populate ``self._result`` and have it surface through
  the property.
"""

from __future__ import annotations

from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor


def test_null_monitor_set_onnx_op_types_is_no_op() -> None:
    """NullEPMonitor inherits the no-op default and ignores the map."""
    mon = NullEPMonitor()
    mon.set_onnx_op_types({"/foo/Conv": "Conv"})  # must not raise
    # No state mutation expected; result still None
    assert mon.result is None


def test_null_monitor_result_default_none() -> None:
    """NullEPMonitor inherits the result-default-None getattr."""
    mon = NullEPMonitor()
    assert mon.result is None


def test_set_onnx_op_types_accepts_empty_dict() -> None:
    """Empty map is a valid input."""
    mon = NullEPMonitor()
    mon.set_onnx_op_types({})
    assert mon.result is None


def test_set_onnx_op_types_returns_none() -> None:
    """The default contract returns None (no return value)."""
    mon = NullEPMonitor()
    assert mon.set_onnx_op_types({"a": "Add"}) is None


def test_result_returns_self_dot_result_when_set() -> None:
    """If a subclass sets self._result, the property returns it."""

    class _FakeMonitor(NullEPMonitor):
        def __init__(self) -> None:
            super().__init__()
            self._result = "sentinel"  # type: ignore[assignment]

    mon = _FakeMonitor()
    assert mon.result == "sentinel"


def test_result_falls_back_when_subclass_omits_result_attr() -> None:
    """A subclass that never sets self._result still gets None via getattr."""

    class _NoResultMonitor(NullEPMonitor):
        pass

    mon = _NoResultMonitor()
    assert mon.result is None


def test_subclass_can_override_set_onnx_op_types() -> None:
    """Subclasses override the no-op default to capture the map."""

    class _CapturingMonitor(NullEPMonitor):
        def __init__(self) -> None:
            super().__init__()
            self.captured: dict[str, str] | None = None

        def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
            self.captured = dict(onnx_op_types)

    mon = _CapturingMonitor()
    mon.set_onnx_op_types({"/layer/Conv": "Conv", "/layer/Relu": "Relu"})
    assert mon.captured == {"/layer/Conv": "Conv", "/layer/Relu": "Relu"}
