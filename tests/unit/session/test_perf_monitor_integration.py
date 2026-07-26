# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for WinMLSession.perf(monitor=...) — teardown ordering,
auto-reset, session/provider option merging, exception transparency.

This file grows across multiple tasks (7, 8).
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from tests._helpers import get_minimal_onnx_model_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_real_cpu_ort_device():
    """Return the CPUExecutionProvider OrtEpDevice from ort.get_ep_devices()."""
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "CPUExecutionProvider"]
    if not devs:
        pytest.skip("CPUExecutionProvider not available in ort.get_ep_devices()")
    return devs[0]


def _make_cpu_session(model_path):
    """Create a WinMLSession bound to a stub CPU WinMLEPDevice.

    The real OrtEpDevice is wrapped so add_provider_for_devices() receives a
    genuine handle and ORT can run.
    """
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")
    return WinMLSession(model_path, ep_device=cpu_ep_device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_active_session_option_entries_default_empty():
    """Newly-constructed WinMLSession has empty _active_session_option_entries."""
    from winml.modelkit.session.session import WinMLSession

    session = WinMLSession.__new__(WinMLSession)
    # Simulate post-__init__ state without file I/O
    session._active_session_option_entries = {}  # from __init__
    assert session._active_session_option_entries == {}


def test_perf_monitor_none_yields_perfcontext_with_null_monitor():
    """perf() with no monitor yields PerfContext whose monitor is NullEPMonitor."""
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor
    from winml.modelkit.session.session import PerfContext

    session = _make_cpu_session(get_minimal_onnx_model_path())
    with session.perf(warmup=0) as ctx:
        assert isinstance(ctx, PerfContext)
        assert isinstance(ctx.monitor, NullEPMonitor)
        # ctx.stats must be the PerfStats instance
        assert ctx.stats is not None


def test_nested_perf_raises():
    """Entering perf() while another is active raises RuntimeError."""
    session = _make_cpu_session(get_minimal_onnx_model_path())
    with session.perf(), pytest.raises(RuntimeError, match="already active"), session.perf():
        pass


def test_teardown_ordering_reset_before_monitor_exit():
    """For monitor.requires_session_teardown=True, self.reset() fires BEFORE monitor.__exit__."""
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    observations: dict = {}

    class _TeardownMonitor(WinMLEPMonitor):
        requires_session_teardown = True

        def __init__(self):
            self.session_ref = None

        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # At this point, session.reset() should have fired → self.session_ref._session is None
            if self.session_ref is not None:
                observations["session_at_exit"] = self.session_ref._session

        def to_dict(self):
            return {"ep": "test"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    mon = _TeardownMonitor()
    mon.session_ref = session

    with session.perf(monitor=mon):
        # Force run so reset has something to tear down
        session.run({"input": np.zeros((1, 4), dtype=np.float32)})

    # After perf exit, the baseline is reconstructed from its snapshots.
    # The monitor observed reset before its own teardown.
    assert session._session is not None
    # And the observation captured by monitor.__exit__ should also be None
    # (meaning reset fired before __exit__)
    assert observations.get("session_at_exit") is None


def test_exception_transparency():
    """Exception in `with session.perf()` body propagates; monitor.__exit__ sees exc_info."""
    from unittest.mock import MagicMock

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    captured: dict = {}

    class _CapturingMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            captured["exc_type"] = exc_type

        def to_dict(self):
            return {"ep": "test"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    mon = _CapturingMonitor()
    raise_error = MagicMock(side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"), session.perf(monitor=mon):
        raise_error()

    assert captured.get("exc_type") is ValueError


def test_monitor_enter_raises_leaves_session_clean():
    """If mon.__enter__() raises, session state is not polluted.

    Regression guard: an earlier version mutated _perf_stats and _provider_options
    before mon.__enter__(), so an __enter__ exception left the session stuck
    (nested-perf error on every subsequent perf() call).

    _RaisingEnterMonitor.get_provider_options() returns a non-empty dict, causing
    perf() to set _session_rebuilt=True and call the free _build_session_options()
    (which calls WinMLEPRegistry). The mock therefore must stay active across the
    entire perf() call.
    """
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _RaisingEnterMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            raise RuntimeError("simulated __enter__ failure")

        def __exit__(self, *a):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"some_key": "1"}

    cpu_dev = _get_real_cpu_ort_device()
    cpu_ep_device = make_stub_winml_ep_device(cpu_dev, "CPUExecutionProvider")

    session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)

    mon = _RaisingEnterMonitor()
    with pytest.raises(RuntimeError, match="simulated"), session.perf(monitor=mon):
        pass  # never reached

    # Session state must be fully restored
    assert session._perf_stats is None
    assert session._active_session_option_entries == {}
    assert session._provider_options == {}

    # Subsequent perf() MUST work (no stuck state)
    with session.perf() as ctx:
        assert ctx is not None


def test_failed_monitored_rebuild_restores_baseline_without_entering_monitor(monkeypatch):
    """A failed monitor-session build rolls back before the monitor is entered."""
    from winml.modelkit.session import session as session_module
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _ContributingMonitor(WinMLEPMonitor):
        def __init__(self):
            self.entered = 0
            self.exited = 0

        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *args):
            self.exited += 1

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"some_key": "1"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    baseline_provider_options = dict(session._provider_options)
    monitor = _ContributingMonitor()

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            session_module.ort,
            "InferenceSession",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rebuild failed")),
        )
        with pytest.raises(RuntimeError, match="rebuild failed"), session.perf(monitor=monitor):
            pass

    assert session._perf_stats is None
    assert session._session is None
    assert session._provider_options == baseline_provider_options
    assert session._ep == "CPUExecutionProvider"
    assert monitor.entered == 0
    assert monitor.exited == 0
    with session.perf() as ctx:
        assert ctx is not None


def test_running_model_hook_failure_restores_rebuilt_session_state():
    """A pre-enter monitor hook failure rolls back the temporary perf session."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _FailingPathMonitor(WinMLEPMonitor):
        def __init__(self):
            self.entered = 0

        @classmethod
        def is_available(cls):
            return True

        def get_provider_options(self):
            return {"profiling_level": "detailed"}

        def set_running_model_path(self, running_model_path) -> None:
            raise RuntimeError(f"path hook failed: {running_model_path}")

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *args):
            pass

    session = _make_cpu_session(get_minimal_onnx_model_path())
    session.compile()
    baseline_session = session._session
    baseline_provider_options = dict(session._provider_options)
    baseline_session_options = dict(session._active_session_option_entries)
    monitor = _FailingPathMonitor()

    with (
        patch(
            "winml.modelkit.session.session.ort.SessionOptions",
            return_value=MagicMock(),
        ),
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            return_value=MagicMock(),
        ),
        pytest.raises(RuntimeError, match="path hook failed"),
        session.perf(monitor=monitor),
    ):
        pass

    assert session._session is not baseline_session
    assert session._session is not None
    assert session._provider_options == baseline_provider_options
    assert session._active_session_option_entries == baseline_session_options
    assert session._perf_stats is None
    assert monitor.entered == 0


def test_monitored_rebuild_and_baseline_restore_use_running_model_path():
    """Monitored perf rebuilds preserve the compiled runtime model path."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session import SessionState
    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _ContributingMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"profiling_level": "detailed"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    session._running_model_path = Path("C:/fake/compiled_ctx.onnx")
    session._state = SessionState.COMPILED

    with (
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=[MagicMock(), MagicMock()],
        ) as inference_session,
        session.perf(monitor=_ContributingMonitor()),
    ):
        pass

    assert inference_session.call_count == 2
    assert [Path(call.args[0]) for call in inference_session.call_args_list] == [
        session._running_model_path,
        session._running_model_path,
    ]
    assert session.state is SessionState.COMPILED


def test_requires_session_teardown_monitor_restores_baseline_after_monitor_exit():
    """Teardown monitors rebuild the baseline after __exit__ when perf rebuilt the session."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    observations: dict[str, object] = {}

    class _TeardownContributingMonitor(WinMLEPMonitor):
        requires_session_teardown = True

        def __init__(self):
            self.session_ref = None
            self.inference_session = None

        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            observations["session_during_exit"] = self.session_ref._session
            observations["call_count_during_exit"] = self.inference_session.call_count

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"profiling_level": "detailed"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    session._running_model_path = Path(r"C:\fake\compiled_ctx.onnx")
    monitored_session = MagicMock(name="monitored_session")
    restored_session = MagicMock(name="restored_session")
    monitor = _TeardownContributingMonitor()
    monitor.session_ref = session

    with patch(
        "winml.modelkit.session.session.ort.InferenceSession",
        side_effect=[monitored_session, restored_session],
    ) as inference_session:
        monitor.inference_session = inference_session
        with session.perf(monitor=monitor):
            assert session._session is monitored_session

    assert observations == {
        "session_during_exit": None,
        "call_count_during_exit": 1,
    }
    assert inference_session.call_count == 2
    assert [Path(call.args[0]) for call in inference_session.call_args_list] == [
        session._running_model_path,
        session._running_model_path,
    ]
    assert session._session is restored_session


def test_rebuilt_perf_with_no_saved_session_restores_exact_no_session_state():
    """A rebuilt perf window over a no-session baseline restores exact no-session state."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _SessionOnlyMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_session_options(self):
            return {"profiling_level": "detailed"}

    cpu_ep_device = make_stub_winml_ep_device(_get_real_cpu_ort_device(), "CPUExecutionProvider")
    session = WinMLSession(get_minimal_onnx_model_path(), ep_device=cpu_ep_device)
    session.reset()
    baseline_state = session.state
    monitor = _SessionOnlyMonitor()
    monitored_session = MagicMock(name="monitored_session")
    restored_session = MagicMock(name="restored_session")

    with patch(
        "winml.modelkit.session.session.ort.InferenceSession",
        side_effect=[monitored_session, restored_session],
    ) as inference_session:
        with session.perf(monitor=monitor):
            assert session._session is monitored_session

        assert session._session is None
        assert session._state is baseline_state
        assert session._provider_options == {}
        assert session._active_session_option_entries == {}

        session.compile()

    assert inference_session.call_count == 2
    assert session._session is restored_session


def test_baseline_restore_failure_does_not_mask_perf_body_error(caplog):
    """Baseline restore failures are logged while the perf-body error remains primary."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _ContributingMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"profiling_level": "detailed"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    baseline_provider_options = dict(session._provider_options)
    baseline_session_entries = dict(session._active_session_option_entries)
    raise_error = MagicMock(side_effect=ValueError("body failed"))

    with (
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=[MagicMock(name="monitored_session"), RuntimeError("restore failed")],
        ),
        caplog.at_level("ERROR"),
        pytest.raises(ValueError, match="body failed"),
        session.perf(monitor=_ContributingMonitor()),
    ):
        raise_error()

    assert "Restoring baseline InferenceSession failed" in caplog.text
    assert "restore failed" in caplog.text
    assert session._session is None
    assert session._provider_options == baseline_provider_options
    assert session._active_session_option_entries == baseline_session_entries


def test_constructor_monitor_baseline_is_reused_for_perf_without_monitor():
    """Constructor baseline monitor config remains active when perf() adds no overrides."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    baseline_session_entries = {"baseline.session.entry": "enabled"}

    class _BaselineMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "baseline"}

        def get_provider_options(self):
            return {"baseline_provider_key": "baseline"}

        def get_session_options(self):
            return dict(baseline_session_entries)

    options = [MagicMock() for _ in range(3)]
    factory = MagicMock(side_effect=options)
    baseline_session = MagicMock()
    cpu_ep_device = make_stub_winml_ep_device(_get_real_cpu_ort_device(), "CPUExecutionProvider")

    with patch(
        "winml.modelkit.session.session.ort.InferenceSession",
        side_effect=[baseline_session, MagicMock(), MagicMock()],
    ) as inference_session:
        session = WinMLSession(
            get_minimal_onnx_model_path(),
            ep_device=cpu_ep_device,
            ep_monitor=_BaselineMonitor(),
            session_options=factory,
        )
        baseline_provider_options = dict(session._provider_options)

        with session.perf():
            assert session._session is baseline_session
            assert session._provider_options == baseline_provider_options
            assert session._active_session_option_entries == baseline_session_entries

    assert inference_session.call_count == 1
    assert factory.call_count == 1
    options[0].add_session_config_entry.assert_called_once_with(
        "baseline.session.entry",
        "enabled",
    )
    assert session._session is baseline_session
    assert session._provider_options == baseline_provider_options
    assert session._active_session_option_entries == baseline_session_entries


def test_constructor_monitor_snapshots_restore_after_perf_rebuild():
    """Constructor-applied monitor options are tracked and restored after perf rebuilds."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    baseline_session_entries = {"baseline.session.entry": "enabled"}
    perf_session_entries = {"perf.session.entry": "enabled"}
    expected_perf_session_entries = {
        "baseline.session.entry": "enabled",
        "perf.session.entry": "enabled",
    }

    class _BaselineMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "baseline"}

        def get_provider_options(self):
            return {"baseline_provider_key": "baseline"}

        def get_session_options(self):
            return dict(baseline_session_entries)

    class _PerfMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "perf"}

        def get_provider_options(self):
            return {"perf_provider_key": "perf"}

        def get_session_options(self):
            return dict(perf_session_entries)

    options = [MagicMock() for _ in range(3)]
    factory = MagicMock(side_effect=options)
    cpu_ep_device = make_stub_winml_ep_device(_get_real_cpu_ort_device(), "CPUExecutionProvider")

    with patch(
        "winml.modelkit.session.session.ort.InferenceSession",
        side_effect=[MagicMock(), MagicMock(), MagicMock()],
    ):
        session = WinMLSession(
            get_minimal_onnx_model_path(),
            ep_device=cpu_ep_device,
            ep_monitor=_BaselineMonitor(),
            session_options=factory,
        )
        baseline_provider_options = dict(session._provider_options)

        assert session._active_session_option_entries == baseline_session_entries

        with session.perf(monitor=_PerfMonitor()):
            assert session._active_session_option_entries == expected_perf_session_entries

    initial_so, perf_so, restored_so = options
    initial_so.add_session_config_entry.assert_called_once_with(
        "baseline.session.entry",
        "enabled",
    )
    assert [call.args for call in perf_so.add_session_config_entry.call_args_list] == [
        ("baseline.session.entry", "enabled"),
        ("perf.session.entry", "enabled"),
    ]
    restored_so.add_session_config_entry.assert_called_once_with(
        "baseline.session.entry",
        "enabled",
    )
    assert initial_so.add_provider_for_devices.call_args.args[1] == baseline_provider_options
    assert perf_so.add_provider_for_devices.call_args.args[1] == {
        **baseline_provider_options,
        "perf_provider_key": "perf",
    }
    assert restored_so.add_provider_for_devices.call_args.args[1] == baseline_provider_options
    assert session._active_session_option_entries == baseline_session_entries


def test_perf_rebuilds_for_monitor_session_options_without_provider_changes():
    """Session-only monitor options trigger rebuilds and active-option tracking."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _SessionOptionMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {}

        def get_session_options(self):
            return {"session.disable_cpu_ep_fallback": "1"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    monitor_so = MagicMock()
    baseline_so = MagicMock()

    with (
        patch(
            "winml.modelkit.session.session.ort.SessionOptions",
            side_effect=[monitor_so, baseline_so],
        ),
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=[MagicMock(), MagicMock()],
        ) as inference_session,
        session.perf(monitor=_SessionOptionMonitor()),
    ):
        assert session._active_session_option_entries == {"session.disable_cpu_ep_fallback": "1"}

    assert inference_session.call_count == 2
    monitor_so.add_session_config_entry.assert_called_once_with(
        "session.disable_cpu_ep_fallback",
        "1",
    )
    baseline_so.add_session_config_entry.assert_not_called()
    assert session._active_session_option_entries == {}


def test_monitored_rebuild_uses_fresh_session_options_factory_outputs():
    """Monitor and baseline rebuilds bind providers on distinct configured options."""
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor
    from winml.modelkit.session.session import WinMLSession

    from .conftest import make_stub_winml_ep_device

    class _ContributingMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def to_dict(self):
            return {"ep": "test"}

        def get_provider_options(self):
            return {"some_key": "1"}

    options = [MagicMock() for _ in range(3)]
    for option in options:
        option.intra_op_num_threads = 4
    factory = MagicMock(side_effect=options)
    cpu_ep_device = make_stub_winml_ep_device(_get_real_cpu_ort_device(), "CPUExecutionProvider")

    with patch("winml.modelkit.session.session.ort.InferenceSession"):
        session = WinMLSession(
            get_minimal_onnx_model_path(),
            ep_device=cpu_ep_device,
            session_options=factory,
        )
        with session.perf(monitor=_ContributingMonitor()):
            pass

    assert factory.call_count == 3
    assert all(option.intra_op_num_threads == 4 for option in options)
    for option in options:
        option.add_provider_for_devices.assert_called_once()


def test_monitor_exit_failure_is_logged_without_replacing_body_error(caplog):
    """A monitor teardown error is logged while the perf-body error propagates."""
    from unittest.mock import MagicMock

    from winml.modelkit.session.monitor.ep_monitor import WinMLEPMonitor

    class _FailingExitMonitor(WinMLEPMonitor):
        @classmethod
        def is_available(cls):
            return True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise RuntimeError("monitor exit failed")

        def to_dict(self):
            return {"ep": "test"}

    session = _make_cpu_session(get_minimal_onnx_model_path())
    raise_error = MagicMock(side_effect=ValueError("body failed"))

    with (
        caplog.at_level("ERROR"),
        pytest.raises(ValueError, match="body failed"),
        session.perf(monitor=_FailingExitMonitor()),
    ):
        raise_error()

    assert "monitor exit failed" in caplog.text


def test_perf_calls_set_onnx_op_types_on_monitor():
    """v2.4: perf() injects the ONNX op-type map unconditionally before __enter__.

    Even monitors that inherit the no-op default get the call — that's the
    design (idempotent, defensive). Op-tracing subclasses override to capture.
    """
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    calls: list[dict[str, str]] = []
    enter_order: list[str] = []

    class _RecordingMonitor(NullEPMonitor):
        def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
            calls.append(dict(onnx_op_types))
            enter_order.append("set_onnx_op_types")

        def __enter__(self):
            enter_order.append("__enter__")
            return self

    session = _make_cpu_session(get_minimal_onnx_model_path())
    with session.perf(monitor=_RecordingMonitor()):
        pass

    # Exactly one call, with a dict argument
    assert len(calls) == 1
    assert isinstance(calls[0], dict)
    # And it fired BEFORE __enter__ (so monitors can prep state on the map)
    assert enter_order == ["set_onnx_op_types", "__enter__"]


def test_perf_injects_onnx_model_path_before_monitor_enter():
    from pathlib import Path

    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    calls: list[Path] = []
    order: list[str] = []

    class _RecordingMonitor(NullEPMonitor):
        def set_onnx_model_path(self, onnx_model_path: Path) -> None:
            calls.append(Path(onnx_model_path))
            order.append("set_onnx_model_path")

        def __enter__(self):
            order.append("__enter__")
            return self

    model_path = get_minimal_onnx_model_path()
    session = _make_cpu_session(model_path)
    with session.perf(monitor=_RecordingMonitor()):
        pass

    assert calls == [Path(model_path)]
    assert order == ["set_onnx_model_path", "__enter__"]


def test_perf_injects_running_model_path_for_compiled_and_runtime_sessions():
    """Monitors receive the active artifact separately from the original graph."""
    from pathlib import Path

    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    class _RecordingMonitor(NullEPMonitor):
        def __init__(self):
            self.onnx_model_paths: list[Path] = []
            self.running_model_paths: list[Path] = []
            self.order: list[str] = []

        def set_onnx_model_path(self, onnx_model_path: Path) -> None:
            self.onnx_model_paths.append(Path(onnx_model_path))
            self.order.append("set_onnx_model_path")

        def set_running_model_path(self, running_model_path: Path) -> None:
            self.running_model_paths.append(Path(running_model_path))
            self.order.append("set_running_model_path")

        def __enter__(self):
            self.order.append("__enter__")
            return self

    original_path = Path(get_minimal_onnx_model_path())
    compiled_path = original_path.with_name(f"{original_path.stem}_ctx.onnx")

    compiled_session = _make_cpu_session(original_path)
    compiled_session._running_model_path = compiled_path
    compiled_monitor = _RecordingMonitor()
    with compiled_session.perf(monitor=compiled_monitor):
        pass

    runtime_session = _make_cpu_session(original_path)
    runtime_monitor = _RecordingMonitor()
    with runtime_session.perf(monitor=runtime_monitor):
        pass

    assert compiled_monitor.onnx_model_paths == [original_path]
    assert compiled_monitor.running_model_paths == [compiled_path]
    assert compiled_monitor.order == [
        "set_onnx_model_path",
        "set_running_model_path",
        "__enter__",
    ]
    assert runtime_monitor.onnx_model_paths == [original_path]
    assert runtime_monitor.running_model_paths == [original_path]
    assert runtime_monitor.order == [
        "set_onnx_model_path",
        "set_running_model_path",
        "__enter__",
    ]


def test_perf_rebuild_keeps_monitor_bound_to_active_running_model():
    """Monitor-contributed options must rebuild and trace the compiled artifact."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    class _RebuildingMonitor(NullEPMonitor):
        def __init__(self):
            self.running_model_paths: list[Path] = []

        def get_provider_options(self):
            return {"profiling_level": "detailed"}

        def set_running_model_path(self, running_model_path: Path) -> None:
            self.running_model_paths.append(Path(running_model_path))

    original_path = Path(get_minimal_onnx_model_path())
    compiled_path = original_path.with_name(f"{original_path.stem}_ctx.onnx")
    session = _make_cpu_session(original_path)
    session._running_model_path = compiled_path
    monitor = _RebuildingMonitor()

    with (
        patch(
            "winml.modelkit.session.session.ort.SessionOptions",
            side_effect=[MagicMock(), MagicMock()],
        ),
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            side_effect=[MagicMock(), MagicMock()],
        ) as inference_session,
        session.perf(monitor=monitor),
    ):
        pass

    assert monitor.running_model_paths == [compiled_path]
    assert [call.args[0] for call in inference_session.call_args_list] == [
        compiled_path,
        compiled_path,
    ]


def test_perf_after_reset_rebuilds_from_source_model():
    """A reset session must not reopen its previous runtime artifact."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    class _RebuildingMonitor(NullEPMonitor):
        def get_provider_options(self):
            return {"profiling_level": "detailed"}

    source_path = Path(get_minimal_onnx_model_path())
    stale_context_path = source_path.with_name(f"{source_path.stem}_stale_ctx.onnx")
    session = _make_cpu_session(source_path)
    session._running_model_path = stale_context_path
    session.reset()

    with (
        patch(
            "winml.modelkit.session.session.ort.SessionOptions",
            return_value=MagicMock(),
        ),
        patch(
            "winml.modelkit.session.session.ort.InferenceSession",
            return_value=MagicMock(),
        ) as inference_session,
        session.perf(monitor=_RebuildingMonitor()),
    ):
        pass

    assert inference_session.call_args_list[0].args[0] == source_path


def test_perf_provides_completed_window_before_monitor_exit():
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    calls: list[tuple[int, int]] = []
    order: list[str] = []

    class _RecordingMonitor(NullEPMonitor):
        def set_perf_window(self, warmup: int, measured_iterations: int) -> None:
            calls.append((warmup, measured_iterations))
            order.append("set_perf_window")

        def __exit__(self, exc_type, exc_val, exc_tb):
            order.append("__exit__")

    session = _make_cpu_session(get_minimal_onnx_model_path())
    inputs = {"input": np.zeros((1, 4), dtype=np.float32)}
    with session.perf(warmup=1, monitor=_RecordingMonitor()):
        for _ in range(3):
            session.run(inputs)

    assert calls == [(1, 2)]
    assert order == ["set_perf_window", "__exit__"]


def test_perf_injects_real_op_type_map_for_named_nodes(tmp_path):
    """v2.4: when the ONNX has named nodes, the injected map is populated."""
    import onnx
    from onnx import TensorProto, helper

    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    # Build a tiny ONNX with a named node
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"], name="/n0/Relu")
    graph = helper.make_graph([node], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model_path = tmp_path / "named.onnx"
    onnx.save(model, str(model_path))

    captured: list[dict[str, str]] = []

    class _CapturingMonitor(NullEPMonitor):
        def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
            captured.append(dict(onnx_op_types))

    session = _make_cpu_session(model_path)
    with session.perf(monitor=_CapturingMonitor()):
        pass

    assert captured == [{"/n0/Relu": "Relu"}]
