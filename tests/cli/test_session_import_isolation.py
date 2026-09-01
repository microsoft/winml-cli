# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Import-boundary regressions for session backends and monitors."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- fixed interpreter and repository code
        [sys.executable, "-X", "utf8", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_hw_monitor_imports_do_not_require_onnx_packages() -> None:
    for import_statement in (
        "from winml.modelkit.session.monitor.hw_monitor import HWMonitor",
        "from winml.modelkit.session import HWMonitor",
    ):
        result = _run_in_subprocess(
            f"""
        import builtins
        import sys

        real_import = builtins.__import__

        def block_onnx(name, *args, **kwargs):
            if name.split('.')[0] in {"onnx", "onnxruntime"}:
                raise ModuleNotFoundError(
                    f'blocked optional dependency: {{name}}', name=name
                )
            return real_import(name, *args, **kwargs)

        builtins.__import__ = block_onnx
        {import_statement}

        assert HWMonitor.__name__ == 'HWMonitor'
        assert not any(name.startswith(('onnx', 'onnxruntime')) for name in sys.modules)
        """
        )

        assert result.returncode == 0, result.stderr


def test_session_facade_imports_no_backends_until_public_symbol_access() -> None:
    result = _run_in_subprocess(
        """
        import sys
        import winml.modelkit.session as session

        eager = {
            'winml.modelkit.session.ep_device',
            'winml.modelkit.session.ep_registry',
            'winml.modelkit.session.genai_session',
            'winml.modelkit.session.monitor.ep_monitor',
            'winml.modelkit.session.monitor.hw_monitor',
            'winml.modelkit.session.monitor.openvino_monitor',
            'winml.modelkit.session.monitor.qnn_monitor',
            'winml.modelkit.session.monitor.vitisai_monitor',
            'winml.modelkit.session.qairt.qairt_session',
            'winml.modelkit.session.session',
            'winml.modelkit.session.stats',
        }
        loaded_modules = set(sys.modules)
        assert eager.isdisjoint(loaded_modules), sorted(eager & loaded_modules)
        assert set(session.__all__) == set(session._LAZY_IMPORTS)
        assert 'HWMonitor' in dir(session)
        loaded_modules = set(sys.modules)
        assert eager.isdisjoint(loaded_modules), sorted(eager & loaded_modules)

        assert session.HWMonitor.__name__ == 'HWMonitor'
        assert 'winml.modelkit.session.monitor.hw_monitor' in sys.modules
        assert 'winml.modelkit.session.session' not in sys.modules
        assert 'winml.modelkit.session.genai_session' not in sys.modules
        assert 'winml.modelkit.session.qairt.qairt_session' not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_monitor_facade_imports_no_concrete_monitors_until_access() -> None:
    result = _run_in_subprocess(
        """
        import sys
        import winml.modelkit.session.monitor as monitor

        eager = {
            'winml.modelkit.session.monitor.ep_monitor',
            'winml.modelkit.session.monitor.op_metrics',
            'winml.modelkit.session.monitor.openvino_monitor',
            'winml.modelkit.session.monitor.report',
        }
        loaded_modules = set(sys.modules)
        assert eager.isdisjoint(loaded_modules), sorted(eager & loaded_modules)
        assert set(monitor.__all__) == set(monitor._LAZY_IMPORTS)
        assert 'OperatorMetrics' in dir(monitor)
        loaded_modules = set(sys.modules)
        assert eager.isdisjoint(loaded_modules), sorted(eager & loaded_modules)

        assert monitor.OperatorMetrics.__name__ == 'OperatorMetrics'
        assert 'winml.modelkit.session.monitor.op_metrics' in sys.modules
        assert 'winml.modelkit.session.monitor.openvino_monitor' not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("symbol", "expected_module"),
    [
        ("WinMLSession", "winml.modelkit.session.session"),
        ("GenaiSession", "winml.modelkit.session.genai_session"),
        ("QNNMonitor", "winml.modelkit.session.monitor.qnn_monitor"),
        ("OpenVinoMonitor", "winml.modelkit.session.monitor.openvino_monitor"),
        ("VitisAIMonitor", "winml.modelkit.session.monitor.vitisai_monitor"),
        ("WinMLQairtSession", "winml.modelkit.session.qairt.qairt_session"),
    ],
)
def test_public_backend_symbol_loads_only_its_selected_module(
    symbol: str,
    expected_module: str,
) -> None:
    result = _run_in_subprocess(
        f"""
        import sys
        import winml.modelkit.session as session

        provider_modules = {{
            'winml.modelkit.session.genai_session',
            'winml.modelkit.session.monitor.openvino_monitor',
            'winml.modelkit.session.monitor.qnn_monitor',
            'winml.modelkit.session.monitor.vitisai_monitor',
            'winml.modelkit.session.qairt.qairt_session',
        }}
        selected = getattr(session, {symbol!r})

        assert selected.__name__ == {symbol!r}
        assert {expected_module!r} in sys.modules
        unexpected = provider_modules - {{{expected_module!r}}}
        loaded_modules = set(sys.modules)
        assert unexpected.isdisjoint(loaded_modules), sorted(unexpected & loaded_modules)
        """
    )

    assert result.returncode == 0, result.stderr


def test_ort_compile_stage_does_not_import_unselected_backends() -> None:
    result = _run_in_subprocess(
        """
        import sys
        import winml.modelkit.compiler.stages.compile as compile_stage

        assert compile_stage._session_class_for_compiler('ort').__name__ == 'WinMLSession'
        assert 'winml.modelkit.session.session' in sys.modules
        assert 'winml.modelkit.session.qairt.qairt_session' not in sys.modules
        assert 'winml.modelkit.session.genai_session' not in sys.modules
        assert 'winml.modelkit.session.monitor.qnn_monitor' not in sys.modules
        assert 'winml.modelkit.session.monitor.vitisai_monitor' not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_cpu_and_dml_monitor_selection_do_not_import_provider_backends() -> None:
    result = _run_in_subprocess(
        """
        import sys
        from pathlib import Path
        from winml.modelkit.commands.perf import _resolve_ep_monitor

        for ep in ('cpu', 'dml'):
            monitor = _resolve_ep_monitor(ep, None, Path('.'))
            assert monitor.__class__.__name__ == 'NullEPMonitor'

        unrelated = {
            'winml.modelkit.session.genai_session',
            'winml.modelkit.session.monitor.openvino_monitor',
            'winml.modelkit.session.monitor.qnn_monitor',
            'winml.modelkit.session.monitor.vitisai_monitor',
            'winml.modelkit.session.qairt.qairt_session',
        }
        loaded_modules = set(sys.modules)
        assert unrelated.isdisjoint(loaded_modules), sorted(unrelated & loaded_modules)
        """
    )

    assert result.returncode == 0, result.stderr


def test_unsupported_op_tracing_does_not_import_qnn_monitor() -> None:
    for ep, device in (("dml", "gpu"), (None, "cpu"), (None, "gpu")):
        result = _run_in_subprocess(
            f"""
        import sys
        from pathlib import Path
        from winml.modelkit.commands.perf import _resolve_ep_monitor

        try:
            _resolve_ep_monitor({ep!r}, 'basic', Path('.'), device={device!r})
        except RuntimeError as exc:
            assert 'Op-tracing not available' in str(exc)
        else:
            raise AssertionError('unsupported op tracing was accepted')

        assert 'winml.modelkit.session.monitor.qnn_monitor' not in sys.modules
        """
        )

        assert result.returncode == 0, result.stderr


def test_qnn_selection_reports_actionable_missing_backend() -> None:
    result = _run_in_subprocess(
        """
        from pathlib import Path
        from winml.modelkit.commands.perf import _resolve_ep_monitor
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        QNNMonitor.is_available = classmethod(lambda cls: False)
        try:
            _resolve_ep_monitor('qnn', 'basic', Path('.'), device='npu')
        except RuntimeError as exc:
            message = str(exc)
            assert 'QNN is not available' in message
            assert 'Install onnxruntime-qnn' in message
        else:
            raise AssertionError('missing QNN backend was accepted')
        """
    )

    assert result.returncode == 0, result.stderr


def test_genai_selection_reports_actionable_missing_dependency(tmp_path) -> None:
    (tmp_path / "genai_config.json").write_text(
        '{"model": {"context_length": 1}}',
        encoding="utf-8",
    )
    result = _run_in_subprocess(
        f"""
        import sys
        from pathlib import Path

        from winml.modelkit.session import GenaiNotInstalledError, GenaiSession

        sys.modules['onnxruntime_genai'] = None
        try:
            GenaiSession(Path({str(tmp_path)!r})).load()
        except GenaiNotInstalledError as exc:
            assert 'Could not import onnxruntime_genai' in str(exc)
        else:
            raise AssertionError('missing onnxruntime_genai was accepted')
        """
    )

    assert result.returncode == 0, result.stderr


def test_qairt_compile_selection_loads_only_qairt_backend() -> None:
    result = _run_in_subprocess(
        """
        import sys
        import winml.modelkit.compiler.stages.compile as compile_stage

        assert compile_stage._session_class_for_compiler('qairt').__name__ == (
            'WinMLQairtSession'
        )
        assert 'winml.modelkit.session.qairt.qairt_session' in sys.modules
        assert 'winml.modelkit.session.genai_session' not in sys.modules
        assert 'winml.modelkit.session.monitor.qnn_monitor' not in sys.modules
        assert 'winml.modelkit.session.monitor.vitisai_monitor' not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr
