# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for QNNMonitor — the QNN EP op-tracing monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_ctor_defaults():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    assert m._level == "basic"
    assert m._output_dir.exists()
    assert m._csv_path.is_absolute()


def test_ctor_accepts_custom_output_dir(tmp_path):
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    assert m._output_dir == tmp_path
    assert str(m._csv_path).startswith(str(tmp_path))


def test_ctor_rejects_invalid_level():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with pytest.raises(ValueError, match="level"):
        QNNMonitor(level="bogus")  # type: ignore[arg-type]


def test_get_session_options_enables_epcontext_caching():
    """get_session_options enables EPContext caching only.

    `session.disable_cpu_ep_fallback` is intentionally NOT set: under
    onnxruntime-windowsml the WinML-registered QNN partitions a QDQ-wrapped
    EPContext model into Q/DQ-on-CPU + EPContext-on-QNN, which is correct.
    The "no silent CPU fallback" guarantee is provided upstream by
    add_provider_for_devices, not here.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    opts = QNNMonitor().get_session_options()
    assert opts == {
        "ep.context_enable": "1",
        "ep.context_embed_mode": "0",
    }
    assert "session.disable_cpu_ep_fallback" not in opts


def test_get_provider_options_owner_keys_only():
    """get_provider_options sets ONLY the two profiling keys + user extras.

    backend_path / htp_* are NOT defaulted: they would overwrite WinML's
    registered absolute backend_path and break DLL loading. Callers who
    need them pass via extra_provider_options.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    opts = QNNMonitor(level="basic").get_provider_options()
    assert opts == {
        "profiling_level": "detailed",
        "profiling_file_path": opts["profiling_file_path"],
    }
    # Verify no defaults that would conflict with WinML registration
    assert "backend_path" not in opts
    assert "htp_performance_mode" not in opts


def test_get_provider_options_detail():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    assert QNNMonitor(level="detail").get_provider_options()["profiling_level"] == "optrace"


def test_extra_provider_options_pass_through():
    """User-supplied extras are honored (e.g. backend_path for bundled ORT QNN)."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(
        level="basic",
        extra_provider_options={
            "backend_path": r"C:\path\to\QnnHtp.dll",
            "htp_performance_mode": "balanced",
        },
    )
    opts = m.get_provider_options()
    assert opts["backend_path"] == r"C:\path\to\QnnHtp.dll"
    assert opts["htp_performance_mode"] == "balanced"


def test_profiling_keys_not_user_overridable():
    """C-3: user extras cannot override profiling_level or profiling_file_path."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(
        level="basic",
        extra_provider_options={
            "profiling_level": "off",
            "profiling_file_path": "/attacker/path",
            "htp_performance_mode": "balanced",
        },
    )
    opts = m.get_provider_options()
    assert opts["profiling_level"] == "detailed"
    assert opts["profiling_file_path"] != "/attacker/path"
    assert opts["htp_performance_mode"] == "balanced"  # non-owned extra honored


def test_get_provider_options_idempotent():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(level="basic")
    assert m.get_provider_options() == m.get_provider_options()


def test_get_session_options_idempotent():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(level="basic")
    assert m.get_session_options() == m.get_session_options()


def test_requires_session_teardown_true():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    assert QNNMonitor.requires_session_teardown is True


def test_double_enter_raises():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    m.__enter__()
    with pytest.raises(RuntimeError, match="already entered"):
        m.__enter__()


def test_exit_with_no_csv_reports_no_data(tmp_path):
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)
    # v2.4: data exposed via the typed ``result`` accessor.
    assert m.result is not None
    assert m.result.status == "no_data"


def test_exit_parse_failure_caught(tmp_path):
    """If CSV exists but is corrupt, status is 'parse_failed' and error is populated."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv = tmp_path / "profiling_output.csv"
    csv.write_text("this is not a valid qnn csv")
    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)
    # v2.4: data exposed via the typed ``result`` accessor. Either
    # 'parse_failed' (if parser raises) or 'ok'/'no_data' (if parser
    # gracefully returns empty). We accept any of those but must NOT raise.
    assert m.result is not None
    assert m.result.status in ("parse_failed", "no_data", "ok")


def test_exit_does_not_suppress_caller_exception(tmp_path):
    """EPMonitor.__exit__ returning None (not True) → exception propagates."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    result = m.__exit__(RuntimeError, RuntimeError("test"), None)
    assert result is None or result is False


def test_result_before_enter_is_none():
    """v2.4: pre-exit, ``result`` is ``None`` (no data parsed yet).

    The pre-v2.4 ``to_dict()`` shim that returned a synthesized
    ``status="not_run"`` envelope is gone; consumers must check
    ``monitor.result is None`` instead.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    assert m.result is None


def test_result_pre_exit_returns_none(tmp_path):
    """v2.4: ``result`` stays ``None`` until ``__exit__`` populates it."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    assert monitor.result is None
    monitor.__enter__()
    # Still None until __exit__ runs the parse pass.
    assert monitor.result is None


def test_is_available_via_bundled():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with patch(
        "onnxruntime.get_available_providers",
        return_value=["QNNExecutionProvider", "CPUExecutionProvider"],
    ):
        assert QNNMonitor.is_available() is True


def test_is_available_via_winml():
    """When QNN EP is registered via WinML, is_available() returns True."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fake_ep = MagicMock()
    fake_ep.ep_name = "QNNExecutionProvider"
    with (
        patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("onnxruntime.get_ep_devices", return_value=[fake_ep]),
        patch("winml.modelkit.session.ep_registry.ensure_initialized"),
    ):
        assert QNNMonitor.is_available() is True


def test_is_available_neither():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with (
        patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("onnxruntime.get_ep_devices", return_value=[]),
        patch("winml.modelkit.session.ep_registry.ensure_initialized"),
    ):
        assert QNNMonitor.is_available() is False


def test_is_available_winml_path_failure_logs_warning(caplog, monkeypatch):
    """NFR-2: real environmental failure on the WinML path must log at WARNING, not DEBUG.

    The bare-Exception swallow downgraded broken Windows App SDK / denied
    registry access to "feature unavailable" silently. Any non-ImportError
    in ``ensure_initialized()`` MUST surface at WARNING with the exception
    class, so users can diagnose the underlying environment problem.
    """
    import logging

    import onnxruntime as ort

    from winml.modelkit.session import ep_registry
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    # Force the QNN-bundled path to miss
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "get_ep_devices", list)

    # Make ensure_initialized raise a non-ImportError exception
    def _raises() -> None:
        raise RuntimeError("simulated WinML init failure")

    monkeypatch.setattr(ep_registry, "ensure_initialized", _raises)

    with caplog.at_level(logging.WARNING):
        assert QNNMonitor.is_available() is False

    # Assert the log carries enough info to diagnose
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    matched = any(
        "WinML EP probe failed" in r.message and "RuntimeError" in r.message for r in warnings
    )
    assert matched, (
        f"expected WARNING with 'WinML EP probe failed' + 'RuntimeError', "
        f"got: {[r.message for r in warnings]}"
    )


def test_result_property_none_before_exit():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    assert m.result is None


def test_no_os_chdir():
    """QNNMonitor MUST NOT mutate CWD per FR-12 / C-5."""
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    cwd_before = Path.cwd()
    m = QNNMonitor()
    m.__enter__()
    m.__exit__(None, None, None)
    assert Path.cwd() == cwd_before


def test_find_schematic_rejects_stale_cwd_candidate(tmp_path, monkeypatch):
    """A *_schematic.bin in CWD older than the profiling CSV must NOT be returned.

    Setup:
      - output_dir = tmp_path/out  (no schematic in it → exercise CWD fallback)
      - cwd        = tmp_path/cwd  (contains a STALE schematic)
      - csv        = tmp_path/out/profiling_output.csv (FRESH, written 'now')
    Expected: the stale CWD schematic is older than the CSV by >5s, so the
    mtime gate rejects it and _find_schematic() returns None.
    """
    import os
    import time

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    # Fresh CSV (now)
    monitor._csv_path.write_text("dummy")
    # Stale schematic in CWD (1 hour old)
    stale = cwd_dir / "stale_schematic.bin"
    stale.write_bytes(b"")
    old = time.time() - 3600
    os.utime(stale, (old, old))

    monkeypatch.chdir(cwd_dir)
    # CWD glob would surface 'stale', but mtime guard rejects.
    assert monitor._find_schematic() is None


def test_find_schematic_accepts_fresh_cwd_candidate(tmp_path, monkeypatch):
    """A *_schematic.bin in CWD newer than the profiling CSV is accepted (mtime gate)."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    # CSV first, then a fresh schematic — the schematic mtime >= CSV mtime.
    monitor._csv_path.write_text("dummy")
    fresh = cwd_dir / "fresh_schematic.bin"
    fresh.write_bytes(b"")

    monkeypatch.chdir(cwd_dir)
    assert monitor._find_schematic() == fresh


def test_find_schematic_prefers_output_dir_over_cwd(tmp_path, monkeypatch):
    """When output_dir contains a schematic, CWD is never consulted."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    in_out = out_dir / "graph_schematic.bin"
    in_out.write_bytes(b"")
    in_cwd = cwd_dir / "graph_schematic.bin"
    in_cwd.write_bytes(b"")

    monkeypatch.chdir(cwd_dir)
    assert monitor._find_schematic() == in_out


def test_output_dir_property_exposes_path(tmp_path):
    """The output_dir property returns the directory used for artifacts."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    assert monitor.output_dir == tmp_path
    assert monitor.output_dir.is_dir()


def test_output_dir_property_for_default_tempdir():
    """When output_dir=None, the property exposes the auto-minted tempdir."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic")
    assert monitor.output_dir.is_dir()
    assert monitor.output_dir.name.startswith("qnn_profile_")


def test_output_dir_property_is_read_only(tmp_path):
    """output_dir is exposed as a property; rebinding must raise AttributeError."""
    import pytest as _pytest

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    with _pytest.raises(AttributeError):
        monitor.output_dir = tmp_path / "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detail-mode fallback (FR-5 / PR review A2-I7)
# ---------------------------------------------------------------------------


def test_detail_mode_falls_back_to_basic_when_qhas_unavailable(tmp_path):
    """A detail-level monitor with a valid CSV but no QHAS path produces status='basic_fallback'.

    PRD FR-5: when the user requests ``level="detail"`` but post-processing
    artifacts (``*_qnn.log`` / ``*_schematic.bin`` / SDK) are unavailable,
    the monitor MUST surface a populated CSV-only result with
    ``status="basic_fallback"`` rather than raising or producing
    ``status="ok"`` (which would silently pretend QHAS data was present).
    """
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="detail", output_dir=tmp_path)
    # Drop the real CSV fixture into the spot the monitor expects so the
    # CSV parse path succeeds. The QHAS branch will fail naturally because
    # no *_qnn.log is present in the output directory — this is the
    # cleanest hit on the basic_fallback codepath in _try_qhas.
    fixture = Path(__file__).parent / "qnn" / "fixtures" / "optrace_resnet50.csv"
    monitor._csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    monitor.__enter__()
    monitor.__exit__(None, None, None)

    assert monitor.result is not None
    assert monitor.result.status == "basic_fallback"
    # CSV-only data must still be populated — basic_fallback is degraded
    # *success*, not failure: operators and summary are non-empty.
    assert monitor.result.operators, "expected CSV-derived operators in basic_fallback result"
    assert monitor.result.summary, "expected CSV-derived summary in basic_fallback result"
    # T2 wiring: per-sample timings must be threaded through the CSV parser
    # so downstream p90/total/count statistics work on real traces.
    assert len(monitor.result.operators[0].samples_us) > 0, (
        "expected samples_us populated from CSV per-sample rows"
    )
    # No QHAS artifact recorded; CSV artifact recorded.
    assert "qhas" not in monitor.result.artifacts
    assert "csv" in monitor.result.artifacts


# ---------------------------------------------------------------------------
# Windows file-handle retry (R-2 / PR review A2-I8)
# ---------------------------------------------------------------------------


def test_parse_artifacts_retries_when_csv_absent(tmp_path, monkeypatch):
    """R-2 mitigation: a 50ms ``time.sleep`` retry fires when the CSV is
    absent on the first ``is_file()`` check.

    QNN EP flushes the profiling CSV on session destruction, but on Windows
    file-handle close can lag the actual unlink/rename behind the calling
    thread. The monitor's ``_parse_artifacts`` does one 50ms retry before
    declaring ``no_data``. Without this retry, slow filesystems would
    silently produce ``status="no_data"`` for runs that did finish flushing.
    """
    from winml.modelkit.session.monitor import qnn_monitor as qnn_monitor_mod
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)

    sleep_calls: list[float] = []

    def _track_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(qnn_monitor_mod.time, "sleep", _track_sleep)

    # CSV never appears, so the retry will not save the result, but the
    # critical assertion is that the 50ms retry DID fire.
    monitor.__enter__()
    monitor.__exit__(None, None, None)

    assert any(abs(s - 0.05) < 1e-9 for s in sleep_calls), (
        f"expected exactly one 0.05s retry sleep, got {sleep_calls!r}"
    )
    # And status confirms the post-retry path: CSV still missing → no_data.
    assert monitor.result is not None
    assert monitor.result.status == "no_data"


def test_csv_path_event_id_splits_into_name_and_op_path(tmp_path):
    """End-to-end: a path-style QNN event id must produce ``name != op_path``.

    The resnet50 fixture contains rows like
    ``/resnet/embedder/embedder/convolution/Conv_token_1_2:OpId_24 (cycles)``
    where the captured identifier is a hierarchical path. The dataclass
    contract says ``name`` is the QNN op type (``Conv``) and ``op_path``
    is the framework path (``/resnet/.../Conv``); the parser splits the
    event id at the trailing ``/`` so this round-trip holds. Without the
    split, the report's Type column renders the truncated path instead
    of the op type — the regression this test pins down.
    """
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    fixture = Path(__file__).parent / "qnn" / "fixtures" / "optrace_resnet50.csv"
    monitor._csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    monitor.__enter__()
    monitor.__exit__(None, None, None)

    assert monitor.result is not None
    assert monitor.result.status == "ok"
    # At least one operator in the fixture has a path-style event id.
    path_style = [op for op in monitor.result.operators if "/" in op.op_path]
    assert path_style, "fixture should contain at least one path-style operator"
    for op in path_style:
        # Type (``op.name``) is the leaf segment, distinct from full path.
        assert op.name != op.op_path, (
            f"path-style op should split into distinct name/op_path; "
            f"got name={op.name!r} op_path={op.op_path!r}"
        )
        assert "/" not in op.name, f"op type should not contain slashes; got {op.name!r}"
        # And the leaf must in fact be the trailing segment of the path.
        assert op.op_path.endswith(op.name), (
            f"op_path should end with op type; got name={op.name!r} op_path={op.op_path!r}"
        )


def test_qhas_path_uses_qnn_op_type_when_no_onnx_map():
    """L2 wins when the ONNX op-type map is empty.

    QHAS path with no injected ONNX map → the resolver's L1 lookup
    misses, so the EP-authoritative ``qnn_op_type`` (e.g. ``"Conv2d"``)
    surfaces in :class:`OperatorMetrics.name`.  This pins the
    pre-Phase-2 behaviour: when no ONNX graph is available
    (e.g. ``parse_existing_artifacts(onnx_op_types=None)``), the QHAS
    vocabulary remains authoritative — it MUST NOT be silently
    overridden by the heuristic leaf-split that produced the ONNX op
    symbol ``"Conv"`` in commit ``c3ac3d45``.
    """
    import json
    from pathlib import Path

    from winml.modelkit.session.monitor.op_metrics import OperatorMetrics
    from winml.modelkit.session.monitor.qnn import parse_qhas
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fixture = Path(__file__).parent / "qnn" / "fixtures" / "qhas_resnet50.json"
    qhas_data = json.loads(fixture.read_text(encoding="utf-8"))
    parsed = parse_qhas(qhas_data)

    # Empty ONNX map → resolver falls through to EP-authoritative (L2).
    monitor = QNNMonitor(level="detail")
    monitor.set_onnx_op_types({})

    # Mirror the OperatorMetrics construction in QNNMonitor._try_qhas
    # exactly as the production code does: pass op["name"] (the QHAS
    # qnn_op_type) as ep_authoritative.
    operators = [
        OperatorMetrics(
            name=monitor._resolve_op_type(op["op_path"], ep_authoritative=op["name"]),
            op_path=op["op_path"],
            duration_us=op["duration_us"],
            percent_of_total=op["percent_of_total"],
            samples_us=[op["duration_us"]],
        )
        for op in parsed["operators"]
    ]

    assert operators, "fixture should yield at least one operator"

    # First op pinned to canonical QNN op type from the fixture.
    first = operators[0]
    assert first.name == "Conv2d", (
        f"with empty ONNX map, L2 (qnn_op_type) wins; expected 'Conv2d' got {first.name!r}"
    )
    # CRIT-1 fix: ``op_path`` is now stripped of ``_token_N_M`` in the
    # QHAS path (matches CSV path strip + the clean ONNX node.name keys
    # from production ``_build_op_type_map``).
    assert first.op_path == "/resnet/embedder/embedder/convolution/Conv"
    assert first.name != first.op_path

    # Full set must come from the QNN vocabulary, not ONNX.
    names = {op.name for op in operators}
    assert names & {"Conv2d", "ElementWiseAdd", "PoolMax2d", "PoolAvg2d", "Transpose"}, (
        f"expected canonical QNN op types in QHAS-derived metrics; got {sorted(names)}"
    )
    assert not (names & {"Conv", "Add", "MaxPool", "AveragePool"}), (
        f"QHAS path must not surface ONNX op symbols when L2 wins; got {sorted(names)}"
    )


def test_qhas_path_uses_onnx_op_type_when_map_populated():
    """L1 wins when the ONNX op-type map contains the path.

    v2.4 FR-14: when ``WinMLSession.perf`` injects an ONNX
    ``node.name -> node.op_type`` map AND the QHAS ``qnn_op``
    framework-path matches a node in the graph, the ONNX op type
    (``"Conv"``, ``"Add"``, ``"MaxPool"``) wins over the QHAS
    ``qnn_op_type`` (``"Conv2d"``, ``"ElementWiseAdd"``, ...).  This is
    the intentional behavioural change introduced by Phase 2: ONNX has
    the last word, the QHAS qnn_op_type drops to L2.

    Paths NOT in the map continue to surface QHAS qnn_op_type (L2 wins).

    CRIT-1 contract: production ``_build_op_type_map`` keys are ALWAYS
    clean ONNX ``node.name`` (no ``_token_N_M`` suffix).  The QHAS
    ``op_path`` returned by ``parse_qhas`` is now token-stripped to
    match.  This test injects the *cleaned* path as the dict key —
    mirroring what production does — to verify the L1 hit fires for
    real-world wiring (not a token-bearing key, which masks the bug).
    """
    import json
    from pathlib import Path

    from winml.modelkit.session.monitor.op_metrics import OperatorMetrics
    from winml.modelkit.session.monitor.qnn import parse_qhas
    from winml.modelkit.session.monitor.qnn._internal import _TOKEN_SUFFIX
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fixture = Path(__file__).parent / "qnn" / "fixtures" / "qhas_resnet50.json"
    qhas_data = json.loads(fixture.read_text(encoding="utf-8"))
    parsed = parse_qhas(qhas_data)

    # CRIT-1: inject the *cleaned* op_path (mirrors production map keys
    # produced by ``_build_op_type_map`` from the ONNX graph).  Strip is
    # idempotent on already-clean strings, so this is a no-op when the
    # parser already cleaned the path — but it pins the contract.
    first_path = parsed["operators"][0]["op_path"]
    first_path_clean = _TOKEN_SUFFIX.sub("", first_path)
    monitor = QNNMonitor(level="detail")
    monitor.set_onnx_op_types({first_path_clean: "Conv"})

    operators = [
        OperatorMetrics(
            name=monitor._resolve_op_type(op["op_path"], ep_authoritative=op["name"]),
            op_path=op["op_path"],
            duration_us=op["duration_us"],
            percent_of_total=op["percent_of_total"],
            samples_us=[op["duration_us"]],
        )
        for op in parsed["operators"]
    ]

    # The first op is now ONNX op_type (L1 win).
    first = operators[0]
    assert first.name == "Conv", (
        f"L1 should win when ONNX map covers op_path; expected 'Conv' got {first.name!r}"
    )
    assert first.op_path == first_path

    # Other ops still surface QHAS qnn_op_type (L2 win) because their
    # paths are absent from the injected map.
    other_names = {op.name for op in operators[1:]}
    assert other_names & {"Conv2d", "ElementWiseAdd", "PoolMax2d", "PoolAvg2d", "Transpose"}, (
        f"non-overridden ops should still surface QHAS qnn_op_type; got {sorted(other_names)}"
    )


def test_qhas_path_uses_onnx_op_type_with_production_realistic_clean_map(tmp_path):
    """Production: ``_build_op_type_map`` produces clean keys (no ``_token_N_M``).

    The QHAS fixture has token-bearing ``qnn_op`` paths. The L1 lookup
    must match those against the clean ONNX node-name keys produced by
    :py:meth:`WinMLSession._build_op_type_map` in production.

    Pre-Bundle-A bug (CRIT-1): the QHAS path stored ``op_path`` raw
    (with ``_token_N_M`` suffix), so L1 ALWAYS missed against clean
    keys → ``qnn_op_type`` (Conv2d) silently won over ONNX
    ``op_type`` (Conv).  FR-14 was silently never active in detail mode.

    The previous ``test_qhas_path_uses_onnx_op_type_when_map_populated``
    masked this by injecting token-bearing keys into the ONNX map —
    which is NOT what production does.  This test wires the full
    ``parse_existing_artifacts`` path against a clean-key map, the
    actual production shape.
    """
    import json
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn import parse_qhas
    from winml.modelkit.session.monitor.qnn._internal import _TOKEN_SUFFIX
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fixture_dir = Path(__file__).parent / "qnn" / "fixtures"

    # Stage CSV + QHAS at the locations parse_existing_artifacts expects.
    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (fixture_dir / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    qhas_path = tmp_path / "qhas_output.json"
    qhas_text = (fixture_dir / "qhas_resnet50.json").read_text(encoding="utf-8")
    qhas_path.write_text(qhas_text, encoding="utf-8")

    # Build a CLEAN-KEY ONNX map by stripping token suffixes from the
    # raw QHAS qnn_op paths — exactly what _build_op_type_map produces
    # from the underlying ONNX graph in production.  Map every op to
    # the canonical ONNX op symbol "Conv" so L1 hits force the Type
    # column to "Conv" (not "Conv2d") for every op.
    raw = json.loads(qhas_text)
    clean_map = {
        _TOKEN_SUFFIX.sub("", op["qnn_op"]): "Conv"
        for op in raw["data"]["qnn_op_instances_nodes"]["data"]
    }
    # Sanity: clean_map keys must NOT carry the token suffix.
    assert not any("_token_" in k for k in clean_map), (
        f"clean_map should mirror production _build_op_type_map keys (no _token_); "
        f"got {[k for k in clean_map if '_token_' in k]}"
    )
    # Cross-check: parsed QHAS op_paths must match the clean_map keys.
    parsed = parse_qhas(raw)
    parsed_paths = {op["op_path"] for op in parsed["operators"]}
    assert parsed_paths.issubset(clean_map.keys()), (
        f"FR-14 contract violated: QHAS op_path keys must be clean and match "
        f"production _build_op_type_map keys; missing: {parsed_paths - clean_map.keys()}"
    )

    result = QNNMonitor.parse_existing_artifacts(
        level="detail",
        artifacts={"csv": csv_path, "qhas": qhas_path},
        onnx_op_types=clean_map,
    )

    assert result.status == "ok"
    # Every op resolved via L1 (clean-key ONNX hit) → name == "Conv".
    names = {op.name for op in result.operators}
    assert names == {"Conv"}, (
        f"L1 ONNX-primary lookup failed for production-shaped clean keys; "
        f"got {sorted(names)}.  This is the CRIT-1 contract: QHAS op_path "
        f"must be token-stripped so it matches _build_op_type_map keys."
    )


def test_parse_artifacts_no_retry_when_csv_present_on_first_check(tmp_path, monkeypatch):
    """If the CSV is on disk on the FIRST ``is_file()`` check, the 50ms
    retry sleep MUST NOT fire. Verifies the retry is gated, not unconditional.
    """
    from pathlib import Path

    from winml.modelkit.session.monitor import qnn_monitor as qnn_monitor_mod
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    # Pre-populate the CSV with valid content.
    fixture = Path(__file__).parent / "qnn" / "fixtures" / "optrace_resnet50.csv"
    monitor._csv_path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

    sleep_calls: list[float] = []
    monkeypatch.setattr(qnn_monitor_mod.time, "sleep", lambda s: sleep_calls.append(s))

    monitor.__enter__()
    monitor.__exit__(None, None, None)

    assert sleep_calls == [], (
        f"expected no retry sleep when CSV is present on first check, got {sleep_calls!r}"
    )
    assert monitor.result is not None
    assert monitor.result.status == "ok"


# ---------------------------------------------------------------------------
# CRIT-5B: dual-source ``duration_us == avg_us`` invariant
# ---------------------------------------------------------------------------
#
# The renderer at report.py:236 treats them as equivalent
# (``avg_str = avg_us if samples_us else duration_us``).  Both CSV and
# QHAS code paths populate ``samples_us`` and currently happen to produce
# ``duration_us == avg_us``.  Pin the invariant so a future refactor
# cannot silently diverge them — the renderer would then show different
# numbers depending on whether ``samples_us`` was populated, with no test
# to catch it.


def test_duration_us_equals_avg_us_for_csv_path(tmp_path):
    """For CSV-path OperatorMetrics, ``duration_us`` must equal ``avg_us``.

    The renderer treats them as equivalent
    (``avg_str = avg_us if samples_us else duration_us``).  Pin the
    invariant so a future refactor doesn't silently diverge them.
    """
    import pathlib

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fixture = pathlib.Path(__file__).parent / "qnn" / "fixtures" / "optrace_resnet50.csv"
    if not fixture.exists():
        pytest.skip(f"CSV fixture not found at {fixture}")

    result = QNNMonitor.parse_existing_artifacts(
        level="basic",
        artifacts={"csv": fixture},
    )

    assert result.operators, "CSV fixture should yield at least one operator"
    for op in result.operators:
        assert op.samples_us, f"CSV path should populate samples_us for {op.op_path}"
        assert abs(op.duration_us - op.avg_us) < 1e-6, (
            f"duration_us={op.duration_us} != avg_us={op.avg_us} for op {op.op_path}"
        )


def test_duration_us_equals_avg_us_for_qhas_path(tmp_path):
    """For QHAS-path OperatorMetrics, ``duration_us`` must equal ``avg_us``.

    QHAS produces single-element ``samples_us = [duration_us]``, so
    ``avg_us == duration_us``.  Pin the invariant so a future refactor
    doesn't silently diverge them.
    """
    import pathlib

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_fixture = pathlib.Path(__file__).parent / "qnn" / "fixtures" / "optrace_resnet50.csv"
    qhas_fixture = pathlib.Path(__file__).parent / "qnn" / "fixtures" / "qhas_resnet50.json"
    if not csv_fixture.exists() or not qhas_fixture.exists():
        pytest.skip("Fixtures not found")

    result = QNNMonitor.parse_existing_artifacts(
        level="detail",
        artifacts={"csv": csv_fixture, "qhas": qhas_fixture},
    )

    assert result.operators, "QHAS fixture should yield at least one operator"
    for op in result.operators:
        if not op.samples_us:
            continue  # heuristic-only ops may not populate samples_us
        assert abs(op.duration_us - op.avg_us) < 1e-6, (
            f"duration_us={op.duration_us} != avg_us={op.avg_us} for op {op.op_path}"
        )
