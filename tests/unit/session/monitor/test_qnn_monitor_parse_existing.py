# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""v2.4 QNNMonitor.parse_existing_artifacts — offline / post-hoc parsing.

Phase 2 addition (spec §3.2 / coreloop §4.3).  Lets callers parse
pre-existing QNN profiling artifacts without running a benchmark, with
optional ONNX op-type map injection for the L1 lookup layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "qnn" / "fixtures"


def test_parse_existing_artifacts_basic_csv(tmp_path):
    """Basic-mode parsing of a pre-existing CSV without an ONNX map.

    Falls through L1 (empty map) → L2 (no ep_authoritative on CSV path)
    → L3 (heuristic).  The leaf segments of path-style event ids
    surface as ``op.name``.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = QNNMonitor.parse_existing_artifacts(level="basic", artifacts={"csv": csv_path})

    assert result.status == "ok"
    assert result.tracing_level == "basic"
    assert result.operators, "fixture should yield operators"

    path_style = [op for op in result.operators if "/" in op.op_path]
    assert path_style, "fixture should contain at least one path-style operator"
    for op in path_style:
        # L3 heuristic wins → leaf segment surfaces in name.
        assert "/" not in op.name, f"leaf-split should drop slashes; got {op.name!r}"


def test_parse_existing_artifacts_uses_injected_onnx_map(tmp_path):
    """Injected ONNX op-type map drives L1 hits for matching paths."""
    from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # Discover one real op_path from the fixture so we can target it.
    parsed = parse_qnn_profiling_csv(csv_path)
    target = next(op for op in parsed["operators"] if "/" in op["op_path"])
    target_path = target["op_path"]

    result = QNNMonitor.parse_existing_artifacts(
        level="basic",
        artifacts={"csv": csv_path},
        onnx_op_types={target_path: "MyCustomOp"},
    )

    assert result.status == "ok"
    by_path = {op.op_path: op for op in result.operators}
    assert by_path[target_path].name == "MyCustomOp", (
        f"L1 should override heuristic; got {by_path[target_path].name!r}"
    )


def test_parse_existing_artifacts_rejects_missing_csv_key():
    """``artifacts`` must contain a 'csv' key — explicit raise on misuse."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with pytest.raises(ValueError, match="csv"):
        QNNMonitor.parse_existing_artifacts(level="basic", artifacts={})


def test_parse_existing_artifacts_default_onnx_map_is_empty(tmp_path):
    """Calling without ``onnx_op_types`` is equivalent to passing an empty map."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    a = QNNMonitor.parse_existing_artifacts(level="basic", artifacts={"csv": csv_path})
    b = QNNMonitor.parse_existing_artifacts(
        level="basic", artifacts={"csv": csv_path}, onnx_op_types={}
    )
    assert [(op.name, op.op_path) for op in a.operators] == [
        (op.name, op.op_path) for op in b.operators
    ]


def test_parse_existing_artifacts_honors_explicit_csv_filename(tmp_path):
    """The constructor pins ``profiling_output.csv``; the classmethod must
    honour the caller's explicit filename so non-default names work."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    # Name the fixture something other than the constructor's default.
    custom = tmp_path / "alternate_name.csv"
    custom.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = QNNMonitor.parse_existing_artifacts(level="basic", artifacts={"csv": custom})
    assert result.status == "ok"
    assert result.artifacts["csv"] == str(custom.resolve())


def test_parse_existing_artifacts_detail_qhas_override(tmp_path):
    """Detail mode with a pre-supplied QHAS JSON skips the viewer shell-out."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    qhas_path = tmp_path / "qhas_output.json"
    qhas_path.write_text(
        (FIXTURE_DIR / "qhas_resnet50.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = QNNMonitor.parse_existing_artifacts(
        level="detail",
        artifacts={"csv": csv_path, "qhas": qhas_path},
    )

    assert result.status == "ok"
    assert result.tracing_level == "detail"
    assert "qhas" in result.artifacts
    # Empty ONNX map → L2 (qnn_op_type) wins.
    first = result.operators[0]
    assert first.name == "Conv2d", (
        f"detail mode without ONNX map should surface QHAS qnn_op_type; got {first.name!r}"
    )


def test_parse_existing_artifacts_returns_failed_result_on_corrupt_csv(tmp_path):
    """When artifacts cannot be parsed, parse_existing_artifacts returns
    OpTraceResult(status='parse_failed', error=...) rather than raising.

    This is the same contract __exit__ honors — both paths must produce
    a typed failure result instead of propagating exceptions.  Pre-Bundle-B
    the offline path called ``_parse_artifacts`` raw, so corrupt artifacts
    raised out of the classmethod and forced every caller (including the
    CLI) to wrap their own try/except.

    Uses non-UTF-8 binary content so the underlying ``csv.DictReader``
    actually raises (a syntactically-empty CSV merely yields zero rows
    and would parse successfully as an empty trace).
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    corrupt_csv = tmp_path / "corrupt.csv"
    # Non-UTF-8 bytes guarantee the parser's open(...encoding="utf-8")
    # raises UnicodeDecodeError, exercising the safe-wrapper path.
    corrupt_csv.write_bytes(b"\xff\xfe\x00not\x00a\x00real\x00csv\xff")

    result = QNNMonitor.parse_existing_artifacts(
        level="basic",
        artifacts={"csv": corrupt_csv},
    )

    assert result.status == "parse_failed"
    assert result.error is not None
    assert len(result.error) > 0  # error message is populated


def test_exit_and_parse_existing_share_parse_failed_contract(tmp_path):
    """__exit__ and parse_existing_artifacts produce the SAME
    OpTraceResult.status when fed identical corrupt input.

    Pins the Bundle B refactor: both paths route through
    ``_parse_artifacts_safe``, so a parse failure on the live path
    (``__exit__``) and the offline path (``parse_existing_artifacts``)
    must yield identical status strings.  Guards against future drift
    where someone forgets to thread one of the two through the safe
    helper.

    Uses non-UTF-8 binary bytes so ``parse_qnn_profiling_csv`` actually
    raises ``UnicodeDecodeError`` — both paths must catch it.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    corrupt_bytes = b"\xff\xfe\x00not\x00a\x00real\x00csv\xff"

    # Live path: write corrupt CSV at the constructor-pinned filename.
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "profiling_output.csv").write_bytes(corrupt_bytes)
    m = QNNMonitor(output_dir=live_dir)
    m.__enter__()
    m.__exit__(None, None, None)
    assert m.result is not None
    live_status = m.result.status

    # Offline path: same corrupt bytes, fed through parse_existing_artifacts.
    offline_csv = tmp_path / "offline.csv"
    offline_csv.write_bytes(corrupt_bytes)
    offline_result = QNNMonitor.parse_existing_artifacts(
        level="basic",
        artifacts={"csv": offline_csv},
    )

    # Pre-Bundle-B these would diverge: live path returned 'parse_failed',
    # offline path raised. Post-Bundle-B both must produce 'parse_failed'.
    assert live_status == "parse_failed"
    assert offline_result.status == "parse_failed"
    assert live_status == offline_result.status


def test_parse_existing_artifacts_detail_with_onnx_map(tmp_path):
    """Detail mode + populated ONNX map → L1 wins for matched paths.

    CRIT-1 contract: production ``_build_op_type_map`` produces clean
    ONNX ``node.name`` keys (no ``_token_N_M`` suffix).  The QHAS path's
    ``op_path`` is now token-stripped to match, so L1 lookup against
    a clean map fires correctly.  Pre-Bundle-A this test injected the
    token-bearing key — masking the production bug because L1 missed
    against the clean keys produced by ``_build_op_type_map``.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(
        (FIXTURE_DIR / "optrace_resnet50.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    qhas_path = tmp_path / "qhas_output.json"
    qhas_path.write_text(
        (FIXTURE_DIR / "qhas_resnet50.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # Production-shape key: clean ONNX node.name, no token suffix.
    target = "/resnet/embedder/embedder/convolution/Conv"

    result = QNNMonitor.parse_existing_artifacts(
        level="detail",
        artifacts={"csv": csv_path, "qhas": qhas_path},
        onnx_op_types={target: "Conv"},
    )

    assert result.status == "ok"
    by_path = {op.op_path: op for op in result.operators}
    # L1 win for the targeted path.
    assert by_path[target].name == "Conv"
