# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""``_split_op_event_id`` is a heuristic-only fallback for the CSV path.

The QNN basic-mode profiling CSV has no op-type column, so we recover
one from the event ID.  QNN compiler emits event IDs as either bare op
types (``"Gelu"``) or hierarchical paths (``"/encoder/layer/Conv"``);
the leaf segment of a path is *usually* the ONNX op symbol — a
best-effort fallback only.

``OperatorMetrics.name`` and ``OperatorMetrics.op_path`` carry distinct
concepts (per the dataclass docstring) and must NOT both be populated
with the raw event ID — doing so causes the report's Type column to
render the truncated path for path-style events.

The QHAS path supplies the authoritative QNN op type via
``qnn_op_type`` and MUST NOT call this helper — see
the QHAS branch of :mod:`winml.modelkit.session.monitor.qnn._internal`.
"""

from winml.modelkit.session.monitor.qnn._internal import _split_op_event_id


def test_path_event_id_extracts_leaf_as_type():
    op_type, op_path = _split_op_event_id("/convnext/embeddings/layernorm/LayerNormalization")
    assert op_type == "LayerNormalization"
    assert op_path == "/convnext/embeddings/layernorm/LayerNormalization"


def test_bare_event_id_is_both_type_and_path():
    op_type, op_path = _split_op_event_id("Gelu")
    assert op_type == "Gelu"
    assert op_path == "Gelu"


def test_trailing_slash_falls_back_gracefully():
    # Defensive: if QNN emits something weird like '/encoder/' it should not crash.
    op_type, op_path = _split_op_event_id("/encoder/")
    # The contract is "no crash, op_path preserved, op_type non-empty".
    assert op_path == "/encoder/"
    assert op_type, "op_type must never be empty for non-empty input"
    # Trailing slash means leaf is empty after split → fallback to full string.
    assert op_type == "/encoder/"


def test_simple_two_segment_path():
    op_type, op_path = _split_op_event_id("/MatMul")
    assert op_type == "MatMul"
    assert op_path == "/MatMul"


def test_deep_path_extracts_only_leaf():
    op_type, op_path = _split_op_event_id("/resnet/embedder/embedder/convolution/Conv")
    assert op_type == "Conv"
    assert op_path == "/resnet/embedder/embedder/convolution/Conv"


def test_empty_string_does_not_crash():
    # Edge: parser should never raise on degenerate input.
    op_type, op_path = _split_op_event_id("")
    assert op_path == ""
    assert op_type == ""


def test_strip_safety_outer_whitespace():
    """Leading/trailing whitespace on the event id is stripped before splitting.

    QNN profiling output occasionally surfaces padding around event ids;
    without ``.strip()`` the path comparison and downstream rendering
    would carry the noise into ``op_path`` and the leaf would be the
    same trailing whitespace, poisoning equality checks.
    """
    op_type, op_path = _split_op_event_id("  /encoder/Conv  ")
    assert op_type == "Conv"
    assert op_path == "/encoder/Conv"


def test_strip_safety_inner_whitespace_around_leaf():
    """Whitespace around the leaf segment is stripped after the split.

    Defensive: e.g. ``"/encoder/Conv "`` → leaf is ``"Conv "`` before
    strip; the helper strips it back to ``"Conv"`` so the Type column
    isn't visually broken.
    """
    op_type, _op_path = _split_op_event_id("/encoder/Conv ")
    assert op_type == "Conv"


def test_strip_whitespace_only_returns_empty():
    """Whitespace-only input is treated as empty after ``.strip()``."""
    op_type, op_path = _split_op_event_id("   ")
    assert op_type == ""
    assert op_path == ""


def test_docstring_marks_helper_as_heuristic_only():
    """The helper's docstring must explicitly call out heuristic-only usage.

    Future maintainers must not mistake this fallback for an
    authoritative source; the QHAS path has ``qnn_op_type`` and should
    use it directly.  Pinning the wording in the docstring guards
    against silent rewordings that would invite reuse on the QHAS path.
    """
    doc = _split_op_event_id.__doc__ or ""
    assert "heuristic" in doc.lower()
    assert "qhas" in doc.lower()
