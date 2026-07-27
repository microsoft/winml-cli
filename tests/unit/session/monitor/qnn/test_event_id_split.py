# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""``_split_op_event_id`` is now a whitespace-strip helper for the CSV path.

The op-type-leaf recovery previously performed here duplicated the logic
in ``QNNMonitor._heuristic_op_type`` (which strips ``_TOKEN_SUFFIX`` first
and is the authoritative resolver for the CSV path).  This helper's sole
remaining responsibility is normalising outer whitespace so downstream
dict keys are stable.
"""

from winml.modelkit.session.monitor.qnn._internal import _split_op_event_id


def test_bare_event_id_is_returned_unchanged():
    assert _split_op_event_id("Gelu") == "Gelu"


def test_path_event_id_is_returned_verbatim():
    event_id = "/convnext/embeddings/layernorm/LayerNormalization"
    assert _split_op_event_id(event_id) == event_id


def test_trailing_slash_is_preserved():
    # No leaf-extraction — the caller relies on _heuristic_op_type downstream.
    assert _split_op_event_id("/encoder/") == "/encoder/"


def test_empty_string_does_not_crash():
    assert _split_op_event_id("") == ""


def test_outer_whitespace_is_stripped():
    """Leading/trailing whitespace on the event id is stripped for stable keys."""
    assert _split_op_event_id("  /encoder/Conv  ") == "/encoder/Conv"


def test_whitespace_only_returns_empty():
    """Whitespace-only input collapses to the empty string after strip."""
    assert _split_op_event_id("   ") == ""
