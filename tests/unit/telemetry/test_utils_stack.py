# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from winml.modelkit.telemetry.utils import _extract_exception_stack


def _raise_chain():
    def inner():
        raise ValueError("boom")

    def outer():
        inner()

    try:
        outer()
    except ValueError as e:
        return e


def test_extract_exception_stack_returns_list_of_dicts():
    exc = _raise_chain()
    frames = _extract_exception_stack(exc.__traceback__)
    assert isinstance(frames, list)
    assert len(frames) >= 2  # outer + inner
    for frame in frames:
        assert set(frame.keys()) == {"file", "line", "function"}
        assert isinstance(frame["line"], int)
        assert isinstance(frame["file"], str)
        assert isinstance(frame["function"], str)


def test_extract_exception_stack_trims_paths():
    exc = _raise_chain()
    frames = _extract_exception_stack(exc.__traceback__)
    # Every frame's file is either package-relative or a basename -
    # never a user-specific absolute path.
    for frame in frames:
        # No drive letter (e.g., "C:") in the first segment.
        assert ":" not in frame["file"].split("/")[0]
        # No absolute-Windows-path prefix.
        assert not frame["file"].startswith(("C:", "c:"))


def test_extract_exception_stack_contains_no_message_or_locals():
    exc = _raise_chain()
    frames = _extract_exception_stack(exc.__traceback__)
    # No surprise fields - strictly {file, line, function}.
    for frame in frames:
        assert "locals" not in frame
        assert "source" not in frame
        assert "message" not in frame
        assert "line_text" not in frame


def test_extract_exception_stack_on_none_returns_empty():
    assert _extract_exception_stack(None) == []
