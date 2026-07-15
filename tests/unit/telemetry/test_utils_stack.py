# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from winml.modelkit.telemetry.utils import _extract_exception_stack, _root_cause


def _raise_chain():
    def inner():
        raise ValueError("boom")

    def outer():
        inner()

    try:
        outer()
    except ValueError as e:
        return e
    # Unreachable: outer() always raises. Explicit return keeps CodeQL happy.
    return None  # pragma: no cover


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


def test_root_cause_no_chain_returns_self():
    exc = ValueError("boom")
    assert _root_cause(exc) is exc


def test_root_cause_follows_explicit_cause():
    root = ValueError("root")
    try:
        try:
            raise root
        except ValueError as e:
            raise RuntimeError("wrapper") from e
    except RuntimeError as outer:
        assert _root_cause(outer) is root


def test_root_cause_follows_implicit_context():
    root = ValueError("root")
    try:
        try:
            raise root
        except ValueError:
            # No `from` — Python sets __context__ implicitly. B904 is the
            # exact pattern under test here, so the lint is suppressed.
            raise RuntimeError("wrapper")  # noqa: B904
    except RuntimeError as outer:
        assert _root_cause(outer) is root


def test_root_cause_honors_suppressed_context():
    # `raise ... from None` sets __suppress_context__=True and __cause__=None.
    # The suppressed inner exception must NOT be walked into — matching
    # Python's own traceback printing and the developer's intent to hide it.
    try:
        try:
            raise ValueError("suppressed inner")
        except ValueError:
            raise RuntimeError("clean outer") from None
    except RuntimeError as outer:
        # No chain surfaces: the outer exception is its own root cause.
        assert _root_cause(outer) is outer


def test_root_cause_walks_multiple_levels_to_innermost():
    innermost = OSError("disk full")
    try:
        try:
            try:
                raise innermost
            except OSError as e1:
                raise RuntimeError("mid") from e1
        except RuntimeError as e2:
            raise ValueError("top") from e2
    except ValueError as outer:
        assert _root_cause(outer) is innermost


def test_root_cause_prefers_cause_over_context():
    cause = ValueError("the cause")
    context = KeyError("the context")
    outer = RuntimeError("outer")
    # Simulate: raised inside handling `context`, but `from cause`.
    outer.__context__ = context
    outer.__cause__ = cause
    assert _root_cause(outer) is cause


def test_root_cause_cycle_guard_terminates():
    a = ValueError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    # Must terminate and return one of the two, not loop forever.
    result = _root_cause(a)
    assert result in (a, b)
