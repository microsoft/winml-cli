# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from winml.modelkit.telemetry.utils import (
    _format_exception_message,
    _scrub_model_ref,
    _scrub_pii,
    _trim_path,
)


@pytest.mark.parametrize(
    "input_path,expected",
    [
        # Windows absolute path -> package-relative
        (
            r"C:\Users\Alice\src\winml\modelkit\commands\build.py",
            "winml/modelkit/commands/build.py",
        ),
        # Already-relative stays relative, slashes normalized
        (
            r"winml\modelkit\commands\build.py",
            "winml/modelkit/commands/build.py",
        ),
        # Path without the package prefix falls back to basename
        (
            r"C:\Users\Alice\scripts\external.py",
            "external.py",
        ),
        # Empty / missing -> empty
        ("", ""),
    ],
)
def test_trim_path(input_path, expected):
    assert _trim_path(input_path) == expected


@pytest.mark.parametrize(
    "before,after",
    [
        # Email
        ("contact alice@example.com for help", "contact <scrubbed> for help"),
        # GUID (case-insensitive)
        (
            "job id 12345678-1234-5678-1234-567812345678 failed",
            "job id <scrubbed> failed",
        ),
        (
            "JOB id 12345678-1234-5678-1234-567812345678 failed",
            "JOB id <scrubbed> failed",
        ),
        # IPv4
        (
            "cannot reach 192.168.1.100 port 8080",
            "cannot reach <scrubbed> port 8080",
        ),
        # IPv6 (common compressed form)
        (
            "cannot reach 2001:db8::1 port 443",
            "cannot reach <scrubbed> port 443",
        ),
        # Long opaque token (24+ alphanumeric/underscore/dash)
        (
            "Authorization: Bearer abcdef0123456789_abcdef_9999",
            "Authorization: Bearer <scrubbed>",
        ),
        # Nothing sensitive -> unchanged
        ("Invalid argument: expected int", "Invalid argument: expected int"),
        # Multiple hits
        (
            "user alice@example.com at 10.0.0.1",
            "user <scrubbed> at <scrubbed>",
        ),
        # Long all-letter identifiers (class / function names) must NOT
        # be scrubbed - they're diagnostic, not secrets. Regression for
        # PR 371 review feedback.
        (
            "TypeError in WinMLImageFeatureExtractionEvaluator.compute",
            "TypeError in WinMLImageFeatureExtractionEvaluator.compute",
        ),
        # Long token WITH at least one digit is still scrubbed.
        (
            "token=abcdef0123456789abcdefghijk",
            "token=<scrubbed>",
        ),
        # Invalid-octet IPv4 is NOT scrubbed (octet > 255 is not a real IP).
        (
            "build number 999.888.777.666",
            "build number 999.888.777.666",
        ),
    ],
)
def test_scrub_pii(before, after):
    assert _scrub_pii(before) == after


def test_format_exception_message_applies_path_trim_and_pii_and_length():
    # Path + email + length - all three pipeline stages exercised.
    msg = (
        "FileNotFoundError: "
        r"C:\Users\Alice\src\winml\modelkit\commands\build.py "
        "email: alice@example.com " + "x" * 500
    )
    result = _format_exception_message(msg)
    # Path trimmed
    assert "winml/modelkit/commands/build.py" in result
    # PII scrubbed
    assert "alice@example.com" not in result
    assert "<scrubbed>" in result
    # Length capped at 200, with truncation marker
    assert len(result) <= 200
    assert result.endswith("…")


def test_format_exception_message_short_passes_through():
    # Short, no PII, no absolute paths.
    assert _format_exception_message("ValueError: expected int") == "ValueError: expected int"


def test_format_exception_message_empty_is_empty():
    assert _format_exception_message("") == ""


def test_format_exception_message_none_safe():
    # Defensive: accepts None by returning empty string.
    assert _format_exception_message(None) == ""


def test_format_exception_message_scrubs_pii_at_cap_boundary():
    """Regression: PII straddling the 200-char cap must be scrubbed
    before truncation. Cap-then-scrub would leak the email's local part
    because the cropped fragment ``alice@exa…`` no longer matches the
    email regex (no TLD). The ``<scrubbed>`` placeholder itself may be
    truncated by the cap - what matters is that the original PII
    fragments are gone."""
    prefix = "x" * 195  # email lands across char 196 onward
    msg = prefix + " alice@example.com"
    result = _format_exception_message(msg)
    assert "alice" not in result
    assert "@" not in result
    assert len(result) <= 200


def test_format_exception_message_scrubs_long_token_at_cap_boundary():
    """Regression: a long opaque token straddling the cap must be scrubbed
    as a whole, not leak its in-cap prefix as a sub-24-char fragment."""
    prefix = "x" * 180  # 30-char token starts at 181, ends at 210
    msg = prefix + " token=abcdef0123456789abcdefghi9"
    result = _format_exception_message(msg)
    assert "abcdef0123456789" not in result
    assert "<scrubbed>" in result
    assert len(result) <= 200


@pytest.mark.parametrize(
    "value,expected",
    [
        # Clean HF ID — passthrough
        ("microsoft/resnet-50", "microsoft/resnet-50"),
        ("google-bert/bert-base-uncased", "google-bert/bert-base-uncased"),
        # Single-segment canonical Hub ids (no org prefix) pass through too —
        # these are real, commonly-used ids and are the documented `perf -m`
        # form. Regression for PR #1108 review.
        ("bert-base-uncased", "bert-base-uncased"),
        ("gpt2", "gpt2"),
        ("mymodel", "mymodel"),
        # Two-segment id with a dot in the name segment is still an id, not a
        # file (the dot is part of the id, not a file extension).
        ("org/model.v2", "org/model.v2"),
        # Windows absolute path with .onnx file
        (r"C:\Users\alice\models\resnet50-int8.onnx", "<local:.onnx>"),
        # POSIX-style absolute path (defensive; leading slash)
        ("/home/x/model.onnx", "<local:.onnx>"),
        # Backslash-relative path with extension
        (r".\output\model.onnx", "<local:.onnx>"),
        # Directory-style reference (no extension) via backslash separator
        (r".\output\qwen3-bundle", "<local:dir>"),
        # Single-segment name carrying a file extension is a local file ref,
        # not a Hub id (Hub ids don't carry file extensions).
        ("model.onnx", "<local:.onnx>"),
        # eval's `role=path` composite: the `=` makes it non-Hub, so the
        # `sub/model.onnx` fragment is never emitted verbatim. Regression for
        # PR #1108 review.
        ("encoder=sub/model.onnx", "<local:.onnx>"),
        # Tuple (multiple=True) — first element classified
        (("microsoft/resnet-50", "other/model"), "microsoft/resnet-50"),
        (("model.onnx",), "<local:.onnx>"),
        # None / empty / empty tuple
        (None, None),
        ("", None),
        ((), None),
    ],
)
def test_scrub_model_ref(value, expected):
    assert _scrub_model_ref(value) == expected


def test_scrub_model_ref_existing_path_is_local(tmp_path, monkeypatch):
    """A single-token relative name that exists on disk is treated as a
    local path, not an HF id — even without a separator."""
    f = tmp_path / "on_disk.onnx"
    f.write_text("x")
    monkeypatch.chdir(tmp_path)
    assert _scrub_model_ref("on_disk.onnx") == "<local:.onnx>"


def test_scrub_model_ref_org_name_that_exists_is_local(tmp_path, monkeypatch):
    """If an `org/name` string happens to exist on disk, prefer the local
    marker — existence wins over the HF-id shape."""
    d = tmp_path / "org"
    d.mkdir()
    (d / "name").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _scrub_model_ref("org/name") == "<local:dir>"
