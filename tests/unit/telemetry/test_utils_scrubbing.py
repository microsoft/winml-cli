# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from winml.modelkit.telemetry.utils import (
    _format_exception_message,
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
