# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the structured attributes on :class:`WinMLEPRegistrationFailed`.

The exception carries ``code`` / ``reason`` / ``fallback_version`` /
``dll_path`` so callers (``winml sys --list-ep`` renderer, session
retry loops) can render a compact ``[failed]`` row directly off the
exception — no re-parsing of the raw ORT message.

Every test constructs the exception through the public constructor and
reads the public attributes. No private-symbol imports (per the
CLAUDE.md src-code import rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.session import WinMLEPRegistrationFailed


# ---------------------------------------------------------------------------
# ORT message shape 1: ``... with error code: N``
#   (OpenVINO / VitisAI paths — shim symbol mismatch under loader collision.)
# ---------------------------------------------------------------------------


class TestErrorCodeShape:
    def test_code_127_sets_symbol_not_resolved_reason(self) -> None:
        e = WinMLEPRegistrationFailed(
            "[ONNXRuntimeError] : 11 : EP_FAIL : Failed to load "
            r"C:\path\plugin_impl.dll with error code: 127"
        )
        assert e.code == 127
        assert e.reason == "symbol not resolved in a dependency DLL (Win32 127)"

    def test_code_without_colon_still_parses(self) -> None:
        e = WinMLEPRegistrationFailed(
            "some ORT variant failed to load foo.dll with error code 127"
        )
        assert e.code == 127


# ---------------------------------------------------------------------------
# ORT message shape 2: ``(Error N: "...")``
#   (Architecture mismatch, missing dep, DllMain fail.)
# ---------------------------------------------------------------------------


class TestParenErrorShape:
    def test_error_193_wrong_architecture(self) -> None:
        e = WinMLEPRegistrationFailed(
            r'Error loading "C:\path\qnn.dll" which is missing. '
            '(Error 193: "%1 is not a valid Win32 application.")'
        )
        assert e.code == 193
        assert e.reason == (
            "wrong architecture — ARM64 DLL in an x64 process (Win32 193)"
        )

    def test_error_1114_dllmain_returned_failure(self) -> None:
        e = WinMLEPRegistrationFailed(
            r'Error loading "C:\path\vitisai.dll" which depends on '
            '"onnxruntime_providers_shared.dll" which is missing. '
            "(Error 1114: A dynamic link library initialization routine failed.)"
        )
        assert e.code == 1114
        assert e.reason == "DllMain returned failure (Win32 1114)"

    def test_error_126_dependency_dll_missing(self) -> None:
        e = WinMLEPRegistrationFailed(
            r'Error loading "C:\path\nvtensorrt.dll" which depends on '
            '"cudart64_12.dll" which is missing. '
            "(Error 126: The specified module could not be found.)"
        )
        assert e.code == 126
        assert e.reason == "dependency DLL not found on disk (Win32 126)"

    def test_error_2_file_not_found(self) -> None:
        e = WinMLEPRegistrationFailed(
            r'Error loading "C:\gone\provider.dll" which is missing. '
            "(Error 2: The system cannot find the file specified.)"
        )
        assert e.code == 2
        assert e.reason == "file not found (Win32 2)"

    def test_error_5_access_denied(self) -> None:
        e = WinMLEPRegistrationFailed(
            r'Error loading "C:\locked\provider.dll" which is missing. '
            "(Error 5: Access is denied.)"
        )
        assert e.code == 5
        assert e.reason == "access denied (Win32 5)"


# ---------------------------------------------------------------------------
# Unknown Win32 code — still surfaces the code plus a generic reason.
# ---------------------------------------------------------------------------


def test_unknown_win32_code_gets_generic_reason() -> None:
    e = WinMLEPRegistrationFailed("Failed to load foo.dll with error code: 9999")
    assert e.code == 9999
    assert e.reason == "DLL load failed (Win32 9999)"


# ---------------------------------------------------------------------------
# Fallback: no Win32 code in the message (e.g. a Python-side exception
# escaped from the ORT layer). Message body becomes the reason.
# ---------------------------------------------------------------------------


class TestNoCodeFallback:
    def test_empty_message_gets_placeholder_reason(self) -> None:
        e = WinMLEPRegistrationFailed("")
        assert e.code is None
        assert e.reason == "(no error message)"

    def test_no_code_returns_first_line_only(self) -> None:
        e = WinMLEPRegistrationFailed(
            "AttributeError: 'NoneType' object has no attribute 'foo'\n"
            "  during ORT init\n  further context"
        )
        assert e.code is None
        assert e.reason.startswith("AttributeError")
        assert "further context" not in e.reason

    def test_no_code_no_newline_truncated_at_200(self) -> None:
        e = WinMLEPRegistrationFailed("X" * 500)
        assert e.code is None
        assert len(e.reason) == 200


# ---------------------------------------------------------------------------
# Positional bias — first Win32 code in the message wins.
# ---------------------------------------------------------------------------


def test_multiple_codes_first_match_wins() -> None:
    e = WinMLEPRegistrationFailed(
        "Failed to load foo.dll with error code: 127; (Error 1114: init failed)"
    )
    assert e.code == 127


# ---------------------------------------------------------------------------
# Case-insensitivity — ORT capitalization has drifted across versions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    ["error code: ", "Error code: ", "ERROR CODE: ", "error CODE:"],
)
def test_case_insensitive_error_code_prefix(prefix: str) -> None:
    e = WinMLEPRegistrationFailed(f"Some ORT message {prefix}193 tail")
    assert e.code == 193


# ---------------------------------------------------------------------------
# ``dll_path`` + ``fallback_version`` — read the PE VS_VERSIONINFO off disk.
# ---------------------------------------------------------------------------


# Windows-ships-with-it system DLL — always present on any Windows box
# (including CI runners), so the "real DLL" test doesn't require a local
# native build drop.
_SYSTEM_DLL = Path(r"C:\Windows\System32\ntdll.dll")


class TestFallbackVersion:
    def test_no_dll_path_leaves_fallback_version_none(self) -> None:
        e = WinMLEPRegistrationFailed("(Error 193: ...)")
        assert e.dll_path is None
        assert e.fallback_version is None

    def test_missing_file_yields_none(self, tmp_path: Path) -> None:
        e = WinMLEPRegistrationFailed(
            "(Error 126: ...)", dll_path=tmp_path / "does-not-exist.dll",
        )
        assert e.fallback_version is None

    def test_non_dll_file_yields_none(self, tmp_path: Path) -> None:
        f = tmp_path / "not-a-dll.txt"
        f.write_text("hello")
        e = WinMLEPRegistrationFailed("(Error 126: ...)", dll_path=f)
        assert e.fallback_version is None

    def test_real_dll_yields_pe_file_version(self) -> None:
        """A Windows system DLL always has a VS_VERSIONINFO resource."""
        import re
        if not _SYSTEM_DLL.is_file():
            pytest.skip(f"{_SYSTEM_DLL} not present — non-Windows or trimmed image")
        e = WinMLEPRegistrationFailed("(Error 193: ...)", dll_path=_SYSTEM_DLL)
        # ntdll.dll's version tracks the OS build (10.0.<build>.<patch>).
        # Anchor on the format rather than a specific version so the test
        # keeps passing across Windows Update revs.
        assert e.fallback_version is not None
        assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", e.fallback_version), (
            f"unexpected version format: {e.fallback_version!r}"
        )


# ---------------------------------------------------------------------------
# Exception base contract — .args, str(), from-raise chain still work.
# ---------------------------------------------------------------------------


def test_str_returns_the_message() -> None:
    e = WinMLEPRegistrationFailed("plain message")
    assert str(e) == "plain message"


def test_args_contains_the_message() -> None:
    e = WinMLEPRegistrationFailed("plain message")
    assert e.args == ("plain message",)
