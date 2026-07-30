# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests the four control-flow branches of :func:`isolated_ep_register`.

``subprocess.run`` is patched — no real child is spawned. Assertions
target call arguments and exception messages only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.commands.sys import isolated_ep_register
from winml.modelkit.session import WinMLEPRegistrationFailed


_EP = "OpenVINOExecutionProvider"
_DLL = Path(r"C:\fake\onnxruntime_providers_openvino_plugin.dll")


def _fake_completed_process(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> MagicMock:
    """Build a mock ``CompletedProcess`` shape for ``subprocess.run``."""
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


# ---------------------------------------------------------------------------
# Branch 1: success path
# ---------------------------------------------------------------------------


class TestSuccess:
    """Child produces JSON on stdout and exits 0 → dict is yielded."""

    def test_success_yields_parsed_dict(self) -> None:
        stdout = (
            '{"plugin_version": "1.4.1+f33af4f",'
            ' "devices": [{"device_type": "NPU", "hardware_name": "AI Boost",'
            ' "vendor": "Intel", "facts": [], "device_facts": []}]}'
        )
        with (
            patch(
                "subprocess.run",
                return_value=_fake_completed_process(stdout=stdout),
            ) as mock_run,
            isolated_ep_register(_EP, _DLL) as result,
        ):
            pass

        # Yielded dict shape.
        assert isinstance(result, dict)
        assert result["plugin_version"] == "1.4.1+f33af4f"
        assert result["devices"][0]["device_type"] == "NPU"

        # Argv shape passed to subprocess.run: sys.executable + `-c` +
        # <worker script> + ep_name + dll_path.
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == "-c"
        assert "_worker()" in cmd[2]  # the injected trailing call
        assert cmd[3] == _EP
        assert cmd[4] == str(_DLL)

        # capture_output + text kwargs so we can read stdout as str.
        kwargs = mock_run.call_args.kwargs
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 30.0  # default


# ---------------------------------------------------------------------------
# Branch 2: worker exited non-zero
# ---------------------------------------------------------------------------


class TestNonZeroExit:
    """Child registration raised → subprocess exits 1 → WinMLEPRegistrationFailed."""

    def test_nonzero_exit_carries_stderr_tail(self) -> None:
        stderr = "some ORT native error\n<traceback>\nOSError: something bad"
        proc = _fake_completed_process(
            stdout="", stderr=stderr, returncode=1,
        )
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL),
        ):
            pass

        msg = str(ei.value)
        assert "exited 1" in msg
        assert "OSError: something bad" in msg
        assert str(_DLL) in msg

    def test_nonzero_exit_stderr_tail_capped_at_500_chars(self) -> None:
        long_stderr = "X" * 2000
        proc = _fake_completed_process(returncode=1, stderr=long_stderr)
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL),
        ):
            pass

        # Only the tail is kept; the message should not be arbitrarily long.
        assert len(str(ei.value)) < 1000

    def test_nonzero_exit_reason_is_clean_exception_line(self) -> None:
        # A worker that fails without a Win32 loader code prints a multi-line
        # Python traceback. The user-facing ``.reason`` (rendered in the
        # ``[failed]`` row) must be the real exception line — not the wrapper
        # prefix or a mid-traceback fragment. Regression guard: the observed
        # "isolated register of ... exited 1: d." mangling, where the parser's
        # first-line-of-wrapper fallback surfaced a broken traceback fragment.
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 9, in _worker\n'
            "    winml_ep = WinMLEPRegistry.instance().register_ep(entry)\n"
            "RuntimeError: EP factory returned no OrtEpDevices\n"
        )
        proc = _fake_completed_process(stderr=stderr, returncode=1)
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL),
        ):
            pass

        exc = ei.value
        assert exc.code is None
        assert exc.reason == "RuntimeError: EP factory returned no OrtEpDevices"
        assert "isolated register of" not in exc.reason
        # Full wrapper message is still preserved for logs / str().
        assert "exited 1" in str(exc)

    def test_nonzero_exit_win32_code_in_stderr_tail_maps_to_reason(self) -> None:
        # A coded DLL-load failure whose Win32 code lands in the last stderr
        # line still maps to the friendly reason via the clean tail.
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 9, in _worker\n'
            "RuntimeError: Failed to load library "
            "(Error 1114: A dynamic link library initialization routine failed.)"
        )
        proc = _fake_completed_process(stderr=stderr, returncode=1)
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL),
        ):
            pass

        assert ei.value.code == 1114
        assert ei.value.reason == "DllMain returned failure (Win32 1114)"


# ---------------------------------------------------------------------------
# Branch 3: worker timed out
# ---------------------------------------------------------------------------


class TestTimeout:
    """Child hangs → TimeoutExpired → we translate and preserve stderr."""

    def test_timeout_translates_to_registration_failed(self) -> None:
        exc = subprocess.TimeoutExpired(
            cmd=["python", "-c", "..."],
            timeout=5.0,
            output=b"",
            stderr=b"driver init log line\nanother log line",
        )
        with (
            patch(
                "subprocess.run",
                side_effect=exc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL, timeout=5.0),
        ):
            pass

        msg = str(ei.value)
        assert "timed out after 5.0s" in msg
        assert str(_DLL) in msg
        # Regression guard: stderr tail from TimeoutExpired must survive.
        assert "driver init log line" in msg

    def test_timeout_with_string_stderr_also_survives(self) -> None:
        """TimeoutExpired.stderr can be str when subprocess.run(text=True)."""
        exc = subprocess.TimeoutExpired(
            cmd=["python"], timeout=1.0,
            output="", stderr="text-mode stderr content",
        )
        with (
            patch(
                "subprocess.run",
                side_effect=exc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL, timeout=1.0),
        ):
            pass

        assert "text-mode stderr content" in str(ei.value)

    def test_timeout_without_stderr_still_reports_timeout(self) -> None:
        """No stderr captured — the timeout itself is still surfaced."""
        exc = subprocess.TimeoutExpired(
            cmd=["python"], timeout=1.0, output=None, stderr=None,
        )
        with (
            patch(
                "subprocess.run",
                side_effect=exc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL, timeout=1.0),
        ):
            pass

        assert "timed out after 1.0s" in str(ei.value)


# ---------------------------------------------------------------------------
# Branch 4: garbled JSON on stdout
# ---------------------------------------------------------------------------


class TestBadJSON:
    """Child exited 0 but stdout wasn't valid JSON → translate error."""

    def test_bad_json_carries_stdout_tail(self) -> None:
        # A plugin's C++ runtime that decides to printf a banner to
        # stdout before the JSON payload would produce something like:
        garbled = "OpenVINO runtime init v1.4.1\n{malformed json"
        proc = _fake_completed_process(stdout=garbled, returncode=0)
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed) as ei,
            isolated_ep_register(_EP, _DLL),
        ):
            pass

        msg = str(ei.value)
        assert "invalid JSON" in msg
        assert "malformed json" in msg  # tail preserved

    def test_completely_empty_stdout_reports_invalid_json(self) -> None:
        """Empty stdout on a 0-exit child is still a JSON-parse failure."""
        proc = _fake_completed_process(stdout="", returncode=0)
        with (
            patch(
                "subprocess.run",
                return_value=proc,
            ),
            pytest.raises(WinMLEPRegistrationFailed),
            isolated_ep_register(_EP, _DLL),
        ):
            pass


# ---------------------------------------------------------------------------
# Timeout parameter propagation
# ---------------------------------------------------------------------------


def test_custom_timeout_passed_to_subprocess_run() -> None:
    """Non-default timeout kwarg propagates to subprocess.run."""
    stdout = '{"plugin_version": null, "devices": []}'
    # Note: {"devices": []} would violate the WinMLEP invariant in a
    # real run — but that check lives in the child, not here. This
    # test only verifies the timeout plumbing.
    with (
        patch(
            "subprocess.run",
            return_value=_fake_completed_process(stdout=stdout),
        ) as mock_run,
        isolated_ep_register(_EP, _DLL, timeout=90.0),
    ):
        pass

    assert mock_run.call_args.kwargs["timeout"] == 90.0


# ---------------------------------------------------------------------------
# Caller-side errors must pass through — the context manager only translates
# the *subprocess's own* invalid-JSON output, never an exception raised inside
# the ``with`` body.
# ---------------------------------------------------------------------------


class TestCallerErrorPassthrough:
    def test_caller_json_decode_error_is_not_trapped(self) -> None:
        # The child produced valid JSON (exit 0), so the dict is yielded.
        # A JSONDecodeError raised inside the ``with`` body must propagate
        # as-is, NOT be misattributed as "subprocess produced invalid JSON".
        proc = _fake_completed_process(
            stdout='{"plugin_version": "1.0", "devices": []}', returncode=0,
        )
        with (
            patch("subprocess.run", return_value=proc),
            pytest.raises(json.JSONDecodeError),
            isolated_ep_register(_EP, _DLL),
        ):
            raise json.JSONDecodeError("caller boom", "doc", 0)
