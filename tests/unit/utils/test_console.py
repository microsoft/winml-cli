# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for safe console output helpers."""

from __future__ import annotations

import errno

import pytest

from winml.modelkit.utils import console as console_module


class _FailingConsole:
    def __init__(self, exc: OSError) -> None:
        self.exc = exc

    def print(self, *args: object, **kwargs: object) -> None:
        raise self.exc


def test_safe_console_print_ignores_expected_windows_console_errors(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    console_module.safe_console_print(_FailingConsole(OSError(1, "Incorrect function")), "x")


def test_safe_console_print_reraises_non_console_oserror(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    with pytest.raises(OSError, match="No space"):
        console_module.safe_console_print(_FailingConsole(OSError(errno.ENOSPC, "No space")), "x")


def test_safe_console_print_does_not_parse_error_code_from_message(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    with pytest.raises(OSError, match="Windows error: 6"):
        console_module.safe_console_print(_FailingConsole(OSError("Windows error: 6")), "x")


def test_safe_console_print_reraises_console_error_on_non_windows(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "linux")

    with pytest.raises(OSError, match="Incorrect function"):
        console_module.safe_console_print(_FailingConsole(OSError(1, "Incorrect function")), "x")


def test_safe_console_ignores_expected_windows_console_errors(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    def fail_print(self: object, *args: object, **kwargs: object) -> None:
        raise OSError(1, "Incorrect function")

    monkeypatch.setattr(console_module.Console, "print", fail_print)

    console_module.SafeConsole().print("x")


def test_safe_console_reraises_non_console_oserror(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    def fail_print(self: object, *args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space")

    monkeypatch.setattr(console_module.Console, "print", fail_print)

    with pytest.raises(OSError, match="No space"):
        console_module.SafeConsole().print("x")


def test_safe_live_ignores_expected_windows_console_errors(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    def fail_start(self: object, refresh: bool = False) -> None:
        raise OSError(1, "Incorrect function")

    monkeypatch.setattr(console_module.Live, "start", fail_start)

    console_module.SafeLive("x").start()


def test_safe_live_has_no_private_alias():
    assert not hasattr(console_module, "_SafeLive")


def test_safe_live_reraises_non_console_oserror_from_wrapped_methods(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    def fail_start(self: object, refresh: bool = False) -> None:
        raise OSError(errno.ENOSPC, "No space")

    monkeypatch.setattr(console_module.Live, "start", fail_start)

    with pytest.raises(OSError, match="No space"):
        console_module.SafeLive("x").start()


def test_safe_live_reraises_non_console_oserror_from_refresh(monkeypatch):
    monkeypatch.setattr(console_module.sys, "platform", "win32")

    def fail_refresh(self: object) -> None:
        raise OSError(errno.ENOSPC, "No space")

    monkeypatch.setattr(console_module.Live, "refresh", fail_refresh)

    with pytest.raises(OSError, match="No space"):
        console_module.SafeLive("x").refresh()
