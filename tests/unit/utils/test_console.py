# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for safe console output helpers."""

from __future__ import annotations

import errno

import pytest

from winml.modelkit.utils import console as console_module


def test_safe_console_print_helper_is_not_part_of_console_api():
    assert not hasattr(console_module, "safe_console_print")


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
