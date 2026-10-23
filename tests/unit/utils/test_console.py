# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for safe console output helpers."""

from __future__ import annotations

import errno
from io import StringIO

import pytest

from winml.modelkit.utils import console as console_module
from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES


@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize("scale", [0, 1, 1000])
def test_ep_progress_counts_have_labels_without_color(width, scale):
    buffer = StringIO()
    console = console_module.Console(file=buffer, width=width, color_system=None)
    stage = console_module.StageLive("optimize", console)
    ep_name = next(iter(EP_SUPPORTED_DEVICES))
    counts = tuple(scale * index for index in range(1, 4))
    total = sum(counts) + scale
    idx = stage.ep_bar_add(ep_name, total=total)
    stage.ep_bar_update(idx, ep_name, *counts, total=total)
    console.print(stage._render())

    output = buffer.getvalue()
    assert ep_name in output
    for label, count in zip(("Supported", "Partial", "Unsupported"), counts, strict=True):
        assert f"{label}: {count}" in output
    assert console_module._build_bar_scaled(*counts, max(total, sum(counts), 1)).plain in output
    assert all(len(line) <= width for line in output.splitlines())


@pytest.mark.parametrize("width", [80, 120])
def test_stage_status_labels_survive_rendering_without_color(width):
    buffer = StringIO()
    console = console_module.Console(file=buffer, width=width, color_system=None)
    name = test_stage_status_labels_survive_rendering_without_color.__name__
    stage = console_module.StageLive(name, console)
    stage._lines = [stage._make_running_line()]
    console.print(stage._render())
    stage.set_done(0)
    console.print(stage._render())
    stage.set_error()
    console.print(stage._render())
    console_module.print_stage_skip(console, name)

    output = buffer.getvalue()
    for label in ("[In progress]", "[Done]", "[Failed]", "[Skipped]"):
        assert label in output
    assert name.capitalize() in output


@pytest.mark.parametrize("width", [80, 120])
def test_shared_markup_status_labels_and_setup_are_plain_text(width):
    buffer = StringIO()
    console = console_module.Console(file=buffer, width=width, color_system=None)
    value = test_shared_markup_status_labels_and_setup_are_plain_text.__name__
    console_module.print_setup(console, model=value, config=value, output=value)
    console_module.print_stages_header(console)
    console_module.print_io_specs_na(console)
    console_module.print_success(console, value)
    console_module.print_error(console, value, hint=value)
    console_module.print_final(console, elapsed=0, artifact=value, config=value)
    stage = console_module.StageLive(value, console)
    stage.artifact(value, 0)
    console.print(stage._render())

    output = buffer.getvalue()
    assert output.count("[Done]") == 2
    assert output.count("[Failed]") == 1
    for label in ("Setup", "Model:", "Config:", "Output:", "Stages", "Summary", "Artifact:"):
        assert label in output
    assert not any(
        0x1F000 <= ord(char) <= 0x1FAFF or 0x2600 <= ord(char) <= 0x27BF
        for char in output
    )


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
