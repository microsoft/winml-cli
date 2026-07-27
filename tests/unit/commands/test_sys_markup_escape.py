# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rich-markup escaping in the ``sys`` text renderers.

Error text originates from ORT / plugin subprocesses and can contain
square-bracket sequences (e.g. ``[ONNXRuntimeError]`` or a Win32 reason)
that Rich would otherwise interpret as style markup — silently dropping
the bracketed span or raising a ``MarkupError``. The renderers must
escape it so it displays literally.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

import winml.modelkit.commands.sys as sysmod


@pytest.fixture
def recording_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Swap the module console for a recording one (no ANSI, wide)."""
    rec = Console(record=True, force_terminal=False, width=200)
    monkeypatch.setattr(sysmod, "console", rec)
    return rec


def test_ep_error_reason_markup_rendered_literally(
    recording_console: Console,
) -> None:
    eps: dict[str, dict[str, Any]] = {
        "TestEP": {
            "entries": [
                {
                    "status": "failed",
                    "source_kind": "FilesystemSource",
                    "error": "raw ORT payload",
                    "error_reason": "load failed [red]boom[/red]",
                }
            ]
        }
    }
    sysmod._output_ep_text(eps)
    out = recording_console.export_text()
    # Escaped → the bracketed span survives verbatim (not consumed as style).
    assert "[red]boom[/red]" in out


def test_device_error_markup_rendered_literally(
    recording_console: Console,
) -> None:
    devices: list[dict[str, Any]] = [
        {
            "priority": 1,
            "type": "NPU",
            "name": "Test NPU",
            "details": {"error": "init failed [red]boom[/red]"},
        }
    ]
    sysmod._output_device_text(devices)
    out = recording_console.export_text()
    assert "[red]boom[/red]" in out


def test_ep_device_facts_markup_rendered_literally(
    recording_console: Console,
) -> None:
    # Per-source device facts come from ORT plugin metadata; a bracketed
    # capability/memory string must render literally, not be consumed as
    # Rich markup.
    lines = sysmod._format_devices_from_handles(
        [{"device_type": "NPU", "facts": ["Capabilities: [bold]FP8[/bold]"]}]
    )
    for line in lines:
        recording_console.print(line)
    out = recording_console.export_text()
    assert "[bold]FP8[/bold]" in out


def test_device_name_markup_rendered_literally(
    recording_console: Console,
) -> None:
    # A hardware name with brackets must not be interpreted as markup.
    devices: list[dict[str, Any]] = [
        {"priority": 1, "type": "NPU", "name": "[bold]Weird[/bold] NPU", "details": {}}
    ]
    sysmod._output_device_text(devices)
    out = recording_console.export_text()
    assert "[bold]Weird[/bold]" in out
