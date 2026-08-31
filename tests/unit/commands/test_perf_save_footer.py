# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Save-to footer prints after op-trace report."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from winml.modelkit.commands.perf import _print_save_to_footer


def _render(
    profiling_csv: str | None = None,
    profiling_json: str | None = None,
) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    _print_save_to_footer(
        console,
        profiling_csv=profiling_csv,
        profiling_json=profiling_json,
    )
    return console.export_text()


def test_csv_path_shown():
    out = _render(r"C:\out\prof.csv")
    assert "prof.csv" in out


def test_json_path_shown():
    out = _render(profiling_json=r"C:\out\onnxruntime_profile.json")
    assert "Profiling JSON:" in out
    assert "onnxruntime_profile.json" in out


def test_footer_omitted_when_csv_is_none():
    out = _render(None)
    assert out.strip() == ""


def test_csv_path_label_present():
    out = _render(r"C:\out\prof.csv")
    assert "CSV" in out or "csv" in out.lower()
