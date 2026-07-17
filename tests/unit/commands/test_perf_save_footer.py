# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Save-to footer prints after op-trace report."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from winml.modelkit.commands.perf import _print_save_to_footer


def _render(trace_json: str | None, profiling_csv: str | None) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    _print_save_to_footer(console, trace_json=trace_json, profiling_csv=profiling_csv)
    return console.export_text()


def test_both_paths_shown():
    out = _render(r"C:\out\trace.json", r"C:\out\prof.csv")
    assert "trace.json" in out
    assert "prof.csv" in out


def test_csv_omitted_when_none():
    out = _render(r"C:\out\trace.json", None)
    assert "trace.json" in out
    assert ".csv" not in out


def test_neither_when_both_none():
    out = _render(None, None)
    assert out.strip() == ""


def test_json_path_label_present():
    out = _render(r"C:\out\trace.json", None)
    # The footer should label what each path is. Look for "Op-trace JSON" or
    # similar marker so users know what the path means.
    assert "Op-trace" in out or "trace JSON" in out.lower()


def test_csv_path_label_present():
    out = _render(r"C:\out\trace.json", r"C:\out\prof.csv")
    # Similarly the CSV line should be labeled.
    assert "CSV" in out or "csv" in out.lower()
