# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Constants pin the mockup-approved chart geometry."""

from types import SimpleNamespace

from winml.modelkit.commands import _live_chart


def test_chart_window_seconds_is_fifteen():
    assert _live_chart._CHART_WINDOW_SECONDS == 15.0


def test_default_chart_width_is_one_hundred_twenty():
    import inspect

    sig = inspect.signature(_live_chart.LiveMonitorDisplay.__init__)
    assert sig.parameters["chart_width"].default == 120


def test_status_cells_wrap_by_terminal_cell_width():
    display = _live_chart.LiveMonitorDisplay(
        total_iterations=1,
        warmup=0,
        model_id="m",
        device="cpu",
    )
    display._console = SimpleNamespace(width=16)

    assert display._pack_status_cells(["宽宽宽", "宽宽宽"]) == ["  宽宽宽", "  宽宽宽"]


def test_status_cells_pad_columns_on_wide_terminal():
    display = _live_chart.LiveMonitorDisplay(
        total_iterations=1,
        warmup=0,
        model_id="m",
        device="cpu",
    )
    display._console = SimpleNamespace(width=160)

    assert display._pack_status_cells(["a", "bb", "ccc"]) == [f"  {'a':<28} | {'bb':<28} | ccc"]
