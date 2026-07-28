# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Constants pin the mockup-approved chart geometry."""

from winml.modelkit.commands import _live_chart


def test_chart_window_seconds_is_fifteen():
    assert _live_chart._CHART_WINDOW_SECONDS == 15.0


def test_default_chart_width_is_one_hundred_twenty():
    import inspect

    sig = inspect.signature(_live_chart.LiveMonitorDisplay.__init__)
    assert sig.parameters["chart_width"].default == 120
