# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Per-sample retention + derived stats on OperatorMetrics."""

import pytest

from winml.modelkit.session.monitor.op_metrics import OperatorMetrics


def test_samples_us_default_empty():
    op = OperatorMetrics(name="Conv2d", op_path="/layer1/conv/Conv")
    assert op.samples_us == []
    assert op.sample_count == 0


def test_avg_us_from_samples():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[100.0, 200.0, 300.0])
    assert op.avg_us == pytest.approx(200.0)


def test_total_us_from_samples():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[10.0, 20.0, 30.0])
    assert op.total_us == pytest.approx(60.0)


def test_p90_us_inclusive_method():
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/x",
        samples_us=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    )
    # Inclusive p90 of 1..10 is 9.1 (statistics.quantiles n=10 method='inclusive' index 8)
    assert op.p90_us == pytest.approx(9.1, abs=0.01)


def test_p90_single_sample():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[42.0])
    assert op.p90_us == pytest.approx(42.0)


def test_p90_empty_samples_returns_zero():
    op = OperatorMetrics(name="Conv2d", op_path="/x", samples_us=[])
    assert op.p90_us == 0.0


def test_duration_us_back_compat_when_samples_present():
    """duration_us should mirror avg_us when samples_us is populated, for back-compat."""
    op = OperatorMetrics(
        name="Conv2d",
        op_path="/x",
        duration_us=200.0,  # explicitly set, mirrors avg
        samples_us=[100.0, 200.0, 300.0],
    )
    # to_dict still serializes duration_us; samples_us is additive
    d = op.to_dict()
    assert d["duration_us"] == 200.0
    assert d["samples_us"] == [100.0, 200.0, 300.0]
