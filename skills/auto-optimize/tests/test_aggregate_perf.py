# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        "aggregate", Path(__file__).parents[1] / "scripts/aggregate_perf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_samples_are_pooled_not_percentiles():
    result = load().aggregate([{"raw_samples_ms": [1, 2, 3]}, {"raw_samples_ms": [4, 100]}])
    assert result["p50_ms"] == 3
    assert result["p90_ms"] == pytest.approx(61.6)
    assert result["throughput_ips"] == pytest.approx(5000 / 110)
    assert result["sample_count"] == 5


@pytest.mark.parametrize("samples", [[], [0], [float("nan")], [True]])
def test_invalid_samples_rejected(samples):
    with pytest.raises(ValueError):
        load().aggregate([{"raw_samples_ms": samples}])
