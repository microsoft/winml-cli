# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DXGI baselines and endpoints must use the same counter backend."""

from unittest.mock import Mock, patch

import pytest

from winml.modelkit.session.monitor.memory_tracker import MIB, ProcessMemoryTracker


@pytest.mark.parametrize("pdh_value", [2 * MIB, None])
@pytest.mark.parametrize("dxgi_value", [MIB, 0, None])
def test_dxgi_baseline_is_independent_of_pdh(pdh_value, dxgi_value):
    reader = Mock()
    reader.sample.return_value = {
        name: {"value_bytes": dxgi_value, "hresult": "0x80004005" if dxgi_value is None else "0x0"}
        for name in ("local", "nonlocal")
    }
    tracker = ProcessMemoryTracker(backend="dxgi")
    with (
        patch("winml.modelkit.session.monitor._dxgi_memory.DxgiMemoryReader", return_value=reader),
        patch(
            "winml.modelkit.session.monitor.memory_tracker.sample_vram",
            return_value={
                name: {
                    "value_bytes": pdh_value,
                    "reason": "unavailable" if pdh_value is None else None,
                }
                for name in ("local", "shared")
            },
        ) as pdh,
        patch("winml.modelkit.session.monitor.memory_tracker.MemoryTracker") as preparation,
    ):
        tracker.checkpoint("baseline")
        startup_rss = tracker.checkpoints["baseline"]["rss"]
        try:
            tracker.bind_before_load("0x00000000_0x00000001", device="gpu")
            tracker.checkpoint("after_compile")
            tracker.checkpoint("after_inference")
            report = tracker.to_dict()
            for key in ("vram_local", "vram_shared"):
                assert report[f"{key}_baseline_mb"] == (
                    dxgi_value / MIB if dxgi_value is not None else None
                )
                for phase in ("model_load", "total"):
                    assert report[f"{key}_{phase}_delta_mb"] == (
                        0 if dxgi_value is not None else None
                    )
                if dxgi_value is None:
                    assert tracker.availability["baseline"][key] == "0x80004005"
                else:
                    assert key not in tracker.availability["baseline"]
            assert tracker.checkpoints["baseline"]["rss"] == startup_rss
            pdh.assert_not_called()
            preparation.assert_not_called()
        finally:
            tracker.close()
        reader.close.assert_called_once()


def test_unbound_dxgi_baseline_does_not_fall_back_to_pdh():
    tracker = ProcessMemoryTracker(backend="dxgi")
    with patch("winml.modelkit.session.monitor.memory_tracker.sample_vram") as pdh:
        tracker.checkpoint("baseline")
        tracker.bind_before_load(None, device="gpu")
        assert tracker.checkpoints["baseline"]["vram_local"] is None
        assert tracker.availability["baseline"]["vram_local"] == "DXGI adapter not bound"
        pdh.assert_not_called()
    tracker.close()
