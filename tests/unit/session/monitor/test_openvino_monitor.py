# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the OpenVINOMonitor stub.

The shipping OpenVINO EP wheels don't implement the CSV-dump surface the
original monitor was written against, so ``OpenVINOMonitor`` was reduced
to a stub: :meth:`is_available` returns ``False`` unconditionally, and
the CLI refuses ``--op-tracing --ep openvino`` at ``commands.perf``
level. These tests pin what remains — construction validation and
provider-option owner-enforcement of PERF_COUNT.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Availability — always False (Item 3 of the cleanup pass)
# ---------------------------------------------------------------------------


def test_is_available_returns_false_unconditionally():
    """``is_available`` is ``False`` regardless of environment.

    The CLI refuses ``--op-tracing --ep openvino`` at ``commands.perf``,
    so this monitor is not a working per-op tracer on any system.
    """
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    assert OpenVINOMonitor.is_available() is False


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_device_raises():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with pytest.raises(ValueError, match="device"):
        OpenVINOMonitor(device="TPU")


def test_level_only_basic():
    """OpenVINOMonitor raises ValueError for any level other than 'basic'."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with pytest.raises(ValueError, match="level"):
        OpenVINOMonitor(level="detail")


def test_ctor_defaults():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor()
    assert m._level == "basic"
    assert m._device == "AUTO"


def test_requires_session_teardown_is_false():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    assert OpenVINOMonitor.requires_session_teardown is False


def test_ep_name_is_openvino():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    assert OpenVINOMonitor.ep_name == "openvino"


# ---------------------------------------------------------------------------
# Provider options — owner-enforced PERF_COUNT (C-3)
# ---------------------------------------------------------------------------


def test_get_provider_options_merges_perf_count():
    """No caller load_config → inserts AUTO.PERF_COUNT=YES."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="AUTO")
    opts = m.get_provider_options()
    assert "load_config" in opts
    lc = json.loads(opts["load_config"])
    assert lc == {"AUTO": {"PERF_COUNT": "YES"}}


def test_get_provider_options_preserves_caller_keys():
    """Caller's CPU.NUM_STREAMS=4 survives alongside owner-injected AUTO.PERF_COUNT=YES."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    caller_lc = json.dumps({"CPU": {"NUM_STREAMS": 4}})
    m = OpenVINOMonitor(device="AUTO", extra_provider_options={"load_config": caller_lc})
    opts = m.get_provider_options()
    lc = json.loads(opts["load_config"])
    assert lc["CPU"]["NUM_STREAMS"] == 4
    assert lc["AUTO"]["PERF_COUNT"] == "YES"


def test_get_provider_options_owner_enforced():
    """C-3: caller PERF_COUNT=NO is overridden to YES; other keys in device cfg are preserved."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    caller_lc = json.dumps({"AUTO": {"PERF_COUNT": "NO", "OTHER": "val"}})
    m = OpenVINOMonitor(device="AUTO", extra_provider_options={"load_config": caller_lc})
    opts = m.get_provider_options()
    lc = json.loads(opts["load_config"])
    assert lc["AUTO"]["PERF_COUNT"] == "YES"
    assert lc["AUTO"]["OTHER"] == "val"


def test_get_provider_options_idempotent():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="CPU")
    assert m.get_provider_options() == m.get_provider_options()


def test_get_provider_options_device_cpu():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="CPU")
    lc = json.loads(m.get_provider_options()["load_config"])
    assert lc == {"CPU": {"PERF_COUNT": "YES"}}
