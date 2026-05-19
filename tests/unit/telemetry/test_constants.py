# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


def test_instrumentation_key_is_empty_in_source():
    """In source / dev installs the iKey is empty. Only official builds
    inject it. This test guards against an iKey being accidentally committed.
    """
    from winml.modelkit.telemetry import constants

    assert constants.INSTRUMENTATION_KEY == ""


def test_telemetry_enabled_default_is_true():
    """Guard the master switch default. Set ``TELEMETRY_ENABLED = False``
    in source only as a deliberate kill-switch; this test prevents it
    being flipped off and forgotten."""
    from winml.modelkit.telemetry import constants

    assert constants.TELEMETRY_ENABLED is True
