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
