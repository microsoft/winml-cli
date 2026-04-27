# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared fixtures for the telemetry integration test package.

Mirrors the autouse singleton reset from
``tests/unit/telemetry/conftest.py`` so the same isolation guarantees
hold for in-process CLI invocations.
"""

import pytest

from winml.modelkit.telemetry import telemetry as telemetry_mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Best-effort cleanup of any leaked singleton from prior tests.
            pass
    telemetry_mod._INSTANCE = None
    yield
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Same rationale as above; teardown must always reach the
            # _INSTANCE = None reset below.
            pass
    telemetry_mod._INSTANCE = None
