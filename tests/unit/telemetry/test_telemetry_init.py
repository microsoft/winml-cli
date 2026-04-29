# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from winml.modelkit.telemetry import Telemetry
from winml.modelkit.telemetry import consent as consent_mod


# `_reset_telemetry_singleton` (autouse) comes from tests/conftest.py.
# `isolated_config` and `clean_env` come from
# tests/unit/telemetry/conftest.py.


def test_empty_ikey_makes_telemetry_disabled(clean_env, isolated_config, monkeypatch):
    """Regression: in dev installs / source checkouts, INSTRUMENTATION_KEY
    is empty and telemetry must stay off even if consent says enabled."""
    consent_mod._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    t = Telemetry.get_or_init()
    assert t.disabled is True
    # No logger was even constructed.
    assert t._logger is None


def test_consent_disabled_makes_telemetry_disabled(clean_env, isolated_config, monkeypatch):
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "o:test-key")
    consent_mod._write_stored_consent("disabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    t = Telemetry.get_or_init()
    assert t.disabled is True
    assert t._logger is None


def test_singleton_is_cached(clean_env, isolated_config, monkeypatch):
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    t1 = Telemetry.get_or_init()
    t2 = Telemetry.get_or_init()
    assert t1 is t2
