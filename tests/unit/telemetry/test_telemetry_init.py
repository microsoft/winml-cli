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


def test_empty_ikey_skips_consent_prompt(clean_env, isolated_config, monkeypatch):
    """Dev install: empty iKey must short-circuit before any consent prompt.

    Pins the design contract: 'INSTRUMENTATION_KEY empty (dev build)
    -> Telemetry stack never initializes; no prompt'. Setup uses an
    interactive TTY and *no* stored consent so the only thing that can
    keep ``_prompt_for_consent`` from running is the empty-iKey guard.
    """
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    prompt_calls: list[int] = []
    monkeypatch.setattr(
        consent_mod,
        "_prompt_for_consent",
        lambda: prompt_calls.append(1) or "enabled",
    )

    t = Telemetry.get_or_init()
    assert t.disabled is True
    assert prompt_calls == []


def test_init_swallows_resource_build_errors(clean_env, isolated_config, monkeypatch):
    """Telemetry init failure must never propagate to the CLI.

    If ``_build_resource`` (which touches the registry for the device ID)
    raises, ``Telemetry.get_or_init()`` must return a disabled instance
    rather than raise. Without this guard a registry permission error or
    transient OS failure would crash every CLI invocation.
    """
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "o:test-key")
    consent_mod._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def boom(self):
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(Telemetry, "_build_resource", boom)

    t = Telemetry.get_or_init()  # must NOT raise
    assert t.disabled is True
    assert t._logger is None
