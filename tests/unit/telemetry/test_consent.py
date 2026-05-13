# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import io
import json

import pytest

from winml.modelkit.telemetry import consent


# --- CI detection --------------------------------------------------------


@pytest.mark.parametrize(
    "var",
    [
        "CI",
        "TF_BUILD",
        "GITHUB_ACTIONS",
        "JENKINS_URL",
        "CODEBUILD_BUILD_ID",
        "BUILDKITE",
        "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
    ],
)
def test_is_ci_environment_detects_known_vars(var, clean_env, monkeypatch):
    monkeypatch.setenv(var, "1")
    assert consent._is_ci_environment() is True


def test_is_ci_environment_empty_env_returns_false(clean_env):
    assert consent._is_ci_environment() is False


# --- config-file read / write -------------------------------------------


def test_read_stored_consent_missing_file_returns_none(isolated_config):
    assert consent._read_stored_consent() is None


def test_write_stored_consent_creates_file_and_roundtrips(isolated_config):
    consent._write_stored_consent("enabled")
    assert consent._read_stored_consent() == "enabled"
    consent._write_stored_consent("disabled")
    assert consent._read_stored_consent() == "disabled"


def test_write_persists_nested_schema(isolated_config):
    consent._write_stored_consent("enabled")
    payload = json.loads(isolated_config.read_text())
    assert payload == {
        "telemetry": {"consent": "enabled", "consent_version": consent._CONSENT_VERSION}
    }


def test_read_preserves_unrelated_config_on_write(isolated_config):
    # A user may have added unrelated keys; we must not clobber them.
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps(
            {
                "unrelated": {"foo": 1},
                "telemetry": {"consent": "enabled"},
            }
        )
    )
    consent._write_stored_consent("disabled")
    payload = json.loads(isolated_config.read_text())
    assert payload["unrelated"] == {"foo": 1}
    assert payload["telemetry"]["consent"] == "disabled"


def test_read_unknown_value_returns_none(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"telemetry": {"consent": "sometimes"}}))
    assert consent._read_stored_consent() is None


def test_read_missing_telemetry_field_returns_none(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"unrelated": {"foo": 1}}))
    assert consent._read_stored_consent() is None


def test_read_malformed_json_returns_none(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text("{ not valid json")
    assert consent._read_stored_consent() is None


# --- consent_version ----------------------------------------------------


def test_read_without_version_field_is_grandfathered(isolated_config):
    # Records predating the version field must not trigger a re-prompt.
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(json.dumps({"telemetry": {"consent": "enabled"}}))
    assert consent._read_stored_consent() == "enabled"


def test_read_older_version_returns_none(isolated_config, monkeypatch):
    monkeypatch.setattr(consent, "_CONSENT_VERSION", 2)
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"telemetry": {"consent": "enabled", "consent_version": 1}})
    )
    assert consent._read_stored_consent() is None


def test_read_same_version_honored(isolated_config):
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps(
            {
                "telemetry": {
                    "consent": "disabled",
                    "consent_version": consent._CONSENT_VERSION,
                }
            }
        )
    )
    assert consent._read_stored_consent() == "disabled"


def test_read_newer_version_honored(isolated_config):
    # A config written by a newer WinML CLI (higher version) must be
    # honored, not silently re-prompted.
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps(
            {
                "telemetry": {
                    "consent": "enabled",
                    "consent_version": consent._CONSENT_VERSION + 5,
                }
            }
        )
    )
    assert consent._read_stored_consent() == "enabled"


def test_read_malformed_version_is_grandfathered(isolated_config):
    # Non-int version (string, None, bool) is ignored rather than
    # causing a re-prompt.
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"telemetry": {"consent": "enabled", "consent_version": "1"}})
    )
    assert consent._read_stored_consent() == "enabled"


def test_write_stamps_current_version(isolated_config):
    consent._write_stored_consent("enabled")
    payload = json.loads(isolated_config.read_text())
    assert payload["telemetry"]["consent_version"] == consent._CONSENT_VERSION


# --- first-run prompt ----------------------------------------------------


def test_prompt_accept_returns_enabled(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent._prompt_for_consent() == "enabled"
    captured = capsys.readouterr()
    assert "Enable telemetry?" in captured.out
    assert "[Y/n]" in captured.out


def test_prompt_decline_returns_disabled(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent._prompt_for_consent() == "disabled"


def test_prompt_empty_defaults_to_enabled(monkeypatch):
    # Default is [Y/n] - empty input = Y (accept).
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent._prompt_for_consent() == "enabled"


def test_prompt_case_insensitive_decline(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("N\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent._prompt_for_consent() == "disabled"


def test_prompt_garbage_input_defaults_to_enabled(monkeypatch):
    # Unknown input falls through to the default (accept). Only explicit
    # 'n' / 'no' declines.
    monkeypatch.setattr("sys.stdin", io.StringIO("banana\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent._prompt_for_consent() == "enabled"


# --- resolve_consent -----------------------------------------------------


def test_resolve_consent_ci_defaults_to_disabled_no_prompt(clean_env, isolated_config, monkeypatch):
    monkeypatch.setenv("CI", "1")
    assert consent.resolve_consent() == "disabled"
    # Must NOT have touched the config file (CI is per-invocation).
    assert not isolated_config.exists()


def test_resolve_consent_non_tty_defaults_to_disabled_no_prompt(
    clean_env, isolated_config, monkeypatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert consent.resolve_consent() == "disabled"
    assert not isolated_config.exists()


def test_resolve_consent_stored_decision_honored(clean_env, isolated_config, monkeypatch):
    consent._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent.resolve_consent() == "enabled"


def test_resolve_consent_first_run_empty_input_accepts_and_persists(
    clean_env, isolated_config, monkeypatch
):
    # Accept-by-default: pressing Enter enables telemetry.
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert consent.resolve_consent() == "enabled"
    assert consent._read_stored_consent() == "enabled"


def test_resolve_consent_reprompts_when_notice_version_bumped(
    clean_env, isolated_config, monkeypatch
):
    # Simulate an older stored decision + a newer notice version. The
    # user should see the prompt again and the decision should be re-stamped.
    isolated_config.parent.mkdir(parents=True, exist_ok=True)
    isolated_config.write_text(
        json.dumps({"telemetry": {"consent": "enabled", "consent_version": 1}})
    )
    monkeypatch.setattr(consent, "_CONSENT_VERSION", 2)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    assert consent.resolve_consent() == "disabled"
    payload = json.loads(isolated_config.read_text())
    assert payload["telemetry"] == {"consent": "disabled", "consent_version": 2}
