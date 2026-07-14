# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for environment variable helpers."""

from __future__ import annotations

import pytest

from winml.modelkit._env import env_flag_enabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_env_flag_enabled_truthy_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Truthy environment flag spellings are accepted consistently."""
    monkeypatch.setenv("WINMLCLI_TEST_FLAG", value)

    assert env_flag_enabled("WINMLCLI_TEST_FLAG") is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything"])
def test_env_flag_enabled_falsey_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset and non-truthy environment flag values are disabled."""
    monkeypatch.setenv("WINMLCLI_TEST_FLAG", value)

    assert env_flag_enabled("WINMLCLI_TEST_FLAG") is False


def test_env_flag_enabled_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing environment flags are disabled."""
    monkeypatch.delenv("WINMLCLI_TEST_FLAG", raising=False)

    assert env_flag_enabled("WINMLCLI_TEST_FLAG") is False
