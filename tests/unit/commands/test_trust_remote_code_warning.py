# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the ``trust_remote_code`` security warning.

Regression for issue #516: passing ``--trust-remote-code`` (CLI) or
``trust_remote_code=True`` (API) must emit a stderr warning before any
remote download or custom-code execution, so a user pasting a command from
a blog post is informed that arbitrary Python may run.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _reset_trust_remote_code_flag():
    """Per-process dedup flag must be reset between tests."""
    import winml.modelkit.utils.cli as cli_utils

    cli_utils._trust_remote_code_warned = False
    yield
    cli_utils._trust_remote_code_warned = False


class TestWarnTrustRemoteCodeHelper:
    """``warn_trust_remote_code`` writes to stderr, deduped per-process."""

    def test_emits_to_stderr(self, capsys) -> None:
        from winml.modelkit.utils.cli import warn_trust_remote_code

        warn_trust_remote_code()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "WARNING" in captured.err
        assert "trust_remote_code" in captured.err

    def test_dedupes_per_process(self, capsys) -> None:
        """Repeated calls in the same process emit only once."""
        from winml.modelkit.utils.cli import warn_trust_remote_code

        warn_trust_remote_code()
        warn_trust_remote_code()
        captured = capsys.readouterr()
        assert captured.err.count("WARNING") == 1


class TestCliTrustRemoteCodeWarning:
    """The ``--trust-remote-code`` click option fires the warning before the
    command body runs. Covers ``build``, ``config``, ``eval``."""

    @pytest.mark.parametrize(
        ("command", "extra_args"),
        [
            ("build", ["-m", "fake/nonexistent"]),
            ("config", ["-m", "fake/nonexistent", "--task", "fake"]),
            ("eval", []),
        ],
    )
    def test_cli_flag_emits_warning(self, command: str, extra_args: list[str]) -> None:
        from winml.modelkit.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [command, "--trust-remote-code", *extra_args],
            obj={"debug": False},
        )
        # The command may exit non-zero (missing args, missing model, etc.)
        # — we only care that the warning was emitted to stderr first.
        combined = (result.stderr or "") + (result.output or "")
        assert "WARNING" in combined
        assert "trust_remote_code" in combined

    @pytest.mark.parametrize(
        ("command", "extra_args"),
        [
            ("build", ["-m", "fake/nonexistent"]),
            ("config", ["-m", "fake/nonexistent", "--task", "fake"]),
        ],
    )
    def test_cli_without_flag_no_warning(self, command: str, extra_args: list[str]) -> None:
        from winml.modelkit.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [command, *extra_args], obj={"debug": False})
        combined = (result.stderr or "") + (result.output or "")
        assert "trust_remote_code is enabled" not in combined


class TestApiTrustRemoteCodeWarning:
    """The public API entry points emit the warning when ``trust_remote_code=True``."""

    def test_load_hf_model_emits_warning(self, monkeypatch, capsys) -> None:
        from winml.modelkit.loader import hf as hf_loader

        # Stub AutoConfig.from_pretrained so we don't hit the network.
        def _fail(*_args, **_kwargs):
            raise RuntimeError("stop after warning")

        monkeypatch.setattr(hf_loader.AutoConfig, "from_pretrained", _fail)

        with pytest.raises(RuntimeError, match="stop after warning"):
            hf_loader.load_hf_model("microsoft/resnet-50", trust_remote_code=True)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "trust_remote_code" in captured.err

    def test_load_hf_model_no_warning_when_flag_false(self, monkeypatch, capsys) -> None:
        from winml.modelkit.loader import hf as hf_loader

        def _fail(*_args, **_kwargs):
            raise RuntimeError("stop before network")

        monkeypatch.setattr(hf_loader.AutoConfig, "from_pretrained", _fail)

        with pytest.raises(RuntimeError):
            hf_loader.load_hf_model("microsoft/resnet-50", trust_remote_code=False)

        captured = capsys.readouterr()
        assert "trust_remote_code is enabled" not in captured.err
