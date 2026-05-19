# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI surface tests for `winml inspect`.

Covers help text, no-args UsageError, invalid option choices, and
--list-tasks — all without downloading any model weights or hitting
the network.

These tests run under the default CI filter (no special marker required).
"""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.inspect import inspect


def _run(*args: str) -> object:
    """Invoke inspect with *args and return the CliRunner Result."""
    return CliRunner().invoke(inspect, list(args), obj={})


# ===========================================================================
# CLI surface
# ===========================================================================


class TestInspectCliSurface:
    """Help text, no-args errors, and format validation."""

    def test_no_args_exits_usage_error(self):
        """Invoked with no arguments inspect must exit 2 with a UsageError."""
        result = _run()
        assert result.exit_code == 2
        assert "At least one of" in result.output

    def test_help_exits_zero(self):
        """--help must exit 0."""
        result = _run("--help")
        assert result.exit_code == 0

    def test_help_documents_model_flag(self):
        """-m / --model appears in help text."""
        result = _run("--help")
        assert "-m" in result.output
        assert "--model" in result.output

    def test_help_documents_format_flag(self):
        """-f / --format appears in help text."""
        result = _run("--help")
        assert "-f" in result.output
        assert "--format" in result.output

    def test_help_documents_model_type_flag(self):
        """--model-type appears in help text."""
        result = _run("--help")
        assert "--model-type" in result.output

    def test_help_documents_model_class_flag(self):
        """--model-class appears in help text."""
        result = _run("--help")
        assert "--model-class" in result.output

    def test_help_documents_list_tasks_flag(self):
        """--list-tasks appears in help text."""
        result = _run("--help")
        assert "--list-tasks" in result.output

    def test_help_documents_verbose_flag(self):
        """-v / --verbose appears in help text."""
        result = _run("--help")
        assert "-v" in result.output
        assert "--verbose" in result.output

    def test_help_documents_hierarchy_flag(self):
        """-H / --hierarchy appears in help text."""
        result = _run("--help")
        assert "-H" in result.output
        assert "--hierarchy" in result.output

    def test_invalid_format_exits_nonzero(self):
        """An unrecognised --format value must exit non-zero."""
        result = _run("--model-type", "bert", "--format", "xml")
        assert result.exit_code != 0

    def test_invalid_format_names_bad_choice(self):
        """Error output mentions the bad format value or 'choice'."""
        result = _run("--model-type", "bert", "--format", "xml")
        output_lower = result.output.lower()
        assert "xml" in output_lower or "choice" in output_lower or "invalid" in output_lower


# ===========================================================================
# --list-tasks
# ===========================================================================


class TestInspectListTasks:
    """--list-tasks must exit 0 and print one task-name per line."""

    def test_list_tasks_exits_zero(self):
        """--list-tasks should not require a model argument and must exit 0."""
        result = _run("--list-tasks")
        assert result.exit_code == 0, f"--list-tasks exited {result.exit_code}:\n{result.output}"

    def test_list_tasks_output_is_nonempty(self):
        """--list-tasks must print at least one task."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert len(lines) > 0, "Expected at least one task line"

    def test_list_tasks_lines_match_task_name_pattern(self):
        """Every line must be a valid HF task-name (lowercase, hyphens only)."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        for line in result.output.splitlines():
            task = line.strip()
            if task:
                assert re.match(r"^[a-z][a-z0-9-]*$", task), (
                    f"Line does not match task-name pattern: {task!r}"
                )

    def test_list_tasks_includes_known_tasks(self):
        """Output must include ModelKit-registered tasks."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        tasks = {line.strip() for line in result.output.splitlines() if line.strip()}
        assert "feature-extraction" in tasks
        assert "mask-generation" in tasks

    @pytest.mark.parametrize("extra_args", [[], ["--model-type", "bert"]])
    def test_list_tasks_is_sorted(self, extra_args: list[str]):
        """Output lines must be in ascending lexicographic order."""
        result = _run("--list-tasks", *extra_args)
        assert result.exit_code == 0
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert lines == sorted(lines), "Task list is not sorted"
