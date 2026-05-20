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

from tests._helpers import run_inspect as _run


# ===========================================================================
# CLI surface
# ===========================================================================


class TestInspectCliSurface:
    """Help text, no-args errors, and format validation."""

    @pytest.fixture(scope="class")
    def help_output(self) -> str:
        """Invoke --help once and share the output across all parametrized cases."""
        return _run("--help").output

    def test_no_args_exits_usage_error(self) -> None:
        """Invoked with no arguments inspect must exit 2 with a UsageError."""
        result = _run()
        assert result.exit_code == 2
        assert "At least one of" in result.output

    def test_help_exits_zero(self) -> None:
        """--help must exit 0."""
        assert _run("--help").exit_code == 0

    @pytest.mark.parametrize(
        "flags",
        [
            ("-m", "--model"),
            ("-f", "--format"),
            ("--model-type",),
            ("--model-class",),
            ("--list-tasks",),
            ("-v", "--verbose"),
            ("-H", "--hierarchy"),
        ],
    )
    def test_help_documents_flag(self, help_output: str, flags: tuple[str, ...]) -> None:
        """Every documented flag appears in --help output."""
        for flag in flags:
            assert flag in help_output

    def test_invalid_format(self) -> None:
        """An unrecognised --format value must exit non-zero and name the bad value."""
        result = _run("--model-type", "bert", "--format", "xml")
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "xml" in output_lower or "choice" in output_lower or "invalid" in output_lower


# ===========================================================================
# --list-tasks
# ===========================================================================


class TestInspectListTasks:
    """--list-tasks must exit 0 and print one task-name per line."""

    def test_list_tasks_exits_zero(self) -> None:
        """--list-tasks should not require a model argument and must exit 0."""
        result = _run("--list-tasks")
        assert result.exit_code == 0, f"--list-tasks exited {result.exit_code}:\n{result.output}"

    def test_list_tasks_output_is_nonempty(self) -> None:
        """--list-tasks must print at least one task."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert len(lines) > 0, "Expected at least one task line"

    def test_list_tasks_lines_match_task_name_pattern(self) -> None:
        """Every line must be a valid HF task-name (lowercase, hyphens only)."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        for line in result.output.splitlines():
            task = line.strip()
            if task:
                assert re.match(r"^[a-z][a-z0-9-]*$", task), (
                    f"Line does not match task-name pattern: {task!r}"
                )

    def test_list_tasks_includes_known_tasks(self) -> None:
        """Output must include ModelKit-registered tasks."""
        result = _run("--list-tasks")
        assert result.exit_code == 0
        tasks = {line.strip() for line in result.output.splitlines() if line.strip()}
        assert "feature-extraction" in tasks
        assert "mask-generation" in tasks

    @pytest.mark.parametrize("extra_args", [[], ["--model-type", "bert"]])
    def test_list_tasks_is_sorted(self, extra_args: list[str]) -> None:
        """Output lines must be in ascending lexicographic order."""
        result = _run("--list-tasks", *extra_args)
        assert result.exit_code == 0
        lines = [line.strip() for line in result.output.splitlines() if line.strip()]
        assert lines == sorted(lines), "Task list is not sorted"
