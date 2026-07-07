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

    @staticmethod
    def _task_lines(output: str) -> list[str]:
        return [line.strip() for line in output.splitlines() if line.strip()]

    def test_list_tasks_model_type_is_strict_subset(self) -> None:
        """--list-tasks --model-type clip must be a strict subset of the full taxonomy."""
        full = set(self._task_lines(_run("--list-tasks").output))
        result = _run("--list-tasks", "--model-type", "clip")
        assert result.exit_code == 0
        clip_tasks = self._task_lines(result.output)
        assert clip_tasks, "Expected clip to have at least one task"
        assert clip_tasks == sorted(clip_tasks)
        clip_set = set(clip_tasks)
        assert clip_set < full, "Model-specific tasks must be a strict subset of all tasks"
        # Optimum-exportable and composite pipeline tasks both surface.
        assert "image-classification" in clip_set
        assert "zero-shot-image-classification" in clip_set

    def test_list_tasks_model_type_is_model_specific(self) -> None:
        """--list-tasks --model-type bert lists NLP tasks and excludes vision-only ones."""
        result = _run("--list-tasks", "--model-type", "bert")
        assert result.exit_code == 0
        bert_set = set(self._task_lines(result.output))
        assert {"fill-mask", "feature-extraction"} <= bert_set
        assert "image-classification" not in bert_set

    def test_list_tasks_excludes_with_past_flavors(self) -> None:
        """Optimum's `-with-past` export flavors are not user-facing tasks and must not leak."""
        result = _run("--list-tasks", "--model-type", "t5")
        assert result.exit_code == 0
        t5_tasks = self._task_lines(result.output)
        assert not any(t.endswith("-with-past") for t in t5_tasks), t5_tasks
        # The real tasks (exportable + composite) still surface.
        assert {"summarization", "translation", "text2text-generation"} <= set(t5_tasks)

    def test_list_tasks_includes_modelkit_registered_tasks(self) -> None:
        """ModelKit's register_onnx_overwrite tasks must surface (needs models.hf import).

        sam -> mask-generation is registered by winml.modelkit.models.hf, not by
        stock Optimum. If the listing only triggers optimum's own registry it is
        silently dropped, so this pins the models.hf import trigger.
        """
        result = _run("--list-tasks", "--model-type", "sam")
        assert result.exit_code == 0
        sam_set = set(self._task_lines(result.output))
        assert "mask-generation" in sam_set, sam_set

    def test_list_tasks_resolves_underscored_model_type(self) -> None:
        """An underscored model_type (depth_anything) must resolve, not return empty."""
        result = _run("--list-tasks", "--model-type", "depth_anything")
        assert result.exit_code == 0
        assert "depth-estimation" in self._task_lines(result.output)


# ===========================================================================
# Progress feedback (banner + spinner) — fixes #543's 14s silence
# ===========================================================================


class TestInspectProgressFeedback:
    """Banner must print to stderr before heavy work; stdout must stay clean."""

    def test_banner_appears_on_stderr_for_model_type(self) -> None:
        """`winml inspect --model-type bert` prints an "Inspecting…" banner on stderr."""
        result = _run("--model-type", "bert")
        assert result.exit_code == 0
        assert "Inspecting" in result.stderr
        assert "bert" in result.stderr

    def test_json_stdout_is_clean(self) -> None:
        """--format json output on stdout must be parseable JSON with no banner."""
        import json

        result = _run("--model-type", "bert", "--format", "json")
        assert result.exit_code == 0
        # Banner must not contaminate stdout — JSON consumers parse this directly.
        assert "Inspecting" not in result.stdout
        # And stdout must in fact be valid JSON.
        json.loads(result.stdout)

    def test_quiet_suppresses_banner(self) -> None:
        """The --quiet flag must suppress the inspect banner on stderr."""
        from click.testing import CliRunner

        from winml.modelkit.cli import main

        # Use the top-level `main` group so --quiet (a group option) is parsed.
        result = CliRunner().invoke(
            main,
            ["--quiet", "inspect", "--model-type", "bert", "--format", "json"],
            obj={},
        )
        assert result.exit_code == 0
        assert "Inspecting" not in result.stderr
