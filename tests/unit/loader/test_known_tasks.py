# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the hand-coded KNOWN_TASKS constant.

KNOWN_TASKS is hand-coded so that ``winml inspect --list-tasks`` does not
need to import ``optimum.exporters`` (which transitively imports
``transformers`` and adds ~10 s of startup latency).

These tests guard against drift:
  * KNOWN_TASKS must be a superset of optimum's TasksManager task set, so
    no canonical task disappears from ``--list-tasks`` when optimum adds one.
  * KNOWN_TASKS must include every task registered in our own
    HF_TASK_DEFAULTS / HF_MODEL_CLASS_MAPPING.
"""

from __future__ import annotations

from winml.modelkit.loader import KNOWN_TASKS


class TestKnownTasksShape:
    """KNOWN_TASKS structural invariants."""

    def test_is_nonempty(self) -> None:
        assert len(KNOWN_TASKS) > 0

    def test_is_frozenset(self) -> None:
        assert isinstance(KNOWN_TASKS, frozenset)

    def test_entries_are_canonical_task_names(self) -> None:
        """Each entry is lowercase, hyphenated, no whitespace."""
        import re

        pattern = re.compile(r"^[a-z][a-z0-9-]*$")
        bad = [t for t in KNOWN_TASKS if not pattern.match(t)]
        assert not bad, f"Non-canonical task names in KNOWN_TASKS: {bad}"


class TestKnownTasksSync:
    """KNOWN_TASKS must remain in sync with authoritative task sources."""

    def test_covers_hf_task_defaults(self) -> None:
        from winml.modelkit.loader.task import HF_TASK_DEFAULTS

        missing = set(HF_TASK_DEFAULTS) - KNOWN_TASKS
        assert not missing, (
            f"HF_TASK_DEFAULTS contains tasks not in KNOWN_TASKS: {sorted(missing)}. "
            "Add them to KNOWN_TASKS in src/winml/modelkit/loader/task.py."
        )

    def test_covers_hf_model_class_mapping(self) -> None:
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        registered = {task for _, task in HF_MODEL_CLASS_MAPPING if task is not None}
        missing = registered - KNOWN_TASKS
        assert not missing, (
            f"HF_MODEL_CLASS_MAPPING uses tasks not in KNOWN_TASKS: {sorted(missing)}. "
            "Add them to KNOWN_TASKS in src/winml/modelkit/loader/task.py."
        )

    def test_covers_optimum_tasks(self) -> None:
        """KNOWN_TASKS must be a superset of optimum's TasksManager task set.

        If optimum upgrades and adds a new canonical task, this test fails
        with a clear message telling the maintainer to update KNOWN_TASKS.
        Importing optimum here is fine — this test is not on the
        ``--list-tasks`` hot path.
        """
        from optimum.exporters.tasks import TasksManager

        optimum_tasks = set(TasksManager.get_all_tasks())
        missing = optimum_tasks - KNOWN_TASKS
        assert not missing, (
            f"optimum exposes tasks not in KNOWN_TASKS: {sorted(missing)}. "
            "Add them to KNOWN_TASKS in src/winml/modelkit/loader/task.py."
        )
