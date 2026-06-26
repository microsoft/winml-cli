# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the hand-coded task registry and its derived views.

``_TASK_REGISTRY`` (and the ``KNOWN_TASKS`` / ``TASK_ABBREV`` views derived
from it) is hand-coded so that ``winml inspect --list-tasks`` does not need to
import ``optimum.exporters`` (which transitively imports ``transformers`` and
adds ~10 s of startup latency).

These tests guard against drift:
  * KNOWN_TASKS must be a superset of optimum's TasksManager task set, so
    no canonical task disappears from ``--list-tasks`` when optimum adds one.
  * KNOWN_TASKS must include every task registered in our own
    HF_TASK_DEFAULTS / HF_MODEL_CLASS_MAPPING.
  * KNOWN_TASKS and TASK_ABBREV must stay derived from the single
    ``_TASK_REGISTRY`` so the two views cannot desync (the root cause of #724).
  * Every task wired into ``inference.tasks`` must be a known task, so a newly
    added canonical task can never be silently rejected by ``validate_task``.
"""

from __future__ import annotations

import pytest

from winml.modelkit.loader import KNOWN_TASKS
from winml.modelkit.loader.task import (
    _TASK_ALIAS_ABBREV,
    _TASK_REGISTRY,
    TASK_ABBREV,
)


# Tasks #724 confirmed are wired into the codebase (inference TaskInputSpec,
# composite-model registration, or e2e testset model tag) and therefore must be
# accepted by validate_task and advertised by ``--list-tasks``.
WIRED_TASKS = (
    "video-classification",
    "keypoint-matching",
    "table-question-answering",
    "zero-shot-audio-classification",
    "any-to-any",
)

# Entries audited as stale in #724 — they only ever existed in the old
# TASK_ABBREV table (no inference spec, model registration, test, or doc) and
# were dropped. Locked here so they are not silently re-added.
DROPPED_TASKS = (
    "instance-segmentation",
    "universal-segmentation",
    "masked-image-modeling",
    "text-encoding",
    "audio-tokenization",
    "multimodal-lm",
    "backbone",
    "time-series-prediction",
)


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
            "Add them to _TASK_REGISTRY in src/winml/modelkit/loader/task.py."
        )

    def test_covers_hf_model_class_mapping(self) -> None:
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        registered = {task for _, task in HF_MODEL_CLASS_MAPPING if task is not None}
        missing = registered - KNOWN_TASKS
        assert not missing, (
            f"HF_MODEL_CLASS_MAPPING uses tasks not in KNOWN_TASKS: {sorted(missing)}. "
            "Add them to _TASK_REGISTRY in src/winml/modelkit/loader/task.py."
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
            "Add them to _TASK_REGISTRY in src/winml/modelkit/loader/task.py."
        )


class TestTaskRegistrySingleSourceOfTruth:
    """KNOWN_TASKS and TASK_ABBREV must derive from one registry (#724).

    The root cause of #724 was two hand-maintained tables (``TASK_ABBREV`` and
    ``KNOWN_TASKS``) that drifted apart. They now both derive from
    ``_TASK_REGISTRY``; these tests lock that invariant.
    """

    def test_known_tasks_equals_registry_keys(self) -> None:
        assert frozenset(_TASK_REGISTRY) == KNOWN_TASKS

    def test_every_canonical_abbrev_key_is_known(self) -> None:
        """Every non-alias key in TASK_ABBREV must be a known task."""
        non_alias = {task for task in TASK_ABBREV if task not in _TASK_ALIAS_ABBREV}
        missing = non_alias - KNOWN_TASKS
        assert not missing, f"TASK_ABBREV canonical keys not in KNOWN_TASKS: {sorted(missing)}"

    def test_aliases_excluded_from_known_tasks(self) -> None:
        """Collapsed aliases keep a cache-key abbrev but are not canonical tasks."""
        overlap = set(_TASK_ALIAS_ABBREV) & KNOWN_TASKS
        assert not overlap, f"Aliases must not appear in KNOWN_TASKS: {sorted(overlap)}"

    def test_abbreviations_are_unique(self) -> None:
        """serve/app.py inverts TASK_ABBREV to {abbrev: task}; collisions lose entries."""
        values = list(TASK_ABBREV.values())
        assert len(values) == len(set(values)), "Duplicate abbreviations in TASK_ABBREV"


class TestWiredTasksAccepted:
    """Regression for #724: the wired tasks must pass validation, not be rejected."""

    @pytest.mark.parametrize("task", WIRED_TASKS)
    def test_wired_task_in_known_tasks(self, task: str) -> None:
        assert task in KNOWN_TASKS

    @pytest.mark.parametrize("task", WIRED_TASKS)
    def test_resolver_validate_task_accepts(self, task: str) -> None:
        from winml.modelkit.inspect.resolver import validate_task

        validate_task(task)  # must not raise

    @pytest.mark.parametrize("task", WIRED_TASKS)
    def test_click_callback_accepts(self, task: str) -> None:
        from winml.modelkit.commands.inspect import _validate_task

        assert _validate_task(None, None, task) == task  # must not raise


class TestStaleTasksDropped:
    """Audit lock for #724: the 8 stale names stay out of every task view."""

    @pytest.mark.parametrize("task", DROPPED_TASKS)
    def test_dropped_from_known_tasks(self, task: str) -> None:
        assert task not in KNOWN_TASKS

    @pytest.mark.parametrize("task", DROPPED_TASKS)
    def test_dropped_from_task_abbrev(self, task: str) -> None:
        assert task not in TASK_ABBREV


class TestInferenceTasksAreKnown:
    """Every inference task must be reachable from the loader task registry.

    This is the cross-source guard #724 was missing: a ``TaskInputSpec`` for a
    genuinely new canonical task (as ``video-classification`` was) added to
    ``inference.tasks`` but not to ``_TASK_REGISTRY`` would be silently rejected
    by ``validate_task``.
    """

    # Pipeline-sugar aliases that live only in inference.tasks (they share another
    # task's input schema; there is no separate canonical task). A NEW canonical
    # task belongs in _TASK_REGISTRY (loader/task.py), NOT in this allowlist.
    INFERENCE_ONLY_ALIASES = frozenset(
        {"sentiment-analysis", "ner", "vqa", "text-to-speech"}
    )

    def test_inference_tasks_are_known_or_alias(self) -> None:
        from winml.modelkit.inference.tasks import TASK_REGISTRY as INFERENCE_TASKS

        recognized = KNOWN_TASKS | set(_TASK_ALIAS_ABBREV) | self.INFERENCE_ONLY_ALIASES
        missing = set(INFERENCE_TASKS) - recognized
        assert not missing, (
            f"inference.tasks defines tasks unknown to the loader registry: {sorted(missing)}. "
            "If these are real canonical tasks, add them to _TASK_REGISTRY in "
            "src/winml/modelkit/loader/task.py (do not extend INFERENCE_ONLY_ALIASES)."
        )
