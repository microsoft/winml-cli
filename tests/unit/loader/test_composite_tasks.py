# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the hand-coded ``COMPOSITE_TASKS`` constant.

``COMPOSITE_TASKS`` (in ``loader/task.py``) is hand-coded so ``--task`` validation
(``commands/inspect.py::_validate_task``) can accept composite pipeline tasks without
importing the composite registry — which pulls in ``transformers`` and would defeat
inspect's fast startup. Because it is hand-coded, these tests lock it in sync with the
live registry so a new ``register_composite_model`` task cannot silently drift out of
the accepted ``--task`` set. Importing the registry here is fine — this is not on the
``--list-tasks`` hot path (mirrors ``test_known_tasks.py::test_covers_optimum_tasks``).
"""

from __future__ import annotations

from winml.modelkit.loader.task import COMPOSITE_TASKS, KNOWN_TASKS


class TestCompositeTasksShape:
    def test_is_nonempty_frozenset(self) -> None:
        assert isinstance(COMPOSITE_TASKS, frozenset)
        assert len(COMPOSITE_TASKS) > 0

    def test_entries_are_canonical_task_names(self) -> None:
        import re

        pattern = re.compile(r"^[a-z][a-z0-9-]*$")
        bad = [t for t in COMPOSITE_TASKS if not pattern.match(t)]
        assert not bad, f"Non-canonical task names in COMPOSITE_TASKS: {bad}"


class TestCompositeTasksSync:
    def test_matches_composite_registry(self) -> None:
        """COMPOSITE_TASKS must equal the distinct tasks in the composite registry.

        Importing ``winml.modelkit.models.hf`` populates ``COMPOSITE_MODEL_REGISTRY``
        as an import side effect (the same trigger the resolver uses).
        """
        import winml.modelkit.models.hf  # noqa: F401  # populates the registry
        from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

        registry_tasks = {task for (_mt, task) in COMPOSITE_MODEL_REGISTRY}
        assert registry_tasks == COMPOSITE_TASKS, (
            "COMPOSITE_TASKS is out of sync with the composite registry. "
            f"Registry-only: {sorted(registry_tasks - COMPOSITE_TASKS)}; "
            f"stale in COMPOSITE_TASKS: {sorted(COMPOSITE_TASKS - registry_tasks)}. "
            "Update COMPOSITE_TASKS in src/winml/modelkit/loader/task.py."
        )

    def test_sam_mask_generation_has_published_graph_roles(self) -> None:
        from winml.modelkit.loader.resolution import resolve_composite

        assert resolve_composite("sam", "mask-generation") == {
            "image-encoder": "image-feature-extraction",
            "prompt-decoder": "mask-generation",
        }


class TestCompositeTasksVsKnownTasks:
    def test_summarization_translation_excluded_from_known_tasks(self) -> None:
        """summarization/translation live only in COMPOSITE_TASKS, not KNOWN_TASKS
        (kept out by the cache-key/alias rules). They are the only composite tasks the
        union actually *adds* to the accepted --task set."""
        added_by_composite = COMPOSITE_TASKS - KNOWN_TASKS
        assert added_by_composite == {"summarization", "translation"}
