# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from winml.modelkit.loader.resolution import TaskResolution, TaskSource


def test_task_source_values():
    assert TaskSource.TASKS_MANAGER.value == "tasks-manager"
    assert TaskSource.SENTINEL_DEFAULT.value == "sentinel-default"
    assert TaskSource.USER_TASK.value == "user-task"
    assert TaskSource.WRAPPED_LIBRARY.value == "wrapped-library"


def test_task_resolution_is_frozen_with_fields():
    tr = TaskResolution(
        task="image-feature-extraction",
        optimum_task="feature-extraction",
        model_class=str,
        source=TaskSource.TASKS_MANAGER,
        composite=None,
    )
    assert tr.task == "image-feature-extraction"
    assert tr.optimum_task == "feature-extraction"
    assert tr.composite is None
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        tr.task = "x"  # type: ignore[misc]
