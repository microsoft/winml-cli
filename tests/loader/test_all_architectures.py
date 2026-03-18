"""Parametrized tests across ALL 166 Optimum architectures for loader/task.py.

Validates that TasksManager has a registration (OnnxConfig constructor) for
every architecture listed in the OPTIMUM_ARCHITECTURES catalog, grouped by
library: transformers, diffusers, sentence_transformers, timm.

Skipped architectures:
- sentence_transformers and timm: different library_name handling
- Architectures whose key contains ":" are duplicates with library qualifiers
  (e.g., "clip:sentence_transformers") and are filtered to the correct library group.
"""

from __future__ import annotations

import pytest

# Trigger OnnxConfig registration with TasksManager (all libraries).
import winml.modelkit.models  # noqa: F401
from tests.assets.optimum_architectures import OPTIMUM_ARCHITECTURES


# ---------------------------------------------------------------------------
# Build parametrize lists
# ---------------------------------------------------------------------------

TRANSFORMERS_ARCHS = [
    (key, info) for key, info in OPTIMUM_ARCHITECTURES.items() if info.library == "transformers"
]

DIFFUSERS_ARCHS = [
    (key, info) for key, info in OPTIMUM_ARCHITECTURES.items() if info.library == "diffusers"
]

SENTENCE_TRANSFORMERS_ARCHS = [
    (key, info)
    for key, info in OPTIMUM_ARCHITECTURES.items()
    if info.library == "sentence_transformers"
]

TIMM_ARCHS = [(key, info) for key, info in OPTIMUM_ARCHITECTURES.items() if info.library == "timm"]


# ---------------------------------------------------------------------------
# Transformers library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_key,arch_info",
    TRANSFORMERS_ARCHS,
    ids=[k for k, _ in TRANSFORMERS_ARCHS],
)
def test_tasksmanager_has_transformers_registration(arch_key, arch_info):
    """Verify TasksManager has an OnnxConfig for every transformers architecture."""
    from optimum.exporters.tasks import TasksManager

    # Handle compound keys like "clip:sentence_transformers" -> extract model_type
    model_type = arch_key.split(":")[0] if ":" in arch_key else arch_key
    first_task = arch_info.tasks[0]

    config_ctor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type=model_type,
        task=first_task,
        library_name="transformers",
    )
    assert config_ctor is not None


# ---------------------------------------------------------------------------
# Diffusers library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_key,arch_info",
    DIFFUSERS_ARCHS,
    ids=[k for k, _ in DIFFUSERS_ARCHS],
)
def test_tasksmanager_has_diffusers_registration(arch_key, arch_info):
    """Verify TasksManager has an OnnxConfig for every diffusers architecture."""
    from optimum.exporters.tasks import TasksManager

    model_type = arch_key.split(":")[0] if ":" in arch_key else arch_key
    first_task = arch_info.tasks[0]

    config_ctor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type=model_type,
        task=first_task,
        library_name="diffusers",
    )
    assert config_ctor is not None


# ---------------------------------------------------------------------------
# Sentence Transformers library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_key,arch_info",
    SENTENCE_TRANSFORMERS_ARCHS,
    ids=[k for k, _ in SENTENCE_TRANSFORMERS_ARCHS],
)
def test_tasksmanager_has_sentence_transformers_registration(arch_key, arch_info):
    """Verify TasksManager has an OnnxConfig for every sentence_transformers architecture."""
    from optimum.exporters.tasks import TasksManager

    model_type = arch_key.split(":")[0] if ":" in arch_key else arch_key
    first_task = arch_info.tasks[0]

    config_ctor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type=model_type,
        task=first_task,
        library_name="sentence_transformers",
    )
    assert config_ctor is not None


# ---------------------------------------------------------------------------
# Timm library
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch_key,arch_info",
    TIMM_ARCHS,
    ids=[k for k, _ in TIMM_ARCHS],
)
def test_tasksmanager_has_timm_registration(arch_key, arch_info):
    """Verify TasksManager has an OnnxConfig for every timm architecture."""
    from optimum.exporters.tasks import TasksManager

    model_type = arch_key.split(":")[0] if ":" in arch_key else arch_key
    first_task = arch_info.tasks[0]

    config_ctor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type=model_type,
        task=first_task,
        library_name="timm",
    )
    assert config_ctor is not None
