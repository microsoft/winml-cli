# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI surface and filter tests for ``winml catalog``.

Covers ``--help``, option validation, and the filtering contract for every
flag (``--model-type``, ``--task``, ``--ep``, ``--device``, ``--output``).
All test values are derived from ``hub_models.json`` at import time, so the
suite asserts *filter invariants* (subset, disjointness, case-insensitivity)
rather than pinning specific catalog entries.

These tests run under the default CI filter (no special marker required).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.catalog import catalog


if TYPE_CHECKING:
    from click.testing import Result


# ---------------------------------------------------------------------------
# Module-level catalog data — same source the CLI reads at runtime
# ---------------------------------------------------------------------------

_HUB_JSON = (
    Path(__file__).resolve().parents[2] / "src" / "winml" / "modelkit" / "data" / "hub_models.json"
)
_CATALOG: list[dict] = json.loads(_HUB_JSON.read_text(encoding="utf-8"))["models"]

_MODEL_TYPES: list[str] = [
    t
    for t, _ in Counter(m["model_type"] for m in _CATALOG if m.get("model_type")).most_common()
    if t
]
_TASKS: list[str] = [
    t for t, _ in Counter(m["task"] for m in _CATALOG if m.get("task")).most_common() if t
]

# First (model_type, task) pair that exists in the catalog.
_TYPE_TASK_PAIR: tuple[str, str] = next(
    (m["model_type"], m["task"]) for m in _CATALOG if m.get("model_type") and m.get("task")
)

# First (ep_key, model_type) pair that exists in the catalog.
_EP_MODEL_TYPE_PAIR: tuple[str, str] = next(
    (sorted((m.get("supported_eps") or {}).keys())[0], m["model_type"])
    for m in _CATALOG
    if (m.get("supported_eps") or {}) and m.get("model_type")
)


def _find_disjoint_type_task() -> tuple[str, str] | None:
    """Return a (model_type, task) pair guaranteed to return no catalog rows."""
    type_tasks: dict[str, set[str]] = defaultdict(set)
    all_tasks: set[str] = set()
    for m in _CATALOG:
        mtype, task = m.get("model_type", ""), m.get("task", "")
        if mtype and task:
            type_tasks[mtype].add(task)
            all_tasks.add(task)
    for mtype in sorted(type_tasks):
        missing = all_tasks - type_tasks[mtype]
        if missing:
            return mtype, sorted(missing)[0]
    return None


_DISJOINT_TYPE_TASK: tuple[str, str] | None = _find_disjoint_type_task()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(*args: str) -> Result:
    """Invoke ``winml catalog <args>`` via CliRunner."""
    return CliRunner().invoke(catalog, list(args), obj={})


def _run_json(out_path: Path, *args: str) -> list[dict]:
    """Run catalog with ``--output`` and return the parsed JSON list."""
    result = _run(*args, "--output", str(out_path))
    assert result.exit_code == 0, f"catalog exited {result.exit_code}\n{result.output}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def _model_ids(models: list[dict]) -> set[str]:
    return {m["model_id"] for m in models}


def _model_types(models: list[dict]) -> set[str]:
    return {m["model_type"] for m in models}


def _tasks(models: list[dict]) -> set[str]:
    return {m["task"] for m in models}


def _ep_keys(model: dict) -> set[str]:
    return set((model.get("supported_eps") or {}).keys())


def _supports_device(model: dict, device: str) -> bool:
    device_upper = device.upper()
    return any(device_upper in devs for devs in (model.get("supported_eps") or {}).values())


# ===========================================================================
# CLI surface
# ===========================================================================


class TestCatalogCliSurface:
    """Parser-level behaviour — no model or EP runtime required."""

    def test_help_exits_zero(self) -> None:
        assert _run("--help").exit_code == 0

    def test_help_documents_model_type_flag(self) -> None:
        assert "--model-type" in _run("--help").output

    def test_help_documents_task_flag(self) -> None:
        assert "--task" in _run("--help").output

    def test_help_documents_ep_flag(self) -> None:
        assert "--ep" in _run("--help").output

    def test_help_documents_device_flag(self) -> None:
        assert "--device" in _run("--help").output

    def test_help_documents_output_flag(self) -> None:
        assert "--output" in _run("--help").output

    def test_no_filter_args_exits_zero(self, tmp_path: Path) -> None:
        """``winml catalog`` with no filters returns the full catalog."""
        models = _run_json(tmp_path / "all.json")
        assert len(models) > 0, "Catalog must not be empty"

    def test_invalid_ep_choice_exits_two(self) -> None:
        result = _run("--ep", "totally_unknown_ep_xyz")
        assert result.exit_code == 2
        assert "Invalid value for '--ep'" in result.output

    def test_invalid_device_choice_exits_two(self) -> None:
        result = _run("--device", "TPU")
        assert result.exit_code == 2
        assert "Invalid value for '--device'" in result.output

    def test_short_flags_accepted(self, tmp_path: Path) -> None:
        """-t / -k short aliases are accepted by the parser."""
        model_type, task = _TYPE_TASK_PAIR
        models = _run_json(tmp_path / "out.json", "-t", model_type, "-k", task)
        assert len(models) > 0, f"Expected at least one {model_type}/{task} model"
        assert all(
            m["model_type"].lower() == model_type.lower() and m["task"].lower() == task.lower()
            for m in models
        )


# ===========================================================================
# --model-type filtering
# ===========================================================================


class TestCatalogFilterModelType:
    @pytest.mark.skipif(not _MODEL_TYPES, reason="hub_models.json has no model_type entries")
    def test_known_type_returns_only_matching_entries(self, tmp_path: Path) -> None:
        model_type = _MODEL_TYPES[0]
        models = _run_json(tmp_path / "mtype.json", "--model-type", model_type)
        assert len(models) > 0, f"Expected at least one {model_type} model"
        assert _model_types(models) == {model_type}

    @pytest.mark.skipif(not _MODEL_TYPES, reason="hub_models.json has no model_type entries")
    def test_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        model_type = _MODEL_TYPES[0]
        lower = _run_json(tmp_path / "lower.json", "--model-type", model_type.lower())
        upper = _run_json(tmp_path / "upper.json", "--model-type", model_type.upper())
        mixed = _run_json(tmp_path / "mixed.json", "--model-type", model_type.swapcase())
        assert _model_ids(lower) == _model_ids(upper) == _model_ids(mixed)

    @pytest.mark.skipif(len(_MODEL_TYPES) < 2, reason="hub_models.json needs ≥2 model_types")
    def test_single_type_is_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        model_type = _MODEL_TYPES[0]
        all_models = _run_json(tmp_path / "all.json")
        mtype_models = _run_json(tmp_path / "mtype.json", "--model-type", model_type)
        assert 0 < len(mtype_models) < len(all_models)

    def test_unknown_type_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _run("--model-type", "nonexistent_arch_xyz", "--output", str(out))
        assert result.exit_code == 0, result.output
        assert json.loads(out.read_text()) == []

    @pytest.mark.skipif(len(_MODEL_TYPES) < 2, reason="hub_models.json needs ≥2 model_types")
    def test_two_distinct_types_are_disjoint(self, tmp_path: Path) -> None:
        type_a, type_b = _MODEL_TYPES[0], _MODEL_TYPES[1]
        a = _run_json(tmp_path / "a.json", "--model-type", type_a)
        b = _run_json(tmp_path / "b.json", "--model-type", type_b)
        assert len(a) > 0, f"Expected at least one {type_a} model"
        assert len(b) > 0, f"Expected at least one {type_b} model"
        assert _model_ids(a).isdisjoint(_model_ids(b))


# ===========================================================================
# --task filtering
# ===========================================================================


class TestCatalogFilterTask:
    @pytest.mark.skipif(not _TASKS, reason="hub_models.json has no task entries")
    def test_known_task_returns_only_matching_entries(self, tmp_path: Path) -> None:
        task = _TASKS[0]
        models = _run_json(tmp_path / "task.json", "--task", task)
        assert len(models) > 0
        assert _tasks(models) == {task}

    @pytest.mark.skipif(not _TASKS, reason="hub_models.json has no task entries")
    def test_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        task = _TASKS[0]
        lower = _run_json(tmp_path / "lower.json", "--task", task.lower())
        upper = _run_json(tmp_path / "upper.json", "--task", task.upper())
        assert _model_ids(lower) == _model_ids(upper)

    @pytest.mark.skipif(len(_TASKS) < 2, reason="hub_models.json needs ≥2 tasks")
    def test_single_task_is_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        task = _TASKS[0]
        all_models = _run_json(tmp_path / "all.json")
        task_models = _run_json(tmp_path / "task.json", "--task", task)
        assert 0 < len(task_models) < len(all_models)

    def test_unknown_task_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _run("--task", "nonexistent-task-xyz", "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    @pytest.mark.skipif(len(_TASKS) < 2, reason="hub_models.json needs ≥2 tasks")
    def test_two_distinct_tasks_are_disjoint(self, tmp_path: Path) -> None:
        task_a, task_b = _TASKS[0], _TASKS[1]
        a = _run_json(tmp_path / "a.json", "--task", task_a)
        b = _run_json(tmp_path / "b.json", "--task", task_b)
        assert len(a) > 0, f"Expected {task_a} models in catalog"
        assert len(b) > 0, f"Expected {task_b} models in catalog"
        assert _model_ids(a).isdisjoint(_model_ids(b))


# ===========================================================================
# --ep filtering
# ===========================================================================


class TestCatalogFilterEp:
    def test_ep_qnn_returns_only_qnn_models(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "qnn.json", "--ep", "qnn")
        assert len(models) > 0
        for m in models:
            assert "qnn" in _ep_keys(m), f"{m['model_id']} missing 'qnn' in supported_eps"

    def test_ep_vitisai_returns_only_vitisai_models(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        assert len(models) > 0
        for m in models:
            assert "vitisai" in _ep_keys(m)

    def test_ep_vitisai_is_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        """vitisai is not universally supported — filtered list must be smaller."""
        all_models = _run_json(tmp_path / "all.json")
        vitisai = _run_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        assert 0 < len(vitisai) < len(all_models)

    def test_ep_alias_ov_equals_openvino(self, tmp_path: Path) -> None:
        ov = _run_json(tmp_path / "ov.json", "--ep", "ov")
        openvino = _run_json(tmp_path / "openvino.json", "--ep", "openvino")
        assert _model_ids(ov) == _model_ids(openvino)

    def test_ep_full_name_equals_alias(self, tmp_path: Path) -> None:
        alias = _run_json(tmp_path / "alias.json", "--ep", "vitisai")
        full = _run_json(tmp_path / "full.json", "--ep", "VitisAIExecutionProvider")
        assert _model_ids(alias) == _model_ids(full)

    def test_no_ep_flag_returns_more_models_than_single_ep(self, tmp_path: Path) -> None:
        all_models = _run_json(tmp_path / "all.json")
        qnn = _run_json(tmp_path / "qnn.json", "--ep", "qnn")
        assert len(all_models) >= len(qnn)


# ===========================================================================
# --device filtering
# ===========================================================================


class TestCatalogFilterDevice:
    def test_device_npu_returns_only_npu_models(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "npu.json", "--device", "NPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "NPU"), f"{m['model_id']} lacks NPU device support"

    def test_device_cpu_returns_only_cpu_models(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "cpu.json", "--device", "CPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "CPU")

    def test_device_gpu_returns_only_gpu_models(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "gpu.json", "--device", "GPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "GPU")

    def test_device_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        upper = _run_json(tmp_path / "upper.json", "--device", "NPU")
        lower = _run_json(tmp_path / "lower.json", "--device", "npu")
        assert _model_ids(upper) == _model_ids(lower)

    def test_unknown_device_choice_exits_two(self) -> None:
        assert _run("--device", "FPGA").exit_code == 2


# ===========================================================================
# Combined filters
# ===========================================================================


class TestCatalogCombinedFilters:
    def test_model_type_and_task_intersection(self, tmp_path: Path) -> None:
        model_type, task = _TYPE_TASK_PAIR
        models = _run_json(
            tmp_path / "combined.json",
            "--model-type",
            model_type,
            "--task",
            task,
        )
        assert len(models) > 0
        for m in models:
            assert m["model_type"].lower() == model_type.lower()
            assert m["task"].lower() == task.lower()

    def test_combined_filter_subset_of_each_alone(self, tmp_path: Path) -> None:
        model_type, task = _TYPE_TASK_PAIR
        mtype_all = _run_json(tmp_path / "mtype.json", "--model-type", model_type)
        task_all = _run_json(tmp_path / "task.json", "--task", task)
        combined = _run_json(
            tmp_path / "combined.json",
            "--model-type",
            model_type,
            "--task",
            task,
        )
        assert len(combined) <= len(mtype_all)
        assert len(combined) <= len(task_all)

    def test_ep_and_model_type_intersection(self, tmp_path: Path) -> None:
        ep, model_type = _EP_MODEL_TYPE_PAIR
        models = _run_json(
            tmp_path / "ep_mtype.json",
            "--ep",
            ep,
            "--model-type",
            model_type,
        )
        assert len(models) > 0, f"Expected at least one {model_type} model supporting {ep}"
        for m in models:
            assert m["model_type"].lower() == model_type.lower()
            assert ep in _ep_keys(m)

    def test_ep_and_device_intersection_satisfies_both(self, tmp_path: Path) -> None:
        models = _run_json(
            tmp_path / "qnn_npu.json",
            "--ep",
            "qnn",
            "--device",
            "NPU",
        )
        for m in models:
            assert "qnn" in (m.get("supported_eps") or {}), (
                f"{m['model_id']} missing qnn in supported_eps"
            )
            assert _supports_device(m, "NPU"), f"{m['model_id']} does not support NPU"

    @pytest.mark.skipif(
        _DISJOINT_TYPE_TASK is None,
        reason="catalog has no type/task pair guaranteed to produce an empty result",
    )
    def test_disjoint_type_task_returns_empty(self, tmp_path: Path) -> None:
        """A (model_type, task) pair with no overlap returns an empty list."""
        model_type, task = _DISJOINT_TYPE_TASK  # type: ignore[misc]
        out = tmp_path / "empty.json"
        result = _run("--model-type", model_type, "--task", task, "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    @pytest.mark.skipif(not _TASKS, reason="hub_models.json has no task entries")
    def test_ep_and_task_combined_subset_of_ep_alone(self, tmp_path: Path) -> None:
        task = _TASKS[0]
        vitisai_all = _run_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        vitisai_task = _run_json(
            tmp_path / "vitisai_task.json",
            "--ep",
            "vitisai",
            "--task",
            task,
        )
        assert len(vitisai_task) <= len(vitisai_all)


# ===========================================================================
# --output file
# ===========================================================================


class TestCatalogOutputFile:
    def test_output_flag_writes_valid_json_list(self, tmp_path: Path) -> None:
        out = tmp_path / "catalog.json"
        result = _run("--output", str(out))
        assert result.exit_code == 0
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

    def test_output_entries_have_required_fields(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "all.json")
        required = {"model_id", "task", "model_type", "supported_eps"}
        for m in models:
            missing = required - m.keys()
            assert not missing, f"Model entry missing keys {missing}: {m}"

    @pytest.mark.skipif(not _MODEL_TYPES, reason="hub_models.json has no model_type entries")
    def test_output_with_filter_writes_only_matching(self, tmp_path: Path) -> None:
        model_type = _MODEL_TYPES[0]
        out = tmp_path / "filtered.json"
        result = _run("--model-type", model_type, "--output", str(out))
        assert result.exit_code == 0
        models = json.loads(out.read_text())
        assert all(m["model_type"].lower() == model_type.lower() for m in models)

    def test_output_empty_filter_writes_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _run("--model-type", "nonexistent_arch_xyz", "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_output_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "catalog.json"
        result = _run("--output", str(out))
        assert result.exit_code == 0
        assert out.is_file()

    def test_short_output_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        result = _run("-o", str(out))
        assert result.exit_code == 0
        assert out.is_file()

    def test_output_writes_json_array_not_dict(self, tmp_path: Path) -> None:
        out = tmp_path / "catalog.json"
        _run("--output", str(out))
        assert isinstance(json.loads(out.read_text()), list)


# ===========================================================================
# EP + device combination
# ===========================================================================


class TestCatalogEpAndDeviceCombination:
    """Verifies filtered model sets when --ep and --device are combined.

    Rich column headers (``Devices`` / ``EPs``) are not asserted here
    because CliRunner does not capture output from the module-level Console.
    """

    def test_ep_only_all_models_contain_ep(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "qnn.json", "--ep", "qnn")
        for m in models:
            assert "qnn" in _ep_keys(m)

    def test_device_only_all_models_support_device(self, tmp_path: Path) -> None:
        models = _run_json(tmp_path / "cpu.json", "--device", "CPU")
        for m in models:
            assert _supports_device(m, "CPU")

    def test_ep_and_device_result_subset_of_each_alone(self, tmp_path: Path) -> None:
        ep_only = _run_json(tmp_path / "ep.json", "--ep", "qnn")
        dev_only = _run_json(tmp_path / "dev.json", "--device", "NPU")
        both = _run_json(tmp_path / "both.json", "--ep", "qnn", "--device", "NPU")
        both_ids = _model_ids(both)
        assert both_ids <= _model_ids(ep_only), "Both-filter must be ⊆ ep-only filter"
        assert both_ids <= _model_ids(dev_only), "Both-filter must be ⊆ device-only filter"

    def test_no_ep_or_device_returns_all_models(self, tmp_path: Path) -> None:
        all_models = _run_json(tmp_path / "all.json")
        qnn = _run_json(tmp_path / "qnn.json", "--ep", "qnn")
        assert len(all_models) >= len(qnn)
