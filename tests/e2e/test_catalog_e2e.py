# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml catalog`` CLI command.

Coverage layout
---------------
* ``TestCatalogCliSurface`` — ``--help``, no-arg invocation, invalid CLI
  choice validation. These rely only on Click's own output.
* ``TestCatalogFilterModelType`` — ``--model-type`` / ``-t`` filtering:
  known type, case-insensitivity, unknown type (empty result).
* ``TestCatalogFilterTask`` — ``--task`` / ``-k`` filtering: known task,
  case-insensitivity, unknown task.
* ``TestCatalogFilterEp`` — ``--ep`` filtering: short aliases, full
  provider names, alias equivalence (``ov`` == ``openvino``).
* ``TestCatalogFilterDevice`` — ``--device`` filtering: CPU / GPU / NPU,
  case-insensitivity.
* ``TestCatalogCombinedFilters`` — Intersecting multiple flags; empty
  intersection produces an empty list.
* ``TestCatalogOutputFile`` — ``--output`` / ``-o``: writes valid JSON,
  content matches filtered model list, parent directory is created.
* ``TestCatalogEpAndDeviceCombination`` — Combined EP+device filter
  behaviour: independent constraints, subset relationships, and the
  no-filter full-catalog baseline.

All data assertions go through the ``--output`` JSON file rather than
parsing Rich console output (which is not captured by ``CliRunner`` when
the ``Console`` is initialised at module level).

Markers
-------
* ``e2e`` — auto-skipped unless ``-m e2e`` is passed (see conftest.py).

Usage::

    uv run pytest tests/e2e/ -m e2e -k catalog
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.catalog import catalog


if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result


pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Helpers — backed by a shared runner to avoid per-call CliRunner() creation
# ---------------------------------------------------------------------------

_RUNNER = CliRunner()


def _invoke(*args: str) -> Result:
    """Run ``winml catalog <args>`` via the shared module-level runner."""
    return _RUNNER.invoke(catalog, list(args), obj={})


def _invoke_json(out_path: Path, *args: str) -> list[dict]:
    """Run ``winml catalog --output <out_path> <args>`` and return parsed JSON.

    Asserts exit 0 and a valid JSON list before returning.
    """
    result = _invoke(*args, "--output", str(out_path))
    assert result.exit_code == 0, f"catalog exited {result.exit_code}\n{result.output}"
    assert out_path.is_file(), "--output file was not written"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"--output JSON must be a list, got {type(data)}"
    return data


def _result_is_subset_of_full_catalog(
    models: list[dict],
    *,
    model_type: str | None = None,
    task: str | None = None,
) -> bool:
    """Return True if every returned model satisfies the given filter criteria."""
    for m in models:
        if model_type and m["model_type"].lower() != model_type.lower():
            return False
        if task and m["task"].lower() != task.lower():
            return False
    return True


def _all_model_ids(models: list[dict]) -> set[str]:
    return {m["model_id"] for m in models}


def _all_model_types(models: list[dict]) -> set[str]:
    return {m["model_type"] for m in models}


def _all_tasks(models: list[dict]) -> set[str]:
    return {m["task"] for m in models}


# ---------------------------------------------------------------------------
# Fixtures — backed by conftest ``runner``; module-scoped ones derive live
# catalog content once and share it across the test session.
# ---------------------------------------------------------------------------


@pytest.fixture
def invoke_catalog(runner: CliRunner) -> object:
    """Catalog invoker backed by the shared conftest ``runner`` fixture."""

    def _call(*args: str) -> Result:
        return runner.invoke(catalog, list(args), obj={})

    return _call


@pytest.fixture(scope="module")
def catalog_data(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """Full catalog fetched once per module via the CLI (source of truth for filter tests)."""
    return _invoke_json(tmp_path_factory.mktemp("catalog") / "all.json")


@pytest.fixture(scope="module")
def two_model_types(catalog_data: list[dict]) -> tuple[str, str]:
    """Two distinct model_type values, chosen by frequency from the live catalog."""
    counts = Counter(m["model_type"] for m in catalog_data if m.get("model_type"))
    distinct = [t for t, _ in counts.most_common() if t]
    if len(distinct) < 2:
        pytest.skip("catalog needs ≥2 distinct model_types for this test")
    return distinct[0], distinct[1]


@pytest.fixture(scope="module")
def two_tasks(catalog_data: list[dict]) -> tuple[str, str]:
    """Two distinct task values, chosen by frequency from the live catalog."""
    counts = Counter(m["task"] for m in catalog_data if m.get("task"))
    distinct = [t for t, _ in counts.most_common() if t]
    if len(distinct) < 2:
        pytest.skip("catalog needs ≥2 distinct tasks for this test")
    return distinct[0], distinct[1]


@pytest.fixture(scope="module")
def type_task_pair(catalog_data: list[dict]) -> tuple[str, str]:
    """A (model_type, task) pair known to exist in the live catalog."""
    for m in catalog_data:
        if m.get("model_type") and m.get("task"):
            return m["model_type"], m["task"]
    pytest.skip("catalog has no models with both model_type and task set")


@pytest.fixture(scope="module")
def ep_model_type_pair(catalog_data: list[dict]) -> tuple[str, str]:
    """An (ep_key, model_type) pair known to exist in the live catalog."""
    for m in catalog_data:
        eps = sorted((m.get("supported_eps") or {}).keys())
        if eps and m.get("model_type"):
            return eps[0], m["model_type"]
    pytest.skip("catalog has no models with both supported_eps and model_type set")


@pytest.fixture(scope="module")
def disjoint_type_task(catalog_data: list[dict]) -> tuple[str, str]:
    """A (model_type, task) pair guaranteed to return no models from the catalog.

    Finds a task that appears in the catalog but not for the chosen model_type,
    so ``--model-type <type> --task <task>`` must return an empty list.
    """
    type_tasks: dict[str, set[str]] = defaultdict(set)
    all_tasks: set[str] = set()
    for m in catalog_data:
        mtype = m.get("model_type", "")
        task = m.get("task", "")
        if mtype and task:
            type_tasks[mtype].add(task)
            all_tasks.add(task)
    for mtype in sorted(type_tasks):
        missing = all_tasks - type_tasks[mtype]
        if missing:
            return mtype, sorted(missing)[0]
    pytest.skip("catalog has no type/task combination guaranteed to produce an empty result")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCatalogCliSurface:
    """Parser-level behaviours — no model or EP runtime required."""

    def test_help_lists_every_documented_option(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0
        for opt in ("--model-type", "--task", "--ep", "--device", "--output"):
            assert opt in result.output, f"--help missing option {opt!r}"

    def test_no_filter_args_returns_all_models(self, tmp_path: Path) -> None:
        """``winml catalog`` with no filter flags returns the full catalog."""
        models = _invoke_json(tmp_path / "all.json")
        assert len(models) > 0, "Catalog must not be empty"

    def test_invalid_ep_choice_exits_two(self) -> None:
        result = _invoke("--ep", "totally_unknown_ep_xyz")
        assert result.exit_code == 2
        assert "Invalid value for '--ep'" in result.output

    def test_invalid_device_choice_exits_two(self) -> None:
        result = _invoke("--device", "TPU")
        assert result.exit_code == 2
        assert "Invalid value for '--device'" in result.output

    def test_short_flags_accepted(self, tmp_path: Path, type_task_pair: tuple[str, str]) -> None:
        """-t and -k short aliases are accepted by the parser."""
        model_type, task = type_task_pair
        models = _invoke_json(tmp_path / "out.json", "-t", model_type, "-k", task)
        assert len(models) > 0, f"Expected at least one {model_type}/{task} model"
        assert _result_is_subset_of_full_catalog(models, model_type=model_type, task=task)


# ---------------------------------------------------------------------------
# Filter by model-type
# ---------------------------------------------------------------------------


class TestCatalogFilterModelType:
    def test_known_model_type_returns_only_matching_entries(
        self, tmp_path: Path, two_model_types: tuple[str, str]
    ) -> None:
        model_type = two_model_types[0]
        models = _invoke_json(tmp_path / "mtype.json", "--model-type", model_type)
        assert len(models) > 0, f"Expected at least one {model_type} model"
        assert _all_model_types(models) == {model_type}, (
            f"All returned models must have model_type='{model_type}'"
        )

    def test_model_type_filter_is_case_insensitive(
        self, tmp_path: Path, two_model_types: tuple[str, str]
    ) -> None:
        model_type = two_model_types[0]
        lower = _invoke_json(tmp_path / "lower.json", "--model-type", model_type.lower())
        upper = _invoke_json(tmp_path / "upper.json", "--model-type", model_type.upper())
        mixed = _invoke_json(tmp_path / "mixed.json", "--model-type", model_type.swapcase())
        # All three invocations must return the same set of models.
        assert _all_model_ids(lower) == _all_model_ids(upper) == _all_model_ids(mixed), (
            "--model-type filtering must be case-insensitive"
        )

    def test_model_type_produces_strict_subset_of_full_catalog(
        self, tmp_path: Path, two_model_types: tuple[str, str]
    ) -> None:
        model_type = two_model_types[0]
        all_models = _invoke_json(tmp_path / "all.json")
        mtype_models = _invoke_json(tmp_path / "mtype.json", "--model-type", model_type)
        assert 0 < len(mtype_models) < len(all_models)

    def test_unknown_model_type_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _invoke("--model-type", "nonexistent_arch_xyz", "--output", str(out))
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert json.loads(out.read_text()) == []

    def test_different_model_types_return_disjoint_sets(
        self, tmp_path: Path, two_model_types: tuple[str, str]
    ) -> None:
        type_a, type_b = two_model_types
        a_models = _invoke_json(tmp_path / "type_a.json", "--model-type", type_a)
        b_models = _invoke_json(tmp_path / "type_b.json", "--model-type", type_b)
        assert len(a_models) > 0, f"Expected at least one {type_a} model"
        assert len(b_models) > 0, f"Expected at least one {type_b} model"
        assert _all_model_ids(a_models).isdisjoint(_all_model_ids(b_models)), (
            f"{type_a} and {type_b} filters must not overlap"
        )


# ---------------------------------------------------------------------------
# Filter by task
# ---------------------------------------------------------------------------


class TestCatalogFilterTask:
    def test_known_task_returns_only_matching_entries(
        self, tmp_path: Path, two_tasks: tuple[str, str]
    ) -> None:
        task = two_tasks[0]
        models = _invoke_json(tmp_path / "task.json", "--task", task)
        assert len(models) > 0
        assert _all_tasks(models) == {task}

    def test_task_filter_is_case_insensitive(
        self, tmp_path: Path, two_tasks: tuple[str, str]
    ) -> None:
        task = two_tasks[0]
        lower = _invoke_json(tmp_path / "lower.json", "--task", task.lower())
        upper = _invoke_json(tmp_path / "upper.json", "--task", task.upper())
        assert _all_model_ids(lower) == _all_model_ids(upper)

    def test_task_produces_strict_subset_of_full_catalog(
        self, tmp_path: Path, two_tasks: tuple[str, str]
    ) -> None:
        task = two_tasks[0]
        all_models = _invoke_json(tmp_path / "all.json")
        task_models = _invoke_json(tmp_path / "task.json", "--task", task)
        assert 0 < len(task_models) < len(all_models)

    def test_unknown_task_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _invoke("--task", "nonexistent-task-xyz", "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_different_tasks_return_disjoint_sets(
        self, tmp_path: Path, two_tasks: tuple[str, str]
    ) -> None:
        """Two distinct tasks that no model can simultaneously satisfy."""
        task_a, task_b = two_tasks
        a_models = _invoke_json(tmp_path / "task_a.json", "--task", task_a)
        b_models = _invoke_json(tmp_path / "task_b.json", "--task", task_b)
        assert len(a_models) > 0, f"Expected {task_a} models in catalog"
        assert len(b_models) > 0, f"Expected {task_b} models in catalog"
        assert _all_model_ids(a_models).isdisjoint(_all_model_ids(b_models))


# ---------------------------------------------------------------------------
# Filter by EP
# ---------------------------------------------------------------------------


def _ep_keys(model: dict) -> set[str]:
    """Return the set of EP keys in a model's ``supported_eps`` dict."""
    return set((model.get("supported_eps") or {}).keys())


class TestCatalogFilterEp:
    def test_ep_alias_qnn_returns_only_qnn_models(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "qnn.json", "--ep", "qnn")
        assert len(models) > 0
        for m in models:
            assert "qnn" in _ep_keys(m), f"Model {m['model_id']} missing 'qnn' in supported_eps"

    def test_ep_alias_vitisai_returns_only_vitisai_models(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        assert len(models) > 0
        for m in models:
            assert "vitisai" in _ep_keys(m)

    def test_ep_vitisai_is_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        """vitisai is not universally supported — filtered list must be smaller."""
        all_models = _invoke_json(tmp_path / "all.json")
        vitisai = _invoke_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        assert 0 < len(vitisai) < len(all_models)

    def test_ep_alias_ov_equals_openvino(self, tmp_path: Path) -> None:
        """``--ep ov`` and ``--ep openvino`` must resolve to the same models."""
        ov = _invoke_json(tmp_path / "ov.json", "--ep", "ov")
        openvino = _invoke_json(tmp_path / "openvino.json", "--ep", "openvino")
        assert _all_model_ids(ov) == _all_model_ids(openvino), (
            "'ov' and 'openvino' aliases must resolve to the same EP"
        )

    def test_ep_full_name_equals_alias(self, tmp_path: Path) -> None:
        """``--ep VitisAIExecutionProvider`` must equal ``--ep vitisai``."""
        alias = _invoke_json(tmp_path / "alias.json", "--ep", "vitisai")
        full = _invoke_json(tmp_path / "full.json", "--ep", "VitisAIExecutionProvider")
        assert _all_model_ids(alias) == _all_model_ids(full)

    def test_no_ep_flag_returns_all_models(self, tmp_path: Path) -> None:
        all_models = _invoke_json(tmp_path / "all.json")
        qnn = _invoke_json(tmp_path / "qnn.json", "--ep", "qnn")
        assert len(all_models) >= len(qnn)


# ---------------------------------------------------------------------------
# Filter by device
# ---------------------------------------------------------------------------


def _supports_device(model: dict, device: str) -> bool:
    device_upper = device.upper()
    return any(device_upper in devs for devs in (model.get("supported_eps") or {}).values())


class TestCatalogFilterDevice:
    def test_device_npu_returns_only_npu_models(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "npu.json", "--device", "NPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "NPU"), f"Model {m['model_id']} lacks NPU device support"

    def test_device_cpu_returns_only_cpu_models(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "cpu.json", "--device", "CPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "CPU")

    def test_device_gpu_returns_only_gpu_models(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "gpu.json", "--device", "GPU")
        assert len(models) > 0
        for m in models:
            assert _supports_device(m, "GPU")

    def test_device_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        upper = _invoke_json(tmp_path / "upper.json", "--device", "NPU")
        lower = _invoke_json(tmp_path / "lower.json", "--device", "npu")
        assert _all_model_ids(upper) == _all_model_ids(lower)

    def test_unknown_device_choice_exits_two(self) -> None:
        result = _invoke("--device", "FPGA")
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------


class TestCatalogCombinedFilters:
    def test_model_type_and_task_intersection(
        self, tmp_path: Path, type_task_pair: tuple[str, str]
    ) -> None:
        """``--model-type <type> --task <task>`` returns only matching models."""
        model_type, task = type_task_pair
        models = _invoke_json(
            tmp_path / "mtype_task.json",
            "--model-type",
            model_type,
            "--task",
            task,
        )
        assert len(models) > 0
        for m in models:
            assert m["model_type"].lower() == model_type.lower()
            assert m["task"].lower() == task.lower()

    def test_model_type_and_task_intersection_is_smaller_than_either_alone(
        self, tmp_path: Path, type_task_pair: tuple[str, str]
    ) -> None:
        model_type, task = type_task_pair
        mtype_all = _invoke_json(tmp_path / "mtype.json", "--model-type", model_type)
        task_all = _invoke_json(tmp_path / "task.json", "--task", task)
        combined = _invoke_json(
            tmp_path / "combined.json",
            "--model-type",
            model_type,
            "--task",
            task,
        )
        assert len(combined) <= len(mtype_all)
        assert len(combined) <= len(task_all)

    def test_ep_and_model_type_intersection(
        self, tmp_path: Path, ep_model_type_pair: tuple[str, str]
    ) -> None:
        """``--ep <ep> --model-type <type>`` returns only matching models."""
        ep, model_type = ep_model_type_pair
        models = _invoke_json(
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

    def test_ep_and_device_both_given(self, tmp_path: Path) -> None:
        """``--ep qnn --device NPU`` returns models that have QNN AND support NPU.

        The two filters are applied independently:
        - ``_filter_by_ep`` keeps models whose ``supported_eps`` contains QNN.
        - ``_filter_by_device`` keeps models where NPU appears in *any* EP's
          device list.
        A model may therefore satisfy both without QNN being the NPU-facing EP.
        """
        models = _invoke_json(
            tmp_path / "qnn_npu.json",
            "--ep",
            "qnn",
            "--device",
            "NPU",
        )
        for m in models:
            eps = m.get("supported_eps") or {}
            assert "qnn" in eps, f"{m['model_id']} missing qnn in supported_eps"
            assert _supports_device(m, "NPU"), f"{m['model_id']} does not support NPU on any EP"

    def test_model_type_and_task_with_no_overlap_returns_empty(
        self, tmp_path: Path, disjoint_type_task: tuple[str, str]
    ) -> None:
        """A type+task pair with no matching models returns an empty list."""
        model_type, task = disjoint_type_task
        out = tmp_path / "empty.json"
        result = _invoke(
            "--model-type",
            model_type,
            "--task",
            task,
            "--output",
            str(out),
        )
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_ep_and_task_intersection_subset_of_ep_alone(
        self, tmp_path: Path, two_tasks: tuple[str, str]
    ) -> None:
        task = two_tasks[0]
        vitisai_all = _invoke_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        vitisai_task = _invoke_json(
            tmp_path / "vitisai_task.json",
            "--ep",
            "vitisai",
            "--task",
            task,
        )
        assert len(vitisai_task) <= len(vitisai_all)


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------


class TestCatalogOutputFile:
    def test_output_flag_writes_valid_json_list(self, tmp_path: Path) -> None:
        out = tmp_path / "catalog.json"
        result = _invoke("--output", str(out))
        assert result.exit_code == 0
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

    def test_output_json_entries_have_required_fields(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "all.json")
        required_keys = {"model_id", "task", "model_type", "supported_eps"}
        for m in models:
            missing = required_keys - m.keys()
            assert not missing, f"Model entry missing keys {missing}: {m}"

    def test_output_with_filter_writes_only_matching_models(
        self, tmp_path: Path, two_model_types: tuple[str, str]
    ) -> None:
        """``--output`` combined with ``--model-type`` writes the filtered subset."""
        model_type = two_model_types[0]
        out = tmp_path / "filtered.json"
        result = _invoke("--model-type", model_type, "--output", str(out))
        assert result.exit_code == 0
        models = json.loads(out.read_text())
        assert all(m["model_type"].lower() == model_type.lower() for m in models)

    def test_output_empty_filter_writes_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _invoke("--model-type", "nonexistent_arch_xyz", "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_output_creates_parent_directories(self, tmp_path: Path) -> None:
        """``--output`` must create intermediate directories that don't exist yet."""
        out = tmp_path / "nested" / "deep" / "catalog.json"
        result = _invoke("--output", str(out))
        assert result.exit_code == 0
        assert out.is_file()

    def test_short_output_flag(self, tmp_path: Path) -> None:
        """-o short alias is accepted."""
        out = tmp_path / "out.json"
        result = _invoke("-o", str(out))
        assert result.exit_code == 0
        assert out.is_file()

    def test_output_json_is_list_not_dict(self, tmp_path: Path) -> None:
        """``--output`` must write a JSON array (list of model entries)."""
        out = tmp_path / "catalog.json"
        _invoke("--output", str(out))
        data = json.loads(out.read_text())
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# EP + device combination
# ---------------------------------------------------------------------------


class TestCatalogEpAndDeviceCombination:
    """Combined EP+device filter behaviour.

    Verifies the underlying filtered model sets rather than Rich column
    headers (which are not captured by CliRunner when Console is
    module-level).
    """

    def test_only_ep_given_all_models_contain_ep(self, tmp_path: Path) -> None:
        """When only --ep is given, every returned model has that EP."""
        models = _invoke_json(tmp_path / "qnn.json", "--ep", "qnn")
        for m in models:
            assert "qnn" in _ep_keys(m)

    def test_only_device_given_all_models_support_device(self, tmp_path: Path) -> None:
        """When only --device is given, every returned model supports that device."""
        models = _invoke_json(tmp_path / "cpu.json", "--device", "CPU")
        for m in models:
            assert _supports_device(m, "CPU")

    def test_both_ep_and_device_returns_intersection(self, tmp_path: Path) -> None:
        """Both --ep and --device → models must satisfy both constraints."""
        ep_only = _invoke_json(tmp_path / "ep.json", "--ep", "qnn")
        dev_only = _invoke_json(tmp_path / "dev.json", "--device", "NPU")
        both = _invoke_json(tmp_path / "both.json", "--ep", "qnn", "--device", "NPU")

        ep_ids = _all_model_ids(ep_only)
        dev_ids = _all_model_ids(dev_only)
        both_ids = _all_model_ids(both)

        assert both_ids <= ep_ids, "Both-filter must be ⊆ ep-only filter"
        assert both_ids <= dev_ids, "Both-filter must be ⊆ device-only filter"

    def test_neither_ep_nor_device_returns_all_models(self, tmp_path: Path) -> None:
        """No EP/device flag → catalog returns all models."""
        all_models = _invoke_json(tmp_path / "all.json")
        ep_all = _invoke_json(tmp_path / "qnn.json", "--ep", "qnn")
        # Without EP filter the count must be >= any single EP's count.
        assert len(all_models) >= len(ep_all)
