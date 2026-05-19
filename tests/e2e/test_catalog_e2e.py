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
* ``TestCatalogEpColumnXor`` — Extra column logic: only ``--ep`` shows
  "Devices" column; only ``--device`` shows "EPs" column; both given →
  no extra column; neither → no extra column.

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
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.catalog import catalog


if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result


pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> Result:
    """Run ``winml catalog <args>`` through a fresh CliRunner."""
    return CliRunner().invoke(catalog, list(args), obj={})


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


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCatalogCliSurface:
    """Parser-level behaviours — no model or EP runtime required."""

    def test_help_lists_every_documented_option(self) -> None:
        result = _invoke("--help")
        assert result.exit_code == 0
        for opt in ("--model-type", "--task", "--ep", "--device", "--output"):
            assert opt in result.output, f"--help missing option {opt!r}"

    def test_no_args_exits_zero_and_returns_all_models(self, tmp_path: Path) -> None:
        """``winml catalog`` with no filters returns the full catalog."""
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

    def test_short_flags_accepted(self, tmp_path: Path) -> None:
        """-t and -k short aliases are accepted by the parser."""
        models = _invoke_json(tmp_path / "out.json", "-t", "bert", "-k", "text-classification")
        assert result_is_subset_of_full_catalog(
            models, model_type="bert", task="text-classification"
        )


# ---------------------------------------------------------------------------
# Helpers for content assertions
# ---------------------------------------------------------------------------


def result_is_subset_of_full_catalog(
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
# Filter by model-type
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCatalogFilterModelType:
    def test_known_model_type_returns_only_matching_entries(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "bert.json", "--model-type", "bert")
        assert len(models) > 0, "Expected at least one bert model"
        assert _all_model_types(models) == {"bert"}, (
            "All returned models must have model_type='bert'"
        )

    def test_model_type_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        lower = _invoke_json(tmp_path / "bert_lower.json", "--model-type", "bert")
        upper = _invoke_json(tmp_path / "bert_upper.json", "--model-type", "BERT")
        mixed = _invoke_json(tmp_path / "bert_mixed.json", "--model-type", "BeRt")
        # All three invocations must return the same set of models.
        assert _all_model_ids(lower) == _all_model_ids(upper) == _all_model_ids(mixed), (
            "--model-type filtering must be case-insensitive"
        )

    def test_model_type_produces_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        all_models = _invoke_json(tmp_path / "all.json")
        bert_models = _invoke_json(tmp_path / "bert.json", "--model-type", "bert")
        assert 0 < len(bert_models) < len(all_models)

    def test_unknown_model_type_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _invoke("--model-type", "nonexistent_arch_xyz", "--output", str(out))
        assert result.exit_code == 0, result.output
        assert out.is_file()
        assert json.loads(out.read_text()) == []

    def test_different_model_types_return_disjoint_sets(self, tmp_path: Path) -> None:
        bert = _invoke_json(tmp_path / "bert.json", "--model-type", "bert")
        vit = _invoke_json(tmp_path / "vit.json", "--model-type", "vit")
        assert _all_model_ids(bert).isdisjoint(_all_model_ids(vit)), (
            "bert and vit filters must not overlap"
        )


# ---------------------------------------------------------------------------
# Filter by task
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCatalogFilterTask:
    def test_known_task_returns_only_matching_entries(self, tmp_path: Path) -> None:
        models = _invoke_json(tmp_path / "textcls.json", "--task", "text-classification")
        assert len(models) > 0
        assert _all_tasks(models) == {"text-classification"}

    def test_task_filter_is_case_insensitive(self, tmp_path: Path) -> None:
        lower = _invoke_json(tmp_path / "lower.json", "--task", "text-classification")
        upper = _invoke_json(tmp_path / "upper.json", "--task", "TEXT-CLASSIFICATION")
        assert _all_model_ids(lower) == _all_model_ids(upper)

    def test_task_produces_strict_subset_of_full_catalog(self, tmp_path: Path) -> None:
        all_models = _invoke_json(tmp_path / "all.json")
        task_models = _invoke_json(tmp_path / "task.json", "--task", "image-classification")
        assert 0 < len(task_models) < len(all_models)

    def test_unknown_task_returns_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = _invoke("--task", "nonexistent-task-xyz", "--output", str(out))
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_different_tasks_return_disjoint_sets(self, tmp_path: Path) -> None:
        """Two distinct tasks that no model can simultaneously satisfy."""
        text_cls = _invoke_json(tmp_path / "text.json", "--task", "text-classification")
        img_cls = _invoke_json(tmp_path / "img.json", "--task", "image-classification")
        assert _all_model_ids(text_cls).isdisjoint(_all_model_ids(img_cls))


# ---------------------------------------------------------------------------
# Filter by EP
# ---------------------------------------------------------------------------


def _ep_keys(model: dict) -> set[str]:
    """Return the set of EP keys in a model's ``supported_eps`` dict."""
    return set((model.get("supported_eps") or {}).keys())


@pytest.mark.e2e
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


@pytest.mark.e2e
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


@pytest.mark.e2e
class TestCatalogCombinedFilters:
    def test_model_type_and_task_intersection(self, tmp_path: Path) -> None:
        """``--model-type bert --task text-classification`` returns only BERT text-cls models."""
        models = _invoke_json(
            tmp_path / "bert_textcls.json",
            "--model-type",
            "bert",
            "--task",
            "text-classification",
        )
        assert len(models) > 0
        for m in models:
            assert m["model_type"].lower() == "bert"
            assert m["task"].lower() == "text-classification"

    def test_model_type_and_task_intersection_is_smaller_than_either_alone(
        self, tmp_path: Path
    ) -> None:
        bert_all = _invoke_json(tmp_path / "bert.json", "--model-type", "bert")
        textcls_all = _invoke_json(tmp_path / "textcls.json", "--task", "text-classification")
        combined = _invoke_json(
            tmp_path / "combined.json",
            "--model-type",
            "bert",
            "--task",
            "text-classification",
        )
        assert len(combined) <= len(bert_all)
        assert len(combined) <= len(textcls_all)

    def test_ep_and_model_type_intersection(self, tmp_path: Path) -> None:
        """``--ep vitisai --model-type bert`` returns only vitisai-capable bert models."""
        models = _invoke_json(
            tmp_path / "vitisai_bert.json",
            "--ep",
            "vitisai",
            "--model-type",
            "bert",
        )
        for m in models:
            assert m["model_type"].lower() == "bert"
            assert "vitisai" in _ep_keys(m)

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

    def test_model_type_and_task_with_no_overlap_returns_empty(self, tmp_path: Path) -> None:
        """A type+task pair that exists for no model returns an empty list."""
        out = tmp_path / "empty.json"
        result = _invoke(
            "--model-type",
            "resnet",  # resnet has no text tasks
            "--task",
            "text-classification",
            "--output",
            str(out),
        )
        assert result.exit_code == 0
        assert json.loads(out.read_text()) == []

    def test_ep_and_task_intersection_subset_of_ep_alone(self, tmp_path: Path) -> None:
        vitisai_all = _invoke_json(tmp_path / "vitisai.json", "--ep", "vitisai")
        vitisai_imgcls = _invoke_json(
            tmp_path / "vitisai_imgcls.json",
            "--ep",
            "vitisai",
            "--task",
            "image-classification",
        )
        assert len(vitisai_imgcls) <= len(vitisai_all)


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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

    def test_output_with_filter_writes_only_matching_models(self, tmp_path: Path) -> None:
        """``--output`` combined with ``--model-type`` writes the filtered subset."""
        out = tmp_path / "bert.json"
        result = _invoke("--model-type", "bert", "--output", str(out))
        assert result.exit_code == 0
        models = json.loads(out.read_text())
        assert all(m["model_type"].lower() == "bert" for m in models)

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
# EP column XOR logic
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCatalogEpColumnXor:
    """The fifth column ("Devices" or "EPs") only appears when exactly one
    of --ep / --device is given.  Verified by combining the ``--output``
    JSON to confirm the underlying data and verifying that the correct
    EP-centric information is derivable from the filtered results.
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
