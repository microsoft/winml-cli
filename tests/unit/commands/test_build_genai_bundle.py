# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Routing tests for ``winml build`` -> genai bundle fast path.

These are CLI-plumbing tests: the heavy orchestrator (``build_genai_bundle``)
and the single-model pipeline (``_run_single_build``) are patched so the tests
only assert *which* path the command takes and *how* the CLI flags map onto the
orchestrator call. No model download, no real build.

The trigger under test (locked design):
    registered decoder-LLM family  AND  explicit ``--device npu``  AND
    explicit ``--ep qnn``  ->  genai bundle.  Every other combination keeps the
    stock single/composite behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}

_BUNDLE_TARGET = "winml.modelkit.models.winml.build_genai_bundle"
_RUN_SINGLE_TARGET = "winml.modelkit.commands.build._run_single_build"
_COMPOSITE_TARGET = "winml.modelkit.loader.resolution.resolve_composite_components"


def _fake_resolve_check_device_ep(*, device: str = "auto", ep: str | None = None):
    resolved = device.lower() if device != "auto" else "npu"
    eps = _DEVICE_TO_EPS.get(resolved, ["CPUExecutionProvider"])
    return resolved, ["npu", "gpu", "cpu"], eps


@pytest.fixture(autouse=True)
def _mock_hardware():
    """Avoid hardware/WinML SDK probing during EP/device resolution."""
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = False
    with (
        patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "gpu", "cpu"]),
        ),
        patch(
            "winml.modelkit.sysinfo.resolve_eps",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
        ),
        patch(
            "winml.modelkit.sysinfo.resolve_check_device_ep",
            side_effect=_fake_resolve_check_device_ep,
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _skip_task_validation():
    """Skip HF task/model preflight so model_type comes from config.loader."""
    with patch(
        "winml.modelkit.commands.build._validate_task_supported_for_model",
        return_value=None,
    ):
        yield


def _write_config(
    tmp_path: Path,
    *,
    model_type: str,
    task: str = "text-generation",
    name: str = "config.json",
) -> str:
    config = {
        "loader": {"task": task, "model_type": model_type},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return str(path)


def _invoke(args: list[str]):
    from winml.modelkit.commands.build import build

    return CliRunner().invoke(build, args, obj={"debug": True}, catch_exceptions=False)


def _record_bundle(store: dict):
    def _fake(model_id, output_dir, recipe, **kwargs):
        store["model_id"] = model_id
        store["output_dir"] = output_dir
        store["recipe"] = recipe
        store["kwargs"] = kwargs
        return Path(output_dir) / "genai_config.json"

    return _fake


def test_registered_family_npu_qnn_routes_to_bundle(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="qwen3")
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            ["-c", cfg, "-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--device", "npu", "--ep", "qnn"]
        )

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1
    run_single.assert_not_called()

    from winml.modelkit.models.winml import resolve_genai_bundle

    assert recorded["model_id"] == "Qwen/Qwen3-0.6B"
    assert Path(recorded["output_dir"]) == out
    assert recorded["recipe"] is resolve_genai_bundle("qwen3")
    kwargs = recorded["kwargs"]
    assert kwargs["ep"] == "qnn"
    assert kwargs["device"] == "npu"
    assert kwargs["force_rebuild"] is False
    assert kwargs["precision"] is None
    assert callable(kwargs["emit"])


def test_rebuild_flag_maps_to_force_rebuild(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="qwen3")
    recorded: dict = {}

    with (
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)),
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-c",
                cfg,
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "b"),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--rebuild",
            ]
        )

    assert result.exit_code == 0, result.output
    assert recorded["kwargs"]["force_rebuild"] is True


def test_precision_override_forwarded(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="qwen3")
    recorded: dict = {}

    with (
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)),
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-c",
                cfg,
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "b"),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--precision",
                "w8a16",
            ]
        )

    assert result.exit_code == 0, result.output
    assert recorded["kwargs"]["precision"] == "w8a16"


def test_npu_without_explicit_ep_does_not_route(tmp_path: Path):
    """Auto-resolved QNN (no explicit --ep) must keep the stock path."""
    cfg = _write_config(tmp_path, model_type="qwen3")

    with (
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            ["-c", cfg, "-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "o"), "--device", "npu"]
        )

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()


def test_cpu_target_does_not_route(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="qwen3")

    with (
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-c",
                cfg,
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "o"),
                "--device",
                "cpu",
                "--ep",
                "cpu",
            ]
        )

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()


def test_unregistered_family_npu_qnn_does_not_route(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="resnet", task="image-classification")

    with (
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-c",
                cfg,
                "-m",
                "microsoft/resnet-50",
                "-o",
                str(tmp_path / "o"),
                "--device",
                "npu",
                "--ep",
                "qnn",
            ]
        )

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()
    run_single.assert_called_once()


def test_use_cache_rejected_for_bundle(tmp_path: Path):
    cfg = _write_config(tmp_path, model_type="qwen3")

    with (
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            ["-c", cfg, "-m", "Qwen/Qwen3-0.6B", "--use-cache", "--device", "npu", "--ep", "qnn"]
        )

    assert result.exit_code != 0
    assert "output-dir" in result.output.lower()
    bundle.assert_not_called()
