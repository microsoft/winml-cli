# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Routing tests for ``winml build`` -> genai bundle fast path.

These are CLI-plumbing tests: the heavy orchestrator (``build_genai_bundle``)
and the single-model pipeline (``_run_single_build``) are patched so the tests
only assert *which* path the command takes and *how* the CLI flags map onto the
orchestrator call. No model download, no real build.

The bundle is fully recipe-driven, so it is built directly from ``-m`` with no
``-c/--config`` (a supplied config would be discarded, and is rejected). Tests
inject ``model_type`` by stubbing the auto config generator instead.

The trigger under test (locked design):
    registered decoder-LLM family  AND  explicit ``--ep qnn``  AND an NPU target
    ->  genai bundle.  The NPU target may be explicit (``--device npu``) or
    resolved from ``auto`` -- whether ``--device auto`` is typed or ``--device``
    is omitted (its default).  Every other combination keeps the stock
    single/composite behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.session import EPDeviceTarget


_BUNDLE_TARGET = "winml.modelkit.models.winml.build_genai_bundle"
_RUN_SINGLE_TARGET = "winml.modelkit.commands.build._run_single_build"
_COMPOSITE_TARGET = "winml.modelkit.loader.resolution.resolve_composite_components"
_GENERATE_TARGET = "winml.modelkit.config.generate_build_config"
# The genai fast path resolves ``--device auto`` via ``session.resolve_device``
# (an ``EPDeviceTarget -> EPDeviceTarget`` deducer).
_RESOLVE_DEVICE_TARGET = "winml.modelkit.session.resolve_device"
_NPU_TARGET = EPDeviceTarget(ep="QNNExecutionProvider", device="npu")
_CPU_TARGET = EPDeviceTarget(ep="CPUExecutionProvider", device="cpu")


@pytest.fixture(autouse=True)
def _mock_resolve_device():
    """Stub device resolution so ``--device auto`` deterministically maps to the
    NPU without probing real hardware. Tests needing a different resolution
    override this patch locally.
    """
    with patch(_RESOLVE_DEVICE_TARGET, return_value=_NPU_TARGET):
        yield


@pytest.fixture(autouse=True)
def _skip_task_validation():
    """Skip HF task/model preflight so model_type comes from config.loader."""
    with patch(
        "winml.modelkit.commands.build._validate_task_supported_for_model",
        return_value=None,
    ):
        yield


def _fake_config(model_type: str, task: str = "text-generation"):
    """A minimal valid ``WinMLBuildConfig`` carrying ``loader.model_type``.

    The genai fast path reads ``model_type`` from the (auto-generated) config's
    loader, so stubbing ``generate_build_config`` to return this is how a test
    selects the routed family -- no ``-c`` needed (and none is accepted).
    """
    from winml.modelkit.config import WinMLBuildConfig

    return WinMLBuildConfig.from_dict(
        {
            "loader": {"task": task, "model_type": model_type},
            "export": {"opset_version": 17, "batch_size": 1},
            "optim": {},
            "quant": None,
            "compile": None,
        }
    )


def _write_config_file(tmp_path: Path, *, model_type: str = "qwen3") -> str:
    """Write a real config JSON file (only for the -c rejection test)."""
    config = {
        "loader": {"task": "text-generation", "model_type": model_type},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    path = tmp_path / "config.json"
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


def _fake_ctx(provided: set[str]):
    """A click.Context stand-in whose parameter sources report ``provided``.

    ``is_cli_provided`` only inspects ``ctx.get_parameter_source(name)``, so this
    lets the recipe-target / precondition helpers be unit-tested without driving
    the whole command.
    """
    ctx = MagicMock(spec=click.Context)
    ctx.get_parameter_source.side_effect = lambda name: (
        click.core.ParameterSource.COMMANDLINE
        if name in provided
        else click.core.ParameterSource.DEFAULT
    )
    return ctx


def test_registered_family_npu_qnn_routes_to_bundle(tmp_path: Path):
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            ["-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--device", "npu", "--ep", "qnn"]
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


def test_explicit_config_file_is_rejected_for_bundle(tmp_path: Path):
    """A ``-c/--config`` file must error, not be silently discarded.

    The bundle is fully recipe-driven, so nothing in a supplied config would be
    honored.  The fast path rejects it (before reaching the orchestrator or the
    single-model pipeline).
    """
    cfg = _write_config_file(tmp_path, model_type="qwen3")

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle({})) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
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
            ]
        )

    assert result.exit_code != 0
    assert "-c/--config" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()


def test_explicit_no_quant_is_rejected_not_silently_ignored(tmp_path: Path):
    """A user-supplied pipeline control must error, not become a silent no-op.

    The genai bundle fixes quantization via its recipe, so ``--no-quant`` cannot
    be honored.  The fast path must reject it instead of quantizing anyway (and
    it must not reach the orchestrator or the single-model pipeline).
    """
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle({})) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "b"),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--no-quant",
            ]
        )

    assert result.exit_code != 0
    assert "--quant/--no-quant" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()


def test_submodel_is_rejected_not_silently_ignored(tmp_path: Path):
    """--submodel must error on the bundle path, not silently build the whole bundle.

    The genai bundle fixes every component via its recipe, so narrowing to one
    sub-model cannot be honored.  The fast path must reject it before reaching the
    orchestrator or the single-model pipeline (the later generic-path --submodel
    handling is never reached on this dispatch).
    """
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle({})) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "b"),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--submodel",
                "decoder",
            ]
        )

    assert result.exit_code != 0
    assert "--submodel" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()


def test_submodel_rejected_for_explicit_optimized_export_type(tmp_path: Path):
    """--submodel + --export-type optimized errors before building the bundle."""
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle({})) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "b"),
                "--export-type",
                "optimized",
                "--submodel",
                "decoder",
            ]
        )

    assert result.exit_code != 0
    assert "--submodel" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)),
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
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
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)),
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
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


def test_registered_family_auto_qnn_routes_to_bundle(tmp_path: Path):
    """``--device auto`` resolving to the NPU, with explicit ``--ep qnn``, routes."""
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            ["-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--device", "auto", "--ep", "qnn"]
        )

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1
    run_single.assert_not_called()
    kwargs = recorded["kwargs"]
    assert kwargs["ep"] == "qnn"
    assert kwargs["device"] == "npu"


def test_auto_qnn_resolving_to_non_npu_does_not_route(tmp_path: Path):
    """``--device auto`` that resolves to a non-NPU device keeps the stock path."""
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
        patch(_RESOLVE_DEVICE_TARGET, return_value=_CPU_TARGET),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "o"),
                "--device",
                "auto",
                "--ep",
                "qnn",
            ]
        )

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()
    run_single.assert_called_once()


def test_registered_family_qnn_without_device_routes_to_bundle(tmp_path: Path):
    """``--ep qnn`` with ``--device`` omitted must route just like ``--device auto``.

    Omitting ``--device`` falls back to the ``auto`` default; the shortcut must
    resolve it (here -> NPU) instead of treating a missing flag as "no NPU
    target". Regression test for ``--ep qnn`` alone silently building generic.
    """
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--ep", "qnn"])

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1
    run_single.assert_not_called()
    kwargs = recorded["kwargs"]
    assert kwargs["ep"] == "qnn"
    assert kwargs["device"] == "npu"


def test_qnn_without_device_resolving_to_non_npu_does_not_route(tmp_path: Path):
    """``--ep qnn`` with ``--device`` omitted still gates on the resolved device."""
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
        patch(_RESOLVE_DEVICE_TARGET, return_value=_CPU_TARGET),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "o"), "--ep", "qnn"])

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()
    run_single.assert_called_once()


def test_npu_without_explicit_ep_does_not_route(tmp_path: Path):
    """Auto-resolved QNN (no explicit --ep) must keep the stock path."""
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "o"), "--device", "npu"])

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()


def test_cpu_target_does_not_route(tmp_path: Path):
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
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
    with (
        patch(
            _GENERATE_TARGET,
            return_value=_fake_config("resnet", task="image-classification"),
        ),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
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
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "--use-cache", "--device", "npu", "--ep", "qnn"])

    assert result.exit_code != 0
    assert "output-dir" in result.output.lower()
    bundle.assert_not_called()


# ---------------------------------------------------------------------------
# --export-type GENERIC | OPTIMIZED (issue #1090)
# ---------------------------------------------------------------------------


def test_export_type_optimized_resolves_target_and_builds(tmp_path: Path):
    """``--export-type optimized`` resolves the host target, then builds its recipe.

    No ``--device``/``--ep`` is pinned, so the target is hardware-probed (here the
    autouse fixture resolves the NPU); the qwen3 -> qnn/npu recipe supports it, so
    the bundle is built for that resolved ``(ep, device)``.
    """
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--export-type", "optimized"])

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1
    run_single.assert_not_called()
    kwargs = recorded["kwargs"]
    assert kwargs["ep"] == "qnn"
    assert kwargs["device"] == "npu"


def test_export_type_optimized_errors_when_resolved_target_unsupported(tmp_path: Path):
    """Optimized resolves the host target first; a host the recipe can't serve errors.

    On a machine without the recipe's accelerator, auto-resolution lands on a
    non-NPU device, so the optimized bundle (qwen3 -> qnn/npu) is unavailable and
    the build fails fast naming the resolved ep/device. Regression test for the
    reviewer note on PR #1104 (resolve first, then error -- not infer-from-recipe).
    """
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
        patch(_RESOLVE_DEVICE_TARGET, return_value=_CPU_TARGET) as probe,
    ):
        result = _invoke(
            ["-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "o"), "--export-type", "optimized"]
        )

    assert result.exit_code != 0
    probe.assert_called_once()
    assert "not supported for" in result.output
    assert "device=cpu" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()


def test_export_type_optimized_explicit_target_builds_on_non_npu_host(tmp_path: Path):
    """Explicit ``--ep qnn --device npu`` builds even when the host has no NPU.

    Pinning the target means auto-resolution (modeled here as landing on CPU) is
    never consulted for the optimized selection, so the bundle can be produced on
    a machine that lacks the accelerator (e.g. CI).
    """
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
        patch(_RESOLVE_DEVICE_TARGET, return_value=_CPU_TARGET),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(out),
                "--export-type",
                "optimized",
                "--ep",
                "qnn",
                "--device",
                "npu",
            ]
        )

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1
    run_single.assert_not_called()
    kwargs = recorded["kwargs"]
    assert kwargs["ep"] == "qnn"
    assert kwargs["device"] == "npu"


def test_export_type_optimized_is_case_insensitive(tmp_path: Path):
    out = tmp_path / "bundle"
    recorded: dict = {}

    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET, side_effect=_record_bundle(recorded)) as bundle,
        patch(_RUN_SINGLE_TARGET),
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(["-m", "Qwen/Qwen3-0.6B", "-o", str(out), "--export-type", "OPTIMIZED"])

    assert result.exit_code == 0, result.output
    assert bundle.call_count == 1


def test_export_type_generic_forces_stock_even_on_npu_qnn(tmp_path: Path):
    """``--export-type generic`` keeps the stock build for a registered family.

    Even the (otherwise routing) ``--ep qnn`` + NPU combination must fall through
    to the single/composite pipeline when generic is explicitly requested.
    """
    with (
        patch(_GENERATE_TARGET, return_value=_fake_config("qwen3")),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "-o",
                str(tmp_path / "o"),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--export-type",
                "generic",
            ]
        )

    assert result.exit_code == 0, result.output
    bundle.assert_not_called()
    run_single.assert_called_once()


def test_export_type_optimized_unregistered_family_errors(tmp_path: Path):
    """Optimized on a family with no recipe fails fast."""
    with (
        patch(
            _GENERATE_TARGET,
            return_value=_fake_config("resnet", task="image-classification"),
        ),
        patch(_BUNDLE_TARGET) as bundle,
        patch(_RUN_SINGLE_TARGET) as run_single,
        patch(_COMPOSITE_TARGET, return_value=None),
    ):
        result = _invoke(
            [
                "-m",
                "microsoft/resnet-50",
                "-o",
                str(tmp_path / "o"),
                "--export-type",
                "optimized",
            ]
        )

    assert result.exit_code != 0
    assert "no optimized recipe" in result.output
    assert "resnet" in result.output
    bundle.assert_not_called()
    run_single.assert_not_called()


def test_resolve_optimized_target_matches_resolved_target():
    from winml.modelkit.commands.build import _resolve_optimized_target
    from winml.modelkit.models.winml import resolve_genai_bundle

    recipe = resolve_genai_bundle("qwen3")
    ep, device = _resolve_optimized_target(recipe, device="npu", ep="QNNExecutionProvider")
    assert (ep, device) == ("qnn", "npu")


def test_resolve_optimized_target_resolves_auto_device_for_pinned_ep():
    """A pinned ``--ep`` with ``--device auto`` resolves the device, then matches."""
    from winml.modelkit.commands.build import _resolve_optimized_target
    from winml.modelkit.models.winml import resolve_genai_bundle

    recipe = resolve_genai_bundle("qwen3")
    # The autouse hardware mock resolves auto -> npu.
    ep, device = _resolve_optimized_target(recipe, device="auto", ep="qnn")
    assert (ep, device) == ("qnn", "npu")


def test_resolve_optimized_target_rejects_unsupported_ep():
    from winml.modelkit.commands.build import _resolve_optimized_target
    from winml.modelkit.models.winml import resolve_genai_bundle

    recipe = resolve_genai_bundle("qwen3")
    with pytest.raises(click.UsageError, match="not supported for ep=dml, device=gpu"):
        _resolve_optimized_target(recipe, device="gpu", ep="dml")


def test_resolve_optimized_target_rejects_unsupported_device():
    from winml.modelkit.commands.build import _resolve_optimized_target
    from winml.modelkit.models.winml import resolve_genai_bundle

    recipe = resolve_genai_bundle("qwen3")
    with pytest.raises(click.UsageError, match="not supported for ep=cpu, device=cpu"):
        _resolve_optimized_target(recipe, device="cpu", ep="cpu")


def test_optimized_rejects_onnx_input():
    from winml.modelkit.commands.build import _maybe_build_genai_bundle

    with pytest.raises(click.UsageError, match=r"pre-exported \.onnx"):
        _maybe_build_genai_bundle(
            _fake_ctx({"export_type"}),
            export_type="optimized",
            model="model.onnx",
            model_is_onnx=True,
            config_or_configs=_fake_config("qwen3"),
            preloaded_hf_config=None,
            output_dir="out",
            use_cache=False,
            device="auto",
            ep=None,
            precision=None,
            rebuild=False,
            submodel=None,
        )


def test_optimized_rejects_module_mode():
    from winml.modelkit.commands.build import _maybe_build_genai_bundle

    with pytest.raises(click.UsageError, match="module mode"):
        _maybe_build_genai_bundle(
            _fake_ctx({"export_type"}),
            export_type="optimized",
            model="Qwen/Qwen3-0.6B",
            model_is_onnx=False,
            config_or_configs=[_fake_config("qwen3")],
            preloaded_hf_config=None,
            output_dir="out",
            use_cache=False,
            device="auto",
            ep=None,
            precision=None,
            rebuild=False,
            submodel=None,
        )


def test_optimized_requires_model():
    from winml.modelkit.commands.build import _maybe_build_genai_bundle

    with pytest.raises(click.UsageError, match="requires -m/--model"):
        _maybe_build_genai_bundle(
            _fake_ctx({"export_type"}),
            export_type="optimized",
            model=None,
            model_is_onnx=False,
            config_or_configs=_fake_config("qwen3"),
            preloaded_hf_config=None,
            output_dir="out",
            use_cache=False,
            device="auto",
            ep=None,
            precision=None,
            rebuild=False,
            submodel=None,
        )
