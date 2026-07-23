# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cross-command CLI option vs ``-c/--config`` JSON value-priority contract.

Every command that accepts both CLI options AND a ``-c/--config`` JSON file
must honor the following 4-tier priority for any field present in the
merge block of its command body:

    Tier 1: CLI option explicitly passed on the command line
    Tier 2: JSON key explicitly present in the ``-c`` file
    Tier 3: CLI option's ``default=`` value (from the click decorator)
    Tier 4: Dataclass field default (must NOT shadow Tier 3)

The Tier-4 leak is the historical bug: when a command reads
``WinMLBuildConfig.from_dict(json)`` and accesses a dataclass field whose
JSON section is empty or missing, the dataclass default is silently
substituted for the JSON value. The merge block then treats that default
as an explicit JSON value and overrides the CLI default.

The fix loads the raw JSON dict alongside the dataclass and checks for
key presence with ``"key" in raw_section`` so missing keys are
distinguishable from dataclass defaults.

This module verifies the 4-tier contract by:

1. Driving each command through ``click.testing.CliRunner`` with the
   downstream business logic mocked.
2. Capturing the effective config object (e.g. ``WinMLCompileConfig``,
   ``BenchmarkConfig``, ``WinMLQuantizationConfig``,
   ``WinMLEvaluationConfig``) right at the boundary where it leaves the
   command body and enters business logic.
3. Asserting the captured field value matches the expected effective
   value for each of the 4 tier-interaction scenarios.

Coverage is structured around *unique merge-logic paths*:

- ``compile``/``perf``/``analyze``/``export``/``quantize`` all use the
  same ``raw_cfg.get(section) or {}`` + ``is_cli_provided`` pattern.
  One representative field per command exercises this shared logic.
- ``eval`` has THREE structurally distinct paths inside
  ``_build_eval_config``: ``build_cfg.compile.ep_config.provider`` (the
  Tier-4-prone path), ``build_cfg.loader.task`` (a dataclass-attribute
  path that happens to be safe because ``task`` defaults to ``None``),
  and ``merge_config(cfg, raw.get("eval"))`` (raw-section merge). Each
  is covered by a dedicated field case.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# =============================================================================
# Shared helpers
# =============================================================================


def _write_json(tmp_path: Path, data: dict | None) -> Path | None:
    """Write *data* to a temp JSON file and return its path; ``None`` -> ``None``."""
    if data is None:
        return None
    p = tmp_path / "bc.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_fake_onnx(tmp_path: Path) -> Path:
    """Create a minimal placeholder file with .onnx extension.

    Commands only check for *file existence*; ONNX parsing happens in the
    downstream business logic which is mocked away by every adapter below.
    """
    p = tmp_path / "m.onnx"
    p.write_bytes(b"fake")
    return p


@dataclass(frozen=True)
class FieldCase:
    """Declarative spec for one ``(command, field)`` priority test case.

    Attributes:
        field: Logical name of the field; key into the dict returned by
            the per-command adapter.
        cli_flag: CLI flag including dashes (e.g. ``"--ep"``).
        cli_value: Value passed on the command line for the Tier-1 test.
        cli_value_effective: What ``field`` is expected to equal in the
            captured config when Tier 1 wins.
        json_section: Top-level JSON section name (e.g. ``"compile"``).
        json_key: Key inside the section (e.g. ``"execution_provider"``).
            May differ from the CLI flag (key remapping).
        json_value: Value placed in the JSON file for the Tier-2 test.
        json_value_effective: What ``field`` is expected to equal in the
            captured config when Tier 2 wins.
        cli_default_effective: What ``field`` is expected to equal when
            neither CLI nor JSON specifies the value (Tier 3).
    """

    field: str
    cli_flag: str
    cli_value: str
    cli_value_effective: Any
    json_section: str
    json_key: str
    json_value: Any
    json_value_effective: Any
    cli_default_effective: Any


# Adapter signature: (cli_args, json_dict, tmp_path) -> dict[str, Any]
AdapterFn = Callable[[list[str], dict | None, Path], dict[str, Any]]


# =============================================================================
# Tier-interaction checks (shared across all commands and fields)
# =============================================================================


def _check_t1_beats_t2(runner: AdapterFn, case: FieldCase, tmp_path: Path) -> None:
    """Tier 1 > Tier 2: explicit CLI option must beat JSON value."""
    eff = runner(
        [case.cli_flag, case.cli_value],
        {case.json_section: {case.json_key: case.json_value}},
        tmp_path,
    )
    assert eff[case.field] == case.cli_value_effective, (
        f"Tier-1 lost to Tier-2 for field={case.field!r}: "
        f"expected {case.cli_value_effective!r}, got {eff[case.field]!r}"
    )


def _check_t2_beats_t3(runner: AdapterFn, case: FieldCase, tmp_path: Path) -> None:
    """Tier 2 > Tier 3: JSON value must beat the CLI option default."""
    eff = runner(
        [],
        {case.json_section: {case.json_key: case.json_value}},
        tmp_path,
    )
    assert eff[case.field] == case.json_value_effective, (
        f"Tier-2 lost to Tier-3 for field={case.field!r}: "
        f"expected {case.json_value_effective!r}, got {eff[case.field]!r}"
    )


def _check_t3_not_shadowed_by_empty_section(
    runner: AdapterFn,
    case: FieldCase,
    tmp_path: Path,
) -> None:
    """Tier 3 > Tier 4: an EMPTY JSON section must not leak a dataclass default.

    JSON ``{section: {}}`` carries no explicit key for this field, so the
    CLI default must survive untouched. This is the original-bug regression
    guard: when ``WinMLBuildConfig.from_dict`` substituted dataclass defaults
    for missing keys, the merge block silently overrode the CLI default.
    """
    eff = runner(
        [],
        {case.json_section: {}},
        tmp_path,
    )
    assert eff[case.field] == case.cli_default_effective, (
        f"Tier-4 dataclass default leaked through empty {case.json_section!r} "
        f"section for field={case.field!r}: expected CLI default "
        f"{case.cli_default_effective!r}, got {eff[case.field]!r}"
    )


def _check_t3_not_shadowed_by_absent_section(
    runner: AdapterFn,
    case: FieldCase,
    tmp_path: Path,
) -> None:
    """Tier 3 > Tier 4: a JSON file *without* the relevant section must not
    leak a dataclass default for this field. The JSON contains an UNRELATED
    section so the file is valid but our target section is absent entirely.
    """
    # Pick any other valid section that's not ours.
    others = ["loader", "export", "optim", "quant", "compile", "eval"]
    other = next(s for s in others if s != case.json_section)
    eff = runner([], {other: {}}, tmp_path)
    assert eff[case.field] == case.cli_default_effective, (
        f"Tier-4 dataclass default leaked when {case.json_section!r} section "
        f"was absent (JSON only had {other!r}) for field={case.field!r}: "
        f"expected CLI default {case.cli_default_effective!r}, "
        f"got {eff[case.field]!r}"
    )


# =============================================================================
# Per-command adapters
# =============================================================================


def _run_compile(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml compile`` and capture ``WinMLCompileConfig`` fields.

    The compile flow normalizes the EP via ``WinMLCompileConfig.for_provider``
    which stores ``ep_config.provider`` in lowercase short form
    (e.g. ``"qnn"``, ``"openvino"``, ``"vitisai"``). The captured
    ``ep`` key reflects that lowercase short form.

    The compile command also calls ``resolve_device``/``resolve_eps`` and
    a registry-availability check; both are mocked so the test does not
    depend on the host's installed EPs.
    """
    from winml.modelkit.commands.compile import compile as compile_cmd

    model = _make_fake_onnx(tmp_path)
    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    def fake_compile_onnx(model_path, output_path=None, config=None, **_kw):
        captured["config"] = config
        r = MagicMock()
        r.success = True
        r.output_path = output_path or (tmp_path / "out.onnx")
        r.compile_time = 0.0
        r.total_time = 0.0
        r.errors = []
        return r

    # Mock all hardware-detection paths. resolve_device returns a fixed
    # device that matches every EP we exercise (npu supports qnn,
    # openvino, vitisai per EP_SUPPORTED_DEVICES).
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = True

    args = ["-m", str(model), "--device", "npu", *cli_args]
    if config_path is not None:
        args.extend(["--config", str(config_path)])

    with (
        patch(
            "winml.modelkit.commands.compile.is_compiled_onnx",
            return_value=False,
        ),
        patch(
            "winml.modelkit.commands.compile.resolve_device",
            return_value=("npu", None),
        ),
        patch(
            "winml.modelkit.commands.compile.resolve_eps",
            return_value=["QNNExecutionProvider"],
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
        patch(
            "winml.modelkit.compiler.compile_onnx",
            side_effect=fake_compile_onnx,
        ),
    ):
        r = CliRunner().invoke(compile_cmd, args, obj={}, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    cfg = captured["config"]
    return {"ep": cfg.ep_config.provider}


def _run_perf(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml perf`` and capture ``BenchmarkConfig``.

    Both HF and ONNX inputs now flow through ``PerfBenchmark`` (see PR #596
    ``fix(perf): unify HF and ONNX paths``), so this adapter patches the
    class itself and captures the ``BenchmarkConfig`` from its constructor.
    """
    from winml.modelkit.commands.perf import perf as perf_cmd

    model = _make_fake_onnx(tmp_path)
    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    def fake_benchmark(config):
        captured["config"] = config
        instance = MagicMock()
        instance.run.return_value = MagicMock()
        captured["instance"] = instance
        return instance

    args = ["-m", str(model), *cli_args]
    if config_path is not None:
        args.extend(["--config", str(config_path)])

    with (
        patch(
            "winml.modelkit.commands.perf.PerfBenchmark",
            side_effect=fake_benchmark,
        ),
        patch("winml.modelkit.commands.perf.display_console_report"),
        patch("winml.modelkit.commands.perf.write_json_report"),
        patch(
            "winml.modelkit.commands.perf.generate_output_path",
            return_value=tmp_path / "out.json",
        ),
    ):
        r = CliRunner().invoke(perf_cmd, args, obj={}, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    # Guard against the perf command short-circuiting before the benchmark
    # runs (e.g. an early return after constructing PerfBenchmark): the
    # captured BenchmarkConfig would still let the priority assertions pass
    # but the command wouldn't be exercising the real flow.
    assert captured["instance"].run.call_count == 1, (
        "PerfBenchmark was constructed but .run() was never invoked"
    )

    cfg = captured["config"]
    return {"ep": cfg.ep, "skip_build": cfg.skip_build}


def _run_analyze(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml analyze`` and capture ``ep`` from the first ``analyzer.analyze`` call.

    Analyze normalizes ``ep`` to its canonical name (e.g. ``"openvino"`` ->
    ``"OpenVINOExecutionProvider"``) before invoking ``analyzer.analyze``,
    so the captured value is the canonical EP. ``--device`` is pinned to
    ``npu`` because every EP used in this suite (qnn/openvino/vitisai)
    supports npu per ``EP_SUPPORTED_DEVICES``.
    """
    from winml.modelkit.commands.analyze import analyze as analyze_cmd

    model = _make_fake_onnx(tmp_path)
    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    mock_result = MagicMock()
    mock_result.is_fully_supported.return_value = True
    mock_result.output.results = []

    def fake_analyze(**kw):
        # Capture only the first call's ep (single execution_pair per test).
        captured.setdefault("ep", kw.get("ep"))
        return mock_result

    args = ["-m", str(model), "--device", "npu", "--quiet", *cli_args]
    if config_path is not None:
        args.extend(["--config", str(config_path)])

    with (
        patch(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            return_value=True,
        ),
        patch(
            "winml.modelkit.analyze.utils.ep_utils.has_any_rule_data",
            return_value=True,
        ),
        # Deterministic Tier-3 default: when ep stays "auto" through the merge
        # block, analyze resolves it via resolve_eps(resolved_device)[0]. Pin the
        # ORT device->EP map so npu -> QNN, fixing the resolved target EP.
        patch(
            "winml.modelkit.sysinfo.device._get_device_ep_map_from_ort",
            return_value={"npu": ("QNNExecutionProvider",)},
        ),
        patch(
            "winml.modelkit.sysinfo.device._get_available_eps",
            return_value=["QNNExecutionProvider"],
        ),
        patch(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            return_value=[("QNNExecutionProvider", "npu")],
        ),
        patch("winml.modelkit.analyze.ONNXStaticAnalyzer") as mock_analyzer_cls,
    ):
        mock_inst = MagicMock()
        mock_inst.analyze.side_effect = fake_analyze
        mock_analyzer_cls.return_value = mock_inst

        r = CliRunner().invoke(analyze_cmd, args, obj={}, catch_exceptions=False)

    # analyze returns 0 (fully supported) or 1 (partial). Either means the
    # command body executed past the merge block.
    assert r.exit_code in (0, 1), r.output
    return {"ep": captured.get("ep")}


def _run_export(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml export`` and capture ``task`` passed to ``load_hf_model``."""
    from winml.modelkit.commands.export import export as export_cmd

    out = tmp_path / "out.onnx"
    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    def fake_load_hf(model_id, task=None, **_kw):
        captured["task"] = task
        return (MagicMock(), MagicMock(), task or "detected-task")

    def fake_resolve(model_id, task=None, shape_config=None):
        return (MagicMock(input_tensors=None, output_tensors=None), None)

    def fake_export(**kw):
        captured["export_config"] = kw.get("export_config")
        return MagicMock()

    args = ["-m", "prajjwal1/bert-tiny", "-o", str(out), *cli_args]
    if config_path is not None:
        args.extend(["-c", str(config_path)])

    with (
        patch("winml.modelkit.loader.load_hf_model", side_effect=fake_load_hf),
        patch(
            "winml.modelkit.export.resolve_export_config",
            side_effect=fake_resolve,
        ),
        patch(
            "winml.modelkit.export.export_pytorch",
            side_effect=fake_export,
        ),
    ):
        r = CliRunner().invoke(
            export_cmd,
            args,
            obj={"debug": False},
            catch_exceptions=False,
        )
    assert r.exit_code == 0, r.output

    export_config = captured.get("export_config")
    return {
        "task": captured.get("task"),
        "enable_hierarchy_tags": export_config.enable_hierarchy_tags
        if export_config is not None
        else None,
        "dynamo": export_config.dynamo if export_config is not None else None,
    }


def _run_quantize(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml quantize`` and capture ``WinMLQuantizationConfig`` fields."""
    from winml.modelkit.commands.quantize import quantize as quantize_cmd

    model = _make_fake_onnx(tmp_path)
    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    def fake_quantize(model_path, output_path=None, config=None, **_kw):
        captured["config"] = config
        r = MagicMock()
        r.success = True
        r.output_path = output_path or (tmp_path / "out_qdq.onnx")
        r.nodes_quantized = 0
        r.total_time_seconds = 0.0
        r.errors = []
        return r

    args = ["-m", str(model), *cli_args]
    if config_path is not None:
        args.extend(["--config", str(config_path)])

    with patch("winml.modelkit.quant.quantize_onnx", side_effect=fake_quantize):
        r = CliRunner().invoke(quantize_cmd, args, obj={}, catch_exceptions=False)
    assert r.exit_code == 0, r.output

    cfg = captured["config"]
    return {"samples": cfg.samples}


def _run_eval(
    cli_args: list[str],
    json_dict: dict | None,
    tmp_path: Path,
) -> dict[str, Any]:
    """Drive ``winml eval`` and capture ``WinMLEvaluationConfig`` fields.

    Eval has three structurally distinct merge paths exercised by the
    field cases below: ``build_cfg.compile.ep_config.provider`` (for
    ``ep``), ``build_cfg.loader.task`` (for ``task``), and
    ``merge_config(cfg, raw.get("eval"))`` (for ``device``).
    """
    from winml.modelkit.commands.eval import eval as eval_cmd

    config_path = _write_json(tmp_path, json_dict)

    captured: dict[str, Any] = {}

    def fake_evaluate(cfg):
        captured["cfg"] = cfg
        result = MagicMock()
        result.config = cfg
        result.metrics = {"accuracy": 1.0}
        result.to_dict.return_value = {
            "metrics": result.metrics,
            "config": cfg.to_dict(),
        }
        return result

    args = ["-m", "microsoft/resnet-50", *cli_args]
    if config_path is not None:
        args.extend(["--config", str(config_path)])

    with (
        patch("winml.modelkit.eval.evaluate", side_effect=fake_evaluate),
        patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
        patch(
            "winml.modelkit.commands.eval._write_and_display",
            return_value=None,
        ),
    ):
        r = CliRunner().invoke(
            eval_cmd,
            args,
            obj={"debug": False},
            catch_exceptions=False,
        )
    assert r.exit_code == 0, r.output

    cfg = captured["cfg"]
    return {
        "ep": cfg.ep,
        "task": cfg.task,
        "device": cfg.device,
        "skip_build": cfg.skip_build,
        "dataset_samples": cfg.dataset.samples,
        "dataset_name": cfg.dataset.name,
    }


# =============================================================================
# Field cases
# =============================================================================


# ``compile`` -- representative field: ``ep`` (5 merge-block fields share
# identical raw_cfg.get("compile") + is_cli_provided logic; ``ep`` exercises
# the JSON-key remap ``ep`` <-> ``execution_provider`` and the original
# Tier-4 leak via ``EPConfig.provider`` default ``"qnn"``).
COMPILE_CASES = [
    FieldCase(
        field="ep",
        cli_flag="--ep",
        cli_value="openvino",
        cli_value_effective="openvino",
        json_section="compile",
        json_key="execution_provider",
        json_value="vitisai",
        json_value_effective="vitisai",
        # When neither CLI nor JSON specifies ep, ``_resolve_compile_provider``
        # falls back to ``resolve_eps(resolved_device)[0]`` -- mocked to
        # ``"QNNExecutionProvider"`` which ``for_provider`` stores as
        # ``ep_config.provider == "qnn"``.
        cli_default_effective="qnn",
    ),
]


# ``perf`` -- representative field: ``ep`` (both perf merge-block fields
# share identical logic; ``ep`` exercises the JSON-key remap).
PERF_CASES = [
    FieldCase(
        field="ep",
        cli_flag="--ep",
        cli_value="openvino",
        cli_value_effective="openvino",
        json_section="compile",
        json_key="execution_provider",
        json_value="vitisai",
        json_value_effective="vitisai",
        cli_default_effective=None,  # CLI default for --ep is None.
    ),
]


# ``analyze`` -- only field in merge block.
ANALYZE_CASES = [
    FieldCase(
        field="ep",
        cli_flag="--ep",
        cli_value="openvino",
        cli_value_effective="OpenVINOExecutionProvider",
        json_section="compile",
        json_key="execution_provider",
        json_value="vitisai",
        json_value_effective="VitisAIExecutionProvider",
        # CLI default is ``"auto"``, expanded to ``_get_available_eps()`` which
        # the adapter mocks to ``["QNNExecutionProvider"]``.
        cli_default_effective="QNNExecutionProvider",
    ),
]


# ``export`` -- representative field: ``task`` (all 3 export merge-block
# fields share identical logic; ``task`` exercises a different JSON
# section (``loader``) than the field under test belongs to in the CLI).
EXPORT_CASES = [
    FieldCase(
        field="task",
        cli_flag="--task",
        cli_value="fill-mask",
        cli_value_effective="fill-mask",
        json_section="loader",
        json_key="task",
        json_value="image-classification",
        json_value_effective="image-classification",
        cli_default_effective=None,
    ),
]


# ``quantize`` -- representative field: ``samples`` (all 8 quantize
# merge-block fields share identical logic; ``samples`` is the simplest
# (int, no JSON-key remap) and is the original-bug context).
QUANTIZE_CASES = [
    FieldCase(
        field="samples",
        cli_flag="--samples",
        cli_value="42",
        cli_value_effective=42,
        json_section="quant",
        json_key="samples",
        json_value=7,
        json_value_effective=7,
        # WinMLQuantizationConfig.samples default is 10 (CLI default also 10).
        cli_default_effective=10,
    ),
]


# ``eval`` -- three distinct merge-logic paths require three field cases.
EVAL_CASES = [
    # Path 1: ``cfg.ep = build_cfg.compile.ep_config.provider`` (dataclass
    # attribute access). When JSON has ``{"compile": {}}``, ``build_cfg.compile``
    # is a default WinMLCompileConfig and ``ep_config.provider`` is "qnn",
    # which would leak through Tier 4 unless the merge logic checks raw JSON.
    FieldCase(
        field="ep",
        cli_flag="--ep",
        cli_value="openvino",
        cli_value_effective="openvino",
        json_section="compile",
        json_key="execution_provider",
        json_value="vitisai",
        json_value_effective="vitisai",
        cli_default_effective=None,
    ),
    # Path 2: ``cfg.task = build_cfg.loader.task`` (dataclass attribute
    # access). ``loader.task`` defaults to ``None``, so this path is
    # structurally safe; included to pin contract.
    FieldCase(
        field="task",
        cli_flag="--task",
        cli_value="fill-mask",
        cli_value_effective="fill-mask",
        json_section="loader",
        json_key="task",
        json_value="image-classification",
        json_value_effective="image-classification",
        cli_default_effective=None,
    ),
    # Path 3: ``merge_config(cfg, raw.get("eval"))`` (raw-section merge).
    # Empty ``{"eval": {}}`` should be a no-op because ``if eval_data:`` is
    # False; included to pin contract for the raw-section merge path.
    FieldCase(
        field="device",
        cli_flag="--device",
        cli_value="cpu",
        cli_value_effective="cpu",
        json_section="eval",
        json_key="device",
        json_value="gpu",
        json_value_effective="gpu",
        # CLI default for --device is ``"auto"``.
        cli_default_effective="auto",
    ),
]


# =============================================================================
# Per-command test classes
#
# Each class runs the 4 tier-interaction checks for every field case of its
# command. The check helpers and the per-command adapters together absorb
# all the duplication, so each test method is a one-liner.
# =============================================================================


class TestCompilePriority:
    """``winml compile`` priority contract."""

    @pytest.mark.parametrize("case", COMPILE_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_compile, case, tmp_path)

    @pytest.mark.parametrize("case", COMPILE_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_compile, case, tmp_path)

    @pytest.mark.parametrize("case", COMPILE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_compile, case, tmp_path)

    @pytest.mark.parametrize("case", COMPILE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_compile, case, tmp_path)


class TestPerfPriority:
    """``winml perf`` priority contract."""

    @pytest.mark.parametrize("case", PERF_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_perf, case, tmp_path)

    @pytest.mark.parametrize("case", PERF_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_perf, case, tmp_path)

    @pytest.mark.parametrize("case", PERF_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_perf, case, tmp_path)

    @pytest.mark.parametrize("case", PERF_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_perf, case, tmp_path)

    # ------------------------------------------------------------------
    # Targeted tests for the ``--skip-build/--no-skip-build`` toggle.
    # Unlike ``ep``, ``skip_build`` has NO JSON config source: perf's merge
    # block (perf.py) only reads ``task`` and ``execution_provider`` from
    # the ``-c`` file, so ``skip_build`` flows CLI -> BenchmarkConfig
    # directly with no Tier-2 path. That, plus the boolean flag not fitting
    # the FieldCase ``[flag, value]`` shape, is why it's tested explicitly
    # here rather than as a PERF_CASES FieldCase. These guard against a
    # param-name mismatch in the ``perf()`` signature and against the CLI
    # option default drifting from the BenchmarkConfig field default.
    # ------------------------------------------------------------------

    def test_skip_build_default_is_true(self, tmp_path: Path) -> None:
        """No flag -> cfg.skip_build keeps the True default."""
        eff = _run_perf([], None, tmp_path)
        assert eff["skip_build"] is True

    def test_no_skip_build_flag_sets_false(self, tmp_path: Path) -> None:
        """``--no-skip-build`` -> cfg.skip_build is False."""
        eff = _run_perf(["--no-skip-build"], None, tmp_path)
        assert eff["skip_build"] is False

    def test_skip_build_flag_sets_true(self, tmp_path: Path) -> None:
        """``--skip-build`` -> cfg.skip_build is True."""
        eff = _run_perf(["--skip-build"], None, tmp_path)
        assert eff["skip_build"] is True


class TestAnalyzePriority:
    """``winml analyze`` priority contract."""

    @pytest.mark.parametrize("case", ANALYZE_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_analyze, case, tmp_path)

    @pytest.mark.parametrize("case", ANALYZE_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_analyze, case, tmp_path)

    @pytest.mark.parametrize("case", ANALYZE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_analyze, case, tmp_path)

    @pytest.mark.parametrize("case", ANALYZE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_analyze, case, tmp_path)


class TestExportPriority:
    """``winml export`` priority contract."""

    @pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_export, case, tmp_path)

    @pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_export, case, tmp_path)

    @pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_export, case, tmp_path)

    @pytest.mark.parametrize("case", EXPORT_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_export, case, tmp_path)

    # ------------------------------------------------------------------
    # Targeted tests for export-section fields (enable_hierarchy_tags,
    # dynamo). These fields are flow through the ``_build_export_dict``
    # layer (raw JSON ``export`` section), which is structurally distinct
    # from the in-line merge block exercised by the ``task`` FieldCase
    # above. Boolean flags (``--no-hierarchy``, ``--dynamo``) don't fit
    # the standard FieldCase ``[flag, value]`` shape, so they're tested
    # explicitly here.
    # ------------------------------------------------------------------

    def test_json_export_enable_hierarchy_tags_false_applies(self, tmp_path: Path) -> None:
        """JSON ``{"export": {"enable_hierarchy_tags": false}}`` reaches cfg."""
        eff = _run_export([], {"export": {"enable_hierarchy_tags": False}}, tmp_path)
        assert eff["enable_hierarchy_tags"] is False

    def test_empty_export_section_keeps_enable_hierarchy_tags_cli_default(
        self, tmp_path: Path
    ) -> None:
        """JSON ``{"export": {}}`` must NOT change ``enable_hierarchy_tags``.

        Guards the fix that switched export.py's Layer-1 source from
        ``_build_export_cfg.to_dict()`` (which would emit every dataclass
        field) to ``_build_export_dict`` (raw JSON keys only).
        """
        eff = _run_export([], {"export": {}}, tmp_path)
        assert eff["enable_hierarchy_tags"] is True  # CLI default

    def test_json_export_dynamo_true_applies(self, tmp_path: Path) -> None:
        """JSON ``{"export": {"dynamo": true}}`` reaches cfg."""
        eff = _run_export([], {"export": {"dynamo": True}}, tmp_path)
        assert eff["dynamo"] is True

    def test_empty_export_section_keeps_dynamo_cli_default(self, tmp_path: Path) -> None:
        """JSON ``{"export": {}}`` must NOT change ``dynamo``.

        Guards the same fix as ``enable_hierarchy_tags`` above.
        """
        eff = _run_export([], {"export": {}}, tmp_path)
        assert eff["dynamo"] is False  # CLI and shared config default


class TestQuantizePriority:
    """``winml quantize`` priority contract."""

    @pytest.mark.parametrize("case", QUANTIZE_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_quantize, case, tmp_path)

    @pytest.mark.parametrize("case", QUANTIZE_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_quantize, case, tmp_path)

    @pytest.mark.parametrize("case", QUANTIZE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_quantize, case, tmp_path)

    @pytest.mark.parametrize("case", QUANTIZE_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_quantize, case, tmp_path)


class TestEvalPriority:
    """``winml eval`` priority contract.

    Three field cases exercise three structurally distinct merge paths in
    ``_build_eval_config``. The ``ep`` case is the most bug-prone because
    its merge path accesses a dataclass attribute (``build_cfg.compile.
    ep_config.provider``) whose default leaks when JSON has only an empty
    ``compile`` section.
    """

    @pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.field)
    def test_t1_beats_t2(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t1_beats_t2(_run_eval, case, tmp_path)

    @pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.field)
    def test_t2_beats_t3(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t2_beats_t3(_run_eval, case, tmp_path)

    @pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_empty_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_empty_section(_run_eval, case, tmp_path)

    @pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.field)
    def test_t3_not_shadowed_by_absent_section(self, case: FieldCase, tmp_path: Path) -> None:
        _check_t3_not_shadowed_by_absent_section(_run_eval, case, tmp_path)

    # ------------------------------------------------------------------
    # Targeted tests for the ``--skip-build/--no-skip-build`` toggle.
    # Unlike perf's ``skip_build`` (no JSON source -> CLI-only), eval's
    # ``skip_build`` has a full Tier-2 path: ``{"eval": {"skip_build": ...}}``
    # flows through ``merge_config(cfg, eval_data)`` in ``_build_eval_config``,
    # and the CLI layer (``collect_cli_overrides``) overrides it. The boolean
    # flag doesn't fit the FieldCase ``[flag, value]`` shape, so the full
    # CLI > config-file > default contract is verified explicitly here.
    # ------------------------------------------------------------------

    def test_skip_build_default_is_true(self, tmp_path: Path) -> None:
        """Tier 3: no flag, no JSON -> cfg.skip_build keeps the True default."""
        assert _run_eval([], None, tmp_path)["skip_build"] is True

    def test_no_skip_build_flag_sets_false(self, tmp_path: Path) -> None:
        """Tier 1: ``--no-skip-build`` -> cfg.skip_build is False."""
        assert _run_eval(["--no-skip-build"], None, tmp_path)["skip_build"] is False

    def test_json_skip_build_false_applies(self, tmp_path: Path) -> None:
        """Tier 2: config file ``{"eval": {"skip_build": false}}`` must take effect."""
        eff = _run_eval([], {"eval": {"skip_build": False}}, tmp_path)
        assert eff["skip_build"] is False

    def test_cli_beats_json_skip_build(self, tmp_path: Path) -> None:
        """Tier 1 > Tier 2: explicit CLI ``--skip-build`` must win over JSON False."""
        eff = _run_eval(["--skip-build"], {"eval": {"skip_build": False}}, tmp_path)
        assert eff["skip_build"] is True

    def test_empty_quant_section_does_not_leak_to_dataset_samples(self, tmp_path: Path) -> None:
        """JSON ``{"quant": {}}`` must NOT change ``cfg.dataset.samples``.

        Pins Bug 2a: today ``WinMLQuantizationConfig.samples`` default ``10``
        leaks through the empty ``quant`` section and overrides the CLI
        default ``100``.
        """
        eff = _run_eval([], {"quant": {}}, tmp_path)
        assert eff["dataset_samples"] == 100, (
            f"Empty quant section leaked into eval dataset.samples: "
            f"expected CLI default 100, got {eff['dataset_samples']!r}"
        )

    def test_explicit_quant_samples_does_not_leak_to_dataset_samples(self, tmp_path: Path) -> None:
        """JSON ``{"quant": {"samples": 50}}`` must NOT change
        ``cfg.dataset.samples``.

        Pins Bug 2: even an explicit ``quant.samples`` value is a
        calibration knob and must not bleed into the eval dataset config.
        Users wanting to set eval samples must use
        ``{"eval": {"dataset": {"samples": ...}}}``.
        """
        eff = _run_eval([], {"quant": {"samples": 50}}, tmp_path)
        assert eff["dataset_samples"] == 100, (
            f"Explicit quant.samples leaked into eval dataset.samples: "
            f"expected CLI default 100, got {eff['dataset_samples']!r}"
        )

    def test_explicit_quant_dataset_name_does_not_leak_to_dataset_name(
        self, tmp_path: Path
    ) -> None:
        """JSON ``{"quant": {"dataset_name": "mrpc"}}`` must NOT change
        ``cfg.dataset.name``.

        Pins Bug 2: ``quant.dataset_name`` is the calibration dataset and
        must not bleed into the eval dataset config. Users wanting to set
        the eval dataset name must use
        ``{"eval": {"dataset": {"name": ...}}}``.
        """
        eff = _run_eval([], {"quant": {"dataset_name": "mrpc"}}, tmp_path)
        assert eff["dataset_name"] is None, (
            f"Explicit quant.dataset_name leaked into eval dataset.name: "
            f"expected CLI default None, got {eff['dataset_name']!r}"
        )
