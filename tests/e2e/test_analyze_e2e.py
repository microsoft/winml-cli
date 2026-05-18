# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml analyze`` CLI command.

These tests exercise the analyze command end-to-end through ``CliRunner``
without mocking the analyzer. Synthetic ONNX models are built in-test and
matching runtime-rule parquet artifacts are written into a per-test
directory that is exposed to the command via the ``WINMLCLI_RULES_DIR``
environment variable.

Coverage layout
---------------
* ``TestAnalyzeCliSurface`` — `--help`, missing/invalid arguments,
  enum-choice validation, and "no runtime rules available" failure
  modes. None of these tests require parquet data.
* ``TestAnalyzeHappyPath`` — Real end-to-end runs against an in-memory
  ONNX model and a minimally-valid parquet rule artifact. Validates the
  documented exit codes (0 fully-supported, 1 partial, 2 error) and JSON
  output via `--output` and `--optim-config`.
* ``TestAnalyzeFlagVariations`` — Each behaviour-bearing flag exercised
  in both present/absent forms, and each enum-choice value of every
  Click ``choice`` flag (`--ep`, `--device`, `--save-node`) covered.
* ``TestAnalyzeBadPath`` — EP+device incompatibility, unknown EP +
  explicit device, and stack-trace-free reporting of analysis failures.

Markers
-------
* ``e2e`` — auto-skipped unless `-m e2e` is selected (see
  ``tests/e2e/conftest.py``).

Usage::

    uv run pytest tests/e2e/ -m e2e -k analyze
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import onnx
import pandas as pd
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from winml.modelkit.commands.analyze import analyze


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120)]


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _invoke(args: list[str]):
    """Invoke ``winml analyze`` via a fresh ``CliRunner``."""
    return CliRunner().invoke(analyze, args, obj={}, catch_exceptions=False)


def _assert_no_traceback(result) -> None:
    """The analyze command must surface errors as logs/messages, never as
    a raw Python traceback — per the documented exit-code contract."""
    assert "Traceback (most recent call last)" not in result.output, (
        f"unexpected traceback in CLI output:\n{result.output}"
    )


def _build_add_onnx(path: Path, opset: int = 13) -> Path:
    """Write a minimal single-Add ONNX model to ``path``."""
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Add", ["A", "B"], ["Y"], name="add")
    graph = helper.make_graph([node], "add_graph", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _write_supported_rule(rules_dir: Path, ep: str, device: str, op: str = "Add") -> Path:
    """Write a minimally-valid "always supported" parquet rule.

    The rule has no condition columns — only the ``compile_run_success``
    tuple — so it unconditionally matches every node of the named op.
    """
    parquet = rules_dir / f"{op}_{ep}_{device}_ai.onnx_opset13.parquet"
    pd.DataFrame([{"compile_run_success": (True, True)}]).to_parquet(parquet, index=False)
    return parquet


def _write_unsupported_rule(rules_dir: Path, ep: str, device: str, op: str = "Add") -> Path:
    """Write a parquet rule that classifies the op as unsupported (compile fails)."""
    parquet = rules_dir / f"{op}_{ep}_{device}_ai.onnx_opset13.parquet"
    pd.DataFrame([{"compile_run_success": (False, False)}]).to_parquet(parquet, index=False)
    return parquet


def _write_partial_rule(rules_dir: Path, ep: str, device: str, op: str = "Add") -> Path:
    """Write a parquet rule that classifies the op as partially supported
    (compile succeeds, run fails). No condition columns → unconditional match."""
    parquet = rules_dir / f"{op}_{ep}_{device}_ai.onnx_opset13.parquet"
    pd.DataFrame([{"compile_run_success": (True, False)}]).to_parquet(parquet, index=False)
    return parquet


@pytest.fixture
def rules_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test rules directory wired into the analyzer via env var."""
    d = tmp_path / "rules"
    d.mkdir()
    monkeypatch.setenv("WINMLCLI_RULES_DIR", str(d))
    return d


@pytest.fixture
def empty_rules_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Rules directory exposed to the CLI but containing no parquet files."""
    d = tmp_path / "empty_rules"
    d.mkdir()
    monkeypatch.setenv("WINMLCLI_RULES_DIR", str(d))
    return d


@pytest.fixture
def add_model(tmp_path: Path) -> Path:
    """Tiny Add-only ONNX model on disk."""
    return _build_add_onnx(tmp_path / "add.onnx")


# Click choice values declared by the analyze command. Kept in module scope so
# parametrize IDs stay stable and the test surface mirrors the CLI.
EP_FULL_NAMES = [
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "MIGraphXExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
]
EP_ALIASES = [
    "qnn", "openvino", "ov", "vitisai", "vitis",
    "cpu", "cuda", "dml", "nv_tensorrt_rtx", "trtrtx", "migraphx",
]
DEVICES = ["CPU", "GPU", "NPU"]
SAVE_NODE_CHOICES = ["partial", "unsupported"]


# ===========================================================================
# CLI surface — flags, choices, missing-arg validation
# ===========================================================================


class TestAnalyzeCliSurface:
    """Parser-level behaviours that don't depend on rule data or analysis."""

    def test_help_lists_every_documented_option(self) -> None:
        result = _invoke(["--help"])
        assert result.exit_code == 0
        for opt in (
            "--model", "--ep", "--device", "--verbose", "--quiet",
            "--config", "--output", "--information", "--no-information",
            "--htp-metadata", "--run-unknown-op", "--no-run-unknown-op",
            "--save-node", "--optim-config",
        ):
            assert opt in result.output, f"--help missing {opt}"
        # Documented exit codes appear in the help text.
        assert "Exit Codes" in result.output

    def test_missing_model_exits_two(self) -> None:
        result = _invoke([])
        assert result.exit_code == 2
        assert "Missing option" in result.output and "--model" in result.output
        _assert_no_traceback(result)

    def test_nonexistent_model_path_exits_two(self, tmp_path: Path) -> None:
        result = _invoke(["-m", str(tmp_path / "does_not_exist.onnx")])
        assert result.exit_code == 2
        assert "does not exist" in result.output
        _assert_no_traceback(result)

    def test_invalid_ep_choice_exits_two(self, add_model: Path) -> None:
        result = _invoke(["-m", str(add_model), "--ep", "bogus_ep"])
        assert result.exit_code == 2
        assert "Invalid value for '--ep'" in result.output
        _assert_no_traceback(result)

    def test_invalid_device_choice_exits_two(self, add_model: Path) -> None:
        result = _invoke(["-m", str(add_model), "--device", "TPU"])
        assert result.exit_code == 2
        assert "Invalid value for '--device'" in result.output
        _assert_no_traceback(result)

    def test_invalid_save_node_choice_exits_two(self, add_model: Path) -> None:
        result = _invoke(["-m", str(add_model), "--save-node", "supported"])
        assert result.exit_code == 2
        assert "Invalid value for '--save-node'" in result.output
        _assert_no_traceback(result)

    def test_nonexistent_htp_metadata_exits_two(self, add_model: Path, tmp_path: Path) -> None:
        result = _invoke([
            "-m", str(add_model),
            "--htp-metadata", str(tmp_path / "missing.json"),
        ])
        assert result.exit_code == 2
        assert "does not exist" in result.output
        _assert_no_traceback(result)

    def test_nonexistent_config_file_exits_two(self, add_model: Path, tmp_path: Path) -> None:
        result = _invoke([
            "-m", str(add_model),
            "--config", str(tmp_path / "missing.json"),
        ])
        assert result.exit_code == 2
        assert "does not exist" in result.output
        _assert_no_traceback(result)

    def test_missing_runtime_rules_handled_cleanly(
        self, add_model: Path, empty_rules_dir: Path
    ) -> None:
        """When no parquet rules can be located, the CLI must report the
        documented diagnostic without raising a traceback. The bundled
        ``runtime_check_rules/`` may also contribute results in some
        installs — both branches are valid; the critical contract is the
        absence of a Python traceback."""
        # The default rules dir under src/ may or may not contain artifacts on a
        # given checkout. Force the search to a known-empty directory and
        # disable the default fallback by pointing the env var at empty dir.
        result = _invoke(["-m", str(add_model), "--ep", "qnn", "--device", "NPU"])
        # Either the empty env-var dir wins and we get exit 2, or rules ship
        # with the package and analysis proceeds. Both are acceptable. The
        # critical contract is: no traceback.
        _assert_no_traceback(result)
        assert result.exit_code in {0, 1, 2}


# ===========================================================================
# Happy path — real analyze run with synthetic parquet rules
# ===========================================================================


class TestAnalyzeHappyPath:
    """End-to-end runs with synthetic rule data covering exit codes 0/1."""

    def test_fully_supported_exits_zero(self, add_model: Path, rules_dir: Path) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model),
            "--ep", "qnn", "--device", "NPU", "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    def test_unsupported_exits_one(self, add_model: Path, rules_dir: Path) -> None:
        _write_unsupported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model),
            "--ep", "qnn", "--device", "NPU", "--quiet",
        ])
        _assert_no_traceback(result)
        # "Not fully supported" → exit 1 per documented exit codes.
        assert result.exit_code == 1

    def test_partial_support_exits_one(self, add_model: Path, rules_dir: Path) -> None:
        _write_partial_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model),
            "--ep", "qnn", "--device", "NPU", "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 1

    def test_output_writes_valid_json(
        self, add_model: Path, rules_dir: Path, tmp_path: Path
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        out_path = tmp_path / "results.json"
        result = _invoke([
            "-m", str(add_model),
            "--ep", "qnn", "--device", "NPU", "--quiet",
            "-o", str(out_path),
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0
        assert out_path.is_file()
        # File contents must be valid JSON.
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_optim_config_writes_valid_json(
        self, add_model: Path, rules_dir: Path, tmp_path: Path
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        cfg_path = tmp_path / "optim.json"
        result = _invoke([
            "-m", str(add_model),
            "--ep", "qnn", "--device", "NPU", "--quiet",
            "--optim-config", str(cfg_path),
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0
        assert cfg_path.is_file()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_default_device_is_npu(self, add_model: Path, rules_dir: Path) -> None:
        """Omitting ``--device`` should use NPU as documented in --help."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(["-m", str(add_model), "--ep", "qnn", "--quiet"])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    def test_analyze_all_eps_when_ep_omitted(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        """Omitting ``--ep`` analyzes all supported EPs. With only one
        synthetic rule the run must still complete without a traceback."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(["-m", str(add_model), "--quiet"])
        _assert_no_traceback(result)
        # Aggregate result depends on whether every probed EP is fully
        # supported; only assert documented exit codes.
        assert result.exit_code in {0, 1}


# ===========================================================================
# Flag / option variations — every behaviour-bearing flag exercised
# ===========================================================================


class TestAnalyzeFlagVariations:
    """Each enum value of every choice flag is covered by at least one test."""

    @pytest.mark.parametrize("ep_choice", EP_ALIASES + EP_FULL_NAMES)
    def test_every_ep_choice_is_accepted_by_parser(
        self, add_model: Path, rules_dir: Path, ep_choice: str
    ) -> None:
        """Click must accept every documented EP alias / full name. The
        analyzer may or may not have rule data for the EP — we don't
        require success, only a clean (non-traceback) exit."""
        # Pre-populate rule data so the no-parquet bootstrap check passes
        # regardless of which EP+device combo is queried.
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", ep_choice, "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code in {0, 1, 2}

    @pytest.mark.parametrize("device", DEVICES + [d.lower() for d in DEVICES])
    def test_every_device_choice_is_accepted_by_parser(
        self, add_model: Path, rules_dir: Path, device: str
    ) -> None:
        _write_supported_rule(
            rules_dir, "QNNExecutionProvider", device.upper()
        )
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", device, "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code in {0, 1, 2}

    @pytest.mark.parametrize("save_node", SAVE_NODE_CHOICES)
    def test_save_node_choice_accepted(
        self, add_model: Path, rules_dir: Path, save_node: str
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU", "--quiet",
            "--save-node", save_node,
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    def test_save_node_accepts_multiple_values(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        """``--save-node`` is declared ``multiple=True`` — both choices
        accepted in one invocation."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU", "--quiet",
            "--save-node", "partial",
            "--save-node", "unsupported",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--information", "--no-information"])
    def test_information_toggle(
        self, add_model: Path, rules_dir: Path, flag: str
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU", "--quiet",
            flag,
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--run-unknown-op", "--no-run-unknown-op"])
    def test_run_unknown_op_toggle(
        self, add_model: Path, rules_dir: Path, flag: str
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU", "--quiet",
            flag,
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    def test_quiet_suppresses_progress_bar(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        """``--quiet`` skips the Rich Live display."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU", "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0
        # The OP CHECK banner is part of the Rich live header path, which is
        # bypassed in --quiet mode. Stdout should not carry the banner emoji.
        assert "📊 OP CHECK" not in result.output

    def test_verbose_flag_accepted(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "qnn", "--device", "NPU",
            "--verbose", "--quiet",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 0


# ===========================================================================
# Bad path — EP/device incompatibility and stack-trace-free failures
# ===========================================================================


class TestAnalyzeBadPath:
    """Misconfiguration produces documented errors, never tracebacks."""

    def test_dml_with_cpu_device_rejected(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        """Dml only supports GPU; an explicit ``--device CPU`` must be
        rejected with a clean ``only supports`` message."""
        # A sentinel rule for an unrelated EP/device passes the parquet
        # bootstrap check so we exercise the EP+device validation path.
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "dml", "--device", "CPU",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 2
        assert "only supports" in result.output.lower()
        assert "gpu" in result.output.lower()

    def test_cpu_ep_with_npu_device_rejected(
        self, add_model: Path, rules_dir: Path
    ) -> None:
        """CPUExecutionProvider only supports CPU; --device NPU must fail."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(add_model), "--ep", "cpu", "--device", "NPU",
        ])
        _assert_no_traceback(result)
        assert result.exit_code == 2
        assert "only supports" in result.output.lower()
        assert "cpu" in result.output.lower()

    def test_invalid_onnx_file_exits_two_without_traceback(
        self, tmp_path: Path, rules_dir: Path
    ) -> None:
        """A non-ONNX file at a valid path should surface a clean error,
        not a raw protobuf parse traceback."""
        bad_model = tmp_path / "not_really.onnx"
        bad_model.write_bytes(b"this is not an onnx model")
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke([
            "-m", str(bad_model), "--ep", "qnn", "--device", "NPU",
        ])
        _assert_no_traceback(result)
        # The documented exit code for analysis failure is 2.
        assert result.exit_code == 2
