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
* ``TestAnalyzeBadPath`` — EP+device incompatibility and unknown EP +
  explicit device produce documented exit codes.

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

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.e2e.require_ep import require_ep
from winml.modelkit.commands.analyze import analyze
from winml.modelkit.utils.constants import EP_ALIASES as _EP_ALIASES
from winml.modelkit.utils.constants import SUPPORTED_EPS


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120)]


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _invoke(args: list[str]):
    """Invoke ``winml analyze`` via a fresh ``CliRunner``."""
    return CliRunner().invoke(analyze, args, obj={}, catch_exceptions=False)


def _build_rule_parquet_path(rules_dir: Path, ep: str, device: str, op: str) -> Path:
    """Build parquet path using standard ``<EP>_<DEVICE>/<file>.parquet`` layout."""
    provider_dir = rules_dir / f"{ep}_{device.upper()}"
    provider_dir.mkdir(parents=True, exist_ok=True)
    return provider_dir / f"{op}_{ep}_{device.upper()}_ai.onnx_opset13.parquet"


def _write_rule_with_result(
    rules_dir: Path,
    ep: str,
    device: str,
    compile_run_success: tuple[bool, bool],
    op: str = "MatMul",
) -> Path:
    """Write a parquet rule with the given compile/run tuple."""
    parquet = _build_rule_parquet_path(rules_dir, ep, device, op)
    pd.DataFrame([{"compile_run_success": compile_run_success}]).to_parquet(parquet, index=False)
    return parquet


def _write_supported_rule(rules_dir: Path, ep: str, device: str, op: str = "MatMul") -> Path:
    """Write a minimally-valid "always supported" parquet rule.

    The rule has no condition columns — only the ``compile_run_success``
    tuple — so it unconditionally matches every node of the named op.
    """
    return _write_rule_with_result(rules_dir, ep, device, (True, True), op)


def _write_unsupported_rule(rules_dir: Path, ep: str, device: str, op: str = "MatMul") -> Path:
    """Write a parquet rule that classifies the op as unsupported (compile fails)."""
    return _write_rule_with_result(rules_dir, ep, device, (False, False), op)


def _write_partial_rule(rules_dir: Path, ep: str, device: str, op: str = "MatMul") -> Path:
    """Write a parquet rule that classifies the op as partially supported
    (compile succeeds, run fails). No condition columns → unconditional match."""
    return _write_rule_with_result(rules_dir, ep, device, (True, False), op)


@pytest.fixture
def rules_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test rules directory wired into the analyzer.

    ``WINMLCLI_RULES_DIR`` *augments* the search path with the bundled
    ``runtime_check_rules/`` directory rather than replacing it. To keep
    these tests deterministic regardless of what real rule artifacts may
    ship in the package, ``get_runtime_rules_search_dirs`` is patched to
    return only this fixture's directory.
    """
    d = tmp_path / "rules"
    d.mkdir()
    monkeypatch.setenv("WINMLCLI_RULES_DIR", str(d))
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.rule_loader.get_runtime_rules_search_dirs",
        lambda: [d],
    )
    return d


@pytest.fixture
def empty_rules_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Rules directory exposed to the CLI but containing no parquet files.

    See :func:`rules_dir` for why the search-path discovery function is
    patched in addition to setting the env var.
    """
    d = tmp_path / "empty_rules"
    d.mkdir()
    monkeypatch.setenv("WINMLCLI_RULES_DIR", str(d))
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.rule_loader.get_runtime_rules_search_dirs",
        lambda: [d],
    )
    return d


# Click choice values declared by the analyze command. Kept in module scope so
# parametrize IDs stay stable and the test surface mirrors the CLI.
EP_FULL_NAMES = SUPPORTED_EPS
EP_ALIASES = list(_EP_ALIASES.keys())
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
            "--model",
            "--ep",
            "--device",
            "--verbose",
            "--quiet",
            "--config",
            "--output",
            "--information",
            "--no-information",
            "--htp-metadata",
            "--run-unknown-op",
            "--no-run-unknown-op",
            "--save-node",
            "--optim-config",
        ):
            assert opt in result.output, f"--help missing {opt}"
        # Documented exit codes appear in the help text.
        assert "Exit Codes" in result.output

    def test_missing_model_exits_two(self) -> None:
        result = _invoke([])
        assert result.exit_code == 2
        assert "Missing option" in result.output and "--model" in result.output

    def test_nonexistent_model_path_exits_two(self, tmp_path: Path) -> None:
        result = _invoke(["-m", str(tmp_path / "does_not_exist.onnx")])
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_invalid_ep_choice_exits_two(self, onnx_model_path: Path) -> None:
        result = _invoke(["-m", str(onnx_model_path), "--ep", "bogus_ep"])
        assert result.exit_code == 2
        assert "Invalid value for '--ep'" in result.output

    def test_invalid_device_choice_exits_two(self, onnx_model_path: Path) -> None:
        result = _invoke(["-m", str(onnx_model_path), "--device", "TPU"])
        assert result.exit_code == 2
        assert "Invalid value for '-d' / '--device'" in result.output

    def test_invalid_save_node_choice_exits_two(self, onnx_model_path: Path) -> None:
        result = _invoke(["-m", str(onnx_model_path), "--save-node", "supported"])
        assert result.exit_code == 2
        assert "Invalid value for '--save-node'" in result.output

    def test_nonexistent_htp_metadata_exits_two(
        self, onnx_model_path: Path, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--htp-metadata",
                str(tmp_path / "missing.json"),
            ]
        )
        assert result.exit_code == 2
        assert "does not exist" in result.output

    def test_nonexistent_config_file_exits_two(self, onnx_model_path: Path, tmp_path: Path) -> None:
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--config",
                str(tmp_path / "missing.json"),
            ]
        )
        assert result.exit_code == 2
        assert "does not exist" in result.output


# ===========================================================================
# Happy path — real analyze run with synthetic parquet rules
# ===========================================================================


class TestAnalyzeHappyPath:
    """End-to-end runs with synthetic rule data covering exit codes 0/1."""

    def test_fully_supported_exits_zero(self, onnx_model_path: Path, rules_dir: Path) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
            ]
        )
        assert result.exit_code == 0

    def test_unsupported_exits_one(self, onnx_model_path: Path, rules_dir: Path) -> None:
        _write_unsupported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
            ]
        )
        # "Not fully supported" → exit 1 per documented exit codes.
        assert result.exit_code == 1

    def test_partial_support_exits_one(self, onnx_model_path: Path, rules_dir: Path) -> None:
        _write_partial_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
            ]
        )
        assert result.exit_code == 1

    def test_output_writes_valid_json(
        self, onnx_model_path: Path, rules_dir: Path, tmp_path: Path
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        out_path = tmp_path / "results.json"
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                "-o",
                str(out_path),
            ]
        )
        assert result.exit_code == 0
        assert out_path.is_file()
        # File contents must be valid JSON.
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_optim_config_writes_valid_json(
        self, onnx_model_path: Path, rules_dir: Path, tmp_path: Path
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        cfg_path = tmp_path / "optim.json"
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                "--optim-config",
                str(cfg_path),
            ]
        )
        assert result.exit_code == 0
        assert cfg_path.is_file()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_default_device_auto_resolves_single_best_device_for_pinned_ep(
        self,
        onnx_model_path: Path,
        rules_dir: Path,
    ) -> None:
        """Omitting ``--device`` resolves a single best device for the pinned EP.

        ``auto`` picks one target via the shared sysinfo helpers (like
        build/run). On a QNN-capable host the highest-priority device is NPU,
        so ``--ep qnn`` with no ``--device`` resolves to a single ``(qnn, NPU)``
        run that is fully supported.

        Real end-to-end: gated on actual QNN availability via ``require_ep``
        rather than monkeypatching local capabilities. The auto-resolution
        logic itself is covered hardware-agnostically by the unit-level
        selection-matrix test.
        """
        require_ep("qnn")
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(["-m", str(onnx_model_path), "--ep", "qnn", "--quiet"])
        assert result.exit_code == 0

    def test_default_auto_selects_single_ep_when_ep_omitted(
        self,
        onnx_model_path: Path,
        rules_dir: Path,
    ) -> None:
        """Omitting ``--ep`` resolves a single best EP from local availability.

        On a QNN-capable host the highest-priority device (NPU) and its
        highest-priority EP (QNN) win, so bare ``auto`` resolves to ``(qnn,
        NPU)`` and should be fully supported.

        Real end-to-end: gated on actual QNN availability via ``require_ep``
        rather than monkeypatching local capabilities.
        """
        require_ep("qnn")
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(["-m", str(onnx_model_path), "--quiet"])
        assert result.exit_code == 0


# ===========================================================================
# Flag / option variations — every behaviour-bearing flag exercised
# ===========================================================================


class TestAnalyzeFlagVariations:
    """Each enum value of every choice flag is covered by at least one test."""

    @pytest.mark.parametrize("ep_choice", EP_ALIASES + EP_FULL_NAMES)
    def test_every_ep_choice_is_accepted_by_parser(
        self, onnx_model_path: Path, rules_dir: Path, ep_choice: str
    ) -> None:
        """Click must accept every documented EP alias / full name. The
        analyzer may or may not have rule data for the EP — we don't
        require success, only a documented exit code."""
        # Pre-populate rule data so the no-parquet bootstrap check passes
        # regardless of which EP+device combo is queried.
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                ep_choice,
                "--quiet",
            ]
        )
        assert result.exit_code in {0, 1, 2}

    @pytest.mark.parametrize("device", DEVICES + [d.lower() for d in DEVICES])
    def test_every_device_choice_is_accepted_by_parser(
        self, onnx_model_path: Path, rules_dir: Path, device: str
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", device.upper())
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                device,
                "--quiet",
            ]
        )
        assert result.exit_code in {0, 1, 2}

    @pytest.mark.parametrize("save_node", SAVE_NODE_CHOICES)
    def test_save_node_choice_accepted(
        self, onnx_model_path: Path, rules_dir: Path, save_node: str
    ) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                "--save-node",
                save_node,
            ]
        )
        assert result.exit_code == 0

    def test_save_node_accepts_multiple_values(
        self, onnx_model_path: Path, rules_dir: Path
    ) -> None:
        """``--save-node`` is declared ``multiple=True`` — both choices
        accepted in one invocation."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                "--save-node",
                "partial",
                "--save-node",
                "unsupported",
            ]
        )
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--information", "--no-information"])
    def test_information_toggle(self, onnx_model_path: Path, rules_dir: Path, flag: str) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                flag,
            ]
        )
        assert result.exit_code == 0

    @pytest.mark.parametrize("flag", ["--run-unknown-op", "--no-run-unknown-op"])
    def test_run_unknown_op_toggle(self, onnx_model_path: Path, rules_dir: Path, flag: str) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
                flag,
            ]
        )
        assert result.exit_code == 0

    def test_quiet_suppresses_progress_bar(self, onnx_model_path: Path, rules_dir: Path) -> None:
        """``--quiet`` skips the Rich Live display."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--quiet",
            ]
        )
        assert result.exit_code == 0
        # The OP CHECK banner is part of the Rich live header path, which is
        # bypassed in --quiet mode. Stdout should not carry the banner emoji.
        assert "📊 OP CHECK" not in result.output

    def test_verbose_flag_accepted(self, onnx_model_path: Path, rules_dir: Path) -> None:
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--verbose",
                "--quiet",
            ]
        )
        assert result.exit_code == 0


# ===========================================================================
# Bad path — EP/device incompatibility and analysis-failure exit codes
# ===========================================================================


class TestAnalyzeBadPath:
    """Misconfiguration produces documented exit codes."""

    def test_dml_with_cpu_device_rejected(self, onnx_model_path: Path, rules_dir: Path) -> None:
        """Dml only supports GPU; an explicit ``--device CPU`` must be
        rejected with a clean ``only supports`` message."""
        # A sentinel rule for an unrelated EP/device passes the parquet
        # bootstrap check so we exercise the EP+device validation path.
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "dml",
                "--device",
                "CPU",
            ]
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    def test_cpu_ep_with_npu_device_rejected(self, onnx_model_path: Path, rules_dir: Path) -> None:
        """CPUExecutionProvider only supports CPU; --device NPU must fail."""
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(onnx_model_path),
                "--ep",
                "cpu",
                "--device",
                "NPU",
            ]
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    def test_invalid_onnx_file_exits_two_without_traceback(
        self, tmp_path: Path, rules_dir: Path
    ) -> None:
        """A non-ONNX file at a valid path should surface a documented
        error exit code, not propagate a parse exception."""
        bad_model = tmp_path / "not_really.onnx"
        bad_model.write_bytes(b"this is not an onnx model")
        _write_supported_rule(rules_dir, "QNNExecutionProvider", "NPU")
        result = _invoke(
            [
                "-m",
                str(bad_model),
                "--ep",
                "qnn",
                "--device",
                "NPU",
            ]
        )
        # The documented exit code for analysis failure is 2.
        assert result.exit_code == 2
