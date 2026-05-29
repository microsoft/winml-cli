# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ``scripts/e2e_eval/run_eval.py``.

The script is not packaged, so we load it via ``importlib`` (same pattern
as ``TestBuildEvalResultEpField`` in ``test_eval.py``) and exercise the
small helpers that gate the ``--no-quant`` / ``--precision`` injection
for EPs run on the unquantized variant (currently VitisAI).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_run_eval():
    """Load scripts/e2e_eval/run_eval.py as a module."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "e2e_eval" / "run_eval.py"

    # run_eval.py does ``sys.path.insert(0, str(Path(__file__).parent))``
    # at import time so its sibling ``utils`` package resolves; mirror that
    # here in case the module is loaded before the script runs.
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    spec = importlib.util.spec_from_file_location("_e2e_run_eval", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_eval():
    return _load_run_eval()


class TestShouldSkipWinmlQuant:
    """Membership test for ``_should_skip_winml_quant``.

    Aliases (case-insensitive) and the canonical ``*ExecutionProvider`` form
    should both resolve via ``normalize_ep_name``.
    """

    @pytest.mark.parametrize(
        "ep",
        ["vitisai", "VitisAI", "VITISAI", "VitisAIExecutionProvider"],
    )
    def test_vitisai_skips_quant(self, run_eval, ep):
        assert run_eval._should_skip_winml_quant(ep) is True

    @pytest.mark.parametrize(
        "ep",
        [None, "", "cpu", "dml", "qnn", "QNNExecutionProvider", "DmlExecutionProvider"],
    )
    def test_other_eps_do_not_skip(self, run_eval, ep):
        assert run_eval._should_skip_winml_quant(ep) is False


class TestResolvePrecision:
    """Behaviour of ``_resolve_precision`` for the new ``ep`` arg."""

    def test_npu_default_unchanged(self, run_eval):
        assert run_eval._resolve_precision("npu", None) == "w8a16"
        assert run_eval._resolve_precision("npu", None, ep=None) == "w8a16"

    def test_cpu_default_unchanged(self, run_eval):
        assert run_eval._resolve_precision("cpu", None) is None
        assert run_eval._resolve_precision("gpu", None) is None

    def test_explicit_precision_takes_precedence_on_npu(self, run_eval):
        assert run_eval._resolve_precision("npu", "fp16") == "fp16"
        assert run_eval._resolve_precision("cpu", "fp16", ep="DmlExecutionProvider") == "fp16"

    def test_skip_quant_ep_drops_default(self, run_eval):
        assert run_eval._resolve_precision("npu", None, ep="vitisai") is None
        assert (
            run_eval._resolve_precision("npu", None, ep="VitisAIExecutionProvider") is None
        )

    def test_skip_quant_ep_drops_explicit_with_warning(self, run_eval, capsys):
        result = run_eval._resolve_precision("npu", "w8a8", ep="vitisai")
        assert result is None
        captured = capsys.readouterr()
        # Warning must mention the dropped value and the EP so the override
        # is visible in the log when an explicit per-model precision is set
        # for an EP that runs on the unquantized variant.
        assert "w8a8" in captured.out
        assert "vitisai" in captured.out


class TestRunBuildNoQuantInjection:
    """``_run_build`` must append ``--no-quant`` to both winml config and
    winml build invocations when the EP is in ``_EPS_SKIP_WINML_QUANT``.
    """

    @staticmethod
    def _make_entry(hf_id="microsoft/resnet-50", task="image-classification"):
        entry = MagicMock()
        entry.hf_id = hf_id
        entry.task = task
        entry.perf_args = []
        return entry

    @staticmethod
    def _make_config_proc(config_path: Path):
        return {
            "exit_code": 0,
            "stdout": f"Generated {config_path}",
            "stderr": "",
            "elapsed": 0.1,
            "command": "winml config ...",
        }

    @staticmethod
    def _make_build_proc():
        return {
            "exit_code": 0,
            "stdout": "Build cache: /tmp/x_model.onnx",
            "stderr": "",
            "elapsed": 0.1,
            "command": "winml build ...",
        }

    def _invoke(self, run_eval, ep, tmp_path):
        entry = self._make_entry()
        # _run_build composes config_path = model_dir / "build_config.json"
        # internally; pre-create it so the post-config glob fallback resolves
        # to a single sub-config and the build loop runs once.
        config_path = tmp_path / "build_config.json"
        config_path.write_text("{}")

        captured_args: list[list[str]] = []

        def fake_subprocess(args, _timeout):
            captured_args.append(list(args))
            if "config" in args:
                return self._make_config_proc(config_path)
            return self._make_build_proc()

        with patch.object(run_eval, "_run_subprocess", side_effect=fake_subprocess), patch.object(
            run_eval, "_extract_onnx_path", return_value=str(tmp_path / "model.onnx")
        ):
            run_eval._run_build(
                entry,
                "npu",
                None,
                300,
                tmp_path,
                ep=ep,
            )
        return captured_args

    def test_vitisai_injects_no_quant_into_both_config_and_build(self, run_eval, tmp_path):
        calls = self._invoke(run_eval, "vitisai", tmp_path)
        config_call = next(args for args in calls if "config" in args)
        build_call = next(args for args in calls if "build" in args)
        assert "--no-quant" in config_call
        assert "--no-quant" in build_call

    def test_other_ep_omits_no_quant(self, run_eval, tmp_path):
        calls = self._invoke(run_eval, "dml", tmp_path)
        assert all("--no-quant" not in args for args in calls)
