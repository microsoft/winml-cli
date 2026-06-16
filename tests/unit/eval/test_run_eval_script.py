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
        assert run_eval._resolve_precision("npu", None, ep="VitisAIExecutionProvider") is None

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
        # Mirror the real ModelEntry default: no pre-exported ONNX, so _run_build
        # takes the HF-id path (a truthy MagicMock would trigger ONNX download).
        entry.onnx_file = None
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

        with (
            patch.object(run_eval, "_run_subprocess", side_effect=fake_subprocess),
            patch.object(run_eval, "_extract_onnx_path", return_value=str(tmp_path / "model.onnx")),
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


def _write_onnx(path: Path, opset: int) -> None:
    """Write a minimal single-node ONNX model at the given opset."""
    import onnx

    helper = onnx.helper
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.save(model, str(path))


class TestEnsureMinOpset:
    """``_ensure_min_opset`` upgrades sub-minimum ONNX to winml's opset floor."""

    def test_below_minimum_is_upgraded(self, run_eval, tmp_path):
        import onnx

        src = tmp_path / "low.onnx"
        _write_onnx(src, opset=11)  # below winml minimum (12)
        out = run_eval._ensure_min_opset(str(src), tmp_path)

        assert out != str(src)  # a new, upgraded file
        assert Path(out).exists()
        upgraded = max(
            i.version for i in onnx.load(out).opset_import if i.domain in ("", "ai.onnx")
        )
        assert upgraded == run_eval._OPSET_UPGRADE_TARGET

    def test_at_or_above_minimum_is_unchanged(self, run_eval, tmp_path):
        src = tmp_path / "ok.onnx"
        _write_onnx(src, opset=run_eval._WINML_MIN_OPSET)  # exactly the floor
        assert run_eval._ensure_min_opset(str(src), tmp_path) == str(src)


class TestOnnxFilePreBuiltModel:
    """Models declaring ``onnx_file`` download the pre-exported ONNX and feed the
    local path to winml config/build via ``-m``, building with ``--output-dir``
    (direct-ONNX configs have no ``loader.task`` for ``--use-cache``).
    """

    @staticmethod
    def _make_entry(onnx_file: str | None = "inference.onnx"):
        entry = MagicMock()
        entry.hf_id = "PaddlePaddle/PP-OCRv5_server_det_onnx"
        entry.task = "image-to-text"
        entry.perf_args = []
        entry.onnx_file = onnx_file
        return entry

    @staticmethod
    def _model_arg(args: list[str]) -> str:
        """Return the value after the *model* ``-m`` (WINML_CLI itself carries a
        leading ``python -m winml.modelkit.cli``, so take the last ``-m``)."""
        idx = len(args) - 1 - args[::-1].index("-m")
        return args[idx + 1]

    def test_no_onnx_file_returns_hf_id(self, run_eval, tmp_path):
        entry = self._make_entry(onnx_file=None)
        assert run_eval._resolve_model_input(entry, tmp_path) == entry.hf_id

    def test_onnx_file_downloads_then_ensures_opset(self, run_eval, tmp_path):
        entry = self._make_entry()
        fake_dl = str(tmp_path / "inference.onnx")
        with (
            patch.object(run_eval, "_ensure_min_opset", return_value="UPGRADED") as ensure,
            patch("huggingface_hub.hf_hub_download", return_value=fake_dl) as download,
        ):
            result = run_eval._resolve_model_input(entry, tmp_path)

        download.assert_called_once_with(repo_id=entry.hf_id, filename="inference.onnx")
        ensure.assert_called_once_with(fake_dl, tmp_path)
        assert result == "UPGRADED"

    def test_run_build_uses_output_dir_and_onnx_path(self, run_eval, tmp_path):
        entry = self._make_entry()
        onnx_path = str(tmp_path / "inference_op17.onnx")
        # winml build writes a deterministic <output-dir>/model.onnx
        build_out = tmp_path / "build"
        build_out.mkdir()
        (build_out / "model.onnx").write_text("x")
        config_path = tmp_path / "build_config.json"
        config_path.write_text("{}")

        captured: list[list[str]] = []

        def fake_subprocess(args, _timeout):
            captured.append(list(args))
            stdout = f"Generated {config_path}" if "config" in args else ""
            return {
                "exit_code": 0,
                "stdout": stdout,
                "stderr": "",
                "elapsed": 0.1,
                "command": "winml ...",
            }

        with (
            patch.object(run_eval, "_resolve_model_input", return_value=onnx_path),
            patch.object(run_eval, "_run_subprocess", side_effect=fake_subprocess),
        ):
            result = run_eval._run_build(entry, "cpu", None, 300, tmp_path, ep=None)

        config_call = next(a for a in captured if "config" in a)
        build_call = next(a for a in captured if "build" in a)
        # -m points at the local ONNX for both config and build
        assert self._model_arg(config_call) == onnx_path
        assert self._model_arg(build_call) == onnx_path
        # build writes to --output-dir, never --use-cache (no loader.task)
        assert "--output-dir" in build_call
        assert "--use-cache" not in build_call
        # artifact resolved deterministically (not via stdout parsing)
        assert result["success"] is True
        assert result["onnx_paths"][""] == str(build_out / "model.onnx")
