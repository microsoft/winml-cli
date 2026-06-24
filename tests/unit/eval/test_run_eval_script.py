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

import argparse
import importlib.util
import json
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


class TestFeedVersionForCombo:
    """``_feed_version_for`` embeds the EP/device combo after the run-stamp."""

    @staticmethod
    def _entry(hf_id="microsoft/resnet-50", task="image-classification"):
        entry = MagicMock()
        entry.hf_id = hf_id
        entry.task = task
        return entry

    def test_combo_label_slugified_after_run_stamp(self, run_eval):
        version = run_eval._feed_version_for(self._entry(), "20260609", "qnn_npu")
        assert version == "0.0.0-20260609-qnn-npu-microsoft-resnet-50-image-classification"

    def test_distinct_combos_yield_distinct_versions(self, run_eval):
        entry = self._entry()
        v1 = run_eval._feed_version_for(entry, "20260609", "qnn_npu")
        v2 = run_eval._feed_version_for(entry, "20260609", "ov_cpu")
        assert v1 != v2
        assert "-qnn-npu-" in v1
        assert "-ov-cpu-" in v2

    def test_task_omitted_when_absent(self, run_eval):
        version = run_eval._feed_version_for(self._entry(task=""), "20260609", "qnn_npu")
        assert version == "0.0.0-20260609-qnn-npu-microsoft-resnet-50"


class TestResultsIO:
    """``_load_results`` / ``_write_results`` round-trip and tolerate junk."""

    def test_round_trip(self, run_eval, tmp_path):
        path = tmp_path / "build_only_results.json"
        data = {"0.0.0-x": {"build_status": "ok", "upload_status": "uploaded"}}
        run_eval._write_results(path, data)
        assert run_eval._load_results(path) == data

    def test_missing_file_returns_empty(self, run_eval, tmp_path):
        assert run_eval._load_results(tmp_path / "nope.json") == {}

    def test_corrupt_file_returns_empty(self, run_eval, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert run_eval._load_results(path) == {}


class TestIsAzUnavailable:
    """A bare timeout is NOT az-unavailable; auth markers and a missing CLI are."""

    def test_pure_timeout_is_not_unavailable(self, run_eval):
        proc = {"exit_code": -1, "timeout": True, "stdout": "", "stderr": ""}
        assert run_eval._is_az_unavailable(proc) is False

    def test_missing_cli_is_unavailable(self, run_eval):
        proc = {"exit_code": 127, "timeout": False, "stdout": "", "stderr": ""}
        assert run_eval._is_az_unavailable(proc) is True

    @pytest.mark.parametrize(
        "blob",
        ["Please run 'az login'", "Not logged in", "AADSTS700082: token has expired"],
    )
    def test_auth_markers_are_unavailable(self, run_eval, blob):
        proc = {"exit_code": 1, "timeout": False, "stdout": "", "stderr": blob}
        assert run_eval._is_az_unavailable(proc) is True


class TestClassifyUpload:
    """``_classify_upload`` maps an ``az`` publish result to an upload status."""

    @staticmethod
    def _args(upload_skip_existing=False):
        return argparse.Namespace(upload_skip_existing=upload_skip_existing)

    def test_success(self, run_eval):
        up = {"exit_code": 0, "timeout": False, "stdout": "", "stderr": ""}
        assert run_eval._classify_upload(up, self._args()) == "uploaded"

    def test_conflict_with_skip_existing(self, run_eval):
        up = {"exit_code": 1, "timeout": False, "stdout": "", "stderr": "PackageVersionExists"}
        assert (
            run_eval._classify_upload(up, self._args(upload_skip_existing=True)) == "exists-skipped"
        )

    def test_conflict_without_skip_existing_is_failed(self, run_eval):
        up = {"exit_code": 1, "timeout": False, "stdout": "", "stderr": "PackageVersionExists"}
        assert run_eval._classify_upload(up, self._args(upload_skip_existing=False)) == "failed"

    def test_auth_is_abort(self, run_eval):
        up = {"exit_code": 1, "timeout": False, "stdout": "", "stderr": "Please run 'az login'"}
        assert run_eval._classify_upload(up, self._args()) == "auth-abort"

    def test_missing_cli_is_abort(self, run_eval):
        up = {"exit_code": 127, "timeout": False, "stdout": "", "stderr": "az not found"}
        assert run_eval._classify_upload(up, self._args()) == "auth-abort"

    def test_timeout(self, run_eval):
        up = {"exit_code": -1, "timeout": True, "stdout": "", "stderr": ""}
        assert run_eval._classify_upload(up, self._args()) == "timeout"

    def test_plain_failure(self, run_eval):
        up = {"exit_code": 1, "timeout": False, "stdout": "", "stderr": "500 server error"}
        assert run_eval._classify_upload(up, self._args()) == "failed"


class TestRunBuildOnlyUploadCleanup:
    """Per-combo upload bounds disk: every outcome cleans up locally + is recorded.

    Exercises the pinned single-combo path (``--ep qnn --device npu``) with
    ``_run_build`` and ``_upload_model_dir`` mocked, asserting the per-combo dir is
    removed and the right status lands in ``build_only_results.json``.
    """

    HF_ID = "microsoft/resnet-50"
    TASK = "image-classification"

    @classmethod
    def _entry(cls):
        entry = MagicMock()
        entry.hf_id = cls.HF_ID
        entry.task = cls.TASK
        entry.priority = "P0"
        entry.group = "vision"
        entry.precision = None
        return entry

    @staticmethod
    def _args(tmp_path, **overrides):
        args = argparse.Namespace(
            output_dir=tmp_path,
            ep="qnn",
            device="npu",
            upload=True,
            continue_run=False,
            keep_local=False,
            upload_skip_existing=False,
            timeout=300,
            verbose=False,
            clean_cache=False,
            run_stamp="20260609",
            feed="Modelkit",
            feed_org="https://dev.azure.com/microsoft",
            feed_project="windows.ai.toolkit",
            package_name="winml-cli-models",
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    @staticmethod
    def _build(success: bool):
        """Fake ``_run_build`` that materialises the combo dir on disk."""

        def _fake(entry, device, precision, timeout, build_dir, ep=None, build_only=False):
            build_dir = Path(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / "quantized.onnx").write_text("x", encoding="utf-8")
            return {
                "success": success,
                "onnx_paths": {"": str(build_dir)} if success else {},
                "stage": "complete" if success else "build",
                "proc": {
                    "exit_code": 0 if success else 1,
                    "stdout": "",
                    "stderr": "" if success else "boom",
                    "timeout": False,
                },
            }

        return _fake

    def _run(self, run_eval, args, build_side_effect, upload_proc):
        with (
            patch.object(run_eval, "save_environment_info"),
            patch.object(run_eval, "_ensure_feed_ready", return_value=None),
            patch.object(run_eval, "_run_build", side_effect=build_side_effect),
            patch.object(run_eval, "_upload_model_dir", return_value=upload_proc),
        ):
            run_eval._run_build_only([self._entry()], args)

    def _model_dir(self, run_eval, tmp_path):
        return run_eval.model_result_dir(tmp_path, self.HF_ID, self.TASK)

    def _record(self, tmp_path):
        results = json.loads((tmp_path / "build_only_results.json").read_text(encoding="utf-8"))
        assert len(results) == 1
        return next(iter(results.values()))

    def test_upload_ok_removes_local_and_records(self, run_eval, tmp_path):
        up = {"exit_code": 0, "stdout": "", "stderr": "", "timeout": False}
        self._run(run_eval, self._args(tmp_path), self._build(True), up)
        assert not self._model_dir(run_eval, tmp_path).exists()
        rec = self._record(tmp_path)
        assert rec["build_status"] == "ok"
        assert rec["upload_status"] == "uploaded"

    def test_upload_timeout_cleans_and_continues(self, run_eval, tmp_path):
        # A timed-out upload must NOT raise SystemExit; it cleans up and is recorded.
        up = {"exit_code": -1, "stdout": "", "stderr": "", "timeout": True}
        self._run(run_eval, self._args(tmp_path), self._build(True), up)
        assert not self._model_dir(run_eval, tmp_path).exists()
        assert self._record(tmp_path)["upload_status"] == "timeout"

    def test_build_failure_cleans_and_records(self, run_eval, tmp_path):
        up = {"exit_code": 0, "stdout": "", "stderr": "", "timeout": False}
        self._run(run_eval, self._args(tmp_path), self._build(False), up)
        assert not self._model_dir(run_eval, tmp_path).exists()
        rec = self._record(tmp_path)
        assert rec["build_status"] == "failed"
        assert rec["upload_status"] == "skipped"

    def test_auth_failure_aborts_after_cleanup(self, run_eval, tmp_path):
        up = {"exit_code": 1, "stdout": "", "stderr": "Please run 'az login'", "timeout": False}
        with pytest.raises(SystemExit):
            self._run(run_eval, self._args(tmp_path), self._build(True), up)
        assert not self._model_dir(run_eval, tmp_path).exists()
        assert self._record(tmp_path)["upload_status"] == "failed"

    def test_keep_local_preserves_dir(self, run_eval, tmp_path):
        up = {"exit_code": 0, "stdout": "", "stderr": "", "timeout": False}
        self._run(run_eval, self._args(tmp_path, keep_local=True), self._build(True), up)
        assert self._model_dir(run_eval, tmp_path).exists()
        assert self._record(tmp_path)["upload_status"] == "uploaded"
