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
    # Register before exec_module so module-level @dataclass definitions can
    # resolve their own __module__ in sys.modules (dataclasses looks it up to
    # detect KW_ONLY/ClassVar); without this, exec raises AttributeError.
    sys.modules["_e2e_run_eval"] = mod
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


class TestResolveOpTracing:
    """Behaviour of ``_resolve_op_tracing`` and the target-key helpers."""

    @staticmethod
    def _entry(run_eval, targets):
        return run_eval.ModelEntry(
            hf_id="acme/model",
            task="image-classification",
            model_type="vit",
            group="test",
            priority="P0",
            op_tracing_targets=list(targets),
        )

    def test_explicit_disabled_overrides_optin(self, run_eval):
        # A model that opts in must still be forced off by an explicit 'disabled'.
        entry = self._entry(run_eval, ["QNNExecutionProvider_npu"])
        assert run_eval._resolve_op_tracing("disabled", entry, "qnn", "npu") is None

    @pytest.mark.parametrize("level", ["basic", "detail"])
    def test_explicit_level_used_as_is(self, run_eval, level):
        # Explicit basic/detail is honoured regardless of opt-in or EP/device.
        entry = self._entry(run_eval, [])
        assert run_eval._resolve_op_tracing(level, entry, "qnn", "npu") == level
        assert run_eval._resolve_op_tracing(level, entry, None, "auto") == level

    def test_unset_auto_enables_on_matching_target(self, run_eval):
        entry = self._entry(run_eval, ["QNNExecutionProvider_npu"])
        assert run_eval._resolve_op_tracing(None, entry, "qnn", "npu") == "basic"

    def test_unset_no_match_stays_off(self, run_eval):
        entry = self._entry(run_eval, ["QNNExecutionProvider_npu"])
        # Different device (dml/npu) and a model without targets both stay off.
        assert run_eval._resolve_op_tracing(None, entry, "dml", "npu") is None
        assert run_eval._resolve_op_tracing(None, self._entry(run_eval, []), "qnn", "npu") is None

    def test_unset_device_auto_does_not_match(self, run_eval):
        # 'auto' is not resolved to a concrete device here, so it never matches
        # a device-specific target such as QNNExecutionProvider_npu.
        entry = self._entry(run_eval, ["QNNExecutionProvider_npu"])
        assert run_eval._resolve_op_tracing(None, entry, "qnn", "auto") is None

    def test_unset_no_ep_stays_off(self, run_eval):
        entry = self._entry(run_eval, ["QNNExecutionProvider_npu"])
        assert run_eval._resolve_op_tracing(None, entry, None, "npu") is None


class TestOpTracingTargetKey:
    """The canonical key builder and the registry target normalizer."""

    def test_target_key_normalizes_ep_and_device(self, run_eval):
        assert run_eval.op_tracing_target_key("qnn", "NPU") == "QNNExecutionProvider_npu"
        full = run_eval.op_tracing_target_key("QNNExecutionProvider", "npu")
        assert full == "QNNExecutionProvider_npu"

    def test_target_key_none_without_ep(self, run_eval):
        assert run_eval.op_tracing_target_key(None, "npu") is None
        assert run_eval.op_tracing_target_key("", "npu") is None

    def test_normalize_target_accepts_alias_and_full_name(self, run_eval):
        from utils.registry import normalize_op_tracing_target

        assert normalize_op_tracing_target("qnn_npu") == "QNNExecutionProvider_npu"
        assert normalize_op_tracing_target("QNNExecutionProvider_npu") == "QNNExecutionProvider_npu"
        assert normalize_op_tracing_target("QNN_NPU") == "QNNExecutionProvider_npu"

    def test_registry_normalizes_targets_on_load(self, run_eval, tmp_path):
        registry = tmp_path / "models.json"
        registry.write_text(
            json.dumps(
                [
                    {
                        "hf_id": "acme/model",
                        "task": "image-classification",
                        "model_type": "vit",
                        "group": "test",
                        "priority": "P0",
                        "op_tracing_targets": ["qnn_npu"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        entries = run_eval.load_registry(registry)
        assert entries[0].op_tracing_targets == ["QNNExecutionProvider_npu"]


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

    def test_invalid_grant_is_unavailable(self, run_eval):
        # invalid_grant is the OAuth2 code Azure AD uses for an expired/revoked
        # refresh token (chosen over the broad 'refresh token' substring).
        proc = {
            "exit_code": 1,
            "timeout": False,
            "stdout": "",
            "stderr": "OAuth error: invalid_grant - token revoked",
        }
        assert run_eval._is_az_unavailable(proc) is True

    def test_informational_refresh_token_is_not_unavailable(self, run_eval):
        # A bare informational MSAL progress line must NOT trigger an abort now
        # that the broad 'refresh token' marker has been narrowed.
        proc = {
            "exit_code": 1,
            "timeout": False,
            "stdout": "Refreshing token for scope https://example",
            "stderr": "",
        }
        assert run_eval._is_az_unavailable(proc) is False


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


class TestFetchFeedVersions:
    """`_fetch_feed_versions`: resolve the package, then list its versions.

    A package missing from the (possibly paginated) listing returns ``None`` --
    not an empty set -- so --continue takes the explicit fallback instead of
    silently rebuilding the whole batch.
    """

    @staticmethod
    def _args():
        return argparse.Namespace(
            feed_org="https://dev.azure.com/microsoft",
            feed_project="windows.ai.toolkit",
            feed="Modelkit",
            package_name="winml-cli-models",
        )

    @staticmethod
    def _ok(stdout):
        return {"exit_code": 0, "stdout": stdout, "stderr": "", "timeout": False}

    def test_package_not_found_returns_none(self, run_eval):
        listing = json.dumps({"value": [{"name": "other-pkg", "protocolType": "upack", "id": "x"}]})

        def fake_az(az_args, _timeout=180):
            return self._ok(listing)

        with patch.object(run_eval, "_run_az", side_effect=fake_az):
            assert run_eval._fetch_feed_versions(self._args(), "20260609") is None

    def test_returns_versions_matching_run_stamp(self, run_eval):
        listing = json.dumps(
            {"value": [{"name": "winml-cli-models", "protocolType": "upack", "id": "PKG"}]}
        )
        versions = json.dumps(
            {
                "value": [
                    {"version": "0.0.0-20260609-qnn-npu-m"},
                    {"version": "0.0.0-20260609-ov-cpu-m"},
                    {"version": "0.0.0-20251231-qnn-npu-m"},
                ]
            }
        )

        def fake_az(az_args, _timeout=180):
            return self._ok(versions if "/versions" in az_args[-1] else listing)

        with patch.object(run_eval, "_run_az", side_effect=fake_az):
            result = run_eval._fetch_feed_versions(self._args(), "20260609")
        assert result == {"0.0.0-20260609-qnn-npu-m", "0.0.0-20260609-ov-cpu-m"}

    def test_query_failure_returns_none(self, run_eval):
        def fake_az(az_args, _timeout=180):
            return {"exit_code": 1, "stdout": "", "stderr": "boom", "timeout": False}

        with patch.object(run_eval, "_run_az", side_effect=fake_az):
            assert run_eval._fetch_feed_versions(self._args(), "20260609") is None


# ---------------------------------------------------------------------------
# Recipe-driven merge: discovery -> jobs -> build -> eval
# ---------------------------------------------------------------------------


def _entry(hf_id="microsoft/resnet-50", task="image-classification"):
    entry = MagicMock()
    entry.hf_id = hf_id
    entry.task = task
    entry.precision = None
    entry.perf_args = []
    entry.eval_args = []
    return entry


class TestModelResultDirPrecision:
    """``model_result_dir`` folds precision into the slug for recipe variants."""

    def test_without_precision(self, run_eval, tmp_path):
        d = run_eval.model_result_dir(tmp_path, "microsoft/resnet-50", "image-classification")
        assert d.name == "microsoft__resnet-50__image-classification"

    def test_with_precision(self, run_eval, tmp_path):
        d = run_eval.model_result_dir(
            tmp_path, "microsoft/resnet-50", "image-classification", "w8a16"
        )
        assert d.name == "microsoft__resnet-50__image-classification__w8a16"


class TestAccuracyStatus:
    """``accuracy_status`` is a coarse, baseline-free status."""

    def test_not_run(self, run_eval):
        assert run_eval.accuracy_status(None) == "NOT_RUN"

    def test_skipped(self, run_eval):
        acc = {"skipped": True, "skip_reason": "perf_failed"}
        assert run_eval.accuracy_status(acc) == "SKIPPED"

    def test_pass(self, run_eval):
        assert run_eval.accuracy_status({"winml_eval_status": "PASS"}) == "PASS"

    def test_fail(self, run_eval):
        assert run_eval.accuracy_status({"winml_eval_status": "FAIL"}) == "FAIL"


class TestRecipeConfigHelpers:
    """Eval-section detection, meta-config pick, and trust-remote-code gate."""

    def _write(self, path: Path, payload: dict):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_has_eval_section(self, run_eval, tmp_path):
        cfg = tmp_path / "a.json"
        self._write(cfg, {"eval": {"task": "x"}})
        assert run_eval._config_has_eval_section(cfg) is True

    def test_missing_eval_section(self, run_eval, tmp_path):
        cfg = tmp_path / "b.json"
        self._write(cfg, {"loader": {"task": "x"}})
        assert run_eval._config_has_eval_section(cfg) is False

    def test_meta_config_picks_eval_component(self, run_eval, tmp_path):
        # Composite: only the decoder config carries the eval section.
        model_dir = tmp_path / "microsoft_trocr-base-printed"
        model_dir.mkdir()
        enc = model_dir / "image-to-text_fp16_config_encoder.json"
        dec = model_dir / "image-to-text_fp16_config_decoder.json"
        self._write(enc, {"loader": {"task": "image-to-text"}})
        self._write(dec, {"eval": {"task": "image-to-text", "dataset": {"path": "x"}}})
        variants = run_eval.discover_recipe_variants(
            tmp_path, "microsoft/trocr-base-printed", "image-to-text"
        )
        assert len(variants) == 1
        assert run_eval._recipe_meta_config(variants[0]) == dec

    def test_needs_trust_remote_code(self, run_eval, tmp_path):
        cfg = tmp_path / "c.json"
        self._write(cfg, {"eval": {"dataset": {"path": "x", "build_script": "build.py"}}})
        assert run_eval._needs_trust_remote_code(cfg) is True

    def test_no_trust_without_build_script(self, run_eval, tmp_path):
        cfg = tmp_path / "d.json"
        self._write(cfg, {"eval": {"dataset": {"path": "x"}}})
        assert run_eval._needs_trust_remote_code(cfg) is False
        assert run_eval._needs_trust_remote_code(None) is False


class TestBuildJobs:
    """``_build_jobs`` expands entries into one job per recipe precision variant."""

    def _make_single_recipe(self, recipes_dir: Path, slug: str, task: str, precisions: list[str]):
        model_dir = recipes_dir / slug
        model_dir.mkdir(parents=True)
        for prec in precisions:
            (model_dir / f"{task}_{prec}_config.json").write_text(
                json.dumps({"eval": {"task": task, "dataset": {"path": "x"}}}), encoding="utf-8"
            )

    def test_recipe_expands_to_one_job_per_precision(self, run_eval, tmp_path):
        self._make_single_recipe(
            tmp_path, "microsoft_resnet-50", "image-classification", ["fp16", "w8a16"]
        )
        entry = _entry()
        jobs = run_eval._build_jobs([entry], tmp_path)
        assert [j.precision for j in jobs] == ["fp16", "w8a16"]
        assert all(j.entry is entry for j in jobs)

    def test_no_recipe_falls_back_to_single_job(self, run_eval, tmp_path):
        entry = _entry("some/model", "text-classification")
        jobs = run_eval._build_jobs([entry], tmp_path)
        assert len(jobs) == 1
        assert jobs[0].variant is None
        assert jobs[0].precision is None

    def test_recipes_disabled_yields_fallback_jobs(self, run_eval, tmp_path):
        self._make_single_recipe(
            tmp_path, "microsoft_resnet-50", "image-classification", ["fp16"]
        )
        entry = _entry()
        # recipes_dir=None disables recipe discovery entirely.
        jobs = run_eval._build_jobs([entry], None)
        assert len(jobs) == 1
        assert jobs[0].variant is None


class TestRunRecipeBuild:
    """``_run_recipe_build`` builds authored configs with ``winml build -c``."""

    def _variant(self, run_eval, tmp_path, *, composite: bool):
        model_dir = tmp_path / "recipes" / "microsoft_resnet-50"
        model_dir.mkdir(parents=True)
        task = "image-to-text" if composite else "image-classification"
        if composite:
            (model_dir / f"{task}_fp16_config_encoder.json").write_text(
                json.dumps({"loader": {"task": task}}), encoding="utf-8"
            )
            (model_dir / f"{task}_fp16_config_decoder.json").write_text(
                json.dumps({"eval": {"task": task, "dataset": {"path": "x"}}}), encoding="utf-8"
            )
        else:
            (model_dir / f"{task}_fp16_config.json").write_text(
                json.dumps({"eval": {"task": task, "dataset": {"path": "x"}}}), encoding="utf-8"
            )
        variants = run_eval.discover_recipe_variants(
            tmp_path / "recipes", "microsoft/resnet-50", task
        )
        assert len(variants) == 1
        return variants[0]

    def _fake_subprocess(self, captured):
        def _fake(args, _timeout):
            captured.append(list(args))
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "elapsed": 0.1,
                "timeout": False,
                "command": " ".join(args),
            }

        return _fake

    def test_single_build_uses_config_and_output_dir(self, run_eval, tmp_path):
        variant = self._variant(run_eval, tmp_path, composite=False)
        out = tmp_path / "out"
        captured: list[list[str]] = []
        with (
            patch.object(run_eval, "_run_subprocess", side_effect=self._fake_subprocess(captured)),
            patch.object(run_eval, "_find_built_onnx", side_effect=lambda bd: bd / "model.onnx"),
        ):
            result = run_eval._run_recipe_build(_entry(), variant, 300, out, ep="openvino")

        assert result["success"] is True
        assert list(result["onnx_paths"]) == [""]  # single model uses "" label
        assert result["meta_config"] == variant.components[0].path
        build_call = captured[0]
        assert "build" in build_call
        assert "-c" in build_call and str(variant.components[0].path) in build_call
        assert "-o" in build_call
        assert "--no-compile" not in build_call  # openvino is not vitisai

    def test_vitisai_adds_no_compile(self, run_eval, tmp_path):
        variant = self._variant(run_eval, tmp_path, composite=False)
        captured: list[list[str]] = []
        with (
            patch.object(run_eval, "_run_subprocess", side_effect=self._fake_subprocess(captured)),
            patch.object(run_eval, "_find_built_onnx", side_effect=lambda bd: bd / "model.onnx"),
        ):
            run_eval._run_recipe_build(_entry(), variant, 300, tmp_path / "o", ep="vitisai")
        assert "--no-compile" in captured[0]

    def test_composite_builds_each_role(self, run_eval, tmp_path):
        variant = self._variant(run_eval, tmp_path, composite=True)
        out = tmp_path / "out"
        captured: list[list[str]] = []
        with (
            patch.object(run_eval, "_run_subprocess", side_effect=self._fake_subprocess(captured)),
            patch.object(run_eval, "_find_built_onnx", side_effect=lambda bd: bd / "model.onnx"),
        ):
            result = run_eval._run_recipe_build(_entry(), variant, 300, out, ep="openvino")

        assert result["success"] is True
        assert set(result["onnx_paths"]) == {"encoder", "decoder"}
        assert len(captured) == 2  # one winml build per component

    def test_build_failure_reported(self, run_eval, tmp_path):
        variant = self._variant(run_eval, tmp_path, composite=False)

        def _fail(args, _timeout):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "boom",
                "elapsed": 0.1,
                "timeout": False,
                "command": " ".join(args),
            }

        with patch.object(run_eval, "_run_subprocess", side_effect=_fail):
            result = run_eval._run_recipe_build(_entry(), variant, 300, tmp_path / "o")
        assert result["success"] is False
        assert result["stage"] == "build"


class TestRunWinmlEvalRecipePath:
    """``_run_winml_eval`` reads the dataset from ``-c`` on the recipe path."""

    def _fake_subprocess_writing_output(self, captured, metrics, dataset):
        def _fake(args, _timeout):
            captured.append(list(args))
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"metrics": metrics, "dataset": dataset}), encoding="utf-8")
            return {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "elapsed": 1.0,
                "timeout": False,
                "command": " ".join(args),
            }

        return _fake

    def test_recipe_path_passes_config_not_dataset_flags(self, run_eval, tmp_path):
        cfg = tmp_path / "recipe.json"
        cfg.write_text("{}", encoding="utf-8")
        captured: list[list[str]] = []
        fake = self._fake_subprocess_writing_output(
            captured, {"accuracy": 0.9}, {"samples": 100}
        )
        with patch.object(run_eval, "_run_subprocess", side_effect=fake):
            res = run_eval._run_winml_eval(
                _entry(),
                "npu",
                300,
                {},
                tmp_path,
                {"": "model.onnx"},
                ep="openvino",
                recipe_config=cfg,
                trust_remote_code=True,
            )
        call = captured[0]
        assert "-c" in call and str(cfg) in call
        assert "--dataset" not in call  # recipe drives the dataset
        assert "--trust-remote-code" in call
        assert res["status"] == "PASS"
        assert res["metrics"] == {"accuracy": 0.9}
        assert res["dataset"] == {"samples": 100}

    def test_fallback_path_uses_dataset_flags(self, run_eval, tmp_path):
        captured: list[list[str]] = []
        fake = self._fake_subprocess_writing_output(captured, {"accuracy": 0.8}, {"samples": 50})
        ds_config = {"dataset": "timm/mini-imagenet", "split": "test", "num_samples": 50}
        with patch.object(run_eval, "_run_subprocess", side_effect=fake):
            res = run_eval._run_winml_eval(
                _entry(), "npu", 300, ds_config, tmp_path, {"": "model.onnx"}, ep="openvino"
            )
        call = captured[0]
        assert "-c" not in call
        assert "--dataset" in call and "timm/mini-imagenet" in call
        assert res["status"] == "PASS"


class TestCleanJobArtifacts:
    """``_clean_job_artifacts`` removes ONNX build outputs but keeps JSON facts."""

    def test_removes_onnx_keeps_json(self, run_eval, tmp_path):
        model_dir = tmp_path / "microsoft__resnet-50__image-classification__fp16"
        model_dir.mkdir()
        # Build artifacts that should be removed.
        (model_dir / "export.onnx").write_text("x")
        (model_dir / "optimized.onnx").write_text("x")
        (model_dir / "model.onnx").write_text("x")
        (model_dir / "model.onnx.data").write_text("x")
        (model_dir / "model_npu_ctx.onnx_VITISAI.bin").write_text("x")
        # Facts that must survive.
        (model_dir / "eval_result.json").write_text("{}")
        (model_dir / "winml_eval_output.json").write_text("{}")

        run_eval._clean_job_artifacts(model_dir)

        survivors = sorted(p.name for p in model_dir.iterdir())
        assert survivors == ["eval_result.json", "winml_eval_output.json"]

    def test_removes_composite_subdir_artifacts(self, run_eval, tmp_path):
        model_dir = tmp_path / "m__image-to-text__fp16"
        (model_dir / "encoder").mkdir(parents=True)
        (model_dir / "decoder").mkdir(parents=True)
        (model_dir / "encoder" / "model.onnx").write_text("x")
        (model_dir / "decoder" / "model.onnx").write_text("x")
        (model_dir / "eval_result.json").write_text("{}")

        run_eval._clean_job_artifacts(model_dir)

        # ONNX gone, empty component dirs pruned, JSON kept.
        assert [p.name for p in model_dir.iterdir()] == ["eval_result.json"]

    def test_missing_dir_is_noop(self, run_eval, tmp_path):
        run_eval._clean_job_artifacts(tmp_path / "nope")  # must not raise


class TestCleanStrayCwdArtifacts:
    """``_clean_stray_cwd_artifacts`` sweeps UUID/sym-shape temps from the cwd."""

    def test_removes_uuid_and_sym_shape_keeps_real(self, run_eval, tmp_path):
        # Stray scratch models libraries leak into the process cwd.
        (tmp_path / "88158373-7198-11f1-ab34-2c9c5846c436.data").write_text("x")
        (tmp_path / "b305c74d-7171-11f1-9869-2c9c5846c436.onnx").write_text("x")
        (tmp_path / "c97746cf-714d-11f1-aa19-2c9c5846c436.onnx.data").write_text("x")
        # QNN EP-context dump form: <uuid>.onnx_<EP>.bin
        (tmp_path / "a1b2c3d4-1234-5678-9abc-def012345678.onnx_QNN.bin").write_text("x")
        (tmp_path / "sym_shape_infer_temp.onnx").write_text("x")
        # Real files that must survive (not UUID-prefixed).
        (tmp_path / "README.md").write_text("x")
        (tmp_path / "model.onnx").write_text("x")
        (tmp_path / "pyproject.toml").write_text("x")

        removed = run_eval._clean_stray_cwd_artifacts(tmp_path)

        assert removed == 5
        survivors = sorted(p.name for p in tmp_path.iterdir())
        assert survivors == ["README.md", "model.onnx", "pyproject.toml"]

    def test_does_not_recurse_into_subdirs(self, run_eval, tmp_path):
        # Only the top-level cwd is swept; a UUID dir/file one level down is left.
        sub = tmp_path / "eval_results"
        sub.mkdir()
        (sub / "88158373-7198-11f1-ab34-2c9c5846c436.data").write_text("x")
        removed = run_eval._clean_stray_cwd_artifacts(tmp_path)
        assert removed == 0
        assert (sub / "88158373-7198-11f1-ab34-2c9c5846c436.data").exists()

    def test_missing_dir_is_noop(self, run_eval, tmp_path):
        assert run_eval._clean_stray_cwd_artifacts(tmp_path / "nope") == 0


class TestRunAccuracyPhaseSchema:
    """``_run_accuracy_phase`` records facts only (no inline PyTorch baseline)."""

    def test_schema_has_metrics_and_no_baseline(self, run_eval, tmp_path):
        winml_ret = {
            "status": "PASS",
            "metric": {"metric": "accuracy", "value": 0.9, "num_samples": 100},
            "metrics": {"accuracy": 0.9},
            "dataset": {"samples": 100},
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "elapsed": 1.0,
            "timeout": False,
            "command": "winml eval ...",
        }
        with patch.object(run_eval, "_run_winml_eval", return_value=winml_ret) as mock_eval:
            acc = run_eval._run_accuracy_phase(
                _entry(),
                "npu",
                300,
                tmp_path,
                {"": "model.onnx"},
                ep="openvino",
                recipe_config=tmp_path / "r.json",
                trust_remote_code=True,
            )
        # New schema: facts only, baseline/delta removed.
        assert acc["winml_eval_status"] == "PASS"
        assert acc["metrics"] == {"accuracy": 0.9}
        assert acc["dataset"] == {"samples": 100}
        assert "pytorch_baseline_status" not in acc
        assert "delta_absolute" not in acc
        # recipe_config + trust flow through to winml eval.
        _, kwargs = mock_eval.call_args
        assert kwargs["recipe_config"] == tmp_path / "r.json"
        assert kwargs["trust_remote_code"] is True
