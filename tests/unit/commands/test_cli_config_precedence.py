# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cross-command CLI <-> build-config value-priority tests.

Documents the contract every command that accepts both CLI options AND a
``-c/--config`` JSON file must honor:

    1. CLI option explicitly provided on the command line
    2. Build-config JSON value explicitly present in the file
    3. CLI option default
    4. Dataclass / build-config default

Tier 4 must NOT shadow tier 3.

These tests probe the actual command body with a real JSON file on disk
(no mocking of ``load_build_config``) so they exercise the same
``WinMLBuildConfig.from_dict`` path the bug lives in.

Note on coverage:
- quantize/eval are covered elsewhere
  (``TestQuantizeCliConfigPrecedence`` in test_compile_quantize_flags.py
  and the ``test_cli_ep_*`` tests in eval).
- compile/perf/analyze have active bugs: dataclass default
  ``EPConfig.provider == "qnn"`` overrides CLI ``--ep`` default ``None``
  whenever the JSON has a ``compile`` section (even empty).
- export currently has only latent bugs (CLI defaults coincide with
  dataclass defaults for ``task``, ``no_hierarchy``, ``dynamo``); the
  positive-contract tests below pin behavior so regressions stay caught.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# =============================================================================
# compile
# =============================================================================


class TestCompileCliConfigPrecedence:
    """``winml compile`` -- value priority CLI > config > CLI-default."""

    @staticmethod
    def _setup(tmp_path: Path) -> Path:
        model = tmp_path / "m.onnx"
        model.write_bytes(b"fake")
        return model

    @staticmethod
    def _captured(args, tmp_path):
        from winml.modelkit.commands.compile import compile as compile_cmd

        captured: dict[str, object] = {}

        def fake_compile_onnx(model_path, output_path=None, config=None, **_kw):
            captured["config"] = config
            r = MagicMock()
            r.success = True
            r.output_path = output_path or (tmp_path / "out.onnx")
            r.compile_time = 0.0
            r.total_time = 0.0
            r.errors = []
            return r

        with (
            patch(
                "winml.modelkit.commands.compile.is_compiled_onnx",
                return_value=False,
            ),
            patch(
                "winml.modelkit.compiler.compile_onnx",
                side_effect=fake_compile_onnx,
            ),
        ):
            r = CliRunner().invoke(
                compile_cmd, args, obj={}, catch_exceptions=False
            )
        assert r.exit_code == 0, r.output
        return captured["config"]

    # ---- ep: active bug ----

    def test_absent_compile_section_matches_no_config_behavior(self, tmp_path):
        """JSON without ``compile`` section must produce identical EP resolution
        to running with no ``--config`` at all.

        With ``--device gpu`` and no ``--ep``, both invocations should refuse
        to compile (DML has no EPContext path). This pins the contract that
        an absent JSON key does NOT alter EP selection.
        """
        from winml.modelkit.commands.compile import compile as compile_cmd

        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"quant": {}}', encoding="utf-8")

        with (
            patch(
                "winml.modelkit.commands.compile.is_compiled_onnx",
                return_value=False,
            ),
            patch("winml.modelkit.compiler.compile_onnx"),
        ):
            no_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu"],
                obj={},
                catch_exceptions=False,
            )
            with_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu", "--config", str(bc)],
                obj={},
                catch_exceptions=False,
            )
        assert no_cfg.exit_code == with_cfg.exit_code
        assert "EPContext" in no_cfg.output
        assert "EPContext" in with_cfg.output

    def test_empty_compile_section_matches_no_config_behavior(self, tmp_path):
        """JSON ``"compile": {}`` (no keys) must NOT alter EP resolution.

        Today this is the BUG: ``WinMLBuildConfig.from_dict`` fills missing
        ``execution_provider`` with the dataclass default ``"qnn"``, which
        the merge block then assigns to ``ep``, forcing QNN even when the
        user picked ``--device gpu``. The without-config run correctly
        refuses (``DML has no EPContext path``) while the with-config run
        silently switches to QNN and "succeeds".
        """
        from winml.modelkit.commands.compile import compile as compile_cmd

        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {}}', encoding="utf-8")

        providers: list[str] = []

        def fake(model_path, output_path=None, config=None, **_kw):
            providers.append(config.ep_config.provider)
            r = MagicMock()
            r.success = True
            r.output_path = tmp_path / "out.onnx"
            r.compile_time = 0.0
            r.total_time = 0.0
            r.errors = []
            return r

        with (
            patch(
                "winml.modelkit.commands.compile.is_compiled_onnx",
                return_value=False,
            ),
            patch(
                "winml.modelkit.compiler.compile_onnx", side_effect=fake
            ),
        ):
            no_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu"],
                obj={},
                catch_exceptions=False,
            )
            with_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu", "--config", str(bc)],
                obj={},
                catch_exceptions=False,
            )
        assert no_cfg.exit_code == with_cfg.exit_code, (
            f"Empty compile section silently changed behavior:\n"
            f"  no-config exit={no_cfg.exit_code} providers={providers!r}\n"
            f"  with-config exit={with_cfg.exit_code}\n"
            f"  with-config tail: {with_cfg.output[-300:]}"
        )

    def test_partial_compile_section_keeps_cli_default_ep(self, tmp_path):
        """JSON with only ``validate: false`` in compile must NOT touch ``ep``."""
        from winml.modelkit.commands.compile import compile as compile_cmd

        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {"validate": false}}', encoding="utf-8")

        providers: list[str] = []

        def fake(model_path, output_path=None, config=None, **_kw):
            providers.append(config.ep_config.provider)
            r = MagicMock()
            r.success = True
            r.output_path = tmp_path / "out.onnx"
            r.compile_time = 0.0
            r.total_time = 0.0
            r.errors = []
            return r

        with (
            patch(
                "winml.modelkit.commands.compile.is_compiled_onnx",
                return_value=False,
            ),
            patch(
                "winml.modelkit.compiler.compile_onnx", side_effect=fake
            ),
        ):
            no_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu"],
                obj={},
                catch_exceptions=False,
            )
            with_cfg = CliRunner().invoke(
                compile_cmd,
                ["-m", str(model), "--device", "gpu", "--config", str(bc)],
                obj={},
                catch_exceptions=False,
            )
        assert no_cfg.exit_code == with_cfg.exit_code, (
            f"Partial compile section silently changed EP resolution. "
            f"providers={providers!r}, with-config output:\n{with_cfg.output[-300:]}"
        )

    def test_config_provides_ep_when_cli_silent(self, tmp_path):
        """JSON ``compile.execution_provider`` applies when ``--ep`` is omitted."""
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "openvino"}}',
            encoding="utf-8",
        )
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep_config.provider == "openvino"

    def test_cli_ep_beats_config(self, tmp_path):
        """Explicit ``--ep`` wins over JSON ``compile.execution_provider``."""
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "openvino"}}',
            encoding="utf-8",
        )
        cfg = self._captured(
            ["-m", str(model), "--ep", "qnn", "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep_config.provider == "qnn"


# =============================================================================
# perf
# =============================================================================


class TestPerfCliConfigPrecedence:
    """``winml perf`` -- value priority for ``task`` and ``ep``."""

    @staticmethod
    def _setup(tmp_path: Path) -> Path:
        model = tmp_path / "m.onnx"
        model.write_bytes(b"fake")
        return model

    @staticmethod
    def _captured(args, tmp_path):
        """Invoke perf with the ONNX-file branch; capture the BenchmarkConfig."""
        from winml.modelkit.commands.perf import perf as perf_cmd

        captured: dict[str, object] = {}

        def fake_run(model_path, *, config=None, **_kw):
            captured["config"] = config
            r = MagicMock()
            return r

        with (
            patch(
                "winml.modelkit.commands.perf._run_onnx_benchmark",
                side_effect=fake_run,
            ),
            patch(
                "winml.modelkit.commands.perf.display_console_report"
            ),
            patch(
                "winml.modelkit.commands.perf.write_json_report"
            ),
            patch(
                "winml.modelkit.commands.perf.generate_output_path",
                return_value=tmp_path / "out.json",
            ),
        ):
            r = CliRunner().invoke(perf_cmd, args, obj={}, catch_exceptions=False)
        assert r.exit_code == 0, r.output
        return captured["config"]

    # ---- ep: active bug ----

    def test_empty_compile_section_keeps_cli_default_ep(self, tmp_path):
        """JSON with ``"compile": {}`` must NOT override CLI default ``ep=None``."""
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {}}', encoding="utf-8")
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep is None, (
            f"Empty compile section must not leak dataclass default ep=qnn; "
            f"got {cfg.ep!r}."
        )

    def test_partial_compile_section_keeps_cli_default_ep(self, tmp_path):
        """JSON ``compile.validate=false`` must not touch ``ep``."""
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {"validate": false}}', encoding="utf-8")
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep is None

    def test_config_ep_applies_when_cli_silent(self, tmp_path):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "dml"}}', encoding="utf-8"
        )
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep == "dml"

    def test_cli_ep_beats_config(self, tmp_path):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "dml"}}', encoding="utf-8"
        )
        cfg = self._captured(
            ["-m", str(model), "--ep", "qnn", "--config", str(bc)],
            tmp_path,
        )
        assert cfg.ep == "qnn"

    # ---- task: latent bug (CLI default coincides with loader.task default) ----

    def test_config_task_applies_when_cli_silent(self, tmp_path):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"loader": {"task": "image-classification"}}', encoding="utf-8"
        )
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.task == "image-classification"

    def test_cli_task_beats_config(self, tmp_path):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"loader": {"task": "image-classification"}}', encoding="utf-8"
        )
        cfg = self._captured(
            ["-m", str(model), "--config", str(bc), "--task", "fill-mask"],
            tmp_path,
        )
        assert cfg.task == "fill-mask"


# =============================================================================
# analyze
# =============================================================================


@pytest.fixture
def _mock_rule_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass rule-data validation so CLI tests don't depend on rule artifacts."""
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        lambda *_a, **_kw: True,
    )
    monkeypatch.setattr(
        "winml.modelkit.commands.analyze._discover_runtime_rule_parquet_files",
        lambda: (
            [Path("runtime_check_rules")],
            [Path("runtime_check_rules/mock.parquet")],
        ),
    )


class TestAnalyzeCliConfigPrecedence:
    """``winml analyze`` -- value priority for ``ep``."""

    @staticmethod
    def _setup(tmp_path: Path) -> Path:
        model = tmp_path / "m.onnx"
        model.write_bytes(b"fake")
        return model

    @staticmethod
    def _captured(args, tmp_path):
        """Run analyze in quiet mode; capture the ``ep`` passed to analyzer.analyze()."""
        from winml.modelkit.commands.analyze import analyze as analyze_cmd

        captured: dict[str, object] = {}

        mock_result = MagicMock()
        mock_result.is_fully_supported.return_value = True
        mock_result.output.results = []

        def fake_analyze(**kw):
            captured.update(kw)
            return mock_result

        with patch(
            "winml.modelkit.analyze.ONNXStaticAnalyzer"
        ) as mock_analyzer_cls:
            mock_inst = MagicMock()
            mock_inst.analyze.side_effect = fake_analyze
            mock_analyzer_cls.return_value = mock_inst

            r = CliRunner().invoke(
                analyze_cmd, [*args, "--quiet"], obj={}, catch_exceptions=False
            )
        assert r.exit_code in (0, 1), r.output  # 1 = partial support
        return captured

    def test_empty_compile_section_keeps_cli_default_ep(
        self, tmp_path, _mock_rule_data
    ):
        """JSON ``"compile": {}`` must NOT override CLI default ``ep=None``."""
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {}}', encoding="utf-8")
        cap = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cap["ep"] is None, (
            f"Empty compile section must not leak dataclass default; "
            f"got ep={cap['ep']!r}."
        )

    def test_partial_compile_section_keeps_cli_default_ep(
        self, tmp_path, _mock_rule_data
    ):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text('{"compile": {"validate": false}}', encoding="utf-8")
        cap = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cap["ep"] is None

    def test_config_ep_applies_when_cli_silent(
        self, tmp_path, _mock_rule_data
    ):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "openvino"}}',
            encoding="utf-8",
        )
        cap = self._captured(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        # ep is normalized inside the command
        assert cap["ep"] == "OpenVINOExecutionProvider"

    def test_cli_ep_beats_config(self, tmp_path, _mock_rule_data):
        model = self._setup(tmp_path)
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"compile": {"execution_provider": "openvino"}}',
            encoding="utf-8",
        )
        cap = self._captured(
            ["-m", str(model), "--config", str(bc), "--ep", "qnn"],
            tmp_path,
        )
        assert cap["ep"] == "QNNExecutionProvider"


# =============================================================================
# export
# =============================================================================


class TestExportCliConfigPrecedence:
    """``winml export`` -- value priority for ``task``, ``no_hierarchy``, ``dynamo``.

    Active failures are not expected today because every CLI default
    coincides with the corresponding dataclass default. These positive-
    contract tests pin the wiring so the next change to a CLI default
    surfaces the latent bug immediately.
    """

    @staticmethod
    def _captured(args, tmp_path):
        from winml.modelkit.commands.export import export as export_cmd

        captured: dict[str, object] = {}

        def fake_load_hf(model_id, task=None, **_kw):
            captured["task"] = task
            return (MagicMock(), MagicMock(), task or "detected-task")

        def fake_resolve(model_id, task=None, shape_config=None):
            return (MagicMock(input_tensors=None, output_tensors=None), None)

        def fake_export(**kw):
            captured["export_kwargs"] = kw
            captured["export_config"] = kw.get("export_config")
            return MagicMock()

        with (
            patch(
                "winml.modelkit.loader.load_hf_model",
                side_effect=fake_load_hf,
            ),
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
                export_cmd, args, obj={"debug": False}, catch_exceptions=False
            )
        assert r.exit_code == 0, r.output
        return captured

    def test_absent_sections_keep_cli_defaults(self, tmp_path):
        """JSON without loader/export sections must not touch CLI-default vars."""
        out = tmp_path / "out.onnx"
        bc = tmp_path / "bc.json"
        bc.write_text('{"quant": {}}', encoding="utf-8")
        cap = self._captured(
            ["-m", "prajjwal1/bert-tiny", "-o", str(out), "-c", str(bc)],
            tmp_path,
        )
        assert cap["task"] is None  # CLI default
        ec = cap["export_config"]
        assert ec.enable_hierarchy_tags is True  # CLI default for --no-hierarchy is False
        assert ec.dynamo is False  # CLI default

    def test_config_task_applies_when_cli_silent(self, tmp_path):
        out = tmp_path / "out.onnx"
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"loader": {"task": "image-classification"}}', encoding="utf-8"
        )
        cap = self._captured(
            ["-m", "prajjwal1/bert-tiny", "-o", str(out), "-c", str(bc)],
            tmp_path,
        )
        assert cap["task"] == "image-classification"

    def test_cli_task_beats_config(self, tmp_path):
        out = tmp_path / "out.onnx"
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"loader": {"task": "image-classification"}}', encoding="utf-8"
        )
        cap = self._captured(
            [
                "-m",
                "prajjwal1/bert-tiny",
                "-o",
                str(out),
                "-c",
                str(bc),
                "--task",
                "fill-mask",
            ],
            tmp_path,
        )
        assert cap["task"] == "fill-mask"

    def test_config_enable_hierarchy_tags_false_applies(self, tmp_path):
        """JSON ``export.enable_hierarchy_tags=false`` applies when CLI silent."""
        out = tmp_path / "out.onnx"
        bc = tmp_path / "bc.json"
        bc.write_text(
            '{"export": {"enable_hierarchy_tags": false}}', encoding="utf-8"
        )
        cap = self._captured(
            ["-m", "prajjwal1/bert-tiny", "-o", str(out), "-c", str(bc)],
            tmp_path,
        )
        assert cap["export_config"].enable_hierarchy_tags is False

    def test_config_dynamo_true_applies(self, tmp_path):
        out = tmp_path / "out.onnx"
        bc = tmp_path / "bc.json"
        bc.write_text('{"export": {"dynamo": true}}', encoding="utf-8")
        cap = self._captured(
            ["-m", "prajjwal1/bert-tiny", "-o", str(out), "-c", str(bc)],
            tmp_path,
        )
        assert cap["export_config"].dynamo is True
