# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for winml.modelkit.commands.eval._resolve_model_path."""

from __future__ import annotations

import json
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.commands.eval import _resolve_model_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def onnx_file(tmp_path):
    """Create a placeholder .onnx file on disk."""
    f = tmp_path / "model.onnx"
    f.write_bytes(b"")
    return f


@pytest.fixture
def onnx_vision(tmp_path):
    f = tmp_path / "vision.onnx"
    f.write_bytes(b"")
    return f


@pytest.fixture
def onnx_text(tmp_path):
    f = tmp_path / "text.onnx"
    f.write_bytes(b"")
    return f


# ---------------------------------------------------------------------------
# Empty -m
# ---------------------------------------------------------------------------


class TestEmptyModel:
    def test_no_model_no_id_raises(self):
        with pytest.raises(click.UsageError, match="model is required"):
            _resolve_model_path(model=(), model_id=None)

    def test_model_id_only(self):
        path, mid = _resolve_model_path(model=(), model_id="openai/clip-vit-base-patch32")
        assert path is None
        assert mid == "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# Single plain -m (HF ID or .onnx file)
# ---------------------------------------------------------------------------


class TestSinglePlain:
    def test_plain_hf_id_no_model_id(self):
        """-m <hf_id> populates model_id when --model-id omitted."""
        path, mid = _resolve_model_path(model=("microsoft/resnet-50",), model_id=None)
        assert path is None
        assert mid == "microsoft/resnet-50"

    def test_plain_hf_id_with_conflicting_model_id_raises(self):
        """Passing both -m <hf_id> and --model-id is rejected as a conflict."""
        with pytest.raises(click.UsageError, match="Cannot pass both"):
            _resolve_model_path(
                model=("microsoft/resnet-50",),
                model_id="Intel/bert-base-uncased-mrpc",
            )

    def test_plain_hf_id_with_matching_model_id_ok(self):
        """Passing --model-id equal to -m <hf_id> is allowed (no-op duplicate)."""
        path, mid = _resolve_model_path(
            model=("microsoft/resnet-50",),
            model_id="microsoft/resnet-50",
        )
        assert path is None
        assert mid == "microsoft/resnet-50"

    def test_plain_onnx_with_model_id(self, onnx_file):
        path, mid = _resolve_model_path(
            model=(str(onnx_file),),
            model_id="microsoft/resnet-50",
        )
        assert path == str(onnx_file)
        assert mid == "microsoft/resnet-50"

    def test_plain_onnx_without_model_id_raises(self, onnx_file):
        with pytest.raises(click.UsageError, match="--model-id is required"):
            _resolve_model_path(model=(str(onnx_file),), model_id=None)

    def test_plain_onnx_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does-not-exist.onnx"
        with pytest.raises(click.BadParameter, match="ONNX file not found"):
            _resolve_model_path(model=(str(missing),), model_id="some/id")

    def test_multiple_plain_raises(self, onnx_file):
        """Multiple plain -m values without role=path are ambiguous."""
        with pytest.raises(click.UsageError, match="role=path"):
            _resolve_model_path(
                model=(str(onnx_file), str(onnx_file)),
                model_id="some/id",
            )


# ---------------------------------------------------------------------------
# Composite -m role=path
# ---------------------------------------------------------------------------


class TestComposite:
    def test_two_roles(self, onnx_vision, onnx_text):
        path, mid = _resolve_model_path(
            model=(
                f"image-encoder={onnx_vision}",
                f"text-encoder={onnx_text}",
            ),
            model_id="openai/clip-vit-base-patch32",
        )
        assert path == {
            "image-encoder": str(onnx_vision),
            "text-encoder": str(onnx_text),
        }
        assert mid == "openai/clip-vit-base-patch32"

    def test_composite_requires_model_id(self, onnx_vision, onnx_text):
        with pytest.raises(click.UsageError, match="--model-id is required"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"text-encoder={onnx_text}",
                ),
                model_id=None,
            )

    def test_duplicate_roles_raise(self, onnx_vision, onnx_text):
        with pytest.raises(click.BadParameter, match="Duplicate role"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"image-encoder={onnx_text}",
                ),
                model_id="some/id",
            )

    def test_missing_path_raises(self, onnx_vision, tmp_path):
        missing = tmp_path / "no.onnx"
        with pytest.raises(click.BadParameter, match="ONNX file not found"):
            _resolve_model_path(
                model=(
                    f"image-encoder={onnx_vision}",
                    f"text-encoder={missing}",
                ),
                model_id="some/id",
            )

    def test_empty_role_raises(self, onnx_vision):
        with pytest.raises(click.BadParameter, match="role and path"):
            _resolve_model_path(
                model=(f"={onnx_vision}",),
                model_id="some/id",
            )

    def test_empty_path_raises(self):
        with pytest.raises(click.BadParameter, match="role and path"):
            _resolve_model_path(
                model=("image-encoder=",),
                model_id="some/id",
            )

    def test_whitespace_stripped(self, onnx_vision):
        """Role and path are trimmed of surrounding whitespace."""
        path, _mid = _resolve_model_path(
            model=(f"  image-encoder  =  {onnx_vision}  ",),
            model_id="some/id",
        )
        assert path == {"image-encoder": str(onnx_vision)}


# ---------------------------------------------------------------------------
# Mixing forms
# ---------------------------------------------------------------------------


class TestMixedForms:
    def test_plain_and_role_path_mixed_raises(self, onnx_file, onnx_vision):
        with pytest.raises(click.UsageError, match="Cannot mix"):
            _resolve_model_path(
                model=(str(onnx_file), f"text-encoder={onnx_vision}"),
                model_id="some/id",
            )


# ---------------------------------------------------------------------------
# Config precedence (CLI > config file > dataclass defaults)
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestEvalHelp:
    def test_model_help_mentions_onnx_model_id_and_role_path(self, runner: CliRunner):
        from winml.modelkit.commands.eval import eval as eval_cmd

        result = runner.invoke(eval_cmd, ["--help"])

        assert result.exit_code == 0, result.output
        assert "requires --model-id" in result.output
        assert "role=path" in result.output


@pytest.fixture
def eval_config_file(tmp_path):
    config = {
        "loader": {
            "task": "feature-extraction",
        },
        "eval": {
            "task": "image-classification",
            "device": "gpu",
            "dataset": {
                "path": "timm/mini-imagenet",
                "split": "test",
                "samples": 33,
            },
        },
    }
    cfg_path = tmp_path / "eval_config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return cfg_path


class TestEvalConfigPrecedence:
    def test_cli_overrides_config_and_config_overrides_defaults(
        self,
        runner: CliRunner,
        eval_config_file,
    ):
        """Validate precedence: CLI > config file > dataclass defaults."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        captured_cfg = {}

        def _fake_evaluate(cfg):
            captured_cfg["cfg"] = cfg

            class _FakeResult:
                def __init__(self, config):
                    self.config = config
                    self.metrics = {"accuracy": 1.0}

                def to_dict(self):
                    return {
                        "metrics": self.metrics,
                        "config": self.config.to_dict(),
                    }

            return _FakeResult(cfg)

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=_fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(
                eval_cmd,
                [
                    "--config",
                    str(eval_config_file),
                    "-m",
                    "microsoft/resnet-50",
                    "--device",
                    "cpu",
                    "--samples",
                    "7",
                    "--split",
                    "train",
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        cfg = captured_cfg["cfg"]

        # CLI > config
        assert cfg.device == "cpu"
        assert cfg.dataset.samples == 7
        assert cfg.dataset.split == "train"

        # config > dataclass defaults (task default is None)
        assert cfg.task == "image-classification"

    def test_cli_default_device_propagates_when_not_explicitly_passed(
        self,
        runner: CliRunner,
    ):
        """The CLI option default must win over any (stale) dataclass default.

        Even when the user doesn't pass ``--device``, the CLI's default value
        ("auto") must be the effective config — never a different dataclass
        default. This guards against the bug where ``--device`` was sourced
        from ``ParameterSource.DEFAULT`` and silently dropped.
        """
        from winml.modelkit.commands.eval import eval as eval_cmd

        captured_cfg = {}

        def _fake_evaluate(cfg):
            captured_cfg["cfg"] = cfg

            class _R:
                config = cfg
                metrics = {"accuracy": 1.0}  # noqa: RUF012

                def to_dict(self):
                    return {"metrics": self.metrics, "config": cfg.to_dict()}

            return _R()

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=_fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(
                eval_cmd,
                ["-m", "microsoft/resnet-50", "--task", "image-classification"],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        cfg = captured_cfg["cfg"]
        # Find CLI default; ``--device`` was not passed, so the resolved
        # config must equal the CLI default (not, e.g., a stale "cpu" dataclass default).
        cli_default = next(p.default for p in eval_cmd.params if p.name == "device")
        assert cfg.device == cli_default

    def test_config_file_device_wins_over_cli_default(
        self,
        runner: CliRunner,
        eval_config_file,
    ):
        """Config-file values must override CLI defaults (but not CLI explicit)."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        captured_cfg = {}

        def _fake_evaluate(cfg):
            captured_cfg["cfg"] = cfg

            class _R:
                config = cfg
                metrics = {"accuracy": 1.0}  # noqa: RUF012

                def to_dict(self):
                    return {"metrics": self.metrics, "config": cfg.to_dict()}

            return _R()

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=_fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(
                eval_cmd,
                ["--config", str(eval_config_file), "-m", "microsoft/resnet-50"],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        cfg = captured_cfg["cfg"]
        # eval_config_file sets device: "gpu"; CLI --device not passed -> "gpu"
        assert cfg.device == "gpu"
        # And config-file dataset.samples (33) wins over CLI default
        assert cfg.dataset.samples == 33

    @pytest.mark.parametrize(
        ("extra_args", "expected"),
        [(["--allow-unsupported-nodes"], True), ([], False)],
    )
    def test_allow_unsupported_nodes_flag_propagates(
        self,
        runner: CliRunner,
        extra_args,
        expected,
    ):
        """``--allow-unsupported-nodes`` maps to the eval config field."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        captured_cfg = {}

        def _fake_evaluate(cfg):
            captured_cfg["cfg"] = cfg

            class _R:
                config = cfg
                metrics = {"accuracy": 1.0}  # noqa: RUF012

                def to_dict(self):
                    return {"metrics": self.metrics, "config": cfg.to_dict()}

            return _R()

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=_fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(
                eval_cmd,
                ["-m", "microsoft/resnet-50", "--task", "image-classification", *extra_args],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        assert captured_cfg["cfg"].allow_unsupported_nodes is expected


# ---------------------------------------------------------------------------
# --label-mapping wiring (Click Path → label_mapping_file str)
# ---------------------------------------------------------------------------


class TestLabelMappingWiring:
    """``--label-mapping`` is a Click ``Path`` that must land in
    ``cfg.dataset.label_mapping_file`` (a ``str``), NOT in
    ``cfg.dataset.label_mapping`` (the *parsed* ``dict[str, int] | None``).

    The Click param name is ``label_mapping_path`` (distinct from the
    ``DatasetConfig.label_mapping`` field) precisely so
    ``cli_utils.collect_cli_overrides`` doesn't accidentally pass a Path
    into the dict field. This test locks in that wiring.
    """

    def test_label_mapping_path_routes_to_file_field_not_dict_field(
        self,
        runner: CliRunner,
        tmp_path,
    ):
        """--label-mapping <file> must set cfg.dataset.label_mapping_file (str)
        and leave cfg.dataset.label_mapping (dict) untouched at this stage."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        # Sentinel mapping file; existence matters because Click validates the path.
        label_file = tmp_path / "labels.json"
        label_file.write_text(json.dumps({"cat": 0, "dog": 1}), encoding="utf-8")

        captured_cfg: dict = {}

        def _fake_evaluate(cfg):
            captured_cfg["cfg"] = cfg

            class _R:
                config = cfg
                metrics = {"accuracy": 1.0}  # noqa: RUF012

                def to_dict(self):
                    return {"metrics": self.metrics, "config": cfg.to_dict()}

            return _R()

        with (
            patch("winml.modelkit.eval.evaluate", side_effect=_fake_evaluate),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch(
                "winml.modelkit.commands.eval._resolve_label_mapping",
                return_value=None,
            ),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(
                eval_cmd,
                [
                    "-m",
                    "microsoft/resnet-50",
                    "--task",
                    "image-classification",
                    "--label-mapping",
                    str(label_file),
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        cfg = captured_cfg["cfg"]

        # The CLI Path must land in label_mapping_file as a str — the field
        # is serialized via to_dict(), so a Path would break JSON output.
        assert cfg.dataset.label_mapping_file == str(label_file)
        assert isinstance(cfg.dataset.label_mapping_file, str)

        # label_mapping is the *parsed* dict and must stay at its default
        # (None) until _resolve_label_mapping loads it at eval time. If the
        # Click Path ever leaks into this field, this assertion fails — that
        # was the bug introduced when ``collect_cli_overrides`` saw a Click
        # param named ``label_mapping`` matching a same-named dataclass field.
        assert cfg.dataset.label_mapping is None


# ---------------------------------------------------------------------------
# Per-task default dataset resolution
# ---------------------------------------------------------------------------


class TestPerTaskDefaultDataset:
    """When the user does not provide --dataset, the per-task default dataset
    (path, split, columns_mapping, ...) must reach the evaluator. Only
    ``samples`` is carried over from the user's CLI value.

    Regression guard for the bug where Click's ``--split validation`` default
    silently clobbered the per-task default split (e.g. coco→"val",
    cifar100→"test"), making the per-task split values dead code.
    """

    @staticmethod
    def _run_and_capture(runner: CliRunner, args: list[str]):
        """Invoke the eval CLI, letting ``evaluate()`` run end-to-end with
        ``_load_model`` and the evaluator class stubbed. Returns the cfg
        observed by the evaluator (i.e. after default-injection)."""
        import importlib

        from winml.modelkit.commands.eval import eval as eval_cmd

        evaluate_mod = importlib.import_module("winml.modelkit.eval.evaluate")

        captured_cfg = {}

        class _FakeEvaluator:
            def __init__(self, cfg, _model):
                captured_cfg["cfg"] = cfg

            def compute(self):
                return {"accuracy": 1.0}

        with (
            patch.object(evaluate_mod, "_load_model", return_value=object()),
            patch.object(
                evaluate_mod,
                "get_evaluator_class",
                return_value=_FakeEvaluator,
            ),
            patch("winml.modelkit.commands.eval._resolve_device", return_value=None),
            patch("winml.modelkit.commands.eval._write_and_display", return_value=None),
        ):
            result = runner.invoke(eval_cmd, args, obj={"debug": False})

        assert result.exit_code == 0, result.output
        return captured_cfg["cfg"]

    @pytest.mark.parametrize(
        ("task", "expected_path", "expected_split"),
        [
            ("object-detection", "detection-datasets/coco", "val"),
            ("zero-shot-classification", "fancyzhx/ag_news", "test"),
            ("zero-shot-image-classification", "uoft-cs/cifar100", "test"),
            ("image-classification", "timm/mini-imagenet", "test"),
        ],
    )
    def test_per_task_default_split_reaches_evaluator(
        self,
        runner: CliRunner,
        task: str,
        expected_path: str,
        expected_split: str,
    ):
        cfg = self._run_and_capture(
            runner,
            ["-m", "some/model", "--task", task],
        )
        assert cfg.dataset.path == expected_path
        assert cfg.dataset.split == expected_split

    def test_user_samples_preserved_when_default_dataset_used(
        self,
        runner: CliRunner,
    ):
        """``--samples N`` must NOT be clobbered by the per-task default's samples
        when the user didn't provide ``--dataset``.
        """
        cfg = self._run_and_capture(
            runner,
            [
                "-m",
                "some/model",
                "--task",
                "image-classification",
                "--samples",
                "4",
            ],
        )
        # Per-task default path filled in:
        assert cfg.dataset.path == "timm/mini-imagenet"
        # ...but user-set --samples preserved.
        assert cfg.dataset.samples == 4

    def test_user_split_ignored_when_default_dataset_used(
        self,
        runner: CliRunner,
    ):
        """When falling back to the per-task default dataset, the default owns
        the split. ``--split`` is intentionally ignored (only ``samples`` is
        carried over from the user). Users wanting a different split must
        also pass ``--dataset``.
        """
        cfg = self._run_and_capture(
            runner,
            [
                "-m",
                "some/model",
                "--task",
                "image-classification",
                "--split",
                "train",
            ],
        )
        assert cfg.dataset.split == "test"  # the default's split wins

    def test_user_column_ignored_when_default_dataset_used(
        self,
        runner: CliRunner,
    ):
        """``--column`` overrides are ignored when the default dataset fills
        in. The default owns ``columns_mapping`` wholesale. To customize
        columns, the user must also pass ``--dataset``.
        """
        cfg = self._run_and_capture(
            runner,
            [
                "-m",
                "some/model",
                "--task",
                "text-classification",
                "--column",
                "input_column=my_text",
            ],
        )
        # Default's columns_mapping wins wholesale; user's --column dropped.
        assert cfg.dataset.columns_mapping == {
            "input_column": "sentence1",
            "second_input_column": "sentence2",
        }

    def test_user_streaming_ignored_when_default_dataset_used(
        self,
        runner: CliRunner,
    ):
        """``--streaming`` is ignored when the default dataset fills in;
        the default's ``streaming`` value wins.
        """
        # fill-mask default has streaming=True; user passing nothing should
        # still get streaming=True (from default), not the Click default False.
        cfg = self._run_and_capture(
            runner,
            ["-m", "some/model", "--task", "fill-mask"],
        )
        assert cfg.dataset.streaming is True

    def test_user_dataset_name_ignored_when_default_dataset_used(
        self,
        runner: CliRunner,
    ):
        """``--dataset-name`` is ignored when the default dataset fills in.
        Only ``samples`` is carried over from the user's config.
        """
        cfg = self._run_and_capture(
            runner,
            [
                "-m",
                "some/model",
                "--task",
                "text-classification",
                "--dataset-name",
                "sst2",
            ],
        )
        # Default's name ("mrpc") wins; user's --dataset-name dropped.
        assert cfg.dataset.name == "mrpc"

    def test_default_dataset_logs_warning(
        self,
        runner: CliRunner,
        caplog,
    ):
        """When falling back to the default dataset, a warning is emitted
        listing the default and the ignored options.
        """
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="winml.modelkit.eval.evaluate"):
            self._run_and_capture(
                runner,
                ["-m", "some/model", "--task", "image-classification"],
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "--dataset not specified" in m
            and "image-classification" in m
            and "timm/mini-imagenet" in m
            for m in msgs
        ), f"expected warning not found in {msgs!r}"


# ---------------------------------------------------------------------------
# --format json
# ---------------------------------------------------------------------------


class TestEvalFormatJson:
    """Test --format json produces structured JSON to stdout."""

    def test_format_json_produces_valid_json(self):
        """_write_and_display with json_mode=True emits parseable JSON."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.eval import _write_and_display

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "mode": "onnx",
            "model_id": "microsoft/resnet-50",
            "metrics": {"top1_accuracy": 0.741},
        }

        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            import io

            buf = io.StringIO()
            with patch(
                "winml.modelkit.commands.eval.click.echo",
                side_effect=lambda x: buf.write(x),
            ):
                _write_and_display(mock_result, None, json_mode=True)

            output = buf.getvalue()
            parsed = json.loads(output)
            assert parsed["model_id"] == "microsoft/resnet-50"
            assert parsed["metrics"]["top1_accuracy"] == 0.741

    def test_format_json_with_output_file(self, tmp_path):
        """--format json + --output should emit JSON to stdout AND save file."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.eval import _write_and_display

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "mode": "onnx",
            "model_id": "test/model",
            "metrics": {"accuracy": 0.9},
        }

        output_file = tmp_path / "result.json"

        import io

        buf = io.StringIO()
        with patch("winml.modelkit.commands.eval.click.echo", side_effect=lambda x: buf.write(x)):
            _write_and_display(mock_result, output_file, json_mode=True)

        # stdout has JSON
        parsed = json.loads(buf.getvalue())
        assert parsed["model_id"] == "test/model"

        # File also has JSON
        assert output_file.exists()
        file_data = json.loads(output_file.read_text())
        assert file_data["model_id"] == "test/model"

    def test_format_text_shows_report(self):
        """json_mode=False should call display_eval_report (default behavior)."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.eval import _write_and_display

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"metrics": {}}
        mock_result.config.model_id = "test"
        mock_result.config.task = "cls"
        mock_result.config.device = "cpu"
        mock_result.config.dataset.path = None
        mock_result.config.dataset.samples = 100
        mock_result.config.model_path = None
        mock_result.metrics = {}

        with patch("winml.modelkit.commands.eval.display_eval_report") as mock_display:
            _write_and_display(mock_result, None, json_mode=False)
            mock_display.assert_called_once()

    def test_help_shows_format_option(self, runner: CliRunner):
        """--format flag must appear in --help output."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        result = runner.invoke(eval_cmd, ["--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "json" in result.output

    def test_invalid_format_rejected(self, runner: CliRunner):
        """An invalid --format value must be rejected by Click."""
        from winml.modelkit.commands.eval import eval as eval_cmd

        result = runner.invoke(eval_cmd, ["-m", "test", "--format", "xml"])
        assert result.exit_code != 0
