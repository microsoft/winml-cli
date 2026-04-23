# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import zipfile
from pathlib import Path

from click.testing import CliRunner

from winml.modelkit.analyze.utils.rule_expander import EXPANDED_MARKER_FILE
from winml.modelkit.cli import main


def _write_json_zip(zip_path: Path, entry_name: str, payload: dict) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(entry_name, json.dumps(payload))


def test_expand_rules_command_expands_in_place_from_env(
    tmp_path: Path, monkeypatch
) -> None:
    rules_dir = tmp_path / "rules_zip"
    rules_dir.mkdir(parents=True, exist_ok=True)

    base_zip = rules_dir / "EP_CPU_ai.onnx_opset12.zip"
    delta_zip = rules_dir / "EP_CPU_ai.onnx_opset13.zip"
    base_entry = "EP_CPU_ai.onnx_opset12_negative_rules.json"
    delta_entry = "EP_CPU_ai.onnx_opset13_negative_rules.json"

    _write_json_zip(base_zip, base_entry, {"Conv": {"v": 1}, "Add": {"v": 2}})
    _write_json_zip(
        delta_zip,
        delta_entry,
        {
            "__snapshot_type__": "delta_v1",
            "__base_opset__": 12,
            "__current_opset__": 13,
            "__changed__": {"Mul": {"v": 3}},
            "__deleted__": ["Add"],
        },
    )

    monkeypatch.setenv("MODELKIT_RULES_DIR", str(rules_dir))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["expand_rules"],
    )

    assert result.exit_code == 0, result.output
    assert "zip_files_processed: 2" in result.output
    assert (rules_dir / EXPANDED_MARKER_FILE).exists()

    with zipfile.ZipFile(delta_zip, "r") as zf:
        payload = json.loads(zf.read(delta_entry).decode("utf-8"))

    assert "__snapshot_type__" not in payload
    assert sorted(payload.keys()) == ["Conv", "Mul"]


def test_expand_rules_command_skips_when_dir_missing(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing_rules_zip"
    monkeypatch.setenv("MODELKIT_RULES_DIR", str(missing))
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["expand_rules"],
    )

    assert result.exit_code == 0
    assert "does not exist, skip" in result.output


def test_expand_rules_command_skips_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("MODELKIT_RULES_DIR", raising=False)
    runner = CliRunner()

    result = runner.invoke(main, ["expand_rules"])

    assert result.exit_code == 0
    assert "MODELKIT_RULES_DIR is not set" in result.output
