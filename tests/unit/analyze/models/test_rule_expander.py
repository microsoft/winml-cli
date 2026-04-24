# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import zipfile
from pathlib import Path

from winml.modelkit.analyze.utils.rule_expander import EXPANDED_MARKER_FILE, expand_rules_zip_dir


def _write_json_zip(zip_path: Path, entry_name: str, payload: dict) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(entry_name, json.dumps(payload))


def test_expand_rules_zip_dir_in_place(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    base_zip = rules_dir / "EP_CPU_ai.onnx_opset12.zip"
    delta_zip = rules_dir / "EP_CPU_ai.onnx_opset13.zip"
    base_entry = "EP_CPU_ai.onnx_opset12_negative_rules.json"
    delta_entry = "EP_CPU_ai.onnx_opset13_negative_rules.json"

    _write_json_zip(
        base_zip,
        base_entry,
        {
            "Conv": {"v": 1},
            "Add": {"v": 2},
        },
    )
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

    summary = expand_rules_zip_dir(rules_dir)

    assert summary.zip_files_processed == 2
    assert summary.zip_files_with_delta == 1
    assert summary.delta_entries_materialized == 1
    assert summary.output_mode.startswith("in-place")

    with zipfile.ZipFile(delta_zip, "r") as zf:
        payload = json.loads(zf.read(delta_entry).decode("utf-8"))

    assert "__snapshot_type__" not in payload
    assert sorted(payload.keys()) == ["Conv", "Mul"]
    marker = rules_dir / EXPANDED_MARKER_FILE
    assert marker.exists()
    assert marker.is_file()


def test_expand_rules_zip_dir_ignores_temp_materialized_zips(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    real_zip = rules_dir / "EP_CPU_ai.onnx_opset12.zip"
    temp_zip = rules_dir / "EP_CPU_ai.onnx_opset13.materialized.abcd.zip"

    _write_json_zip(real_zip, "EP_CPU_ai.onnx_opset12_negative_rules.json", {"Conv": {"v": 1}})
    _write_json_zip(
        temp_zip,
        "EP_CPU_ai.onnx_opset13_negative_rules.json",
        {
            "__snapshot_type__": "delta_v1",
            "__base_opset__": 12,
            "__current_opset__": 13,
            "__changed__": {"Mul": {"v": 1}},
            "__deleted__": [],
        },
    )

    summary = expand_rules_zip_dir(rules_dir)

    assert summary.zip_files_processed == 1
    assert [item[0] for item in summary.per_zip] == [real_zip.name]
    assert (rules_dir / EXPANDED_MARKER_FILE).exists()


def test_expand_rules_zip_dir_no_zip_does_not_create_marker(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    summary = expand_rules_zip_dir(rules_dir)

    assert summary.zip_files_processed == 0
    assert not (rules_dir / EXPANDED_MARKER_FILE).exists()
