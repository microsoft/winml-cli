# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLManifest dataclass (utils/manifest.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

import pytest

from winml.modelkit.utils.manifest import (
    MANIFEST_FILENAME,
    ManifestStage,
    WinMLManifest,
)


class TestManifestStage:
    """ManifestStage construction and serialisation."""

    def test_minimal(self) -> None:
        s = ManifestStage(name="export", status="completed")
        assert s.name == "export"
        assert s.filename is None

    def test_with_quant_metrics(self) -> None:
        s = ManifestStage(
            name="quantize",
            status="completed",
            filename="quantized.onnx",
            elapsed_seconds=1.5,
            nodes_quantized=42,
            nodes_skipped=3,
        )
        assert s.nodes_quantized == 42
        assert s.nodes_skipped == 3


class TestWinMLManifestRoundTrip:
    """Serialise → deserialise round-trip."""

    def test_to_dict_injects_schema_version(self) -> None:
        m = WinMLManifest(source="hf", final_artifact="model.onnx")
        d = m.to_dict()
        assert d["schema_version"] == 1
        assert d["source"] == "hf"

    def test_none_fields_omitted(self) -> None:
        m = WinMLManifest(source="export", final_artifact="model.onnx")
        d = m.to_dict()
        assert "model_id" not in d
        assert "cache_key" not in d

    def test_stage_none_fields_omitted(self) -> None:
        m = WinMLManifest(
            source="hf",
            final_artifact="model.onnx",
            stages=[ManifestStage(name="optimize", status="skipped")],
        )
        d = m.to_dict()
        stage = d["stages"][0]
        assert "filename" not in stage
        assert "elapsed_seconds" not in stage

    def test_round_trip(self) -> None:
        original = WinMLManifest(
            source="hf",
            model_id="microsoft/resnet-50",
            task="image-classification",
            final_artifact="model.onnx",
            elapsed_seconds=12.345,
            stages=[
                ManifestStage(
                    name="export",
                    status="completed",
                    filename="export.onnx",
                    elapsed_seconds=5.0,
                ),
                ManifestStage(name="quantize", status="skipped"),
            ],
        )
        d = original.to_dict()
        restored = WinMLManifest.from_dict(d)
        assert restored.source == original.source
        assert restored.model_id == original.model_id
        assert restored.task == original.task
        assert restored.elapsed_seconds == original.elapsed_seconds
        assert len(restored.stages) == 2
        assert restored.stages[0].name == "export"
        assert restored.stages[0].filename == "export.onnx"
        assert restored.stages[1].status == "skipped"

    def test_forward_compat_extras(self) -> None:
        """Unknown keys in JSON land in ``extras`` instead of crashing."""
        d = {
            "schema_version": 1,
            "source": "hf",
            "final_artifact": "model.onnx",
            "future_field": "hello",
        }
        m = WinMLManifest.from_dict(d)
        assert m.extras["future_field"] == "hello"

    def test_forward_compat_stage_extras(self) -> None:
        """Unknown keys inside a stage land in stage ``extras``."""
        d = {
            "schema_version": 1,
            "source": "hf",
            "final_artifact": "model.onnx",
            "stages": [
                {
                    "name": "export",
                    "status": "completed",
                    "future_metric": 42,
                }
            ],
        }
        m = WinMLManifest.from_dict(d)
        assert m.stages[0].extras["future_metric"] == 42
        # Round-trip preserves the extra
        rt = WinMLManifest.from_dict(m.to_dict())
        assert rt.stages[0].extras["future_metric"] == 42


class TestWinMLManifestIO:
    """File I/O helpers."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        m = WinMLManifest(
            source="onnx",
            final_artifact="model.onnx",
            input_onnx="input.onnx",
            stages=[
                ManifestStage(
                    name="optimize",
                    status="completed",
                    filename="optimized.onnx",
                    elapsed_seconds=2.0,
                )
            ],
        )
        path = tmp_path / MANIFEST_FILENAME
        m.save(path)
        assert path.exists()

        loaded = WinMLManifest.load(path)
        assert loaded.source == "onnx"
        assert loaded.input_onnx == "input.onnx"
        assert len(loaded.stages) == 1

    def test_save_sanitizes_path_objects(self, tmp_path: Path) -> None:
        """Path values in export_stats are coerced to strings, not dropped."""
        m = WinMLManifest(
            source="export",
            final_artifact="model.onnx",
            export_stats={"output_path": tmp_path / "model.onnx", "count": 5},
        )
        path = tmp_path / MANIFEST_FILENAME
        m.save(path)
        data = json.loads(path.read_text())
        assert isinstance(data["export_stats"]["output_path"], str)
        assert data["export_stats"]["count"] == 5  # numeric stays numeric

    def test_save_sanitizes_numpy_scalars(self, tmp_path: Path) -> None:
        """Numpy scalars in export_stats are coerced to native Python types."""
        np = pytest.importorskip("numpy")
        m = WinMLManifest(
            source="export",
            final_artifact="model.onnx",
            export_stats={
                "accuracy": np.float32(0.95),
                "count": np.int64(42),
                "flag": np.bool_(True),
            },
        )
        path = tmp_path / MANIFEST_FILENAME
        m.save(path)
        data = json.loads(path.read_text())
        assert data["export_stats"]["accuracy"] == pytest.approx(0.95, abs=1e-5)
        assert data["export_stats"]["count"] == 42
        assert isinstance(data["export_stats"]["count"], int)
        assert data["export_stats"]["flag"] is True

    def test_manifest_path_for_plain(self, tmp_path: Path) -> None:
        p = WinMLManifest.manifest_path_for(tmp_path)
        assert p.name == MANIFEST_FILENAME

    def test_manifest_path_for_prefixed(self, tmp_path: Path) -> None:
        p = WinMLManifest.manifest_path_for(tmp_path, prefix="imgcls_abc123")
        assert p.name == "imgcls_abc123_build_manifest.json"

    def test_find_discovers_multiple(self, tmp_path: Path) -> None:
        for name in ["build_manifest.json", "feat_aaa_build_manifest.json"]:
            (tmp_path / name).write_text(
                json.dumps({"schema_version": 1, "source": "hf", "final_artifact": "model.onnx"})
            )
        found = WinMLManifest.find(tmp_path)
        assert len(found) == 2

    def test_find_skips_corrupt(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text("not json{{{")
        found = WinMLManifest.find(tmp_path)
        assert found == []

    def test_json_on_disk_matches_schema(self, tmp_path: Path) -> None:
        """Verify the on-disk JSON has schema_version and is valid."""
        m = WinMLManifest(
            source="export",
            model_id="test/model",
            task="image-classification",
            final_artifact="model.onnx",
            export_stats={"tagged_nodes": 10},
        )
        path = tmp_path / MANIFEST_FILENAME
        m.save(path)
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == 1
        assert raw["source"] == "export"
        assert raw["export_stats"]["tagged_nodes"] == 10
