"""Tests for module summary report generation."""

import json
from pathlib import Path

from winml.modelkit.build.module_summary import write_module_summary


class TestModuleSummary:
    def test_write_summary(self, tmp_path: Path) -> None:
        instances = [
            {
                "module_path": "encoder.layer.0.attention",
                "class_name": "BertAttention",
                "output_dir": "BertAttention_0",
                "build_elapsed_s": 5.2,
            },
            {
                "module_path": "encoder.layer.1.attention",
                "class_name": "BertAttention",
                "output_dir": "BertAttention_1",
                "build_elapsed_s": 4.8,
            },
        ]

        output_path = tmp_path / "module_summary.json"
        write_module_summary(
            output_path=output_path,
            model_id="bert-base-uncased",
            module_class="BertAttention",
            instances=instances,
        )

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["model_id"] == "bert-base-uncased"
        assert data["module_class"] == "BertAttention"
        assert data["instance_count"] == 2
        assert len(data["instances"]) == 2
        assert data["instances"][0]["module_path"] == "encoder.layer.0.attention"

    def test_write_summary_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "dir" / "summary.json"
        write_module_summary(
            output_path=output_path,
            model_id="test",
            module_class="Test",
            instances=[],
        )
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["instance_count"] == 0
