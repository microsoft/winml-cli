# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end integration test for per-module config generation.

Verifies the full config generation pipeline for submodules without
running actual ONNX build (which requires runtime dependencies).
"""

import json
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_device_resolution() -> None:
    """Keep module-build tests independent of host EP discovery."""
    with (
        patch("winml.modelkit.sysinfo.resolve_device", return_value=("cpu", ["cpu"])),
        patch(
            "winml.modelkit.config.precision.resolve_eps",
            return_value=["CPUExecutionProvider"],
        ),
    ):
        yield


@pytest.mark.slow
class TestModuleConfigE2E:
    """End-to-end: generate_build_config(module=...) produces valid configs."""

    def test_config_module_generates_array_with_module_path(self) -> None:
        """Verify winml config --module outputs a JSON array with module_path."""
        from winml.modelkit.config import generate_build_config

        # Use model_type only (no download, uses default HF config with random weights)
        configs = generate_build_config(
            model_type="bert",
            task="fill-mask",
            module="BertAttention",
        )

        assert isinstance(configs, list)
        assert len(configs) > 0

        # Each config should have inherited parent context (except task)
        for cfg in configs:
            # Submodules don't have tasks — task intentionally omitted
            assert cfg.loader.task is None
            assert cfg.loader.model_type == "bert"
            assert cfg.loader.model_class == "BertAttention"
            assert cfg.loader.module_path is not None
            assert "attention" in cfg.loader.module_path

            # Should pass validation (relaxed for submodules)
            cfg.validate()

        # Serialization roundtrip
        serialized = [cfg.to_dict() for cfg in configs]
        json_str = json.dumps(serialized, indent=2)
        restored = json.loads(json_str)
        assert len(restored) == len(configs)
        for d in restored:
            assert "loader" in d
            assert d["loader"]["module_path"] is not None
            # task is omitted from loader.to_dict() when None
            assert d["loader"]["model_type"] == "bert"

    def test_config_module_no_match_raises_with_available_classes(self) -> None:
        """Module class that doesn't exist raises with discovered classes."""
        from winml.modelkit.config import SubmoduleClassNotFoundError, generate_build_config

        with pytest.raises(SubmoduleClassNotFoundError) as exc_info:
            generate_build_config(
                model_type="bert",
                task="fill-mask",
                module="NonExistentModule",
            )

        assert exc_info.value.class_name == "NonExistentModule"
        assert exc_info.value.available_classes
