# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from winml.modelkit.config.build import WinMLBuildConfig
from winml.modelkit.export.config import WinMLExportConfig
from winml.modelkit.export.policy import ExportCompatibilityConfig


class TestExportCompatibilityBuildConfig:
    def test_export_compatibility_changes_cache_key(self) -> None:
        default_config = WinMLBuildConfig(export=WinMLExportConfig())
        eager_config = WinMLBuildConfig(
            export=WinMLExportConfig(
                compatibility=ExportCompatibilityConfig(transformers_attention="eager")
            )
        )

        assert default_config.generate_cache_key() != eager_config.generate_cache_key()

    def test_registered_export_merge_preserves_override_compatibility(self) -> None:
        from winml.modelkit.config.build import _merge_export_config

        base = WinMLExportConfig()
        override = WinMLExportConfig(
            compatibility=ExportCompatibilityConfig(transformers_attention="eager")
        )

        merged = _merge_export_config(base, override)

        assert merged.compatibility.transformers_attention == "eager"
