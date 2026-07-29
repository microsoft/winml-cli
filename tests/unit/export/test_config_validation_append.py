# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from winml.modelkit.export.config import WinMLExportConfig
from winml.modelkit.export.policy import ExportCompatibilityConfig


class TestExportCompatibilitySerialization:
    def test_compatibility_round_trips_when_present(self) -> None:
        cfg = WinMLExportConfig(
            compatibility=ExportCompatibilityConfig(transformers_attention="eager")
        )

        data = cfg.to_dict()
        round_tripped = WinMLExportConfig.from_dict(data)

        assert data["compatibility"] == {"transformers_attention": "eager"}
        assert round_tripped.compatibility.transformers_attention == "eager"

    def test_empty_compatibility_is_omitted_from_export_dict(self) -> None:
        cfg = WinMLExportConfig()

        assert "compatibility" not in cfg.to_dict()

    def test_invalid_compatibility_value_raises(self) -> None:
        with pytest.raises(ValueError, match="transformers_attention"):
            WinMLExportConfig.from_dict({"compatibility": {"transformers_attention": "sdpa"}})
