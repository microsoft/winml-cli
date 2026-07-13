# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for HuggingFace hierarchy extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch import nn

from winml.modelkit.inspect.hierarchy import extract_hierarchy


@pytest.mark.parametrize(
    ("kwargs", "trust_remote_code"),
    [
        ({}, False),
        ({"trust_remote_code": True}, True),
    ],
)
def test_extract_hierarchy_forwards_remote_code_consent(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, bool],
    trust_remote_code: bool,
) -> None:
    """Every hierarchy load must use the caller's explicit trust setting."""
    from winml.modelkit.inspect import hierarchy

    pretrained_loader = MagicMock(side_effect=OSError("not cached"))
    config_loader = MagicMock(return_value=object())
    model_loader = MagicMock(return_value=nn.Linear(1, 1))
    monkeypatch.setattr(hierarchy.AutoModel, "from_pretrained", pretrained_loader)
    monkeypatch.setattr(hierarchy.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(hierarchy.AutoModel, "from_config", model_loader)

    extract_hierarchy("test/model", **kwargs)

    pretrained_loader.assert_called_once_with(
        "test/model",
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    config_loader.assert_called_once_with("test/model", trust_remote_code=trust_remote_code)
    model_loader.assert_called_once_with(
        config_loader.return_value,
        trust_remote_code=trust_remote_code,
    )
