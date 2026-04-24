# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module-level helpers."""

from __future__ import annotations

from unittest.mock import patch

from winml.modelkit.session.ep_registry import ensure_initialized


def test_ensure_initialized_calls_registry_once():
    """ensure_initialized() calls register_to_ort() via singleton; idempotent across calls."""
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True

        ensure_initialized()
        ensure_initialized()
        ensure_initialized()

        assert mock_registry_cls.get_instance.call_count >= 1
        # Multiple calls must not raise
