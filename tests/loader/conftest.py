"""Pytest configuration for loader tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (download models)")


@pytest.fixture
def make_mock_config():
    """Factory fixture for mock HF PretrainedConfig objects.

    Returns:
        A factory callable that creates MagicMock objects mimicking
        transformers.PretrainedConfig with ``model_type`` and
        ``architectures`` attributes set.
    """

    def _make(
        model_type: str,
        architectures: list[str] | None = None,
        **kwargs,
    ) -> MagicMock:
        config = MagicMock()
        config.model_type = model_type
        config.architectures = architectures
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config

    return _make
