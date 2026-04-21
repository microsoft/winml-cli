# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for the LoggerProvider factory."""

from collections.abc import Callable

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.resources import Resource

from winml.modelkit.telemetry.library import (
    DEFAULT_ENDPOINT,
    create_logger_provider,
)


@pytest.fixture
def make_provider() -> Callable[..., LoggerProvider]:
    """Build providers that are guaranteed to shut down after the test.

    `create_logger_provider` wires a `BatchLogRecordProcessor` which spawns a
    background daemon thread; without `shutdown()` each test leaks one.
    """
    created: list[LoggerProvider] = []

    def _make(**kwargs) -> LoggerProvider:
        provider = create_logger_provider(**kwargs)
        created.append(provider)
        return provider

    yield _make

    for provider in created:
        provider.shutdown()


def test_default_endpoint_points_at_one_collector():
    assert DEFAULT_ENDPOINT == ("https://mobile.events.data.microsoft.com/OneCollector/1.0/")


def test_create_logger_provider_returns_configured_provider(make_provider):
    resource = Resource.create({"app_version": "0.0.1"})
    provider = make_provider(ikey="o:test", resource=resource)
    assert isinstance(provider, LoggerProvider)
    assert provider.resource.attributes.get("app_version") == "0.0.1"


def test_create_logger_provider_with_custom_endpoint(make_provider):
    provider = make_provider(
        ikey="o:test",
        endpoint="https://example.invalid/OneCollector/1.0/",
    )
    assert isinstance(provider, LoggerProvider)
