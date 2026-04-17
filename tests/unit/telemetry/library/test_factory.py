# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for the LoggerProvider factory."""

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.resources import Resource

from winml.modelkit.telemetry.library import (
    DEFAULT_ENDPOINT,
    create_logger_provider,
)


def test_default_endpoint_points_at_one_collector():
    assert DEFAULT_ENDPOINT == ("https://mobile.events.data.microsoft.com/OneCollector/1.0/")


def test_create_logger_provider_returns_configured_provider():
    resource = Resource.create({"app_version": "0.0.1"})
    provider = create_logger_provider(ikey="o:test", resource=resource)
    assert isinstance(provider, LoggerProvider)
    assert provider.resource.attributes.get("app_version") == "0.0.1"


def test_create_logger_provider_with_custom_endpoint():
    provider = create_logger_provider(
        ikey="o:test",
        endpoint="https://example.invalid/OneCollector/1.0/",
    )
    assert isinstance(provider, LoggerProvider)
