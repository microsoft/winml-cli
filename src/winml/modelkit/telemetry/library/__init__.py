# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Public surface for the OneCollector client.

Consumers (the rest of the telemetry module) only import from this file.
Everything else in `library/` is private implementation detail.
"""

from __future__ import annotations

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from .exporter import OneCollectorLogExporter


__all__ = ["DEFAULT_ENDPOINT", "OneCollectorLogExporter", "create_logger_provider"]

DEFAULT_ENDPOINT = "https://mobile.events.data.microsoft.com/OneCollector/1.0/"


def create_logger_provider(
    ikey: str,
    endpoint: str = DEFAULT_ENDPOINT,
    resource: Resource | None = None,
) -> LoggerProvider:
    """Create a LoggerProvider wired to a OneCollectorLogExporter.

    The returned provider's `get_logger(name)` produces loggers whose
    `.emit(LogRecord(...))` calls are batched and sent to the OneCollector
    endpoint. Shutdown the provider to flush pending events.
    """
    provider = LoggerProvider(resource=resource or Resource.get_empty())
    exporter = OneCollectorLogExporter(ikey=ikey, endpoint=endpoint)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    return provider
