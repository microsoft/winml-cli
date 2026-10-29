# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test doubles for GGUF evidence: a llama log API that calls back through ctypes."""

from __future__ import annotations

from winml.modelkit.commands._gguf_native_log import LOG_CALLBACK


# Stand-in address of the Runtime adapter's own llama log callback.
RUNTIME_CALLBACK = 0x7920


class FakeLogApi:
    """In-memory ``llama_log_get``/``llama_log_set`` pair.

    ``emit`` invokes the installed callback through a real ctypes function
    pointer, exactly as native code would.
    """

    def __init__(self) -> None:
        self.callback: int | None = RUNTIME_CALLBACK
        self.user_data: int | None = None
        self.set_calls: list[tuple[int | None, int | None]] = []

    def get(self) -> tuple[int | None, int | None]:
        return self.callback, self.user_data

    def set(self, callback: int | None, user_data: int | None) -> None:
        self.set_calls.append((callback, user_data))
        self.callback = callback
        self.user_data = user_data

    def emit(self, level: int, text: bytes | None) -> None:
        assert self.callback not in (None, RUNTIME_CALLBACK), "no capture callback is installed"
        LOG_CALLBACK(self.callback)(level, text, None)
