# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E-only CLI entry point with the OpenVINO GPU allocator workaround.

OpenVINO 1.8.95 can expose two hardware handles for the same GPU with different
allocator IDs. Inference on one handle can request the other handle's allocator.
Include ORT environment allocators in E2E sessions until the EP handles this.
The installed CLI and product source retain their normal session behavior.
"""

from __future__ import annotations

import functools
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch


if TYPE_CHECKING:
    from collections.abc import Iterator


def cli_command() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]


@contextmanager
def openvino_gpu_allocators() -> Iterator[None]:
    """Scope the workaround to an E2E invocation, preserving explicit options."""
    from winml.modelkit.session import session as session_module

    original = session_module._build_session_options
    reported = False

    @functools.wraps(original)
    def build_options(ep_device, *args, **kwargs):
        nonlocal reported
        options = original(ep_device, *args, **kwargs)
        handle = ep_device.device.ort_handle
        if handle.ep_name == "OpenVINOExecutionProvider" and handle.device.type.name == "GPU":
            try:
                options.get_session_config_entry("session.use_env_allocators")
            except RuntimeError:
                options.add_session_config_entry("session.use_env_allocators", "1")
                if not reported:
                    print(
                        "[E2E] OpenVINO GPU allocator workaround: session.use_env_allocators=1",
                        file=sys.stderr,
                    )
                    reported = True
        return options

    with patch.object(session_module, "_build_session_options", build_options):
        yield


if __name__ == "__main__":
    from winml.modelkit.cli import main

    with openvino_gpu_allocators():
        main()
