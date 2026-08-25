# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security tests for the serve command's Uvicorn configuration."""

from __future__ import annotations

from typing import Any

import pytest
import uvicorn
from click.testing import CliRunner

from winml.modelkit.commands.serve import serve


@pytest.mark.parametrize("args", [[], ["--multi"]])
def test_serve_disables_proxy_headers(
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls: list[dict[str, Any]] = []

    def record_run(_app: Any, **kwargs: Any) -> None:
        run_calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", record_run)

    result = CliRunner().invoke(serve, args)

    assert result.exit_code == 0, result.output
    assert len(run_calls) == 1
    assert run_calls[0]["proxy_headers"] is False
