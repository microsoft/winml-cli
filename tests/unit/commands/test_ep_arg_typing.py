# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression pin for T-01 widening: click callbacks on the 4 commands
that wire ``--ep`` through :class:`EpAtSourceParamType` must annotate
the callback's ``ep`` parameter with the tuple return type of
``EpAtSourceParamType.convert()`` — not ``str | None``.

If the annotation is wrong AND any code in that command's body
consumes ``ep`` before unpacking (e.g. ``f"EP: {ep}"``), the render
leaks a tuple like ``EP: ('qnn', 'pypi')`` into user-visible output.
"""
from __future__ import annotations

import importlib
import inspect
from typing import get_type_hints

import pytest
from click.testing import CliRunner


COMMANDS_TO_CHECK = [
    ("winml.modelkit.commands.perf", "perf"),
    ("winml.modelkit.commands.compile", "compile"),
    ("winml.modelkit.commands.build", "build"),
    ("winml.modelkit.commands.config", "config"),
]


@pytest.mark.parametrize("module_path,cmd_name", COMMANDS_TO_CHECK)
def test_ep_annotation_matches_tuple_return(module_path: str, cmd_name: str) -> None:
    """The click callback's ``ep`` parameter must be typed as the tuple
    :class:`EpAtSourceParamType.convert()` actually returns.
    """
    mod = importlib.import_module(module_path)
    cmd = getattr(mod, cmd_name)
    fn = cmd.callback
    sig = inspect.signature(fn)
    assert "ep" in sig.parameters, f"{cmd_name} has no `ep` parameter"

    # Resolve stringified annotations (from __future__ annotations).
    try:
        hints = get_type_hints(fn, include_extras=False)
    except Exception:
        hints = {}
    ann = hints.get("ep", sig.parameters["ep"].annotation)
    ann_str = str(ann)

    assert "tuple" in ann_str.lower(), (
        f"{module_path}.{cmd_name}: `ep` annotated as {ann_str!r}; "
        "expected tuple[str, str | None] | None per EpAtSourceParamType.convert()."
    )


def test_no_tuple_leak_in_compile_pre_run_block() -> None:
    """When ``--ep foo@bar`` is provided, the compile command's Rich
    console pre-run block must render the EP as ``foo@bar`` (or better,
    the resolved provider) — never as the raw tuple ``('foo', 'bar')``.

    Uses a helper stub for compile_onnx so we don't need real EPs on the
    host; the goal is to exercise the Rich block and grep the output.
    """
    from unittest.mock import MagicMock, patch

    from winml.modelkit.commands.compile import compile as compile_cmd

    runner = CliRunner()

    # Stub the heavy path: resolver, config factory, compile_onnx. We just
    # want the pre-run print block to render.
    fake_resolved = MagicMock(device="npu", ep="QNNExecutionProvider")
    fake_result = MagicMock(success=True, output_path=None, compile_time=1.0, total_time=1.0)

    with (
        patch("winml.modelkit.commands.compile.resolve_device", return_value=fake_resolved),
        patch("winml.modelkit.compiler.WinMLCompileConfig.for_ep_device") as _cfg_factory,
        patch("winml.modelkit.compiler.compile_onnx", return_value=fake_result),
        patch("winml.modelkit.commands.compile.is_compiled_onnx", return_value=False),
    ):
        cfg = MagicMock()
        cfg.ep_config.enable_ep_context = False
        _cfg_factory.return_value = cfg

        # A file that exists — the model input must be a real path.
        with runner.isolated_filesystem():
            from pathlib import Path

            Path("m.onnx").write_bytes(b"\x08\x01")
            result = runner.invoke(
                compile_cmd, ["-m", "m.onnx", "--ep", "qnn@pypi", "--device", "npu"]
            )

    out = result.output or ""
    assert "EP: (" not in out, (
        "compile command leaked the raw (ep, source) tuple into user-visible "
        f"output (T-01 regression). Full output:\n{out}"
    )
