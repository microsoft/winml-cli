# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI routing and path ownership for optional GGUF native prefill evidence."""

from pathlib import Path
from random import Random
from typing import Any

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import perf


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "model.onnx"],
        ["-m", "model.gguf", "--runtime", "ort-genai"],
        ["-m", "model.gguf", "--runtime", "dxcgc"],
    ],
)
def test_prefill_trace_is_rejected_on_other_routes(arguments: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(perf, [*arguments, "--gguf-prefill-trace", str(tmp_path / "trace")])
    assert result.exit_code == 2
    assert "requires a GGUF model on winml-runtime" in result.output
    assert not (tmp_path / "trace").exists()


def test_native_trace_option_reaches_gguf_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(Random(71).randbytes(32))
    trace = tmp_path / "native.jsonl"
    seen: list[Any] = []
    monkeypatch.setattr(
        "winml.modelkit.commands._perf_gguf.run_gguf_perf",
        lambda config, **kwargs: seen.append(config),
    )
    result = CliRunner().invoke(
        perf,
        [
            "-m",
            str(model),
            "--gguf-prefill-trace",
            str(trace),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen[0].prefill_trace_path == trace


@pytest.mark.parametrize("collision", ["model", "output", "context", "prompt"])
def test_native_trace_cannot_overwrite_model_output_or_prompt(
    tmp_path: Path, collision: str
) -> None:
    model = tmp_path / "model.gguf"
    original = Random(23).randbytes(32)
    model.write_bytes(original)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Generated test prompt", encoding="utf-8")
    output = tmp_path / "result.json"
    trace = tmp_path / "trace.jsonl"
    if collision == "model":
        trace = model
    elif collision == "output":
        trace = output
    elif collision == "context":
        output = trace.with_name(trace.name + ".context.json")
    else:
        trace = prompt
    result = CliRunner().invoke(
        perf,
        [
            "-m",
            str(model),
            "--prompt-file",
            str(prompt),
            "--gguf-prefill-trace",
            str(trace),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert "must differ" in result.output
    assert model.read_bytes() == original
    assert prompt.read_text(encoding="utf-8") == "Generated test prompt"
