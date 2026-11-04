# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for routing ``winml perf --backend cgc`` to CGC text generation.

A recorder replaces ``run_cgc_perf``, so no Windows ML code runs; the benchmark
itself is tested in ``test_perf_cgc.py``. Every test that expects the CGC route
fails if the command reaches llama.cpp or resolves another runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.commands import _perf_cgc as perf_cgc
from winml.modelkit.commands import _perf_gguf as perf_gguf
from winml.modelkit.commands.perf import perf
from winml.modelkit.utils import cli as cli_utils


if TYPE_CHECKING:
    from click.testing import Result
    from rich.console import Console

    from winml.modelkit.commands._perf_cgc import CGCPerfConfig


perf_module = importlib.import_module("winml.modelkit.commands.perf")
cgc_session = importlib.import_module("winml.modelkit.session.cgc_session")

_CHECKPOINT = "Qwen2.5-0.5B-Instruct"
_LUID = "0x00000000_0x0000ABCD"


class _RoutedPast(Exception):  # noqa: N818 - a routing sentinel, not an error
    """Raised where ``perf()`` resolves ``--runtime``: the command left the CGC route."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _gguf(tmp_path: Path, name: str = "model.gguf") -> Path:
    model = tmp_path / name
    model.write_bytes(b"GGUF")
    return model


def _checkpoint(
    tmp_path: Path,
    name: str = _CHECKPOINT,
    *,
    config: bool = True,
    weights: tuple[str, ...] = ("model.safetensors",),
    index: bool = False,
) -> Path:
    """Write a placeholder Hugging Face checkpoint folder."""
    folder = tmp_path / name
    folder.mkdir()
    if config:
        (folder / "config.json").write_text("{}", encoding="utf-8")
    for weight in weights:
        (folder / weight).write_bytes(b"\0")
    if index:
        (folder / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    for tokenizer_file in ("tokenizer.json", "tokenizer_config.json"):
        (folder / tokenizer_file).write_text("{}", encoding="utf-8")
    return folder


def _cache_snapshot(tmp_path: Path) -> Path:
    """Write a Hugging Face cache snapshot: each file is a relative symlink into ``blobs/``.

    Creating a symlink on Windows needs Developer Mode or an elevated shell.
    """
    repo = tmp_path / "hub" / "models--org--checkpoint"
    snapshot = repo / "snapshots" / "rev"
    (repo / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        content = name.encode()
        blob = hashlib.sha256(content).hexdigest()
        (repo / "blobs" / blob).write_bytes(content)
        (snapshot / name).symlink_to(Path("..", "..", "blobs", blob))
    return snapshot


def _ids_file(tmp_path: Path, ids: list[int]) -> Path:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(ids), encoding="utf-8")
    return path


def _expand(args: list[str], inputs: dict[str, str]) -> list[str]:
    return [inputs.get(arg, arg) for arg in args]


def _text(result: Result) -> str:
    """The command output with line wrapping undone."""
    return " ".join(result.output.split())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, str]:
    """Every placeholder the parametrized command lines name, mapped to a path."""
    (tmp_path / "model.mlir").write_text("module {}", encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "inputs.npz").write_bytes(b"")
    return {
        "{gguf}": str(_gguf(tmp_path)),
        "{folder}": str(_checkpoint(tmp_path)),
        "{index_only_folder}": str(_checkpoint(tmp_path, "sharded", weights=(), index=True)),
        "{mlir}": str(tmp_path / "model.mlir"),
        "{onnx}": str(tmp_path / "model.onnx"),
        "{npz}": str(tmp_path / "inputs.npz"),
        "{ev}": str(tmp_path / "evidence.json"),
        "{etw}": str(tmp_path / "etw"),
    }


@pytest.fixture
def capture_cgc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace run_cgc_perf; record the config, console and json_mode it receives."""
    captured: dict[str, Any] = {}

    def record(config: CGCPerfConfig, *, console: Console, json_mode: bool) -> None:
        captured.update(config=config, console=console, json_mode=json_mode)

    monkeypatch.setattr(perf_cgc, "run_cgc_perf", record)
    return captured


@pytest.fixture
def no_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("--backend cgc fell back to llama.cpp")

    monkeypatch.setattr(perf_gguf, "run_gguf_perf", fail)


@pytest.fixture
def routed_past(monkeypatch: pytest.MonkeyPatch) -> None:
    def leave(runtime: str, model: str) -> str:
        raise _RoutedPast(runtime, model)

    monkeypatch.setattr(perf_module, "_resolve_runtime", leave)


# ---------------------------------------------------------------------------
# Routing (every command of the routing table)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize(
    ("args", "fmt"),
    [
        (["{gguf}", "--backend", "cgc"], "gguf"),
        (["{gguf}", "--runtime", "winml-runtime", "--backend", "cgc"], "gguf"),
        (["{folder}", "--backend", "cgc"], "safetensors"),
        pytest.param(
            ["{index_only_folder}", "--backend", "cgc"], "safetensors", id="index_only_folder"
        ),
    ],
)
def test_backend_cgc_selects_cgc_text_generation(
    runner: CliRunner,
    inputs: dict[str, str],
    capture_cgc: dict[str, Any],
    args: list[str],
    fmt: str,
) -> None:
    result = runner.invoke(perf, ["-m", *_expand(args, inputs)])

    assert result.exit_code == 0, result.output
    config = capture_cgc["config"]
    assert config.model_format == fmt
    assert config.model_path == Path(inputs[args[0]])


@pytest.mark.parametrize(
    "args",
    [
        ["{gguf}"],
        ["{folder}"],
        ["{folder}", "--runtime", "winml-runtime", "--backend", "cgc"],
        ["{folder}", "--runtime", "winml-runtime"],
        ["{mlir}", "--backend", "cgc"],
        ["{onnx}", "--backend", "cgc"],
        ["{gguf}", "--runtime", "winml-ort", "--backend", "cgc"],
    ],
)
def test_other_commands_keep_their_route(
    runner: CliRunner,
    inputs: dict[str, str],
    capture_cgc: dict[str, Any],
    routed_past: None,
    args: list[str],
) -> None:
    result = runner.invoke(perf, ["-m", *_expand(args, inputs)])

    assert isinstance(result.exception, _RoutedPast), result.output
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama")
def test_gguf_with_backend_ort_names_both_supported_choices(
    runner: CliRunner, inputs: dict[str, str], capture_cgc: dict[str, Any]
) -> None:
    result = runner.invoke(perf, ["-m", inputs["{gguf}"], "--backend", "ort"])

    assert result.exit_code == 2, result.output
    assert "--backend cannot be used with a GGUF model except --backend cgc" in _text(result)
    assert "llama.cpp" in result.output
    assert "config" not in capture_cgc


# ---------------------------------------------------------------------------
# Device and input rules
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize("kind", ["gguf", "folder"])
@pytest.mark.parametrize("device", ["cpu", "npu", "config"])
def test_cgc_runs_only_on_a_gpu(
    runner: CliRunner,
    inputs: dict[str, str],
    capture_cgc: dict[str, Any],
    device: str,
    kind: str,
) -> None:
    result = runner.invoke(
        perf, ["-m", inputs[f"{{{kind}}}"], "--backend", "cgc", "--device", device]
    )

    assert result.exit_code == 2, result.output
    assert f"--device {device} is not supported with --backend cgc" in _text(result)
    # Only a .gguf file has a CPU engine to point to.
    assert ("omit --backend cgc" in _text(result)) is (kind == "gguf")
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_single_safetensors_file_asks_for_its_folder(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    folder = _checkpoint(tmp_path)
    result = runner.invoke(perf, ["-m", str(folder / "model.safetensors"), "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert "not a single .safetensors file" in _text(result)
    assert f'-m "{folder}"' in result.output
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_hugging_face_id_is_not_accepted_yet(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_cgc: dict[str, Any],
) -> None:
    monkeypatch.chdir(tmp_path)  # no local folder may shadow the ID
    result = runner.invoke(perf, ["-m", f"Qwen/{_CHECKPOINT}", "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert "Hugging Face model IDs are not supported yet" in _text(result)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize("missing", ["config", "weights"])
def test_folder_precheck_names_the_required_files(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any], missing: str
) -> None:
    folder = _checkpoint(
        tmp_path,
        config=missing != "config",
        weights=() if missing == "weights" else ("model.safetensors",),
    )
    result = runner.invoke(perf, ["-m", str(folder), "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert "config.json" in result.output
    assert "*.safetensors" in result.output
    assert "pass the file itself" not in _text(result)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_folder_holding_only_a_gguf_points_to_the_file(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    folder = tmp_path / "downloads"
    folder.mkdir()
    _gguf(folder)
    result = runner.invoke(perf, ["-m", str(folder), "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert "pass the file itself" in _text(result)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize(
    ("name", "message"),
    [("missing.gguf", "GGUF file not found"), ("missing", "does not exist")],
)
def test_missing_inputs_are_usage_errors(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any], name: str, message: str
) -> None:
    result = runner.invoke(perf, ["-m", str(tmp_path / name), "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert message in _text(result)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_other_files_are_neither_input(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"\0")
    result = runner.invoke(perf, ["-m", str(model), "--backend", "cgc"])

    assert result.exit_code == 2, result.output
    assert f"'{model}' is neither" in result.output
    assert "config" not in capture_cgc


# ---------------------------------------------------------------------------
# Option rules
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize(
    ("args", "flag"),
    [
        (["--ep", "cpu"], "--ep"),
        (["--ep-options", "k=v"], "--ep-options"),
        (["--input-data", "{npz}"], "--input-data"),
        (["--module", "X"], "--module"),
        (["--submodel", "decoder"], "--submodel"),
        (["--evidence", "{ev}"], "--evidence"),
        (["--etw-tracing", "{etw}"], "--etw-tracing"),
        (["--allow-cpu-fallback"], "--allow-cpu-fallback"),
        (["--cgc-lowering-target", "auto"], "--cgc-lowering-target"),
        (["--memory-backend", "dxgi"], "--memory-backend dxgi"),
        (["--warmup-seconds", "1"], "--warmup-seconds"),
    ],
)
def test_options_cgc_cannot_honor_are_rejected(
    runner: CliRunner,
    inputs: dict[str, str],
    capture_cgc: dict[str, Any],
    args: list[str],
    flag: str,
) -> None:
    result = runner.invoke(
        perf, ["-m", inputs["{gguf}"], "--backend", "cgc", *_expand(args, inputs)]
    )

    assert result.exit_code == 2, result.output
    assert f"{flag} cannot be used with --backend cgc text generation" in _text(result)
    # Only --evidence has a llama.cpp run to point to.
    assert ("omit --backend cgc" in _text(result)) is (flag == "--evidence")
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama")
def test_evidence_with_a_checkpoint_folder_points_to_no_other_route(
    runner: CliRunner, inputs: dict[str, str], capture_cgc: dict[str, Any]
) -> None:
    # --evidence records the llama.cpp native log, which a checkpoint folder never
    # runs on: neither rejection may send the user to the other one.
    evidence = ["--evidence", inputs["{ev}"]]
    with_cgc = runner.invoke(perf, ["-m", inputs["{folder}"], "--backend", "cgc", *evidence])
    without_cgc = runner.invoke(perf, ["-m", inputs["{folder}"], *evidence])

    assert with_cgc.exit_code == 2, with_cgc.output
    assert "--evidence cannot be used with --backend cgc text generation" in _text(with_cgc)
    assert "--evidence applies only to a .gguf file on llama.cpp" in _text(with_cgc)
    assert "omit --backend cgc" not in _text(with_cgc)
    assert without_cgc.exit_code == 2, without_cgc.output
    assert "--evidence applies only to a .gguf file on llama.cpp" in _text(without_cgc)
    assert "--backend cgc" not in _text(without_cgc)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_explicit_pdh_memory_backend_is_accepted(
    runner: CliRunner, inputs: dict[str, str], capture_cgc: dict[str, Any]
) -> None:
    result = runner.invoke(
        perf, ["-m", inputs["{gguf}"], "--backend", "cgc", "--memory-backend", "pdh"]
    )

    assert result.exit_code == 0, result.output
    assert capture_cgc["config"].memory is True


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_build_options_are_ignored_without_loading_the_build_config(
    runner: CliRunner,
    inputs: dict[str, str],
    tmp_path: Path,
    capture_cgc: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_config_load(_path: Path) -> None:
        pytest.fail("--backend cgc must not load the ignored ONNX build configuration")

    monkeypatch.setattr(cli_utils, "load_build_config", unexpected_config_load)
    config_path = tmp_path / "onnx-build.json"
    config_path.write_text(
        json.dumps({"compile": {"device": "npu", "execution_provider": "qnn"}}), encoding="utf-8"
    )
    result = runner.invoke(
        perf,
        [
            "-m",
            inputs["{folder}"],
            "--backend",
            "cgc",
            "--precision",
            "w8a16",
            "--batch-size",
            "2",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    text = _text(result)
    assert "ignored with --backend cgc text generation" in text
    assert [flag for flag in ("--precision", "--batch-size", "--config") if flag not in text] == []
    assert capture_cgc["config"].device == "gpu"


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_defaults_follow_the_gguf_protocol(
    runner: CliRunner, inputs: dict[str, str], capture_cgc: dict[str, Any]
) -> None:
    result = runner.invoke(perf, ["-m", inputs["{folder}"], "--backend", "cgc"])

    assert result.exit_code == 0, result.output
    config = capture_cgc["config"]
    assert (config.iterations, config.warmup, config.max_new_tokens) == (10, 2, 128)
    assert (config.device, config.requested_device, config.memory) == ("gpu", "auto", True)
    assert (config.device_luid, config.program_cache_dir, config.prompt_ids) == (None, None, None)
    assert (config.apply_template, config.context_length, config.ignore_eos) == (True, None, False)
    assert config.output_path.parent == Path.home() / ".cache" / "winml" / "perf" / _CHECKPOINT
    assert capture_cgc["json_mode"] is False


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_protocol_options_are_forwarded(
    runner: CliRunner, inputs: dict[str, str], tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    cache = tmp_path / "programs"
    output = tmp_path / "report.json"
    result = runner.invoke(
        perf,
        [
            "-m",
            inputs["{folder}"],
            "--backend",
            "cgc",
            "--prompt-ids-file",
            str(_ids_file(tmp_path, [1, 2, 3])),
            "--context-length",
            "256",
            "--ignore-eos",
            "--device-luid",
            _LUID,
            "--program-cache",
            str(cache),
            "--device",
            "gpu",
            "--max-new-tokens",
            "16",
            "--iterations",
            "3",
            "--warmup",
            "0",
            "--no-memory",
            "-o",
            str(output),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    config = capture_cgc["config"]
    assert (config.prompt_ids, config.apply_template) == ((1, 2, 3), False)
    assert (config.context_length, config.ignore_eos) == (256, True)
    assert (config.device_luid, config.program_cache_dir) == (_LUID, cache.resolve())
    assert (config.device, config.requested_device) == ("gpu", "gpu")
    assert (config.iterations, config.warmup, config.max_new_tokens) == (3, 0, 16)
    assert (config.memory, config.output_path) == (False, output)
    assert capture_cgc["json_mode"] is True


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_program_cache_is_recorded_as_an_absolute_path(
    runner: CliRunner,
    inputs: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_cgc: dict[str, Any],
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        perf, ["-m", inputs["{gguf}"], "--backend", "cgc", "--program-cache", "programs"]
    )

    assert result.exit_code == 0, result.output
    assert capture_cgc["config"].program_cache_dir == (tmp_path / "programs").resolve()


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_json_format_keeps_stdout_for_the_report(
    runner: CliRunner, inputs: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(config: CGCPerfConfig, *, console: Console, json_mode: bool) -> None:
        # What the real entry point writes in JSON mode.
        click.echo(json.dumps({"backend": "cgc"}))
        console.print(f"[green]Results saved to:[/green] {config.output_path}")

    monkeypatch.setattr(perf_cgc, "run_cgc_perf", run)
    result = runner.invoke(
        perf,
        [
            "-m",
            inputs["{gguf}"],
            "--backend",
            "cgc",
            "--precision",
            "w8a16",
            "-o",
            str(tmp_path / "report.json"),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"backend": "cgc"}
    stderr = " ".join(result.stderr.split())
    assert "ignored with --backend cgc text generation" in stderr
    assert "Results saved to:" in stderr


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_session_warnings_show_at_the_default_verbosity(
    runner: CliRunner, inputs: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session warns when Windows ML reports no capacity or vocabulary size."""
    message = "Windows ML did not report the tokenizer vocabulary size"

    def run(config: CGCPerfConfig, *, console: Console, json_mode: bool) -> None:
        cgc_session.logger.warning(message)

    monkeypatch.setattr(perf_cgc, "run_cgc_perf", run)
    result = runner.invoke(perf, ["-m", inputs["{folder}"], "--backend", "cgc", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert f"WARNING winml.modelkit.session.cgc_session] {message}" in " ".join(
        result.stderr.split()
    )
    assert result.stdout == ""


@pytest.mark.parametrize("model", ["{gguf}", "{onnx}"])
def test_program_cache_needs_backend_cgc(
    runner: CliRunner,
    inputs: dict[str, str],
    tmp_path: Path,
    capture_cgc: dict[str, Any],
    no_llama: None,
    model: str,
) -> None:
    result = runner.invoke(
        perf, ["-m", inputs[model], "--program-cache", str(tmp_path / "programs")]
    )

    assert result.exit_code == 2, result.output
    assert "--program-cache only applies to --backend cgc" in _text(result)
    assert "config" not in capture_cgc


# ---------------------------------------------------------------------------
# Output paths and input protection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize("layout", ["files", "cache_snapshot"])
@pytest.mark.parametrize("target", ["report.json", "config.json"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_output_inside_the_checkpoint_folder_is_refused(
    runner: CliRunner,
    tmp_path: Path,
    capture_cgc: dict[str, Any],
    overwrite: bool,
    target: str,
    layout: str,
) -> None:
    # In a cache snapshot, config.json is a symlink to a blob outside the folder.
    folder = _checkpoint(tmp_path) if layout == "files" else _cache_snapshot(tmp_path)
    before = {path: path.read_bytes() for path in folder.iterdir()}
    args = ["-m", str(folder), "--backend", "cgc", "--output", str(folder / target)]
    result = runner.invoke(perf, [*args, "--overwrite"] if overwrite else args)

    assert result.exit_code == 2, result.output
    assert "inside the --model checkpoint folder" in _text(result)
    assert "config" not in capture_cgc
    assert {path: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_output_in_a_linked_subfolder_of_the_checkpoint_is_refused(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    folder = _checkpoint(tmp_path)
    # The subfolder is a directory symlink that leads out of the checkpoint folder.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (folder / "extra").symlink_to(elsewhere, target_is_directory=True)
    output = folder / "extra" / "report.json"
    result = runner.invoke(perf, ["-m", str(folder), "--backend", "cgc", "--output", str(output)])

    assert result.exit_code == 2, result.output
    assert "inside the --model checkpoint folder" in _text(result)
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize("link", ["symlink", "hard_link", "cache_blob"])
def test_output_linked_to_a_checkpoint_file_is_refused(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any], link: str
) -> None:
    # Each --output sits outside the folder but names its config.json, so a
    # report written there would replace the model's config.json.
    if link == "cache_blob":
        folder = _cache_snapshot(tmp_path)
        output = (folder / "config.json").resolve()
    else:
        folder = _checkpoint(tmp_path)
        output = tmp_path / "report.json"
        if link == "symlink":
            output.symlink_to(folder / "config.json")
        else:
            output.hardlink_to(folder / "config.json")
    before = {path: path.read_bytes() for path in folder.iterdir()}
    result = runner.invoke(
        perf, ["-m", str(folder), "--backend", "cgc", "--output", str(output), "--overwrite"]
    )

    assert result.exit_code == 2, result.output
    assert "--model checkpoint folder" in _text(result)
    # A path that resolves into the folder is inside it; a hard link or the
    # blob behind a symlink is the same file as the folder's config.json.
    named = "is inside the" if link == "symlink" else f"same file as '{folder / 'config.json'}'"
    assert named in result.output
    assert "config" not in capture_cgc
    assert {path: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_existing_report_outside_the_checkpoint_folder_can_be_overwritten(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any]
) -> None:
    folder = _cache_snapshot(tmp_path)
    # A blob removed from the cache leaves a dangling symlink, which names no file.
    (folder / "generation_config.json").symlink_to(Path("..", "..", "blobs", "removed"))
    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        perf, ["-m", str(folder), "--backend", "cgc", "--output", str(output), "--overwrite"]
    )

    assert result.exit_code == 0, result.output
    assert Path(capture_cgc["config"].output_path) == output


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize("cache", ["programs", ".", "link"])
def test_program_cache_inside_the_checkpoint_folder_is_refused(
    runner: CliRunner,
    inputs: dict[str, str],
    tmp_path: Path,
    capture_cgc: dict[str, Any],
    cache: str,
) -> None:
    folder = Path(inputs["{folder}"])
    program_cache = folder / cache
    if cache == "link":
        # A directory symlink outside the folder: the programs would land in the folder.
        program_cache = tmp_path / "programs"
        program_cache.symlink_to(folder, target_is_directory=True)
    result = runner.invoke(
        perf,
        [
            "-m",
            str(folder),
            "--backend",
            "cgc",
            "--program-cache",
            str(program_cache),
            "-o",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 2, result.output
    text = _text(result)
    assert "--program-cache" in text
    assert "inside the --model checkpoint folder" in text
    assert "config" not in capture_cgc


@pytest.mark.usefixtures("no_llama", "routed_past")
@pytest.mark.parametrize(
    ("kind", "target"),
    [("gguf", "model"), ("gguf", "shard"), ("gguf", "ids"), ("folder", "ids")],
)
def test_output_cannot_overwrite_inputs(
    runner: CliRunner, tmp_path: Path, capture_cgc: dict[str, Any], kind: str, target: str
) -> None:
    # The IDs file sits outside the checkpoint folder.
    ids = _ids_file(tmp_path, [1, 2])
    if kind == "gguf":
        model = _gguf(tmp_path, "m-00001-of-00002.gguf")
        protected = {"model": model, "shard": _gguf(tmp_path, "m-00002-of-00002.gguf"), "ids": ids}
    else:
        model = _checkpoint(tmp_path)
        protected = {"ids": ids}
    before = {path: path.read_bytes() for path in protected.values()}
    result = runner.invoke(
        perf,
        [
            "-m",
            str(model),
            "--backend",
            "cgc",
            "--prompt-ids-file",
            str(ids),
            "--output",
            str(protected[target]),
            "--overwrite",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "would overwrite the" in _text(result)
    assert "config" not in capture_cgc
    assert {path: path.read_bytes() for path in protected.values()} == before


# ---------------------------------------------------------------------------
# Stable windowsml (end to end through the real run_cgc_perf)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_llama", "routed_past")
def test_stable_windowsml_exits_with_the_preview_requirement(
    runner: CliRunner, inputs: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pinned stable windowsml has no runtime subpackage.
    monkeypatch.setitem(sys.modules, "windowsml.runtime", None)

    def select_gpu(device_luid: str | None) -> Any:
        pytest.fail("the GPU was selected before the windowsml check")

    monkeypatch.setattr(cgc_session, "open_gpu_adapter", select_gpu)
    output = tmp_path / "report.json"
    result = runner.invoke(perf, ["-m", inputs["{gguf}"], "--backend", "cgc", "-o", str(output)])

    assert result.exit_code == 1, result.output
    assert "2.8.183.dev0" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


class TestHelp:
    BACKEND_HELP: ClassVar[list[str]] = [
        "Without --backend, a .gguf file runs on llama.cpp",
        "With --runtime auto (the default), cgc runs a .gguf file or a local Hugging Face "
        "checkpoint folder as text generation on the Windows ML CGC backend on a GPU",
        "with --runtime winml-runtime a checkpoint folder is still exported to ONNX",
    ]
    OTHER_HELP: ClassVar[list[str]] = [
        # --runtime
        "With --backend cgc, 'auto' runs a .gguf file or a local checkpoint folder as CGC text "
        "generation on a GPU",
        # --device
        "'auto' means cpu for a .gguf file on llama.cpp and gpu with --backend cgc",
        # --device-luid
        "With --backend cgc, pins the GPU used for text generation",
    ]

    @pytest.fixture
    def help_text(self) -> str:
        result = CliRunner().invoke(perf, ["--help"])
        assert result.exit_code == 0, result.output
        # Undo Click's line wrapping, which can also break after the hyphen in winml-runtime
        # depending on the terminal width.
        return re.sub(r"(?<=\w)- (?=\w)", "-", " ".join(result.output.split()))

    def test_help_states_the_gguf_default_and_backend_cgc(self, help_text: str) -> None:
        missing = [
            phrase for phrase in [*self.BACKEND_HELP, *self.OTHER_HELP] if phrase not in help_text
        ]
        assert missing == []
        # Both stay hidden.
        assert "--cgc-lowering-target" not in help_text
        assert "--allow-cpu-fallback" not in help_text

    def test_program_cache_help_promises_to_keep_the_directory(
        self, help_text: str, tmp_path: Path
    ) -> None:
        assert "--program-cache DIR" in help_text
        assert "never deletes this directory" in help_text
        with (
            click.Context(main) as root,
            perf.make_context("perf", ["--program-cache", str(tmp_path)], parent=root) as ctx,
        ):
            assert ctx.params["program_cache"] == tmp_path
