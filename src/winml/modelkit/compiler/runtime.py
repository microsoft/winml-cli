# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Device-targeted compilation of existing CGC Input IR through Runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import click
from filelock import FileLock

from ..export.cgc.artifacts import cgc_metadata_path
from ..export.cgc.foundry import (
    FoundryCompileError,
    FoundryCompiler,
    FoundryToolboxUnavailableError,
)
from ..session._runtime_import import import_runtime
from ..session.runtime_session import _DXCoreAdapter, _translate_native_errors
from ..sysinfo.dxcore_adapters import enumerate_compute_adapters
from ..utils.cli import guard_output


if TYPE_CHECKING:
    from ..sysinfo.dxcore_adapters import DXCoreAdapterInfo

logger = logging.getLogger(__name__)


def _file_identity(path: Path) -> dict[str, str | int]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest}


def _validate_input_ir(path: Path) -> None:
    try:
        with FoundryCompiler() as compiler:
            text = compiler.read_cgir(path.read_bytes()).decode("utf-8")
    except (FoundryCompileError, FoundryToolboxUnavailableError, UnicodeDecodeError) as e:
        raise click.ClickException(f"Cannot parse CGC Input IR '{path}': {e}") from e
    if re.search(r"#cgc\.interface<cgc_output>", text) or not re.search(
        r'(?m)^\s*(?:"cgc\.entry_point"|cgc\.entry_point\b)', text
    ):
        raise click.ClickException(
            "--input-format cgc-input-ir requires a CGC Input IR entry point, "
            "not ONNX or already compiled Output IR."
        )


def _select_gpu(device_luid: str | None) -> DXCoreAdapterInfo:
    adapters = [adapter for adapter in enumerate_compute_adapters() if adapter.device_type == "GPU"]
    if device_luid is not None:
        adapters = [
            adapter for adapter in adapters if adapter.luid.casefold() == device_luid.casefold()
        ]
    if len(adapters) != 1:
        raise click.ClickException(
            "CGC compilation requires one unambiguous GPU. Use --device-luid with "
            f"a current GPU LUID from 'winml sys'; found {len(adapters)} matching adapters."
        )
    return adapters[0]


def _same_file(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def _publish(staging: Path, output: Path, overwrite: bool, protected: tuple[Path, ...]) -> None:
    files = [path for path in staging.iterdir() if path.is_file()]
    if any(path.is_dir() for path in staging.iterdir()):
        raise click.ClickException("Cannot publish a nested Runtime artifact layout.")
    for path in files:
        destination = output.parent / path.name
        if any(_same_file(destination, source) for source in protected):
            raise click.ClickException(f"Output would overwrite a source artifact: {destination}")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise click.ClickException(f"Output artifact is not a regular file: {destination}")
        guard_output(destination, overwrite)
    backup = staging / "backup"
    backup.mkdir()
    restored: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, path in enumerate(files):
            destination = output.parent / path.name
            if destination.exists():
                saved = backup / str(index)
                destination.replace(saved)
                restored.append((saved, destination))
            path.replace(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink()
        for saved, destination in reversed(restored):
            saved.replace(destination)
        raise


def compile_cgc_ir(
    model_path: Path,
    output_path: Path,
    *,
    device_luid: str | None = None,
    validate: bool = True,
    overwrite: bool = False,
) -> Path:
    """Compile genuine CGC Input IR to a persistent device-targeted bytecode bundle.

    Validation reloads the artifact, builds a pipeline and compares I/O schemas;
    it does not run inference, numerical accuracy checks or performance tests.
    """
    started = time.perf_counter()
    if not model_path.is_file():
        raise click.ClickException(f"Input IR file not found: {model_path}")
    if _same_file(model_path, output_path):
        raise click.ClickException(
            "Input and output must be different files, even with --overwrite."
        )
    if output_path.suffix.lower() not in {".mlir", ".mlirbc"}:
        raise click.UsageError("CGC Output IR output must use .mlir or .mlirbc.")
    metadata_path = cgc_metadata_path(model_path)
    metadata_identity = _file_identity(metadata_path) if metadata_path.is_file() else None
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as e:
            raise click.ClickException(f"Invalid CGC metadata '{metadata_path}': {e}") from e
        if not isinstance(metadata, dict):
            raise click.ClickException(f"CGC metadata must be an object: {metadata_path}")
    guard_output(output_path, overwrite)
    source_identity = _file_identity(model_path)
    _validate_input_ir(model_path)
    wr = import_runtime()
    adapter_info = _select_gpu(device_luid)
    high, low = adapter_info.luid.split("_")
    luid = (int(high, 16) << 32) | int(low, 16)
    logger.info("Compiling CGC Input IR on %s / %s", adapter_info.name, adapter_info.luid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(output_path) + ".compile.lock"),
        TemporaryDirectory(
            prefix=f".{output_path.stem}-compile-", dir=output_path.parent
        ) as temporary,
    ):
        staging = Path(temporary)
        artifact = staging / output_path.name
        with ExitStack() as stack:
            runtime = wr.Runtime()
            stack.callback(runtime.close)
            adapter = _DXCoreAdapter.from_luid(luid)
            stack.callback(adapter.close)
            with _translate_native_errors("build", device_class="gpu"):
                target = runtime.create_target_from_adapter(adapter.pointer)
                stack.callback(target.close)
                if target.kind != wr.ExecutionTargetKind.GPU:
                    raise click.ClickException("Selected adapter is not a GPU Runtime target.")
                compiler = target.model_compiler()
                stack.callback(compiler.close)
                if not compiler.supports_form(wr.CompiledModelForm.DEVICE_TARGETED):
                    raise click.ClickException(
                        "Runtime does not support DEVICE_TARGETED Output IR."
                    )
                options = compiler.create_options()
                stack.callback(options.close)
                options.form = wr.CompiledModelForm.DEVICE_TARGETED
            with _translate_native_errors("load"):
                model = runtime.load_model(str(model_path))
                stack.callback(model.close)
                source_schema = model.schema()
                stack.callback(source_schema.close)
            if metadata is not None:
                for field, count in (
                    ("inputs", source_schema.input_count),
                    ("outputs", source_schema.output_count),
                ):
                    entries = metadata.get(field)
                    if (
                        not isinstance(entries, list)
                        or len(entries) != count
                        or any(
                            not isinstance(entry, dict)
                            or not isinstance(entry.get("name"), str)
                            or not entry["name"]
                            or type(entry.get("index")) is not int
                            or entry["index"] != index
                            for index, entry in enumerate(entries)
                        )
                        or len({entry["name"] for entry in entries}) != count
                    ):
                        raise click.ClickException(
                            f"CGC metadata {field} do not match Input IR: {metadata_path}"
                        )
            with _translate_native_errors("build", device_class="gpu"):
                compiler.compile_to_file(model, str(artifact), options=options)
            if not artifact.is_file() or not artifact.stat().st_size:
                raise click.ClickException("Runtime returned no nonempty Output IR artifact.")
            with artifact.open("rb") as stream:
                if stream.read(4) != b"ML\xefR":
                    raise click.ClickException("Runtime did not produce MLIR bytecode Output IR.")
            if metadata is not None:
                cgc_metadata_path(artifact).write_text(
                    json.dumps({**metadata, "format": "cgc-output-ir"}, indent=2),
                    encoding="utf-8",
                )
            if validate:
                with _translate_native_errors("load"):
                    compiled = runtime.load_model(str(artifact))
                    stack.callback(compiled.close)
                with _translate_native_errors("build", device_class="gpu"):
                    builder = runtime.create_pipeline_builder()
                    stack.callback(builder.close)
                    stage = builder.add_model_stage(compiled, target)
                    stack.callback(stage.close)
                    pipeline = builder.build()
                    stack.callback(pipeline.close)
                    compiled_schema = stage.schema()
                    stack.callback(compiled_schema.close)
                    if (
                        source_schema.inputs() != compiled_schema.inputs()
                        or source_schema.outputs() != compiled_schema.outputs()
                    ):
                        raise click.ClickException("Output IR I/O schema differs from Input IR.")
        if _file_identity(model_path) != source_identity:
            raise click.ClickException("Input IR changed during compilation; output not published.")
        current_metadata_identity = (
            _file_identity(metadata_path) if metadata_path.is_file() else None
        )
        if current_metadata_identity != metadata_identity:
            raise click.ClickException("Input IR metadata changed; output not published.")
        receipt = {
            "input_format": "cgc-input-ir",
            "target": "cgc-output-ir",
            "compiler": "winml-runtime",
            "compiled_form": "DEVICE_TARGETED",
            "output_encoding": "mlir-bytecode",
            "input": source_identity,
            "gpu": {"name": adapter_info.name, "luid": adapter_info.luid},
            "validation": {
                "status": "passed" if validate else "skipped",
                "scope": "reload, pipeline build and I/O schema; no inference or accuracy",
            },
            "artifacts": [_file_identity(path) for path in staging.iterdir() if path.is_file()],
            "elapsed_seconds": time.perf_counter() - started,
        }
        (staging / f"{output_path.stem}_compile.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        _publish(staging, output_path, overwrite, (model_path, metadata_path))
    return output_path
