# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX-to-CGC export orchestration."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
from rich.console import Console

from .artifacts import cgc_metadata_path
from .foundry import FoundryCompileError, FoundryCompiler


if TYPE_CHECKING:
    from torch import nn

    from ..config import WinMLExportConfig


@dataclass(frozen=True)
class CGCOptions:
    """Configuration shared by CGC export steps."""

    external_weights: bool = False
    topo_sort_nodes: bool = False
    update_opset: bool = True
    freeze_dims: str = ""


@dataclass(frozen=True)
class CGCExportResult:
    """Metadata produced by a CGC export."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    export_stats: dict[str, Any] | None = None


class CGCExporter:
    """Generate standalone CGC Input IR."""

    options_type = CGCOptions

    def __init__(self, options: CGCOptions) -> None:
        """Initialize the exporter with typed CGC options."""
        self.options = options
        self._freeze_dims = _parse_freeze_dims(options.freeze_dims)
        self.console = Console(width=100, highlight=False)

    def export_pytorch(
        self,
        *,
        model: nn.Module,
        output_path: str | Path,
        export_config: WinMLExportConfig,
        model_id: str,
        task: str | None,
        verbose: bool,
        enable_reporting: bool,
        **onnx_kwargs: Any,
    ) -> CGCExportResult:
        """Export PyTorch to CGC without exposing the intermediate representation."""
        from .. import export_pytorch as export_onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".winml-cgc-source-",
            dir=output_path.parent,
        ) as temporary:
            intermediate_path = Path(temporary) / "model.onnx"
            export_stats = export_onnx(
                model=model,
                output_path=intermediate_path,
                export_config=export_config,
                model_id=model_id,
                task=task,
                verbose=verbose,
                enable_reporting=enable_reporting,
                **onnx_kwargs,
            )
            result = self.export_onnx(
                model=intermediate_path,
                output_path=output_path,
            )
            if enable_reporting:
                self._publish_onnx_reports(intermediate_path, output_path)
        return replace(result, export_stats=export_stats)

    def export_onnx(
        self,
        model: str | Path,
        output_path: str | Path,
    ) -> CGCExportResult:
        """Export an ONNX model to standalone CGC MLIR."""
        source_path = Path(model)
        if not source_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {source_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.console.print("\n" + "=" * 80)
        self.console.print("🚀 [bold cyan]ONNX TO CGC EXPORT PROCESS[/bold cyan]")
        self.console.print("=" * 80)
        self.console.print(f"[bold blue]Input:[/bold blue] {source_path}")
        self.console.print(f"[bold blue]Output:[/bold blue] {output_path}")
        self.console.print("[bold blue]Format:[/bold blue] CGC MLIR")
        export_start = time.monotonic()

        model_proto = onnx.load(str(source_path), load_external_data=False)
        initializer_names = {
            initializer.name for initializer in model_proto.graph.initializer
        }
        initializer_names.update(
            initializer.values.name
            for initializer in model_proto.graph.sparse_initializer
        )
        self._freeze_dims = _parse_freeze_dims(self.options.freeze_dims)
        self._freeze_batch_size(model_proto, initializer_names)
        result = CGCExportResult(
            input_names=tuple(
                value.name
                for value in model_proto.graph.input
                if value.name not in initializer_names
            ),
            output_names=tuple(value.name for value in model_proto.graph.output),
        )

        self._export_mlir(
            source_path,
            output_path,
            result,
        )

        self.console.print("\n[bold green]✅ CGC EXPORT COMPLETE[/bold green]")
        self.console.print(f"[dim]Total time: {time.monotonic() - export_start:.2f}s[/dim]")
        return result

    def _freeze_batch_size(
        self, model: onnx.ModelProto, initializer_names: set[str],
    ) -> None:
        """Supplement explicit dimension overrides with a detected batch_size default."""
        if "batch_size" not in self._freeze_dims and any(
            dimension.dim_param == "batch_size"
            for value in model.graph.input
            if value.name not in initializer_names and value.type.HasField("tensor_type")
            for dimension in value.type.tensor_type.shape.dim
        ):
            self._freeze_dims["batch_size"] = 1
            self.console.print("[dim]Freeze input dimension: batch_size=1[/dim]")

    def export(
        self,
        model: str | Path,
        output_path: str | Path,
    ) -> None:
        """Export ONNX to CGC; retained as a compatibility alias."""
        self.export_onnx(model=model, output_path=output_path)

    def output_artifacts(self, output_path: str | Path) -> tuple[Path, ...]:
        """Return primary and potential sidecar artifacts for output guarding."""
        output_path = Path(output_path)
        artifacts = [output_path]
        if self.options.external_weights:
            artifacts.append(output_path.with_name(f"{output_path.name}.data"))
        artifacts.append(cgc_metadata_path(output_path))
        return tuple(artifacts)

    def _write_io_metadata(
        self,
        result: CGCExportResult,
        output_path: Path,
    ) -> None:
        """Persist source ONNX names for ordinal-only Runtime artifacts."""
        cgc_metadata_path(output_path).write_text(
            json.dumps(
                {
                    "format": "cgc-input-ir",
                    "inputs": [
                        {"name": name, "index": index}
                        for index, name in enumerate(result.input_names)
                    ],
                    "outputs": [
                        {"name": name, "index": index}
                        for index, name in enumerate(result.output_names)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _publish_onnx_reports(
        intermediate_path: Path,
        output_path: Path,
    ) -> None:
        """Move requested HTP reports out of the temporary ONNX directory."""
        intermediate_base = intermediate_path.with_suffix("")
        output_base = output_path.with_suffix("")
        for suffix in ("_htp_metadata.json", "_htp_export_report.md"):
            source = intermediate_base.with_name(f"{intermediate_base.name}{suffix}")
            destination = output_base.with_name(f"{output_base.name}{suffix}")
            source.replace(destination)

    def _export_mlir(
        self,
        source_path: Path,
        output_path: Path,
        result: CGCExportResult,
    ) -> None:
        """Generate standalone CGC MLIR with optional Foundry weight data."""
        final_weights = (
            output_path.with_name(f"{output_path.name}.data")
            if self.options.external_weights
            else None
        )
        final_metadata = cgc_metadata_path(output_path)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        ) as temporary:
            staging_dir = Path(temporary)
            staged_output = staging_dir / output_path.name
            staged_weights = (
                staging_dir / final_weights.name if final_weights is not None else None
            )
            staged_metadata = cgc_metadata_path(staged_output)

            ir = self._convert_to_cgir(
                source_path,
                output_data_file=staged_weights,
            )
            staged_output.write_text(ir, encoding="utf-8", newline="\n")
            self._write_io_metadata(result, staged_output)

            artifacts = [(staged_metadata, final_metadata), (staged_output, output_path)]
            if staged_weights is not None and final_weights is not None:
                artifacts.insert(0, (staged_weights, final_weights))
            for _, destination in artifacts:
                if destination.exists() and not destination.is_file():
                    raise ValueError(f"Output artifact exists but is not a file: {destination}")

            backup_dir = staging_dir / "backup"
            backup_dir.mkdir()
            backups: list[tuple[Path, Path]] = []
            published: list[Path] = []
            try:
                for index, (_, destination) in enumerate(artifacts):
                    if destination.exists() or destination.is_symlink():
                        backup = backup_dir / str(index)
                        destination.replace(backup)
                        backups.append((backup, destination))
                for staged, destination in artifacts:
                    if staged.exists():
                        staged.replace(destination)
                        published.append(destination)
            except BaseException:
                for destination in reversed(published):
                    destination.unlink(missing_ok=True)
                for backup, destination in reversed(backups):
                    backup.replace(destination)
                raise

    def _convert_to_cgir(
        self,
        source_path: Path,
        *,
        output_data_file: Path | None = None,
    ) -> str:
        """Compile ONNX to textual CGC Input IR with FoundryToolbox."""
        try:
            with FoundryCompiler() as compiler:
                # model_directory lets Foundry resolve source ONNX external-data
                # locations while lazy external mode avoids loading those weights.
                serialized = compiler.compile_onnx(
                    source_path.read_bytes(),
                    model_directory=source_path.parent,
                    update_opset=self.options.update_opset,
                    topo_sort_nodes=self.options.topo_sort_nodes,
                    include_initializers=not self.options.external_weights,
                    enable_lazy_external_data=self.options.external_weights,
                    output_data_file=output_data_file,
                    freeze_dims=self._freeze_dims,
                )
        except FoundryCompileError as e:
            raise RuntimeError(self._format_foundry_error(e)) from e

        try:
            ir = serialized.decode("utf-8")
        except UnicodeDecodeError as e:
            raise RuntimeError(
                "FoundryToolbox returned non-UTF-8 CGC textual MLIR"
            ) from e
        if not ir.strip():
            raise RuntimeError("FoundryToolbox returned empty CGC textual MLIR")
        if "cgc." not in ir.lower() and "#cgc" not in ir.lower():
            raise RuntimeError("FoundryToolbox output does not contain the CGC dialect")
        return ir

    def _format_foundry_error(self, error: FoundryCompileError) -> str:
        """Format native Foundry diagnostics for an export user."""
        details = [f"Foundry failed to convert ONNX to CGC IR [{error.result_name}]."]
        if error.unsupported_op:
            details.append(f"ONNX operator '{error.unsupported_op}' is not supported.")
        elif error.missing_external_data:
            details.append(
                f"ONNX external weights file was not found: "
                f"'{error.missing_external_data}'."
            )
        if error.native_message:
            details.append(error.native_message)
        if error.result_name == "SHAPE_INFERENCE" and not self._freeze_dims:
            details.append(
                "If symbolic dimensions caused this failure, retry with "
                "--options freeze-dims=batch=1,seq=128 "
                "using the model's dimension names."
            )
        return " ".join(details)

def export_cgc(
    model: str | Path,
    output_path: str | Path,
    options: CGCOptions,
) -> None:
    """Export an ONNX model using :class:`CGCExporter`."""
    CGCExporter(options).export_onnx(
        model=model,
        output_path=output_path,
    )


def _parse_freeze_dims(value: str) -> dict[str, int]:
    """Parse comma-separated symbolic dimension overrides."""
    if not value.strip():
        return {}
    overrides: dict[str, int] = {}
    for assignment in value.split(","):
        name, separator, raw_size = assignment.partition("=")
        name = name.strip()
        raw_size = raw_size.strip()
        if not separator or not name or not raw_size:
            raise ValueError(
                "freeze-dims expects comma-separated NAME=SIZE values, "
                f"got {assignment!r}."
            )
        try:
            size = int(raw_size)
        except ValueError as e:
            raise ValueError(
                f"freeze-dims value for {name!r} must be an integer, got {raw_size!r}."
            ) from e
        if size <= 0:
            raise ValueError(
                f"freeze-dims value for {name!r} must be greater than zero."
            )
        if name in overrides:
            raise ValueError(f"freeze-dims contains duplicate dimension {name!r}.")
        overrides[name] = size
    return overrides


__all__ = ["CGCExportResult", "CGCExporter", "CGCOptions", "export_cgc"]
