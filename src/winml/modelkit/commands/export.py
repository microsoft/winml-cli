# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Export command for winml CLI.

This module provides the export command that uses export_onnx() as the single
implementation path for HuggingFace to ONNX model conversion.

Features:
- Uses export_onnx() from winml.modelkit.export.pytorch as single implementation path
- Leverages WinMLExportConfig for unified configuration
- Supports MODEL_BUILD_CONFIGS lookup for input_tensors fallback

Usage:
    winml export --model MODEL --output PATH [--verbose] [--with-report]

Examples:
    winml export -m prajjwal1/bert-tiny -o model.onnx
    winml export -m facebook/convnext-tiny-224 -o convnext.onnx -v --with-report
    winml export -m bert-base-uncased -o bert.onnx --input-specs inputs.json
    winml export -m bert-base-uncased -o bert.onnx --export-config config.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console

from ..utils import cli as cli_utils


logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=str,
    help="HuggingFace model name or local path (e.g., prajjwal1/bert-tiny)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output ONNX file path (e.g., model.onnx)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose console output (8-step format)",
)
@click.option(
    "--with-report",
    is_flag=True,
    default=False,
    help="Generate full export reports (markdown, JSON, console tree)",
)
@click.option(
    "--clean-onnx",
    "--no-hierarchy",
    "no_hierarchy",
    is_flag=True,
    default=False,
    help="Skip embedding hierarchy_tag metadata in ONNX (clean ONNX output)",
)
@click.option(
    "--dynamo",
    "dynamo",
    is_flag=True,
    default=False,
    help="Enable PyTorch 2.9+ dynamo export for rich node metadata",
)
@click.option(
    "--torch-module",
    type=str,
    default=None,
    help="Include torch.nn modules in hierarchy (comma-separated, e.g., LayerNorm,Embedding)",
)
@click.option(
    "--input-specs",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON file with input specifications (auto-generates if not provided)",
)
@click.option(
    "--task",
    "-t",
    type=str,
    default=None,
    help="Override auto-detected task (e.g., image-feature-extraction, feature-extraction)",
)
@click.option(
    "--export-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="ONNX export configuration JSON (opset_version, do_constant_folding, etc.)",
)
@click.option(
    "--shape-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='JSON with shape overrides (e.g., {"sequence_length": 2048, "height": 640}).',
)
@cli_utils.build_config_option
@click.pass_context
def export(
    ctx: click.Context,
    model: str,
    output: Path,
    verbose: bool,
    with_report: bool,
    no_hierarchy: bool,
    dynamo: bool,
    torch_module: str | None,
    task: str | None,
    input_specs: Path | None,
    export_config: Path | None,
    shape_config: Path | None,
    build_config_file: Path | None,
) -> None:
    r"""Export HuggingFace model to ONNX format with HTP.

    This command converts a HuggingFace model to ONNX format using the
    Hierarchy-preserving Tags Protocol (HTP) with optional full reporting.

    The export process (8 steps):
    1. Model Preparation - Load and configure model
    2. Input Generation - Generate example inputs
    3. Hierarchy Building - Trace module execution
    4. ONNX Export - Convert to ONNX format (TorchScript by default)
    5. Node Tagger Creation - Create tagger from hierarchy
    6. Node Tagging - Apply hierarchy tags to nodes
    7. Tag Injection - Embed tags in ONNX node metadata_props
    8. Metadata Generation - Generate reports (if --with-report)

    \b
    Examples:
        # Basic export
        winml export --model prajjwal1/bert-tiny --output model.onnx

        # Short form
        winml export -m prajjwal1/bert-tiny -o model.onnx

        # With verbose output and full reporting
        winml export -m facebook/convnext-tiny-224 -o convnext.onnx -v --with-report

        # Clean ONNX output (no hierarchy metadata, for optimization)
        winml export -m prajjwal1/bert-tiny -o model.onnx --clean-onnx

        # Use PyTorch dynamo export (for rich node metadata)
        winml export -m prajjwal1/bert-tiny -o model.onnx --dynamo

        # Include torch.nn modules in hierarchy
        winml export -m prajjwal1/bert-tiny -o model.onnx --torch-module LayerNorm,Embedding

        # Custom input specifications from JSON file
        winml export -m bert-base-uncased -o bert.onnx --input-specs inputs.json

        # Custom ONNX export configuration
        winml export -m bert-base-uncased -o bert.onnx --export-config config.json
    """
    # Inherit debug mode from parent
    if ctx.obj.get("debug"):
        verbose = True

    # Apply build config defaults (CLI explicit options take precedence)
    _build_export_cfg = None
    if build_config_file is not None:
        build_cfg = cli_utils.load_build_config(build_config_file)
        if build_cfg.export:
            _build_export_cfg = build_cfg.export
        if build_cfg.loader and not cli_utils.is_cli_provided(ctx, "task"):
            task = task or build_cfg.loader.task
        if _build_export_cfg and not cli_utils.is_cli_provided(ctx, "no_hierarchy"):
            no_hierarchy = not _build_export_cfg.enable_hierarchy_tags
        if _build_export_cfg and not cli_utils.is_cli_provided(ctx, "dynamo"):
            dynamo = _build_export_cfg.dynamo

    from ..export.config import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
    from ..export.pytorch import export_pytorch as export_onnx
    from ..loader import load_hf_model

    # Configure logging based on verbose flag
    if verbose:
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    # Show export info
    console.print(f"[bold blue]Model:[/bold blue] {model}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")
    if with_report:
        console.print("[bold blue]Report:[/bold blue] Enabled (md, json, console)")
    if input_specs:
        console.print(f"[bold blue]Input specs:[/bold blue] {input_specs}")
    if export_config:
        console.print(f"[bold blue]Export config:[/bold blue] {export_config}")

    # Create output directory if needed
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load export configuration from JSON file if provided, or create default
    export_config_dict: dict = {}
    if export_config:
        try:
            with export_config.open() as f:
                export_config_dict = json.load(f)
            console.print(f"[dim]Loaded export config: {list(export_config_dict.keys())}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to load export config:[/bold red] {e}")
            raise click.ClickException(f"Failed to load export config: {e}") from e

    # Load input/output specifications
    input_tensors: list[InputTensorSpec] | None = None
    output_tensors: list[OutputTensorSpec] | None = None
    if input_specs:
        try:
            with input_specs.open() as f:
                input_specs_dict = json.load(f)
            # Convert dict format to InputTensorSpec list
            input_tensors = []
            for name, spec in input_specs_dict.items():
                input_tensors.append(
                    InputTensorSpec(
                        name=name,
                        dtype=spec.get("dtype", "float32"),
                        shape=tuple(spec.get("shape", [])) if spec.get("shape") else None,
                    )
                )
            console.print(f"[dim]Loaded {len(input_tensors)} input specifications[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to load input specs:[/bold red] {e}")
            raise click.ClickException(f"Failed to load input specs: {e}") from e
    else:
        # Load shape overrides from JSON (outside catch-all so errors propagate)
        shape_overrides = None
        if shape_config:
            try:
                with shape_config.open() as f:
                    shape_overrides = json.load(f)
                if not isinstance(shape_overrides, dict):
                    raise click.ClickException(
                        f"--shape-config must contain a JSON object, "
                        f"got {type(shape_overrides).__name__}"
                    )
            except json.JSONDecodeError as e:
                raise click.ClickException(
                    f"Invalid JSON in --shape-config: {shape_config}: {e}"
                ) from e
            console.print(f"[dim]Shape overrides: {shape_overrides}[/dim]")

        # Auto-resolve input/output tensors via loader + Optimum
        try:
            from ..export.config import resolve_export_config as resolve_cfg

            auto_export_cfg, _ = resolve_cfg(
                model_id=model,
                task=task,
                shape_config=shape_overrides,
            )
            if auto_export_cfg.input_tensors:
                input_tensors = auto_export_cfg.input_tensors
                console.print(
                    f"[dim]Auto-resolved input specs: {[t.name for t in input_tensors]}[/dim]"
                )
            if auto_export_cfg.output_tensors:
                output_tensors = auto_export_cfg.output_tensors
                console.print(
                    f"[dim]Auto-resolved output specs: {[t.name for t in output_tensors]}[/dim]"
                )
        except Exception as e:
            logger.debug("I/O tensor auto-resolution failed: %s", e)

    # Build WinMLExportConfig from loaded settings
    config_kwargs = {}
    # Layer 1: build config defaults (lowest precedence)
    if _build_export_cfg is not None:
        config_kwargs.update(_build_export_cfg.to_dict())
    # Layer 2: --export-config file overrides
    config_kwargs.update(export_config_dict)
    # Layer 3: explicit CLI options (highest precedence)
    config_kwargs["enable_hierarchy_tags"] = not no_hierarchy
    config_kwargs["verbose"] = verbose
    if cli_utils.is_cli_provided(ctx, "dynamo"):
        config_kwargs["dynamo"] = dynamo

    # Add input/output tensors if we resolved them
    if input_tensors:
        config_kwargs["input_tensors"] = input_tensors
    if output_tensors:
        config_kwargs["output_tensors"] = output_tensors

    try:
        cfg = WinMLExportConfig.from_dict(config_kwargs)
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        logger.exception("Failed to create export config")
        raise click.ClickException(f"Configuration error: {e}") from e

    # Parse torch-module option
    # TODO: export_onnx() does not currently support torch_module parameter.
    # This would need to be passed through to HTPExporter.
    # For now, we note this as a limitation and log a warning if used.
    if torch_module:
        console.print(
            "[yellow]Warning:[/yellow] --torch-module is not yet supported in export_onnx(). "
            "This option will be ignored."
        )
        logger.warning(
            "torch_module parameter (%s) is not supported by export_onnx(). "
            "TODO: Add torch_module support to export_onnx() and WinMLExportConfig.",
            torch_module,
        )

    # Handle --dynamo flag
    if dynamo:
        console.print(
            "[yellow]Warning:[/yellow] --dynamo is not yet supported in export_onnx(). "
            "export_onnx() defaults to dynamo=False for QNN compatibility."
        )
        logger.warning(
            "dynamo=True is not supported by export_onnx(). "
            "TODO: Add dynamo support to WinMLExportConfig if needed."
        )

    # Execute export
    try:
        console.print("\n[bold]Starting HTP export...[/bold]")

        # Load model with task detection (CLI is the orchestration layer)
        pytorch_model, _, detected_task = load_hf_model(model, task=task)
        if task:
            console.print(f"[dim]Task (override): {detected_task}[/dim]")
        else:
            console.print(f"[dim]Detected task: {detected_task}[/dim]")

        # Export using export_onnx() - the single implementation path
        result_path = export_onnx(
            model=pytorch_model,
            output_path=output_path,
            export_config=cfg,
            model_id=model,  # For metadata
            task=detected_task,  # Use detected task for proper OnnxConfig lookup
            verbose=verbose,
            enable_reporting=with_report,
        )

        # Show results
        console.print(f"\n[bold green]Success![/bold green] Model exported to: {result_path}")

        # Show report file locations if enabled
        if with_report:
            base_name = output_path.stem
            report_dir = output_path.parent
            console.print("\n[bold]Generated reports:[/bold]")
            md_report = report_dir / f"{base_name}_htp_export_report.md"
            json_metadata = report_dir / f"{base_name}_htp_metadata.json"
            if md_report.exists():
                console.print(f"  Markdown: {md_report}")
            if json_metadata.exists():
                console.print(f"  JSON: {json_metadata}")

    except Exception as e:
        console.print(f"\n[bold red]Export failed:[/bold red] {e}")
        logger.exception("Export failed")
        raise click.ClickException(f"Export failed: {e}") from e
