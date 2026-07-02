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
from ..utils.logging import configure_logging


logger = logging.getLogger(__name__)
console = Console()


def _delete_onnx_with_external_data(onnx_path: Path) -> None:
    """Delete an ONNX file and its external data files."""
    import onnx
    from onnx.external_data_helper import ExternalDataInfo

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
        ext_files: set[str] = set()
        for tensor in model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                ext_files.add(ExternalDataInfo(tensor).location)
        for name in ext_files:
            data_path = onnx_path.parent / name
            if data_path.exists():
                data_path.unlink()
    except Exception:
        logger.debug("Could not parse external data from %s", onnx_path, exc_info=True)

    if onnx_path.exists():
        onnx_path.unlink()


@click.command()
@cli_utils.model_option(
    required=True,
    help_text="HuggingFace model name or local path (e.g., prajjwal1/bert-tiny)",
)
@cli_utils.output_option("Output ONNX file path (e.g., model.onnx)", required=True)
@cli_utils.overwrite_option()
@click.option(
    "--with-report/--no-with-report",
    default=False,
    show_default=True,
    help="Generate full export reports (markdown, JSON, console tree)",
)
@click.option(
    "--hierarchy/--no-hierarchy",
    "hierarchy",
    default=True,
    show_default=True,
    help="Embed hierarchy_tag metadata in ONNX output",
)
@click.option(
    "--clean-onnx",
    "clean_onnx",
    is_flag=True,
    default=False,
    hidden=True,
    help="Deprecated alias for --no-hierarchy.",
)
@click.option(
    "--dynamo/--no-dynamo",
    "dynamo",
    default=False,
    show_default=True,
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
@click.option(
    "--submodel",
    type=str,
    default=None,
    help=(
        "Export a specific sub-model from a composite model "
        "(e.g., 'encoder', 'decoder'). "
        "Omit to export all sub-models automatically."
    ),
)
@cli_utils.build_config_option()
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@click.pass_context
def export(
    ctx: click.Context,
    model: str,
    output: Path,
    overwrite: bool,
    verbose: int,
    quiet: bool,
    with_report: bool,
    hierarchy: bool,
    clean_onnx: bool,
    dynamo: bool,
    torch_module: str | None,
    task: str | None,
    input_specs: Path | None,
    export_config: Path | None,
    shape_config: Path | None,
    submodel: str | None,
    config_file: Path | None,
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

        # Export all sub-models of a composite model (auto-detected)
        winml export -m google-t5/t5-small --task translation -o t5.onnx

        # Export only the encoder sub-model
        winml export -m google-t5/t5-small --task translation -o t5.onnx --submodel encoder
    """
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    # Apply build config defaults (CLI explicit options take precedence).
    # Read raw JSON so missing keys are distinguishable from dataclass defaults.
    _build_export_dict: dict = {}
    if config_file is not None:
        _, raw_cfg = cli_utils.load_build_config(config_file)
        lc = raw_cfg.get("loader") or {}
        ec = raw_cfg.get("export") or {}
        _build_export_dict = ec
        if not cli_utils.is_cli_provided(ctx, "task") and "task" in lc:
            task = lc["task"]
        if (
            not cli_utils.is_cli_provided(ctx, "hierarchy")
            and not cli_utils.is_cli_provided(ctx, "clean_onnx")
            and "enable_hierarchy_tags" in ec
        ):
            hierarchy = ec["enable_hierarchy_tags"]
        if not cli_utils.is_cli_provided(ctx, "dynamo") and "dynamo" in ec:
            dynamo = ec["dynamo"]

    from ..export import export_pytorch as export_onnx
    from ..loader import load_hf_model

    if clean_onnx:
        click.echo(
            "warning: --clean-onnx is deprecated; use --no-hierarchy instead.",
            err=True,
        )
        if not cli_utils.is_cli_provided(ctx, "hierarchy"):
            hierarchy = False

    # Configure logging — stderr only, shared format with the rest of the CLI.
    configure_logging(verbosity=verbose, quiet=quiet)

    # Show export info
    console.print(f"[bold blue]Model:[/bold blue] {model}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")
    if with_report:
        console.print("[bold blue]Report:[/bold blue] Enabled (md, json, console)")
    if input_specs:
        console.print(f"[bold blue]Input specs:[/bold blue] {input_specs}")
    if export_config:
        console.print(f"[bold blue]Export config:[/bold blue] {export_config}")

    output_path = Path(output)

    # Create output directory if needed
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

    # Load shape overrides from JSON
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

    # Parse torch-module option
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

    # ── Composite model detection ──────────────────────────────────────
    # Resolve whether this is a composite (multi-component) model.
    # When --submodel is given, validate it against the composite's
    # components and export only that one.  When omitted and the model
    # is composite, export every sub-model automatically.
    composite_components = _resolve_composite_model_components(model, task)

    if submodel is not None:
        if composite_components is None:
            raise click.ClickException(
                f"--submodel '{submodel}' was specified, but '{model}' "
                f"is not a composite model (no sub-models detected)."
            )
        if submodel not in composite_components:
            raise click.ClickException(
                f"Unknown sub-model '{submodel}'. "
                f"Available: {', '.join(composite_components.keys())}"
            )

    if composite_components is not None:
        # Determine which components to export
        if submodel is not None:
            targets = {submodel: composite_components[submodel]}
        else:
            targets = composite_components

        console.print(f"[bold blue]Composite model:[/bold blue] {', '.join(targets.keys())}")

        exported_paths: list[Path] = []
        for component_name, component_task in targets.items():
            component_output = output_path.with_stem(f"{output_path.stem}_{component_name}")
            cli_utils.guard_output(component_output, overwrite)
            console.print(
                f"\n[bold]Exporting sub-model '{component_name}' (task={component_task})...[/bold]"
            )
            _export_single_model(
                ctx=ctx,
                model_id=model,
                task=component_task,
                output_path=component_output,
                overwrite=overwrite,
                verbose=verbose,
                with_report=with_report,
                hierarchy=hierarchy,
                dynamo=dynamo,
                input_specs=input_specs,
                export_config_dict=export_config_dict,
                shape_overrides=shape_overrides,
                build_export_dict=_build_export_dict,
                export_onnx_fn=export_onnx,
                load_hf_model_fn=load_hf_model,
            )
            exported_paths.append(component_output)

        console.print("\n[bold green]All sub-models exported![/bold green]")
        for p in exported_paths:
            console.print(f"  {p}")
        return

    # ── Single (non-composite) model export ────────────────────────────
    cli_utils.guard_output(output_path, overwrite)
    _export_single_model(
        ctx=ctx,
        model_id=model,
        task=task,
        output_path=output_path,
        overwrite=overwrite,
        verbose=verbose,
        with_report=with_report,
        hierarchy=hierarchy,
        dynamo=dynamo,
        input_specs=input_specs,
        export_config_dict=export_config_dict,
        shape_overrides=shape_overrides,
        build_export_dict=_build_export_dict,
        export_onnx_fn=export_onnx,
        load_hf_model_fn=load_hf_model,
    )
    console.print(f"\n[bold green]Success![/bold green] Model exported to: {output_path}")

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


def _resolve_composite_model_components(
    hf_model: str,
    task: str | None,
) -> dict[str, str] | None:
    """Resolve the composite ``_SUB_MODEL_CONFIG`` for a model, else ``None``.

    Two detection paths:
    - Explicit ``--task``: direct registry lookup via ``resolve_composite``.
    - No ``--task``: ``resolve_task`` auto-detects the task and returns
      the composite bridge (its ``.composite`` field), if any.
    """
    from transformers import AutoConfig

    from ..loader.resolution import resolve_composite, resolve_task

    try:
        hf_config = AutoConfig.from_pretrained(hf_model)
    except Exception:
        return None

    if task is not None:
        return resolve_composite(hf_config.model_type, task)

    return resolve_task(hf_config).composite


def _export_single_model(
    *,
    ctx: click.Context,
    model_id: str,
    task: str | None,
    output_path: Path,
    overwrite: bool,
    verbose: int,
    with_report: bool,
    hierarchy: bool,
    dynamo: bool,
    input_specs: Path | None,
    export_config_dict: dict,
    shape_overrides: dict | None,
    build_export_dict: dict,
    export_onnx_fn: object,
    load_hf_model_fn: object,
) -> None:
    """Export a single (non-composite) model to ONNX.

    Shared by both the single-model path and the per-component loop
    for composite models.
    """
    from ..export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig

    # ── I/O tensor auto-resolution ─────────────────────────────────────
    input_tensors: list[InputTensorSpec] | None = None
    output_tensors: list[OutputTensorSpec] | None = None

    from ..export import ONNXConfigNotFoundError

    try:
        from ..export import resolve_export_config as resolve_cfg

        auto_export_cfg, _ = resolve_cfg(
            model_id=model_id,
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
    except ONNXConfigNotFoundError as e:
        logger.debug("I/O tensor auto-resolution unavailable: %s", e)
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    except Exception as e:
        logger.debug("I/O tensor auto-resolution failed: %s", e)

    # --input-specs overrides individual fields on the auto-resolved input_tensors.
    if input_specs:
        try:
            with input_specs.open() as f:
                input_specs_dict = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Failed to load input specs:[/bold red] {e}")
            raise click.ClickException(f"Failed to load input specs: {e}") from e

        if input_tensors is None:
            input_tensors = []
        by_name = {t.name: t for t in input_tensors}
        for name, spec in input_specs_dict.items():
            shape = tuple(spec["shape"]) if spec.get("shape") else None
            dtype = spec.get("dtype")
            if name in by_name:
                existing = by_name[name]
                if dtype is not None:
                    existing.dtype = dtype
                if shape is not None:
                    existing.shape = shape
            else:
                input_tensors.append(
                    InputTensorSpec(name=name, dtype=dtype or "float32", shape=shape)
                )
        console.print(f"[dim]Applied input-spec overrides: {list(input_specs_dict.keys())}[/dim]")

    # ── Build WinMLExportConfig ────────────────────────────────────────
    config_kwargs: dict = {}
    config_kwargs.update(build_export_dict)
    config_kwargs.update(export_config_dict)
    if cli_utils.is_cli_provided(ctx, "hierarchy") or cli_utils.is_cli_provided(ctx, "clean_onnx"):
        config_kwargs["enable_hierarchy_tags"] = hierarchy
    if cli_utils.is_cli_provided(ctx, "verbose"):
        config_kwargs["verbose"] = bool(verbose)
    if cli_utils.is_cli_provided(ctx, "dynamo"):
        config_kwargs["dynamo"] = dynamo

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

    # ── Execute export ─────────────────────────────────────────────────
    try:
        console.print("[bold]Starting HTP export...[/bold]")

        pytorch_model, _, detected_task = load_hf_model_fn(model_id, task=task)
        if task:
            console.print(f"[dim]Task (override): {detected_task}[/dim]")
        else:
            console.print(f"[dim]Detected task: {detected_task}[/dim]")

        export_stats = export_onnx_fn(
            model=pytorch_model,
            output_path=output_path,
            export_config=cfg,
            model_id=model_id,
            task=detected_task,
            verbose=bool(verbose),
            enable_reporting=with_report,
        )
        logger.debug("Export stats: %s", export_stats)

        console.print(f"[bold green]Exported:[/bold green] {output_path}")

        if with_report:
            base_name = output_path.stem
            report_dir = output_path.parent
            md_report = report_dir / f"{base_name}_htp_export_report.md"
            json_metadata = report_dir / f"{base_name}_htp_metadata.json"
            if md_report.exists():
                console.print(f"  Report: {md_report}")
            if json_metadata.exists():
                console.print(f"  Metadata: {json_metadata}")

    except Exception as e:
        console.print(f"\n[bold red]Export failed:[/bold red] {e}")
        debug_mode = bool((ctx.obj or {}).get("debug"))
        if debug_mode:
            logger.exception("Export failed")
        else:
            logger.error("Export failed: %s", e)
        raise click.ClickException(f"Export failed: {e}") from e
