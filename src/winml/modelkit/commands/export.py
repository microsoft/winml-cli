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


def _warn_partial_composite(completed: list[Path]) -> None:
    """Warn that a composite export failed mid-run, listing what was written.

    We deliberately do NOT delete anything: the targets may be pre-existing files
    the user chose to ``--overwrite``, and a component can fail before touching its
    file, so auto-deleting could destroy artifacts this run never actually wrote.
    Instead we surface the completed sub-models and let the user decide whether to
    keep or remove the partial composite.
    """
    if not completed:
        return
    console.print(
        "\n[yellow]Warning:[/yellow] composite export did not finish; "
        f"{len(completed)} sub-model(s) were written/updated by this run:"
    )
    for onnx_path in completed:
        console.print(f"  • {onnx_path}")
    console.print(
        "[yellow]The export did not complete for every sub-model.[/yellow] "
        "Review these files and remove them if you don't want to keep the "
        "partial export."
    )


def _load_json_object(path: Path, option_name: str) -> dict:
    """Load a JSON object from a CLI option path."""
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in {option_name}: {path}: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Failed to load {option_name}: {e}") from e

    if not isinstance(data, dict):
        raise click.ClickException(
            f"{option_name} must contain a JSON object, got {type(data).__name__}"
        )
    return data


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
    "--dynamic-axes",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "JSON dynamic axes mapping for ONNX export "
        '(e.g., {"input_ids": {"0": "batch", "1": "sequence"}}).'
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
    dynamic_axes: Path | None,
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

        # Dynamic dimensions from a dedicated JSON file
        winml export -m bert-base-uncased -o bert.onnx --dynamic-axes dynamic_axes.json
    """
    # Classify the -m value once (existence-first). Export only works with
    # HuggingFace model IDs — reject ONNX files and folders early.
    if model:
        model_input = cli_utils.classify_model_input(model)
        if model_input.kind is cli_utils.ModelInputKind.ONNX_FILE:
            raise click.UsageError(
                "export requires a HuggingFace model ID, not an ONNX file. "
                "Use 'winml inspect -m model.onnx' to inspect an existing ONNX model."
            )
        if model_input.kind is cli_utils.ModelInputKind.FOLDER:
            raise click.UsageError(
                "export requires a HuggingFace model ID, not a directory. "
                "Provide a HuggingFace model ID (e.g., prajjwal1/bert-tiny)."
            )

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

    from ..export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
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
    if dynamic_axes:
        console.print(f"[bold blue]Dynamic axes:[/bold blue] {dynamic_axes}")

    output_path = Path(output)

    # Load export configuration from JSON file if provided (task-independent).
    export_config_dict: dict = {}
    if export_config:
        export_config_dict = _load_json_object(export_config, "--export-config")
        console.print(f"[dim]Loaded export config: {list(export_config_dict.keys())}[/dim]")

    # Load shape overrides from JSON (task-independent).
    shape_overrides = None
    if shape_config:
        shape_overrides = _load_json_object(shape_config, "--shape-config")
        console.print(f"[dim]Shape overrides: {shape_overrides}[/dim]")

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = _load_json_object(dynamic_axes, "--dynamic-axes")
        console.print(f"[dim]Dynamic axes: {dynamic_axes_dict}[/dim]")

    # One-time warnings (apply to every sub-model in a composite export).
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
    if dynamo:
        console.print(
            "[yellow]Warning:[/yellow] --dynamo is not yet supported in export_onnx(). "
            "export_onnx() defaults to dynamo=False for QNN compatibility."
        )
        logger.warning(
            "dynamo=True is not supported by export_onnx(). "
            "TODO: Add dynamo support to WinMLExportConfig if needed."
        )

    def _run_component_export(component_task: str | None, out_path: Path) -> None:
        """Resolve I/O, build config, load, and export one (model, task) to ``out_path``."""
        from ..export import ONNXConfigNotFoundError

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load input/output specifications.
        #
        # We ALWAYS run Optimum auto-resolution because it provides authoritative
        # output_tensors (names that match the actual ONNX graph). --input-specs
        # then overrides input_tensors only; output_tensors stays from Optimum so
        # tasks like feature-extraction don't trip torch.onnx.export with extra
        # dataclass field names that aren't in the traced graph.
        input_tensors: list[InputTensorSpec] | None = None
        output_tensors: list[OutputTensorSpec] | None = None

        try:
            from ..export import resolve_export_config as resolve_cfg

            auto_export_cfg, _ = resolve_cfg(
                model_id=model,
                task=component_task,
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
            # model_type is not registered in Optimum's TasksManager (e.g. CLIP/SigLIP
            # sub-encoder variants like clip-text-model / clip-vision-model that only
            # live in MODEL_BUILD_CONFIGS, or a model_type from a newer transformers
            # release we don't know yet). Fall through: downstream MODEL_BUILD_CONFIGS
            # lookup or user-supplied --input-specs takes over.
            logger.debug("I/O tensor auto-resolution unavailable: %s", e)
        except ValueError as e:
            # Mirrors `winml config`: surface (model, task) incompatibility raised by
            # Optimum's TasksManager as a clean usage error instead of letting it fall
            # through to a misleading "Unrecognized configuration class" traceback
            # later in load_hf_model.
            raise click.UsageError(str(e)) from e
        except Exception as e:
            logger.debug("I/O tensor auto-resolution failed: %s", e)

        # --input-specs overrides individual fields on the auto-resolved input_tensors.
        # Names matched against auto-resolve get their dtype/shape patched; unknown
        # names are appended. output_tensors are left untouched.
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
            console.print(
                f"[dim]Applied input-spec overrides: {list(input_specs_dict.keys())}[/dim]"
            )

        # Build WinMLExportConfig from loaded settings
        config_kwargs: dict = {}
        # Layer 1: build config defaults (lowest precedence)
        config_kwargs.update(_build_export_dict)
        # Layer 2: --export-config file overrides
        config_kwargs.update(export_config_dict)
        # Layer 3: explicit CLI options (highest precedence)
        if cli_utils.is_cli_provided(ctx, "hierarchy") or cli_utils.is_cli_provided(
            ctx, "clean_onnx"
        ):
            config_kwargs["enable_hierarchy_tags"] = hierarchy
        if cli_utils.is_cli_provided(ctx, "verbose"):
            config_kwargs["verbose"] = bool(verbose)
        if cli_utils.is_cli_provided(ctx, "dynamo"):
            config_kwargs["dynamo"] = dynamo
        if dynamic_axes_dict is not None:
            config_kwargs["dynamic_axes"] = dynamic_axes_dict

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

        # Load model with task detection (CLI is the orchestration layer)
        pytorch_model, _, detected_task = load_hf_model(model, task=component_task)
        if component_task:
            console.print(f"[dim]Task (override): {detected_task}[/dim]")
        else:
            console.print(f"[dim]Detected task: {detected_task}[/dim]")

        export_stats = export_onnx(
            model=pytorch_model,
            output_path=out_path,
            export_config=cfg,
            model_id=model,
            task=detected_task,
            verbose=bool(verbose),
            enable_reporting=with_report,
        )
        logger.debug("Export stats: %s", export_stats)

        console.print(f"\n[bold green]Success![/bold green] Model exported to: {out_path}")

        # Show report file locations if enabled
        if with_report:
            base_name = out_path.stem
            report_dir = out_path.parent
            console.print("\n[bold]Generated reports:[/bold]")
            md_report = report_dir / f"{base_name}_htp_export_report.md"
            json_metadata = report_dir / f"{base_name}_htp_metadata.json"
            if md_report.exists():
                console.print(f"  Markdown: {md_report}")
            if json_metadata.exists():
                console.print(f"  JSON: {json_metadata}")

    # Detect a composite pipeline (registry-driven). A composite fans out into one
    # ONNX per sub-component, each written next to <output> with a _<component>
    # stem suffix; a plain model exports to the single output path as before.
    # Detection suppresses only the expected "not a resolvable HF config" case
    # (OSError — e.g. the model reference isn't a hub id / has no local config);
    # intentional loud guards (empty registry, model-task incompatibility) and any
    # unexpected failure are surfaced rather than masked as "not composite".
    components = None
    try:
        from ..loader.resolution import resolve_composite_components

        components = resolve_composite_components(model, task=task)
    except click.ClickException:
        raise
    except ValueError as e:
        # A genuine (model, task) incompatibility, or an ambiguous composite that
        # needs an explicit --task, is surfaced as a usage error (mirroring
        # `winml config`) instead of being silently downgraded to a single export.
        raise click.UsageError(str(e)) from e
    except RuntimeError:
        # The empty-COMPOSITE_MODEL_REGISTRY guard is meant to fail loudly (a
        # registrations moved/renamed refactor mistake); never mask it as
        # "not composite".
        raise
    except OSError as e:
        # Expected "unavailable" case: the model reference can't be loaded as a HF
        # config (not a hub id / no local config.json). Fall through to the
        # single-model export path, which surfaces its own load error if the
        # reference is truly invalid.
        logger.debug("Composite detection unavailable (config not resolvable): %s", e)
    except Exception as e:
        # Anything else is unexpected: surface it loudly rather than silently
        # producing a single-model artifact for a possibly-composite model.
        raise click.ClickException(f"Composite model detection failed unexpectedly: {e}") from e

    try:
        console.print("\n[bold]Starting HTP export...[/bold]")

        if components:
            if input_specs:
                raise click.UsageError(
                    "--input-specs is not supported for composite models; each sub-model "
                    "resolves its own I/O. Export a sub-model individually with --task if "
                    "you need custom input specs."
                )
            console.print(
                f"[dim]Composite model: {len(components)} sub-models "
                f"(fanned out from {output_path.name} with a _<component> suffix)[/dim]"
            )
            # Fan out flat, suffixing the output stem per component to match the
            # sibling `winml config` layout and the runtime's `*_model.onnx`
            # discovery: e.g. `model.onnx` -> `model_decoder_prefill.onnx`.
            # Each component is loaded and exported independently because a
            # composite's sub-tasks map to distinct model classes/heads (e.g.
            # decoder-prefill vs. decoder-with-past), so the full model is
            # reloaded per component — this N-load cost is intentional.
            sub_outputs = {
                name: output_path.with_stem(f"{output_path.stem}_{name}") for name in components
            }
            # Guard every target up front so an overwrite collision on a later
            # component can't leave an earlier one already written.
            for sub_out in sub_outputs.values():
                cli_utils.guard_output(sub_out, overwrite)

            # Track sub-models this invocation actually completes. On a mid-run
            # failure we do NOT delete anything (the targets may be pre-existing
            # files the user chose to --overwrite, and a component can fail before
            # touching its file). Instead we warn and list what was written so the
            # user decides whether to keep or remove the partial composite.
            completed: list[Path] = []
            try:
                for name, component_task in components.items():
                    sub_out = sub_outputs[name]
                    console.print(
                        f"\n[bold blue]Sub-model:[/bold blue] {name} (task={component_task})"
                    )
                    _run_component_export(component_task, sub_out)
                    completed.append(sub_out)
            except BaseException:
                _warn_partial_composite(completed)
                raise
        else:
            cli_utils.guard_output(output_path, overwrite)
            _run_component_export(task, output_path)

    except (click.UsageError, click.ClickException):
        raise
    except Exception as e:
        console.print(f"\n[bold red]Export failed:[/bold red] {e}")
        debug_mode = bool((ctx.obj or {}).get("debug"))
        if debug_mode:
            logger.exception("Export failed")
        else:
            logger.error("Export failed: %s", e)
        raise click.ClickException(f"Export failed: {e}") from e
