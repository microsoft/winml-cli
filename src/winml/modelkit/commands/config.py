# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Config generation command (v2, Rich UI) for WinML CLI.

Generates WinMLBuildConfig for a HuggingFace model or a pre-exported ONNX file
by auto-detecting task, model class, and I/O specifications.

When -m points to an existing .onnx file, generates a simpler config with
export=None (marking it as an ONNX build that skips the export stage).

Usage:
    winml config -m microsoft/resnet-50
    winml config -m bert-base-uncased --task text-classification
    winml config -m model.onnx
    winml config --model-type bert
    winml config --model-type bert --task fill-mask
    winml config -m microsoft/resnet-50 --module ResNetConvLayer
    winml config -m bert-base-uncased -o config.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

from ..utils import cli as cli_utils
from ..utils.console import (
    get_console,
    print_command_header,
    print_error,
    print_io_specs_detail,
    print_io_specs_na,
    print_kv,
    print_success,
)


logger = logging.getLogger(__name__)
console = get_console()


def _apply_stage_overrides(cfg: Any, *, no_quant: bool, no_compile: bool) -> None:
    """Apply --no-quant and --no-compile CLI overrides to a config."""
    if no_quant:
        cfg.quant = None
    if no_compile:
        cfg.compile = None


def _is_onnx_file(model_input: str) -> bool:
    """Check if input is a path to an existing .onnx file."""
    path = Path(model_input)
    return path.suffix == ".onnx" and path.exists()


@click.command("config")
@click.option(
    "-m",
    "--model",
    "hf_model",
    default=None,
    help="HuggingFace model ID (e.g., microsoft/resnet-50) or path to .onnx file. "
    "Optional when --model-type is provided.",
)
@click.option(
    "-t",
    "--task",
    default=None,
    help="Override auto-detected task (e.g., image-classification, text-classification)",
)
@click.option(
    "--model-class",
    "model_class",
    default=None,
    help="Override auto-detected model class (e.g., CLIPTextModelWithProjection)",
)
@click.option(
    "--model-type",
    "model_type",
    default=None,
    help="Override auto-detected model type (e.g., bert, resnet). "
    "Can be used without -m to generate config from default HF settings. "
    "When used without --task, the first supported task is auto-selected.",
)
@click.option(
    "--module",
    default=None,
    help="Generate configs for submodules matching this class name (e.g., ResNetConvLayer)",
)
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True),
    default=None,
    help="JSON config file with overrides (WinMLBuildConfig format)",
)
@click.option(
    "--shape-config",
    "shape_config_file",
    type=click.Path(exists=True),
    default=None,
    help="JSON file with shape overrides passed to dummy input generation. "
    "Valid keys — text: sequence_length; "
    "vision: height, width, num_channels; "
    "audio: feature_size, nb_max_frames, audio_sequence_length.",
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.Choice(["auto", "npu", "gpu", "cpu"], case_sensitive=False),
    default="auto",
    help="Target device (affects quant/compile config). Default: auto (no changes to config).",
)
@click.option(
    "--ep",
    "ep",
    type=str,
    default=None,
    help="Force specific execution provider "
    "(qnn, dml, migraphx, nv_tensorrt_rtx, vitisai, openvino, cpu). "
    "Overrides device-to-provider mapping. "
    "When used without --device, device is inferred from EP.",
)
@click.option(
    "-p",
    "--precision",
    "precision",
    type=click.Choice(
        ["auto", "fp32", "fp16", "int8", "int16", "w4a16", "w8a8", "w8a16"],
        case_sensitive=False,
    ),
    default="auto",
    help="Precision: auto, fp32, fp16, int8, int16, or w{x}a{y} (e.g., w8a16). "
    "Default: auto (based on device when device is specified).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output JSON file path (default: stdout)",
)
@click.option(
    "--library",
    "library_name",
    default="transformers",
    help="Source library for TasksManager (default: transformers)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
@click.option(
    "--no-quant",
    is_flag=True,
    default=False,
    help="Exclude quantization from generated config (sets quant=None)",
)
@click.option(
    "--no-compile/--compile",
    "no_compile",
    default=True,
    help="Exclude compilation from generated config (sets compile=None). Default: exclude.",
)
@cli_utils.trust_remote_code_option()
def config(
    hf_model: str | None,
    task: str | None,
    model_class: str | None,
    model_type: str | None,
    module: str | None,
    config_file: str | None,
    shape_config_file: str | None,
    device: str,
    ep: str | None,
    precision: str,
    output: str | None,
    library_name: str,
    verbose: bool,
    no_quant: bool,
    no_compile: bool,
    trust_remote_code: bool,
) -> None:
    r"""Generate WinMLBuildConfig for a HuggingFace model or .onnx file.

    This command auto-detects the task, model class, and I/O specifications
    from a HuggingFace model and generates a complete build configuration.
    When -m points to an existing .onnx file, generates a config with
    export=None for the ONNX build path.

    Requires at least one of -m/--model, --model-type, or --model-class.

    \b
    Examples:
        # Basic usage - auto-detect everything
        winml config -m microsoft/resnet-50

        # Override task
        winml config -m bert-base-uncased --task text-classification

        # Target NPU with w8a8 quantization (NPU supports auto/fp16/w8a8/w8a16)
        winml config -m microsoft/resnet-50 --device npu --precision w8a8

        # Target GPU with fp16 (no quantization)
        winml config -m bert-base-uncased --device gpu --precision fp16

        # Model type only (uses default HF config, auto-detects task)
        winml config --model-type bert

        # Model type + task
        winml config --model-type bert --task fill-mask

        # Override with JSON config file
        winml config -m bert-base-uncased -c overrides.json

        # Vision model with shape overrides ({"height": 224, "width": 224})
        winml config --model-type resnet -t image-classification --shape-config shapes.json

        # Save to file
        winml config -m bert-base-uncased -o config.json

        # Generate configs for submodules
        winml config -m microsoft/resnet-50 --module ResNetConvLayer
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate: at least one of -m, --model-type, or --model-class is required
    if hf_model is None and model_type is None and model_class is None:
        # Show header even for errors
        print_command_header(console, "\U0001f4cb CONFIG GENERATION")
        print_error(
            console,
            "Missing required input",
            hint="Provide one of: -m/--model, --model-type, or --model-class",
        )
        console.print()
        raise click.UsageError(
            "At least one of -m/--model, --model-type, or --model-class is required."
        )

    try:
        from ..config import (
            WinMLBuildConfig,
            generate_hf_build_config,
            generate_onnx_build_config,
        )

        # Load override config from JSON file if provided
        override = None
        _override_file: str | None = None
        _shape_config_file: str | None = None
        if config_file:
            config_path = Path(config_file)
            try:
                content = config_path.read_text()
                if not content.strip():
                    raise click.UsageError(f"Config file is empty: {config_path}")
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise click.UsageError(
                        f"Config file must contain a JSON object, "
                        f"got {type(data).__name__}: {config_path}"
                    )
                override = WinMLBuildConfig.from_dict(data)
            except json.JSONDecodeError as e:
                raise click.UsageError(f"Invalid JSON in config file {config_path}: {e}") from e
            _override_file = config_path.name

        # Load shape_config (shape overrides) from JSON file if provided
        shape_config = None
        if shape_config_file:
            shape_config_path = Path(shape_config_file)
            try:
                content = shape_config_path.read_text()
                if not content.strip():
                    raise click.UsageError(f"I/O config file is empty: {shape_config_path}")
                shape_config = json.loads(content)
                if not isinstance(shape_config, dict):
                    raise click.UsageError(
                        f"I/O config file must contain a JSON object, "
                        f"got {type(shape_config).__name__}: {shape_config_path}"
                    )
            except json.JSONDecodeError as e:
                raise click.UsageError(
                    f"Invalid JSON in I/O config file {shape_config_path}: {e}"
                ) from e
            _shape_config_file = shape_config_path.name

        # ONNX file detection: generate simpler config without loader/export
        if hf_model and _is_onnx_file(hf_model) and module:
            raise click.UsageError(
                "--module is not supported with ONNX file input. "
                "Module discovery requires a HuggingFace model."
            )
        if hf_model and _is_onnx_file(hf_model):
            config_obj = generate_onnx_build_config(
                hf_model,
                task=task,
                device=device,
                precision=precision,
                ep=ep,
                override=override,
            )

            # Apply --no-quant / --no-compile overrides
            _apply_stage_overrides(config_obj, no_quant=no_quant, no_compile=no_compile)

            output_data = config_obj.to_dict()
            _is_onnx_mode = True
            _resolved_task = None
            _resolved_model_class = None
            _export_cfg = None
            configs: list = []  # defensive — ONNX + module is rejected above
            _n_modules = 0
        else:
            _is_onnx_mode = False

            # Check composite model registry: (model_type, task) -> multi-config
            pipeline_components = _resolve_composite_model_components(
                hf_model, model_type, task, trust_remote_code=trust_remote_code
            )
            if pipeline_components:
                # composite model: generate one config per sub-component
                _generate_pipeline_configs(
                    pipeline_components,
                    hf_model=hf_model,
                    model_class=model_class,
                    model_type=model_type,
                    override=override,
                    shape_config=shape_config,
                    library_name=library_name,
                    device=device,
                    precision=precision,
                    trust_remote_code=trust_remote_code,
                    ep=ep,
                    no_quant=no_quant,
                    no_compile=no_compile,
                    output=output,
                    console=console,
                )
                return

            # Generate config(s) - returns single or list based on module parameter
            result = generate_hf_build_config(
                model_id=hf_model,
                task=task,
                model_class=model_class,
                model_type=model_type,
                module=module,
                override=override,
                shape_config=shape_config,
                library_name=library_name,
                device=device,
                precision=precision,
                trust_remote_code=trust_remote_code,
                ep=ep,
            )

            # Handle output format
            if module:
                # Module mode: result is list[WinMLBuildConfig]
                configs = result
                for cfg in configs:
                    _apply_stage_overrides(cfg, no_quant=no_quant, no_compile=no_compile)
                output_data = [cfg.to_dict() for cfg in configs]
                _n_modules = len(configs)
                # Use first config for display metadata
                config_obj = configs[0] if configs else None
            else:
                # Normal mode: result is WinMLBuildConfig
                config_obj = result
                configs = []
                _apply_stage_overrides(config_obj, no_quant=no_quant, no_compile=no_compile)
                output_data = config_obj.to_dict()
                _n_modules = 0

            _resolved_task = config_obj.loader.task if config_obj else None
            _resolved_model_class = config_obj.loader.model_class if config_obj else None
            _export_cfg = config_obj.export if config_obj else None

        # ── Rich console output ──────────────────────────────────────
        subtitle = "ONNX mode" if _is_onnx_mode else ("module mode" if module else None)
        print_command_header(console, "\U0001f4cb CONFIG GENERATION", subtitle)

        # Model identity
        model_label = hf_model or model_type or model_class or "?"
        print_kv(console, "Model:", model_label, icon="\U0001f4e6")

        if _is_onnx_mode:
            print_kv(console, "Mode:", "Direct ONNX", note="export=None", icon="\U0001f527")
        else:
            # Fix #1: Model class before Task
            if module:
                print_kv(console, "Module:", module, icon="\U0001f9e9")
            elif _resolved_model_class:
                mc_note = None if model_class else "auto-detected"
                print_kv(
                    console,
                    "Model class:",
                    _resolved_model_class,
                    note=mc_note,
                    icon="\U0001f9e9",
                )
            # Fix #2: no trailing space after 🏷️
            if _resolved_task:
                task_note = None if task else "auto-detected"
                print_kv(
                    console,
                    "Task:",
                    _resolved_task,
                    note=task_note,
                    icon="\U0001f3f7\ufe0f",
                )

        # Override files
        if config_file:
            console.print(
                f"   \U0001f4c1 [bold]Overrides:[/bold]    {_override_file}  [green]\u2713[/green]"
            )
        if shape_config_file:
            console.print(
                f"   \U0001f4c1 [bold]Shape config:[/bold] "
                f"{_shape_config_file}  [green]\u2713[/green]"
            )

        console.print()

        # I/O specs (always full detail)
        if _is_onnx_mode:
            print_io_specs_na(console)
        elif _export_cfg is not None:
            print_io_specs_detail(console, _export_cfg)

        console.print()

        # Resolution — read directly from the config object.
        # No inference or reverse mapping — display what the config contains.
        _ref_config = config_obj if not module else (configs[0] if configs else None)
        if _ref_config is not None:
            _quant = _ref_config.quant

            console.print("   \u2699\ufe0f  [bold]Resolution:[/bold]")

            # Fix #4: Device from resolve_device (existing API)
            from ..sysinfo import resolve_device as _rd

            _resolved_dev, _ = _rd(device, ep=ep)
            console.print(f"      Device:     [cyan]{_resolved_dev.upper()}[/cyan]")

            # EP — only shown when user explicitly passed --ep
            if ep:
                from ..utils.constants import normalize_ep_name

                _ep_full = normalize_ep_name(ep) or ep
                console.print(f"      EP:         [cyan]{_ep_full}[/cyan]")

            # Quant types — display exactly what config contains
            if _quant:
                console.print(
                    f"      Quant:      "
                    f"[cyan]{_quant.weight_type}/{_quant.activation_type}"
                    f"[/cyan]  [dim](weight/activation)[/dim]"
                )
            else:
                console.print("      Quant:      [dim]none[/dim]")

        # Module mode: show submodule list
        if module and not _is_onnx_mode and _n_modules > 0:
            console.print()
            console.print(
                f"   \U0001f9e9 [bold]Submodules:[/bold] "
                f"[green]{_n_modules}[/green] matching '{module}'"
            )

        console.print()

        # ── Serialize and output ─────────────────────────────────────
        config_json = json.dumps(output_data, indent=2)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(config_json)
            suffix = f"  [dim]({_n_modules} submodules)[/dim]" if _n_modules else ""
            print_success(console, f"Config saved to: [bold]{output}[/bold]{suffix}")
        else:
            print_success(console, "Config written to stdout")
            # Print to stdout (not stderr where console prints)
            print(config_json)

        console.print()

    except click.UsageError:
        raise  # Let click handle its own errors
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    except Exception as e:
        if verbose:
            logger.exception("Unexpected error during config generation")
        raise click.ClickException(f"Unexpected error: {e}") from e


def _resolve_composite_model_components(
    hf_model: str | None,
    model_type: str | None,
    task: str | None,
    trust_remote_code: bool = False,
) -> dict[str, str] | None:
    """Check if (model_type, task) is a registered composite model.

    Returns _SUB_MODEL_CONFIG dict if found, None otherwise.
    """
    if task is None:
        return None

    import winml.modelkit.models.hf  # noqa: F401  # trigger pipeline registrations

    from ..models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

    # Resolve model_type from HF config if not provided
    resolved_type = model_type
    if resolved_type is None and hf_model is not None:
        from transformers import AutoConfig

        resolved_type = AutoConfig.from_pretrained(
            hf_model, trust_remote_code=trust_remote_code
        ).model_type

    if resolved_type is None:
        return None

    cls = COMPOSITE_MODEL_REGISTRY.get((resolved_type, task))
    return cls._SUB_MODEL_CONFIG if cls is not None else None


def _generate_pipeline_configs(
    components: dict[str, str],
    *,
    hf_model: str | None,
    model_class: str | None,
    model_type: str | None,
    override: Any,
    shape_config: dict | None,
    library_name: str,
    device: str,
    precision: str,
    trust_remote_code: bool,
    ep: str | None,
    no_quant: bool,
    no_compile: bool,
    output: str | None,
    console: Any,
) -> None:
    """Generate and save one config file per pipeline sub-component."""
    from ..config import generate_hf_build_config

    for component_name, component_task in components.items():
        console.print(
            f"[dim]Generating config for component '{component_name}' "
            f"(task={component_task})...[/dim]"
        )

        cfg = generate_hf_build_config(
            model_id=hf_model,
            task=component_task,
            model_class=model_class,
            model_type=model_type,
            override=override,
            shape_config=shape_config,
            library_name=library_name,
            device=device,
            precision=precision,
            trust_remote_code=trust_remote_code,
            ep=ep,
        )
        _apply_stage_overrides(cfg, no_quant=no_quant, no_compile=no_compile)

        config_json = json.dumps(cfg.to_dict(), indent=2)

        if output:
            out_path = Path(output)
            suffixed = out_path.with_stem(f"{out_path.stem}_{component_name}")
            suffixed.parent.mkdir(parents=True, exist_ok=True)
            tmp = suffixed.with_suffix(".json.tmp")
            tmp.write_text(config_json)
            tmp.replace(suffixed)
            console.print(f"[green]Config saved to:[/green] {suffixed}")
        else:
            console.print(f"[bold]--- {component_name} ({component_task}) ---[/bold]")
            print(config_json)
