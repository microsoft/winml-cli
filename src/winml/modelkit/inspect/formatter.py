# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Output formatting for inspect command.

Provides table and JSON output formatters using Rich library.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .types import HierarchyInfo, InspectResult, ModuleInfo, SupportLevel


# Status icons
STATUS_ICONS = {
    SupportLevel.SUPPORTED: "[green]+[/green]",
    SupportLevel.DEFAULT: "[yellow]-[/yellow]",
    SupportLevel.GENERIC: "[blue]*[/blue]",
    SupportLevel.UNSUPPORTED: "[red]x[/red]",
}

STATUS_LABELS = {
    SupportLevel.SUPPORTED: "[green]Supported[/green]",
    SupportLevel.DEFAULT: "[yellow]Default[/yellow]",
    SupportLevel.GENERIC: "[blue]Generic[/blue]",
    SupportLevel.UNSUPPORTED: "[red]Unsupported[/red]",
}


def _format_params(params: int) -> str:
    """Format parameter count as human-readable string."""
    if params >= 1_000_000_000:
        return f"{params / 1_000_000_000:.1f}B"
    if params >= 1_000_000:
        return f"{params / 1_000_000:.1f}M"
    if params >= 1_000:
        return f"{params / 1_000:.1f}K"
    return str(params)


def _output_processor_table(console: Console, result: InspectResult) -> None:
    """Output processor information as table.

    Args:
        console: Rich console instance
        result: InspectResult with processor info
    """
    if not result.processor:
        return

    processor = result.processor

    processor_table = Table(show_header=False, box=None, padding=(0, 2))
    processor_table.add_column("Field", style="cyan")
    processor_table.add_column("Value")

    def _src_tag(source: str | None) -> str:
        return f" [dim](via {source})[/dim]" if source else ""

    if processor.processor_class:
        src = _src_tag(processor.processor_source)
        processor_table.add_row("Processor", f"{processor.processor_class}{src}")
    if processor.tokenizer_class:
        src = _src_tag(processor.tokenizer_source)
        processor_table.add_row("Tokenizer", f"{processor.tokenizer_class}{src}")
    if processor.image_processor_class:
        src = _src_tag(processor.image_processor_source)
        processor_table.add_row("Image Processor", f"{processor.image_processor_class}{src}")
    if processor.feature_extractor_class:
        src = _src_tag(processor.feature_extractor_source)
        processor_table.add_row("Feature Extractor", f"{processor.feature_extractor_class}{src}")

    # Only show panel if we have at least one processor class
    if any(
        [
            processor.processor_class,
            processor.tokenizer_class,
            processor.image_processor_class,
            processor.feature_extractor_class,
        ]
    ):
        console.print(Panel(processor_table, title="Data Processing", border_style="dim"))


def _output_io_config_table(console: Console, result: InspectResult) -> None:
    """Output IO configuration as table.

    Args:
        console: Rich console instance
        result: InspectResult with io_config info
    """
    if not result.io_config:
        return

    io_config = result.io_config

    io_table = Table(show_header=False, box=None, padding=(0, 2))
    io_table.add_column("Field", style="cyan")
    io_table.add_column("Value")

    has_content = False

    # Text-related config
    if io_config.max_position_embeddings is not None:
        io_table.add_row("Max Sequence Length", str(io_config.max_position_embeddings))
        has_content = True
    if io_config.vocab_size is not None:
        io_table.add_row("Vocab Size", f"{io_config.vocab_size:,}")
        has_content = True

    # Vision-related config
    if io_config.image_size is not None:
        if isinstance(io_config.image_size, tuple):
            img_size_str = f"{io_config.image_size[0]} x {io_config.image_size[1]}"
        else:
            img_size_str = f"{io_config.image_size} x {io_config.image_size}"
        io_table.add_row("Image Size", img_size_str)
        has_content = True
    if io_config.patch_size is not None:
        io_table.add_row("Patch Size", str(io_config.patch_size))
        has_content = True
    if io_config.num_channels is not None:
        io_table.add_row("Channels", str(io_config.num_channels))
        has_content = True

    # Audio-related config
    if io_config.sampling_rate is not None:
        io_table.add_row("Sampling Rate", f"{io_config.sampling_rate:,} Hz")
        has_content = True

    # General config
    if io_config.hidden_size is not None:
        io_table.add_row("Hidden Size", str(io_config.hidden_size))
        has_content = True
    if io_config.hidden_sizes is not None:
        sizes_str = " → ".join(str(s) for s in io_config.hidden_sizes)
        io_table.add_row("Hidden Sizes", sizes_str)
        has_content = True

    # Extra attrs discovered dynamically from OnnxConfig
    if io_config.extra:
        for key, val in sorted(io_config.extra.items()):
            label = key.replace("_", " ").title()
            io_table.add_row(label, str(val))
            has_content = True

    # Only show panel if we have content
    if has_content:
        console.print(Panel(io_table, title="IO Configuration", border_style="dim"))


def _output_cache_table(console: Console, result: InspectResult) -> None:
    """Output cache status as table.

    Args:
        console: Rich console instance
        result: InspectResult with cache info
    """
    if not result.cache:
        return

    cache = result.cache

    cache_table = Table(show_header=False, box=None, padding=(0, 2))
    cache_table.add_column("Field", style="cyan")
    cache_table.add_column("Value")

    # Summary row
    if cache.total_cached > 0:
        status = f"[green]{cache.total_cached}/{len(cache.stages)} stages cached[/green]"
        cache_table.add_row("Status", status)
        cache_table.add_row("Total Size", f"{cache.total_size_mb} MB")
    else:
        cache_table.add_row("Status", "[dim]No cached artifacts[/dim]")

    # Individual stages
    cache_table.add_row("", "")  # Spacer
    cache_table.add_row("[bold]Pipeline Stages[/bold]", "")

    for stage in cache.stages:
        if stage.cached:
            icon = "[green]+[/green]"
            info = f"{stage.size_mb} MB"
        else:
            icon = "[dim]-[/dim]"
            info = "[dim]not cached[/dim]"
        cache_table.add_row(f"  {icon} {stage.stage}", info)

    console.print(Panel(cache_table, title="Cache Status", border_style="dim"))


def _output_composite_panel(console: Console, result: InspectResult) -> None:
    """Render the Composite Pipeline panel (Variant 1) for composite model_types.

    Explains that the Loader / Exporter panels below describe a single exported
    component, lists the component -> export-task breakdown, and shows which
    pipeline tasks the model_type serves plus how to build one.
    """
    composite = result.composite
    if composite is None:  # defensive; callers guard on result.composite
        return

    intro = Text.from_markup(
        f"Composite model — the Loader / Exporter panels below describe a single "
        f"exported component ([cyan]{result.task}[/cyan]). The runnable model is a "
        f"multi-component pipeline:"
    )

    comp_table = Table(show_header=True, box=None, padding=(0, 2), header_style="bold")
    comp_table.add_column("Component", style="cyan")
    comp_table.add_column("Export Task")
    for name, task in composite.components.items():
        comp_table.add_row(name, task)

    pipelines = " · ".join(composite.pipeline_tasks)
    footer = Text.from_markup(
        f"[bold]Serves pipelines:[/bold] {pipelines}\n"
        f"[bold]Build one:[/bold] [cyan]winml config -m <model> --task <pipeline>[/cyan]"
    )

    console.print(
        Panel(
            Group(intro, "", comp_table, "", footer),
            title="Composite Pipeline",
            border_style="dim",
        )
    )


def output_table(console: Console, result: InspectResult, verbose: bool = False) -> None:
    """Output result as rich table.

    Args:
        console: Rich console instance
        result: InspectResult to format
        verbose: If True, show full build config
    """
    # Header panel
    console.print(
        Panel(
            f"[bold]{result.model_id}[/bold]",
            title="Model",
            border_style="blue",
        )
    )

    # Model Information table
    model_table = Table(show_header=False, box=None, padding=(0, 2))
    model_table.add_column("Field", style="cyan")
    model_table.add_column("Value")

    model_table.add_row("Model Type", result.model_type)
    # Guard on pipeline_tasks too: output_table is public, so a directly-constructed
    # InspectResult with an empty-pipeline_tasks CompositeInfo must not render a
    # bare " [composite]" Task row — fall back to the plain task in that case.
    if result.composite and result.composite.pipeline_tasks:
        # Variant 1 (Pipeline-led): the Task row surfaces the pipeline tasks the
        # model_type serves, and an Export row shows the granular component tasks.
        pipelines = " · ".join(result.composite.pipeline_tasks)
        model_table.add_row("Task", f"{pipelines} [magenta]\\[composite][/magenta]")
        components = ", ".join(
            f"{name}: {task}" for name, task in result.composite.components.items()
        )
        if components:
            model_table.add_row("Export", f"{components} [dim](via {result.task_source})[/dim]")
    else:
        model_table.add_row("Task", f"{result.task} [dim](via {result.task_source})[/dim]")
    archs = ", ".join(result.architectures) if result.architectures else "-"
    model_table.add_row("Architectures", archs)
    model_table.add_row("Overall Support", STATUS_LABELS[result.overall_support])

    console.print(Panel(model_table, title="Model Information", border_style="dim"))

    # Composite Pipeline panel (only for composite model_types with pipeline tasks)
    if result.composite and result.composite.pipeline_tasks:
        _output_composite_panel(console, result)

    # Loader Configuration table
    loader_table = Table(show_header=False, box=None, padding=(0, 2))
    loader_table.add_column("Field", style="cyan")
    loader_table.add_column("Value")

    loader_table.add_row("HF Model Class", result.loader.hf_model_class)
    loader_table.add_row("Source", result.loader.hf_model_class_source)
    loader_table.add_row("Status", STATUS_LABELS[result.loader.support_level])

    console.print(Panel(loader_table, title="Loader Configuration", border_style="dim"))

    # Exporter Configuration table
    exporter_table = Table(show_header=False, box=None, padding=(0, 2))
    exporter_table.add_column("Field", style="cyan")
    exporter_table.add_column("Value")

    exporter_table.add_row("ONNX Config", result.exporter.onnx_config_class or "-")
    exporter_table.add_row("Source", result.exporter.onnx_config_source)
    exporter_table.add_row("Status", STATUS_LABELS[result.exporter.support_level])
    exporter_table.add_row("OPSET Version", str(result.exporter.opset_version))

    # Input tensors
    if result.exporter.input_tensors:
        exporter_table.add_row("", "")  # Spacer
        exporter_table.add_row("[bold]Input Tensors[/bold]", "")
        for tensor in result.exporter.input_tensors:
            # Prefer shape_desc (dynamic), fall back to shape (concrete)
            if tensor.shape_desc:
                shape_str = tensor.shape_desc
            elif tensor.shape:
                shape_str = str(list(tensor.shape))
            else:
                shape_str = "-"
            dtype_str = tensor.dtype or "-"
            extra = ""
            if tensor.value_range is not None:
                extra = f"  [dim]range {tensor.value_range}[/dim]"
            exporter_table.add_row(f"  {tensor.name}", f"{dtype_str}  {shape_str}{extra}")

    # Output tensors
    if result.exporter.output_tensors:
        exporter_table.add_row("", "")  # Spacer
        exporter_table.add_row("[bold]Output Tensors[/bold]", "")
        for tensor in result.exporter.output_tensors:
            if tensor.shape_desc:
                shape_str = tensor.shape_desc
            elif tensor.shape:
                shape_str = str(list(tensor.shape))
            else:
                shape_str = "-"
            exporter_table.add_row(f"  {tensor.name}", shape_str)

    console.print(Panel(exporter_table, title="Exporter Configuration", border_style="dim"))

    # WinML Inference Class table
    winml_table = Table(show_header=False, box=None, padding=(0, 2))
    winml_table.add_column("Field", style="cyan")
    winml_table.add_column("Value")

    winml_table.add_row("Class", result.winml.winml_class)
    winml_table.add_row("Source", result.winml.winml_class_source)
    winml_table.add_row("Status", STATUS_LABELS[result.winml.support_level])

    console.print(Panel(winml_table, title="WinML Inference Class", border_style="dim"))

    # Processor Information table
    if result.processor:
        _output_processor_table(console, result)

    # IO Configuration table
    if result.io_config:
        _output_io_config_table(console, result)

    # Cache Status table
    if result.cache:
        _output_cache_table(console, result)

    # Hierarchy (if present)
    if result.hierarchy:
        _output_hierarchy_table(console, result)

    # Verbose: Full build config
    if verbose and result.build_config:
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column("Section", style="cyan")
        config_table.add_column("Config")

        for section, config in result.build_config.items():
            if config is not None:
                if isinstance(config, dict):
                    config_str = json.dumps(config, indent=2)
                else:
                    config_str = str(config)
                config_table.add_row(section, config_str)

        console.print(Panel(config_table, title="Full Build Configuration", border_style="dim"))

    # Support notes
    if result.support_notes:
        notes_text = "\n".join(f"• {note}" for note in result.support_notes)
        console.print(Panel(notes_text, title="Notes", border_style="yellow"))


def _output_hierarchy_table(console: Console, result: InspectResult) -> None:
    """Output hierarchy as tree view.

    Args:
        console: Rich console instance
        result: InspectResult with hierarchy
    """
    if not result.hierarchy:
        return

    hierarchy = result.hierarchy

    # Summary table
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Field", style="cyan")
    summary_table.add_column("Value")

    summary_table.add_row("Root Class", hierarchy.root_class)
    summary_table.add_row("Total Parameters", _format_params(hierarchy.total_parameters))
    summary_table.add_row("HF Modules", str(hierarchy.hf_module_count))
    summary_table.add_row("NN Modules", f"{hierarchy.nn_module_count} [dim](filtered)[/dim]")

    console.print(Panel(summary_table, title="HF Module Hierarchy", border_style="dim"))

    # Tree view
    if hierarchy.hf_modules:
        tree = Tree(f"[bold]{hierarchy.root_class}[/bold]")
        _build_tree(tree, hierarchy.hf_modules)
        console.print(Panel(tree, title="Module Tree (HF modules only)", border_style="dim"))


def _build_tree(parent: Tree, modules: list[ModuleInfo]) -> None:
    """Recursively build Rich tree from module hierarchy.

    Args:
        parent: Parent tree node
        modules: List of ModuleInfo to add
    """
    for module in modules:
        # Format: name (ClassName)
        label = f"{module.name}  [dim]{module.class_name}[/dim]"
        if module.num_parameters > 0:
            label += f"  [cyan]{_format_params(module.num_parameters)}[/cyan]"

        branch = parent.add(label)

        if module.children:
            _build_tree(branch, module.children)


def output_json(result: InspectResult, verbose: bool = False) -> str:
    """Output result as JSON string.

    Args:
        result: InspectResult to format
        verbose: If True, include full build config

    Returns:
        JSON string
    """
    data: dict[str, Any] = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "architectures": result.architectures,
        "task": result.task,
        "task_source": result.task_source,
        # task stays the granular machine task (e.g. text2text-generation); the
        # composite/pipeline view is additive so machine consumers are unaffected.
        # pipeline_tasks is a flat convenience alias of composite.pipeline_tasks.
        "pipeline_tasks": (result.composite.pipeline_tasks if result.composite else None),
        "composite": (
            {
                "pipeline_tasks": result.composite.pipeline_tasks,
                "components": result.composite.components,
            }
            if result.composite
            else None
        ),
        "overall_support": result.overall_support.value,
        "support_notes": result.support_notes,
        "loader": {
            "hf_model_class": result.loader.hf_model_class,
            "hf_model_class_source": result.loader.hf_model_class_source,
            "support_level": result.loader.support_level.value,
        },
        "exporter": {
            "onnx_config_class": result.exporter.onnx_config_class,
            "onnx_config_source": result.exporter.onnx_config_source,
            "support_level": result.exporter.support_level.value,
            "opset_version": result.exporter.opset_version,
            "input_tensors": [
                {
                    "name": t.name,
                    "dtype": t.dtype,
                    "shape": list(t.shape) if t.shape else None,
                    "shape_desc": t.shape_desc,
                    "dynamic_axes": t.dynamic_axes,
                    "value_range": list(t.value_range) if t.value_range else None,
                }
                for t in result.exporter.input_tensors
            ],
            "output_tensors": [
                {
                    "name": t.name,
                    "shape_desc": t.shape_desc,
                    "dynamic_axes": t.dynamic_axes,
                }
                for t in result.exporter.output_tensors
            ],
        },
        "winml": {
            "winml_class": result.winml.winml_class,
            "winml_class_source": result.winml.winml_class_source,
            "support_level": result.winml.support_level.value,
        },
    }

    # Add cache info
    if result.cache:
        data["cache"] = {
            "cache_dir": result.cache.cache_dir,
            "total_cached": result.cache.total_cached,
            "total_size_mb": result.cache.total_size_mb,
            "stages": [
                {
                    "stage": s.stage,
                    "cached": s.cached,
                    "path": s.path,
                    "size_mb": s.size_mb,
                    "created": s.created,
                }
                for s in result.cache.stages
            ],
        }
    else:
        data["cache"] = None

    # Add hierarchy if present
    if result.hierarchy:
        data["hierarchy"] = _hierarchy_to_dict(result.hierarchy)
    else:
        data["hierarchy"] = None

    # Add processor info
    if result.processor:
        data["processor"] = {
            "processor_class": result.processor.processor_class,
            "tokenizer_class": result.processor.tokenizer_class,
            "image_processor_class": result.processor.image_processor_class,
            "feature_extractor_class": result.processor.feature_extractor_class,
            "processor_source": result.processor.processor_source,
            "tokenizer_source": result.processor.tokenizer_source,
            "image_processor_source": result.processor.image_processor_source,
            "feature_extractor_source": result.processor.feature_extractor_source,
        }
    else:
        data["processor"] = None

    # Add IO config
    if result.io_config:
        io_config = result.io_config
        data["io_config"] = {
            "max_position_embeddings": io_config.max_position_embeddings,
            "vocab_size": io_config.vocab_size,
            "image_size": (
                list(io_config.image_size)
                if isinstance(io_config.image_size, tuple)
                else io_config.image_size
            ),
            "patch_size": io_config.patch_size,
            "num_channels": io_config.num_channels,
            "sampling_rate": io_config.sampling_rate,
            "hidden_size": io_config.hidden_size,
            "hidden_sizes": io_config.hidden_sizes,
            "extra": io_config.extra,
        }
    else:
        data["io_config"] = None

    # Add build config if verbose
    if verbose and result.build_config:
        data["build_config"] = result.build_config

    return json.dumps(data, indent=2)


def _hierarchy_to_dict(hierarchy: HierarchyInfo) -> dict[str, Any]:
    """Convert HierarchyInfo to dictionary.

    Args:
        hierarchy: HierarchyInfo instance

    Returns:
        Dictionary representation
    """

    def module_to_dict(module: ModuleInfo) -> dict[str, Any]:
        return {
            "name": module.name,
            "class_name": module.class_name,
            "module_path": module.module_path,
            "depth": module.depth,
            "num_parameters": module.num_parameters,
            "children": [module_to_dict(c) for c in module.children],
        }

    return {
        "root_class": hierarchy.root_class,
        "total_parameters": hierarchy.total_parameters,
        "hf_module_count": hierarchy.hf_module_count,
        "nn_module_count": hierarchy.nn_module_count,
        "hf_modules": [module_to_dict(m) for m in hierarchy.hf_modules],
    }
