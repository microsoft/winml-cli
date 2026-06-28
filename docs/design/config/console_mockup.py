# ruff: noqa
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mock: Proposed wmk config console output.

Run:  uv run python temp/mock_config_output.py [--onnx] [--verbose] [--module] [--error]

Demonstrates the redesigned config command output with:
- Command header with model identity
- Auto-detected values labeled
- Resolution summary (device, EP, precision)
- Verbose resolution chain (-v)
- Module mode output
- Error output with actionable hints
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table


console = Console(stderr=True)

# ── Shared styling constants (would live in modelkit/utils/console.py) ────

HEAVY_SEP = "═" * 60
LIGHT_SEP = "─" * 60


def print_command_header(
    title: str,
    subtitle: str | None = None,
) -> None:
    """Print a command header block matching analyze style."""
    console.print()
    console.print(HEAVY_SEP)
    label = f"[bold]{title}[/bold]"
    if subtitle:
        label += f"  [dim]({subtitle})[/dim]"
    console.print(label)
    console.print(HEAVY_SEP)


def print_kv(
    label: str,
    value: str,
    note: str | None = None,
    icon: str = "",
) -> None:
    """Print a key-value line with optional note."""
    line = f"   {icon} [bold]{label:<14}[/bold] [cyan]{value}[/cyan]"
    if note:
        line += f"  [dim]({note})[/dim]"
    console.print(line)


def print_success(message: str) -> None:
    console.print(f"   [green]✅ {message}[/green]")


def print_error(message: str, hint: str | None = None) -> None:
    console.print(f"   [red]❌ {message}[/red]")
    if hint:
        console.print(f"   [dim]💡 {hint}[/dim]")


# ── Scenario: Normal HF model ────────────────────────────────────────────


def print_io_specs(
    inputs: list[tuple[str, str, str]],
    output_names: list[str],
) -> None:
    """Print resolved I/O specs.

    Args:
        inputs: list of (name, shape_str, dtype) for input tensors
        output_names: list of output tensor names (no shape/dtype available)
    """
    for i, (name, shape, dtype) in enumerate(inputs):
        label = "Input:        " if i == 0 else "              "
        console.print(f"   {label}[cyan]{name:<18}[/cyan] {shape:<14} [dim]{dtype}[/dim]")
    # Fix #3: Output tensors have name only (OutputTensorSpec lacks shape/dtype)
    for i, name in enumerate(output_names):
        label = "Output:       " if i == 0 else "              "
        console.print(f"   {label}[cyan]{name}[/cyan]")


# Example I/O data for demos
_BERT_INPUTS = [
    ("input_ids", "[1, 128]", "int64"),
    ("attention_mask", "[1, 128]", "int64"),
    ("token_type_ids", "[1, 128]", "int64"),
]
_BERT_OUTPUTS = ["logits"]

_RESNET_INPUTS = [
    ("pixel_values", "[1, 3, 224, 224]", "float32"),
]
_RESNET_OUTPUTS = ["logits"]


def demo_normal(verbose: bool = False) -> None:
    """Simulate: wmk config -m bert-base-uncased."""
    print_command_header("📋 CONFIG GENERATION")

    # Fix #1: Model class before Task. Fix #2: no trailing space on 🏷️
    print_kv("Model:", "bert-base-uncased", icon="📦")
    print_kv("Model class:", "BertForMaskedLM", note="auto-detected", icon="🧩")
    print_kv("Task:", "fill-mask", note="auto-detected", icon="🏷️")

    console.print()

    # Fix #3: Output name only (no shape/dtype)
    print_io_specs(_BERT_INPUTS, _BERT_OUTPUTS)

    console.print()

    console.print("   ⚙️  [bold]Resolution:[/bold]")
    console.print("      Device:     [cyan]NPU[/cyan]")
    console.print("      Quant:      [cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")

    console.print()
    print_success("Config saved to: [bold]output/config.json[/bold]")
    console.print()


# ── Scenario: ONNX file input ────────────────────────────────────────────


def demo_onnx() -> None:
    """Simulate: wmk config -m model.onnx."""
    print_command_header("📋 CONFIG GENERATION", subtitle="ONNX mode")

    print_kv("Model:", "model.onnx", icon="📦")
    print_kv("Mode:", "Direct ONNX", note="export=None", icon="🔧")

    console.print()
    console.print(
        "   📐 [bold]I/O specs:[/bold]    [dim]N/A — inferred from ONNX graph at build time[/dim]"
    )

    console.print()
    console.print("   ⚙️  [bold]Resolution:[/bold]")
    console.print("      Device:     [cyan]NPU[/cyan]")
    console.print("      Quant:      [cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")

    console.print()
    print_success("Config saved to: [bold]output/config.json[/bold]")
    console.print()


# ── Scenario: Module mode ────────────────────────────────────────────────


def demo_module() -> None:
    """Simulate: wmk config -m microsoft/resnet-50 --module ResNetConvLayer."""
    print_command_header("📋 CONFIG GENERATION", subtitle="module mode")

    print_kv("Model:", "microsoft/resnet-50", icon="📦")
    print_kv("Module:", "ResNetConvLayer", icon="🧩")
    print_kv("Task:", "image-classification", note="auto-detected", icon="🏷️")

    console.print()
    print_io_specs(_RESNET_INPUTS, _RESNET_OUTPUTS)

    console.print()
    console.print("   ⚙️  [bold]Resolution:[/bold]")
    console.print("      Device:     [cyan]NPU[/cyan]")
    console.print("      Quant:      [cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")

    console.print()

    # Module discovery results
    console.print(
        "   🧩 [bold]Submodules found:[/bold] [green]3[/green] matching 'ResNetConvLayer'"
    )
    console.print()

    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", width=4, justify="right")
    table.add_column("Module path", width=30)
    table.add_column("Class", width=20)

    table.add_row("[dim]1[/dim]", "encoder.stages.0.layers.0.conv", "ResNetConvLayer")
    table.add_row("[dim]2[/dim]", "encoder.stages.1.layers.0.conv", "ResNetConvLayer")
    table.add_row("[dim]3[/dim]", "encoder.stages.2.layers.0.conv", "ResNetConvLayer")

    console.print(table)

    console.print()
    print_success("Config saved to: [bold]output/config.json[/bold]  [dim](3 submodules)[/dim]")
    console.print()


# ── Scenario: Override files ─────────────────────────────────────────────


def demo_overrides() -> None:
    """Simulate: wmk config -m bert-base-uncased -c overrides.json --shape-config shapes.json."""
    print_command_header("📋 CONFIG GENERATION")

    print_kv("Model:", "bert-base-uncased", icon="📦")
    print_kv("Model class:", "BertForMaskedLM", note="auto-detected", icon="🧩")
    print_kv("Task:", "fill-mask", note="auto-detected", icon="🏷️")

    console.print()

    # Override files
    console.print("   📁 [bold]Overrides:[/bold]    overrides.json  [green]✓[/green]")
    console.print("   📁 [bold]Shape config:[/bold] shapes.json  [green]✓[/green]")

    console.print()
    print_io_specs(_BERT_INPUTS, _BERT_OUTPUTS)

    console.print()
    console.print("   ⚙️  [bold]Resolution:[/bold]")
    console.print("      Device:     [cyan]NPU[/cyan]")
    console.print("      Quant:      [cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")

    console.print()
    print_success("Config written to stdout")
    console.print()


# ── Scenario: Error — resolution failure ─────────────────────────────────


def demo_error() -> None:
    """Simulate: wmk config -m unknown-model --task custom-task."""
    print_command_header("📋 CONFIG GENERATION")

    print_kv("Model:", "unknown-model", icon="📦")
    print_kv("Task:", "custom-task", note="user-provided", icon="🏷️")

    console.print()
    print_error(
        "I/O spec resolution failed:",
        hint="Try: --model-class MyModelClass  or  --shape-config shapes.json",
    )
    console.print(
        "      [dim]Could not find OnnxConfig for model_type='unknown', task='custom-task'[/dim]"
    )
    console.print()


# ── Scenario: Error — missing input ──────────────────────────────────────


def demo_missing_input() -> None:
    """Simulate: wmk config (no arguments)."""
    print_command_header("📋 CONFIG GENERATION")

    print_error(
        "Missing required input",
        hint="Provide one of: -m/--model, --model-type, or --model-class",
    )
    console.print()


# ── Scenario: Auto device (default, no --device) ────────────────────────


def demo_auto_device() -> None:
    """Simulate: wmk config -m bert-base-uncased (no device/precision flags)."""
    print_command_header("📋 CONFIG GENERATION")

    print_kv("Model:", "bert-base-uncased", icon="📦")
    print_kv("Model class:", "BertForMaskedLM", note="auto-detected", icon="🧩")
    print_kv("Task:", "fill-mask", note="auto-detected", icon="🏷️")

    console.print()
    print_io_specs(_BERT_INPUTS, _BERT_OUTPUTS)

    console.print()
    console.print("   ⚙️  [bold]Resolution:[/bold]")
    console.print("      Device:     [cyan]NPU[/cyan]")
    console.print("      Quant:      [dim]none[/dim]")

    console.print()
    print_success("Config written to stdout")
    console.print()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        console.print("[bold]Usage:[/bold] uv run python temp/mock_config_output.py [OPTIONS]")
        console.print()
        console.print("  [dim](no flags)[/dim]   Normal HF model with NPU/int8")
        console.print("  --auto        Default device (CPU/fp32, no --device)")
        console.print("  --onnx        ONNX file input")
        console.print("  --module      Module mode (submodule discovery)")
        console.print("  --overrides   With override files")
        console.print("  --verbose     Verbose resolution chain")
        console.print("  --error       Resolution failure")
        console.print("  --missing     Missing required input")
        console.print("  --all         Run all scenarios")
        sys.exit(0)

    if "--all" in args:
        scenarios = [
            ("Normal HF model (--device npu --precision int8)", demo_normal),
            ("Default device (CPU/fp32)", demo_auto_device),
            ("ONNX file input", demo_onnx),
            ("Module mode", demo_module),
            ("With override files", demo_overrides),
            ("Verbose resolution chain", lambda: demo_normal(verbose=True)),
            ("Error: resolution failure", demo_error),
            ("Error: missing input", demo_missing_input),
        ]
        for label, fn in scenarios:
            console.print()
            console.print(f"[bold yellow]▶ Scenario: {label}[/bold yellow]")
            fn()
            console.print()
        sys.exit(0)

    if "--onnx" in args:
        demo_onnx()
    elif "--module" in args:
        demo_module()
    elif "--overrides" in args:
        demo_overrides()
    elif "--verbose" in args:
        demo_normal(verbose=True)
    elif "--error" in args:
        demo_error()
    elif "--missing" in args:
        demo_missing_input()
    elif "--auto" in args:
        demo_auto_device()
    else:
        demo_normal()
