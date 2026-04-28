# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""System information command for ModelKit CLI.

Displays detailed information about the system environment, including:
- Python and OS information
- Core ML library versions (torch, transformers, onnx, etc.)
- Hardware capabilities (CPU, GPU, memory)
- Backend SDK availability (QNN, OpenVINO)
- Export readiness assessment
- Available devices and execution providers

Usage:
    winml sys
    winml sys --format json
    winml sys --format compact
    winml sys --verbose
    winml sys --list-device
    winml sys --list-ep
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from ..sysinfo import OS, get_ep_device_map


logger = logging.getLogger(__name__)
console = Console()


def _get_python_info() -> dict[str, Any]:
    """Gather Python environment information."""
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _get_platform_info() -> dict[str, Any]:
    """Gather OS and platform information."""
    system = platform.system()
    release = platform.release()

    # For Windows, use OS class for accurate Windows 11 detection
    # platform.release() may incorrectly report '10' on some Python versions
    if system == "Windows":
        try:
            os_info = OS.get()
            # Only override if it's actually Windows 11
            # Otherwise keep the original platform.release() value
            if os_info.is_windows_11():
                release = "11"
        except Exception:
            # Fallback to platform.release() if OS detection fails
            pass

    return {
        "system": system,
        "release": release,
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
    }


def _get_library_versions() -> dict[str, str | None]:
    """Gather versions of key ML libraries."""
    from importlib.metadata import PackageNotFoundError, version

    libraries: dict[str, str | None] = {}

    # Core libraries
    lib_names = [
        "torch",
        "transformers",
        "onnx",
        "optimum",
        "numpy",
        "click",
        "rich",
    ]

    for lib in lib_names:
        try:
            libraries[lib] = version(lib)
        except PackageNotFoundError:  # noqa: PERF203
            libraries[lib] = None

    # onnxruntime has multiple distribution variants
    ort_variants = [
        "onnxruntime",
        "onnxruntime-windowsml",
        "onnxruntime-gpu",
        "onnxruntime-silicon",
    ]
    libraries["onnxruntime"] = None
    for variant in ort_variants:
        try:
            ver = version(variant)
            # Include variant suffix if not base onnxruntime
            if variant != "onnxruntime":
                suffix = variant.replace("onnxruntime-", "")
                libraries["onnxruntime"] = f"{ver} ({suffix})"
            else:
                libraries["onnxruntime"] = ver
            break
        except PackageNotFoundError:
            continue

    return libraries


def _get_torch_info() -> dict[str, Any]:
    """Gather PyTorch-specific information including CUDA."""
    info: dict[str, Any] = {"available": False}

    try:
        import torch

        info["available"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        logger.debug("PyTorch not available")

    return info


def _check_qnn_sdk() -> dict[str, Any]:
    """Check QNN SDK availability."""
    import os

    info: dict[str, Any] = {"installed": False}

    # Check common environment variables and paths
    qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
    qairt_sdk_root = os.environ.get("QAIRT_SDK_ROOT")

    if qnn_sdk_root:
        info["installed"] = True
        info["path"] = qnn_sdk_root
        info["source"] = "QNN_SDK_ROOT"
    elif qairt_sdk_root:
        info["installed"] = True
        info["path"] = qairt_sdk_root
        info["source"] = "QAIRT_SDK_ROOT"

    # TODO: Check for qnn-convert executable
    # TODO: Parse version from SDK

    return info


def _check_openvino() -> dict[str, Any]:
    """Check OpenVINO availability."""
    info: dict[str, Any] = {"installed": False}

    try:
        import openvino  # type: ignore[import-not-found]

        info["installed"] = True
        info["version"] = openvino.__version__
    except ImportError:
        logger.debug("OpenVINO not available")

    return info


def _gather_system_info(verbose: bool = False) -> dict[str, Any]:
    """Gather all system information.

    Args:
        verbose: Include additional diagnostic information

    Returns:
        Dictionary containing all system information
    """
    info = {
        "python": _get_python_info(),
        "platform": _get_platform_info(),
        "libraries": _get_library_versions(),
        "torch": _get_torch_info(),
        "backends": {
            "qnn": _check_qnn_sdk(),
            "openvino": _check_openvino(),
        },
    }

    # Assess export readiness
    libs = info["libraries"]
    info["export_readiness"] = {
        "onnx_export": all(
            [
                libs.get("torch"),
                libs.get("onnx"),
                libs.get("transformers"),
            ]
        ),
        "qnn_ready": info["backends"]["qnn"]["installed"],
        "openvino_ready": info["backends"]["openvino"]["installed"],
    }

    return info


def _output_text(info: dict[str, Any], verbose: bool = False) -> None:
    """Output system info in human-readable text format."""
    # Title
    console.print(
        Panel.fit(
            "[bold]ModelKit System Information[/bold]",
            border_style="blue",
        )
    )

    # Python & Platform
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Python Version", info["python"]["version"])
    table.add_row("Python Executable", info["python"]["executable"])
    table.add_row("OS", f"{info['platform']['system']} {info['platform']['release']}")
    table.add_row("Machine", info["platform"]["machine"])

    console.print("\n[bold blue]Environment[/bold blue]")
    console.print(table)

    # Libraries
    lib_table = Table(show_header=True, box=None, padding=(0, 2))
    lib_table.add_column("Library", style="bold")
    lib_table.add_column("Version")
    lib_table.add_column("Status")

    for lib, version in info["libraries"].items():
        status = "[green]OK[/green]" if version else "[red]Not installed[/red]"
        lib_table.add_row(lib, version or "-", status)

    console.print("\n[bold blue]ML Libraries[/bold blue]")
    console.print(lib_table)

    # PyTorch details
    torch_info = info["torch"]
    if torch_info["available"]:
        console.print("\n[bold blue]PyTorch Details[/bold blue]")
        torch_table = Table(show_header=False, box=None, padding=(0, 2))
        torch_table.add_column("Key", style="bold")
        torch_table.add_column("Value")

        torch_table.add_row("CUDA Available", str(torch_info["cuda_available"]))
        if torch_info["cuda_available"]:
            torch_table.add_row("CUDA Version", torch_info.get("cuda_version", "N/A"))
            torch_table.add_row("GPU Count", str(torch_info.get("gpu_count", 0)))
            for i, gpu in enumerate(torch_info.get("gpu_devices", [])):
                torch_table.add_row(f"GPU {i}", gpu)

        console.print(torch_table)

    # Backend SDKs
    console.print("\n[bold blue]Backend SDKs[/bold blue]")
    backend_table = Table(show_header=False, box=None, padding=(0, 2))
    backend_table.add_column("Backend", style="bold")
    backend_table.add_column("Status")
    backend_table.add_column("Details")

    qnn = info["backends"]["qnn"]
    qnn_status = "[green]Installed[/green]" if qnn["installed"] else "[yellow]Not found[/yellow]"
    qnn_details = qnn.get("path", "-")[:50] if qnn["installed"] else "-"
    backend_table.add_row("QNN SDK", qnn_status, qnn_details)

    ov = info["backends"]["openvino"]
    ov_status = "[green]Installed[/green]" if ov["installed"] else "[yellow]Not found[/yellow]"
    ov_details = ov.get("version", "-") if ov["installed"] else "-"
    backend_table.add_row("OpenVINO", ov_status, ov_details)

    console.print(backend_table)

    # Export Readiness
    console.print("\n[bold blue]Export Readiness[/bold blue]")
    readiness = info["export_readiness"]
    ready_table = Table(show_header=False, box=None, padding=(0, 2))
    ready_table.add_column("Capability", style="bold")
    ready_table.add_column("Status")

    onnx_ready = "[green]Ready[/green]" if readiness["onnx_export"] else "[red]Not ready[/red]"
    qnn_ready = (
        "[green]Ready[/green]" if readiness["qnn_ready"] else "[yellow]SDK required[/yellow]"
    )
    ov_ready = (
        "[green]Ready[/green]" if readiness["openvino_ready"] else "[yellow]Not installed[/yellow]"
    )

    ready_table.add_row("ONNX Export", onnx_ready)
    ready_table.add_row("QNN Compilation", qnn_ready)
    ready_table.add_row("OpenVINO Conversion", ov_ready)

    console.print(ready_table)


def _output_json(info: dict[str, Any]) -> None:
    """Output system info as JSON."""
    click.echo(json.dumps(info, indent=2))


def _output_compact(info: dict[str, Any]) -> None:
    """Output system info in compact format."""
    py = info["python"]
    plat = info["platform"]
    libs = info["libraries"]
    torch_info = info["torch"]
    readiness = info["export_readiness"]

    lines = [
        f"Python: {py['version']} ({plat['system']})",
        f"torch: {libs.get('torch', 'N/A')} | "
        f"transformers: {libs.get('transformers', 'N/A')} | "
        f"onnx: {libs.get('onnx', 'N/A')}",
    ]

    if torch_info["available"] and torch_info["cuda_available"]:
        lines.append(
            f"CUDA: {torch_info.get('cuda_version', 'N/A')} | "
            f"GPU: {torch_info.get('gpu_count', 0)} device(s)"
        )

    qnn = info["backends"]["qnn"]
    ov = info["backends"]["openvino"]
    lines.append(
        f"QNN: {'OK' if qnn['installed'] else 'N/A'} | "
        f"OpenVINO: {ov.get('version', 'N/A') if ov['installed'] else 'N/A'}"
    )

    onnx_status = "OK" if readiness["onnx_export"] else "FAIL"
    lines.append(f"Export Ready: ONNX {onnx_status}")

    for line in lines:
        click.echo(line)


# --- Device listing ---


def _gather_device_info() -> list[dict[str, Any]]:
    """Gather available device information in priority order.

    Returns:
        List of device dicts with type, priority, and details.
    """
    from ..sysinfo import CPU, GPU, NPU

    result: list[dict[str, Any]] = []
    priority = 1

    # Query hardware directly in NPU > GPU > CPU priority order.
    # This avoids depending on _get_available_devices() and eliminates
    # redundant PowerShell queries (we need the details anyway).
    hw_queries: list[tuple[str, type]] = [
        ("NPU", NPU),
        ("GPU", GPU),
        ("CPU", CPU),
    ]

    for device_label, hw_class in hw_queries:
        try:
            items = hw_class.get_all()
        except Exception as e:
            logger.warning("Failed to get %s details: %s", device_label, e)
            # Only append an error entry if this was expected to have results
            # CPU always exists, NPU/GPU may not
            if device_label == "CPU":
                result.append(
                    {
                        "priority": priority,
                        "type": device_label,
                        "name": "(detection error)",
                        "details": {"error": str(e)},
                    }
                )
                priority += 1
            continue

        for item in items:
            entry: dict[str, Any] = {
                "priority": priority,
                "type": device_label,
                "name": item.name,
                "details": {},
            }
            if device_label in ("NPU", "GPU"):
                entry["details"] = {
                    "driver": item.driver_version,
                    "manufacturer": item.manufacturer,
                }
            elif device_label == "CPU":
                entry["details"] = {
                    "cores": item.core_count,
                    "threads": item.thread_count,
                    "architecture": item.architecture.name,
                }
            result.append(entry)
            priority += 1

    return result


def _output_device_text(devices: list[dict[str, Any]]) -> None:
    """Display device list in rich text format."""
    console.print("\n[bold blue]Available Devices (priority order)[/bold blue]")
    for dev in devices:
        console.print(
            f"  [bold]#{dev['priority']}[/bold]  [cyan]{dev['type']:5s}[/cyan] {dev['name']}"
        )
        details = dev.get("details", {})
        if "error" in details:
            console.print(f"             [red]Error: {details['error']}[/red]")
        elif dev["type"] in ("NPU", "GPU"):
            console.print(
                f"             Driver: {details.get('driver', 'N/A')} | "
                f"Manufacturer: {details.get('manufacturer', 'N/A')}"
            )
        elif dev["type"] == "CPU":
            console.print(
                f"             Cores: {details.get('cores', 'N/A')} | "
                f"Threads: {details.get('threads', 'N/A')} | "
                f"Architecture: {details.get('architecture', 'N/A')}"
            )


# --- EP listing ---


def _gather_ep_info() -> list[dict[str, Any]]:
    """Gather execution provider information.

    Tries WinMLEPRegistry first, falls back to ORT get_available_providers.

    Returns:
        List of EP dicts with name, device, and optional path.
    """
    eps: list[dict[str, Any]] = []
    winml_eps: dict[str, str] = {}

    # Try WinML EP Registry first
    try:
        from ..session import WinMLEPRegistry

        registry = WinMLEPRegistry.get_instance()
        winml_eps = registry.get_available_eps()
    except Exception as e:
        logger.debug("WinML EP registry unavailable: %s", e)

    # Get ORT available providers as fallback / supplement
    ort_providers: list[str] = []
    try:
        import onnxruntime as ort

        ort_providers = ort.get_available_providers()
    except Exception as e:
        logger.debug("ORT not available: %s", e)

    # Merge: WinML EPs first (they have paths), then ORT-only EPs
    ep_device_map = get_ep_device_map()
    seen: set[str] = set()

    for ep_name, ep_path in winml_eps.items():
        device = ep_device_map.get(ep_name, "unknown").upper()
        eps.append({"name": ep_name, "device": device, "path": ep_path})
        seen.add(ep_name)

    for ep_name in ort_providers:
        if ep_name not in seen:
            device = ep_device_map.get(ep_name, "unknown").upper()
            eps.append({"name": ep_name, "device": device, "path": None})
            seen.add(ep_name)

    return eps


def _output_ep_text(eps: list[dict[str, Any]]) -> None:
    """Display EP list in rich text format."""
    console.print("\n[bold blue]Available Execution Providers[/bold blue]")
    if not eps:
        console.print("  [yellow]No execution providers found.[/yellow]")
        return

    for ep in eps:
        name_padded = ep["name"].ljust(30)
        console.print(f"  [bold]{name_padded}[/bold] [dim]->[/dim] [cyan]{ep['device']}[/cyan]")
        if ep.get("path"):
            console.print(f"    Path: {ep['path']}")
        else:
            console.print("    [dim](built-in)[/dim]")


@click.command()  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "compact"], case_sensitive=False),
    default="text",
    help="Output format: text (human-readable), json, or compact",
)
@click.option(  # type: ignore[misc]
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Include additional diagnostic information",
)
@click.option(  # type: ignore[misc]
    "--list-device",
    is_flag=True,
    default=False,
    help="List available devices in priority order",
)
@click.option(  # type: ignore[misc]
    "--list-ep",
    is_flag=True,
    default=False,
    help="List available execution providers",
)
@click.pass_context  # type: ignore[misc]
def sysinfo(
    ctx: click.Context,
    output_format: str,
    verbose: bool,
    list_device: bool,
    list_ep: bool,
) -> None:
    r"""Display system information for ModelKit export.

    This command gathers and displays information relevant to ONNX model
    export, including Python version, library versions, hardware
    capabilities, and backend SDK availability.

    Use this to diagnose issues with model export or verify your
    environment is properly configured.

    \b
    Examples:
        # Display system info (human-readable format)
        winml sys

        # Get output as JSON for scripting
        winml sys --format json

        # Show detailed info
        winml sys --verbose

        # Compact format for quick overview
        winml sys --format compact

        # List available devices
        winml sys --list-device

        # List execution providers as JSON
        winml sys --list-ep --format json
    """
    # Inherit debug mode from parent
    if ctx.obj.get("debug"):
        verbose = True

    # Route winml.modelkit logs through Rich so they never interleave with CLI output.
    # In normal mode suppress everything below WARNING; in debug mode show all levels.
    # Restore logger state on exit so tests using caplog are not affected.
    log_level = logging.DEBUG if verbose else logging.WARNING
    pkg_logger = logging.getLogger("winml.modelkit")
    _saved_handlers = pkg_logger.handlers[:]
    _saved_level = pkg_logger.level
    _saved_propagate = pkg_logger.propagate
    pkg_logger.handlers = [h for h in pkg_logger.handlers if not isinstance(h, RichHandler)]
    rich_handler = RichHandler(console=console, show_path=False)
    rich_handler.setLevel(log_level)
    pkg_logger.setLevel(log_level)
    pkg_logger.addHandler(rich_handler)
    pkg_logger.propagate = False

    try:
        use_json = output_format.lower() == "json"

        # Handle --list-device and/or --list-ep (combinable)
        if list_device or list_ep:
            if use_json:
                # Combine both into a single JSON object so output is always valid JSON
                result: dict[str, Any] = {}
                if list_device:
                    try:
                        result["devices"] = _gather_device_info()
                    except Exception as e:
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep:
                    try:
                        result["executionProviders"] = _gather_ep_info()
                    except Exception as e:
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
                click.echo(json.dumps(result, indent=2))
            elif output_format.lower() == "compact":
                if list_device:
                    try:
                        devices = _gather_device_info()
                        parts = [f"{d['type']}: {d['name'].strip()}" for d in devices]
                        click.echo(" | ".join(parts) if parts else "No devices found")
                    except Exception as e:
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep:
                    try:
                        eps = _gather_ep_info()
                        parts = [f"{ep['name']}({ep['device']})" for ep in eps]
                        click.echo("EPs: " + ", ".join(parts) if parts else "EPs: none")
                    except Exception as e:
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
            else:
                if list_device:
                    try:
                        devices = _gather_device_info()
                        _output_device_text(devices)
                    except Exception as e:
                        console.print(f"[bold red]Error detecting devices:[/bold red] {e}")
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep:
                    try:
                        eps = _gather_ep_info()
                        _output_ep_text(eps)
                    except Exception as e:
                        err_msg = f"[bold red]Error detecting execution providers:[/bold red] {e}"
                        console.print(err_msg)
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
            return

        # Default: full sysinfo including devices and EPs
        try:
            info = _gather_system_info(verbose=verbose)

            if use_json:
                # Add devices and EPs to JSON output
                try:
                    info["devices"] = _gather_device_info()
                except Exception:
                    info["devices"] = []
                try:
                    info["executionProviders"] = _gather_ep_info()
                except Exception:
                    info["executionProviders"] = []
                _output_json(info)
            elif output_format.lower() == "compact":
                _output_compact(info)
            else:
                _output_text(info, verbose=verbose)
                # Append devices and EPs to text output
                console.print()
                try:
                    devices = _gather_device_info()
                    _output_device_text(devices)
                except Exception:
                    logger.debug("Device detection failed in default output")
                console.print()
                try:
                    eps = _gather_ep_info()
                    _output_ep_text(eps)
                except Exception:
                    logger.debug("EP detection failed in default output")

        except Exception as e:
            console.print(f"[bold red]Error gathering system information:[/bold red] {e}")
            logger.exception("Failed to gather system information")
            raise click.ClickException(f"Error gathering system information: {e}") from e

    finally:
        pkg_logger.handlers = _saved_handlers
        pkg_logger.setLevel(_saved_level)
        pkg_logger.propagate = _saved_propagate
