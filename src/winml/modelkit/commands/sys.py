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
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from ..ep_path import (
    DirectorySource,
    EPEntry,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
)
from ..session import EP_DEVICE_SPECS, WinMLEP, WinMLEPRegistry
from ..sysinfo import OS


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
        except PackageNotFoundError:
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

    # Enrich each device entry with WinMLDevice.device_facts() from any
    # registered EP whose .devices saw this hardware. First successful
    # match per (device_type, hardware_name) pair wins; later EPs binding
    # the same device can't override architecture/driver because those
    # facts are device-intrinsic per docs/design/session/4_winml_device.md
    # §4.1. ``setdefault`` ensures sysinfo-provided values aren't
    # clobbered when both sources have the same key (e.g. ``driver``).
    #
    # TODO(session-public-surface): reaches into ``registry._registered``,
    # which is registry-internal. Consider exposing a small public
    # accessor like ``registered_eps()`` — consistent with the existing
    # ``_entries``-reach smell in this module.
    try:
        from ..session import WinMLEPRegistry

        registry = WinMLEPRegistry.instance()
        for entry in result:
            match_type = entry["type"]
            sysinfo_name = entry["name"]
            for winml_ep in registry._registered.values():
                # Match by device_type AND a fuzzy name relation —
                # exact match fails when OpenVINO appends suffixes like
                # "(iGPU)" to FULL_DEVICE_NAME that sysinfo's WMI query
                # doesn't include. Substring-in-either-direction covers
                # both bias cases.
                matched = next(
                    (
                        d for d in winml_ep.devices
                        if d.device_type == match_type
                        and (
                            d.hardware_name == sysinfo_name
                            or sysinfo_name in d.hardware_name
                            or d.hardware_name in sysinfo_name
                        )
                    ),
                    None,
                )
                if matched is not None:
                    # device_facts() yields ``"Label: Value"`` strings;
                    # split on the first colon so the renderer can merge
                    # by lowercased key alongside sysinfo's details.
                    for fact in matched.device_facts():
                        label, _, value = fact.partition(": ")
                        if label and value:
                            entry["details"].setdefault(label.lower(), value)
                    break
    except Exception as e:
        logger.warning("device_facts enrichment failed: %s", e)

    return result


def _output_device_text(devices: list[dict[str, Any]]) -> None:
    """Display device list in rich text format.

    Per ``docs/design/session/4_winml_device.md`` §4.1, the *Available
    Devices* section renders device-intrinsic facts (Architecture +
    Driver). When :func:`_gather_device_info` enriched the entry from a
    registered ``WinMLDevice.device_facts()`` call the values land in the
    same ``details`` dict alongside sysinfo's keys (``driver``, ``manufacturer``,
    ``cores``, ``threads``, ``architecture``), so we read them
    uniformly here.
    """
    console.print("\n[bold blue]Available Devices (priority order)[/bold blue]")
    for dev in devices:
        console.print(
            f"  [bold]#{dev['priority']}[/bold]  [cyan]{dev['type']:5s}[/cyan] {dev['name']}"
        )
        details = dev.get("details", {})
        if "error" in details:
            console.print(f"             [red]Error: {details['error']}[/red]")
        elif dev["type"] in ("NPU", "GPU"):
            parts = [
                f"Driver: {details.get('driver', 'N/A')}",
                f"Manufacturer: {details.get('manufacturer', 'N/A')}",
            ]
            if arch := details.get("architecture"):
                parts.append(f"Architecture: {arch}")
            console.print(f"             {' | '.join(parts)}")
        elif dev["type"] == "CPU":
            console.print(
                f"             Cores: {details.get('cores', 'N/A')} | "
                f"Threads: {details.get('threads', 'N/A')} | "
                f"Architecture: {details.get('architecture', 'N/A')}"
            )


# --- EP listing ---


def _format_device_types(ep_name: str) -> str:
    """Return ``"<vendor> <DEV1>/<DEV2>"`` device-type string for an EP.

    Enumerates every :data:`EP_DEVICE_SPECS` entry whose ``ep`` matches in
    catalog order, so multi-target EPs (e.g. OpenVINO targets NPU/GPU/CPU)
    surface all of their supported devices. EPs absent from the catalog
    (custom/unknown plugins, or ORT built-ins like Azure) render as
    ``"unknown"``. A short vendor qualifier ("Intel ", "Qualcomm ", …) is
    prepended when the catalog has a vendor requirement for ``ep_name``;
    EPs with no vendor requirement (CPU, DML, Azure) render bare.
    """
    from ..ep_path import EP_CATALOG

    devices = [spec.device.upper() for spec in EP_DEVICE_SPECS if spec.ep == ep_name]
    # De-duplicate while preserving catalog order in case a future catalog
    # entry pairs the same EP with the same device twice (defensive).
    seen: set[str] = set()
    unique = [d for d in devices if not (d in seen or seen.add(d))]
    raw = "/".join(unique) if unique else "unknown"

    # Vendor prefix: pick the shortest alias when multiple are listed
    # (e.g. AMD); empty string when the EP has no vendor requirement.
    vendors = EP_CATALOG.vendor_requirements_for(ep_name)
    prefix = f"{sorted(vendors, key=len)[0]} " if vendors else ""
    return f"{prefix}{raw}"


def _describe_source(entry: EPEntry) -> dict[str, Any]:
    """Build a JSON-friendly per-source descriptor for ``--list-ep``.

    Reads ``entry.version`` as the single source-of-truth for version
    metadata — each EPSource subclass populates it at ``.resolve()`` time
    from its own canonical source (importlib.metadata for PyPI, cache
    subdir name for NuGet, ``Package.Id.Version`` for MSIX). Subclass
    dispatch here only adds source-kind-specific identifying fields
    (distribution, family prefix, catalog name, etc.) — no version
    recovery.
    """
    source = entry.source
    desc: dict[str, Any] = {"source_kind": type(source).__name__}
    if entry.version is not None:
        desc["version"] = entry.version
    if isinstance(source, PyPISource):
        desc["distribution"] = source.distribution
    elif isinstance(source, MSIXPackageSource):
        desc["family_name_prefix"] = source.family_name_prefix
    elif isinstance(source, NuGetSource):
        desc["nuget_id"] = source.distribution
    elif isinstance(source, WinMLCatalogSource):
        desc["catalog_name"] = source.catalog_name
    elif isinstance(source, DirectorySource):
        desc["root"] = str(source.root)
        if source.env_var:
            desc["env_var"] = source.env_var
    return desc


def _gather_ep_info() -> dict[str, dict[str, Any]]:
    """Gather comprehensive EP inventory across every source.

    Walks :func:`discover_all_eps` (PyPI / NuGet / WinMLCatalog /
    DirectorySource / live MSIX enumeration via :func:`list_msix_eps` in
    the default source list) plus any ``WINMLCLI_EP_PATH`` env-var
    override. Each discovered :class:`EPEntry` is fed through
    :meth:`WinMLEPRegistry.register_ep` (Path B — broad enumeration); the
    DLL is loaded so we can surface per-device facts from the resulting
    :class:`WinMLEP`. Registration failures (incompatible hardware,
    missing runtime, etc.) are captured as ``status="incompatible"``
    rows.

    Built-in EPs (CPU, Azure, DML) are appended from
    :meth:`WinMLEPRegistry.builtin_eps` — snapshotted at registry init.
    This command does not import onnxruntime directly. Compatibility is
    queried straight from :data:`EP_CATALOG`.

    Status derivation follows ``2_coreloop.md`` §7.1: first successful
    registration per ep_name is ``"primary"``, subsequent successes are
    ``"shadowed"``, registration failures are ``"incompatible"``. The
    discovery-time ``EPEntry.status`` field is intentionally ignored —
    it reflects precedence-position in the source list, not whether the
    DLL actually loaded.

    Returns:
        Dict ``ep_name -> {compatible, device_types, entries: [...]}``.
    """
    from ..ep_path import EP_CATALOG, discover_all_eps
    from ..session import WinMLEPRegistrationFailed

    # --- Tier 2: register every discovered entry; capture failures as data.
    registry = WinMLEPRegistry.instance()
    results: list[WinMLEP] = []
    failures: list[tuple[EPEntry, Exception]] = []
    for entry in discover_all_eps():
        try:
            results.append(registry.register_ep(entry))
        except WinMLEPRegistrationFailed as e:
            failures.append((entry, e))
        except Exception as e:
            # A broken plugin must not abort the whole inventory walk.
            logger.warning(
                "register_ep raised non-WinMLEPRegistrationFailed for %s: %s",
                entry.ep_name, e,
            )
            failures.append((entry, e))

    # --- Group by ep_name; derive status fresh per §7.1 (ignore
    # EPEntry.status — that's discovery-time, not registration outcome).
    ep_records: dict[str, list[tuple[EPEntry, WinMLEP | None, Exception | None, str]]] = {}
    primary_seen: set[str] = set()
    for winml_ep in results:
        entry = winml_ep.source
        status = "shadowed" if entry.ep_name in primary_seen else "primary"
        primary_seen.add(entry.ep_name)
        ep_records.setdefault(entry.ep_name, []).append(
            (entry, winml_ep, None, status)
        )
    for entry, err in failures:
        ep_records.setdefault(entry.ep_name, []).append(
            (entry, None, err, "incompatible")
        )

    # --- Tag DLL paths that came via WinMLCatalogSource (for the
    # "(catalog default)" annotation in the renderer).
    catalog_default_paths: set[Path] = {
        entry.dll_path
        for rows in ep_records.values()
        for entry, _, _, _ in rows
        if isinstance(entry.source, WinMLCatalogSource)
    }

    # --- Build per-EP output dicts.
    record_by_ep: dict[str, dict[str, Any]] = {}
    for ep_name, rows in ep_records.items():
        first_entry = rows[0][0]
        try:
            compatible = first_entry.source.is_compatible()
        except Exception as e:
            logger.warning(
                "is_compatible() raised for EP %s; treating as compatible: %s",
                ep_name, e,
            )
            compatible = True

        entries_out: list[dict[str, Any]] = []
        for entry, winml_ep, err, derived_status in rows:
            desc = _describe_source(entry)
            # Two independent failure layers per 2_coreloop.md §7.1.1:
            #   L1 = register_ep raised        -> status "failed"
            #   L2 = EP-level vendor rule says wrong hardware
            #        (source.is_compatible() returned False, but the DLL
            #        loaded with a generic fallback) -> "incompatible"
            # L1 wins when both fire.
            if err is not None:
                desc["status"] = "failed"
                desc["compatible"] = False
                desc["error"] = f"{type(err).__name__}: {err}"
            elif not compatible:
                desc["status"] = "incompatible"
                desc["compatible"] = False
            else:
                desc["status"] = derived_status
                desc["compatible"] = True
            desc["dll_path"] = str(entry.dll_path)
            if entry.dll_path in catalog_default_paths:
                desc["is_catalog_default"] = True
            if winml_ep is not None:
                desc["devices"] = [
                    {
                        "device_type": d.device_type,
                        "hardware_name": d.hardware_name,
                        "vendor": d.vendor,
                        # Per docs/design/session/4_winml_device.md §4.1:
                        # the per-source EP rows surface only Memory +
                        # Capabilities (EP-mediated). Architecture +
                        # Driver are device-intrinsic and rendered once
                        # per physical device in the Devices section
                        # via ``_gather_device_info`` enrichment below.
                        "facts": list(d.ep_facts()),
                    }
                    for d in winml_ep.devices
                ]
            entries_out.append(desc)

        record_by_ep[ep_name] = {
            "compatible": compatible,
            "device_types": _format_device_types(ep_name),
            "entries": entries_out,
        }

    # --- Append ORT built-ins (CPU, Azure, DML) — the registry wraps
    # ORT, so we ask it for the built-in set rather than importing
    # onnxruntime directly. If a plugin already provided this ep_name,
    # the plugin record wins.
    for ep_name in registry.builtin_eps():
        if ep_name in record_by_ep:
            continue
        record_by_ep[ep_name] = {
            "compatible": EP_CATALOG.is_compatible(ep_name),
            "device_types": _format_device_types(ep_name),
            "entries": [
                {
                    "status": "primary",
                    "source_kind": "built-in",
                    "dll_path": None,
                }
            ],
        }

    return record_by_ep


_SOURCE_KIND_LABEL = {
    "PyPISource": "PyPI",
    "MSIXPackageSource": "MSIX",
    "NuGetSource": "NuGet",
    "WinMLCatalogSource": "Catalog",
    "DirectorySource": "FS",
    "built-in": "built-in",
}


def _format_devices_from_handles(devices: list[dict[str, Any]]) -> list[str]:
    """Render device-level lines from the WinMLDevice metadata in an entry.

    Each device dict carries ``device_type``, ``hardware_name``, ``vendor``,
    and a ``facts`` list joined by ``"  |  "``. Returns one rich-markup
    line per device, ready for ``console.print``.
    """
    lines: list[str] = []
    for d in devices:
        dev_type = d.get("device_type", "?")
        hw_name = d.get("hardware_name", "<unknown>")
        vendor = d.get("vendor") or ""
        head = f"[cyan]{dev_type}[/cyan] {hw_name}"
        if vendor and vendor not in hw_name:
            head += f"  [dim]({vendor})[/dim]"
        lines.append(f"              [dim]Device:[/dim] {head}")
        facts = d.get("facts") or []
        if facts:
            lines.append(f"                {'  |  '.join(facts)}")
    return lines


def _output_ep_text(eps: dict[str, dict[str, Any]]) -> None:
    """Display the comprehensive EP inventory in rich text format."""
    console.print("\n[bold blue]Available Execution Providers[/bold blue]")
    if not eps:
        console.print("  [yellow]No execution providers found.[/yellow]")
        return

    for ep_name, record in eps.items():
        # Rich treats square brackets as markup; escape the literal status
        # tags with backslashes so [primary] / [failed] etc. render as text.
        # See 2_coreloop.md §7.1.1 for the L1 (failed) vs L2 (incompatible)
        # split — the EP-level tag here is L2 only.
        compat_tag = (
            "" if record["compatible"]
            else r"  [bold red]\[incompatible][/bold red]"
        )
        device_part = f"[cyan]{record['device_types']}[/cyan]"
        console.print(f"  [bold]{ep_name}[/bold]{compat_tag}  [dim]->[/dim] {device_part}")

        for entry in record["entries"]:
            status = entry.get("status", "?")
            kind = entry.get("source_kind", "?")
            # Status colour mirrors §7.1.2:
            #   primary       — green  (this EP's precedence-winner)
            #   shadowed      — yellow (registered cleanly; not Scenario A's pick)
            #   failed        — red    (register_ep raised; carries error field)
            #   incompatible  — red    (vendor rule overrides a successful register)
            status_color = {
                "primary": "green",
                "shadowed": "yellow",
                "failed": "red",
                "incompatible": "red",
            }.get(status, "white")
            tag = f"[{status_color}]\\[{status}][/{status_color}]"

            extras: list[str] = []
            # ``entry["version"]`` is the single source of truth for any
            # version string (populated per-EPSource-subclass at
            # ``.resolve()`` time and copied verbatim by
            # :func:`_describe_source`); render "?" when absent.
            ver = entry.get("version") or "?"
            if "distribution" in entry:
                extras.append(f"{entry['distribution']} {ver}")
            if "nuget_id" in entry:
                extras.append(f"{entry['nuget_id']} {ver}")
            if "family_name_prefix" in entry:
                # Drop the trailing ``_<publisherId>`` (e.g. ``8wekyb3d8bbwe``)
                # for compact CLI display; show the prefix verbatim if no
                # underscore is present.
                family = entry["family_name_prefix"]
                head, sep, _publisher = family.rpartition("_")
                short_family = head if sep else family
                ver = entry.get("version") or "?"
                extras.append(f"{short_family} v{ver}")
                if entry.get("is_catalog_default"):
                    extras.append("[dim](catalog default)[/dim]")
            elif entry.get("is_catalog_default") and kind == "WinMLCatalogSource":
                extras.append("[dim](catalog default)[/dim]")
            if "root" in entry:
                extras.append(f"root={entry['root']}")

            short_kind = _SOURCE_KIND_LABEL.get(kind, kind)
            extras_str = "  ".join(extras) if extras else ""
            console.print(f"    {tag} [bold]{short_kind:9}[/bold] {extras_str}")
            if entry.get("dll_path"):
                console.print(f"              [dim]Path:[/dim] {entry['dll_path']}")
            if entry.get("error"):
                console.print(f"              [red]Error:[/red] {entry['error']}")
            for line in _format_devices_from_handles(entry.get("devices") or []):
                console.print(line)


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
                # Combine both into a single JSON object so output is always valid JSON.
                # ``_gather_ep_info`` runs first when both are requested so the
                # registry is populated before ``_gather_device_info``'s
                # device_facts enrichment pass runs (see text-path comment).
                result: dict[str, Any] = {}
                eps_json: dict[str, dict[str, Any]] | None = None
                if list_ep:
                    try:
                        eps_json = _gather_ep_info()
                    except Exception as e:
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
                if list_device:
                    try:
                        result["devices"] = _gather_device_info()
                    except Exception as e:
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep and eps_json is not None:
                    result["executionProviders"] = eps_json
                click.echo(json.dumps(result, indent=2))
            elif output_format.lower() == "compact":
                # Same ordering as text/json paths — gather EPs first to
                # warm the registry for device_facts enrichment.
                eps_compact: dict[str, dict[str, Any]] | None = None
                if list_ep:
                    try:
                        eps_compact = _gather_ep_info()
                    except Exception as e:
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
                if list_device:
                    try:
                        devices = _gather_device_info()
                        parts = [f"{d['type']}: {d['name'].strip()}" for d in devices]
                        click.echo(" | ".join(parts) if parts else "No devices found")
                    except Exception as e:
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep and eps_compact is not None:
                    parts = [
                        f"{name}({record['device_types']})"
                        for name, record in eps_compact.items()
                    ]
                    click.echo("EPs: " + ", ".join(parts) if parts else "EPs: none")
            else:
                # When both sections are requested we gather EP info first
                # (which populates ``WinMLEPRegistry._registered``) so that
                # ``_gather_device_info``'s device_facts enrichment loop
                # can read Architecture/Driver off any successfully
                # registered WinMLDevice. Rendering order stays Devices →
                # EPs per the design in 4_winml_device.md §4.1.
                eps: dict[str, dict[str, Any]] | None = None
                if list_ep:
                    try:
                        eps = _gather_ep_info()
                    except Exception as e:
                        err_msg = f"[bold red]Error detecting execution providers:[/bold red] {e}"
                        console.print(err_msg)
                        logger.exception("Failed to detect execution providers")
                        msg = f"Error detecting execution providers: {e}"
                        raise click.ClickException(msg) from e
                if list_device:
                    try:
                        devices = _gather_device_info()
                        _output_device_text(devices)
                    except Exception as e:
                        console.print(f"[bold red]Error detecting devices:[/bold red] {e}")
                        logger.exception("Failed to detect devices")
                        raise click.ClickException(f"Error detecting devices: {e}") from e
                if list_ep and eps is not None:
                    _output_ep_text(eps)
            return

        # Default: full sysinfo including devices and EPs
        try:
            info = _gather_system_info(verbose=verbose)

            if use_json:
                # Add devices and EPs to JSON output. Gather EP info
                # first so the registry is populated before
                # ``_gather_device_info``'s device_facts enrichment runs.
                try:
                    eps_default_json = _gather_ep_info()
                except Exception:
                    eps_default_json = {}
                try:
                    info["devices"] = _gather_device_info()
                except Exception:
                    info["devices"] = []
                info["executionProviders"] = eps_default_json
                _output_json(info)
            elif output_format.lower() == "compact":
                _output_compact(info)
            else:
                _output_text(info, verbose=verbose)
                # Gather EPs first so device_facts enrichment in
                # ``_gather_device_info`` can read off the registry; the
                # render order stays Devices → EPs.
                try:
                    eps_default = _gather_ep_info()
                except Exception as e:
                    eps_default = None
                    logger.warning("EP detection failed: %s", e)
                console.print()
                try:
                    devices = _gather_device_info()
                    _output_device_text(devices)
                except Exception as e:
                    logger.warning("Device detection failed: %s", e)
                    console.print(
                        "[yellow]Device detection failed — re-run with "
                        "[bold]-v[/bold] for the full traceback.[/yellow]"
                    )
                console.print()
                if eps_default is not None:
                    _output_ep_text(eps_default)
                else:
                    console.print(
                        "[yellow]EP detection failed — re-run with "
                        "[bold]-v[/bold] for the full traceback.[/yellow]"
                    )

        except Exception as e:
            console.print(f"[bold red]Error gathering system information:[/bold red] {e}")
            logger.exception("Failed to gather system information")
            raise click.ClickException(f"Error gathering system information: {e}") from e

    finally:
        pkg_logger.handlers = _saved_handlers
        pkg_logger.setLevel(_saved_level)
        pkg_logger.propagate = _saved_propagate
