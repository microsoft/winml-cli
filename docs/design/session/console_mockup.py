"""Console mockup: proposed `winml sys` output format (v3 layout).

The v3 layout has two top-level sections:

    1. **Available Devices** — a flat enumeration of the host's hardware
       devices, one row per device (NPU / GPU / CPU). Each row shows the
       type tag, the hardware name, the vendor, and a second line with
       architecture/driver/cores details.

    2. **Available Execution Providers** — one block per EP name. Each
       block lists every discovered source (PyPI, MSIX, Catalog,
       Directory, NuGet, built-in) underneath with a status tag
       (``[primary]`` / ``[shadowed]`` / ``[incompatible]``). Under every
       source we render three rows: ``Version:`` (runtime plugin version
       from ORT ``ep_metadata['version']``), ``Path:`` (loaded DLL), and
       ``Devices:`` (nested ``NPU:`` / ``GPU:`` / ``CPU:`` per-device-type
       facts the EP published).

The nested ``Devices:`` → ``NPU:`` / ``GPU:`` / ``CPU:`` shape is
deliberate: the top section answers "what hardware is on this box?", and
the per-source block under each EP answers "what does this particular
EP source see and expose for each device type?". These are different
projections of the same hardware and should not be conflated.

**Type-taxonomy note (2026-06-07).** The classes below — ``DeviceListing``,
``PerDeviceFacts``, ``SourceRow``, ``EPBlock`` — are **render-time DTOs**.
They are produced from the ``(results, failures)`` tuple returned by the
Path B inline loop (``list[WinMLEP]``, ``list[(EPEntry, Exception)]`` —
see `2_coreloop.md` §5.1 for the loop and §2 for the six-class taxonomy),
not from raw tuples. Specifically:

  - ``SourceRow.status`` is **derived** by the renderer per
    `2_coreloop.md` §5.2 ("primary" = first source under an EP name in
    ``results``; "shadowed" = subsequent source under same EP name in
    ``results``; "incompatible" = source appears in the ``failures``
    list). Status is NOT a field on ``WinMLEP``; ``WinMLEP`` is
    success-only (``len(.devices) >= 1``) and failures live as
    ``(EPEntry, Exception)`` pairs alongside.
  - ``PerDeviceFacts`` is produced by aggregating
    ``WinMLDevice.device_facts()`` (device-intrinsic: Architecture,
    Driver) and ``WinMLDevice.ep_facts()`` (EP-mediated: Memory,
    Capabilities) for each ``WinMLDevice`` in ``WinMLEP.devices``,
    by type tag. See `4_winml_device.md` §4.1 for the attribution
    table.
  - ``SourceRow.version``, ``kind_label``, ``source_label``, ``path``
    come from ``WinMLEP.source`` (the per-source ``EPEntry`` attribution
    record produced by ``discover_all_eps()``). Incompatible entries
    pull the same fields from the raw ``EPEntry`` in the failures list.
  - ``SourceRow.plugin_version`` comes from
    ``WinMLEP.devices[0]._ort.ep_metadata['version']`` — the runtime
    version of the loaded DLL. Distinct from ``SourceRow.version``,
    which is the packaging version and is ``None`` for
    ``DirectorySource`` / built-in EPs.
  - ``DeviceListing`` rows are produced from the host hardware probe
    (independent of any EP), so the top section renders identically
    regardless of which EPs are installed.

For the canonical class reference, see ``3_design_classes.md``.

Run:

    uv run python docs/design/session/console_mockup.py

Compare with the current output to validate the design direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from rich.console import Console


# ---------------------------------------------------------------------------
# DTOs.
# ---------------------------------------------------------------------------


Status = Literal["primary", "shadowed", "incompatible"]


@dataclass(frozen=True)
class DeviceListing:
    """One row in the ``Available Devices`` section."""

    type_tag: str               # "NPU" | "GPU" | "CPU"
    hardware_name: str
    vendor: str
    architecture: str
    driver: str | None          # populated for NPU/GPU; None for CPU
    cores: str | None           # populated for CPU; None for NPU/GPU


@dataclass(frozen=True)
class PerDeviceFacts:
    """Facts an EP source publishes for one device-type slot.

    A ``memory`` of ``None`` means the EP doesn't publish a memory value
    for this slot. An empty ``capabilities`` tuple combined with
    ``memory is None`` is the built-in "no metadata published" shape and
    is rendered as a single annotation.
    """

    type_tag: str               # "NPU" | "GPU" | "CPU"
    memory: str | None
    capabilities: tuple[str, ...]


# Sentinel rendered as one collapsed line under a shadowed source whose DLL
# bytes are byte-identical to the primary's. We use a singleton tuple of a
# marker so the renderer can detect collapse by identity
# (``source.devices is _COLLAPSE_TO_PRIMARY``) and distinguish it from the
# "no devices" / empty-marker case which is plain ``()``.
_COLLAPSE_MARKER: Final[PerDeviceFacts] = PerDeviceFacts(
    type_tag="__collapse__", memory=None, capabilities=()
)
_COLLAPSE_TO_PRIMARY: Final[tuple[PerDeviceFacts, ...]] = (_COLLAPSE_MARKER,)


@dataclass(frozen=True)
class SourceRow:
    """One (EP, source) pair — one bracketed entry under an EP heading."""

    status: Status
    kind_label: str             # "PyPI" / "NuGet" / "MSIX" / "Catalog" / "Directory" / "bundled"
    source_label: str | None    # distribution / family name; None for bundled
    version: str | None         # packaging version (PyPI/MSIX/NuGet); None for Directory/bundled
    plugin_version: str | None  # ORT runtime ep_metadata['version']; None when unpublished
    catalog_default: bool       # true → render "(catalog default)" annotation
    path: str | None            # None for bundled
    devices: tuple[PerDeviceFacts, ...]   # empty tuple → collapse / none
    error: str | None = None    # rendered as "Error: …" when present


@dataclass(frozen=True)
class EPBlock:
    """One EP name + all its discovered source rows."""

    name: str
    incompatible: bool          # true → render "[incompatible]" next to name
    sources: tuple[SourceRow, ...]


@dataclass(frozen=True)
class MockListing:
    device_listings: tuple[DeviceListing, ...]
    ep_blocks: tuple[EPBlock, ...]


# ---------------------------------------------------------------------------
# Mock data — drawn from the empirical probe at temp/probe_output.txt
# (Intel Core Ultra 7 258V machine, OpenVINO PyPI installed, no Qualcomm HW).
# ---------------------------------------------------------------------------


_PYPI_OPENVINO_PATH = (
    r"C:\Users\zhengte\BYOM\ModelKits\winml\.venv\Lib\site-packages"
    r"\onnxruntime_ep_openvino\onnxruntime_providers_openvino_plugin.dll"
)

_CATALOG_OPENVINO_PATH = (
    r"C:\Program Files\WindowsApps"
    r"\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.79.0"
    r"_x64__8wekyb3d8bbwe\ExecutionProvider"
    r"\onnxruntime_providers_openvino_plugin.dll"
)

_MSIX_WORKLOAD_OPENVINO_PATH = (
    r"C:\Program Files\WindowsApps"
    r"\WindowsWorkload.EP.Intel.OpenVINO.1.8_1.8.61.0"
    r"_x64__8wekyb3d8bbwe\ExecutionProvider"
    r"\onnxruntime_providers_openvino_plugin.dll"
)

_QNN_PYPI_PATH = (
    r"C:\Users\zhengte\BYOM\ModelKits\winml\.venv\Lib\site-packages"
    r"\onnxruntime_qnn\libs\amd64\onnxruntime_providers_qnn.dll"
)


_OPENVINO_FACTS: Final[tuple[PerDeviceFacts, ...]] = (
    PerDeviceFacts(
        type_tag="NPU",
        memory="16.0 GiB",
        capabilities=("FP16", "INT8"),
    ),
    PerDeviceFacts(
        type_tag="GPU",
        memory="16.3 GiB",
        capabilities=("FP32", "FP16", "BIN", "INT8", "MatMul", "USM"),
    ),
    PerDeviceFacts(
        type_tag="CPU",
        memory=None,
        capabilities=("BF16", "FP32", "FP16", "INT8", "BIN"),
    ),
)


MOCK_LISTING: Final[MockListing] = MockListing(
    device_listings=(
        DeviceListing(
            type_tag="NPU",
            hardware_name="Intel(R) AI Boost",
            vendor="Intel Corporation",
            architecture="NPU gen-4 (4000)",
            driver="32.0.100.4724",
            cores=None,
        ),
        DeviceListing(
            type_tag="GPU",
            hardware_name="Intel(R) Arc(TM) 140V GPU (iGPU, 16 GB)",
            vendor="Intel Corporation",
            architecture="Xe2 (v20.4.4)",
            driver="32.0.101.8424",
            cores=None,
        ),
        DeviceListing(
            type_tag="CPU",
            hardware_name="Intel(R) Core(TM) Ultra 7 258V",
            vendor="Intel Corporation",
            architecture="x86_64 (intel64)",
            driver=None,
            cores="8 phys / 8 logical",
        ),
    ),
    ep_blocks=(
        EPBlock(
            name="OpenVINOExecutionProvider",
            incompatible=False,
            sources=(
                SourceRow(
                    status="primary",
                    kind_label="PyPI",
                    source_label="onnxruntime-ep-openvino",
                    version="1.4.1",
                    plugin_version="1.5.2+c6549bd",
                    catalog_default=False,
                    path=_PYPI_OPENVINO_PATH,
                    devices=_OPENVINO_FACTS,
                ),
                SourceRow(
                    status="shadowed",
                    kind_label="Catalog",
                    source_label=None,
                    version=None,
                    plugin_version="1.5.2+c6549bd",
                    catalog_default=True,
                    path=_CATALOG_OPENVINO_PATH,
                    devices=_COLLAPSE_TO_PRIMARY,
                ),
                SourceRow(
                    status="shadowed",
                    kind_label="MSIX",
                    source_label="WindowsWorkload.EP.Intel.OpenVINO.1.8",
                    version="1.8.61.0",
                    plugin_version="1.5.2+c6549bd",
                    catalog_default=False,
                    path=_MSIX_WORKLOAD_OPENVINO_PATH,
                    devices=_OPENVINO_FACTS,
                ),
            ),
        ),
        EPBlock(
            name="QNNExecutionProvider",
            incompatible=True,
            sources=(
                SourceRow(
                    status="incompatible",
                    kind_label="PyPI",
                    source_label="onnxruntime-qnn",
                    version="2.1.1",
                    plugin_version=None,
                    catalog_default=False,
                    path=_QNN_PYPI_PATH,
                    devices=(),
                    error="Qualcomm hardware not present on this host",
                ),
            ),
        ),
        EPBlock(
            name="DmlExecutionProvider",
            incompatible=False,
            sources=(
                SourceRow(
                    status="primary",
                    kind_label="bundled",
                    source_label=None,
                    version=None,
                    plugin_version=None,
                    catalog_default=False,
                    path=None,
                    devices=(
                        PerDeviceFacts(
                            type_tag="GPU",
                            memory=None,
                            capabilities=(),
                        ),
                    ),
                ),
            ),
        ),
        EPBlock(
            name="CPUExecutionProvider",
            incompatible=False,
            sources=(
                SourceRow(
                    status="primary",
                    kind_label="bundled",
                    source_label=None,
                    version=None,
                    plugin_version=None,
                    catalog_default=False,
                    path=None,
                    devices=(
                        PerDeviceFacts(
                            type_tag="CPU",
                            memory=None,
                            capabilities=(),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------


console = Console(width=120, soft_wrap=False, highlight=False)

# Widths for column alignment.
_INDEX_W = 4         # "#1  "
_TYPE_W = 4          # "NPU "
_DEVICE_NAME_W = 50  # left column for hardware name in Available Devices
_KIND_W = 7          # min width for kind label (PyPI / MSIX / Catalog ...)


def _pad_visible(s: str, width: int) -> str:
    """Pad ``s`` with trailing spaces so its visible length is ``width``.

    ``s`` may contain Rich markup; we use ``Console.render_str`` to
    measure the rendered cell length, then append plain spaces.
    """
    visible = len(console.render_str(s))
    if visible >= width:
        return s
    return s + " " * (width - visible)


def _render_devices_section(listings: tuple[DeviceListing, ...]) -> None:
    console.print("[bold blue]Available Devices[/bold blue]")
    for i, dev in enumerate(listings, start=1):
        index_cell = _pad_visible(f"[dim]#{i}[/dim]", _INDEX_W)
        type_cell = _pad_visible(f"[bold cyan]{dev.type_tag}[/bold cyan]", _TYPE_W)
        name_cell = _pad_visible(dev.hardware_name, _DEVICE_NAME_W)
        line1 = (
            f"  {index_cell}[dim]Device:[/dim] {type_cell} "
            f"{name_cell} [dim]{dev.vendor}[/dim]"
        )
        console.print(line1)

        arch_field = (
            f"[dim]Architecture:[/dim] {_pad_visible(dev.architecture, 24)}"
        )
        if dev.driver is not None:
            tail = f"[dim]Driver:[/dim] {dev.driver}"
        else:
            tail = f"[dim]Cores:[/dim] {dev.cores or ''}"
        # The 14-space indent below the index aligns the second-line labels
        # under the hardware name column (matches the EP-source indent so
        # the visual gutter is uniform throughout the screen).
        console.print(f"              {arch_field} {tail}")

        if i < len(listings):
            console.print()


def _status_tag(status: Status) -> str:
    if status == "primary":
        return "[green]\\[primary][/green]"
    if status == "shadowed":
        return "[yellow]\\[shadowed][/yellow]"
    return "[red]\\[incompatible][/red]"


def _render_source_header(source: SourceRow) -> str:
    status = _status_tag(source.status)
    kind = _pad_visible(f"[bold]{source.kind_label}[/bold]", _KIND_W)
    parts = [f"    {status} {kind}"]
    if source.catalog_default:
        parts.append("[dim](catalog default)[/dim]")
    else:
        label = source.source_label or ""
        version = source.version or ""
        text = f"{label} {version}".strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _render_facts_block(devices: tuple[PerDeviceFacts, ...]) -> None:
    """Render the ``Devices:`` → per-type-tag facts block under a source."""
    console.print("              [dim]Devices:[/dim]")
    for facts in devices:
        type_label = _pad_visible(
            f"[bold cyan]{facts.type_tag}:[/bold cyan]", 5
        )
        fields: list[str] = []
        if facts.memory is not None:
            fields.append(f"[dim]Memory:[/dim] {facts.memory}")
        if facts.capabilities:
            fields.append(
                f"[dim]Capabilities:[/dim] {', '.join(facts.capabilities)}"
            )
        if not fields:
            body = "[dim](no metadata published)[/dim]"
        else:
            body = "  |  ".join(fields)
        console.print(f"                {type_label} {body}")


def _render_source(source: SourceRow) -> None:
    console.print(_render_source_header(source))

    # Runtime plugin version from ORT ``ep_metadata['version']`` — rendered
    # on its own row (semantically distinct from ``source.version`` which
    # is the packaging version already inside the header).
    if source.plugin_version is not None:
        console.print(f"              [dim]Version:[/dim] {source.plugin_version}")

    if source.path is not None:
        # Compact the path with an ellipsis so the rendered cell stays in one
        # screen width; the path "rule" in the mockup uses a leading "…\\".
        displayed = _shorten_path(source.path)
        console.print(f"              [dim]Path:[/dim]    {displayed}")

    if source.devices is _COLLAPSE_TO_PRIMARY:
        console.print(
            "              [dim]Devices: (identical facts — "
            "same DLL bytes as primary)[/dim]"
        )
        return

    if not source.devices:
        # Incompatible (or otherwise device-less) source: render the empty
        # marker carrying the reason if we have one, else a bare "(none)".
        reason = source.error or "no devices reported"
        console.print(
            f"              [dim]Devices: (none — {reason})[/dim]"
        )
        return

    # Non-incompatible source with a separate ``Error:`` annotation
    # (rare — a source that registered but published a warning).
    if source.error is not None:
        console.print(f"              [red]Error:[/red] {source.error}")

    _render_facts_block(source.devices)


def _shorten_path(path: str) -> str:
    """Trim a Windows absolute path to its tail with a leading ellipsis.

    Pure presentation aid for the mockup — keeps long site-packages and
    WindowsApps paths from blowing past the terminal width while still
    making it obvious where the DLL came from.
    """
    markers = (
        r"\.venv\Lib",
        r"\site-packages",
        r"\WindowsApps",
    )
    for marker in markers:
        idx = path.find(marker)
        if idx > 0:
            return "…" + path[idx:]
    return path


def _render_ep_block(block: EPBlock) -> None:
    name = f"[bold]{block.name}[/bold]"
    if block.incompatible:
        name = f"{name}  [bold red]\\[incompatible][/bold red]"
    console.print(f"  {name}")

    for i, source in enumerate(block.sources):
        _render_source(source)
        if i < len(block.sources) - 1:
            console.print()


def render(
    listings: tuple[DeviceListing, ...],
    blocks: tuple[EPBlock, ...],
) -> None:
    _render_devices_section(listings)
    console.print()
    console.print()
    console.print("[bold blue]Available Execution Providers[/bold blue]")
    for i, block in enumerate(blocks):
        _render_ep_block(block)
        if i < len(blocks) - 1:
            console.print()


def main() -> None:
    render(MOCK_LISTING.device_listings, MOCK_LISTING.ep_blocks)


if __name__ == "__main__":
    main()
