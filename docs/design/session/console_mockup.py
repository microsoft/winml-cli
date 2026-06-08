"""Console mockup: proposed `winml sys --list-ep` output format.

Captures four user-stated requirements:

    1. **DLL-oriented, not catalog-oriented.** Only EPs whose plugin DLL is
       discoverable (or are bundled with ORT) appear. No phantom rows for
       MIGraphX / VitisAI / Tensorrt / NvTensorRtRtx on machines where
       those DLLs aren't installed.

    2. **No device exploration for incompatible EPs.** When a source's DLL
       can't load or the hardware vendor isn't present, mark the entry as
       ``incompatible`` and skip the devices block entirely.

    3. **Shadowed candidates must be try-registered.** A shadowed source
       gets its own numbered entry only if its DLL actually registers.
       Failed shadowed entries appear as ``#N incompatible`` rather than
       being silently dropped.

    4. **Per-device detail.** Each registered (ep, device) pair gets its
       own line with type, hardware name, memory, capabilities — drawing
       on the rich ``ep_metadata`` fields ORT actually surfaces.

Each EP is followed by one or more **numbered entries** (``#1 primary``,
``#2 shadowed``, ``#3 incompatible``), one per discovered source for that
EP. The numbering reflects discovery precedence; only the ``#1`` entry's
devices participate in actual session selection.

**Type-taxonomy note (2026-06-07).** The classes below — ``EntryRow``,
``DeviceRow``, ``EpBlock`` — are **render-time DTOs**. They are produced
from the ``(results, failures)`` tuple returned by the Path B inline loop
(``list[WinMLEP]``, ``list[(EPEntry, Exception)]`` — see
`2_coreloop.md` §5.1 for the loop and §2 for the six-class taxonomy),
not from raw tuples. Specifically:

  - ``EntryRow.status`` is **derived** by the renderer per
    `2_coreloop.md` §5.2 ("primary" = first source under an EP name in
    ``results``; "shadowed" = subsequent source under same EP name
    in ``results``; "incompatible" = source appears in the
    ``failures`` list). Status is NOT a field on ``WinMLEP``;
    ``WinMLEP`` is success-only (``len(.devices) >= 1``) and failures
    live as ``(EPEntry, Exception)`` pairs alongside.
  - ``DeviceRow.facts`` is produced from ``WinMLDevice.facts()`` (the
    runtime adapter for each ``WinMLDevice`` in ``WinMLEP.devices``,
    reached via ``WinMLEP.ep_devices()`` -> ``WinMLEPDevice.device``);
    see `4_winml_device.md` for the ABC.
  - ``EntryRow.version``, ``source_kind``, ``source_label``, ``path``
    come from ``WinMLEP.source`` (the per-source ``EPEntry`` attribution
    record produced by ``discover_all_eps()``). Incompatible entries
    pull the same fields from the raw ``EPEntry`` in the failures list.
  - The flat ``WinMLEPDevice(ep: WinMLEP, device: WinMLDevice)`` pair
    is what the ``WinMLSession(onnx_path, ep_device, ...)`` direct
    constructor consumes downstream; the renderer only reads
    ``WinMLDevice`` (via the pair's ``.device``) since it doesn't
    construct sessions.

For the canonical class reference, see ``3_design_classes.md``.

Run:

    uv run python docs/design/session/console_mockup.py

Compare with the current output to validate the design direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


# ---------------------------------------------------------------------------
# Mock data — drawn from the empirical probe at temp/probe_output.txt
# (Intel Core Ultra 7 258V machine, OpenVINO PyPI installed, no Qualcomm HW).
# ---------------------------------------------------------------------------


Status = Literal["primary", "shadowed", "incompatible"]


@dataclass(frozen=True)
class DeviceRow:
    """One device exposed by a registered EP source."""

    type_tag: str               # "NPU" | "GPU" | "CPU"
    name: str                   # Description / FULL_DEVICE_NAME
    facts: tuple[str, ...]      # zero or more facts shown on the second line


@dataclass(frozen=True)
class EntryRow:
    """One (EP, source) pair — one numbered entry under an EP heading.

    Status semantics:
      - ``primary``: precedence winner; its devices are what sessions use.
      - ``shadowed``: registered cleanly; available as fallback but not bound.
      - ``incompatible``: failed to register, or its hardware vendor isn't
        present on this host. Devices block is omitted.
    """

    status: Status
    version: str                # package / distribution version
    source_kind: str            # "PyPI" / "NuGet" / "MSIX" / "Catalog" / "Directory"
    source_label: str           # distribution / family name / catalog name
    path: str
    devices: tuple[DeviceRow, ...]      # empty for incompatible
    incompatible_reason: str = ""       # populated when status == "incompatible"


@dataclass(frozen=True)
class EpBlock:
    """One EP name + all its discovered source entries."""

    name: str
    vendor_tag: str             # "Intel" / "Qualcomm" / "(built-in)" / ""
    entries: tuple[EntryRow, ...]


# Concrete OpenVINO devices reused for entries where the same DLL contributes
# the same devices.
_OPENVINO_DEVICES: Final[tuple[DeviceRow, ...]] = (
    DeviceRow(
        type_tag="NPU",
        name="Intel(R) AI Boost",
        facts=(
            "Memory: 16.0 GiB",
            "Driver: 1004724",
            "Compiler: 458781",
            "Capabilities: FP16, INT8",
        ),
    ),
    DeviceRow(
        type_tag="GPU",
        name="Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)",
        facts=(
            "Memory: 16.3 GiB",
            "Architecture: v20.4.4",
            "Execution units: 64",
            "Capabilities: FP32, FP16, BIN, INT8, MatMul, USM",
        ),
    ),
    DeviceRow(
        type_tag="CPU",
        name="Intel(R) Core(TM) Ultra 7 258V",
        facts=(
            "Architecture: intel64",
            "Capabilities: BF16, FP32, FP16, INT8, BIN",
        ),
    ),
)


_PYPI_OPENVINO_PATH = (
    r"C:\Users\zhengte\BYOM\ModelKits\winml\.venv\Lib\site-packages\\"
    r"onnxruntime_ep_openvino\onnxruntime_providers_openvino_plugin.dll"
)

_MSIX_CATALOG_OPENVINO_PATH = (
    r"C:\Program Files\WindowsApps\\"
    r"MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.79.0_x64__"
    r"8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll"
)

_MSIX_WORKLOAD_OPENVINO_PATH = (
    r"C:\Program Files\WindowsApps\\"
    r"WindowsWorkload.EP.Intel.OpenVINO.1.8_1.8.61.0_x64__"
    r"8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll"
)


MOCK_LISTING: Final[tuple[EpBlock, ...]] = (
    EpBlock(
        name="OpenVINOExecutionProvider",
        vendor_tag="Intel",
        entries=(
            EntryRow(
                status="primary",
                version="1.4.1",
                source_kind="PyPI",
                source_label="onnxruntime-ep-openvino",
                path=_PYPI_OPENVINO_PATH,
                devices=_OPENVINO_DEVICES,
            ),
            EntryRow(
                status="shadowed",
                version="1.8.79.0",
                source_kind="MSIX",
                source_label="MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8",
                path=_MSIX_CATALOG_OPENVINO_PATH,
                devices=_OPENVINO_DEVICES,
            ),
            EntryRow(
                status="incompatible",
                version="1.8.61.0",
                source_kind="MSIX",
                source_label="WindowsWorkload.EP.Intel.OpenVINO.1.8",
                path=_MSIX_WORKLOAD_OPENVINO_PATH,
                devices=(),
                incompatible_reason="DLL load failed: missing OS-side runtime stack",
            ),
        ),
    ),
    EpBlock(
        name="CPUExecutionProvider",
        vendor_tag="(built-in)",
        entries=(
            EntryRow(
                status="primary",
                version="",
                source_kind="",
                source_label="",
                path="",
                devices=(
                    DeviceRow(
                        type_tag="CPU",
                        name="Intel(R) Core(TM) Ultra 7 258V",
                        facts=("Cores: 8", "Architecture: x64"),
                    ),
                ),
            ),
        ),
    ),
    EpBlock(
        name="DmlExecutionProvider",
        vendor_tag="(built-in)",
        entries=(
            EntryRow(
                status="primary",
                version="",
                source_kind="",
                source_label="",
                path="",
                devices=(
                    DeviceRow(
                        type_tag="GPU",
                        name="Intel(R) Arc(TM) 140V GPU (16GB)",
                        facts=(
                            "Memory: 128 MB shared",
                            "DXGI adapter #0",
                            "High-performance preference",
                        ),
                    ),
                ),
            ),
        ),
    ),
    EpBlock(
        name="QNNExecutionProvider",
        vendor_tag="Qualcomm",
        entries=(
            EntryRow(
                status="incompatible",
                version="2.1.1",
                source_kind="PyPI",
                source_label="onnxruntime-qnn",
                path=(
                    r"C:\Users\zhengte\BYOM\ModelKits\winml\.venv\Lib\site-packages\\"
                    r"onnxruntime_qnn\libs\amd64\onnxruntime_providers_qnn.dll"
                ),
                devices=(),
                incompatible_reason="no Qualcomm hardware detected on this machine",
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------


_INDENT = "  "


def _render_header(blocks: tuple[EpBlock, ...]) -> str:
    primaries = sum(
        sum(1 for e in b.entries if e.status == "primary") for b in blocks
    )
    shadowed = sum(
        sum(1 for e in b.entries if e.status == "shadowed") for b in blocks
    )
    incompatible = sum(
        sum(1 for e in b.entries if e.status == "incompatible") for b in blocks
    )
    return (
        f"Execution Providers  "
        f"({primaries} primary, {shadowed} shadowed, {incompatible} incompatible)\n"
        + "=" * 78
    )


def _wrap_path(path: str, lead: int) -> str:
    """Break long paths after site-packages or ExecutionProvider for readability."""
    width = 78
    if len(path) + lead <= width:
        return path
    # Simple heuristic: split on first occurrence of '\site-packages\\' or
    # '\ExecutionProvider\\' so the wrap point lands at a natural boundary.
    for marker in (r"\site-packages\\", r"\ExecutionProvider\\"):
        if marker in path:
            head, tail = path.split(marker, 1)
            return f"{head}{marker}\n{' ' * lead}{tail}"
    return path  # give up; let it overflow


def _render_entry(idx: int, entry: EntryRow) -> str:
    """Render one numbered entry under an EP heading."""
    lines: list[str] = []
    header = f"{_INDENT}#{idx} {entry.status}"
    lines.append(header)

    inner = _INDENT * 2

    # Built-in entries (CPU, DML, etc.) have no source/version/path metadata
    # — the bundled-with-ORT label on the EP heading covers them.
    if entry.source_kind:
        lines.append(f"{inner}Version:  {entry.version}")
        lines.append(
            f"{inner}Source:   {entry.source_kind} / {entry.source_label}"
        )
        wrapped = _wrap_path(entry.path, lead=len(inner) + len("Path:     "))
        lines.append(f"{inner}Path:     {wrapped}")

    if entry.status == "incompatible":
        lines.append(f"{inner}Status:   {entry.incompatible_reason}")
        return "\n".join(lines)

    if entry.devices:
        lines.append(f"{inner}Devices:")
        dev_inner = inner + "  "
        for dev in entry.devices:
            lines.append(f"{dev_inner}[{dev.type_tag}]   {dev.name}")
            if dev.facts:
                fact_indent = dev_inner + "        "
                # Show facts as continuation indented under the device name.
                # Pipe-separated within one line; wrap to a new line if needed.
                first = True
                buf: list[str] = []
                width = 78 - len(fact_indent)
                for fact in dev.facts:
                    candidate = "  |  ".join([*buf, fact]) if buf else fact
                    if len(candidate) > width and buf:
                        lines.append(fact_indent + "  |  ".join(buf))
                        buf = [fact]
                    else:
                        buf.append(fact)
                if buf:
                    lines.append(fact_indent + "  |  ".join(buf))

    return "\n".join(lines)


def _render_ep_block(block: EpBlock) -> str:
    title = block.name
    pad = max(1, 78 - len(title) - len(block.vendor_tag))
    header = f"{title}{' ' * pad}{block.vendor_tag}"
    body = "\n\n".join(_render_entry(i + 1, e) for i, e in enumerate(block.entries))
    return f"\n{header}\n{body}"


def main() -> None:
    print(_render_header(MOCK_LISTING))
    for block in MOCK_LISTING:
        print(_render_ep_block(block))
    print()


if __name__ == "__main__":
    main()
