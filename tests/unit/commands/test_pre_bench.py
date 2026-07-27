# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pre-bench identity block: HF and ONNX-file paths + Option B provenance."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel

from winml.modelkit.commands._pre_bench import print_pre_bench_block


_PLUGIN_DLL = (
    r"C:\Users\zhengte\BYOM\ModelKits\winml\x64\Release"
    r"\onnxruntime_providers_openvino_plugin.dll"
)


def _render_hf() -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id="facebook/convnext-base-224",
        task="image-classification",
        opset=17,
        inputs=[("pixel_values", "float32", (1, 3, 224, 224))],
        outputs=[("logits", "float32", (1, 1000))],
        cached_onnx_path=r"C:\Users\u\.cache\winml\artifacts\convnext.onnx",
        onnx_file=None,
        device="npu",
        hardware_name="NPU Compute Accelerator Device",
        ep="qnn",
        ep_source="pypi",
        ep_version="2.29.0",
        ep_dll_path=r"C:\Users\u\onnxruntime_providers_qnn.dll",
    )
    return console.export_text()


def _render_onnx_file() -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id=None,
        task=None,
        opset=None,
        inputs=None,
        outputs=None,
        cached_onnx_path=None,
        onnx_file=r"C:\models\my_model.onnx",
        device="cpu",
        hardware_name="Intel(R) Core(TM) Ultra 9 285H",
        ep="cpu",
        ep_source="bundled",
        ep_version=None,
        ep_dll_path="",
    )
    return console.export_text()


def test_hf_block_shows_model_id():
    out = _render_hf()
    assert "facebook/convnext-base-224" in out
    assert "image-classification" in out
    assert "17" in out  # opset
    assert "convnext.onnx" in out


def test_hf_block_shows_inputs_and_outputs():
    out = _render_hf()
    assert "pixel_values" in out
    assert "logits" in out


def test_onnx_file_block_shows_path_only():
    out = _render_onnx_file()
    assert "my_model.onnx" in out
    assert "facebook" not in out
    assert "image-classification" not in out


def test_device_block_shows_device_and_ep():
    out = _render_hf()
    assert "npu" in out
    assert "qnn" in out


def test_onnx_file_device_and_ep_present():
    out = _render_onnx_file()
    assert "cpu" in out


def test_dynamic_batch_dim_renders_as_question_mark():
    """Dynamic dims (None in shape) render as '?' not '0'."""
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id="test/model",
        task=None,
        opset=None,
        inputs=[("input_ids", "int64", ("?", 128))],
        outputs=[("logits", "float32", ("?", 1000))],
        cached_onnx_path=None,
        onnx_file=None,
        device="npu",
        hardware_name="",
        ep="qnn",
        ep_source="pypi",
        ep_version=None,
        ep_dll_path=r"C:\fake\qnn.dll",
    )
    out = console.export_text()
    # The "?" sentinel must appear; the raw "0" must NOT (would be wrong).
    assert "?" in out
    # No "(0, 128)" or "(0, 1000)" substrings indicating broken int-coercion
    assert "(0, 128)" not in out
    assert "(0, 1000)" not in out


# =============================================================================
# Option B — provenance rows (Device / EP / EP DLL)
# =============================================================================


def _render(*, ep: str, ep_source: str, ep_version: str | None, ep_dll_path: str) -> str:
    """Minimal HF invocation with configurable EP-provenance fields."""
    buf = StringIO()
    console = Console(file=buf, width=140, force_terminal=False, record=True)
    print_pre_bench_block(
        console,
        model_id="facebook/convnext-base-224",
        task=None,
        opset=None,
        inputs=None,
        outputs=None,
        cached_onnx_path=None,
        onnx_file=None,
        device="npu",
        hardware_name="Intel(R) AI Boost",
        ep=ep,
        ep_source=ep_source,
        ep_version=ep_version,
        ep_dll_path=ep_dll_path,
    )
    return console.export_text()


def test_renders_bundled_ep():
    """Built-in EPs (ep_dll_path='') render as ``(bundled with ORT)`` and never leak 'auto'."""
    out = _render(ep="cpu", ep_source="bundled", ep_version=None, ep_dll_path="")
    assert "EP DLL:" in out
    assert "(bundled with ORT)" in out
    assert "auto" not in out


def test_renders_plugin_ep_with_source():
    """Plugin EPs render ``<short>@<source>``, a ``v<version>`` chunk, and the DLL path."""
    out = _render(
        ep="openvino",
        ep_source="directory",
        ep_version="1.4.1+f33af4f",
        ep_dll_path=_PLUGIN_DLL,
    )
    assert "openvino@directory" in out
    assert "v1.4.1+f33af4f" in out
    assert "onnxruntime_providers_openvino_plugin.dll" in out


def test_never_shows_auto_literal():
    """Every canonical source tag must render without the literal ``"auto"`` slipping in."""
    for source_tag in (
        "bundled",
        "pypi",
        "nuget",
        "directory",
        "winml-catalog",
        "msix",
    ):
        out = _render(
            ep="openvino",
            ep_source=source_tag,
            ep_version="1.4.1",
            ep_dll_path=_PLUGIN_DLL,
        )
        assert "auto" not in out, f"'auto' leaked when ep_source={source_tag!r}"


def test_omits_version_chunk_when_none():
    """``ep_version=None`` drops the ``v<version>`` suffix from the EP row."""
    out = _render(
        ep="openvino",
        ep_source="directory",
        ep_version=None,
        ep_dll_path=_PLUGIN_DLL,
    )
    assert "openvino@directory" in out
    # No " v" chunk anywhere on the EP row means neither the letter 'v' nor a
    # version number follows the source tag.
    ep_row = next(line for line in out.splitlines() if "EP:" in line and "DLL" not in line)
    assert "v" not in ep_row.split("@")[1]


# =============================================================================
# Structural: two Rich Panels with titles "Model" and "Device"
# =============================================================================


def test_renders_two_panels_with_model_and_device_titles():
    """The block prints two top-level ``Panel`` renderables titled Model / Device.

    ``export_text()`` strips Panel borders, so substring assertions on the
    exported text can't distinguish "wrapped in a Panel" from "printed as
    plain lines". Intercept ``console.print`` instead and inspect the
    positional arg — it must be a ``Panel`` with the expected title. Guards
    against silent removal of either ``Panel(...)`` wrapping in
    ``_pre_bench.print_pre_bench_block``.
    """
    console = Console(file=StringIO(), width=120, force_terminal=False)
    captured: list[object] = []
    original_print = console.print

    def _capture_print(*args: object, **kwargs: object) -> None:
        captured.extend(args)
        original_print(*args, **kwargs)  # type: ignore[arg-type]

    console.print = _capture_print  # type: ignore[method-assign]

    print_pre_bench_block(
        console,
        model_id="facebook/convnext-base-224",
        task="image-classification",
        opset=17,
        inputs=[("pixel_values", "float32", (1, 3, 224, 224))],
        outputs=[("logits", "float32", (1, 1000))],
        cached_onnx_path=None,
        onnx_file=None,
        device="npu",
        hardware_name="Intel(R) AI Boost",
        ep="openvino",
        ep_source="pypi",
        ep_version="1.4.1",
        ep_dll_path=r"C:\fake\ov.dll",
    )

    panels = [obj for obj in captured if isinstance(obj, Panel)]
    titles = [str(p.title) for p in panels]
    assert "Model" in titles, f"expected a Panel titled 'Model', got titles={titles!r}"
    assert "Device" in titles, f"expected a Panel titled 'Device', got titles={titles!r}"
    assert titles.index("Model") < titles.index("Device"), (
        f"Model Panel must be printed before Device Panel; got order={titles!r}"
    )
