# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pre-bench identity block: HF and ONNX-file paths."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from winml.modelkit.commands._pre_bench import print_pre_bench_block


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
        device="NPU",
        ep="QNN",
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
        device="CPU",
        ep="ORT-CPU",
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
    assert "NPU" in out
    assert "QNN" in out


def test_onnx_file_device_and_ep_present():
    out = _render_onnx_file()
    assert "CPU" in out
    assert "ORT-CPU" in out


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
        device="NPU",
        ep="QNN",
    )
    out = console.export_text()
    # The "?" sentinel must appear; the raw "0" must NOT (would be wrong).
    assert "?" in out
    # No "(0, 128)" or "(0, 1000)" substrings indicating broken int-coercion
    assert "(0, 128)" not in out
    assert "(0, 1000)" not in out
