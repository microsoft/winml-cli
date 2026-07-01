# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Subprocess worker functions for EP-context compilation.

Functions in this module are intentionally defined at module scope so they
can be serialised by ``multiprocessing.spawn`` on Windows (which uses
pickle to transfer the target callable to the child process).

Do **not** move these into class bodies or nest them inside other functions.
"""

from __future__ import annotations


def qnn_compile_to_ep_context(src: str, dst: str, qnn_options: dict) -> None:
    """Compile *src* ONNX to an EPContext ONNX at *dst* using QNN HTP.

    Designed to run in a subprocess spawned by
    :meth:`GenaiSession._compile_stage`.  All imports are deferred to inside
    the function body so that the child process only loads what it needs.

    Args:
        src: Absolute path to the source ONNX file.
        dst: Absolute path where the compiled EPContext ONNX should be written.
        qnn_options: QNN provider options forwarded verbatim from
            ``genai_config.json`` (e.g. ``backend_path``,
            ``htp_performance_mode``, ``soc_model``).
    """
    import onnxruntime as ort

    from .session.ep_registry import WinMLEPRegistry
    from .winml import add_ep_for_device

    registry = WinMLEPRegistry.get_instance()
    registry.register_execution_providers()
    so = ort.SessionOptions()
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", dst)
    add_ep_for_device(so, "QNNExecutionProvider", ort.OrtHardwareDeviceType.NPU, qnn_options)
    mc = ort.ModelCompiler(so, src, embed_compiled_data_into_model=False)
    mc.compile_to_file(dst)
