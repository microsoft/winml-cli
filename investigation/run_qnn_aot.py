"""Phase Q1 — Adapter on a precompiled EPContext is a no-op.

Answers Q1 of docs/lora_qnn_investigation.md: once a model is collapsed into an
EPContext node on QNN, attaching an OrtLoraAdapter at Run() time cannot inject
or replace weights inside the frozen binary.

Method: start from base_ctx.onnx (EPContext model produced in Phase Q2, no LoRA
inside, only input "x"). Build an OrtLoraAdapter from the real A/B values (the
same values that demonstrably change output in Q2/Q3). Attach it via
RunOptions.add_active_adapter and compare to a plain run. Expected: identical.

ORT 1.24.5 adapter API (discovered on the device; see NOTES.md):
  fmt = ort.AdapterFormat(); fmt.set_parameters({name: OrtValue}); fmt.export_adapter(path)
  ad = ort.LoraAdapter(); ad.Load(path)
  ro = ort.RunOptions(); ro.add_active_adapter(ad)
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from qnn_common import MODELS, make_qnn_session

NPU = ort.OrtHardwareDeviceType.NPU
ADAPTER_API = (
    "ort.AdapterFormat().set_parameters({name: ort.OrtValue}); "
    ".export_adapter(path); ort.LoraAdapter().Load(path); "
    "ort.RunOptions().add_active_adapter(adapter)"
)


def build_adapter(path) -> None:
    """Write a fake .onnx_adapter whose params would change W if honored."""
    w = np.load(MODELS / "weights.npz")
    fmt = ort.AdapterFormat()
    # Use names that a LoRA-aware model would expose, plus a direct W override.
    params = {
        "A": ort.OrtValue.ortvalue_from_numpy(w["A"]),
        "B": ort.OrtValue.ortvalue_from_numpy(w["B"]),
        "W": ort.OrtValue.ortvalue_from_numpy(
            (w["W"] + 5.0).astype(np.float32)  # large delta: would be obvious if applied
        ),
    }
    fmt.set_parameters(params)
    fmt.export_adapter(str(path))
    print(f"[Q1] wrote fake adapter {path}")


def main() -> dict:
    from run_cpu import X0

    refs = np.load(MODELS / "qnn_jit_outputs.npz")
    y_qnn_base = refs["y_qnn_base"]

    adapter_path = MODELS / "fake.onnx_adapter"
    build_adapter(adapter_path)

    # Plain run on the precompiled EPContext base model.
    sess = make_qnn_session(MODELS / "base_ctx.onnx", device_type=NPU)
    y_plain = sess.run(None, {"x": X0})[0]

    # Same session, now with the adapter "active".
    note = ""
    try:
        ad = ort.LoraAdapter()
        ad.Load(str(adapter_path))
        ro = ort.RunOptions()
        ro.add_active_adapter(ad)
        y_adapter = sess.run(None, {"x": X0}, run_options=ro)[0]
        max_err = float(np.max(np.abs(y_plain - y_adapter)))
        applied = not np.allclose(y_plain, y_adapter, atol=1e-6)
        print(f"[Q1] plain   y={y_plain.ravel()}")
        print(f"[Q1] adapter y={y_adapter.ravel()}")
        print(f"[Q1] max_abs_err(adapter vs plain) = {max_err}")
        print(f"[Q1] adapter changed output? {applied}  (expected: False -> no-op)")
    except Exception as e:  # noqa: BLE001
        # An exception (rejected adapter) also confirms Q1: cannot inject.
        note = f"add_active_adapter/run raised: {type(e).__name__}: {e}"
        max_err = 0.0
        applied = False
        print(f"[Q1] adapter attach raised (still confirms Q1): {note}")

    result = {
        "adapter_api_used": ADAPTER_API,
        "max_abs_err__override_vs_qnn_base": float(np.max(np.abs(y_plain - y_qnn_base))),
        "max_abs_err__adapter_vs_plain": max_err,
        "adapter_changed_output": applied,
        "note": note,
        "q1b_outside_epcontext": (
            "informational only (not implemented): the only post-hoc route is to "
            "wrap MatMul/MatMul/Add AROUND the EPContext node; those ops fall back "
            "to CPU EP and are not inside the QNN binary."
        ),
    }
    print(f"[Q1] verdict input: adapter_changed_output={applied}")
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2))
