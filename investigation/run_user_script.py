"""Phase Q3 — Graph-input LoRA + the Appendix A user script.

Answers Q3 of docs/lora_qnn_investigation.md: when A/B are real ONNX graph
inputs, a normal user can feed adapter values from a safetensors file at
inference time on both CPU and QNN NPU, and swapping adapters inside one
session triggers no recompile.

This exercises the VERBATIM Appendix A loader (inference/lora_loader.py) — the
full regex + transpose + scaling path — by building a PEFT-named switchable
model and PEFT-style adapter files. Targets the QNN NPU device.

  Q3.a  build switchable_peft.onnx + adapter.safetensors + adapter2.safetensors
  Q3.b  Appendix A user call on CPU EP            -> matches CPU baked refs
  Q3.c  same call on QNN NPU (JIT, in-session swap) + timings (median of >=3)
  Q3.d  AOT EPContext dump, fresh session, same swap still works
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file

from inspect_graph import inspect_model
from qnn_common import MODELS, ROOT, cpu_session, make_qnn_session
from run_cpu import X0

sys.path.insert(0, str(ROOT))
from inference.lora_loader import load_adapter  # noqa: E402

NPU = ort.OrtHardwareDeviceType.NPU
LORA_A = "layer0.weight_lora_A"
LORA_B = "layer0.weight_lora_B"


# ---------------------------------------------------------------- Q3.a
def build_switchable_peft() -> Path:
    w = np.load(MODELS / "weights.npz")
    W = w["W"]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    a = helper.make_tensor_value_info(LORA_A, TensorProto.FLOAT, [8, 4])
    b = helper.make_tensor_value_info(LORA_B, TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    nodes = [
        helper.make_node("MatMul", ["x", LORA_A], ["t0"]),
        helper.make_node("MatMul", ["t0", LORA_B], ["t1"]),
        helper.make_node("MatMul", ["x", "W"], ["t2"]),
        helper.make_node("Add", ["t2", "t1"], ["y"]),
    ]
    g = helper.make_graph(nodes, "switchable_peft", [x, a, b], [y],
                          initializer=[numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    path = MODELS / "switchable_peft.onnx"
    onnx.save(m, str(path))
    print(f"[Q3.a] wrote {path}")
    return path


def build_peft_adapter(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    """Write a PEFT-style safetensors so the verbatim loader yields A,B exactly.

    PEFT stores lora_A:[r,in], lora_B:[out,r]; loader transposes back to ONNX
    [in,r]/[r,out]. With alpha==rank the folded scaling is 1.0.
    """
    tensors = {
        "base_model.model.layer0.lora_A.weight": a.T.copy(),   # [r,in]
        "base_model.model.layer0.lora_B.weight": b.T.copy(),   # [out,r]
    }
    save_file(tensors, str(path))
    print(f"[Q3.a] wrote {path}")


def q3a() -> Path:
    path = build_switchable_peft()
    w = np.load(MODELS / "weights.npz")
    build_peft_adapter(w["A"], w["B"], MODELS / "adapter.safetensors")
    build_peft_adapter(w["A2"], w["B2"], MODELS / "adapter2.safetensors")
    (MODELS / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4, "r": 4}))
    return path


# ---------------------------------------------------------------- Q3.b
def q3b(model_path: Path) -> dict:
    refs = np.load(MODELS / "cpu_refs.npz")
    sess = cpu_session(model_path)
    names = [i.name for i in sess.get_inputs()]

    lora1 = load_adapter(MODELS / "adapter.safetensors", names)
    lora2 = load_adapter(MODELS / "adapter2.safetensors", names)
    print(f"[Q3.b] loader produced feeds for: {sorted(lora1)}")
    assert set(lora1) == {LORA_A, LORA_B}, f"loader matched wrong inputs: {lora1.keys()}"

    y1 = sess.run(None, {"x": X0, **lora1})[0]
    y2 = sess.run(None, {"x": X0, **lora2})[0]
    e1 = float(np.max(np.abs(y1 - refs["y_cpu_baked"])))
    e2 = float(np.max(np.abs(y2 - refs["y_cpu_baked2"])))
    print(f"[Q3.b] CPU adapter1 vs cpu_baked max_err={e1}; adapter2 vs cpu_baked2={e2}")
    assert np.allclose(y1, refs["y_cpu_baked"], atol=1e-5)
    assert np.allclose(y2, refs["y_cpu_baked2"], atol=1e-5)
    return {"cpu_adapter1_vs_cpu_baked": e1, "cpu_adapter2_vs_cpu_baked2": e2}


# ---------------------------------------------------------------- Q3.c
def _median_run(sess, feeds, trials=3):
    dts = []
    last = None
    for _ in range(trials):
        t0 = time.perf_counter()
        last = sess.run(None, feeds)[0]
        dts.append(time.perf_counter() - t0)
    return last, statistics.median(dts), dts


def q3c(model_path: Path, *, aot: bool = False) -> dict:
    refs = np.load(MODELS / "cpu_refs.npz")
    cfg = None
    if aot:
        ctx = MODELS / "switchable_ctx.onnx"
        if ctx.exists():
            ctx.unlink()
        cfg = {"ep.context_enable": "1", "ep.context_file_path": str(ctx),
               "ep.context_embed_mode": "1"}

    t0 = time.perf_counter()
    sess = make_qnn_session(model_path, device_type=NPU, session_config=cfg)
    t_compile = time.perf_counter() - t0
    names = [i.name for i in sess.get_inputs()]

    lora1 = load_adapter(MODELS / "adapter.safetensors", names)
    lora2 = load_adapter(MODELS / "adapter2.safetensors", names)
    base = {"x": X0}

    y1, dt1, dts1 = _median_run(sess, {**base, **lora1})
    y2, dt2, dts2 = _median_run(sess, {**base, **lora2})
    y3, dt3, dts3 = _median_run(sess, {**base, **lora1})  # round-trip

    e1 = float(np.max(np.abs(y1 - refs["y_cpu_baked"])))
    e2 = float(np.max(np.abs(y2 - refs["y_cpu_baked2"])))
    e3 = float(np.max(np.abs(y3 - refs["y_cpu_baked"])))
    swap_differs = float(np.max(np.abs(y1 - y2)))

    tag = "Q3.d(AOT)" if aot else "Q3.c(JIT)"
    print(f"[{tag}] t_compile={t_compile*1000:.1f}ms  "
          f"run medians ms: a1={dt1*1000:.3f} a2={dt2*1000:.3f} a1again={dt3*1000:.3f}")
    print(f"[{tag}] max_err a1={e1:.4g} a2={e2:.4g} a1again={e3:.4g} swap_differs={swap_differs:.4g}")

    out = {
        "max_abs_err": {
            "qnn_adapter1_vs_cpu_baked": e1,
            "qnn_adapter2_vs_cpu_baked2": e2,
            "qnn_adapter1again_vs_cpu_baked": e3,
            "qnn_adapter1_vs_adapter2": swap_differs,
        },
        "timings_ms": {
            "session_build": t_compile * 1000,
            "run_adapter1": dt1 * 1000,
            "run_adapter2": dt2 * 1000,
            "run_adapter1_again": dt3 * 1000,
            "run_adapter1_trials": [d * 1000 for d in dts1],
            "run_adapter2_trials": [d * 1000 for d in dts2],
            "run_adapter1_again_trials": [d * 1000 for d in dts3],
        },
        "no_recompile_on_swap": (max(dt1, dt2, dt3) < 0.1 * t_compile),
        "compile_dominates": (t_compile > 10 * max(dt1, dt2, dt3)),
    }
    if aot:
        info = inspect_model(MODELS / "switchable_ctx.onnx")
        out["ctx_structure"] = {
            "EPContext": info["EPContext"], "MatMul": info["MatMul"],
            "ctx_inputs": info["ctx_inputs"], "initializers": info["initializers"],
        }
        print(f"[{tag}] ctx structure: {out['ctx_structure']}")
    return out


def main() -> dict:
    model_path = q3a()
    q3_b = q3b(model_path)
    q3_c = q3c(model_path, aot=False)
    q3_d = q3c(model_path, aot=True)
    return {"q3_b": q3_b, "q3_c": q3_c, "q3_d": q3_d, "lora_inputs": [LORA_A, LORA_B]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
