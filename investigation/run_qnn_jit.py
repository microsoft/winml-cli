"""Phase Q2 — Uncompiled model + LoRA-as-initializers, JIT compiled by QNN NPU.

Answers Q2 of docs/lora_qnn_investigation.md:
  Q2.a  JIT QNN bakes the LoRA branch (A,B initializers) into the EPContext;
        output matches the CPU reference (loosened for NPU/HTP fp16).
  Q2.b  Switching adapter (different initializer values) needs a fresh compile;
        there is no in-session swap path. Timing is load-bearing.

Targets the QNN NPU device (per user request) via the WinML registration in
qnn_common. Emits *_ctx.onnx AOT dumps used by Phase Q1.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import onnxruntime as ort

from inspect_graph import inspect_model
from qnn_common import MODELS, make_qnn_session
from run_cpu import X0

NPU = ort.OrtHardwareDeviceType.NPU


def _jit_and_dump(src_name: str, ctx_name: str) -> tuple[np.ndarray, float, dict]:
    """JIT-compile src on QNN NPU, dump EPContext, return (y, compile_s, info)."""
    src = MODELS / src_name
    ctx = MODELS / ctx_name
    if ctx.exists():
        ctx.unlink()
    cfg = {
        "ep.context_enable": "1",
        "ep.context_file_path": str(ctx),
        "ep.context_embed_mode": "1",
    }
    t0 = time.perf_counter()
    sess = make_qnn_session(src, device_type=NPU, session_config=cfg)
    t_compile = time.perf_counter() - t0
    y = sess.run(None, {"x": X0})[0]
    del sess
    info = inspect_model(ctx) if ctx.exists() else {"error": "no ctx dumped"}
    return y, t_compile, info


def run_q2a() -> dict:
    refs = np.load(MODELS / "cpu_refs.npz")
    results: dict = {"node_counts": {}, "outputs": {}, "compile_times_s": {}}

    for src, ctx, key in [
        ("base.onnx", "base_ctx.onnx", "base"),
        ("baked.onnx", "baked_ctx.onnx", "baked"),
        ("baked2.onnx", "baked2_ctx.onnx", "baked2"),
    ]:
        y, tc, info = _jit_and_dump(src, ctx)
        results["outputs"][key] = y
        results["compile_times_s"][key] = tc
        results["node_counts"][ctx.replace(".onnx", "")] = {
            "EPContext": info.get("EPContext"),
            "MatMul": info.get("MatMul"),
            "ctx_inputs": info.get("ctx_inputs"),
            "initializers": info.get("initializers"),
            "op_counts": info.get("op_counts"),
        }
        print(f"[Q2.a] {src}: compile={tc*1000:.1f}ms  y={y.ravel()}")
        print(f"        ctx structure: {results['node_counts'][ctx.replace('.onnx','')]}")

    yb, yb2, ybase = (
        results["outputs"]["baked"],
        results["outputs"]["baked2"],
        results["outputs"]["base"],
    )
    err = {
        "qnn_base_vs_cpu_base": float(np.max(np.abs(ybase - refs["y_cpu_base"]))),
        "qnn_baked_vs_cpu_baked": float(np.max(np.abs(yb - refs["y_cpu_baked"]))),
        "qnn_baked2_vs_cpu_baked2": float(np.max(np.abs(yb2 - refs["y_cpu_baked2"]))),
        "qnn_baked_vs_qnn_baked2": float(np.max(np.abs(yb - yb2))),
    }
    results["max_abs_err"] = err
    print(f"[Q2.a] max_abs_err: {err}")
    return results


def run_q2b(trials: int = 3) -> dict:
    """Time fresh QNN-NPU session builds; no EPContext caching options."""
    seq = ["baked.onnx", "baked2.onnx", "baked.onnx"]
    per_model: dict[str, list[float]] = {s: [] for s in set(seq)}
    sequence_log: list[dict] = []
    for trial in range(trials):
        for src in seq:
            t0 = time.perf_counter()
            sess = make_qnn_session(MODELS / src, device_type=NPU)
            dt = time.perf_counter() - t0
            sess.run(None, {"x": X0})
            del sess
            per_model[src].append(dt)
            sequence_log.append({"trial": trial, "src": src, "build_s": dt})
            print(f"[Q2.b] trial{trial} {src}: build={dt*1000:.1f}ms")
    medians = {s: statistics.median(v) for s, v in per_model.items()}
    print(f"[Q2.b] median build times (s): {medians}")
    return {
        "trials": trials,
        "sequence": seq,
        "per_model_build_s": per_model,
        "median_build_s": medians,
        "compile_times_ms": [round(statistics.median(per_model[s]) * 1000, 1) for s in seq],
        "sequence_log": sequence_log,
    }


def main() -> dict:
    q2a = run_q2a()
    q2b = run_q2b()
    np.savez(
        MODELS / "qnn_jit_outputs.npz",
        y_qnn_base=q2a["outputs"]["base"],
        y_qnn_baked=q2a["outputs"]["baked"],
        y_qnn_baked2=q2a["outputs"]["baked2"],
    )
    return {"q2_a": q2a, "q2_b": q2b}


if __name__ == "__main__":
    import json

    out = main()
    # Strip numpy arrays for printing.
    printable = {
        "q2_a": {k: v for k, v in out["q2_a"].items() if k != "outputs"},
        "q2_b": out["q2_b"],
    }
    print(json.dumps(printable, indent=2, default=str))
