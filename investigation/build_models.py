"""Step 1 — build the toy source ONNX models for the LoRA/QNN investigation.

All models are tiny fp32 (8x8) and deterministic. See Section 3 of
docs/lora_qnn_investigation.md. Produces:

  models/lora_test/base.onnx        y = x·W                       (no LoRA)
  models/lora_test/baked.onnx       y = x·W + (x·A)·B   (A,B init) (fixed LoRA)
  models/lora_test/baked2.onnx      same shape, A2,B2 init        (2nd adapter)
  models/lora_test/switchable.onnx  y = x·W + (x·A)·B   (A,B input)(graph-input)
  models/lora_test/weights.npz      raw A,B,A2,B2 for adapter.safetensors
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qnn_common import MODELS

OPSET = 17
IR_VERSION = 9

# Deterministic, seeded per Section 3.1.
W = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
A = np.random.default_rng(1).standard_normal((8, 4)).astype(np.float32)
B = np.random.default_rng(2).standard_normal((4, 8)).astype(np.float32)
# Second adapter: different seeds so its output differs end-to-end.
A2 = np.random.default_rng(11).standard_normal((8, 4)).astype(np.float32)
B2 = np.random.default_rng(12).standard_normal((4, 8)).astype(np.float32)


def _vi(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save(model: onnx.ModelProto, path) -> None:
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"[build] wrote {path}")


def build_base() -> None:
    nodes = [helper.make_node("MatMul", ["x", "W"], ["y"])]
    g = helper.make_graph(
        nodes, "base", [_vi("x", [1, 8])], [_vi("y", [1, 8])],
        initializer=[numpy_helper.from_array(W, "W")],
    )
    _save(helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)]),
          MODELS / "base.onnx")


def build_baked(a: np.ndarray, b: np.ndarray, path) -> None:
    nodes = [
        helper.make_node("MatMul", ["x", "A"], ["t0"]),
        helper.make_node("MatMul", ["t0", "B"], ["t1"]),
        helper.make_node("MatMul", ["x", "W"], ["t2"]),
        helper.make_node("Add", ["t2", "t1"], ["y"]),
    ]
    g = helper.make_graph(
        nodes, "baked_lora", [_vi("x", [1, 8])], [_vi("y", [1, 8])],
        initializer=[
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(a, "A"),
            numpy_helper.from_array(b, "B"),
        ],
    )
    _save(helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)]), path)


def build_switchable() -> None:
    nodes = [
        helper.make_node("MatMul", ["x", "A"], ["t0"]),
        helper.make_node("MatMul", ["t0", "B"], ["t1"]),
        helper.make_node("MatMul", ["x", "W"], ["t2"]),
        helper.make_node("Add", ["t2", "t1"], ["y"]),
    ]
    g = helper.make_graph(
        nodes, "switchable_lora",
        [_vi("x", [1, 8]), _vi("A", [8, 4]), _vi("B", [4, 8])],
        [_vi("y", [1, 8])],
        initializer=[numpy_helper.from_array(W, "W")],
    )
    _save(helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)]),
          MODELS / "switchable.onnx")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    build_base()
    build_baked(A, B, MODELS / "baked.onnx")
    build_baked(A2, B2, MODELS / "baked2.onnx")
    build_switchable()
    np.savez(MODELS / "weights.npz", W=W, A=A, B=B, A2=A2, B2=B2)
    print(f"[build] wrote {MODELS / 'weights.npz'}")


if __name__ == "__main__":
    main()
