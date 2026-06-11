"""Step 2 — CPU EP reference outputs for all toy models.

Establishes the ground-truth tensors every QNN phase compares against.
See Section 4 / Step 2 of docs/lora_qnn_investigation.md.
"""

from __future__ import annotations

import numpy as np

from qnn_common import MODELS, cpu_session

# Fixed deterministic input.
X0 = (np.arange(8, dtype=np.float32) / 8.0).reshape(1, 8)


def main() -> dict[str, np.ndarray]:
    w = np.load(MODELS / "weights.npz")
    A, B = w["A"], w["B"]

    y_cpu_base = cpu_session(MODELS / "base.onnx").run(None, {"x": X0})[0]
    y_cpu_baked = cpu_session(MODELS / "baked.onnx").run(None, {"x": X0})[0]
    y_cpu_baked2 = cpu_session(MODELS / "baked2.onnx").run(None, {"x": X0})[0]
    y_cpu_switch = cpu_session(MODELS / "switchable.onnx").run(
        None, {"x": X0, "A": A, "B": B}
    )[0]

    # Assertions (Step 2.5).
    assert np.allclose(y_cpu_baked, y_cpu_switch, atol=1e-6), \
        "baked vs switchable must be mathematically identical"
    assert not np.allclose(y_cpu_base, y_cpu_baked, atol=1e-3), \
        "LoRA must change the output vs base"
    assert not np.allclose(y_cpu_baked, y_cpu_baked2, atol=1e-3), \
        "second adapter must produce a different output"

    out = {
        "y_cpu_base": y_cpu_base,
        "y_cpu_baked": y_cpu_baked,
        "y_cpu_baked2": y_cpu_baked2,
        "y_cpu_switch": y_cpu_switch,
    }
    np.savez(MODELS / "cpu_refs.npz", **out)
    print("[cpu] references computed and asserted:")
    for k, v in out.items():
        print(f"  {k} = {v.ravel()}")
    print(f"[cpu] saved {MODELS / 'cpu_refs.npz'}")
    return out


if __name__ == "__main__":
    main()
