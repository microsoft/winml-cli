"""Shared ONNX graph inspection helper (Section 4.x)."""

from __future__ import annotations

from pathlib import Path

import onnx


def inspect_model(path: str | Path) -> dict:
    """Return op-type counts, EPContext input lists, and initializer names."""
    m = onnx.load(str(path))
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    ep_ctx = [n for n in m.graph.node if n.op_type == "EPContext"]
    info = {
        "path": str(path),
        "op_counts": ops,
        "EPContext": ops.get("EPContext", 0),
        "MatMul": ops.get("MatMul", 0),
        "ctx_inputs": [list(n.input) for n in ep_ctx],
        "initializers": [t.name for t in m.graph.initializer],
        "graph_inputs": [i.name for i in m.graph.input],
        "graph_outputs": [o.name for o in m.graph.output],
    }
    return info


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(inspect_model(sys.argv[1]), indent=2))
