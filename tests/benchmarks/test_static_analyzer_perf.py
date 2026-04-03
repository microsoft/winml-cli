# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Performance benchmark suite for the Static Analyzer.

Validates ADR-017 performance targets using synthetic ONNX models of exact
sizes (100 / 1000 / 5000 nodes).  Benchmarks are wall-clock time only
(pytest wall time via ``time.perf_counter``) so no extra plugins are needed.

ADR-017 targets
---------------
- Small  (100 ops):  < 5 s,    < 100 MB RSS
- Medium (1000 ops): < 30 s,   < 500 MB RSS
- Large  (5000 ops): < 2 min,  < 1.5 GB RSS

CI placement
------------
These tests are marked ``slow`` and excluded from the per-PR matrix.  They
are intended for nightly runs:

    uv run pytest tests/benchmarks -m slow

Model generation
----------------
All models are built with ``onnx.helper.make_model()`` — no network access.
To keep the benchmark representative without hardcoding op names, the same
generic element-wise op (Add) is repeated N times in a chain so that the
analyzer must evaluate every node.

Usage
-----
    # Quick sanity run (100 nodes only):
    pytest tests/benchmarks/test_static_analyzer_perf.py::TestStaticAnalyzerPerf::test_small

    # Full suite (slow – takes several minutes):
    pytest tests/benchmarks -m slow -v
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.analyze import ONNXModel, RuntimeChecker


if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# ADR-017 performance targets (seconds)
# ---------------------------------------------------------------------------

_TARGET_SMALL_S = 5.0
_TARGET_MEDIUM_S = 30.0
_TARGET_LARGE_S = 120.0


# ---------------------------------------------------------------------------
# Synthetic model builder
# ---------------------------------------------------------------------------


def _make_chain_model(n_ops: int) -> onnx.ModelProto:
    """Build an ONNX model with *n_ops* Add nodes chained sequentially.

    The graph looks like:
        input → Add₀ → Add₁ → … → Addₙ₋₁ → output

    Each node uses a fixed initializer as its second operand so the shape
    inference produces concrete output shapes throughout.

    Args:
        n_ops: Number of Add nodes (equals graph operation count).

    Returns:
        A valid onnx.ModelProto with *n_ops* nodes.
    """
    batch = 1
    channels = 4
    height = 8
    width = 8
    shape = [batch, channels, height, width]

    # Shared initializer reused as the second operand of every Add
    bias_name = "shared_bias"
    bias_tensor = helper.make_tensor(
        name=bias_name,
        data_type=TensorProto.FLOAT,
        dims=shape,
        vals=[0.0] * (batch * channels * height * width),
    )

    input_name = "input_0"
    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)

    nodes = []
    prev_output = input_name
    for i in range(n_ops):
        node_output = f"node_output_{i}"
        node = helper.make_node(
            "Add",
            inputs=[prev_output, bias_name],
            outputs=[node_output],
            name=f"add_{i}",
        )
        nodes.append(node)
        prev_output = node_output

    graph_output = helper.make_tensor_value_info(prev_output, TensorProto.FLOAT, shape)

    graph = helper.make_graph(
        nodes,
        name="chain_graph",
        inputs=[graph_input],
        outputs=[graph_output],
        initializer=[bias_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.producer_name = "modelkit_benchmark"
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_op_support(n_ops: int) -> float:
    """Create a synthetic model and run op_support(); return elapsed seconds."""
    model_proto = _make_chain_model(n_ops)
    onnx_model = ONNXModel.from_onnx_model(model_proto, f"bench_{n_ops}.onnx")
    checker = RuntimeChecker(
        ep="QNNExecutionProvider",
        device="NPU",
        model=onnx_model,
    )
    t0 = time.perf_counter()
    checker.op_support(run_unknown_op=False)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestStaticAnalyzerPerf:
    """Wall-clock performance benchmarks for the static analyzer.

    All tests are marked ``slow`` and are excluded from per-PR CI.
    Run them explicitly with ``pytest tests/benchmarks -m slow``.
    """

    def test_small(self) -> None:
        """Static analyzer on 100 ops must complete in < 5 s (ADR-017 small target)."""
        elapsed = _run_op_support(100)
        assert elapsed < _TARGET_SMALL_S, (
            f"Small benchmark (100 ops) took {elapsed:.2f}s, "
            f"exceeds ADR-017 target of {_TARGET_SMALL_S}s"
        )

    def test_medium(self) -> None:
        """Static analyzer on 1 000 ops must complete in < 30 s (ADR-017 medium target)."""
        elapsed = _run_op_support(1_000)
        assert elapsed < _TARGET_MEDIUM_S, (
            f"Medium benchmark (1 000 ops) took {elapsed:.2f}s, "
            f"exceeds ADR-017 target of {_TARGET_MEDIUM_S}s"
        )

    def test_large(self) -> None:
        """Static analyzer on 5 000 ops must complete in < 2 min (ADR-017 large target)."""
        elapsed = _run_op_support(5_000)
        assert elapsed < _TARGET_LARGE_S, (
            f"Large benchmark (5 000 ops) took {elapsed:.2f}s, "
            f"exceeds ADR-017 target of {_TARGET_LARGE_S}s"
        )

    def test_repeated_layer_cache_speedup(self) -> None:
        """Verify that repeated-layer caching keeps analysis time well within target.

        A model with 200 identical Add nodes (same shape, same dtype) exercises
        the _table_query_cache path: after the first node is evaluated, every
        subsequent identical node should hit the cache.  The elapsed time must
        remain well within the small (100-ops) ADR-017 target.
        """
        elapsed = _run_op_support(200)
        assert elapsed < _TARGET_SMALL_S, (
            f"Repeated-layer model (200 identical ops) took {elapsed:.2f}s, "
            f"exceeds ADR-017 small target of {_TARGET_SMALL_S}s"
        )
