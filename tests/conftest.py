# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared pytest fixtures for all tests.

This conftest.py provides common fixtures used across multiple test modules.
Fixtures are organized by scope for optimal performance.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto


# =============================================================================
# WINML SDK INITIALIZATION GUARD
# =============================================================================
# winml.modelkit.models.winml.base imports WinMLSession at module level,
# which triggers WinMLEPRegistry._discover_eps() → WinML SDK runtime init.
# This can hang on CI environments without the SDK installed.
# Mock it only for unit tests; other test groups use real initialization.


@pytest.fixture(autouse=True)
def _reset_telemetry_singleton():
    """Reset the Telemetry singleton between tests so each starts clean.

    Lives in the top-level ``conftest.py`` (rather than one per telemetry
    test package) so the singleton can never leak between tests, and so
    the fixture is not duplicated. No-op for tests that never touch
    Telemetry — ``_INSTANCE`` is ``None`` and the function returns
    immediately.

    Calls ``shutdown()`` on any pre-existing instance so a real
    ``BatchLogRecordProcessor`` thread (created when a test exercises
    the real LoggerProvider path) does not leak across tests.
    """
    from winml.modelkit.telemetry import telemetry as telemetry_mod

    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Best-effort cleanup: a half-initialized singleton from a
            # prior test must not block resetting state for this test.
            pass
    telemetry_mod._INSTANCE = None
    yield
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Same rationale as above; teardown must always reach the
            # _INSTANCE = None reset below.
            pass
    telemetry_mod._INSTANCE = None


@pytest.fixture(autouse=True)
def _skip_winml_ep_init(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock WinML EP initialization for unit tests only."""
    nodeid = request.node.nodeid.replace("\\", "/")
    if not nodeid.startswith("tests/unit/"):
        return
    try:
        from winml.modelkit.session import ep_registry as ep_registry_mod

        def _mock_discover_eps(registry) -> None:
            registry._ep_paths = {}
            registry._registered_eps = []
            registry._winml_available = False
            registry._catalog = None

        monkeypatch.setattr(ep_registry_mod, "_winml_ep_registry", None)
        monkeypatch.setattr(
            ep_registry_mod.WinMLEPRegistry,
            "_discover_eps",
            _mock_discover_eps,
        )
    except ImportError as e:
        import warnings

        warnings.warn(f"Could not mock WinMLEPRegistry discovery: {e}", stacklevel=2)

    try:
        monkeypatch.setattr(
            "winml.modelkit.session.session.WinMLSession._init_winml_eps_once",
            classmethod(lambda cls: None),
        )
    except ImportError as e:
        import warnings

        warnings.warn(f"Could not mock _init_winml_eps_once: {e}", stacklevel=2)

    try:
        monkeypatch.setattr(
            "winml.modelkit.analyze.core.runtime_checker_query.RuntimeCheckerQuery._is_ep_available_locally",
            lambda self: False,
        )
    except ImportError as e:
        import warnings

        warnings.warn(f"Could not mock _is_ep_available_locally: {e}", stacklevel=2)


# =============================================================================
# ORT GRAPH OPTIMIZATION TEST FIXTURES (Module-scoped)
# =============================================================================

ORT_MODEL_PATH = (
    Path(__file__).parent.parent
    / "temp"
    / "ort_test_patterns"
    / "ort_graph_optim_all_patterns.onnx"
)


@pytest.fixture(scope="module")
def ort_original_model() -> onnx.ModelProto:
    """Load original unoptimized ORT test model (shared across module)."""
    if not ORT_MODEL_PATH.exists():
        pytest.skip(f"ORT test model not found at {ORT_MODEL_PATH}")
    return onnx.load(str(ORT_MODEL_PATH))


@pytest.fixture(scope="module")
def ort_optimized_model(ort_original_model: onnx.ModelProto) -> onnx.ModelProto:
    """Optimize model with ORT level 2 (shared across module)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.onnx"
        output_path = Path(tmpdir) / "output.onnx"

        onnx.save(ort_original_model, str(input_path))

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_opts.optimized_model_filepath = str(output_path)

        _ = ort.InferenceSession(str(input_path), sess_opts, providers=["CPUExecutionProvider"])

        return onnx.load(str(output_path))


@pytest.fixture(scope="module")
def ort_model_inputs(ort_original_model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Generate random inputs for ORT model inference."""
    rng = np.random.RandomState(42)
    inputs = {}

    for inp in ort_original_model.graph.input:
        name = inp.name
        if any(init.name == name for init in ort_original_model.graph.initializer):
            continue

        shape = [
            dim.dim_value if dim.dim_value > 0 else 1 for dim in inp.type.tensor_type.shape.dim
        ]

        dtype = inp.type.tensor_type.elem_type
        if dtype == TensorProto.FLOAT:
            inputs[name] = rng.randn(*shape).astype(np.float32)
        elif dtype == TensorProto.INT64:
            inputs[name] = rng.randint(0, 10, size=shape).astype(np.int64)
        else:
            inputs[name] = rng.randn(*shape).astype(np.float32)

    return inputs


@pytest.fixture(scope="module")
def ort_original_outputs(
    ort_original_model: onnx.ModelProto, ort_model_inputs: dict
) -> dict[str, np.ndarray]:
    """Run inference on original ORT model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(ort_original_model, str(model_path))

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            str(model_path), sess_opts, providers=["CPUExecutionProvider"]
        )

        outputs = session.run(None, ort_model_inputs)
        output_names = [o.name for o in session.get_outputs()]

        return dict(zip(output_names, outputs, strict=True))


@pytest.fixture(scope="module")
def ort_optimized_outputs(
    ort_optimized_model: onnx.ModelProto, ort_model_inputs: dict
) -> dict[str, np.ndarray]:
    """Run inference on optimized ORT model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(ort_optimized_model, str(model_path))

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        session = ort.InferenceSession(
            str(model_path), sess_opts, providers=["CPUExecutionProvider"]
        )

        outputs = session.run(None, ort_model_inputs)
        output_names = [o.name for o in session.get_outputs()]

        return dict(zip(output_names, outputs, strict=True))


# =============================================================================
# ORT HELPER FUNCTIONS (available as module-level utilities)
# =============================================================================


def get_op_counts(model: onnx.ModelProto) -> dict[str, int]:
    """Get operation type counts from model."""
    return dict(Counter(node.op_type for node in model.graph.node))


def get_nodes_by_prefix(model: onnx.ModelProto, prefix: str) -> list[onnx.NodeProto]:
    """Get all nodes matching a prefix."""
    return [node for node in model.graph.node if node.name.startswith(prefix)]


def get_op_counts_by_prefix(model: onnx.ModelProto, prefix: str) -> dict[str, int]:
    """Get op type counts for nodes matching a prefix."""
    ops = [node.op_type for node in model.graph.node if node.name.startswith(prefix)]
    return dict(Counter(ops))


def verify_instance_fusion(
    original: onnx.ModelProto,
    optimized: onnx.ModelProto,
    prefix: str,
    expected_before: int,
    expected_after: int,
) -> None:
    """Verify a specific instance was correctly fused."""
    before_count = len(get_nodes_by_prefix(original, prefix))
    after_count = len(get_nodes_by_prefix(optimized, prefix))

    assert before_count == expected_before, (
        f"Instance {prefix}: Expected {expected_before} nodes before, got {before_count}"
    )
    assert after_count <= expected_after, (
        f"Instance {prefix}: Expected at most {expected_after} nodes after, got {after_count}"
    )


def verify_numeric_output(
    original_out: np.ndarray,
    optimized_out: np.ndarray,
    output_name: str,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Verify outputs match within tolerance."""
    assert original_out.shape == optimized_out.shape, (
        f"Output {output_name}: Shape mismatch {original_out.shape} vs {optimized_out.shape}"
    )
    assert np.allclose(original_out, optimized_out, rtol=rtol, atol=atol), (
        f"Output {output_name}: Values differ beyond tolerance "
        f"(max_diff={np.max(np.abs(original_out - optimized_out)):.2e})"
    )
