# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared pytest fixtures and utilities for optimization tests.

This module provides common fixtures and helper functions for testing ONNX Runtime
optimizations. It uses RAW ORT API only - NO imports from winml.modelkit.optim.pipes.

Fixtures:
    sample_model: Simple ONNX model for basic testing (function-scoped)
    all_patterns_model: ONNX model with all test patterns (session-scoped)

Helper Functions:
    generate_random_inputs: Generate random inputs for model
    run_onnx_inference: Run inference on ONNX model
    verify_capability_effect: Verify optimization effect
    verify_numeric: Verify numerical equivalence between outputs
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest


if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# Path Constants
# =============================================================================

ASSETS_DIR = Path(__file__).parent / "assets"
TEMP_DIR = Path(__file__).parent.parent.parent / "temp" / "pytest" / "optim"
ORT_MODEL_PATH = TEMP_DIR / "all_patterns.onnx"

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Project root for cleanup
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Session Cleanup - Remove ONNX external data files from project root
# =============================================================================
# ONNX creates .data files with UUID names in the current working directory
# when saving models with large tensors. These need to be cleaned up.


@pytest.fixture(scope="session", autouse=True)
def cleanup_external_data_files():
    """Clean up ONNX external data files from project root after tests."""
    yield  # Run tests first

    # Clean up .data files from project root (ONNX external data)
    for data_file in PROJECT_ROOT.glob("*.data"):
        # Only delete UUID-like .data files (36 char UUID + .data)
        if len(data_file.stem) == 36 and "-" in data_file.stem:
            try:
                data_file.unlink()
            except OSError:
                pass  # Ignore errors during cleanup


# =============================================================================
# Pattern Prefixes
# =============================================================================

# Pattern prefixes from tests/optim/assets/graphpipe/generate_patterns.py
# These match the PATTERN_REGISTRY keys with f"p{idx:02d}_{name}_" format
ALL_PATTERNS = (
    # Phase 1: Core patterns (p01-p09)
    "p01_identity_",
    "p02_constfold_",
    "p03_cse_",
    "p04_convbn_",
    "p05_convaddrelu_",
    "p06_matmuladdrelu_",
    "p07_reshape_",
    "p08_biasgelu_",
    "p09_skiplayernorm_",
    # Phase 2: Activation and optimizer patterns (p10-p15)
    "p10_softmax_",
    "p11_reluclip_",
    "p12_matmulact_",
    "p13_transpose_",
    "p14_simpln_",
    "p15_reducesoftmax_",
    # Phase 3: Specialized patterns (p16-p19)
    "p16_gemmact_",
    "p17_gatherslice_",
    "p18_padconv_",
    "p19_qdqpairs_",
    # Phase 3: GEMM and Conv variant patterns (p20-p25)
    "p20_gemmsum_",
    "p21_gemmtrans_",
    "p22_convmul_",
    "p23_convadd_",
    "p24_convact_",
    "p25_convaddact_",
    # Phase 3: MatMul variant patterns (p26-p29)
    "p26_matmulbn_",
    "p27_matmulscale_",
    "p28_matmultrans_",
    "p29_dynquant_",
    # Phase 4: GELU variant patterns (p30-p33)
    "p30_fastgelu_",
    "p31_quickgelu_",
    "p32_geluapprox_",
    "p33_biasdropout_",
    # Phase 4: Layout patterns (p34)
    "p34_nchwc_",
    # Phase 4: Misc patterns (p35-p36)
    "p35_notwhere_",
    "p36_noop_",
    # Phase 4: Attention patterns (p37-p40)
    "p37_attention_",
    "p38_mha_",
    "p39_rotary_",
    "p40_biasskiln_",
    # Phase 5: Gather and Slice patterns (p41-p42)
    "p41_gathersplit_",
    "p42_concatslice_",
    # Phase 6: Elimination patterns (p43-p46)
    "p43_sliceelim_",
    "p44_unsqueezeelim_",
    "p45_reshapeelim_",
    "p46_concatsliceelim_",
    # Phase 7: Special patterns - REMOVED from GraphPipe
    # EmbedLayerNormFusion removed from GraphPipe, uses FusionPipe instead.
)

# =============================================================================
# Core Optimization Function (RAW ORT)
# =============================================================================


def optimize_at_level(
    model: onnx.ModelProto,
    level: int,
    disabled_optimizers: Sequence[str] | None = None,
) -> onnx.ModelProto:
    """Optimize ONNX model at specified level using RAW ORT API.

    This function uses ONNX Runtime's file-based optimization API directly,
    without any modelkit.optim.pipes imports.

    Args:
        model: Input ONNX model to optimize.
        level: Optimization level (0=disable, 1=basic, 2=extended, 99=all).
        disabled_optimizers: List of optimizer names to disable.

    Returns:
        Optimized ONNX model.
    """
    # Create temporary files for input and output
    with (
        tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as input_file,
        tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as output_file,
    ):
        input_path = Path(input_file.name)
        output_path = Path(output_file.name)

    try:
        # Save input model
        onnx.save(model, str(input_path))

        # Configure session options with optimization level
        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = str(output_path)

        # Map level to correct ORT constant
        level_names = {
            0: "ORT_DISABLE_ALL",
            1: "ORT_ENABLE_BASIC",
            2: "ORT_ENABLE_EXTENDED",
            99: "ORT_ENABLE_ALL",
        }
        level_name = level_names.get(level, "ORT_ENABLE_EXTENDED")
        sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level_name)

        # Disable specific optimizers if requested
        # ORT uses semicolon ";" as separator for disabled_optimizers
        if disabled_optimizers:
            sess_options.add_session_config_entry(
                "optimization.disable_specified_optimizers",
                ";".join(disabled_optimizers),
            )

        # Create session to trigger optimization
        _ = ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])

        # Load and return optimized model
        return onnx.load(str(output_path))

    finally:
        # Clean up temporary files
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def sample_model() -> onnx.ModelProto:
    """Create a simple ONNX model for testing.

    Creates a minimal valid ONNX model dynamically using code,
    following Cardinal Rule #2 (no hardcoded results).

    Returns:
        A simple ONNX model with a single Add operation.
    """
    from onnx import TensorProto, helper

    # Create a simple Add operation: output = input1 + input2
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    add_node = helper.make_node(
        "Add",
        inputs=["input1", "input2"],
        outputs=["output"],
        name="add_node",
    )

    graph = helper.make_graph(
        nodes=[add_node],
        name="test_graph",
        inputs=[input1, input2],
        outputs=[output],
    )

    # Use opset 17 and set IR version 8 for ORT compatibility
    model = helper.make_model(
        graph, producer_name="test", opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8  # ORT 1.23.2 supports IR version up to 11
    return model


@pytest.fixture(scope="session")
def all_patterns_model() -> onnx.ModelProto:
    """Load or generate ONNX model with all test patterns.

    This fixture generates a comprehensive test model containing all optimization
    patterns once per test module and caches it.

    Returns:
        ONNX model with all test patterns.
    """
    # Import here to avoid circular dependencies
    from .assets.graphpipe.generate_patterns import create_all_patterns_model

    # Generate or load cached model
    if not ORT_MODEL_PATH.exists():
        model = create_all_patterns_model()
        onnx.save(model, str(ORT_MODEL_PATH))
    else:
        model = onnx.load(str(ORT_MODEL_PATH))

    return model


# =============================================================================
# Helper Functions - Verification
# =============================================================================


def generate_random_inputs(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Generate random inputs based on model input specifications.

    Args:
        model: ONNX model to generate inputs for.

    Returns:
        Dictionary mapping input names to random numpy arrays.
    """
    inputs: dict[str, np.ndarray] = {}
    initializer_names = {init.name for init in model.graph.initializer}

    for inp in model.graph.input:
        # Skip initializers (they're not runtime inputs)
        if inp.name in initializer_names:
            continue

        shape: list[int] = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # Use 1 for dynamic dims

        dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
        inputs[inp.name] = np.random.randn(*shape).astype(dtype)

    return inputs


def run_onnx_inference(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run ONNX model inference and return outputs.

    Args:
        model: ONNX model to run.
        inputs: Dictionary mapping input names to numpy arrays.

    Returns:
        Dictionary mapping output names to numpy arrays.
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = Path(f.name)
        onnx.save(model, str(temp_path))

    try:
        session = ort.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
        output_names = [out.name for out in session.get_outputs()]
        results = session.run(output_names, inputs)
        return dict(zip(output_names, results, strict=True))
    finally:
        temp_path.unlink(missing_ok=True)


def verify_capability_effect(
    model_before: onnx.ModelProto,
    model_after: onnx.ModelProto,
    existence_list: Sequence[str],
    non_existence_list: Sequence[str],
    min_node_reduction: int = 0,
    verify_numeric: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Verify capability effect using 4-criteria system.

    This function validates that an optimization capability had the expected
    effect on the model, using differential testing between baseline and
    optimized models.

    4-Criteria Verification (from design doc Section 9.4):
        1. Node reduction: optimized model should have fewer nodes
        2. Target effect: specific fused ops MUST exist (existence_list)
        3. Isolation: other fused ops MUST NOT exist (non_existence_list)
        4. Numeric verification: outputs must match within tolerance

    Args:
        model_before: Baseline model (with target capability disabled).
        model_after: Model after optimization with target capability enabled.
        existence_list: Fused ops that MUST exist (Criterion 2).
        non_existence_list: Other fused ops that MUST NOT exist (Criterion 3).
        min_node_reduction: Minimum expected node reduction (Criterion 1).
        verify_numeric: If True, run inference and compare outputs (Criterion 4).
        rtol: Relative tolerance for numeric comparison (default 1e-5).
        atol: Absolute tolerance for numeric comparison (default 1e-5).

    Raises:
        AssertionError: If any criterion fails, with full report of all criteria.

    Example:
        >>> verify_capability_effect(
        ...     model_before=baseline,
        ...     model_after=optimized,
        ...     existence_list=["Gelu"],
        ...     non_existence_list=["BiasGelu", "FastGelu"],
        ...     min_node_reduction=1,
        ...     verify_numeric=True,
        ... )
    """
    # Collect all criteria results before asserting
    failures: list[str] = []
    results: list[str] = []

    # Criterion 1: Node reduction
    nodes_before = len(model_before.graph.node)
    nodes_after = len(model_after.graph.node)
    reduction = nodes_before - nodes_after

    if reduction >= min_node_reduction:
        results.append(
            f"  [PASS] Criterion 1: Node reduction {reduction} >= {min_node_reduction} "
            f"(Nodes: {nodes_before} -> {nodes_after})"
        )
    else:
        msg = (
            f"  [FAIL] Criterion 1: Node reduction {reduction} < {min_node_reduction} "
            f"(Nodes: {nodes_before} -> {nodes_after})"
        )
        results.append(msg)
        failures.append(msg)

    # Get all op types in both models
    ops_before = {node.op_type for node in model_before.graph.node}
    ops_after = {node.op_type for node in model_after.graph.node}
    new_ops = ops_after - ops_before

    # Criterion 2: Target effect - existence check
    found_ops = [op for op in existence_list if op in ops_after]
    missing_ops = [op for op in existence_list if op not in ops_after]

    if not missing_ops:
        if existence_list:
            results.append(f"  [PASS] Criterion 2: Found expected ops {found_ops}")
        else:
            results.append("  [PASS] Criterion 2: No existence check required")
    else:
        msg = f"  [FAIL] Criterion 2: Missing expected ops {missing_ops}"
        results.append(msg)
        failures.append(msg)

    # Criterion 3: Isolation - check for NEW fused ops that shouldn't be added
    unexpected_ops = [op for op in non_existence_list if op in new_ops]

    if not unexpected_ops:
        if non_existence_list:
            results.append("  [PASS] Criterion 3: No unexpected fused ops found")
        else:
            results.append("  [PASS] Criterion 3: No isolation check required")
    else:
        msg = f"  [FAIL] Criterion 3: Unexpected NEW ops {unexpected_ops} (isolation failure)"
        results.append(msg)
        failures.append(msg)

    # Criterion 4: Numeric verification - outputs must match within tolerance
    if verify_numeric:
        numeric_errors: list[str] = []

        try:
            # Generate random inputs and run inference internally
            inputs = generate_random_inputs(model_before)
            original_outputs = run_onnx_inference(model_before, inputs)
            optimized_outputs = run_onnx_inference(model_after, inputs)

            # Check output names match
            if set(original_outputs.keys()) != set(optimized_outputs.keys()):
                numeric_errors.append(
                    f"Output names mismatch: {set(original_outputs.keys())} vs "
                    f"{set(optimized_outputs.keys())}"
                )
            else:
                for name in original_outputs:
                    orig = original_outputs[name]
                    opt = optimized_outputs[name]

                    # Check shapes match
                    if orig.shape != opt.shape:
                        numeric_errors.append(
                            f"Shape mismatch for '{name}': {orig.shape} vs {opt.shape}"
                        )
                    elif not np.allclose(orig, opt, rtol=rtol, atol=atol):
                        max_diff = float(np.max(np.abs(orig - opt)))
                        numeric_errors.append(
                            f"Numeric mismatch for '{name}': max_diff={max_diff:.2e} "
                            f"(rtol={rtol}, atol={atol})"
                        )

            if not numeric_errors:
                results.append(
                    f"  [PASS] Criterion 4: Numeric verification passed "
                    f"({len(original_outputs)} outputs, rtol={rtol}, atol={atol})"
                )
            else:
                msg = f"  [FAIL] Criterion 4: Numeric verification failed: {numeric_errors}"
                results.append(msg)
                failures.append(msg)
        except Exception as e:
            msg = f"  [FAIL] Criterion 4: Inference error: {e}"
            results.append(msg)
            failures.append(msg)
    else:
        results.append("  [SKIP] Criterion 4: Numeric verification not requested")

    # Build report - only show violations, not all ops
    if failures:
        report_lines = [
            "",
            "=" * 60,
            "4-CRITERIA VERIFICATION REPORT",
            "=" * 60,
            *results,
            "-" * 60,
        ]
        # Only show violations if there are isolation failures
        if unexpected_ops:
            report_lines.append(f"Unexpected NEW ops: {sorted(unexpected_ops)}")
        report_lines.append("=" * 60)
        raise AssertionError("\n".join(report_lines))


def verify_numeric(
    original_outputs: dict[str, np.ndarray],
    optimized_outputs: dict[str, np.ndarray],
    output_name: str | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Verify numerical equivalence between original and optimized outputs.

    Args:
        original_outputs: Original model outputs.
        optimized_outputs: Optimized model outputs.
        output_name: Specific output to verify (None = verify all).
        rtol: Relative tolerance for np.allclose.
        atol: Absolute tolerance for np.allclose.

    Raises:
        AssertionError: If outputs don't match within tolerance.
    """
    if output_name is not None:
        # Verify specific output
        assert output_name in original_outputs, f"Output '{output_name}' not in original outputs"
        assert output_name in optimized_outputs, f"Output '{output_name}' not in optimized outputs"

        original = original_outputs[output_name]
        optimized = optimized_outputs[output_name]

        assert original.shape == optimized.shape, (
            f"Shape mismatch for '{output_name}': {original.shape} vs {optimized.shape}"
        )

        assert np.allclose(original, optimized, rtol=rtol, atol=atol), (
            f"Numerical mismatch for '{output_name}': "
            f"max_diff={np.max(np.abs(original - optimized))}"
        )
    else:
        # Verify all outputs
        assert set(original_outputs.keys()) == set(optimized_outputs.keys()), "Output name mismatch"

        for name in original_outputs:
            verify_numeric(original_outputs, optimized_outputs, name, rtol, atol)
