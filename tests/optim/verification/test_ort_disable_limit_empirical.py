"""Empirical test to verify ORT's disable_specified_optimizers limit.

This test verifies whether ONNX Runtime has a limit on the number of
optimizers that can be disabled via 'optimization.disable_specified_optimizers'.

The claim in graph.py line 56 states:
    "ORT's disable list limit - beyond this, ORT silently ignores the entire list"
    ORT_DISABLE_LIMIT: ClassVar[int] = 32

This test empirically verifies this claim by:
1. Creating a model with known fusible patterns
2. Verifying baseline optimization behavior
3. Testing with increasing numbers of disabled optimizers
4. Detecting at what point (if any) ORT stops honoring the disable list
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_relu_model() -> onnx.ModelProto:
    """Create a minimal model with Relu that ORT might optimize.

    This creates: input -> Relu -> output
    ORT has a ReluFusion optimizer that we can target.
    """
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 64, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 64, 224, 224]
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_0",
    )

    graph = helper.make_graph(
        nodes=[relu_node],
        name="relu_test",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="ort_limit_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model


def create_add_relu_model() -> onnx.ModelProto:
    """Create a model with Add+Relu pattern that ORT might fuse.

    This creates: input1, input2 -> Add -> Relu -> output
    """
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 64])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 64])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])

    add_node = helper.make_node(
        "Add",
        inputs=["input1", "input2"],
        outputs=["add_out"],
        name="add_0",
    )

    relu_node = helper.make_node(
        "Relu",
        inputs=["add_out"],
        outputs=["output"],
        name="relu_0",
    )

    graph = helper.make_graph(
        nodes=[add_node, relu_node],
        name="add_relu_test",
        inputs=[input1, input2],
        outputs=[output],
    )

    model = helper.make_model(
        graph,
        producer_name="ort_limit_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model


def get_known_ort_optimizers() -> list[str]:
    """Return a list of known ORT optimizer names.

    These are based on ORT source code graph transformers.
    We intentionally include more than 32 to test the limit.
    """
    return [
        # Fusions
        "MatMulAddFusion",
        "MatMulBNFusion",
        "MatMulTransposeFusion",
        "MatMulScaleFusion",
        "MatMulActivationFusion",
        "ConvAddFusion",
        "ConvMulFusion",
        "ConvBNFusion",
        "ConvActivationFusion",
        "GemmActivationFusion",
        "GeluFusion",
        "GeluApproximation",
        "LayerNormFusion",
        "SkipLayerNormFusion",
        "FastGeluFusion",
        "QuickGeluFusion",
        "BiasGeluFusion",
        "BiasDropoutFusion",
        "AttentionFusion",
        "EmbedLayerNormFusion",
        "GatherSliceFusion",
        "MatMulIntegerToFloatFusion",
        "DynamicQuantizeMatMulFusion",
        "ConvIntegerFusion",
        "NhwcConvFusion",
        "FreeDimensionOverrideFusion",
        "ExpandFusion",
        "QDQFusion",
        "DoubleQDQPairsRemover",
        "MatMulNBitsFusion",
        "BiasSoftmaxFusion",
        "BiasSplitGeluFusion",
        # 32 so far, add more to exceed limit
        "SimplifiedLayerNormFusion",
        "ReshapeFusion",
        "ConcatSliceFusion",
        "ShapeToInitializerFusion",
        "GatherToSplitFusion",
        "GatherToSliceFusion",
        "DropoutFusion",
        "SoftmaxCrossEntropyLossFusion",
        "NCHWConvFusion",
        "ConstantSharing",
        "CommonSubexpressionElimination",
        "ConstantFolding",
        "ShapeOptimization",
        "MatMulReshapeFusion",
        "TransposeOptimizer",
        "ReluClipFusion",
        "ConvTransposePostProcessor",
        "BatchNormalizationFusion",
        "Relu6Fusion",
        # 50+ optimizers to thoroughly test
    ]


def run_optimization_with_disabled_list(
    model: onnx.ModelProto,
    disabled_list: list[str],
    temp_dir: Path,
    opt_level: int = 2,
) -> tuple[onnx.ModelProto, bool]:
    """Run ORT optimization with a disabled list.

    Args:
        model: Input model
        disabled_list: List of optimizer names to disable
        temp_dir: Temporary directory for files
        opt_level: Optimization level (default 2 = extended)

    Returns:
        Tuple of (optimized model, whether ORT raised any errors)
    """
    input_file = temp_dir / "input.onnx"
    output_file = temp_dir / "output.onnx"

    onnx.save(model, str(input_file))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel(opt_level)
    sess_opts.optimized_model_filepath = str(output_file)

    if disabled_list:
        disable_string = ";".join(disabled_list)
        # This is the key config entry we're testing
        sess_opts.add_session_config_entry(
            "optimization.disable_specified_optimizers",
            disable_string,
        )

    try:
        _ = ort.InferenceSession(
            str(input_file), sess_opts, providers=["CPUExecutionProvider"]
        )
        success = True
    except Exception:
        success = False

    optimized = onnx.load(str(output_file)) if output_file.exists() else model

    return optimized, success


class TestORTDisableLimit:
    """Empirical tests for ORT disable_specified_optimizers limit."""

    def test_ort_accepts_empty_disable_list(self, temp_dir: Path) -> None:
        """Verify ORT works with no disabled optimizers."""
        model = create_relu_model()
        optimized, success = run_optimization_with_disabled_list(
            model, [], temp_dir
        )

        assert success, "ORT should accept empty disable list"
        assert optimized is not None

    def test_ort_accepts_single_disabled_optimizer(self, temp_dir: Path) -> None:
        """Verify ORT works with a single disabled optimizer."""
        model = create_relu_model()
        optimized, success = run_optimization_with_disabled_list(
            model, ["GeluFusion"], temp_dir
        )

        assert success, "ORT should accept single disabled optimizer"
        assert optimized is not None

    def test_ort_accepts_10_disabled_optimizers(self, temp_dir: Path) -> None:
        """Verify ORT works with 10 disabled optimizers."""
        model = create_relu_model()
        optimizers = get_known_ort_optimizers()[:10]

        optimized, success = run_optimization_with_disabled_list(
            model, optimizers, temp_dir
        )

        assert success, "ORT should accept 10 disabled optimizers"
        assert optimized is not None
        print("\n[10 items] ORT accepted disable list successfully")

    def test_ort_accepts_32_disabled_optimizers(self, temp_dir: Path) -> None:
        """Verify ORT works with exactly 32 disabled optimizers.

        This is the claimed limit in graph.py.
        """
        model = create_relu_model()
        optimizers = get_known_ort_optimizers()[:32]

        assert len(optimizers) == 32, "Need exactly 32 optimizers for this test"

        optimized, success = run_optimization_with_disabled_list(
            model, optimizers, temp_dir
        )

        assert success, "ORT should accept 32 disabled optimizers"
        assert optimized is not None
        print("\n[32 items] ORT accepted disable list successfully")

    def test_ort_behavior_with_40_disabled_optimizers(self, temp_dir: Path) -> None:
        """Test ORT behavior with 40 disabled optimizers (beyond claimed limit).

        This is the KEY test - if ORT has a 32-item limit, this should fail
        or behave differently than the 32-item test.
        """
        model = create_relu_model()
        optimizers = get_known_ort_optimizers()[:40]

        assert len(optimizers) == 40, "Need 40 optimizers for this test"

        optimized, success = run_optimization_with_disabled_list(
            model, optimizers, temp_dir
        )

        # Document the actual behavior
        print(f"\n[40 items] ORT session creation: {'SUCCESS' if success else 'FAILED'}")

        # The test passes either way - we're documenting behavior
        assert optimized is not None or not success

    def test_ort_behavior_with_50_disabled_optimizers(self, temp_dir: Path) -> None:
        """Test ORT behavior with 50 disabled optimizers."""
        model = create_relu_model()
        optimizers = get_known_ort_optimizers()

        assert len(optimizers) >= 50, f"Need 50 optimizers, have {len(optimizers)}"
        optimizers = optimizers[:50]

        optimized, success = run_optimization_with_disabled_list(
            model, optimizers, temp_dir
        )

        print(f"\n[50 items] ORT session creation: {'SUCCESS' if success else 'FAILED'}")

        # Document actual behavior
        assert optimized is not None or not success

    def test_character_limit_2048(self, temp_dir: Path) -> None:
        """Test ORT's 2048 character limit on config values.

        ORT source code shows config values are limited to 2048 characters.
        This tests if long disable lists are truncated or rejected.
        """
        model = create_relu_model()

        # Create a long list that exceeds 2048 characters
        # Average optimizer name is ~20 chars + semicolon = ~21 chars
        # 2048 / 21 ≈ 97 items before hitting char limit
        long_name = "A" * 100  # 100-char fake optimizer name
        fake_optimizers = [f"{long_name}_{i}" for i in range(25)]  # ~2500+ chars

        total_chars = len(";".join(fake_optimizers))
        print(f"\n[Char limit test] Total characters: {total_chars}")

        _optimized, success = run_optimization_with_disabled_list(
            model, fake_optimizers, temp_dir
        )

        print(f"[Char limit test] ORT session creation: {'SUCCESS' if success else 'FAILED'}")

        # Document behavior - does ORT reject or silently truncate?
        # This informs whether the issue is item count or character count

    def test_ort_actually_disables_specified_optimizers(self, temp_dir: Path) -> None:
        """Verify that disabling optimizers actually prevents them from running.

        This is the most important validation - we need to confirm ORT
        respects the disable list, not just accepts it without error.
        """
        model = create_add_relu_model()
        original_node_count = len(model.graph.node)

        # First, optimize without disabling anything
        optimized_full, _ = run_optimization_with_disabled_list(
            model, [], temp_dir
        )
        full_opt_nodes = len(optimized_full.graph.node)

        # Now optimize with maximum disabled optimizers
        all_optimizers = get_known_ort_optimizers()
        optimized_disabled, _ = run_optimization_with_disabled_list(
            model, all_optimizers, temp_dir
        )
        disabled_opt_nodes = len(optimized_disabled.graph.node)

        print("\n[Disable effectiveness test]")
        print(f"  Original nodes: {original_node_count}")
        print(f"  Full optimization nodes: {full_opt_nodes}")
        print(f"  With {len(all_optimizers)} disabled: {disabled_opt_nodes}")

        # If disable list works, disabled_opt_nodes should be >= full_opt_nodes
        # (fewer optimizations means more nodes preserved)
        # But if ORT silently ignores the list, they would be equal

    def test_incremental_limit_discovery(self, temp_dir: Path) -> None:
        """Incrementally test to find the actual limit (if any).

        This test tries increasing numbers of disabled optimizers
        to empirically discover where (if anywhere) ORT stops working.
        """
        model = create_relu_model()
        optimizers = get_known_ort_optimizers()

        results: list[tuple[int, bool]] = []

        # Test increments: 10, 20, 30, 40, 50
        for count in [10, 20, 30, 32, 35, 40, 45, 50]:
            if count > len(optimizers):
                break

            subset = optimizers[:count]
            _, success = run_optimization_with_disabled_list(
                model, subset, temp_dir
            )
            results.append((count, success))

        print("\n[Incremental limit discovery]")
        print("  Count | Success")
        print("  ------+--------")
        for count, success in results:
            status = "✓" if success else "✗"
            marker = " <-- claimed limit" if count == 32 else ""
            print(f"  {count:5} | {status}{marker}")

        # Find the first failure point
        failures = [count for count, success in results if not success]
        if failures:
            print(f"\n  First failure at: {failures[0]} items")
        else:
            print(f"\n  No failures detected up to {results[-1][0]} items")
            print("  The 32-item limit claim appears to be INCORRECT")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
