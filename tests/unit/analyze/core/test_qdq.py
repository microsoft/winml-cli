# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for QDQ (Quantize-Dequantize) support functions.

Tests verify:
- QDQGenerator class functionality
- QDQParameterConfig class functionality
- QDQTypeInfo class functionality
- _get_qdq_query_conditions_for_node function
- _collect_qdq_types functionality via RuntimeCheckerQuery
"""

import pytest
from onnx import TensorProto, helper

from winml.modelkit.analyze.core.runtime_checker_query import (
    QDQTypeInfo,
    RuntimeCheckerQuery,
    _get_qdq_query_conditions_for_node,
)
from winml.modelkit.onnx import ONNXDomain, dtypes
from winml.modelkit.pattern.op_input_gen.op_input_gen import (
    QDQParameterConfig,
)
from winml.modelkit.pattern.op_input_gen.qdq_gen import QDQGenerator


class TestQDQGenerator:
    """Tests for QDQGenerator class."""

    @pytest.fixture
    def qdq_generator(self) -> QDQGenerator:
        """Create a QDQGenerator instance for testing."""
        return QDQGenerator(opset_version=17, domain=ONNXDomain.AI_ONNX)

    def test_initialization_and_schemas(self, qdq_generator: QDQGenerator) -> None:
        """Test QDQGenerator initialization and schema loading."""
        # Basic initialization
        assert qdq_generator.domain == ONNXDomain.AI_ONNX
        assert qdq_generator.opset_version >= 1

        # Schemas loaded correctly
        assert qdq_generator.dequantize_linear_schema.name == "DequantizeLinear"
        assert qdq_generator.quantize_linear_schema.name == "QuantizeLinear"

        # Type lists populated from schema
        assert isinstance(qdq_generator.weight_onnx_types, list)
        assert isinstance(qdq_generator.activation_onnx_types, list)
        assert isinstance(qdq_generator.dq_output_onnx_types, list)
        assert isinstance(qdq_generator.q_input_onnx_types, list)

    def test_initialization_does_not_print_to_stdout(self, capsys) -> None:
        """QDQGenerator.__init__ must not write to stdout (regression: #231)."""
        QDQGenerator(opset_version=17, domain=ONNXDomain.AI_ONNX)
        captured = capsys.readouterr()
        assert captured.out == "", (
            "QDQGenerator printed to stdout during init — "
            "debug print() statements must be converted to logger.debug()"
        )

    def test_type_lists_validity(self, qdq_generator: QDQGenerator) -> None:
        """Test that type lists contain valid quantization types."""
        assert len(qdq_generator.weight_onnx_types) > 0
        assert len(qdq_generator.activation_onnx_types) > 0

        # Weight types should be in supported set
        for weight_type in qdq_generator.weight_onnx_types:
            assert weight_type in QDQGenerator.SUPPORTED_WEIGHT_TYPES

        # Activation types should be in supported set
        for activation_type in qdq_generator.activation_onnx_types:
            assert activation_type in QDQGenerator.SUPPORTED_ACTIVATION_TYPES

        # Types should be convertible to SupportedONNXType
        for weight_type in qdq_generator.weight_onnx_types:
            onnx_type = dtypes.SupportedONNXType.from_onnx_type(weight_type)
            assert onnx_type is not None

        for activation_type in qdq_generator.activation_onnx_types:
            onnx_type = dtypes.SupportedONNXType.from_onnx_type(activation_type)
            assert onnx_type is not None

    @pytest.mark.parametrize("opset_version", [13, 17, 21])
    def test_different_opset_versions(self, opset_version: int) -> None:
        """Test QDQGenerator works with different opset versions."""
        gen = QDQGenerator(opset_version=opset_version, domain=ONNXDomain.AI_ONNX)
        assert len(gen.weight_onnx_types) > 0
        assert len(gen.activation_onnx_types) > 0


class TestQDQParameterConfig:
    """Tests for QDQParameterConfig class."""

    def test_default_initialization(self) -> None:
        """Test default initialization with all False flags."""
        config = QDQParameterConfig(support_non_qdq=True)
        assert config.support_weight is False
        assert config.support_activation is False
        assert config.support_non_qdq is True
        assert config.qdq_types is None

    def test_support_flags(self) -> None:
        """Test support_weight and support_activation flags."""
        # Weight only
        config_w = QDQParameterConfig(support_weight=True)
        assert config_w.support_weight is True
        assert config_w.support_activation is False

        # Activation only
        config_a = QDQParameterConfig(support_activation=True)
        assert config_a.support_weight is False
        assert config_a.support_activation is True

        # Both
        config_both = QDQParameterConfig(support_weight=True, support_activation=True)
        assert config_both.support_weight is True
        assert config_both.support_activation is True

    def test_qdq_types_field(self) -> None:
        """Test that qdq_types stores the specified type list."""
        # standalone qdq_types — no support_weight or support_activation needed
        config_standalone = QDQParameterConfig(qdq_types=[dtypes.SupportedONNXType.INT32])
        assert config_standalone.support_weight is False
        assert config_standalone.support_activation is False
        assert config_standalone.qdq_types == [dtypes.SupportedONNXType.INT32]

        # combined with support_weight — yields both True and the specific types
        config_w = QDQParameterConfig(
            support_weight=True, qdq_types=[dtypes.SupportedONNXType.INT8]
        )
        assert config_w.support_weight is True
        assert config_w.qdq_types == [dtypes.SupportedONNXType.INT8]

        config_multi = QDQParameterConfig(
            support_weight=True,
            qdq_types=[dtypes.SupportedONNXType.INT8, dtypes.SupportedONNXType.INT32],
        )
        assert config_multi.qdq_types == [
            dtypes.SupportedONNXType.INT8,
            dtypes.SupportedONNXType.INT32,
        ]


class TestQDQTypeInfo:
    """Tests for QDQTypeInfo class."""

    def test_initialization_and_defaults(self) -> None:
        """Test QDQTypeInfo initialization and default values."""
        # Full initialization
        info = QDQTypeInfo(
            type_annotation="I8",
            domain=ONNXDomain.AI_ONNX,
        )
        assert info.type_annotation == "I8"
        assert info.domain == ONNXDomain.AI_ONNX

    def test_repr_and_str(self) -> None:
        """Test __repr__ and __str__ methods."""
        info = QDQTypeInfo(type_annotation="I8", domain=ONNXDomain.AI_ONNX)
        repr_str = repr(info)

        assert "QDQTypeInfo" in repr_str
        assert "I8" in repr_str
        assert "onnx" in repr_str.lower()
        assert str(info) == repr(info)


class TestGetQDQQueryConditionsForNode:
    """Tests for _get_qdq_query_conditions_for_node function."""

    @pytest.fixture
    def add_schema(self):
        """Get the Add operator schema."""
        return ONNXDomain.AI_ONNX.get_op_schema("Add", 17)

    @pytest.fixture
    def relu_schema(self):
        """Get the Relu operator schema."""
        return ONNXDomain.AI_ONNX.get_op_schema("Relu", 17)

    def test_no_qdq_returns_empty_dict(self, add_schema) -> None:
        """Test that no QDQ patterns returns empty dict."""
        node = helper.make_node("Add", ["a", "b"], ["c"])
        result = _get_qdq_query_conditions_for_node(node, add_schema, {}, {})
        assert result == {}

    def test_quantized_inputs_and_outputs(self, add_schema, relu_schema) -> None:
        """Test nodes with quantized inputs and/or outputs."""
        # Single input quantized
        node_add = helper.make_node("Add", ["a", "b"], ["c"])
        input_to_dq = {"a": QDQTypeInfo("I8", ONNXDomain.AI_ONNX)}
        result = _get_qdq_query_conditions_for_node(node_add, add_schema, input_to_dq, {})
        assert result["QDQ_A"] == "I8"
        assert result["QDQ_B"] is None  # Not quantized

        # Output quantized
        node_relu = helper.make_node("Relu", ["x"], ["y"])
        output_to_q = {"y": QDQTypeInfo("U8", ONNXDomain.AI_ONNX)}
        result = _get_qdq_query_conditions_for_node(node_relu, relu_schema, {}, output_to_q)
        assert result["QDQ_Y"] == "U8"

        # Both input and output quantized
        input_to_dq_relu = {"x": QDQTypeInfo("I8", ONNXDomain.AI_ONNX)}
        result = _get_qdq_query_conditions_for_node(
            node_relu, relu_schema, input_to_dq_relu, output_to_q
        )
        assert result["QDQ_X"] == "I8"
        assert result["QDQ_Y"] == "U8"

    def test_conditions_use_schema_names(self) -> None:
        """Test that QDQ conditions use schema input/output names, not tensor names."""
        schema = ONNXDomain.AI_ONNX.get_op_schema("Add", 17)
        node = helper.make_node("Add", ["tensor_a", "tensor_b"], ["tensor_c"])
        input_to_dq = {"tensor_a": QDQTypeInfo("I8", ONNXDomain.AI_ONNX)}

        result = _get_qdq_query_conditions_for_node(node, schema, input_to_dq, {})

        # Conditions use schema names (A, B), not tensor names
        assert "QDQ_A" in result
        assert result["QDQ_A"] == "I8"
        assert "QDQ_B" in result

    def test_optional_input_not_provided_recorded_as_none(self) -> None:
        """Optional schema input absent from node is recorded as None when other inputs are QDQ.

        Gemm has optional C input. When C is not in node.input but A is quantized,
        QDQ_C should appear as None (not omitted entirely).
        """
        gemm_schema = ONNXDomain.AI_ONNX.get_op_schema("Gemm", 17)

        # Node with only A and B - C is completely absent
        node = helper.make_node("Gemm", ["a", "b"], ["y"])
        input_to_dq = {"a": QDQTypeInfo("I8", ONNXDomain.AI_ONNX)}
        result = _get_qdq_query_conditions_for_node(node, gemm_schema, input_to_dq, {})

        assert result["QDQ_A"] == "I8"
        assert result["QDQ_B"] is None  # present but not quantized
        assert result["QDQ_C"] is None  # optional and absent -> recorded as None

        # Node with C explicitly as empty string (ONNX convention for "not provided")
        node_empty_c = helper.make_node("Gemm", ["a", "b", ""], ["y"])
        result2 = _get_qdq_query_conditions_for_node(node_empty_c, gemm_schema, input_to_dq, {})

        assert result2["QDQ_C"] is None  # same result regardless of how C is omitted

    def test_optional_input_not_provided_no_qdq_returns_empty(self) -> None:
        """When no inputs are quantized, optional absent input does not trigger a result."""
        gemm_schema = ONNXDomain.AI_ONNX.get_op_schema("Gemm", 17)
        node = helper.make_node("Gemm", ["a", "b"], ["y"])
        result = _get_qdq_query_conditions_for_node(node, gemm_schema, {}, {})
        assert result == {}


class TestCollectQDQTypes:
    """Tests for _collect_qdq_types method via RuntimeCheckerQuery."""

    def _make_dq_model(self):
        """Create a model with DequantizeLinear node."""
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 3, 4, 4])
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])

        dq_node = helper.make_node("DequantizeLinear", ["x", "scale"], ["dq_out"], name="dq_node")
        relu_node = helper.make_node("Relu", ["dq_out"], ["y"], name="relu_node")

        graph = helper.make_graph([dq_node, relu_node], "test_dq", [x], [y], [scale])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def _make_q_model(self):
        """Create a model with QuantizeLinear node."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
        zp = helper.make_tensor("zp", TensorProto.INT8, [], [0])
        y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 3, 4, 4])

        relu_node = helper.make_node("Relu", ["x"], ["relu_out"], name="relu_node")
        q_node = helper.make_node(
            "QuantizeLinear", ["relu_out", "scale", "zp"], ["y"], name="q_node"
        )

        graph = helper.make_graph([relu_node, q_node], "test_q", [x], [y], [scale, zp])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def _make_qdq_model(self):
        """Create a model with full QDQ pattern (DQ -> Op -> Q)."""
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 3, 4, 4])
        dq_scale = helper.make_tensor("dq_scale", TensorProto.FLOAT, [], [0.1])
        q_scale = helper.make_tensor("q_scale", TensorProto.FLOAT, [], [0.1])
        zp = helper.make_tensor("zp", TensorProto.INT8, [], [0])
        y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 3, 4, 4])

        dq_node = helper.make_node(
            "DequantizeLinear", ["x", "dq_scale"], ["dq_out"], name="dq_node"
        )
        relu_node = helper.make_node("Relu", ["dq_out"], ["relu_out"], name="relu_node")
        q_node = helper.make_node(
            "QuantizeLinear", ["relu_out", "q_scale", "zp"], ["y"], name="q_node"
        )

        graph = helper.make_graph(
            [dq_node, relu_node, q_node], "test_qdq", [x], [y], [dq_scale, q_scale, zp]
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_collect_dq_and_q_types(self) -> None:
        """Test _collect_qdq_types collects DQ outputs and Q inputs."""
        # Test DQ collection
        dq_query = RuntimeCheckerQuery(
            model_proto=self._make_dq_model(),
            ep_name="QNNExecutionProvider",
            device_type="NPU",
        )
        assert "dq_out" in dq_query.input_to_dq_type
        assert isinstance(dq_query.input_to_dq_type["dq_out"], QDQTypeInfo)

        # Test Q collection
        q_query = RuntimeCheckerQuery(
            model_proto=self._make_q_model(),
            ep_name="QNNExecutionProvider",
            device_type="NPU",
        )
        assert "relu_out" in q_query.output_to_q_type
        assert isinstance(q_query.output_to_q_type["relu_out"], QDQTypeInfo)

        # Test full QDQ pattern
        qdq_query = RuntimeCheckerQuery(
            model_proto=self._make_qdq_model(),
            ep_name="QNNExecutionProvider",
            device_type="NPU",
        )
        assert "dq_out" in qdq_query.input_to_dq_type
        assert "relu_out" in qdq_query.output_to_q_type

    def test_dq_weight_vs_activation(self) -> None:
        """Test _collect_qdq_types distinguishes weights from activations via initializers."""
        # Create model where DQ input is from initializer (weight)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
        weight = helper.make_tensor("weight", TensorProto.INT8, [3, 3, 1, 1], [1] * 9)
        dq_scale = helper.make_tensor("dq_scale", TensorProto.FLOAT, [], [0.1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])

        dq_node = helper.make_node(
            "DequantizeLinear", ["weight", "dq_scale"], ["dq_out"], name="dq_node"
        )
        conv_node = helper.make_node(
            "Conv", ["x", "dq_out"], ["y"], name="conv_node", kernel_shape=[1, 1], pads=[0, 0, 0, 0]
        )

        graph = helper.make_graph(
            [dq_node, conv_node], "test_weight_dq", [x], [y], [weight, dq_scale]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        query = RuntimeCheckerQuery(
            model_proto=model, ep_name="QNNExecutionProvider", device_type="NPU"
        )

        # DQ output from initializer should be detected
        assert "dq_out" in query.input_to_dq_type
        assert isinstance(query.input_to_dq_type["dq_out"], QDQTypeInfo)


class TestIterShouldQDQCombinations:
    """Unit tests for _iter_should_qdq_combinations.

    This method generates the cross-product of which inputs/outputs should
    have QDQ nodes applied, based on QDQParameterConfig flags. The result
    is a flat dict {schema_name: bool}.
    """

    @pytest.fixture
    def tanh_gen(self):
        """TanhInputGenerator — single input, single output."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op
        from winml.modelkit.pattern.op_input_gen.qdq_gen import QDQGenerator

        schema = ONNXDomain.AI_ONNX.get_op_schema("Tanh", 22)
        qdq_gen = QDQGenerator(opset_version=17, domain=ONNXDomain.AI_ONNX)
        return get_runtime_checker_op("Tanh")(schema, qdq_generator=qdq_gen)

    @pytest.fixture
    def gather_gen(self):
        """GatherInputGenerator — two inputs (data: activation, indices: pass-through)."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op
        from winml.modelkit.pattern.op_input_gen.qdq_gen import QDQGenerator

        schema = ONNXDomain.AI_ONNX.get_op_schema("Gather", 22)
        qdq_gen = QDQGenerator(opset_version=1, domain=ONNXDomain.COM_MICROSOFT)
        return get_runtime_checker_op("Gather")(schema, qdq_generator=qdq_gen)

    def _first_kwargs(self, gen):
        kwargs, _ = next(iter(gen.iter()))
        return kwargs

    def test_empty_input_config_output_still_defaults_true(self, tanh_gen) -> None:
        """Empty qdq_config: no input entries, but the output still gets should_qdq=True."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        output_name = gen.schema.outputs[0].name
        results = list(gen._iter_should_qdq_combinations(kwargs, {}))
        assert len(results) == 1
        assert results[0] == {output_name: True}

    def test_activation_only_input_yields_true(self, tanh_gen) -> None:
        """support_activation=True → only True is yielded for that input."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        input_vals = [r[input_name] for r in results]
        assert input_vals == [True]

    def test_non_qdq_only_input_yields_false(self, tanh_gen) -> None:
        """support_non_qdq=True only → only False is yielded for that input."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        qdq_config = {input_name: QDQParameterConfig(support_non_qdq=True)}
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        input_vals = [r[input_name] for r in results]
        assert input_vals == [False]

    def test_both_flags_yields_true_and_false(self, tanh_gen) -> None:
        """support_activation=True + support_non_qdq=True → both True and False yielded."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        qdq_config = {input_name: QDQParameterConfig(support_activation=True, support_non_qdq=True)}
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        input_vals = [r[input_name] for r in results]
        assert sorted(input_vals) == [False, True]

    def test_output_not_in_config_always_true(self, tanh_gen) -> None:
        """Output absent from qdq_config always gets should_qdq=True."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        assert all(r[output_name] is True for r in results)

    def test_output_with_non_qdq_yields_false_combo(self, tanh_gen) -> None:
        """Output configured with support_non_qdq=True yields a False combination."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        qdq_config = {
            input_name: QDQParameterConfig(support_activation=True),
            output_name: QDQParameterConfig(support_non_qdq=True),
        }
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        output_vals = [r[output_name] for r in results]
        assert False in output_vals

    def test_cross_product_two_inputs(self, gather_gen) -> None:
        """Two inputs with independent options produce the full cross-product."""
        gen = gather_gen
        kwargs = self._first_kwargs(gen)
        # Gather: data=activation, indices=pass-through
        qdq_config = gen.get_qdq_config()
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        # data: only True; indices: only False -> 1 x 1 = 1 input combo
        # output: only True → total 1
        assert len(results) >= 1
        # data (activation-only) must always be True
        data_vals = {r.get("data") for r in results}
        assert data_vals == {True}

    def test_output_name_in_config_not_treated_as_input(self, tanh_gen) -> None:
        """An output name in qdq_config is handled as an output, not duplicated as input."""
        gen = tanh_gen
        kwargs = self._first_kwargs(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        qdq_config = {
            input_name: QDQParameterConfig(support_activation=True),
            output_name: QDQParameterConfig(support_activation=True),
        }
        results = list(gen._iter_should_qdq_combinations(kwargs, qdq_config))
        # Both input and output appear in each result, each exactly once
        assert all(input_name in r for r in results)
        assert all(output_name in r for r in results)
        # No extra keys beyond input + output
        assert all(set(r.keys()) == {input_name, output_name} for r in results)


class TestIterQDQCombinationsUnit:
    """Unit tests for the iter_qdq_combinations method of OpInputGenerator."""

    @pytest.fixture
    def qdq_gen(self) -> QDQGenerator:
        """QDQGenerator using AI_ONNX domain at opset 17."""
        return QDQGenerator(opset_version=17, domain=ONNXDomain.AI_ONNX)

    @pytest.fixture
    def tanh_gen(self, qdq_gen: QDQGenerator):
        """TanhInputGenerator with QDQ support enabled."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op

        schema = ONNXDomain.AI_ONNX.get_op_schema("Tanh", 22)
        return get_runtime_checker_op("Tanh")(schema, qdq_generator=qdq_gen)

    @pytest.fixture
    def tanh_gen_no_qdq(self):
        """TanhInputGenerator without a QDQ generator."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op

        schema = ONNXDomain.AI_ONNX.get_op_schema("Tanh", 22)
        return get_runtime_checker_op("Tanh")(schema)

    def _first_kwargs_tags(self, gen):
        """Return the first (kwargs, tags) pair produced by gen.iter()."""
        return next(iter(gen.iter()))

    def _float_kwargs_tags(self, gen):
        """Return (kwargs, tags) from gen.iter() where every type var resolves to FLOAT.

        iter_qdq_combinations only processes float inputs (SUPPORT_DQ_OUTPUT_TYPES = {float}).
        Tests that expect yielded models must use a FLOAT type-variable combo.

        Also injects 'dynamic_axes': {} because iter_qdq_combinations now requires this
        key in tags (normally injected by iter_const_and_dynamic_models in production).
        """
        float_annotation = dtypes.SupportedONNXType.FLOAT.annotation
        for kwargs, tags in gen.iter():
            if all(v == float_annotation for v in tags[gen.type_vars_key].values()):
                return kwargs, {**tags, "dynamic_axes": {}}
        raise ValueError("No all-FLOAT type-var combo found in gen.iter()")

    def test_yields_nothing_without_qdq_generator(self, tanh_gen_no_qdq) -> None:
        """iter_qdq_combinations returns immediately when qdq_generator is None."""
        gen = tanh_gen_no_qdq
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert results == []

    def test_yields_nothing_when_qdq_config_is_none(self, tanh_gen) -> None:
        """iter_qdq_combinations returns immediately when qdq_config is None."""
        gen = tanh_gen
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        results = list(gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, None, set()))
        assert results == []

    def test_yields_nothing_when_input_absent_from_config(self, tanh_gen) -> None:
        """iter_qdq_combinations returns immediately when an input is not in qdq_config."""
        gen = tanh_gen
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        results = list(gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, {}, set()))
        assert results == []

    def test_yields_nothing_when_constant_input_lacks_weight_support(self, tanh_gen) -> None:
        """iter_qdq_combinations returns when a constant input has no weight support in config."""
        gen = tanh_gen
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: True}  # constant = weight
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}  # activation only
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert results == []

    def test_yields_nothing_when_nonconstant_input_lacks_activation_support(self, tanh_gen) -> None:
        """iter_qdq_combinations returns when a non-constant input has no activation support."""
        gen = tanh_gen
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}  # non-constant = activation
        qdq_config = {input_name: QDQParameterConfig(support_weight=True)}  # weight only
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert results == []

    def test_activation_input_yields_one_per_activation_type(self, tanh_gen) -> None:
        """A non-constant activation input produces one model per activation type."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert len(results) == len(gen.qdq_generator.activation_onnx_types)

    def test_weight_input_yields_weight_times_activation_count(self, tanh_gen) -> None:
        """A constant weight input produces |weight_types| x |activation_types| unique models."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: True}
        qdq_config = {input_name: QDQParameterConfig(support_weight=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        n_weight = len(gen.qdq_generator.weight_onnx_types)
        n_activation = len(gen.qdq_generator.activation_onnx_types)
        assert len(results) == n_weight * n_activation

    def test_yielded_tags_contain_qdq_types_key(self, tanh_gen) -> None:
        """Each yielded (model, tags) tuple includes a non-empty 'qdq_types' dict in tags."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert len(results) > 0
        for _, final_tags in results:
            assert "qdq_types" in final_tags
            assert isinstance(final_tags["qdq_types"], dict)
            assert len(final_tags["qdq_types"]) > 0

    def test_qdq_types_maps_both_input_and_output_schema_names(self, tanh_gen) -> None:
        """qdq_types tag contains entries for the schema input name and the schema output name."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert len(results) > 0
        for _, final_tags in results:
            qdq_types = final_tags["qdq_types"]
            assert input_name in qdq_types
            assert output_name in qdq_types
            assert isinstance(qdq_types[input_name], str)
            assert isinstance(qdq_types[output_name], str)

    def test_deduplication_prevents_second_pass_yields(self, tanh_gen) -> None:
        """Sharing qdq_tested_types between two calls prevents any yields on the second call."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        qdq_tested_types: set = set()
        first_pass = list(
            gen.iter_qdq_combinations(
                kwargs, tags, is_constant_map, {}, qdq_config, qdq_tested_types
            )
        )
        second_pass = list(
            gen.iter_qdq_combinations(
                kwargs, tags, is_constant_map, {}, qdq_config, qdq_tested_types
            )
        )
        assert len(first_pass) > 0
        assert second_pass == []

    def test_fresh_qdq_tested_types_allows_full_iteration(self, tanh_gen) -> None:
        """A fresh qdq_tested_types set always produces the complete set of combinations."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        first = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        second = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert len(first) == len(second) > 0

    def test_qdq_types_skips_weight_iteration(self, tanh_gen) -> None:
        """qdq_types yields |activation_types| models, not |weight| x |activation|."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: True}
        qdq_config = {input_name: QDQParameterConfig(qdq_types=[dtypes.SupportedONNXType.INT8])}
        should_qdq = {input_name: dtypes.SupportedONNXType.INT8}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, should_qdq, qdq_config, set())
        )
        assert len(results) == len(gen.qdq_generator.activation_onnx_types)

    def test_qdq_types_sets_correct_weight_type_in_tags(self, tanh_gen) -> None:
        """qdq_types for a constant input appears verbatim in qdq_types tags."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: True}
        qdq_config = {input_name: QDQParameterConfig(qdq_types=[dtypes.SupportedONNXType.INT8])}
        should_qdq = {input_name: dtypes.SupportedONNXType.INT8}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, should_qdq, qdq_config, set())
        )
        assert len(results) > 0
        for _, final_tags in results:
            assert final_tags["qdq_types"][input_name] == dtypes.SupportedONNXType.INT8.annotation

    def test_qdq_types_sets_correct_activation_type_in_tags(self, tanh_gen) -> None:
        """qdq_types for a non-constant input appears verbatim in qdq_types tags."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(qdq_types=[dtypes.SupportedONNXType.UINT8])}
        should_qdq = {input_name: dtypes.SupportedONNXType.UINT8}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, should_qdq, qdq_config, set())
        )
        assert len(results) > 0
        for _, final_tags in results:
            assert final_tags["qdq_types"][input_name] == dtypes.SupportedONNXType.UINT8.annotation

    def test_passthrough_input_recorded_as_empty_in_qdq_types_tag(self, tanh_gen) -> None:
        """Pass-through inputs (no weight/activation support) appear in qdq_types as None."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_non_qdq=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, tags, is_constant_map, {}, qdq_config, set())
        )
        assert len(results) > 0
        for _, final_tags in results:
            assert final_tags["qdq_types"][input_name] is None

    def test_output_no_qdq_suppresses_q_node(self, tanh_gen) -> None:
        """Output with should_qdq=False in should_qdq_map produces no QuantizeLinear node.

        When should_qdq_map[output_name] is False, the generated model must NOT
        have a QuantizeLinear node wrapping the operator output, and the output qdq_types tag
        must be None for that output.
        """
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        is_constant_map = {input_name: False}
        qdq_config = {
            input_name: QDQParameterConfig(support_activation=True),
        }
        should_qdq_map = {
            input_name: True,
            output_name: False,  # suppress Q on output
        }
        results = list(
            gen.iter_qdq_combinations(
                kwargs, tags, is_constant_map, should_qdq_map, qdq_config, set()
            )
        )
        assert len(results) > 0

        for model, final_tags in results:
            # qdq_types tag must record None for the suppressed output
            assert final_tags["qdq_types"][output_name] is None

            # Model graph must not contain any QuantizeLinear node
            node_op_types = [node.op_type for node in model.graph.node]
            assert "QuantizeLinear" not in node_op_types, (
                f"Expected no QuantizeLinear node, but found nodes: {node_op_types}"
            )

            # Model must still contain a DequantizeLinear node (input is still quantized)
            assert "DequantizeLinear" in node_op_types

            # Graph output must be the raw float output of the operator (not a quantized type)
            import onnx

            assert len(model.graph.output) == 1
            assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    def test_input_suppressed_via_should_qdq_map_records_none_in_qdq_types(self, tanh_gen) -> None:
        """should_qdq_map[input]=False suppresses DQ and records None in qdq_types tag."""
        gen = tanh_gen
        kwargs, tags = self._float_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        output_name = gen.schema.outputs[0].name
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        should_qdq_map = {
            input_name: False,  # suppress DQ on input
            output_name: True,
        }
        results = list(
            gen.iter_qdq_combinations(
                kwargs, tags, is_constant_map, should_qdq_map, qdq_config, set()
            )
        )
        assert len(results) > 0
        for model, final_tags in results:
            assert final_tags["qdq_types"][input_name] is None
            node_op_types = [node.op_type for node in model.graph.node]
            assert "DequantizeLinear" not in node_op_types
            assert "QuantizeLinear" in node_op_types

    def test_yields_nothing_when_input_type_not_in_dq_output_types(self, tanh_gen) -> None:
        """iter_qdq_combinations yields nothing when the resolved input type is not float."""
        gen = tanh_gen
        kwargs, tags = self._first_kwargs_tags(gen)
        input_name = gen.op_input_names[0]
        # Force the type variable to INT8, which is not in SUPPORT_DQ_OUTPUT_TYPES
        type_var_key = next(iter(tags[gen.type_vars_key]))
        int8_annotation = dtypes.SupportedONNXType.INT8.annotation
        int8_tags = {**tags, gen.type_vars_key: {type_var_key: int8_annotation}}
        is_constant_map = {input_name: False}
        qdq_config = {input_name: QDQParameterConfig(support_activation=True)}
        results = list(
            gen.iter_qdq_combinations(kwargs, int8_tags, is_constant_map, {}, qdq_config, set())
        )
        assert results == []


class TestIterQDQCombinationsTagSchema:
    """Tests verifying output tag structure matches the Generated Table Schema in qdq.md.

    Uses Gather and Gemm to cover the three tag schema cases:
    - QDQ input: present in 'qdq_types' with a type annotation
    - Pass-through input: present in 'qdq_types' as '', present in 'input_is_constant'
    - Optional QDQ input not provided: present in 'qdq_types' as ''

    Note: When an operator has pass-through inputs, 'input_is_constant' contains only
    those pass-through inputs (Gather). When an optional input supports both QDQ and
    non-QDQ (pass-through) modes (Gemm C), 'input_is_constant' contains that input only
    in the non-QDQ combination; pure QDQ inputs (A, B) never appear in 'input_is_constant'.
    """

    @pytest.fixture
    def qdq_gen(self) -> QDQGenerator:
        """QDQGenerator using COM_MICROSOFT domain at opset 1 (production domain)."""
        return QDQGenerator(opset_version=1, domain=ONNXDomain.COM_MICROSOFT)

    @pytest.fixture
    def gather_gen(self, qdq_gen: QDQGenerator):
        """GatherInputGenerator with QDQ support enabled."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op

        schema = ONNXDomain.AI_ONNX.get_op_schema("Gather", 22)
        return get_runtime_checker_op("Gather")(schema, qdq_generator=qdq_gen)

    @pytest.fixture
    def gemm_gen(self, qdq_gen: QDQGenerator):
        """GemmInputGenerator with QDQ support enabled."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op

        schema = ONNXDomain.AI_ONNX.get_op_schema("Gemm", 22)
        return get_runtime_checker_op("Gemm")(schema, qdq_generator=qdq_gen)

    def _gather_float_data_kwargs_tags(self, gen):
        """Return (kwargs, tags) from Gather iter() where data resolves to FLOAT.

        Gather has two type vars (T for data, Tind for indices), so only checking
        the data-specific type var is sufficient instead of requiring all to be FLOAT.
        """
        float_ann = dtypes.SupportedONNXType.FLOAT.annotation
        data_template = gen.type_annotations["data"]
        for kwargs, tags in gen.iter():
            resolved = gen._apply_type_var_combination(data_template, tags[gen.type_vars_key])
            if resolved == float_ann:
                return kwargs, tags
        raise ValueError("No FLOAT data type found in Gather iter()")

    def _gemm_float_c_provided_kwargs_tags(self, gen):
        """Return (kwargs, tags) from Gemm iter() where T_Gemm is FLOAT and C is provided."""
        float_ann = dtypes.SupportedONNXType.FLOAT.annotation
        for kwargs, tags in gen.iter():
            if not all(v == float_ann for v in tags[gen.type_vars_key].values()):
                continue
            if kwargs.get("C") is not None:
                return kwargs, tags
        raise ValueError("No FLOAT Gemm with C provided in iter()")

    def _gemm_float_c_none_kwargs_tags(self, gen):
        """Return (kwargs, tags) from Gemm iter() where T_Gemm is FLOAT and C is None."""
        float_ann = dtypes.SupportedONNXType.FLOAT.annotation
        for kwargs, tags in gen.iter():
            if not all(v == float_ann for v in tags[gen.type_vars_key].values()):
                continue
            if kwargs.get("C") is None:
                return kwargs, tags
        raise ValueError("No FLOAT Gemm with C=None in iter()")

    # ---- Gather: QDQ input (data) ----

    def test_gather_qdq_data_present_in_qdq_types(self, gather_gen) -> None:
        """data (QDQ activation input) appears in qdq_types with a non-empty annotation."""
        gen = gather_gen
        kwargs, tags = self._gather_float_data_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert "data" in final_tags["qdq_types"]
            assert len(final_tags["qdq_types"]["data"]) > 0

    def test_gather_qdq_data_absent_from_input_is_constant(self, gather_gen) -> None:
        """data (QDQ activation input) does not appear in input_is_constant."""
        gen = gather_gen
        kwargs, tags = self._gather_float_data_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            if "input_is_constant" in final_tags:
                assert "data" not in final_tags["input_is_constant"]

    # ---- Gather: pass-through input (indices) ----

    def test_gather_passthrough_indices_present_in_input_is_constant(self, gather_gen) -> None:
        """indices (pass-through input) is present in input_is_constant."""
        gen = gather_gen
        kwargs, tags = self._gather_float_data_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert "input_is_constant" in final_tags
            assert "indices" in final_tags["input_is_constant"]
            assert isinstance(final_tags["input_is_constant"]["indices"], bool)

    def test_gather_passthrough_indices_recorded_as_empty_in_qdq_types(self, gather_gen) -> None:
        """indices (pass-through input) appears in qdq_types as ''."""
        gen = gather_gen
        kwargs, tags = self._gather_float_data_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert final_tags["qdq_types"]["indices"] is None

    # ---- Gather: output tag ----

    def test_gather_output_has_valid_activation_type(self, gather_gen) -> None:
        """The output schema name is in qdq_types with a valid activation type annotation."""
        gen = gather_gen
        output_name = gen.schema.outputs[0].name
        kwargs, tags = self._gather_float_data_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        valid_anns = {
            dtypes.SupportedONNXType.from_onnx_type(t).annotation
            for t in gen.qdq_generator.activation_onnx_types
        }
        for _, final_tags in results:
            assert output_name in final_tags["qdq_types"]
            assert final_tags["qdq_types"][output_name] in valid_anns

    # ---- Gemm: QDQ inputs (A activation, B weight) ----

    def test_gemm_activation_a_present_in_qdq_types(self, gemm_gen) -> None:
        """A (QDQ activation) is present in qdq_types with a non-empty annotation."""
        gen = gemm_gen
        kwargs, tags = self._gemm_float_c_provided_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert "A" in final_tags["qdq_types"]
            assert final_tags["qdq_types"]["A"] != ""

    def test_gemm_weight_b_present_in_qdq_types(self, gemm_gen) -> None:
        """B (QDQ weight) is present in qdq_types with a non-empty annotation."""
        gen = gemm_gen
        kwargs, tags = self._gemm_float_c_provided_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert "B" in final_tags["qdq_types"]
            assert final_tags["qdq_types"]["B"] != ""

    def test_gemm_qdq_inputs_not_in_input_is_constant(self, gemm_gen) -> None:
        """A (activation) and B (weight) never appear in input_is_constant.

        C may appear in input_is_constant for its non-QDQ (pass-through) combination,
        but pure QDQ inputs A and B are never pass-through.
        """
        gen = gemm_gen
        kwargs, tags = self._gemm_float_c_provided_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            ic = final_tags.get("input_is_constant", {})
            assert "A" not in ic
            assert "B" not in ic

    # ---- Gemm: optional QDQ input (C) ----

    def test_gemm_optional_c_provided_has_int32_type(self, gemm_gen) -> None:
        """When optional C is provided and quantized, qdq_types['C'] is INT32 annotation.

        C supports both QDQ (INT32) and non-QDQ (pass-through) modes. In the non-QDQ
        combination qdq_types['C'] is None; when quantized it must be INT32.
        """
        gen = gemm_gen
        int32_ann = dtypes.SupportedONNXType.INT32.annotation
        kwargs, tags = self._gemm_float_c_provided_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        int32_seen = False
        for _, final_tags in results:
            assert "C" in final_tags["qdq_types"]
            c_type = final_tags["qdq_types"]["C"]
            if c_type is not None:
                assert c_type == int32_ann
                int32_seen = True
        assert int32_seen, "Expected at least one result with C quantized as INT32"

    def test_gemm_optional_c_not_provided_recorded_as_empty_in_qdq_types(self, gemm_gen) -> None:
        """When optional C is not provided (None), qdq_types['C'] is '' (not omitted)."""
        gen = gemm_gen
        kwargs, tags = self._gemm_float_c_none_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        for _, final_tags in results:
            assert final_tags["qdq_types"]["C"] is None

    # ---- Gemm: output tag ----

    def test_gemm_output_y_has_valid_activation_type(self, gemm_gen) -> None:
        """Output Y is in qdq_types with a valid activation type annotation."""
        gen = gemm_gen
        output_name = gen.schema.outputs[0].name  # "Y"
        kwargs, tags = self._gemm_float_c_provided_kwargs_tags(gen)
        results = list(gen.iter_const_and_dynamic_models(kwargs, tags))
        assert len(results) > 0
        valid_anns = {
            dtypes.SupportedONNXType.from_onnx_type(t).annotation
            for t in gen.qdq_generator.activation_onnx_types
        }
        for _, final_tags in results:
            assert output_name in final_tags["qdq_types"]
            assert final_tags["qdq_types"][output_name] in valid_anns


unary_input_shapes = 8
binary_input_shapes = 42


class TestIterQDQCombinations:
    """Tests for iter_qdq_combinations and iter_const_and_dynamic_models methods."""

    @pytest.mark.parametrize(
        "op_name,expected_count",
        [
            # All binary use this and it is enough
            ("Add", binary_input_shapes * (16 * 2 - 4)),  # 1176
            (
                "AveragePool",
                1152,
            ),  # QDQ 4 * shape 3 * combo 3 * finite attributes 2 * 2 * 2 * optional strides, pads 4
            (
                "Cast",
                (1 + 12 * 2) * 4 * unary_input_shapes,
            ),  # float->float, others (12 types) only 1 direction supported
            ("Clip", unary_input_shapes * 4 * (4 + 4 + 4 + 1)),  # act 4 * weight 13
            ("Concat", 240),  # 15 base shapes/axes * 4 variadic counts * 4 activation types
            (
                "Conv",
                1536 * 4,
            ),  # shape 3 * attrs 4 * 2 * kernel shape 2 * opt B 2 * 16 * B/Y non qdq 4
            (
                "ConvTranspose",
                3328,
            ),  # base: shape 3 * auto_pad 4 * group_opts 2 * output 4 * optional b 2 * 16 +
            # even-kernel: 2 * output 4 * optional b 2 * 16
            (
                "CumSum",
                2816,
            ),  # qdq 4 * is_constant 4 * attributes 4 * (2 + 2 + 4 + 6 + 8 + 10 + 12)
            # All comparison use this
            ("Equal", binary_input_shapes * 16),  # 672
            ("Expand", 328),  # case 41 * QDQ 4 * is_constant shape 2
            ("Flatten", 28 * 4),  # 112
            (
                "Gather",
                1184,
            ),  # (2*7+3*(6+5+4+3+2)) input combos
            # * 2 Tind types * 2 Tind optional * 4 act types
            (
                "GatherElements",
                864,
            ),  # (7+6+5+4+3+2)*2 input combos * 2 Tind types * 2 Tind optional * 4 activation types
            ("Gelu", unary_input_shapes * 4 * 2),  # 64
            (
                "Gemm",
                36 * 16 * (4 + 3 * 2),
            ),  # attributes (2 * 2 * 3 * 3) * QDQ * C (qdq + non-qdq * opt)
            ("GlobalAveragePool", 3 * 4),  # 12
            ("InstanceNormalization", 3 * 16),  # 48
            ("LayerNormalization", 5 * 2 * 2 * 16),  # 320
            ("MatMul", 36 * (16 * 2 - 4)),  # 1008
            (
                "MaxPool",
                768,
            ),  # shape 3 * finite attributes 2 * 2 * 2 * optional combinations 2 * 2 * 2 * 4
            (
                "Pad",
                512,
            ),  # shape 8 * mode 4 * QDQ 4 * is_constant pads 2 * Tind 2 (actually axes not used)
            # All Reduce* use this and it is enough
            (
                "ReduceSum",
                1440,
            ),  # (3+3+6*4) * 4 QDQ * 4 attr combos
            # * 3 (axes none, const, not const)
            ("Relu", unary_input_shapes * 4 * 2),
            ("Reshape", 36 * 4 * 2 * 2),  # allowzero 2 * is_constant 2
            (
                "Resize",
                3456,
            ),  # shape 4 * T2 3 * QDQ 4 * antialias 2
            # * attribute 6 * (optional input 4 + 2)
            (
                "ScatterND",
                1680,
            ),  # qdq 4 * is_constant 2 * reduction 5 * q 2 * (1 + 2 + 3 + 4 + 5 + 6)
            ("Shape", 37 * 4 * 2),  # optional end
            (
                "Slice",
                6240,
            ),  # QDQ 4 * Tind 2 * (48 with_axes * 16 is_const + 3 without * 4 is_const)
            ("Softmax", (unary_input_shapes - 1) * 11 * 4),  # 704 actually 172
            ("Split", 352),  # QDQ 4 * (30 * is_constant 2 + 28)
            ("Squeeze", 216),  # QDQ 4 * split 3 state * 18 cases
            # All unary use this and it is enough
            ("Tanh", unary_input_shapes * 4),  # 32
            ("TopK", 768),  # QDQ 4 * example 12 * k is_constant 2 * parameter 8
            ("Transpose", 11 * 4 * 2 * 2),  # cases * QDQ * opt perm * non_qdq data
            ("Unsqueeze", 208),  # 26 * 4 QDQ types * 2 is_constant axes
            (
                "Where",
                (42 * 2 - 7) * 4 * (2 + 4 + 4 + 7),
            ),  # 5236, 7 cases with same shape, different qdq for x,y
        ],
    )
    def test_qdq_total_count(self, op_name: str, expected_count: int) -> None:
        """Test the total count of QDQ combinations generated for various operators."""
        from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op

        schema = ONNXDomain.AI_ONNX.get_op_schema(op_name, 22)
        qdq_gen = QDQGenerator(opset_version=1, domain=ONNXDomain.COM_MICROSOFT)
        generator = get_runtime_checker_op(op_name)(schema, qdq_generator=qdq_gen)

        # Count yielded models via iter_const_and_dynamic_models
        count = 0
        for kwargs, tags in generator.iter():
            for _model, final_tags in generator.iter_const_and_dynamic_models(kwargs, tags):
                assert "qdq_types" in final_tags
                count += 1

        # For rerun, could track in https://github.com/gim-home/ModelKit/issues/278
        assert count == expected_count, "If changes, either bug or need to rerun"
