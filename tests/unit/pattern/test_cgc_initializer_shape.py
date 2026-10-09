# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.pattern.cgc.cgc_constant_folding import _ConstantParameters


def test_initializer_shape_without_value_info():
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Shape", ["weights"], ["shape"])],
            "initializer_shape",
            [],
            [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
            [numpy_helper.from_array(np.zeros((2, 3), np.float32), "weights")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    result = _ConstantParameters(model, static_shapes=True).evaluate("shape", set(), set())
    np.testing.assert_array_equal(result, np.asarray([2, 3], np.int64))
