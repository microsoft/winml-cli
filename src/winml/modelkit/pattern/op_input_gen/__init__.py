# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from .attention_input_generator import AttentionInputGenerator
from .binary_input_generator import *
from .binary_like_input_generator import *
from .constant_of_shape_input_generator import ConstantOfShapeInputGenerator
from .conv_input_generator import *
from .expand_input_generator import ExpandInputGenerator
from .flatten_input_generator import FlattenInputGenerator
from .global_pooling_input_generator import *
from .indexing_input_generator import (
    GatherInputGenerator,
    ScatterNDInputGenerator,
    SplitInputGenerator,
    UnsqueezeInputGenerator,
)
from .matmul_input_generator import *
from .normalization_input_generator import *
from .op_input_gen import *
from .pad_input_generator import PadInputGenerator
from .pooling_input_generator import *
from .reduction_input_generator import *
from .reshape_input_generator import ReshapeInputGenerator
from .resize_input_generator import ResizeInputGenerator
from .rotary_embedding_input_generator import RotaryEmbeddingInputGenerator
from .shape_input_generator import ShapeInputGenerator
from .slice_input_generator import SliceInputGenerator
from .squeeze_input_generator import SqueezeInputGenerator
from .transpose_input_generator import TransposeInputGenerator
from .unary_input_generator import *
from .unary_like_input_generator import *
from .variadic_input_generator import *
