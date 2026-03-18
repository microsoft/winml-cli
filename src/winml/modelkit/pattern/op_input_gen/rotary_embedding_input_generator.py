"""Input generator for RotaryEmbedding ONNX operator (opset 23+).

RotaryEmbedding applies rotary position encoding to input tensors.
It supports both 3D (batch_size, sequence_length, hidden_size) and
4D (batch_size, num_heads, sequence_length, head_size) input formats.
"""

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class RotaryEmbeddingInputGenerator(OpInputGenerator):
    """Input generator for RotaryEmbedding operator.

    Signature: RotaryEmbedding(X, cos_cache, sin_cache, position_ids?,
                                *, interleaved=0, rotary_embedding_dim=0, num_heads=0)

    Applies rotary position encoding to input embeddings.

    Inputs:
    - X: Input tensor
        4D: (batch_size, num_heads, sequence_length, head_size)
        3D: (batch_size, sequence_length, hidden_size) when num_heads > 0
    - cos_cache: Cosine values (max_position_id + 1, head_size / 2)
    - sin_cache: Sine values (max_position_id + 1, head_size / 2)
    - position_ids: Optional position indices (batch_size, sequence_length)

    Attributes:
    - interleaved: 0 for non-interleaved (default), 1 for interleaved
    - rotary_embedding_dim: Partial rotation dim, 0 means full rotation (default)
    - num_heads: Number of heads for 3D input, 0 means inferred from 4D shape (default)
    """

    op_name = "RotaryEmbedding"
    # Disable optional expansion: num_heads is in optional_attrs_without_defaults
    # but ORT requires it when rotary_embedding_dim > 0 or input is 3D.
    # We explicitly control which combos include num_heads.
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute values for RotaryEmbedding.

        interleaved controls whether rotary uses interleaved or split-half layout.
        rotary_embedding_dim and num_heads are handled per-shape in input combinations.
        """
        return {
            "interleaved": [0, 1],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for RotaryEmbedding.

        Tests both 4D and 3D input formats with varying rotary_embedding_dim.
        cos_cache/sin_cache shape: (batch, seq_len, rotary_embedding_dim / 2)
        for partial rotation, or (batch, seq_len, head_size / 2) for full.
        """
        combinations = []

        # 4D input: (batch, num_heads, seq_len, head_size)
        batch, heads, seq_len, head_size = 2, 4, 8, 16
        # rotary_embedding_dim: 0 = full rotation, or partial dims (must be even)
        # ORT requires num_heads when rotary_embedding_dim > 0
        for rot_dim in (0, 8, head_size):
            cache_dim = head_size // 2 if rot_dim == 0 else rot_dim // 2
            cache_shape = (batch, seq_len, cache_dim)
            combo: dict = {
                "X": InputShapeConstraint((batch, heads, seq_len, head_size)),
                "cos_cache": InputShapeConstraint(cache_shape),
                "sin_cache": InputShapeConstraint(cache_shape),
                "rotary_embedding_dim": rot_dim,
            }
            if rot_dim > 0:
                combo["num_heads"] = heads
            combinations.append(combo)

        # 3D input: (batch, seq_len, hidden_size) — num_heads required
        batch_3d, seq_len_3d, hidden_size, num_heads = 2, 8, 32, 4
        h_size = hidden_size // num_heads
        # rotary_embedding_dim: 0 = full, half, full explicit
        for rot_dim in (0, h_size // 2, h_size):
            cache_dim = h_size // 2 if rot_dim == 0 else rot_dim // 2
            cache_shape = (batch_3d, seq_len_3d, cache_dim)

            combinations.append(
                {
                    "X": InputShapeConstraint((batch_3d, seq_len_3d, hidden_size)),
                    "cos_cache": InputShapeConstraint(cache_shape),
                    "sin_cache": InputShapeConstraint(cache_shape),
                    "num_heads": num_heads,
                    "rotary_embedding_dim": rot_dim,
                }
            )

        # TODO: add position_ids to combinations

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for RotaryEmbedding testing.

        Args:
            properties: Base properties containing X_shape

        Returns:
            Updated properties with RotaryEmbedding-specific derived values
        """
        item = properties.copy()
        input_name = self.op_input_names[0]
        x_shape = item[f"{input_name}_shape"]
        x_dim = len(x_shape)
        item[f"{input_name}_dim"] = x_dim

        rot_dim = item.get("attr_rotary_embedding_dim", 0)
        # Determine head_size from input shape
        if x_dim == 4:
            head_size = x_shape[-1]
        elif x_dim == 3 and "attr_num_heads" in item and item["attr_num_heads"] > 0:
            head_size = x_shape[-1] // item["attr_num_heads"]
        else:
            head_size = 0
        item["rotary_embedding_dim_is_zero"] = rot_dim == 0
        item["rotary_embedding_dim_is_full"] = rot_dim == head_size
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values."""
        return (
            [f"{input_name}_value" for input_name in self.op_input_names]
            + [f"{input_name}_shape" for input_name in self.op_input_names]
            + ["attr_num_heads", "attr_rotary_embedding_dim"]
        )
