"""FusionPipe pattern builders for testing.

All builders follow the universal pattern:
    def builder(input_name, output_name, prefix, initializers, **kwargs) -> list[NodeProto]

Model factories create complete ModelProto:
    def create_model(**kwargs) -> ModelProto
"""

from .attention import (
    bert_attention_builder,
    create_bert_attention_model,
    create_gpt2_attention_model,
    gpt2_attention_builder,
)
from .layernorm import (
    create_decomposed_layernorm_model,
    create_simplified_layernorm_model,
    create_skip_layernorm_model,
    decomposed_layernorm_builder,
    simplified_layernorm_builder,
    skip_layernorm_builder,
)


__all__ = [
    "bert_attention_builder",
    "create_bert_attention_model",
    "create_decomposed_layernorm_model",
    "create_gpt2_attention_model",
    "create_simplified_layernorm_model",
    "create_skip_layernorm_model",
    "decomposed_layernorm_builder",
    "gpt2_attention_builder",
    "simplified_layernorm_builder",
    "skip_layernorm_builder",
]
