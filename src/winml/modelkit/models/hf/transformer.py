# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom tabular Transformer export support.

The SnowFlash383935/DigitalEduTransformers checkpoint publishes a custom
``model_type='transformer'`` architecture whose public ``forward`` accepts a
Python list, performs NumPy normalization, and returns Python booleans. That is
useful for the model card example but is not an ONNX export contract. The
wrapper below exposes the tensor path directly: normalized tabular features in,
logits out.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator

from ...export import register_onnx_overwrite


def _ensure_legacy_remote_model_compat(
    auto_model: Any,
    config: Any,
    *,
    trust_remote_code: bool,
) -> None:
    model_class = type(
        auto_model.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}


class TabularInputGenerator(DummyInputGenerator):  # type: ignore[misc]
    """Generate floating tabular feature tensors."""

    SUPPORTED_INPUT_NAMES = ("features",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **_: Any,
    ) -> None:
        """Initialize the generator from Optimum's OnnxConfig factory args."""
        self.task = task
        self.normalized_config = normalized_config
        self.batch_size = batch_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> Any:
        """Generate a zero-valued tabular feature tensor."""
        import torch

        if input_name != "features":
            raise ValueError(f"Unsupported input for tabular transformer: {input_name}")
        input_dim = int(getattr(self.normalized_config, "input_dim", 7))
        return torch.zeros((self.batch_size, input_dim), dtype=torch.float32)


class TabularTransformerWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper around the remote tabular model."""

    main_output_name = "logits"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.config = model.config
        self.input_proj = cast("nn.Module", model.input_proj)
        self.transformer = cast("nn.Module", model.transformer)
        self.head = cast("nn.Module", model.head)
        mean = torch.tensor(self.config.mean, dtype=torch.float32)
        std = torch.tensor(self.config.std, dtype=torch.float32)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> TabularTransformerWrapper:
        """Load the remote model and wrap its tensorizable submodules."""
        from transformers import AutoConfig, AutoModel

        from ...loader import load_hf_config

        config = kwargs.pop("config", None)
        if config is None:
            config = load_hf_config(
                AutoConfig,
                model_name_or_path,
                trust_remote_code=kwargs.get("trust_remote_code", False),
                revision=kwargs.get("revision"),
            )

        _ensure_legacy_remote_model_compat(
            AutoModel,
            config,
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

        model = AutoModel.from_pretrained(model_name_or_path, config=config, **kwargs)
        wrapper = cls(model)
        wrapper.eval()
        return wrapper

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run normalized tabular features through the Transformer classifier."""
        mean = cast("torch.Tensor", self.mean)
        std = cast("torch.Tensor", self.std)
        x = (features - mean.to(features.device, features.dtype)) / (
            std.to(features.device, features.dtype) + 1e-8
        )
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        if torch.onnx.is_in_onnx_export():
            fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                x = self.transformer(x)
            finally:
                torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
        else:
            x = self.transformer(x)
        x = x.squeeze(1)
        return cast("torch.Tensor", self.head(x))


@register_onnx_overwrite("transformer", "text-classification", library_name="transformers")
class TabularTransformerIOConfig(OnnxConfig):  # type: ignore[misc]
    """ONNX config for tensorized tabular classification."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        input_dim="input_dim",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (TabularInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """ONNX input names and dynamic axes."""
        return {"features": {0: "batch_size"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """ONNX output names and dynamic axes."""
        return {"logits": {0: "batch_size"}}


MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("transformer", "tabular-classification"): TabularTransformerWrapper,
    ("transformer", "text-classification"): TabularTransformerWrapper,
    ("transformer", None): TabularTransformerWrapper,
}


__all__ = [
    "MODEL_CLASS_MAPPING",
    "TabularInputGenerator",
    "TabularTransformerIOConfig",
    "TabularTransformerWrapper",
]
