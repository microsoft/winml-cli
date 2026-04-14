# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HF module hierarchy extraction.

Extracts the HuggingFace module hierarchy from a model,
filtering out standard torch.nn modules.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .types import HierarchyInfo, ModuleInfo


logger = logging.getLogger(__name__)


# Standard torch.nn module classes to filter out
# These are the "leaf" modules that don't contain HF-specific logic
TORCH_NN_MODULES: set[type] = {
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.LSTMCell,
    nn.GRUCell,
    nn.RNNCell,
    nn.MultiheadAttention,
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Tanh,
    nn.Sigmoid,
    nn.Softmax,
    nn.LogSoftmax,
    nn.Softplus,
    nn.Softsign,
    nn.PReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.CELU,
    nn.Hardtanh,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.Flatten,
    nn.Unflatten,
    nn.Identity,
    nn.ModuleList,
    nn.ModuleDict,
    nn.Sequential,
    nn.ParameterList,
    nn.ParameterDict,
}


def _is_hf_module(module: nn.Module) -> bool:
    """Determine if a module is an HF-specific module.

    Strategy:
    1. Check if module class is in TORCH_NN_MODULES set -> exclude
    2. Check if module's class is defined in torch.nn -> exclude
    3. Otherwise -> include (assumed to be HF-specific)

    Args:
        module: PyTorch module to check

    Returns:
        True if this is an HF-specific module, False if torch.nn
    """
    module_class = type(module)

    # Direct check against known torch.nn modules
    if module_class in TORCH_NN_MODULES:
        return False

    # Check module's defining package
    module_package = module_class.__module__

    # Exclude if defined in torch.nn
    if module_package.startswith("torch.nn"):
        return False

    # Include if from transformers or other HF packages
    if module_package.startswith("transformers"):
        return True

    # Include other custom modules (could be HF-derived)
    return True


def extract_hierarchy(model_id: str, trust_remote_code: bool = False) -> HierarchyInfo:
    """Extract the HF module hierarchy from a model.

    If the model is already cached locally, loads pretrained weights.
    Otherwise, uses AutoModel.from_config() with random weights to avoid
    downloading large model weight files.

    Args:
        model_id: HuggingFace model identifier
        trust_remote_code: If True, allow running custom code from the model repository

    Returns:
        HierarchyInfo with the module hierarchy
    """
    logger.info("Extracting hierarchy for: %s", model_id)

    # Try to load from local cache first
    try:
        logger.debug("Checking if model is cached locally...")
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        logger.info("Using cached pretrained model")
    except Exception as e:
        # Model not cached locally or other loading issue - use random weights
        logger.debug("Model not cached or load failed (%s), using random weights", type(e).__name__)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModel.from_config(config)
        logger.info("Using random weights (model not downloaded)")

    model.eval()

    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Counters
    hf_module_count = 0
    nn_module_count = 0

    def process_module(
        name: str,
        module: nn.Module,
        depth: int,
        parent_path: str = "",
    ) -> ModuleInfo | None:
        """Recursively process modules, filtering out torch.nn modules."""
        nonlocal hf_module_count, nn_module_count

        full_path = f"{parent_path}.{name}" if parent_path else name
        class_name = type(module).__name__

        # Check if this is an HF module
        is_hf = _is_hf_module(module)

        if not is_hf:
            nn_module_count += 1
            # Still process children — they might have HF modules nested.
            # Collect any HF descendants and propagate them up to the
            # nearest HF ancestor by returning a sentinel list.
            hf_children: list[ModuleInfo] = []
            for child_name, child_module in module.named_children():
                result = process_module(
                    child_name,
                    child_module,
                    depth,  # don't increment depth for skipped nn modules
                    full_path,
                )
                if isinstance(result, list):
                    # Grandchildren propagated up from a deeper nn module
                    hf_children.extend(result)
                elif result is not None:
                    hf_children.append(result)
            # Return list of HF children (not None) so parent can adopt them
            return hf_children if hf_children else None

        hf_module_count += 1

        # Count parameters in this module only (excluding children)
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))

        # Process children (handle both single ModuleInfo and list from skipped nn modules)
        children: list[ModuleInfo] = []
        for child_name, child_module in module.named_children():
            result = process_module(
                child_name,
                child_module,
                depth + 1,
                full_path,
            )
            if isinstance(result, list):
                # HF children propagated up from a skipped nn module (ModuleList, Sequential)
                children.extend(result)
            elif result is not None:
                children.append(result)

        return ModuleInfo(
            name=name,
            class_name=class_name,
            module_path=full_path,
            depth=depth,
            num_parameters=direct_params,
            children=children,
        )

    # Process root modules
    hf_modules: list[ModuleInfo] = []
    for name, module in model.named_children():
        result = process_module(name, module, depth=0)
        if isinstance(result, list):
            hf_modules.extend(result)
        elif result is not None:
            hf_modules.append(result)

    logger.info(
        "Hierarchy extracted: %d HF modules, %d nn modules filtered",
        hf_module_count,
        nn_module_count,
    )

    return HierarchyInfo(
        root_class=type(model).__name__,
        total_parameters=total_params,
        hf_modules=hf_modules,
        hf_module_count=hf_module_count,
        nn_module_count=nn_module_count,
    )


def flatten_hierarchy(modules: list[ModuleInfo], max_depth: int = -1) -> list[ModuleInfo]:
    """Flatten the hierarchy tree into a list for display.

    Args:
        modules: List of ModuleInfo trees
        max_depth: Maximum depth to include (-1 for unlimited)

    Returns:
        Flattened list of ModuleInfo in tree order
    """
    result: list[ModuleInfo] = []

    def walk(module: ModuleInfo, depth: int = 0) -> None:
        if max_depth >= 0 and depth > max_depth:
            return
        result.append(module)
        for child in module.children:
            walk(child, depth + 1)

    for module in modules:
        walk(module)

    return result
