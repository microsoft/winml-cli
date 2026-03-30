# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ...config import WinMLBuildConfig
from ...optim import WinMLOptimizationConfig


# =============================================================================
# WinML Build Config
# =============================================================================
VISION_ENCODER_DECODER_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        reshape_mergedreshape=True,
    ),
)
