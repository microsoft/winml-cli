# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Evaluation Module.

Provides accuracy evaluation using HuggingFace pipeline + evaluate library.
"""

from .config import WinMLEvaluationConfig
from .evaluate import EvalResult, evaluate


__all__ = [
    "EvalResult",
    "WinMLEvaluationConfig",
    "evaluate",
]
