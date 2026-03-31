# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Exceptions for static analyzer."""


class OpOptionalInputSupportError(Exception):
    """Raised when optional attributes or inputs are not supported."""


class OpLackOfRequiredInformationError(Exception):
    """Raised when required information (shape, dtype, etc.) is missing from the model.

    This commonly occurs in:
    - Quantized models where DequantizeLinear outputs lack valueinfo
    - Models with incomplete shape inference
    - Models with dynamic/symbolic dimensions
    """


class OpUnsupportedError(Exception):
    """Raised when an unsupported operator is encountered."""
