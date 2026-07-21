# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatible re-export for runtime checker result sanitizers."""

from ...utils.result_sanitizer import sanitize_check_result_payload, sanitize_result_text


__all__ = ["sanitize_check_result_payload", "sanitize_result_text"]
