# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Any


# Example: "(123 us)"
_DURATION_US_RE = re.compile(r"\(\s*\d+\s*us\)")

# Date + time styles we commonly see in ORT/EP logs.
_DATETIME_PATTERNS = (
    # 2026-07-08 13:23:06.596 / 2026/07/08 13:23:06 / 2026-07-08T13:23:06Z
    re.compile(
        r"\b\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?\b"
    ),
    # 7/8/2026 1:23:06.596 PM / 07-08-2026 13:23:06
    re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[T\s]\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:\s?(?:AM|PM))?\b",
        re.IGNORECASE,
    ),
    # Time only: 13:23:06.596
    re.compile(r"(?<!\d)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?!\d)"),
)


def sanitize_result_text(text: str) -> str:
    """Normalize volatile timing fragments in result/log text for stable diffs."""
    cleaned = _DURATION_US_RE.sub("", text)
    for pattern in _DATETIME_PATTERNS:
        cleaned = pattern.sub("<ts>", cleaned)
    return cleaned


def sanitize_check_result_payload(result: Any) -> None:
    """In-place sanitize of one runtime-check stage payload.

    Expected input shape:
      {
        "result": {"success": bool | None, "reason": str | None, ...},
        "stdout": str | None,
        "stderr": str | None,
      }
    """
    if not isinstance(result, dict):
        return

    for key in ("stdout", "stderr"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = sanitize_result_text(value)

    result_payload = result.get("result")
    if isinstance(result_payload, dict):
        reason = result_payload.get("reason")
        if isinstance(reason, str):
            result_payload["reason"] = sanitize_result_text(reason)


__all__ = ["sanitize_check_result_payload", "sanitize_result_text"]
