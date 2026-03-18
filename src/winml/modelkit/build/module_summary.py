"""Module summary report generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path


def write_module_summary(
    output_path: Path,
    model_id: str,
    module_class: str,
    instances: list[dict[str, Any]],
) -> None:
    """Write module build summary to JSON.

    Args:
        output_path: Path to write the summary JSON.
        model_id: HuggingFace model ID.
        module_class: Module class name that was built.
        instances: List of per-instance result dicts.
    """
    summary = {
        "model_id": model_id,
        "module_class": module_class,
        "instance_count": len(instances),
        "instances": instances,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
