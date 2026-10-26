# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model config introspection shared by loading and config orchestration."""

from annotationlib import get_annotations
from typing import Any


def _get_model_config_class(model_class: type) -> Any:
    """Resolve Transformers config classes despite deferred class annotations."""
    config_cls = getattr(model_class, "config_class", None)
    if (
        isinstance(model_class, type)
        and model_class.__module__.startswith("transformers.")
        and config_cls is getattr(super(model_class, model_class), "config_class", None)
    ):
        # Transformers 4.x reads __dict__["__annotations__"] before Python 3.14
        # evaluates it, incorrectly retaining the parent model's config class.
        annotated_config = get_annotations(model_class, eval_str=True).get("config")
        if isinstance(annotated_config, type):
            return annotated_config
    return config_cls
