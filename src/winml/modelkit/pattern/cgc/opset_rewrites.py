# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model-level opset metadata rewrites for CGC compatibility."""

from onnx import ModelProto


def deduplicate_opset_imports(model: ModelProto) -> ModelProto:
    """Remove identical model opset imports, rejecting conflicting versions."""
    versions: dict[str, int] = {}
    for opset in model.opset_import:
        if opset.domain in versions and versions[opset.domain] != opset.version:
            raise ValueError(
                f"Conflicting opset imports for domain {opset.domain!r}: "
                f"{versions[opset.domain]} and {opset.version}"
            )
        versions[opset.domain] = opset.version
    if len(versions) == len(model.opset_import):
        return model
    result = ModelProto()
    result.CopyFrom(model)
    del result.opset_import[:]
    seen: set[str] = set()
    for opset in model.opset_import:
        if opset.domain not in seen:
            result.opset_import.append(opset)
            seen.add(opset.domain)
    return result
