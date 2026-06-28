# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX domain identifiers shared across ModelKit modules.

This module provides the ONNXDomain enum, extracted from the analyze
onnx_opset package so that other packages (pattern, optim) can use it without
depending on the full analyze.

It also registers custom ONNX domain schemas (e.g., com.microsoft) from
ONNX Runtime so that get_schema() calls succeed for non-standard operators.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from onnx.defs import OpSchema, get_schema


if TYPE_CHECKING:
    from onnx import ModelProto


# ---------------------------------------------------------------------------
# Custom ONNX domain schema registration
# ---------------------------------------------------------------------------

_CUSTOM_DOMAINS_TO_REGISTER = {"com.microsoft"}

# Private cache for custom-domain schemas that we intentionally do NOT register
# with onnx.defs (because they share an op name with the default domain and
# onnx.reference.op_run._build_schemas() would crash).  Looked up by
# ONNXDomain.get_op_schema() as a fallback.
_custom_schema_cache: dict[tuple[str, str, int], OpSchema] = {}


def _init_custom_schemas(domains: set[str] | None = None) -> None:
    """Register custom ONNX domain operator schemas from ONNX Runtime.

    Uses ``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema``
    to fetch schemas for custom domains and registers them with ``onnx.defs``
    so that ``get_schema()`` calls succeed (e.g., for com.microsoft.Gelu).

    This is a no-op when ONNX Runtime is not installed.

    Args:
        domains: Set of domain names to register. Defaults to
                 ``_CUSTOM_DOMAINS_TO_REGISTER``.
    """
    if domains is None:
        domains = _CUSTOM_DOMAINS_TO_REGISTER

    if not domains:
        return

    try:
        from onnxruntime.capi.onnxruntime_pybind11_state import (  # type: ignore[import]
            get_all_operator_schema,
        )
    except ImportError:
        return

    import onnx.defs as _onnx_defs

    # Collect op names that already exist in the default ("") ONNX domain.
    # We must NOT register com.microsoft schemas for these ops because
    # onnx.reference.op_run._build_schemas() assumes every op name is unique
    # across all registered domains.  Registering a duplicate (e.g.
    # QLinearConv in both "" and "com.microsoft") causes a
    # NotImplementedError at import time for any code that touches
    # onnx.reference.
    _default_op_names: set[str] = {
        s.name for s in _onnx_defs.get_all_schemas_with_history() if s.domain == ""
    }

    for ort_schema in get_all_operator_schema():
        if ort_schema.domain not in domains:
            continue
        try:
            inputs = [
                OpSchema.FormalParameter(
                    inp.name,
                    inp.typeStr,
                    inp.description,
                    param_option=OpSchema.FormalParameterOption(inp.option.value),
                )
                for inp in ort_schema.inputs
            ]
            outputs = [
                OpSchema.FormalParameter(
                    out.name,
                    out.typeStr,
                    out.description,
                    param_option=OpSchema.FormalParameterOption(out.option.value),
                )
                for out in ort_schema.outputs
            ]
            type_constraints = [
                (tc.type_param_str, tc.allowed_type_strs, tc.description)
                for tc in ort_schema.type_constraints
            ]
            attributes = [
                OpSchema.Attribute(
                    attr.name,
                    OpSchema.AttrType(attr.type.value),
                    attr.description,
                    required=attr.required,
                )
                for attr in ort_schema.attributes.values()
            ]
            schema = OpSchema(
                ort_schema.name,
                ort_schema.domain,
                ort_schema.since_version,
                ort_schema.doc or "",
                inputs=inputs,
                outputs=outputs,
                type_constraints=type_constraints,
                attributes=attributes,
            )
            if ort_schema.name in _default_op_names:
                # Don't register with onnx.defs — would break
                # onnx.reference.op_run._build_schemas().  Cache privately
                # so ONNXDomain.get_op_schema() can still find it.
                _custom_schema_cache[
                    (ort_schema.name, ort_schema.domain, ort_schema.since_version)
                ] = schema
            else:
                _onnx_defs.register_schema(schema)
        except Exception:
            continue


# Register com.microsoft schemas eagerly so that get_schema() works for any
# module that imports ONNXDomain (e.g. modelkit.pattern.gelu_patterns).
_init_custom_schemas(_CUSTOM_DOMAINS_TO_REGISTER)


class ONNXDomain(StrEnum):
    """ONNX domain identifiers with helper methods for domain-specific operations.

    This enum encapsulates domain name conversions between different ONNX APIs:
    - User-facing domain: "ai.onnx", "com.microsoft"
    - Schema API domain: "", "com.microsoft" (empty string for default domain)
    - Model opset_import domain: "onnx", "com.microsoft"
    """

    AI_ONNX = "ai.onnx"
    COM_MICROSOFT = "com.microsoft"

    @property
    def schema_domain(self) -> str:
        """Get domain string for ONNX schema API (empty string for ai.onnx)."""
        return "" if self is self.__class__.AI_ONNX else self.value

    @property
    def name(self) -> str:
        """Get domain name string."""
        return self.value

    @classmethod
    def from_str(cls, domain_str: str) -> ONNXDomain:
        """Create ONNXDomain enum from domain string.

        Args:
            domain_str: Domain string (e.g., "ai.onnx", "com.microsoft")

        Returns:
            Corresponding ONNXDomain enum value

        Raises:
            ValueError: If the domain string is not recognized
        """
        if domain_str == "":
            return cls.AI_ONNX
        for domain in cls:
            if domain.value == domain_str:
                return domain
        raise ValueError(f"Unsupported ONNX domain: {domain_str}")

    def get_op_schema(self, op_name: str, opset_version: int) -> OpSchema:
        """Get operator schema for this domain.

        Tries ``onnx.defs.get_schema`` first.  Falls back to the private
        ``_custom_schema_cache`` for com.microsoft ops that share a name with
        the default domain (these are intentionally NOT registered in
        ``onnx.defs`` to avoid breaking ``onnx.reference``).

        Args:
            op_name: Name of the ONNX operator
            opset_version: ONNX opset version number

        Returns:
            OpSchema object for the specified operator
        """
        try:
            return get_schema(op_name, opset_version, self.schema_domain)
        except Exception:
            # onnx.defs.get_schema raises SchemaError (subclass of Exception,
            # NOT RuntimeError) when schema not found.  Fall back to private cache
            # for com.microsoft ops that share names with the default domain.
            # Cache keys use since_version; find highest version <= requested.
            best: OpSchema | None = None
            for (name, domain, ver), schema in _custom_schema_cache.items():
                if (
                    name == op_name
                    and domain == self.schema_domain
                    and ver <= opset_version
                    and (best is None or ver > best.since_version)
                ):
                    best = schema
            if best is not None:
                return best
            raise

    @classmethod
    def get_model_domain_opset_versions(cls, model: ModelProto) -> dict[ONNXDomain, int]:
        """Get opset version from model for this domain.

        Args:
            model: ONNX ModelProto object

        Returns:
            Opset version number for this domain, or -1 if not found
        """
        opset_versions = {}
        if hasattr(model, "opset_import"):
            for domain in cls:
                for opset in model.opset_import:
                    if domain.schema_domain == opset.domain:
                        opset_versions[domain] = opset.version
        return opset_versions
