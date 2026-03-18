"""Pattern entities - PatternType, Pattern, SubgraphPattern, OperatorPattern.

These pure Pydantic data models are shared between modelkit.pattern and
modelkit.analyze.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class PatternType(str, Enum):
    """Pattern type enum."""

    OPERATOR = "operator"
    SUBGRAPH = "subgraph"


class Pattern(BaseModel):
    """Base class for all pattern types (Pydantic model, not the ABC Pattern).

    Attributes:
        pattern_id: Pattern identifier
        pattern_type: Type of pattern (operator or subgraph)
        description: Pattern explanation
    """

    pattern_id: str = Field(..., description="Pattern identifier")
    pattern_type: PatternType = Field(..., description="Pattern type")
    description: str = Field(default="", description="Pattern explanation")
    semantic_label: str | None = Field(
        default=None, description="Optional pytorch label for the pattern match"
    )

    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type_consistency(cls, v: PatternType, info: object) -> PatternType:
        """Validate pattern_type matches pattern_id prefix."""
        if hasattr(info, "data") and "pattern_id" in info.data:
            pattern_id = info.data["pattern_id"]
            if v == PatternType.OPERATOR and not pattern_id.startswith("OP/"):
                raise ValueError(
                    f"Pattern type 'operator' requires pattern_id "
                    f"starting with 'OP/', got {pattern_id}"
                )
            if v == PatternType.SUBGRAPH and not pattern_id.startswith("SUBGRAPH/"):
                raise ValueError(
                    f"Pattern type 'subgraph' requires pattern_id "
                    f"starting with 'SUBGRAPH/', got {pattern_id}"
                )
        return v


class SubgraphPattern(Pattern):
    """Represents multi-operator subgraph pattern with topology.

    Attributes:
        pattern_id: Pattern identifier in format SUBGRAPH/<name>
        pattern_name: Human-readable name (e.g., "Gelu_Erf_Based")
        node_topology: Dict mapping node roles to operator types
        edge_topology: List of edges between nodes
    """

    pattern_id: str = Field(..., pattern=r"^SUBGRAPH/[^/]+$", description="Pattern ID")
    pattern_type: PatternType = Field(default=PatternType.SUBGRAPH, description="Pattern type")
    pattern_name: str = Field(..., description="Human-readable pattern name")
    # Topology fields are optional when using semantic_label for hierarchy_tag matching
    node_topology: dict[str, str] = Field(
        default_factory=dict, description="Node role to op type mapping"
    )
    edge_topology: list[tuple[str, str]] = Field(
        default_factory=list, description="Edge list between nodes"
    )


class OperatorPattern(Pattern):
    """Represents operator-level pattern with unique combination of properties.

    Attributes:
        pattern_id: Pattern identifier in format OP/<namespace>/<op_type>
        namespace: ONNX operator namespace (e.g., "ai.onnx")
        op_type: Operator type name (e.g., "Conv")
    """

    pattern_id: str = Field(..., pattern=r"^OP/[^/]+/[^/]+$", description="Pattern ID")
    pattern_type: PatternType = Field(default=PatternType.OPERATOR, description="Pattern type")
    namespace: str = Field(..., description="ONNX operator namespace")
    op_type: str = Field(..., description="Operator type name")

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace is one of supported values."""
        valid_namespaces = {"ai.onnx", "com.microsoft"}
        if v not in valid_namespaces:
            raise ValueError(f"Namespace must be one of {valid_namespaces}, got {v}")
        return v
