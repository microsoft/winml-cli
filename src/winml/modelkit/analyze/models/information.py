"""Information entity.

Represents actionable information for fixing unsupported patterns.
"""

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from .runtime_checks import PatternRuntime
from .support_level import SupportLevel


class ActionLevel(str, Enum):
    """Action priority level."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    WARNING = "warning"


class ActionItem(BaseModel):
    """Represents a specific transformation or optimization step.

    Attributes:
        type: Type of transformation (e.g., OrtTransformersOptimization)
        optimization_options: Configuration options for the transformation
    """

    type: str = Field(
        ..., description="Type of transformation or optimizatio, e.g. Olive pass name"
    )
    optimization_options: dict[str, object] | None = Field(
        default=None, description="Configuration options"
    )


class Action(BaseModel):
    """Represents a specific action to improve pattern support.

    Attributes:
        action_id: Unique action identifier (UUID)
        pattern_from_id: Original pattern identifier
        pattern_to_id: Target pattern identifier after transformation
        level: Action priority level
        action_items: List of transformation steps
        status: Expected support level after applying action
        enabled: Whether this action is enabled
        details: Detailed explanation of the action
    """

    action_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique action ID")
    pattern_from_id: str = Field(..., description="Original pattern identifier")
    pattern_to_id: str = Field(..., description="Target pattern identifier")
    level: ActionLevel | None = Field(default=None, description="Action priority level")
    action_items: list[ActionItem] = Field(default_factory=list, description="List of action steps")
    status: SupportLevel | None = Field(
        default=None, description="Expected support level after action"
    )
    enabled: bool = Field(default=True, description="Whether this action is enabled")
    details: str = Field(..., description="Detailed explanation")


class Information(BaseModel):
    """Represents actionable information for fixing unsupported patterns.

    Attributes:
        Information_id: Unique identifier (UUID)
        Information_type: Action priority
        actions: What to do
        explanation: Detailed explanation
        pattern_id: Pattern ID this applies to
    """

    Information_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID")
    actions: list[Action] | None = Field(default=None, description="What to do")
    explanation: str = Field(..., description="Detailed explanation")
    pattern_id: str | None = Field(default=None, description="Original pattern identifier")
    enabled: bool = Field(default=True, description="Whether this information is enabled")
    status: SupportLevel | None = Field(
        default=None, description="Support status for current pattern"
    )
    pattern_list: list[PatternRuntime] = Field(
        default_factory=list, exclude=True, description="Private list of pattern runtime results"
    )
    pattern_node_list: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Node names from patterns, list of lists where each inner list contains node names"
        ),
    )

    def model_post_init(self, __context: object) -> None:
        """Post-initialization to compute pattern_node_list from pattern_list."""
        if self.pattern_list and not self.pattern_node_list:
            self.pattern_node_list = [
                [node.node_name for node in pattern.pattern_match.matched_node_names]
                for pattern in self.pattern_list
                if pattern.pattern_match is not None
            ]
