# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom node checker protocol and registry for RuntimeCheckerQuery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import onnx

    from winml.modelkit.onnx.domains import ONNXDomain
    from winml.modelkit.pattern.match import PatternMatchResult

    from ...models.runtime_checks import PatternAlternative, PatternRuntime


class NodeChecker(ABC):
    """Base class for custom node checkers.

    Custom checkers can intercept node checking logic before the standard
    negative rule + table lookup. This allows for specialized checking logic
    that doesn't fit the standard pattern.

    To create a custom checker:
    1. Subclass NodeChecker
    2. Implement can_check() to determine if this checker handles the node
    3. Implement check() with your custom logic
    4. Register via RuntimeCheckerQuery.register_custom_checker()

    Example:
        class MyOpChecker(NodeChecker):
            def can_check(self, node, op_domain):
                return node.op_type == "MyOp"

            def check(self, node, conditions, pattern_match, alternatives):
                # Custom logic here
                return PatternRuntime(...)
    """

    @abstractmethod
    def can_check(self, node: onnx.NodeProto, op_domain: ONNXDomain, opset_version: int) -> bool:
        """Determine if this checker can handle the node.

        Args:
            node: ONNX node to check
            op_domain: ONNX domain of the node

        Returns:
            True if this checker should handle the node, False otherwise
        """
        ...

    @abstractmethod
    def check(
        self,
        node: onnx.NodeProto,
        op_domain: ONNXDomain,
        opset_version: int,
        pattern_match: PatternMatchResult,
        alternatives: list[PatternAlternative],
        **kwargs: dict[str, Any],
    ) -> PatternRuntime:
        """Execute custom checking logic for the node.

        Args:
            node: ONNX node to check
            conditions: Extracted node conditions (attributes, inputs, etc.)
            pattern_match: Pattern match information for the node
            alternatives: List of pattern alternatives

        Returns:
            PatternRuntime with check results if handled, None to fall through
            to default negative rule + table logic
        """
        ...
