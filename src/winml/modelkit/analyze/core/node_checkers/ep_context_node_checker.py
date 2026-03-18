# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any

from winml.modelkit.onnx.domains import ONNXDomain

from ...models.runtime_checks import PatternAlternative, PatternRuntime, RuntimeTestResult
from .base import NodeChecker
from .registry import NodeCheckerRegistry


if TYPE_CHECKING:
    import onnx

    from winml.modelkit.pattern.match import PatternMatchResult


@NodeCheckerRegistry.register_checker()
class EpContextNodeChecker(NodeChecker):
    """Checker for validating EPContext nodes based on their attributes.

    This checker applies to EPContext nodes in the com.microsoft domain and
    validates that the node's ``partition_name`` attribute is consistent with
    the execution provider name (``ep_name``). If ``partition_name`` starts
    with ``"<ep_name>_"``, the node is considered valid and eligible for
    execution and compilation; otherwise, it reports a mismatch in the
    runtime test result.
    """

    def can_check(
        self, node: "onnx.NodeProto", op_domain: "ONNXDomain", opset_version: int
    ) -> bool:
        return node.op_type == "EPContext" and op_domain == ONNXDomain.COM_MICROSOFT

    def check(
        self,
        node: "onnx.NodeProto",
        op_domain: "ONNXDomain",
        opset_version: int,
        pattern_match: "PatternMatchResult",
        alternatives: "list[PatternAlternative]",
        **kwargs: dict[str, Any],
    ) -> "PatternRuntime":
        ep_name = kwargs.get("ep_name")
        partition_name = self.get_attribute_value(node, "partition_name")
        if partition_name is None:
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=False,
                    compile=False,
                    no_data=True,
                    reason="Missing 'partition_name' attribute",
                ),
                alternatives=alternatives,
                pattern_match=pattern_match,
            )

        # https://github.com/microsoft/onnxruntime/blob/7e1d818ba7923c1e44187284a6ca77bbe13aa1eb/onnxruntime/core/framework/graph_partitioner.cc#L378
        if partition_name.startswith(ep_name + "_"):
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=True,
                    compile=True,
                    no_data=False,
                    reason=None,
                    filter=None,
                ),
                alternatives=alternatives,
                pattern_match=pattern_match,
            )
        return PatternRuntime(
            pattern_id=pattern_match.pattern.pattern_id,
            result=RuntimeTestResult(
                run=False,
                compile=False,
                no_data=False,
                reason=f"partition_name '{partition_name}' does not match ep_name '{ep_name}'",
            ),
            alternatives=alternatives,
            pattern_match=pattern_match,
        )

    def get_attribute_value(self, node: "onnx.NodeProto", attr_name: str) -> str | None:
        attr = next((attr for attr in node.attribute if attr.name == attr_name), None)
        if not attr:
            return None

        # Some attributes may not be of string type; guard against missing or non-byte "s".
        s_value = getattr(attr, "s", None)
        if isinstance(s_value, (bytes, bytearray)):
            return s_value.decode("utf-8")

        return None
