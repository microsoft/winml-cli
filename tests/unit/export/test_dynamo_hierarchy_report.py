# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Writer-level tests for the dynamo module-hierarchy report.

These exercise the actual ``MetadataWriter`` tree output (not just the flat map
``build_module_hierarchy`` returns), covering the two failure modes the shared
writers previously had:

  * same-class named siblings (an attention block's query/key/value Linears)
    collapsing to a single tree entry because children were keyed by class name,
    and
  * a sparse hierarchy whose intermediate container scope (a ``ModuleList``)
    never emitted its own entry, orphaning the whole subtree from the tree.
"""

from __future__ import annotations

from onnx import ModelProto, NodeProto, StringStringEntryProto, helper

from winml.modelkit.core.hierarchy_utils import find_immediate_children
from winml.modelkit.core.onnx_node_tagger import DynamoMetadataTagger
from winml.modelkit.export.htp.metadata_writer import MetadataWriter
from winml.modelkit.export.htp.step_data import ModuleInfo


NAME_SCOPES_KEY = "pkg.torch.onnx.name_scopes"
CLASS_HIERARCHY_KEY = "pkg.torch.onnx.class_hierarchy"


def _node(op_type: str, name: str, name_scopes: list[str], class_hierarchy: list[str]) -> NodeProto:
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], name=name)
    node.metadata_props.append(StringStringEntryProto(key=NAME_SCOPES_KEY, value=repr(name_scopes)))
    node.metadata_props.append(
        StringStringEntryProto(key=CLASS_HIERARCHY_KEY, value=repr(class_hierarchy))
    )
    return node


def _model(nodes: list[NodeProto]) -> ModelProto:
    return helper.make_model(helper.make_graph(nodes, "g", inputs=[], outputs=[]))


def _attention_model() -> ModelProto:
    """One block with an attention module exposing query/key/value Linears.

    The ``blocks`` ModuleList never appears as its own scope (torch skips
    container modules), so the root's only child is the compound ``blocks.0``.
    """
    return _model(
        [
            _node(
                "MatMul",
                f"n_{name}",
                ["", "blocks.0", "blocks.0.attn", f"blocks.0.attn.{name}", "linear"],
                [
                    "pkg.Net",
                    "pkg.Blk",
                    "pkg.Attention",
                    "torch.nn.modules.linear.Linear",
                    "aten.linear.default",
                ],
            )
            for name in ("query", "key", "value")
        ]
    )


def _to_module_info(flat: dict[str, dict]) -> dict[str, ModuleInfo]:
    return {
        scope: ModuleInfo(
            class_name=info["class_name"],
            traced_tag=info["traced_tag"],
            execution_order=info["execution_order"],
        )
        for scope, info in flat.items()
    }


class TestFindImmediateChildrenSparse:
    """Nearest-present-ancestor nesting for sparse/compound scopes."""

    def test_compound_root_child_attaches_to_root(self) -> None:
        # "blocks.0" has no "blocks" ancestor entry, so it is a root child and
        # its subtree is not dropped.
        hierarchy = {"": {}, "blocks.0": {}, "blocks.0.attn": {}}
        assert find_immediate_children("", hierarchy) == ["blocks.0"]
        assert find_immediate_children("blocks.0", hierarchy) == ["blocks.0.attn"]

    def test_present_container_reparents_index(self) -> None:
        # When the "blocks" container IS present, "blocks.0" nests under it.
        hierarchy = {"": {}, "blocks": {}, "blocks.0": {}}
        assert find_immediate_children("", hierarchy) == ["blocks"]
        assert find_immediate_children("blocks", hierarchy) == ["blocks.0"]


class TestMetadataWriterTree:
    """The persisted MetadataWriter tree preserves every reconstructed module."""

    def _tree(self) -> dict:
        flat = DynamoMetadataTagger().build_module_hierarchy(_attention_model())
        writer = MetadataWriter("unused.json")
        return writer._build_hierarchical_modules(_to_module_info(flat))

    def test_same_class_siblings_all_serialized(self) -> None:
        tree = self._tree()
        # root -> Blk.0 -> Attention -> {Linear.query, Linear.key, Linear.value}
        attn = tree["children"]["Blk.0"]["children"]["Attention"]
        linears = attn["children"]
        assert set(linears) == {"Linear.query", "Linear.key", "Linear.value"}
        scopes = {child["scope"] for child in linears.values()}
        assert scopes == {
            "blocks.0.attn.query",
            "blocks.0.attn.key",
            "blocks.0.attn.value",
        }

    def test_sparse_root_subtree_present(self) -> None:
        tree = self._tree()
        # The compound root child "blocks.0" is present, not orphaned.
        assert "Blk.0" in tree["children"]
        assert tree["children"]["Blk.0"]["scope"] == "blocks.0"
