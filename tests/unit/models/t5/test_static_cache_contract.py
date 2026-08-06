# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static-cache contract tests for the T5 encoder-decoder wrapper.

These tests intentionally inspect source structure so they can run in minimal
environments where torch wheels are unavailable. The end-to-end export tests
exercise the same contract with the real libraries on supported hosts.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
T5_SOURCE = REPO_ROOT / "src" / "winml" / "modelkit" / "models" / "hf" / "t5.py"
KV_CACHE_SOURCE = REPO_ROOT / "src" / "winml" / "modelkit" / "models" / "winml" / "kv_cache.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls.name}.{name} not found")


def test_t5_decoder_declares_cache_position_input() -> None:
    inputs = _method(_class(_tree(T5_SOURCE), "T5DecoderIOConfig"), "inputs")

    string_constants = {
        node.value
        for node in ast.walk(inputs)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "cache_position" in string_constants


def test_t5_model_defaults_to_static_cache() -> None:
    get_cache_class = _method(_class(_tree(T5_SOURCE), "WinMLT5Model"), "get_cache_class")

    returns = [node.value for node in ast.walk(get_cache_class) if isinstance(node, ast.Return)]

    assert any(isinstance(value, ast.Name) and value.id == "WinMLStaticCache" for value in returns)


def test_t5_decoder_forward_uses_static_cache_and_threads_trace_position() -> None:
    forward = _method(_class(_tree(T5_SOURCE), "T5DecoderWrapper"), "forward")

    names = {node.id for node in ast.walk(forward) if isinstance(node, ast.Name)}
    called_attrs = {
        node.func.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "WinMLStaticCache" in names
    assert "set_trace_position" in called_attrs


def test_t5_decoder_forward_reads_cache_position_before_past_kv() -> None:
    forward = _method(_class(_tree(T5_SOURCE), "T5DecoderWrapper"), "forward")
    assignments = {
        target.id: value
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        for target, value in [(node.targets[0], node.value)]
        if isinstance(target, ast.Name)
    }

    cache_position = assignments["cache_position"]
    kv_start = assignments["kv_start"]

    assert isinstance(cache_position, ast.Subscript)
    assert isinstance(cache_position.value, ast.Name)
    assert cache_position.value.id == "args"
    assert isinstance(cache_position.slice, ast.Constant)
    assert cache_position.slice.value == 4
    assert isinstance(kv_start, ast.Constant)
    assert kv_start.value == 5


def test_static_cache_trace_position_drives_hf_seq_length_queries() -> None:
    static_cache = _class(_tree(KV_CACHE_SOURCE), "WinMLStaticCache")
    get_seq_length = _method(static_cache, "get_seq_length")

    attrs = {node.attr for node in ast.walk(get_seq_length) if isinstance(node, ast.Attribute)}
    string_constants = {
        node.value
        for node in ast.walk(get_seq_length)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "_trace_position" in attrs | string_constants
    assert "step" in attrs


def test_cache_base_normalizes_traced_dimensions_before_early_initialization() -> None:
    tree = _tree(KV_CACHE_SOURCE)
    cache_base = _class(tree, "WinMLCache")
    early_initialization = _method(cache_base, "early_initialization")
    module_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    normalized_args = {
        keyword.arg
        for call in ast.walk(early_initialization)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "early_initialization"
        for keyword in call.keywords
        if isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "_as_static_dim"
    }

    assert "_as_static_dim" in module_functions
    assert normalized_args == {"batch_size", "num_heads", "head_dim"}
