# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml optimize`` CLI command.

These tests use synthetic ONNX models built in-test via ``onnx.helper`` — no
network, no HuggingFace download.

Success criteria for any successful invocation:
    * Command exits with code 0.
    * The requested (or default) ONNX file exists at the expected path.
    * Test-specific extra invariants (per-case).

Failure criteria for any failing invocation:
    * Command exits with a non-zero code.
    * No ONNX file is left where success would have placed it.

Markers:
    e2e: End-to-end test invoking the CLI through CliRunner.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.commands.optimize import optimize


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.timeout(120)]


# ---------------------------------------------------------------------------
# CLI helpers (DRY)
# ---------------------------------------------------------------------------


def _invoke(args: list[str], *, catch: bool = False):
    """Invoke the optimize CLI with a fresh runner and standard ctx.obj."""
    runner = CliRunner()
    return runner.invoke(optimize, args, obj={"debug": False}, catch_exceptions=catch)


def _assert_succeeds(args: list[str], output_path: Path) -> onnx.ModelProto:
    """Assert exit==0, output ONNX exists, return the loaded optimized model."""
    result = _invoke(args)
    assert result.exit_code == 0, f"optimize failed (exit {result.exit_code}):\n{result.output}"
    assert output_path.exists(), f"optimized ONNX not found at {output_path}"
    return onnx.load(str(output_path))


def _count_op(model: onnx.ModelProto, op_type: str) -> int:
    return sum(1 for n in model.graph.node if n.op_type == op_type)


def _ops(model: onnx.ModelProto) -> list[str]:
    return [n.op_type for n in model.graph.node]


def _default_opt_path(input_path: Path) -> Path:
    """The path the CLI writes to when ``-o`` is omitted."""
    return input_path.parent / f"{input_path.stem}_opt.onnx"


def _save(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, str(path))
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Synthetic ONNX builders + fixtures
# ---------------------------------------------------------------------------


def _build_const_folding_model() -> onnx.ModelProto:
    """``out = x + ConstantOfShape(shape=[2,3], value=7.0)``.

    Per the ONNX spec, ConstantOfShape takes a single 1-D int64 input whose
    *values* are the desired output dimensions, and a ``value`` attribute
    (a length-1 1-D or 0-D tensor) holding the scalar fill value. Here we
    ask for a 2x3 float32 tensor filled with 7.0. The shape input is a
    constant initializer, so constant-folding collapses the node into a
    literal initializer.
    """
    target_shape = [2, 3]
    fill_value = 7.0

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, target_shape)
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape)

    # 1-D int64 tensor whose values describe the desired output shape.
    shape_init = helper.make_tensor(
        "shape_init",
        TensorProto.INT64,
        [len(target_shape)],
        target_shape,
    )

    # The `value` attribute is a length-1 tensor carrying the fill scalar.
    value_tensor = helper.make_tensor("value", TensorProto.FLOAT, [1], [fill_value])
    cos = helper.make_node("ConstantOfShape", ["shape_init"], ["y"], name="cos")
    cos.attribute.append(helper.make_attribute("value", value_tensor))

    add = helper.make_node("Add", ["x", "y"], ["out"], name="add")
    graph = helper.make_graph([cos, add], "cst_folding", [x_info], [out_info], [shape_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_expanded_gelu_model() -> onnx.ModelProto:
    """Decomposed GELU pattern: ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    This is the canonical shape that ORT's ``GeluFusionL2`` (aka
    ``gelu-fusion``) collapses into a single ``Gelu`` op. Mirrors
    ``tests/unit/optim/assets/graphpipe/builders/gelu.gelu_fusion_builder``.
    """
    in_name, out_name, p = "x", "out", "g_"
    initializers = [
        numpy_helper.from_array(np.array([math.sqrt(2)], dtype=np.float32), f"{p}sqrt2"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), f"{p}one"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), f"{p}half"),
    ]
    nodes = [
        helper.make_node("Div", [in_name, f"{p}sqrt2"], [f"{p}div"], name=f"{p}div"),
        helper.make_node("Erf", [f"{p}div"], [f"{p}erf"], name=f"{p}erf"),
        helper.make_node("Add", [f"{p}erf", f"{p}one"], [f"{p}add1"], name=f"{p}add1"),
        helper.make_node("Mul", [in_name, f"{p}add1"], [f"{p}mul1"], name=f"{p}mul1"),
        helper.make_node("Mul", [f"{p}mul1", f"{p}half"], [out_name], name=f"{p}mul2"),
    ]
    in_info = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 64])
    out_info = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, 64])
    graph = helper.make_graph(nodes, "expanded_gelu", [in_info], [out_info], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.fixture
def cst_folding_path(tmp_path: Path) -> Path:
    """Path to a freshly-written ``cst_folding.onnx`` for the test."""
    return _save(_build_const_folding_model(), tmp_path / "cst_folding.onnx")


@pytest.fixture
def expanded_gelu_path(tmp_path: Path) -> Path:
    """Path to a freshly-written ``expanded_gelu.onnx`` for the test."""
    return _save(_build_expanded_gelu_model(), tmp_path / "expanded_gelu.onnx")


# ===========================================================================
# Documentation commands
# ===========================================================================


class TestOptimizeDoc:
    """The three meta commands each render their corresponding documentation."""

    def test_help(self):
        result = _invoke(["--help"])
        assert result.exit_code == 0, f"--help failed:\n{result.output}"
        assert "Usage:" in result.output
        assert "--enable-constant-folding" in result.output
        # The default annotation added to capability_options must surface here.
        assert "Default: enabled" in result.output
        assert "Default: disabled" in result.output

    def test_list_capabilities(self):
        result = _invoke(["--list-capabilities"])
        assert result.exit_code == 0, f"--list-capabilities failed:\n{result.output}"
        assert "Available optimization flags" in result.output
        # constant-folding is the lone default-enabled cap; the compact listing
        # shows the "flip-the-default" form (--disable-...).
        assert "--disable-constant-folding" in result.output

    def test_list_rewrites(self):
        result = _invoke(["--list-rewrites"])
        assert result.exit_code == 0, f"--list-rewrites failed:\n{result.output}"
        assert "Rewrite capabilities" in result.output


# ===========================================================================
# Happy path 1: constant folding (default-enabled)
# ===========================================================================


class TestOptimizeConstFolding:
    """Happy-path 1: ``winml optimize -m cst_folding.onnx`` and its variants.

    Default behaviour: ``constant-folding`` is on, so the ``ConstantOfShape``
    node disappears from the optimized graph.
    """

    def test_minimal(self, cst_folding_path: Path):
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(["-m", str(cst_folding_path)], out)
        assert _count_op(model, "ConstantOfShape") == 0, (
            f"expected ConstantOfShape to be folded by default; ops: {_ops(model)}"
        )

    def test_missing_model_fails(self, tmp_path: Path):
        # No -m supplied: command must reject and write no ONNX anywhere.
        result = _invoke([], catch=True)
        assert result.exit_code != 0, (
            f"expected failure for missing -m, got exit=0:\n{result.output}"
        )
        leftover = list(tmp_path.glob("*.onnx"))
        assert not leftover, f"unexpected ONNX files in {tmp_path}: {leftover}"

    def test_custom_output(self, cst_folding_path: Path, tmp_path: Path):
        custom = tmp_path / "explicit.onnx"
        model = _assert_succeeds(["-m", str(cst_folding_path), "-o", str(custom)], custom)
        # -o redirects: the default-named file must NOT be created.
        assert not _default_opt_path(cst_folding_path).exists(), (
            "-o specified but default `{stem}_opt.onnx` was still written"
        )
        assert _count_op(model, "ConstantOfShape") == 0

    def test_verbose(self, cst_folding_path: Path):
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(["-m", str(cst_folding_path), "-v"], out)
        assert _count_op(model, "ConstantOfShape") == 0

    def test_enable_constant_folding(self, cst_folding_path: Path):
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(["-m", str(cst_folding_path), "--enable-constant-folding"], out)
        assert _count_op(model, "ConstantOfShape") == 0

    def test_disable_constant_folding(self, cst_folding_path: Path):
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(["-m", str(cst_folding_path), "--disable-constant-folding"], out)
        assert _count_op(model, "ConstantOfShape") == 1, (
            "expected ConstantOfShape to survive with --disable-constant-folding; "
            f"ops: {_ops(model)}"
        )

    def test_config_disables_folding(self, cst_folding_path: Path, tmp_path: Path):
        cfg = _write_json(tmp_path / "opt.json", {"constant-folding": False})
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(["-m", str(cst_folding_path), "--config", str(cfg)], out)
        assert _count_op(model, "ConstantOfShape") == 1, (
            "config `constant-folding: false` should leave ConstantOfShape intact; "
            f"ops: {_ops(model)}"
        )

    def test_cli_overrides_config(self, cst_folding_path: Path, tmp_path: Path):
        cfg = _write_json(tmp_path / "opt.json", {"constant-folding": False})
        out = _default_opt_path(cst_folding_path)
        model = _assert_succeeds(
            [
                "-m",
                str(cst_folding_path),
                "--config",
                str(cfg),
                "--enable-constant-folding",
            ],
            out,
        )
        # CLI --enable-* must override the config's `false`.
        assert _count_op(model, "ConstantOfShape") == 0, (
            f"CLI --enable-constant-folding must override config false; ops: {_ops(model)}"
        )


# ===========================================================================
# Happy path 2: GELU fusion (default-disabled)
# ===========================================================================


class TestOptimizeGeluFusion:
    """Happy-path 2: ``winml optimize -m expanded_gelu.onnx`` and its variants.

    Default behaviour: ``gelu-fusion`` is off, so the expanded pattern is
    preserved (the single Erf node remains in the optimized graph).
    """

    def test_minimal(self, expanded_gelu_path: Path):
        out = _default_opt_path(expanded_gelu_path)
        model = _assert_succeeds(["-m", str(expanded_gelu_path)], out)
        assert _count_op(model, "Erf") == 1, (
            f"expected Erf to survive (gelu-fusion default-off); ops: {_ops(model)}"
        )

    def test_enable_gelu_fusion(self, expanded_gelu_path: Path):
        # --enable-gelu-fusion fuses the expanded pattern → single Gelu, no Erf.
        out = _default_opt_path(expanded_gelu_path)
        model = _assert_succeeds(["-m", str(expanded_gelu_path), "--enable-gelu-fusion"], out)
        assert _count_op(model, "Erf") == 0, f"expected Erf to be fused away; ops: {_ops(model)}"

    def test_disable_gelu_fusion(self, expanded_gelu_path: Path):
        # --disable-gelu-fusion explicitly off → pattern preserved, Erf survives.
        out = _default_opt_path(expanded_gelu_path)
        model = _assert_succeeds(["-m", str(expanded_gelu_path), "--disable-gelu-fusion"], out)
        assert _count_op(model, "Erf") == 1, f"expected Erf to survive; ops: {_ops(model)}"
