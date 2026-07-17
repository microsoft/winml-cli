# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the export CLI command.

These tests exercise the full ``winml export`` pipeline with a real model
(``microsoft/resnet-50``) downloaded from HuggingFace Hub.

Success criteria for any successful invocation:
    * Command exits with code 0.
    * The requested ONNX file exists at the given path.
    * Test-specific extra invariants (per-case).

Failure criteria for any failing invocation:
    * Command exits with a non-zero code.
    * No ONNX file is left at the requested path.

Markers:
    e2e:     Full end-to-end test with real models
    slow:    Tests that take > 30 seconds
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import onnx
import pytest
from click.testing import CliRunner

from winml.modelkit.commands.export import export


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.network, pytest.mark.timeout(1800)]


_MODEL = "microsoft/resnet-50"

# Full WinMLBuildConfig used by ``-c PATH`` tests. Sets opset_version=18 so the
# resulting ONNX model can be verified via its default-domain opset.
_BUILD_CONFIG: dict = {
    "export": {
        "opset_version": 18,
        "batch_size": 1,
        "export_params": True,
        "do_constant_folding": True,
        "verbose": False,
        "dynamo": False,
        "enable_hierarchy_tags": True,
        "clean_onnx": False,
        "hierarchy_tag_format": "full",
        "input_tensors": [
            {
                "name": "pixel_values",
                "dtype": "float32",
                "shape": [1, 3, 224, 224],
                "value_range": [0, 1],
            }
        ],
        "output_tensors": [{"name": "logits"}],
    },
    "optim": {},
    "quant": None,
    "compile": None,
    "loader": {
        "task": "image-classification",
        "model_class": "AutoModelForImageClassification",
        "model_type": "resnet",
    },
}


# ---------------------------------------------------------------------------
# Helpers (DRY)
# ---------------------------------------------------------------------------


def _invoke(args: list[str], *, catch: bool = False):
    """Invoke the export CLI with a fresh runner and standard ctx.obj."""
    runner = CliRunner()
    return runner.invoke(export, args, obj={"debug": False}, catch_exceptions=catch)


def _happy_args(onnx_path: Path, *extra: str) -> list[str]:
    """Build the happy-path args ``-m <model> -o <onnx_path>`` plus any extras."""
    return ["-m", _MODEL, "-o", str(onnx_path), *extra]


def _assert_succeeds(args: list[str], onnx_path: Path) -> onnx.ModelProto:
    """Assert exit==0, ONNX file is produced, and return the loaded model."""
    result = _invoke(args)
    assert result.exit_code == 0, f"export failed (exit {result.exit_code}):\n{result.output}"
    assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"
    return onnx.load(str(onnx_path))


def _assert_fails(args: list[str], onnx_path: Path) -> None:
    """Assert exit!=0 and the ONNX file is absent."""
    result = _invoke(args, catch=True)
    assert result.exit_code != 0, f"expected failure, got exit=0:\n{result.output}"
    assert not onnx_path.exists(), f"ONNX file unexpectedly present at {onnx_path}"


def _opset_version(model: onnx.ModelProto) -> int:
    """Return the default-domain (ai.onnx) opset version of the model."""
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    msg = "default-domain opset import not found"
    raise AssertionError(msg)


def _node_has_metadata(node: onnx.NodeProto, key: str) -> bool:
    return any(prop.key == key for prop in node.metadata_props)


def _nodes_missing_metadata(model: onnx.ModelProto, key: str) -> list[onnx.NodeProto]:
    """Return the list of graph nodes that do NOT have ``key`` in metadata_props."""
    return [n for n in model.graph.node if not _node_has_metadata(n, key)]


def _nodes_with_metadata(model: onnx.ModelProto, key: str) -> list[onnx.NodeProto]:
    """Return the list of graph nodes that have ``key`` in metadata_props."""
    return [n for n in model.graph.node if _node_has_metadata(n, key)]


def _assert_all_nodes_have(model: onnx.ModelProto, key: str) -> None:
    """Assert every graph node has ``key`` in its metadata_props."""
    nodes = list(model.graph.node)
    assert nodes, "model has zero graph nodes"
    missing = _nodes_missing_metadata(model, key)
    assert not missing, (
        f"{len(missing)}/{len(nodes)} nodes missing metadata_props key {key!r}; "
        f"first few: {[(n.name, n.op_type) for n in missing[:5]]}"
    )


def _assert_no_node_has(model: onnx.ModelProto, key: str) -> None:
    """Assert no graph node has ``key`` in its metadata_props."""
    nodes = list(model.graph.node)
    assert nodes, "model has zero graph nodes"
    present = _nodes_with_metadata(model, key)
    assert not present, (
        f"{len(present)}/{len(nodes)} nodes unexpectedly have metadata_props key {key!r}; "
        f"first few: {[(n.name, n.op_type) for n in present[:5]]}"
    )


def _assert_some_node_has(model: onnx.ModelProto, key: str) -> None:
    """Assert at least one graph node has ``key`` in its metadata_props.

    Use only for keys that are intentionally per-subset (e.g. onnxscript
    rewriter rule_name, which only annotates rewritten nodes).
    """
    nodes = list(model.graph.node)
    assert nodes, "model has zero graph nodes"
    present = _nodes_with_metadata(model, key)
    assert present, (
        f"no node has metadata_props key {key!r}; "
        f"sample keys observed: "
        f"{sorted({p.key for n in nodes for p in n.metadata_props})[:10]}"
    )


def _output_names(model: onnx.ModelProto) -> list[str]:
    return [out.name for out in model.graph.output]


def _input_shape_dims(model: onnx.ModelProto) -> list[int]:
    """Return the first input's shape as a list of int dim_values (or -1)."""
    first = model.graph.input[0]
    return [
        d.dim_value if d.HasField("dim_value") else -1 for d in first.type.tensor_type.shape.dim
    ]


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _symbolic_dims(tensors, name: str) -> dict[int, str]:
    """Return ``{axis: dim_param}`` for the symbolic dims of the named tensor."""
    for tensor in tensors:
        if tensor.name == name:
            return {
                i: d.dim_param
                for i, d in enumerate(tensor.type.tensor_type.shape.dim)
                if d.dim_param
            }
    raise AssertionError(f"tensor {name!r} not found in {[t.name for t in tensors]}")


def _static_dims(tensors, name: str) -> dict[int, int]:
    """Return ``{axis: dim_value}`` for the static (non-symbolic) dims of the tensor."""
    for tensor in tensors:
        if tensor.name == name:
            return {
                i: d.dim_value
                for i, d in enumerate(tensor.type.tensor_type.shape.dim)
                if not d.dim_param
            }
    raise AssertionError(f"tensor {name!r} not found in {[t.name for t in tensors]}")


# ===========================================================================
# --help
# ===========================================================================


class TestExportHelp:
    """``winml export --help`` prints help and exits 0 without producing an ONNX."""

    def test_help_works(self):
        result = _invoke(["--help"])
        assert result.exit_code == 0, f"--help failed:\n{result.output}"
        # Help output should mention the command and at least one flag we care about
        assert "--model" in result.output
        assert "--output" in result.output


# ===========================================================================
# Happy path
# ===========================================================================


class TestExportHappyPath:
    """Minimal: ``winml export -m microsoft/resnet-50 -o <tmp>/model.onnx``."""

    def test_minimal_resnet50(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path), onnx_path)

        # Every graph node must have winml.hierarchy.tag and winml.hierarchy.depth.
        _assert_all_nodes_have(model, "winml.hierarchy.tag")
        _assert_all_nodes_have(model, "winml.hierarchy.depth")


class TestExportDinoV2:
    MODEL = "facebook/dinov2-base"

    def test_image_feature_extraction(self, tmp_path: Path):
        """``-t image-feature-extraction`` must produce a valid ONNX export."""
        onnx_path = tmp_path / "model.onnx"
        result = _invoke(["-m", self.MODEL, "-o", str(onnx_path), "-t", "image-feature-extraction"])
        assert result.exit_code == 0, f"export failed (exit {result.exit_code}):\n{result.output}"
        assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"

        model = onnx.load(str(onnx_path))
        # Optimum-driven OnnxConfig for dinov2/feature-extraction produces
        # last_hidden_state. If the patcher had fallen back to nullcontext,
        # the trace-inferred output names (last_hidden_state, pooler_output)
        # would have been used instead.
        assert _output_names(model) == ["last_hidden_state"], (
            f"expected outputs ['last_hidden_state'], got {_output_names(model)} "
            "— Optimum patcher likely fell back to nullcontext because the "
            "task wasn't normalised before TasksManager lookup."
        )


# ===========================================================================
# Required-option failures
# ===========================================================================


class TestExportRequiredOptions:
    """``-m`` and ``-o`` are required; omitting either should fail cleanly."""

    def test_missing_model_fails(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        _assert_fails(["-o", str(onnx_path)], onnx_path)

    def test_missing_output_fails(self, tmp_path: Path):
        # No -o supplied — assert nothing was written anywhere under tmp_path.
        result = _invoke(["-m", _MODEL], catch=True)
        assert result.exit_code != 0, (
            f"expected failure for missing -o, got exit=0:\n{result.output}"
        )
        onnx_files = list(tmp_path.glob("*.onnx"))
        assert not onnx_files, f"unexpected ONNX files in {tmp_path}: {onnx_files}"


# ===========================================================================
# Flag variants on the happy path
# ===========================================================================


class TestExportFlagVariants:
    """Each flag below is layered onto the happy-path invocation."""

    def test_verbose(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        _assert_succeeds(_happy_args(onnx_path, "-v"), onnx_path)

    def test_with_report(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        _assert_succeeds(_happy_args(onnx_path, "--with-report"), onnx_path)
        # Report files are named ``{stem}_htp_export_report.md`` and
        # ``{stem}_htp_metadata.json`` in the output directory.
        md_report = tmp_path / "model_htp_export_report.md"
        json_report = tmp_path / "model_htp_metadata.json"
        assert md_report.exists() or json_report.exists(), (
            f"--with-report produced no report file in {tmp_path}; "
            f"dir contents: {[p.name for p in tmp_path.iterdir()]}"
        )

    def test_clean_onnx(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path, "--clean-onnx"), onnx_path)
        _assert_no_node_has(model, "winml.hierarchy.tag")
        _assert_no_node_has(model, "winml.hierarchy.depth")

    def test_no_hierarchy(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path, "--no-hierarchy"), onnx_path)
        _assert_no_node_has(model, "winml.hierarchy.tag")
        _assert_no_node_has(model, "winml.hierarchy.depth")

    def test_dynamo(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path, "--dynamo"), onnx_path)
        # Only rewritten nodes carry this key; "at least one" is the correct check.
        _assert_some_node_has(model, "pkg.onnxscript.rewriter.rule_name")

    def test_torch_module_warning(self, tmp_path: Path):
        # --torch-module is currently a no-op; the command must still succeed
        # but emit a warning identifying the option.
        onnx_path = tmp_path / "model.onnx"
        result = _invoke(_happy_args(onnx_path, "--torch-module", "LayerNorm,Embedding"))
        assert result.exit_code == 0, f"export failed:\n{result.output}"
        assert onnx_path.exists()
        assert "torch-module" in result.output, (
            f"expected warning mentioning '--torch-module' in output, got:\n{result.output}"
        )


# ===========================================================================
# Task overrides (-t)
# ===========================================================================


class TestExportTaskOverride:
    """``-t`` selects which Optimum OnnxConfig is used; outputs differ by task."""

    def test_image_classification(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path, "-t", "image-classification"), onnx_path)
        assert _output_names(model) == ["logits"], (
            f"expected outputs ['logits'], got {_output_names(model)}"
        )

    def test_feature_extraction(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path, "-t", "feature-extraction"), onnx_path)
        assert _output_names(model) == ["last_hidden_state"], (
            f"expected outputs ['last_hidden_state'], got {_output_names(model)}"
        )

    def test_translation_fails(self, tmp_path: Path):
        # ResNet-50 is a vision model and does not support translation.
        onnx_path = tmp_path / "model.onnx"
        _assert_fails(_happy_args(onnx_path, "-t", "translation"), onnx_path)


# ===========================================================================
# Config files: --shape-config / --export-config / -c
# ===========================================================================


class TestExportConfigFiles:
    """Validate the three JSON-config inputs and their interactions."""

    def test_shape_config(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        width, height = 448, 896
        shape_cfg = _write_json(tmp_path / "shape.json", {"width": width, "height": height})
        model = _assert_succeeds(
            _happy_args(onnx_path, "--shape-config", str(shape_cfg)), onnx_path
        )
        # ONNX vision models use NCHW layout: [batch, channels, height, width]
        assert _input_shape_dims(model) == [1, 3, height, width], (
            f"expected input shape [1, 3, {height}, {width}], got {_input_shape_dims(model)}"
        )

    def test_input_specs(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        shape = [1, 3, 448, 896]
        specs = _write_json(
            tmp_path / "specs.json",
            {"pixel_values": {"dtype": "float32", "shape": shape}},
        )
        model = _assert_succeeds(_happy_args(onnx_path, "--input-specs", str(specs)), onnx_path)
        assert _input_shape_dims(model) == shape, (
            f"expected input shape {shape}, got {_input_shape_dims(model)}"
        )

    def test_input_specs_with_feature_extraction(self, tmp_path: Path):
        # Regression: --input-specs must not bypass Optimum-driven output_tensors
        # resolution. For ResNet feature-extraction the dataclass exposes
        # last_hidden_state + pooler_output but the ONNX graph has only the
        # former — passing both names to torch.onnx.export raises
        # "number of output names provided (2) exceeded number of outputs (1)".
        onnx_path = tmp_path / "model.onnx"
        shape = [1, 3, 448, 896]
        specs = _write_json(
            tmp_path / "specs.json",
            {"pixel_values": {"dtype": "float32", "shape": shape}},
        )
        model = _assert_succeeds(
            _happy_args(onnx_path, "--input-specs", str(specs), "-t", "feature-extraction"),
            onnx_path,
        )
        assert _input_shape_dims(model) == shape, (
            f"expected input shape {shape}, got {_input_shape_dims(model)}"
        )
        assert _output_names(model) == ["last_hidden_state"], (
            f"expected outputs ['last_hidden_state'], got {_output_names(model)}"
        )

    def test_export_config(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        export_cfg = _write_json(tmp_path / "export.json", {"opset_version": 18})
        model = _assert_succeeds(
            _happy_args(onnx_path, "--export-config", str(export_cfg)), onnx_path
        )
        assert _opset_version(model) == 18, (
            f"expected opset 18 from --export-config, got {_opset_version(model)}"
        )

    def test_build_config(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        cfg = _write_json(tmp_path / "build.json", _BUILD_CONFIG)
        model = _assert_succeeds(_happy_args(onnx_path, "-c", str(cfg)), onnx_path)
        assert _opset_version(model) == 18, (
            f"expected opset 18 from -c build config, got {_opset_version(model)}"
        )

    def test_build_config_with_dynamo(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        cfg = _write_json(tmp_path / "build.json", _BUILD_CONFIG)
        model = _assert_succeeds(_happy_args(onnx_path, "-c", str(cfg), "--dynamo"), onnx_path)
        assert _opset_version(model) == 18, (
            f"expected opset 18 with -c + --dynamo, got {_opset_version(model)}"
        )
        _assert_some_node_has(model, "pkg.onnxscript.rewriter.rule_name")


# ===========================================================================
# Dynamic axes: --dynamic-axes
# ===========================================================================


class TestExportDynamicAxes:
    """``--dynamic-axes`` marks the named tensor axes symbolic in the ONNX graph.

    ResNet-50 has a single input ``pixel_values`` of shape [1, 3, 224, 224] and
    a ``logits`` output. Making axis 0 dynamic turns the static batch dim into a
    symbolic ``dim_param`` that also propagates to the output.
    """

    def test_batch_axis_symbolic(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        axes = _write_json(tmp_path / "axes.json", {"pixel_values": {"0": "batch"}})
        model = _assert_succeeds(_happy_args(onnx_path, "--dynamic-axes", str(axes)), onnx_path)
        # Axis 0 of pixel_values becomes symbolic; channel/spatial dims stay static.
        assert _symbolic_dims(model.graph.input, "pixel_values") == {0: "batch"}
        assert _static_dims(model.graph.input, "pixel_values") == {1: 3, 2: 224, 3: 224}
        # The symbolic batch dim propagates to the logits output.
        assert _symbolic_dims(model.graph.output, "logits") == {0: "batch"}

    def test_static_batch_without_flag(self, tmp_path: Path):
        # Baseline contrast: absent the flag, every input dim is a fixed integer.
        onnx_path = tmp_path / "model.onnx"
        model = _assert_succeeds(_happy_args(onnx_path), onnx_path)
        assert _symbolic_dims(model.graph.input, "pixel_values") == {}
        assert _input_shape_dims(model) == [1, 3, 224, 224]

    def test_multiple_axes_symbolic(self, tmp_path: Path):
        onnx_path = tmp_path / "model.onnx"
        axes = _write_json(
            tmp_path / "axes.json",
            {"pixel_values": {"0": "batch", "2": "height", "3": "width"}},
        )
        model = _assert_succeeds(_happy_args(onnx_path, "--dynamic-axes", str(axes)), onnx_path)
        assert _symbolic_dims(model.graph.input, "pixel_values") == {
            0: "batch",
            2: "height",
            3: "width",
        }
        # Only the channel dim remains static.
        assert _static_dims(model.graph.input, "pixel_values") == {1: 3}

    def test_invalid_dynamic_axes_fails(self, tmp_path: Path):
        # An empty symbolic dim name is rejected by WinMLExportConfig validation,
        # so the command must fail cleanly without writing an ONNX file.
        onnx_path = tmp_path / "model.onnx"
        bad = _write_json(tmp_path / "axes.json", {"pixel_values": {"0": ""}})
        _assert_fails(_happy_args(onnx_path, "--dynamic-axes", str(bad)), onnx_path)
