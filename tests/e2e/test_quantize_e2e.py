# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E functional tests for the `winml quantize` CLI command.

Tests invoke ``winml quantize`` via ``CliRunner`` against a tiny hand-built
FP32 ONNX, asserting exit code + structural properties of the output graph.

Shared structural assertions live in ``_assert_quantized_output``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from winml.modelkit.commands.quantize import quantize as quantize_cmd


if TYPE_CHECKING:
    from click.testing import CliRunner


pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Tiny FP32 ONNX fixture
# ---------------------------------------------------------------------------


def _build_tiny_onnx(path: Path, *, with_metadata: bool = True) -> None:
    rng = np.random.default_rng(42)
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 16])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 4])
    w1 = onnx.numpy_helper.from_array(rng.standard_normal((16, 8), dtype=np.float32), "W1")
    b1 = onnx.numpy_helper.from_array(rng.standard_normal((8,), dtype=np.float32), "B1")
    w2 = onnx.numpy_helper.from_array(rng.standard_normal((8, 4), dtype=np.float32), "W2")
    b2 = onnx.numpy_helper.from_array(rng.standard_normal((4,), dtype=np.float32), "B2")
    nodes = [
        onnx.helper.make_node("MatMul", ["input", "W1"], ["mm1"], name="MatMul_1"),
        onnx.helper.make_node("Add", ["mm1", "B1"], ["add1"], name="Add_1"),
        onnx.helper.make_node("Relu", ["add1"], ["relu1"], name="Relu_1"),
        onnx.helper.make_node("MatMul", ["relu1", "W2"], ["mm2"], name="MatMul_2"),
        onnx.helper.make_node("Add", ["mm2", "B2"], ["output"], name="Add_2"),
    ]
    graph = onnx.helper.make_graph(nodes, "tiny_quantizable", [x], [y], [w1, b1, w2, b2])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    if with_metadata:
        meta = model.metadata_props.add()
        meta.key = "test_marker"
        meta.value = "preserved"
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("tiny_quant")
    p = d / "tiny.onnx"
    _build_tiny_onnx(p)
    return p


@pytest.fixture(scope="session")
def tiny_onnx_external(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """At least one weight > 1024 B so ORT's external-data threshold trips."""
    d = tmp_path_factory.mktemp("tiny_quant_ext")
    p = d / "tiny_ext.onnx"
    rng = np.random.default_rng(43)
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])
    w1 = onnx.numpy_helper.from_array(rng.standard_normal((64, 32), dtype=np.float32), "W1")
    b1 = onnx.numpy_helper.from_array(rng.standard_normal((32,), dtype=np.float32), "B1")
    w2 = onnx.numpy_helper.from_array(rng.standard_normal((32, 8), dtype=np.float32), "W2")
    b2 = onnx.numpy_helper.from_array(rng.standard_normal((8,), dtype=np.float32), "B2")
    nodes = [
        onnx.helper.make_node("MatMul", ["input", "W1"], ["mm1"]),
        onnx.helper.make_node("Add", ["mm1", "B1"], ["add1"]),
        onnx.helper.make_node("Relu", ["add1"], ["relu1"]),
        onnx.helper.make_node("MatMul", ["relu1", "W2"], ["mm2"]),
        onnx.helper.make_node("Add", ["mm2", "B2"], ["output"]),
    ]
    graph = onnx.helper.make_graph(nodes, "tiny_ext_quantizable", [x], [y], [w1, b1, w2, b2])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(p))
    onnx.save(
        onnx.load(str(p)),
        str(p),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{p.name}.data",
    )
    return p


# ---------------------------------------------------------------------------
# Real HF-exported ONNX fixtures for per-task calibration dataset coverage
#
# Each fixture lazily exports the model via `winml export` to a persistent
# cache under <project>/temp/test_fixtures/quantize/ so that subsequent test
# runs reuse the file (cold first run: ~30-90s per model; warm: ~0s).
# Cache lives in the project tree (per CLAUDE.md convention) so CI cleanup
# and `.gitignore` can find it.
# Marked with `network` because the first run downloads from HuggingFace.
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_CACHE_ROOT = _PROJECT_ROOT / "temp" / "test_fixtures" / "quantize"


def _export_hf_to_onnx(hf_id: str, task: str, slug: str) -> Path:
    cache = _FIXTURE_CACHE_ROOT / slug
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / "model.onnx"
    if out.exists():
        return out
    from click.testing import CliRunner

    from winml.modelkit.commands.export import export

    args = ["-m", hf_id, "-o", str(out), "--task", task]
    r = CliRunner().invoke(export, args, obj={}, catch_exceptions=False)
    if r.exit_code != 0 or not out.exists():
        raise RuntimeError(
            f"winml export failed for {hf_id}: exit={r.exit_code}\n{r.output}"
        )
    return out


@pytest.fixture(scope="session")
def onnx_imgcls() -> Path:
    return _export_hf_to_onnx("microsoft/resnet-50", "image-classification", "resnet50")


@pytest.fixture(scope="session")
def onnx_txtcls() -> Path:
    return _export_hf_to_onnx(
        "Intel/bert-base-uncased-mrpc", "text-classification", "bert_mrpc"
    )


@pytest.fixture(scope="session")
def onnx_objdet() -> Path:
    return _export_hf_to_onnx("hustvl/yolos-small", "object-detection", "yolos_small")


@pytest.fixture(scope="session")
def onnx_imgseg() -> Path:
    return _export_hf_to_onnx(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "image-segmentation",
        "segformer_b0",
    )


@pytest.fixture(scope="session")
def onnx_dinov2() -> Path:
    return _export_hf_to_onnx(
        "facebook/dinov2-small", "image-feature-extraction", "dinov2_small",
    )


# ---------------------------------------------------------------------------
# Standard assertions
# ---------------------------------------------------------------------------


_QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


def _dtype_of_init(model: onnx.ModelProto, name: str) -> int:
    for init in model.graph.initializer:
        if init.name == name:
            return int(init.data_type)
    raise AssertionError(f"initializer not found: {name}")


def _zero_point_dtype(model: onnx.ModelProto, node_op: str) -> int:
    for node in model.graph.node:
        if node.op_type == node_op and len(node.input) >= 3:
            return _dtype_of_init(model, node.input[2])
    raise AssertionError(f"no {node_op} node with zero_point found")


def _weight_dq_zero_point_dtype(model: onnx.ModelProto) -> int:
    """Return dtype of zero_point on the first DequantizeLinear consuming a weight init.

    Weights in our fixture are named W1/W2.
    """
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0].startswith("W"):
            return _dtype_of_init(model, node.input[2])
    raise AssertionError("no weight DequantizeLinear found")


def _assert_quantized_output(
    *,
    input_onnx: Path,
    output_onnx: Path,
    stdout: str,
    run_inference: bool = True,
) -> onnx.ModelProto:
    """Shared structural assertions on the quantized output model.

    (Exit-code is asserted at the invoke site.)
    """
    # Output is a loadable ONNX
    model = onnx.load(str(output_onnx))

    # Has at least one QDQ node; stdout count matches
    qdq_count = sum(1 for n in model.graph.node if n.op_type in _QDQ_OPS)
    assert qdq_count >= 1, f"no QDQ nodes: {[n.op_type for n in model.graph.node]}"
    if "QDQ nodes inserted:" in stdout:
        reported = int(stdout.split("QDQ nodes inserted:")[1].split()[0])
        assert reported == qdq_count

    # Passes onnx.checker full validation
    onnx.checker.check_model(model, full_check=True)

    # Preserves input/output tensor names
    input_model = onnx.load(str(input_onnx))
    in_before = [i.name for i in input_model.graph.input]
    out_before = [o.name for o in input_model.graph.output]
    assert [i.name for i in model.graph.input] == in_before
    assert [o.name for o in model.graph.output] == out_before

    # No UNDEFINED QDQ scale/zp dtypes; scales finite & strictly positive
    qdq_scale_names: set[str] = set()
    qdq_init_names: set[str] = set()
    for node in model.graph.node:
        if node.op_type in _QDQ_OPS:
            qdq_init_names.update(node.input[1:3])
            if len(node.input) >= 2:
                qdq_scale_names.add(node.input[1])
    for init in model.graph.initializer:
        if init.name in qdq_init_names:
            assert init.data_type != onnx.TensorProto.UNDEFINED, init.name
        if init.name in qdq_scale_names:
            arr = onnx.numpy_helper.to_array(init)
            assert np.all(np.isfinite(arr)), f"non-finite scale {init.name}"
            assert np.all(arr > 0), f"non-positive scale {init.name}: {arr}"

    # ORT-CPU runs and two distinct inputs produce non-identical outputs
    if run_inference:
        sess = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(7)

        # ORT type names -> numpy generator. Integer inputs (token IDs,
        # attention masks, token_type_ids) require integer data; float inputs
        # (image pixels, embeddings) require float data.
        # Integer range is constrained to [0, 1] so embedding lookups
        # (e.g. BERT token_type_ids with vocab=2) stay in bounds across all
        # model architectures.
        def _gen_input(ort_type: str, shape: list[int]) -> np.ndarray:
            if "int64" in ort_type:
                return rng.integers(0, 2, size=shape, dtype=np.int64)
            if "int32" in ort_type:
                return rng.integers(0, 2, size=shape, dtype=np.int32)
            if "float16" in ort_type:
                return rng.standard_normal(shape).astype(np.float16)
            return rng.standard_normal(shape).astype(np.float32)

        outs_runs: list[list[np.ndarray]] = []
        for _ in range(2):
            feed: dict[str, np.ndarray] = {}
            for inp in sess.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                feed[inp.name] = _gen_input(inp.type, shape)
            outs = sess.run(None, feed)
            for arr in outs:
                if np.issubdtype(arr.dtype, np.floating):
                    assert np.isfinite(arr).all()
            outs_runs.append(outs)
        differ = any(
            not np.array_equal(a, b)
            for a, b in zip(outs_runs[0], outs_runs[1], strict=True)
        )
        assert differ, "outputs identical across two distinct inputs (degenerate)"

    return model


def _invoke(runner: CliRunner, args: list[str], *, expect_success: bool = True):
    result = runner.invoke(quantize_cmd, args, obj={}, catch_exceptions=True)
    if expect_success and result.exit_code != 0:
        raise AssertionError(
            f"winml quantize exited {result.exit_code}\nargs: {args}\n{result.output}"
        )
    return result


# ===========================================================================
# Precision routing
# ===========================================================================


class TestPrecision:
    def test_default_precision_is_uint8(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path):
        out = tmp_path / "a1.onnx"
        r = _invoke(runner, ["-m", str(tiny_onnx), "-o", str(out), "--samples", "4"])
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert _zero_point_dtype(model, "QuantizeLinear") == onnx.TensorProto.UINT8
        meta = {p.key: p.value for p in model.metadata_props}
        assert meta.get("test_marker") == "preserved"

    def test_precision_int8_preset(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path):
        out = tmp_path / "a2.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--precision", "int8", "--samples", "4"],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert _zero_point_dtype(model, "QuantizeLinear") == onnx.TensorProto.UINT8

    def test_precision_int16_preset(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path):
        out = tmp_path / "a3.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--precision", "int16", "--samples", "4"],
        )
        model = _assert_quantized_output(
            input_onnx=tiny_onnx, output_onnx=out, stdout=r.output, run_inference=False
        )
        assert _weight_dq_zero_point_dtype(model) == onnx.TensorProto.INT16

    def test_precision_w8a16_preset(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path):
        out = tmp_path / "a4.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--precision", "w8a16", "--samples", "4"],
        )
        model = _assert_quantized_output(
            input_onnx=tiny_onnx, output_onnx=out, stdout=r.output, run_inference=False
        )
        weight_inits = {"W1", "W2", "B1", "B2"}
        for node in model.graph.node:
            if node.op_type == "QuantizeLinear" and node.input[0] not in weight_inits:
                assert _dtype_of_init(model, node.input[2]) == onnx.TensorProto.UINT16
                break
        else:
            raise AssertionError("no activation QuantizeLinear")

    def test_explicit_weight_activation_type_override_precision(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out = tmp_path / "a5.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--precision", "int8",
                "--weight-type", "int8",
                "--activation-type", "uint8",
                "--samples", "4",
            ],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert _weight_dq_zero_point_dtype(model) == onnx.TensorProto.INT8

    def test_non_quant_precision_rejected(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        """Float precisions like fp16 must be rejected at CLI parse time.

        Replaces the legacy ``test_unknown_precision_falls_back_to_uint8`` which
        documented the silent-fallback bug that PR #680 fixed.
        """
        out = tmp_path / "a6.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--precision", "fp16", "--samples", "4"],
            expect_success=False,
        )
        assert r.exit_code != 0
        assert "not a supported quantization precision" in r.output
        assert not out.exists()


# ===========================================================================
# Calibration method
# ===========================================================================


class TestCalibrationMethod:
    @pytest.mark.parametrize("method", ["minmax", "entropy", "percentile"])
    def test_method(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path, method: str):
        out = tmp_path / f"b_{method}.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--method", method, "--samples", "4"],
        )
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)


# ===========================================================================
# Quant options
# ===========================================================================


class TestQuantOptions:
    def test_per_channel_produces_vector_scale(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out = tmp_path / "c1.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--per-channel", "--samples", "4"],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        scale_initnames = {
            node.input[1]
            for node in model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0].startswith("W")
        }
        assert scale_initnames, "no weight DequantizeLinear found"
        has_vector = False
        for init in model.graph.initializer:
            if init.name in scale_initnames:
                total = 1
                for d in init.dims:
                    total *= d
                if total > 1:
                    has_vector = True
                    break
        assert has_vector

    def test_symmetric_int8_zero_point_is_zero(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out = tmp_path / "c2.onnx"
        # int8 weights so symmetric -> zp=0 unambiguously (uint8 symmetric centers on 127).
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--symmetric",
                "--weight-type", "int8",
                "--samples", "4",
            ],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        weight_zp_names = {
            node.input[2]
            for node in model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0].startswith("W")
        }
        for init in model.graph.initializer:
            if init.name in weight_zp_names:
                arr = onnx.numpy_helper.to_array(init)
                assert np.all(arr == 0), f"zp {init.name}: {arr}"


# ===========================================================================
# Per-task calibration datasets
# ===========================================================================


class TestPerTaskDatasets:
    def test_task_random_uses_random_dataset(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out = tmp_path / "d1.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--task", "random", "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert "Creating random dataset with RandomDataset" in r.output, r.output

    @pytest.mark.network
    def test_task_image_classification_dataset(
        self, runner: CliRunner, onnx_imgcls: Path, tmp_path: Path
    ):
        out = tmp_path / "d2.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(onnx_imgcls), "-o", str(out),
                "--task", "image-classification",
                "--model-name", "microsoft/resnet-50",
                "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=onnx_imgcls, output_onnx=out, stdout=r.output)
        assert "Creating image-classification dataset with ImageDataset" in r.output, r.output

    @pytest.mark.network
    def test_task_text_classification_dataset(
        self, runner: CliRunner, onnx_txtcls: Path, tmp_path: Path
    ):
        out = tmp_path / "d3.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(onnx_txtcls), "-o", str(out),
                "--task", "text-classification",
                "--model-name", "Intel/bert-base-uncased-mrpc",
                "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=onnx_txtcls, output_onnx=out, stdout=r.output)
        assert "Creating text-classification dataset with TextDataset" in r.output, r.output

    @pytest.mark.network
    def test_task_object_detection_dataset(
        self, runner: CliRunner, onnx_objdet: Path, tmp_path: Path
    ):
        out = tmp_path / "d4.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(onnx_objdet), "-o", str(out),
                "--task", "object-detection",
                "--model-name", "hustvl/yolos-small",
                "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=onnx_objdet, output_onnx=out, stdout=r.output)
        assert (
            "Creating object-detection dataset with ObjectDetectionDataset" in r.output
        ), r.output

    @pytest.mark.network
    def test_task_image_segmentation_dataset(
        self, runner: CliRunner, onnx_imgseg: Path, tmp_path: Path
    ):
        out = tmp_path / "d5.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(onnx_imgseg), "-o", str(out),
                "--task", "image-segmentation",
                "--model-name", "nvidia/segformer-b0-finetuned-ade-512-512",
                "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=onnx_imgseg, output_onnx=out, stdout=r.output)
        assert (
            "Creating image-segmentation dataset with ImageSegmentationDataset" in r.output
        ), r.output

    def test_unsupported_task_falls_back_to_random_dataset(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out = tmp_path / "d6.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--task", "automatic-speech-recognition",
                "--samples", "4",
            ],
        )
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert "falling back to RandomDataset" in r.output, (
            f"fallback warning not emitted in CLI output:\n{r.output}"
        )

    @pytest.mark.network
    def test_image_feature_extraction_uses_image_dataset(
        self, runner: CliRunner, onnx_dinov2: Path, tmp_path: Path
    ):
        # A vision embedding model's canonical task is image-feature-extraction
        # (what `winml inspect` and auto-detection report); it maps directly to
        # ImageDataset for calibration. Under the modality-aware task vocabulary
        # 'feature-extraction' is text-only, so it is intentionally not a valid
        # task for a vision model (its calibration dataset would be TextDataset).
        out = tmp_path / "d7.onnx"

        r = _invoke(
            runner,
            [
                "-m", str(onnx_dinov2), "-o", str(out),
                "--task", "image-feature-extraction",
                "--model-name", "facebook/dinov2-small",
                "--samples", "4", "-v",
            ],
        )
        _assert_quantized_output(input_onnx=onnx_dinov2, output_onnx=out, stdout=r.output)
        assert (
            "Creating image-feature-extraction dataset with ImageDataset" in r.output
        ), r.output


# ===========================================================================
# Output behavior
# ===========================================================================


class TestOutputBehavior:
    def test_default_output_path_alongside_input(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        target_dir = tmp_path / "e1"
        target_dir.mkdir()
        local = target_dir / "tiny.onnx"
        local.write_bytes(tiny_onnx.read_bytes())
        r = _invoke(runner, ["-m", str(local), "--samples", "4"])
        expected = target_dir / "tiny_qdq.onnx"
        assert expected.exists()
        _assert_quantized_output(input_onnx=local, output_onnx=expected, stdout=r.output)

    def test_explicit_output_path(self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out = out_dir / "custom.onnx"
        r = _invoke(runner, ["-m", str(tiny_onnx), "-o", str(out), "--samples", "4"])
        assert out.exists()
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)

    def test_explicit_output_path_auto_creates_parent_dir(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        # Parent dir does not exist; command must create it.
        out = tmp_path / "missing" / "nested" / "custom.onnx"
        assert not out.parent.exists()
        r = _invoke(runner, ["-m", str(tiny_onnx), "-o", str(out), "--samples", "4"])
        assert out.exists(), f"command did not auto-create parent dir: {out.parent}"
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)

    def test_external_data_sidecar_written(
        self, runner: CliRunner, tiny_onnx_external: Path, tmp_path: Path
    ):
        out_dir = tmp_path / "out_ext"
        out_dir.mkdir()
        out = out_dir / "quant_ext.onnx"
        r = _invoke(
            runner, ["-m", str(tiny_onnx_external), "-o", str(out), "--samples", "4"]
        )
        assert out.exists()
        assert (out_dir / f"{out.name}.data").exists()
        _assert_quantized_output(input_onnx=tiny_onnx_external, output_onnx=out, stdout=r.output)


# ===========================================================================
# Build-config precedence (CLI vs config file)
# ===========================================================================


def _write_build_config(path: Path, quant: dict) -> None:
    cfg = {
        "loader": {"task": None},
        "export": {},
        "optim": {},
        "quant": quant,
        "compile": {},
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


class TestBuildConfigPrecedence:
    def test_cli_samples_overrides_config_and_config_method_used(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        bc = tmp_path / "bc.json"
        _write_build_config(bc, {"samples": 50, "calibration_method": "entropy"})
        out = tmp_path / "f1.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--config", str(bc),
                "--samples", "4",
            ],
        )
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert "Samples: 4" in r.output
        assert "Method: entropy" in r.output

    def test_cli_precision_wins_over_empty_config(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        """Regression: --precision int16 must win even when --config is given.

        Before the fix, dataclass defaults (weight_type='uint8') leaked from
        ``WinMLQuantizationConfig.from_dict`` and silently overrode --precision.
        """
        bc = tmp_path / "bc_f2.json"
        _write_build_config(bc, {})  # empty quant section
        out = tmp_path / "f2.onnx"
        r = _invoke(
            runner,
            [
                "-m", str(tiny_onnx), "-o", str(out),
                "--config", str(bc),
                "--precision", "int16",
                "--samples", "4",
            ],
        )
        # uint16 activations may not run on CPU EP â€” skip S7/S9
        model = _assert_quantized_output(
            input_onnx=tiny_onnx, output_onnx=out, stdout=r.output, run_inference=False
        )
        assert _weight_dq_zero_point_dtype(model) == onnx.TensorProto.INT16


# ===========================================================================
# Errors
# ===========================================================================


class TestErrors:
    def test_missing_model_option(self, runner: CliRunner):
        r = runner.invoke(quantize_cmd, [], obj={}, catch_exceptions=True)
        assert r.exit_code != 0
        assert "Missing option" in r.output and "--model" in r.output

    def test_model_path_does_not_exist(self, runner: CliRunner, tmp_path: Path):
        r = runner.invoke(
            quantize_cmd,
            ["-m", str(tmp_path / "nope.onnx")],
            obj={},
            catch_exceptions=True,
        )
        assert r.exit_code != 0
        assert "does not exist" in r.output

    def test_invalid_method_value(self, runner: CliRunner, tiny_onnx: Path):
        r = runner.invoke(
            quantize_cmd,
            ["-m", str(tiny_onnx), "--method", "gaussian"],
            obj={},
            catch_exceptions=True,
        )
        assert r.exit_code != 0
        assert "Invalid value for" in r.output and "--method" in r.output

    def test_invalid_weight_type_value(self, runner: CliRunner, tiny_onnx: Path):
        r = runner.invoke(
            quantize_cmd,
            ["-m", str(tiny_onnx), "--weight-type", "float8"],
            obj={},
            catch_exceptions=True,
        )
        assert r.exit_code != 0
        assert "Invalid value for" in r.output and "--weight-type" in r.output

    def test_malformed_onnx_input(self, runner: CliRunner, tmp_path: Path):
        bad = tmp_path / "bad.onnx"
        bad.write_bytes(b"\x00\x01\x02 not a real onnx " * 10)
        r = runner.invoke(
            quantize_cmd,
            ["-m", str(bad), "--samples", "4"],
            obj={},
            catch_exceptions=True,
        )
        assert r.exit_code != 0
        assert "Quantization failed" in r.output
        # Must surface a parse-related cause, not just the generic prefix.
        lowered = r.output.lower()
        assert any(
            kw in lowered
            for kw in ("parse", "protobuf", "decode", "load", "invalid")
        ), f"expected parse-related cause in output, got:\n{r.output}"


# ===========================================================================
# Build-config key absorption sweep
# ===========================================================================


class TestConfigPrecedenceSweep:
    """For each build-config quant.* key, assert it is consumed when CLI omits it.

    Verifies via structural inspection of the produced model, not stdout.
    """

    def test_weight_type_from_config(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        bc = tmp_path / "bc.json"
        _write_build_config(bc, {"weight_type": "int8"})
        out = tmp_path / "f3a.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--config", str(bc), "--samples", "4"],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert _weight_dq_zero_point_dtype(model) == onnx.TensorProto.INT8

    def test_per_channel_from_config(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        bc = tmp_path / "bc.json"
        _write_build_config(bc, {"per_channel": True})
        out = tmp_path / "f3b.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--config", str(bc), "--samples", "4"],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        scale_initnames = {
            node.input[1]
            for node in model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0].startswith("W")
        }
        has_vector = False
        for init in model.graph.initializer:
            if init.name in scale_initnames:
                total = 1
                for d in init.dims:
                    total *= d
                if total > 1:
                    has_vector = True
                    break
        assert has_vector, "per_channel from config not applied (scales are scalar)"

    def test_symmetric_from_config(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        bc = tmp_path / "bc.json"
        # symmetric only unambiguously yields zp==0 with int8 weights
        _write_build_config(bc, {"symmetric": True, "weight_type": "int8"})
        out = tmp_path / "f3c.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--config", str(bc), "--samples", "4"],
        )
        model = _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        weight_zp_names = {
            node.input[2]
            for node in model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0].startswith("W")
        }
        for init in model.graph.initializer:
            if init.name in weight_zp_names:
                arr = onnx.numpy_helper.to_array(init)
                assert np.all(arr == 0), f"symmetric from config not applied; zp={arr}"

    def test_task_from_config(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        """task='automatic-speech-recognition' from config must trigger fallback warning."""
        bc = tmp_path / "bc.json"
        _write_build_config(bc, {"task": "automatic-speech-recognition"})
        out = tmp_path / "f3d.onnx"
        r = _invoke(
            runner,
            ["-m", str(tiny_onnx), "-o", str(out), "--config", str(bc), "--samples", "4"],
        )
        _assert_quantized_output(input_onnx=tiny_onnx, output_onnx=out, stdout=r.output)
        assert "falling back to RandomDataset" in r.output, (
            f"task from config did not flow through to dataset selection:\n{r.output}"
        )


# ===========================================================================
# Verbose flag
# ===========================================================================


class TestVerbose:
    def test_verbose_emits_more_output(
        self, runner: CliRunner, tiny_onnx: Path, tmp_path: Path
    ):
        out_q = tmp_path / "quiet.onnx"
        out_v = tmp_path / "verbose.onnx"
        r_quiet = _invoke(runner, ["-m", str(tiny_onnx), "-o", str(out_q), "--samples", "4"])
        r_verbose = _invoke(
            runner, ["-m", str(tiny_onnx), "-o", str(out_v), "--samples", "4", "-v"]
        )
        assert len(r_verbose.output) > len(r_quiet.output), (
            f"verbose did not increase output\n--- quiet ---\n{r_quiet.output}\n"
            f"--- verbose ---\n{r_verbose.output}"
        )

