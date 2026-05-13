# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E functional tests for the `winml eval` CLI command.

Plan: ``temp/eval_e2e_commands.md``.

Each test invokes ``winml eval`` end-to-end against a real (small) model +
~4 dataset samples and asserts exit code + expected output keys + finite
metric values. These tests do NOT assert metric magnitudes — that's the
accuracy regression suite under ``scripts/e2e_eval/run_eval.py``.

Markers:
    e2e: Auto-skipped unless ``pytest -m e2e`` is passed.

Group layout (see plan for rationale):
    A. Per-task success (13 cases)
    B. Alternative model-input forms (2 cases)
    C. Output behavior (1 case)
    D. ``--schema`` parametrized (13 sub-tests)
    E. ``--device`` / ``--ep`` (2 cases)
    F. Additional options & branches (8 cases)
    G. CLI-validation error paths (5 cases, fast)
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from winml.modelkit.commands.eval import eval as eval_cmd

from .conftest import find_cache_dir


if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


pytestmark = [pytest.mark.e2e]

# 10 samples keeps each e2e run short while giving enough signal for
# range-based assertions. Shuffle uses a fixed seed=42 (see
# ``DatasetConfig.seed`` default), so the sampled subset is deterministic
# across runs on the same dataset.
SAMPLES = "10"
ADE20K_LABEL_MAP = "scripts/e2e_eval/datasets/ade20k_gt_to_model_label.json"


@pytest.fixture
def tiny_textcls_script(tmp_path: Path) -> Path:
    """Write a minimal dataset-build script to disk and return its path.

    Used to exercise the ``--dataset-script`` CLI code path without
    committing a build-script artifact to the repo.

    The script writes a 10-row text-classification dataset with non-default
    columns (``text_a``, ``text_b``) so the test must also pass
    ``--column input_column=text_a --column second_input_column=text_b``.
    """
    script = tmp_path / "build_tiny_textcls.py"
    script.write_text(
        '''import argparse
from datasets import Dataset

ROWS = [
    {"text_a": "The movie was great.", "text_b": "I loved the film.", "label": 1},
    {"text_a": "Terrible weather today.", "text_b": "It is raining heavily.", "label": 0},
    {"text_a": "The cat sat on the mat.", "text_b": "A feline rested on the rug.", "label": 1},
    {"text_a": "Stocks are rising.", "text_b": "I went to the supermarket.", "label": 0},
    {"text_a": "She loves chocolate cake.", "text_b": "Her favorite dessert is cake.", "label": 1},
    {"text_a": "The car needs repair.", "text_b": "It is a sunny afternoon.", "label": 0},
    {"text_a": "He plays guitar in a band.", "text_b": "He is a musician.", "label": 1},
    {"text_a": "I finished the report.", "text_b": "The dog chased the ball.", "label": 0},
    {"text_a": "Coffee tastes bitter.", "text_b": "This drink is not sweet.", "label": 1},
    {"text_a": "She studies mathematics.", "text_b": "The garden is full of flowers.", "label": 0},
]

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
Dataset.from_list(ROWS).save_to_disk(args.output)
''',
        encoding="utf-8",
    )
    return script


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(runner: CliRunner, args: list[str], *, expect_success: bool = True) -> object:
    """Invoke ``winml eval`` with ``obj={}`` (the command uses @click.pass_context).

    ``catch_exceptions=True`` so Click error-handling produces non-zero exit
    instead of raising — required for error-path assertions.
    """
    result = runner.invoke(eval_cmd, args, obj={}, catch_exceptions=True)
    if expect_success and result.exit_code != 0:
        raise AssertionError(
            f"winml eval exited with {result.exit_code}\n"
            f"args: {args}\n--- output ---\n{result.output}",
        )
    return result


def _assert_metrics_present(output_path: Path, required_keys: list[str]) -> dict:
    """Load eval JSON; assert required metric keys are present + finite."""
    assert output_path.exists(), f"output file not created: {output_path}"
    data = json.loads(output_path.read_text())
    assert "metrics" in data, f"missing 'metrics': {data}"
    metrics = data["metrics"]
    for key in required_keys:
        assert key in metrics, f"missing metric {key!r}; got {sorted(metrics)}"
        value = metrics[key]
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f"metric {key} is not finite: {value}"
    return data


def _assert_in_range(
    metrics: dict, key: str, lo: float, hi: float,
) -> None:
    """Assert ``metrics[key]`` is a finite number within ``[lo, hi]``.

    Use this for tasks where the metric is a bounded score (e.g. accuracy 0..1
    or top-k accuracy 0..100). The bounds are deliberately wide — they prove
    the value is sane, not that it matches a baseline.
    """
    assert key in metrics, f"missing metric {key!r}; got {sorted(metrics)}"
    value = metrics[key]
    assert isinstance(value, (int, float)), (
        f"metric {key} not numeric: {value!r} ({type(value).__name__})"
    )
    assert math.isfinite(value), f"metric {key} is not finite: {value}"
    assert lo <= value <= hi, (
        f"metric {key}={value} outside expected range [{lo}, {hi}]"
    )


# ===========================================================================
# A. Per-task success path
# ===========================================================================


class TestEvalPerTask:
    """One end-to-end run per task in ``_EVALUATOR_REGISTRY``.

    No ``--device`` / ``--ep`` — CLI auto-picks hardware.
    """

    def test_image_classification(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.1 — HF evaluate.evaluator("image-classification") returns `accuracy`.
        # --streaming avoids caching full mini-imagenet (~1-2 GB).
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "google/vit-base-patch16-224",
            "--task", "image-classification",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        # ViT-base full ImageNet ≈ 0.81; floor at 0.5 still catches
        # broken-pipeline regressions on 10 samples.
        _assert_in_range(data["metrics"], "accuracy", 0.5, 1.0)

    def test_text_classification(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.2 — model aligned with CLI default dataset (nyu-mll/glue/mrpc).
        # HF evaluate.evaluator("text-classification") returns `accuracy`.
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        # bert-mrpc full MRPC ≈ 0.86; MRPC majority baseline ≈ 0.68.
        _assert_in_range(data["metrics"], "accuracy", 0.6, 1.0)

    def test_token_classification(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.3
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "dslim/bert-base-NER",
            "--task", "token-classification",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(
            out,
            ["overall_precision", "overall_recall", "overall_f1", "overall_accuracy"],
        )
        # bert-base-NER full CoNLL: f1 ≈ 0.91, accuracy ≈ 0.98.
        for k in ("overall_precision", "overall_recall", "overall_f1"):
            _assert_in_range(data["metrics"], k, 0.5, 1.0)
        _assert_in_range(data["metrics"], "overall_accuracy", 0.8, 1.0)

    def test_object_detection(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.4 — COCO val is ~6 GB; --streaming keeps only the bytes needed
        # for the sampled subset.
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "hustvl/yolos-small",
            "--task", "object-detection",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["map", "map_50", "mar_100"])
        # COCO mAP / mAR are bounded by [0, 1]; torchmetrics may report -1
        # when no positives are sampled, which is acceptable for tiny N.
        for k in ("map", "map_50", "mar_100"):
            v = data["metrics"][k]
            assert -1.0 <= v <= 1.0, f"{k}={v} outside [-1, 1]"

    def test_image_segmentation(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.5
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "nvidia/segformer-b1-finetuned-ade-512-512",
            "--task", "image-segmentation",
            "--dataset", "danjacobellis/scene_parse_150",
            "--split", "validation",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["mean_iou"])
        _assert_in_range(data["metrics"], "mean_iou", 0.0, 1.0)

    def test_question_answering(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.6
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "distilbert/distilbert-base-cased-distilled-squad",
            "--task", "question-answering",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["exact_match", "f1"])
        # distilbert-squad full SQuAD v1: EM ≈ 77, F1 ≈ 85 (percentages).
        # Both are harsh on N=10 (heavy per-sample variance with seed=42).
        # Loose floors guard against degenerate output, not magnitude.
        _assert_in_range(data["metrics"], "exact_match", 5.0, 100.0)
        _assert_in_range(data["metrics"], "f1", 5.0, 100.0)

    def test_feature_extraction(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.7
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "sentence-transformers/all-MiniLM-L6-v2",
            "--task", "feature-extraction",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        # Spearman correlation reported as percentage in [-100, 100].
        # MiniLM-L6-v2 full STSB ≈ 80; 10-sample noise can be large.
        data = _assert_metrics_present(out, ["cosine_spearman"])
        _assert_in_range(data["metrics"], "cosine_spearman", 40.0, 100.0)

    def test_sentence_similarity(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.8 (alias for feature-extraction)
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "sentence-transformers/all-MiniLM-L6-v2",
            "--task", "sentence-similarity",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["cosine_spearman"])
        _assert_in_range(data["metrics"], "cosine_spearman", 40.0, 100.0)

    def test_image_feature_extraction(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # A.9 — kNN accuracies reported as percentages 0..100.
        # --streaming avoids caching mini-imagenet.
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "facebook/dinov2-small",
            "--task", "image-feature-extraction",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(
            out, ["knn_top1_accuracy", "knn_top5_accuracy"],
        )
        # DINOv2 features cluster strongly on mini-imagenet.
        _assert_in_range(data["metrics"], "knn_top1_accuracy", 30.0, 100.0)
        _assert_in_range(data["metrics"], "knn_top5_accuracy", 60.0, 100.0)
        top1 = data["metrics"]["knn_top1_accuracy"]
        top5 = data["metrics"]["knn_top5_accuracy"]
        assert top1 <= top5, f"top1 ({top1}) must be <= top5 ({top5})"

    def test_image_to_text_fp16(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.10 — only test that exercises non-auto --precision
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Salesforce/blip-image-captioning-base",
            "--task", "image-to-text",
            "--dataset", "lmms-lab/flickr30k",
            "--split", "test",
            "--streaming",
            "--samples", SAMPLES,
            "--precision", "fp16",
            "-o", str(out),
        ])
        # CER and CIDEr depend on optional libs (jiwer, pycocoevalcap), and
        # the BLIP captioning model currently falls back to a generic
        # Seq2Seq path in WinML (text2text-generation), which may produce
        # zero valid samples on tiny N. The CLI contract here is "exit 0
        # and produce the metric keys"; magnitude is exercised in the
        # accuracy regression suite, not in this functional test.
        data = _assert_metrics_present(out, ["cer", "cider", "n_samples"])
        m = data["metrics"]
        for k, hi in (("cer", 10.0), ("cider", 20.0)):
            v = m[k]
            assert v is None or (
                isinstance(v, (int, float))
                and math.isfinite(v)
                and 0.0 <= v <= hi
            ), f"metric {k}={v!r} not None or in [0,{hi}]"
        assert isinstance(m["n_samples"], int) and m["n_samples"] >= 0

    def test_fill_mask(self, runner: CliRunner, tmp_path: Path) -> None:
        # A.11 — pseudo-perplexity >= 1 (perplexity is exp of non-neg NLL).
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "distilbert/distilbert-base-uncased",
            "--task", "fill-mask",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["pseudo_perplexity", "nll"])
        # Pseudo-perplexity over a 10-sample wikitext stream can vary
        # widely (we observed ~3000 with seed=42). Cap is set well above
        # observed to catch genuine numerical blowup, not normal noise.
        _assert_in_range(data["metrics"], "pseudo_perplexity", 1.0, 1e5)
        _assert_in_range(data["metrics"], "nll", 0.0, 15.0)

    def test_zero_shot_classification(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # A.12 — zero-shot uses ClassificationMetric → accuracy + f1.
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "cross-encoder/nli-deberta-v3-small",
            "--task", "zero-shot-classification",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy", "f1"])
        # nli-deberta-v3-small zero-shot on AG News, N=10. 4-class random
        # baseline = 0.25; tiny-N variance can push real models below
        # baseline. Use a very loose floor here.
        _assert_in_range(data["metrics"], "accuracy", 0.1, 1.0)
        _assert_in_range(data["metrics"], "f1", 0.1, 1.0)

    def test_zero_shot_image_classification(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # A.13
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "openai/clip-vit-base-patch32",
            "--task", "zero-shot-image-classification",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["top1_accuracy", "top5_accuracy"])
        # CLIP-ViT-B/32 zero-shot on CIFAR-100: top1 ≈ 0.63, top5 ≈ 0.88
        # (full set). Floors leave headroom for tiny-N variance.
        _assert_in_range(data["metrics"], "top1_accuracy", 30.0, 100.0)
        _assert_in_range(data["metrics"], "top5_accuracy", 60.0, 100.0)


# ===========================================================================
# B. Alternative model-input forms
# ===========================================================================


class TestEvalModelInputForms:
    """Coverage for the two non-default ``-m`` forms."""

    def test_onnx_file_mode_monolithic(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # B.1
        hf_id = "google/vit-base-patch16-224"
        task = "image-classification"

        # Warm cache via HF id (use streaming to avoid mini-imagenet cache).
        _invoke(runner, [
            "-m", hf_id, "--task", task, "--streaming", "--samples", SAMPLES,
        ])

        cache_dir = find_cache_dir(hf_id, task=task)
        assert cache_dir is not None, "expected cache after warm run"
        onnx_files = list(cache_dir.glob("*_model.onnx"))
        assert onnx_files, f"no *_model.onnx in {cache_dir}"

        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", str(onnx_files[0]),
            "--model-id", hf_id,
            "--task", task,
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.5, 1.0)

    def test_onnx_file_mode_split_encoder(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # B.2
        hf_id = "openai/clip-vit-base-patch32"
        task = "zero-shot-image-classification"

        _invoke(runner, ["-m", hf_id, "--task", task, "--samples", SAMPLES])

        cache_dir = find_cache_dir(hf_id, task=task)
        assert cache_dir is not None, "expected cache after warm run"

        image_candidates = list(cache_dir.glob("*image*encoder*.onnx")) + list(
            cache_dir.glob("*vision*.onnx"),
        )
        text_candidates = [
            p for p in cache_dir.glob("*text*.onnx")
            if "encoder" in p.name.lower() or "text" in p.name.lower()
        ]
        if not image_candidates or not text_candidates:
            pytest.skip(
                f"split-encoder ONNX files not found in {cache_dir}; "
                "build may produce a monolithic ONNX for this model.",
            )

        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", f"image-encoder={image_candidates[0]}",
            "-m", f"text-encoder={text_candidates[0]}",
            "--model-id", hf_id,
            "--task", task,
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["top1_accuracy"])
        _assert_in_range(data["metrics"], "top1_accuracy", 30.0, 100.0)


# ===========================================================================
# C. Output behavior
# ===========================================================================


class TestEvalOutput:
    """``-o`` path creation + JSON validity."""

    def test_creates_nested_output_dir(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # C.1
        out = tmp_path / "nested" / "subdir" / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        assert out.exists(), "nested output dir not auto-created"
        data = json.loads(out.read_text())
        assert "metrics" in data
        assert "model_id" in data
        assert "dataset" in data


# ===========================================================================
# D. --schema mode (parametrized x 13)
# ===========================================================================


_ALL_TASKS = [
    "image-classification",
    "text-classification",
    "token-classification",
    "object-detection",
    "image-segmentation",
    "question-answering",
    "feature-extraction",
    "sentence-similarity",
    "image-feature-extraction",
    "image-to-text",
    "fill-mask",
    "zero-shot-classification",
    "zero-shot-image-classification",
]


class TestEvalSchema:
    @pytest.mark.parametrize("task", _ALL_TASKS)
    def test_schema_for_each_task(self, runner: CliRunner, task: str) -> None:
        # D
        result = _invoke(runner, ["--schema", "--task", task])
        assert "Dataset schema" in result.output


# ===========================================================================
# E. --device / --ep coverage
# ===========================================================================


class TestEvalDeviceAndEp:
    def test_device_cpu(self, runner: CliRunner, tmp_path: Path) -> None:
        # E.1 — CPU works on every box. ResNet-50 is a small, fast image
        # classifier well-suited to a CPU smoke test (no per-token forward
        # passes like fill-mask).
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "microsoft/resnet-50",
            "--task", "image-classification",
            "--device", "cpu",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        # ResNet-50 full ImageNet ≈ 0.76; mini-imagenet is shifted, floor 0.4.
        _assert_in_range(data["metrics"], "accuracy", 0.4, 1.0)

    def test_device_npu_and_ep_qnn(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # E.2 — combined --device + --ep
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "google/vit-base-patch16-224",
            "--task", "image-classification",
            "--device", "npu",
            "--ep", "qnn",
            "--streaming",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.5, 1.0)


# ===========================================================================
# F. Additional options & branches
# ===========================================================================


class TestEvalAdditionalOptions:
    def test_dataset_name_explicit(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # F.1
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--dataset", "nyu-mll/glue",
            "--dataset-name", "mrpc",
            "--column", "input_column=sentence1",
            "--column", "second_input_column=sentence2",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.6, 1.0)

    def test_label_mapping_image_segmentation(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # F.2
        from pathlib import Path as _Path

        label_map = _Path(ADE20K_LABEL_MAP)
        if not label_map.exists():
            pytest.skip(f"label-mapping file not in repo: {label_map}")

        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "nvidia/segformer-b1-finetuned-ade-512-512",
            "--task", "image-segmentation",
            "--dataset", "danjacobellis/scene_parse_150",
            "--split", "validation",
            "--streaming",
            "--label-mapping", str(label_map),
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["mean_iou"])
        _assert_in_range(data["metrics"], "mean_iou", 0.0, 1.0)

    def test_config_file_basic(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # F.3 — `eval` section provides task + samples
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({
            "loader": {"task": "text-classification"},
            "eval": {"dataset": {"samples": 5}},
        }))
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--config", str(cfg),
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.6, 1.0)
        assert data["dataset"]["samples"] == 5, (
            f"expected samples=5 from config, got {data['dataset']['samples']}"
        )

    def test_config_file_cli_override(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # F.4 — CLI wins over config file
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({
            "loader": {"task": "text-classification"},
            "eval": {"dataset": {"samples": 5}},
        }))
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--config", str(cfg),
            "--samples", "7",
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.6, 1.0)
        assert data["dataset"]["samples"] == 7, (
            f"expected CLI override samples=7, got {data['dataset']['samples']}"
        )

    def test_auto_task_detection(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # F.5 — no --task flag; CLI infers from HF model
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--samples", SAMPLES,
            "-o", str(out),
        ])
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.6, 1.0)
        assert data.get("task") == "text-classification", (
            f"expected auto-detected task, got {data.get('task')!r}"
        )

    def test_precision_warning_for_prebuilt_onnx(
        self, runner: CliRunner, tmp_path: Path, caplog,
    ) -> None:
        # F.6 — pre-built ONNX + --precision emits warning, still succeeds
        import logging as _logging

        hf_id = "google/vit-base-patch16-224"
        task = "image-classification"

        _invoke(runner, ["-m", hf_id, "--task", task, "--streaming", "--samples", SAMPLES])

        cache_dir = find_cache_dir(hf_id, task=task)
        assert cache_dir is not None
        onnx_files = list(cache_dir.glob("*_model.onnx"))
        assert onnx_files

        out = tmp_path / "result.json"
        with caplog.at_level(_logging.WARNING, logger="winml.modelkit.commands.eval"):
            _invoke(runner, [
                "-m", str(onnx_files[0]),
                "--model-id", hf_id,
                "--task", task,
                "--precision", "fp16",
                "--streaming",
                "--samples", SAMPLES,
                "-o", str(out),
            ])
        # Warning is emitted via ``logger.warning(...)``; capture from log records.
        msgs = [r.getMessage().lower() for r in caplog.records]
        assert any(
            "precision" in m and ("ignor" in m or "pre-built" in m)
            for m in msgs
        ), f"expected precision-ignored warning, got:\n{msgs!r}"
        _assert_metrics_present(out, ["accuracy"])

    def test_dataset_script_with_column_remap(
        self, runner: CliRunner, tmp_path: Path, tiny_textcls_script: Path,
    ) -> None:
        # F.7 — --dataset-script + --column + --trust-remote-code (good path)
        ds_path = tmp_path / "tiny_textcls"
        out = tmp_path / "result.json"
        _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--dataset-script", str(tiny_textcls_script),
            "--dataset", str(ds_path),
            "--trust-remote-code",
            "--column", "input_column=text_a",
            "--column", "second_input_column=text_b",
            "--samples", "10",
            "-o", str(out),
        ])
        assert ds_path.exists(), "dataset script did not write to --dataset path"
        data = _assert_metrics_present(out, ["accuracy"])
        _assert_in_range(data["metrics"], "accuracy", 0.0, 1.0)

    def test_dataset_script_without_trust_remote_code(
        self, runner: CliRunner, tmp_path: Path, tiny_textcls_script: Path,
    ) -> None:
        # F.8 — error path
        ds_path = tmp_path / "tiny_textcls"
        result = _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--dataset-script", str(tiny_textcls_script),
            "--dataset", str(ds_path),
            "--samples", "10",
        ], expect_success=False)
        assert result.exit_code != 0
        assert "trust-remote-code" in result.output.lower(), result.output


# ===========================================================================
# G. CLI-validation error paths (fast — no model load)
# ===========================================================================


class TestEvalErrorPaths:
    def test_bad_column_format(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # G.1
        result = _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--column", "foo",  # missing '='
            "--samples", "1",
        ], expect_success=False)
        assert result.exit_code != 0
        assert "key=value" in result.output.lower() or "invalid" in result.output.lower(), (
            result.output
        )

    def test_missing_label_mapping_file(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # G.2
        missing = tmp_path / "does-not-exist.json"
        result = _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--label-mapping", str(missing),
            "--samples", "1",
        ], expect_success=False)
        assert result.exit_code != 0
        out_lower = result.output.lower()
        assert ("does not exist" in out_lower
                or "not found" in out_lower
                or "no such file" in out_lower), result.output

    def test_bogus_dataset_name(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # G.3
        result = _invoke(runner, [
            "-m", "Intel/bert-base-uncased-mrpc",
            "--task", "text-classification",
            "--dataset", "nyu-mll/glue",
            "--dataset-name", "not_a_real_glue_config",
            "--samples", "1",
        ], expect_success=False)
        assert result.exit_code != 0
        # Loose: exact wording depends on datasets lib version
        assert "config" in result.output.lower() or "not_a_real_glue_config" in result.output, (
            result.output
        )

    def test_schema_without_task(self, runner: CliRunner) -> None:
        # G.4
        result = _invoke(runner, ["--schema"], expect_success=False)
        assert result.exit_code != 0
        assert "--task" in result.output, result.output

    def test_schema_bogus_task(self, runner: CliRunner) -> None:
        # G.6 — get_evaluator_class ValueError wrapped as UsageError
        result = _invoke(
            runner, ["--schema", "--task", "not-a-real-task"],
            expect_success=False,
        )
        assert result.exit_code != 0
        out_lower = result.output.lower()
        assert ("not-a-real-task" in out_lower
                or "unknown" in out_lower
                or "unsupported" in out_lower
                or "invalid" in out_lower), result.output

    def test_onnx_file_without_model_id(
        self, runner: CliRunner, tmp_path: Path,
    ) -> None:
        # G.5 — needs a real .onnx file path that exists; reuse warmed cache
        hf_id = "google/vit-base-patch16-224"
        task = "image-classification"
        _invoke(runner, ["-m", hf_id, "--task", task, "--streaming", "--samples", SAMPLES])
        cache_dir = find_cache_dir(hf_id, task=task)
        assert cache_dir is not None
        onnx_files = list(cache_dir.glob("*_model.onnx"))
        assert onnx_files

        result = _invoke(runner, [
            "-m", str(onnx_files[0]),
            "--task", task,
            "--samples", "1",
        ], expect_success=False)
        assert result.exit_code != 0
        assert "model-id" in result.output.lower(), result.output
