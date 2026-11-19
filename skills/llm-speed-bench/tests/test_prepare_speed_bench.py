# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        "prepare_speed_bench", Path(__file__).parents[1] / "scripts/prepare_speed_bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look up their module in sys.modules while the module executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


psb = load()


def make_rows(prefix, layout, masked=()):
    """Synthetic rows: layout maps (category, sub_category) to a row count."""
    rows = []
    for (category, sub_category), count in layout.items():
        for i in range(count):
            qid = f"{prefix}-{category}-{sub_category}-{i}"
            text = f"{psb.PLACEHOLDER}\n\n{{x}}" if qid in masked else f"prompt {qid}"
            rows.append(
                {
                    "question_id": qid,
                    "category": category,
                    "sub_category": sub_category,
                    "turns": [text, f"second turn {qid}"],
                    "source": "https://example.org/src",
                }
            )
    return rows


QUALITATIVE = {("math", None): 5, ("coding", None): 5, ("writing", "a"): 3, ("writing", "b"): 3}
THROUGHPUT = {
    ("low_entropy", "code_completion"): 4,
    ("low_entropy", "code_debug"): 4,
    ("low_entropy", "sort"): 4,
    ("mixed", "NIAH"): 4,
    ("high_entropy", "open_books"): 4,
}


def rows_by_config():
    data = {"qualitative": make_rows("q", QUALITATIVE)}
    for config in psb.THROUGHPUT_CONFIGS:
        data[config] = make_rows(config, THROUGHPUT)
    return data


def test_selection_is_deterministic_and_independent_of_row_order():
    data = rows_by_config()
    shuffled = {config: random.Random(7).sample(rows, len(rows)) for config, rows in data.items()}
    for subset in psb.SUBSETS.values():
        first = [(c, r["question_id"]) for c, r in psb.select_subset(data, subset)]
        second = [(c, r["question_id"]) for c, r in psb.select_subset(shuffled, subset)]
        assert first == second


def test_mini_takes_per_group_rows_and_alternates_families():
    selected = psb.select_subset(rows_by_config(), psb.SUBSETS["mini"])
    counts = Counter((config, row["category"]) for config, row in selected)
    assert counts == {
        **{("qualitative", category): 2 for category in ("math", "coding", "writing")},
        **{
            (config, category): 2
            for config in ("throughput_1k", "throughput_8k", "throughput_16k")
            for category in ("low_entropy", "mixed", "high_entropy")
        },
    }
    writing = [r["sub_category"] for c, r in selected if r["category"] == "writing"]
    assert sorted(writing) == ["a", "b"]
    low = [
        r["sub_category"]
        for c, r in selected
        if c == "throughput_1k" and r["category"] == "low_entropy"
    ]
    assert {psb.family_of(s) for s in low} == {"code", "sort"}


def test_selection_follows_stable_hash_order_within_a_group():
    data = {"qualitative": make_rows("q", {("math", None): 9})}
    selected = psb.select_subset(data, psb.SUBSETS["mini"])
    ids = sorted(
        (r["question_id"] for r in data["qualitative"]),
        key=lambda q: hashlib.sha256(q.encode()).hexdigest(),
    )
    assert [r["question_id"] for _, r in selected] == ids[:2]


def test_selected_rows_are_interleaved_across_groups():
    selected = psb.select_subset(rows_by_config(), psb.SUBSETS["mini"])
    first_round = selected[:12]
    assert len({(c, r["category"]) for c, r in first_round}) == 12


def test_throughput_subset_groups_by_family_and_full_takes_every_row():
    data = rows_by_config()
    throughput = psb.select_subset(data, psb.SUBSETS["throughput"])
    groups = Counter((c, r["category"], psb.family_of(r["sub_category"])) for c, r in throughput)
    assert set(groups.values()) == {2}
    assert len(groups) == 4 * len(psb.THROUGHPUT_CONFIGS)
    code = [
        r["sub_category"]
        for c, r in throughput
        if c == "throughput_2k" and r["sub_category"].startswith("code")
    ]
    assert sorted(code) == ["code_completion", "code_debug"]
    full = psb.select_subset(data, psb.SUBSETS["full"])
    assert len(full) == sum(len(rows) for rows in data.values())


def test_first_turn_extraction():
    assert psb.first_turn(["one", "two"]) == "one"
    assert (
        psb.first_turn([{"role": "user", "content": "hi"}, {"role": "user", "content": "x"}])
        == "hi"
    )
    assert psb.first_turn([]) == ""


def test_finalize_drops_placeholder_and_empty_rows_and_keeps_first_turn():
    rows = make_rows("throughput_8k", {("mixed", "NIAH"): 3}, masked={"throughput_8k-mixed-NIAH-1"})
    rows[2]["turns"] = ["  "]
    out, dropped = psb.finalize_rows([("throughput_8k", row) for row in rows])
    assert dropped == 2
    assert out == [
        {
            "id": "throughput_8k-mixed-NIAH-0",
            "prompt": "prompt throughput_8k-mixed-NIAH-0",
            "category": "mixed",
            "sub_category": "NIAH",
            "bucket": "8k",
            "source": "https://example.org/src",
        }
    ]


def test_finalize_rejects_duplicate_ids():
    row = make_rows("q", {("math", None): 1})[0]
    with pytest.raises(ValueError, match="twice"):
        psb.finalize_rows([("qualitative", row), ("throughput_1k", row)])


def test_exclude_sources_skips_only_masked_rows_that_need_them():
    rows = make_rows("q", {("math", None): 3}, masked={"q-math-None-0", "q-math-None-1"})
    rows[0]["source"] = "https://huggingface.co/datasets/org/gated"
    rows[2]["source"] = "https://huggingface.co/datasets/org/gated"
    kept, skipped = psb.exclude_sources(rows, ["org/gated"], ["org/gated", "org/open"])
    assert [r["question_id"] for r in kept] == ["q-math-None-1", "q-math-None-2"]
    assert skipped == {"org/gated": 1}


def test_write_subset_manifest_records_pins_counts_and_command(tmp_path):
    subset = psb.SUBSETS["mini"]
    pairs = psb.select_subset(rows_by_config(), subset)
    rows, dropped = psb.finalize_rows(pairs)
    manifest = psb.write_subset(
        tmp_path, "mini", subset, rows, skipped_sources={"org/gated": 4}, dropped_unresolved=dropped
    )
    data = (tmp_path / "mini" / "test.jsonl").read_bytes()
    lines = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    assert [list(line) for line in lines] == [list(psb.OUTPUT_COLUMNS)] * len(rows)
    assert json.loads((tmp_path / "mini" / "manifest.json").read_text("utf-8")) == manifest
    assert manifest["row_count"] == len(rows) == 24
    assert manifest["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["counts"]["bucket"] == {"16k": 6, "1k": 6, "8k": 6, "qualitative": 6}
    assert manifest["counts"]["bucket_category"]["8k/mixed"] == 2
    assert manifest["skipped_gated_rows"] == {"org/gated": 4}
    assert manifest["pins"] == {
        "speed_bench_dataset": "nvidia/SPEED-Bench",
        "speed_bench_revision": psb.SPEED_BENCH_REVISION,
        "prepare_py_repo": psb.PREPARE_PY_REPO,
        "prepare_py_commit": psb.PREPARE_PY_COMMIT,
        "prepare_py_url": psb.PREPARE_PY_URL,
        "prepare_py_sha256": psb.PREPARE_PY_SHA256,
    }
    assert psb.PREPARE_PY_COMMIT in psb.PREPARE_PY_URL
    command = manifest["recommended_command"]
    assert f"--dataset {tmp_path / 'mini'}" in command
    assert "--max-new-tokens 256" in command
    assert "--column cohort_column=bucket,category" in command


def test_prepare_module_with_wrong_hash_is_not_imported(tmp_path):
    (tmp_path / "prepare.py").write_text("raise SystemExit('must not run')\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        psb.load_prepare_module(tmp_path)
