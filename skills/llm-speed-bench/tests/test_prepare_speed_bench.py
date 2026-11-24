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
from types import SimpleNamespace

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


@pytest.mark.parametrize("subset_name", psb.SUBSETS)
def test_prefix_spans_buckets_across_subset_parts(subset_name):
    data = rows_by_config()
    subset = psb.SUBSETS[subset_name]
    configs = {config for part in subset.parts for config in part.configs}
    selected = psb.select_subset(data, subset)
    assert {config for config, _ in selected[: len(configs)]} == configs
    assert {category for category, _ in THROUGHPUT} <= {
        row["category"] for _, row in selected[: len(configs)]
    }
    expected_ids = {
        row["question_id"]
        for part in subset.parts
        for group in psb.select_part(data, part)
        for _, row in group
    }
    assert Counter(row["question_id"] for _, row in selected) == Counter(expected_ids)


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
    tmp_path = tmp_path / "output with spaces"
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
    assert f'--dataset "{tmp_path / "mini"}"' in command
    assert "--max-new-tokens 256" in command
    assert "--column cohort_column=bucket,category" in command


def test_prepare_module_with_wrong_hash_is_not_imported(tmp_path):
    (tmp_path / "prepare.py").write_text("raise SystemExit('must not run')\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        psb.load_prepare_module(tmp_path)


def test_prepare_module_executes_verified_bytes_and_is_registered(tmp_path, monkeypatch):
    source = f"TURNS_PLACEHOLDER = {psb.PLACEHOLDER!r}\n"
    path = tmp_path / "prepare.py"
    path.write_text(source, encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify = psb.verify_sha256

    def verify_and_replace(data):
        verify(data, expected)
        path.write_text("raise RuntimeError('unverified code ran')\n", encoding="utf-8")

    monkeypatch.setattr(psb, "verify_sha256", verify_and_replace)
    monkeypatch.setitem(sys.modules, psb.PREPARE_MODULE_NAME, None)
    module = psb.load_prepare_module(tmp_path)
    assert sys.modules[module.__name__] is module
    assert module.TURNS_PLACEHOLDER == psb.PLACEHOLDER
    assert module.__file__ == str(path)


def test_prepare_module_restores_registration_on_error(tmp_path, monkeypatch):
    path = tmp_path / "prepare.py"
    path.write_text("raise RuntimeError('upstream failed')\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify = psb.verify_sha256
    monkeypatch.setattr(psb, "verify_sha256", lambda data: verify(data, expected))
    previous = SimpleNamespace()
    monkeypatch.setitem(sys.modules, psb.PREPARE_MODULE_NAME, previous)
    with pytest.raises(RuntimeError, match="upstream failed"):
        psb.load_prepare_module(tmp_path)
    assert sys.modules[psb.PREPARE_MODULE_NAME] is previous


def test_prepare_module_pin_changes_resolver_fingerprint(tmp_path, monkeypatch):
    from datasets.fingerprint import Hasher

    verify = psb.verify_sha256
    hashes = []
    for version in range(2):
        source = (
            f"TURNS_PLACEHOLDER = {psb.PLACEHOLDER!r}\n"
            f"def resolve(row):\n    return {{**row, 'version': {version}}}\n"
        ).encode()
        digest = hashlib.sha256(source).hexdigest()
        name = f"speed_bench_prepare_{digest}"
        monkeypatch.setattr(psb, "PREPARE_MODULE_NAME", name)
        monkeypatch.setattr(psb, "verify_sha256", lambda data, digest=digest: verify(data, digest))
        monkeypatch.setitem(sys.modules, name, None)
        (tmp_path / "prepare.py").write_bytes(source)
        module = psb.load_prepare_module(tmp_path)
        fingerprint = Hasher.hash(module.resolve)
        assert fingerprint == Hasher.hash(psb.load_prepare_module(tmp_path).resolve)
        hashes.append(fingerprint)
    assert len(set(hashes)) == len(hashes)


def test_hub_sources_only_probes_masked_hub_dataset_urls():
    rows = make_rows("q", {("category", None): 4})
    for row in rows[:3]:
        row["turns"] = [psb.PLACEHOLDER]
    rows[0]["source"] = "https://huggingface.co/datasets/org/gated/resolve/main/data.json"
    rows[1]["source"] = "http://example.org/org/label/data.json"
    rows[2]["source"] = "https://huggingface.co.evil.test/datasets/org/label"
    rows[3]["source"] = "https://huggingface.co/datasets/org/unmasked"
    assert psb._hub_sources(rows) == ["org/gated"]


@pytest.mark.parametrize("force_skip", [False, True])
def test_gated_lookup_failure_is_not_treated_as_accessible(monkeypatch, force_skip):
    import huggingface_hub

    def fail(*args, **kwargs):
        raise TimeoutError("Hub unavailable")

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", fail)
    with pytest.raises(RuntimeError, match="Cannot inspect Hub dataset org/gated"):
        psb._gated_unavailable(["org/gated"], force_skip)


@pytest.mark.parametrize("force_skip", [False, True])
def test_gated_access_and_forced_skip(monkeypatch, force_skip):
    import huggingface_hub

    checked = []
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-token")
    monkeypatch.setattr(
        huggingface_hub.HfApi, "dataset_info", lambda self, source: SimpleNamespace(gated=True)
    )
    monkeypatch.setattr(
        huggingface_hub.HfApi, "auth_check", lambda self, source, **kwargs: checked.append(source)
    )
    assert psb._gated_unavailable(["org/gated"], force_skip) == (
        ["org/gated"] if force_skip else []
    )
    assert checked == ([] if force_skip else ["org/gated"])


@pytest.mark.parametrize("status", [403, 503])
def test_gated_access_denial_is_distinct_from_service_failure(monkeypatch, status):
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-token")
    monkeypatch.setattr(
        huggingface_hub.HfApi, "dataset_info", lambda self, source: SimpleNamespace(gated=True)
    )

    def fail(*args, **kwargs):
        raise HfHubHTTPError(
            "access check failed",
            response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co")),
        )

    monkeypatch.setattr(huggingface_hub.HfApi, "auth_check", fail)
    if status == 403:
        assert psb._gated_unavailable(["org/gated"], False) == ["org/gated"]
    else:
        with pytest.raises(RuntimeError, match="Cannot check access"):
            psb._gated_unavailable(["org/gated"], False)


def test_prepare_probes_metadata_sources_and_explains_forced_skip(tmp_path, monkeypatch, capsys):
    import datasets
    import huggingface_hub

    rows = make_rows("q", {("category", None): 3}, masked={"q-category-None-0"})
    rows[0]["source"] = "https://huggingface.co/datasets/org/gated/resolve/main/data.json"
    rows[1]["source"] = "https://example.org/org/nonhub"
    metadata = datasets.Dataset.from_list(rows)
    loads = []
    probes = []
    module = SimpleNamespace(
        BenchmarkDataset=[SimpleNamespace(value=value) for value in ("org/gated", "org/nonhub")],
        _resolve_external_data=lambda selected, config: selected,
    )

    def load(dataset, config, *, split, revision):
        loads.append((dataset, config, split, revision))
        return metadata

    def info(self, source):
        probes.append(source)
        return SimpleNamespace(gated=True)

    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr(psb, "load_prepare_module", lambda path: module)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", info)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "test-token")
    subset = psb.Subset("test", (psb.Part((psb.QUALITATIVE,), psb.BY_CATEGORY, 2),), 256)
    monkeypatch.setitem(psb.SUBSETS, "test", subset)
    manifests = psb.prepare(tmp_path, ["test"], skip_gated=True)
    assert loads == [(psb.SPEED_BENCH_DATASET, psb.QUALITATIVE, "test", psb.SPEED_BENCH_REVISION)]
    assert probes == ["org/gated"]
    assert manifests[0]["row_count"] == 2
    assert manifests[0]["skipped_gated_rows"] == {"org/gated": 1}
    assert manifests[0]["dropped_unresolved_rows"] == 0
    output = capsys.readouterr().out
    assert "--skip-gated was requested" in output
    assert "cannot read it" not in output
