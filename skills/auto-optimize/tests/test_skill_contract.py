# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Contract tests for the minimal auto-optimize skill."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = SKILL_ROOT / "knowledge"
ROLE_ROOT = SKILL_ROOT / "roles"
SCRIPT_ROOT = SKILL_ROOT / "scripts"
REFERENCE_ROOT = SKILL_ROOT / "references"
CASE_FIELDS = {
    "id",
    "status",
    "scope",
    "observation",
    "mechanism",
    "transformation",
    "expected_evidence",
    "outcome",
    "safety",
    "counterexamples",
    "provenance",
    "generic_review",
    "discovery",
}


def _load_hotspot_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_hotspot", SCRIPT_ROOT / "plan_hotspot.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_content_sha256(case: dict[str, Any]) -> str:
    content = {key: value for key, value in case.items() if key != "generic_review"}
    canonical = (
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", text, re.DOTALL)
    assert match is not None, "SKILL.md must begin with YAML frontmatter"
    values: dict[str, str] = {}
    for raw_line in match.group("body").splitlines():
        key, separator, value = raw_line.partition(":")
        assert separator, f"invalid frontmatter line: {raw_line}"
        values[key.strip()] = value.strip().strip("'\"")
    return values


def test_skill_is_small_and_single_agent() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()
    metadata = _frontmatter(text)

    assert metadata["name"] == SKILL_ROOT.name == "auto-optimize"
    assert metadata["description"].startswith("Use when ")
    assert (
        metadata["description"]
        == "Use when optimizing ONNX latency with WinML for a target EP/device, "
        "including QNN NPU profiling and graph interactions."
    )
    description = metadata["description"].lower()
    for keyword in ("onnx", "winml", "qnn", "npu", "latency"):
        assert keyword in description, keyword

    for required in (
        "at most three",
        "correctness before performance",
        "a/b",
        "b/a",
        "cache identity",
        "trace delta",
        "interaction",
        "one adjacent",
        "explicit user approval",
        "graph scout",
        "perf arbiter",
        "feature gap",
        "draft pr",
        "ready for check-in",
        "report.json",
        "report.html",
        "save_case.py",
        "ponytail",
        "bundled knowledge is model-agnostic",
        "run-local",
    ):
        assert required in lowered, required

    for forbidden in (
        "search-planner",
        "scheduler",
        "phase 0 html",
        "typed plan",
    ):
        assert forbidden not in lowered, forbidden


def test_auto_optimize_is_checked_in_as_a_repository_skill() -> None:
    skills_root = SKILL_ROOT.parent
    repository_root = skills_root.parent

    assert skills_root.name == "skills"
    assert (repository_root / "pyproject.toml").is_file()
    assert not (skills_root / "auto-config").exists()


def test_hypotheses_require_winml_evidence_first() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()

    for required in (
        "winml inspect",
        "winml analyze",
        "--check-optim",
        "winml perf",
        "evidence brief",
        "ihv sdk",
        "detail profile",
        "hardware time",
        "memory time",
        "dram",
        "evidence gap",
        "unattributed provider work",
        "not evidence of no hotspot",
    ):
        assert required in lowered, required

    evidence_gate = lowered.index("evidence brief")
    hypothesis_loop = lowered.index("maintain at most three active hypotheses")
    assert evidence_gate < hypothesis_loop


def test_confirmed_candidate_can_lead_within_noise() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for required in (
        "confirmed against baseline",
        "lower point estimate",
        "statistical tie",
        "do not claim superiority",
    ):
        assert required in text, required


def test_provisional_quality_delivery_requires_disclosed_unavailable_evaluator() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for required in (
        "task evaluator unavailable",
        "provisional-quality",
        "tensor validation",
        "disclose the evidence gap",
    ):
        assert required in text, required


def test_every_measurement_reuses_frozen_provider_options_explicitly() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for required in (
        "frozen provider options",
        "explicitly",
        "every wall/perf/profile command",
        "compiled-context profiling",
    ):
        assert required in text, required


def test_material_leaders_trigger_staged_capability_rediscovery() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for required in (
        "material leader",
        "llm",
        "capability closure review",
        "verbose registry",
        "residual topology",
        "at most three",
        "probe_required",
        "graph-changing probes",
        "deferred_budget",
    ):
        assert required in text, required
    assert "capability_closure.py" not in text
    assert not (SCRIPT_ROOT / "capability_closure.py").exists()

    reference = (REFERENCE_ROOT / "capability-closure.md").read_text(encoding="utf-8").lower()
    for required in (
        "list-capabilities --verbose",
        "analyzer non-reporting is detector evidence only",
        "inverses",
        "interactions",
        "at most three",
        "probe_required",
        "closed_ineligible",
        "closed_already_tested",
        "closed_registry_absent",
        "deferred_budget",
        "material_omission_found",
        "insufficient_evidence",
        "no_material_omission",
    ):
        assert required in reference, required
    assert "gather-slice-to-split" not in reference


def test_dominant_hotspot_fast_lane_is_bounded_and_evidence_gated() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for phrase in (
        "## planning router - evaluate before loading cases or proposing hypotheses",
        "70 percent",
        "at most two probes",
        "provider-attributed detail trace",
        "priority only",
        "does not prune",
        "qdq-boundary",
        "write `hotspot_evidence.json`",
        "resolve [plan_hotspot.py](./scripts/plan_hotspot.py)",
        "run `python ./scripts/plan_hotspot.py hotspot_evidence.json --output hotspot_plan.json`",
        "adopt the helper result as the current plan",
        (
            "if mode is `dominant-hotspot-fast-lane`, execute only its steps and "
            "exit instruction before loading cases or proposing normal-loop hypotheses."
        ),
        "if mode is `normal-hypothesis-loop`, continue normally.",
    ):
        assert phrase in text, phrase

    for phrase in (
        (
            "execute only its steps and exit instruction before loading cases or "
            "proposing normal-loop hypotheses."
        ),
        "current plan",
        "hotspot_plan.json",
    ):
        assert phrase in text, phrase

    assert (
        text.index("write an evidence brief on bottlenecks, provider work, gaps before")
        < text.index("## planning router - evaluate before loading cases or proposing hypotheses")
        < text.index(
            "run `python ./scripts/plan_hotspot.py hotspot_evidence.json --output "
            "hotspot_plan.json`"
        )
        < text.index(
            "read [`knowledge/index.json`](./knowledge/index.json), match ep/device anchors,"
        )
        < text.index("maintain at most three active hypotheses")
    )


def test_dominant_hotspot_hard_gate_stops_before_normal_loop() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    active_gate_window = text[
        text.index(
            "run `python ./scripts/plan_hotspot.py hotspot_evidence.json --output "
            "hotspot_plan.json`"
        ) : text.index("maintain at most three active hypotheses")
    ]
    for forbidden in (
        "active hypotheses:",
        "accepted baseline:",
        "load cases:",
        "list cases:",
        "analyzer options:",
        "layout follow-up:",
        "provider follow-up:",
        "quantization-format follow-up:",
        "experiments a",
        "experiments b",
        "experiments c",
        "experiments d",
        "experiments e",
        "feature gap",
    ):
        assert forbidden not in active_gate_window, forbidden


def test_hotspot_helper_invocation_is_linked_resolved_and_byte_verified() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()

    for phrase in (
        "[plan_hotspot.py](./scripts/plan_hotspot.py)",
        "resolve that linked file path",
        "do not infer a workspace-root `scripts/` directory",
        "exit code 0",
        "stdout parses as json",
        "stdout bytes equal the `--output` file bytes",
        "never synthesize or rewrite the plan json",
    ):
        assert phrase in lowered, phrase

    helper_window = lowered[
        lowered.index("[plan_hotspot.py](./scripts/plan_hotspot.py)") : lowered.index(
            "read [`knowledge/index.json`](./knowledge/index.json), match ep/device anchors,"
        )
    ]
    for forbidden in (
        '"evidence"',
        '"probes"',
        '"exit_after"',
    ):
        assert forbidden not in helper_window, forbidden


def test_graph_scout_enforces_capability_closure_before_stop() -> None:
    text = (ROLE_ROOT / "graph-scout.md").read_text(encoding="utf-8").lower()

    for required in (
        "residual topology inventory",
        "capability closure ledger",
        "analyzer non-reporting",
        "detector evidence",
        "probe_required",
        "deferred_budget",
        "no_material_omission",
        "insufficient_evidence",
    ):
        assert required in text, required
    restriction = text.index("no_material_omission")
    assert text.index("probe_required") < restriction
    assert text.index("deferred_budget") < restriction


def test_llm_closure_pressure_scenario_covers_analyzer_false_negative() -> None:
    scenario = (
        (SKILL_ROOT / "tests" / "pressure" / "capability-closure-false-negative.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    for required in (
        "no reference model",
        "check-optim",
        "unrelated capabilities",
        "verbose registry",
        "probe_required",
        "detector evidence",
        "public-output preservation",
        "material_omission_found",
        "future registry entry",
        "--enable-static-pad-into-conv",
        "second",
        "probe_required",
        "do not require a production code change",
    ):
        assert required in scenario, required


def test_bundle_publication_precedes_promotion() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert text.index("finalize_output.py") < text.index("promotion.py create")
    routing = (SKILL_ROOT / "references" / "pr-routing.md").read_text(encoding="utf-8")
    assert "new versioned bundle and handoff" in routing


def test_final_output_bundle_is_mandatory_before_stop() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    lowered = text.lower()

    for required in (
        "finalize_output.py",
        "champion.onnx",
        "companion files",
        "winml_config.json",
        "rebuild_config.json",
        "repro.ps1",
        "repro-run.ps1",
        "repro.lock.json",
        "perf_input.npz",
        "eval_inputs.npz",
        "inputs_manifest.json",
        "generated wrapper",
        "replay body",
        "validateonly",
        "manifest.json",
        "fresh temporary directory",
        "requires-unmerged-pr",
        "semantically separate",
        "validate the published bundle",
    ):
        assert required in lowered, required
    assert (SCRIPT_ROOT / "finalize_output.py").is_file()
    assert lowered.index("finalize_output.py") < lowered.index("## stop")


def test_event_roles_are_small_and_have_closed_outputs() -> None:
    contracts = {
        "graph-scout.md": (
            "baseline",
            "new leader",
            "before stopping",
            "read-only",
            "at most three",
            "no_material_omission",
        ),
        "perf-arbiter.md": (
            "confirmed",
            "inconclusive",
            "harness_error",
            "regression",
            "a/b",
            "b/a",
            "cache identity",
        ),
        "feature-gap-engineer.md": (
            "isolated worktree",
            "origin/main",
            "test-driven",
            "generic",
            "main agent",
            "paired",
            "ponytail",
            "complexity-review.md",
        ),
        "checkin-reviewer.md": (
            "ready_for_check_in",
            "changes_requested",
            "perf_not_proven",
            "blocked",
            "direct baseline",
            "confidence interval",
            "trace",
            "complexity review",
        ),
    }

    for filename, required_phrases in contracts.items():
        text = (ROLE_ROOT / filename).read_text(encoding="utf-8").lower()
        for phrase in required_phrases:
            assert phrase in text, f"{filename}:{phrase}"


def test_ponytail_reference_is_optional_versioned_and_safe() -> None:
    text = (REFERENCE_ROOT / "ponytail.md").read_text(encoding="utf-8").lower()
    assert len(re.findall(r"\S+", text)) < 350
    for required in (
        "https://github.com/dietrichgebert/ponytail",
        "copilot plugin list",
        "/ponytail:ponytail full",
        "/ponytail:ponytail-review",
        "do not install",
        "fallback",
        "understand the root cause",
        "correctness",
        "security",
        "performance",
        "complexity-review.md",
        "plugin version",
        "net removable lines",
    ):
        assert required in text, required


def test_knowledge_index_is_small_lazy_and_valid() -> None:
    index = _load_json(KNOWLEDGE_ROOT / "index.json")

    assert index["version"] == 1
    assert index["max_cases_per_round"] == 3
    entries = index["cases"]
    assert len(entries) == 6
    assert len({entry["id"] for entry in entries}) == len(entries)

    loaded_cases: list[dict[str, Any]] = []
    for entry in entries:
        assert {
            "id",
            "status",
            "ep",
            "device",
            "anchor_ops",
            "keywords",
            "lesson",
            "path",
            "sha256",
        } <= entry.keys()
        assert entry["anchor_ops"]
        assert entry["lesson"]

        relative_path = Path(entry["path"])
        assert not relative_path.is_absolute()
        case_path = (KNOWLEDGE_ROOT / relative_path).resolve()
        assert KNOWLEDGE_ROOT.resolve() in case_path.parents
        assert case_path.is_file()

        case = _load_json(case_path)
        assert hashlib.sha256(case_path.read_bytes()).hexdigest() == entry["sha256"]
        assert case.keys() >= CASE_FIELDS
        assert case["id"] == entry["id"]
        assert case["status"] == entry["status"]
        assert case["discovery"] == {
            "ep": entry["ep"],
            "device": entry["device"],
            "anchor_ops": entry["anchor_ops"],
            "keywords": entry["keywords"],
            "lesson": entry["lesson"],
        }
        for field in CASE_FIELDS - {"id"}:
            assert case[field] not in (None, "", [], {}), f"{case['id']}:{field}"
        loaded_cases.append(case)

    statuses = {case["status"] for case in loaded_cases}
    assert "confirmed" in statuses
    assert "rejected" in statuses
    exp_case = next(case for case in loaded_cases if case["id"] == "positive-exp-scale")
    assert exp_case["status"] == "confirmed-performance-provisional-quality"
    quality = exp_case["outcome"]["quality"].lower()
    assert "requires" in quality
    assert "each matching run" in quality


def test_grouped_conv_qdq_case_is_indexed() -> None:
    index = _load_json(KNOWLEDGE_ROOT / "index.json")

    entry = next(item for item in index["cases"] if item["id"] == "grouped-conv-qdq-boundary")
    assert entry["anchor_ops"] == [
        "Conv",
        "Slice",
        "Split",
        "Concat",
        "QuantizeLinear",
        "DequantizeLinear",
    ]


def test_bundled_knowledge_is_generic_only() -> None:
    case_paths = sorted((KNOWLEDGE_ROOT / "cases").glob("*.json"))
    assert case_paths

    def inspect(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                inspect(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{location}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            assert not isinstance(value, (int, float)), f"{location}: numeric run value"
        elif isinstance(value, str):
            lowered = value.lower()
            assert not re.fullmatch(r"[0-9a-f]{64}", lowered), f"{location}: exact hash"
            assert not re.search(
                r"(?:[a-z]:[\\/]|(?:^|\s)/[^\s]+|\b[^\s]+\.onnx\b)",
                value,
                re.IGNORECASE,
            ), f"{location}: model/path-specific value"
            assert not re.search(r"\bv?\d+\.\d+(?:\.\d+)+\b", value), (
                f"{location}: provider/toolchain version"
            )
            assert not re.search(r"\bpattern-\d+\b", value, re.IGNORECASE), (
                f"{location}: historical pattern ID"
            )
            assert not re.search(r"\bnode_[a-z0-9_]+\b", value, re.IGNORECASE), (
                f"{location}: concrete node name"
            )

    for case_path in case_paths:
        case = _load_json(case_path)
        assert set(case["scope"]) == {"ep", "device", "graph_requirements"}
        assert set(case["provenance"]) == {"evidence_class", "scope_note"}
        assert case["generic_review"] == {
            "verdict": "GENERIC_CASE_APPROVED",
            "reviewer": "independent-graph-scout",
            "content_sha256": _case_content_sha256(case),
        }
        inspect(
            {key: value for key, value in case.items() if key != "generic_review"},
            case_path.name,
        )


def test_qnn_reference_preserves_only_high_value_decisions() -> None:
    text = (KNOWLEDGE_ROOT / "qnn-npu.md").read_text(encoding="utf-8").lower()

    for required in (
        "node count",
        "split",
        "slice",
        "transpose",
        "partition",
        "cache identity",
        "profiled wall latency",
    ):
        assert required in text, required
    assert "[planning router]" in text
    assert "python scripts/plan_hotspot.py" not in text


def test_dominant_hotspot_pressure_scenario_requires_two_step_recipe() -> None:
    text = (
        (SKILL_ROOT / "tests" / "pressure" / "dominant-hotspot-fast-lane.md")
        .read_text(encoding="utf-8")
        .lower()
    )

    for required in (
        "invoke the helper",
        "write `hotspot_evidence.json`",
        "resolve [`plan_hotspot.py`](../../scripts/plan_hotspot.py)",
        "python ./scripts/plan_hotspot.py hotspot_evidence.json --output hotspot_plan.json",
        (
            "adopt the helper result only after exit code 0, stdout parses as json, "
            "and stdout bytes equal `hotspot_plan.json` bytes"
        ),
        "never synthesize, rewrite, or replace the helper result with a free-form plan",
        "70 percent",
        "at most two probes",
        "priority only",
        "does not prune",
        "qdq-boundary",
    ):
        assert required in text, required

    disallowed_current_plan = (
        "layout",
        "provider option",
        "quantization format",
        "active hypotheses",
        "experiments a",
        "experiments b",
        "experiments c",
        "experiments d",
        "experiments e",
        "feature gap",
    )
    current_plan_window = text[text.index("success requires:") : text.index("failure criteria:")]
    for forbidden in disallowed_current_plan:
        assert forbidden not in current_plan_window, forbidden


def test_dominant_hotspot_pressure_scenario_embeds_exact_helper_json() -> None:
    module = _load_hotspot_module()
    pressure_evidence = {
        "schema_version": 1,
        "provider_attribution": "valid",
        "dominant_accelerator_time_pct": 92,
        "fallback_is_larger_explanation": False,
        "partitioning_is_larger_explanation": False,
        "transfers_are_larger_explanation": False,
        "quantized": True,
        "outcomes": {
            "representation": None,
            "qdq-boundary": None,
        },
    }
    expected = (
        json.dumps(
            module.plan_hotspot(pressure_evidence),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    raw_text = (SKILL_ROOT / "tests" / "pressure" / "dominant-hotspot-fast-lane.md").read_text(
        encoding="utf-8"
    )

    assert expected in raw_text


def test_deterministic_support_scripts_are_present() -> None:
    assert (SCRIPT_ROOT / "plan_hotspot.py").is_file()
    assert (SCRIPT_ROOT / "save_case.py").is_file()
    assert (SCRIPT_ROOT / "render_report.py").is_file()
    assert (SCRIPT_ROOT / "finalize_output.py").is_file()
    assert (SCRIPT_ROOT / "promotion.py").is_file()


def test_pr_routing_has_two_exclusive_owners_and_local_merge_proof() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()
    routing = (REFERENCE_ROOT / "pr-routing.md").read_text(encoding="utf-8").lower()

    assert len(routing.splitlines()) <= 100
    for required in (
        "model-support-promotion-v1",
        "model-opt-by-skill",
        "model-scale-by-skill",
        "auto-optimize creates and reviews only optimizer prs",
        "adding-model-support creates and reviews only recipe prs",
        "fetch current main before",
        "does not fetch",
        "--optimizer-repo",
        "--merged-commit",
        "--current-main-commit",
    ):
        assert required in routing, required
    assert "[pr routing](./references/pr-routing.md)" in skill


def test_public_cli_promotion_and_pr_label_contract_are_mandatory() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8").lower()

    for required in (
        "clean directory",
        "public cli",
        "exact serialized build config",
        "only that public-path artifact may become final leader",
        "prototype artifacts remain in experiment lineage",
        "optimizer pr",
        "gh label list",
        "must not create the label automatically",
        "gh pr create --draft --label model-opt-by-skill",
        "gh pr view <url> --json labels",
        "missing or unavailable label blocks handoff",
        "missing post-create label verification blocks handoff",
        "promotion.py",
        "promotion_handoff.json",
    ):
        assert required in text, required


def test_feature_gap_engineer_requires_clean_public_cli_validation_and_verified_label() -> None:
    text = (ROLE_ROOT / "feature-gap-engineer.md").read_text(encoding="utf-8").lower()

    for required in (
        "clean directory",
        "public cli",
        "exact effective serialized config",
        "clean-directory validation evidence",
        "main agent",
        "optimizer pr",
    ):
        assert required in text
    assert "gh pr create" not in text


def test_checkin_reviewer_blocks_prototype_promotion_and_missing_label_evidence() -> None:
    text = (ROLE_ROOT / "checkin-reviewer.md").read_text(encoding="utf-8").lower()

    for required in (
        "clean public cli composition",
        "prototype",
        "model-opt-by-skill",
        "label verification evidence",
        "optimizer pr",
        "final target evidence",
    ):
        assert required in text, required


def test_entry_routes_before_measurement_and_loads_scout():
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert text.index("## Entry") < text.index("Run `winml inspect`")
    assert "[Graph Scout](./roles/graph-scout.md)" in text


def test_engineering_does_not_create_pr_before_bundle():
    text = (ROLE_ROOT / "feature-gap-engineer.md").read_text(encoding="utf-8")
    assert "gh pr create" not in text
    assert "public" in text


def test_repeated_rebuild_has_explicit_evidence_contract():
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "[reproduction](./references/reproduction.md)" in text
    reference = (REFERENCE_ROOT / "reproduction.md").read_text(encoding="utf-8")
    assert "independent clean builds" in reference
    assert "calibration" in reference


def test_skill_text_has_no_encoding_corruption():
    roots = [SKILL_ROOT, SKILL_ROOT.parents[1] / "docs" / "getting-started" / "agent-skill"]
    bad_sequences = (
        chr(0x00E2) + chr(0x20AC),
        chr(0xFFFD),
        chr(0x00EF) + chr(0x00BB) + chr(0x00BF),
    )
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".md", ".py", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in bad_sequences), str(path)
