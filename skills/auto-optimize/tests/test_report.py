# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deterministic optimization report tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from types import ModuleType


SKILL_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_ROOT / "scripts" / "render_report.py"


@pytest.mark.parametrize("field", ["p50_ms", "gain_pct", "ci_low_pct", "ci_high_pct"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_final_report_rejects_invalid_performance(report_module, field, value):
    report = _report()
    report["leader"][field] = value
    with pytest.raises(report_module.ReportError):
        report_module.validate_report(report, final=True)


@pytest.fixture(scope="module")
def report_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_optimize_render_report", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "updates",
    [
        {"p50_ms": 0},
        {"p50_ms": -1},
        {"ci_low_pct": 10, "ci_high_pct": 5},
        {"ci_low_pct": None, "ci_high_pct": 5},
    ],
)
def test_final_report_rejects_invalid_metric_bounds(report_module, updates):
    report = _report()
    report["leader"].update(updates)
    with pytest.raises(report_module.ReportError):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0, -1])
def test_final_report_rejects_invalid_baseline(report_module, value):
    report = _report()
    report["baseline"]["p50_ms"] = value
    with pytest.raises(report_module.ReportError):
        report_module.validate_report(report, final=True)


def _report() -> dict[str, Any]:
    report = {
        "schema_version": 2,
        "title": "Model <unsafe> optimization",
        "updated_at": "2026-08-12T00:00:00Z",
        "model": {
            "path": "model.onnx",
            "sha256": "a" * 64,
            "opset": 18,
            "node_count": 195,
            "inputs": [{"name": "x", "shape": [1, 4], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [1, 4], "dtype": "float32"}],
            "op_counts": {"Conv": 52, "Slice": 8},
            "components": [{"name": "tail", "nodes": 20, "note": "Gaussian outputs"}],
        },
        "target": {
            "ep": "QNNExecutionProvider",
            "device": "NPU",
            "latency_target_ms": 25.0,
            "provider_options": {"htp_performance_mode": "burst"},
        },
        "baseline": {
            "p50_ms": 27.2,
            "p90_ms": 28.1,
            "p99_ms": 29.4,
            "throughput_ips": 36.8,
            "protocol": "20 warmups / 200 iterations",
            "hotspots": [
                {
                    "name": "Slice",
                    "hardware_time_us": 2000,
                    "memory_time_us": 800,
                    "dram_bytes": 4096,
                    "share_pct": 31.0,
                }
            ],
            "trace": {"partitions": 1, "transpose": 6},
        },
        "evidence": {
            "diagnosis": "The routed output tail dominates accelerator time.",
            "execution": {
                "accelerator_pct": 78.0,
                "host_overhead_pct": 22.0,
                "partition_count": 1,
                "fallback_nodes": 0,
                "transfers": 2,
            },
            "analyzer": {
                "runtime_support": "supported",
                "coverage": [
                    {"classification": "supported", "count": 188},
                    {"classification": "partial", "count": 7},
                ],
                "optimizations": [
                    {
                        "name": "channel affine folding",
                        "status": "available",
                        "instances": "routed tail",
                    }
                ],
            },
            "detail_profile": {
                "status": "available",
                "hardware_time_us": 12400,
                "memory_time_us": 4100,
                "ddr_read_bytes": 8192,
                "ddr_write_bytes": 4096,
                "artifacts": ["qhas_output.json", "qnn_htp_summary.html"],
            },
            "ranked_levers": [
                {
                    "rank": 1,
                    "lever": "Normalize routes and fold affine leaves",
                    "evidence": "Analyzer opportunity overlaps the hottest tail.",
                    "confidence": "high",
                }
            ],
            "gaps": ["Task-quality acceptance remains run-local."],
        },
        "leader": {
            "id": "c1",
            "status": "confirmed",
            "model_path": "winner.onnx",
            "p50_ms": 8.0,
            "gain_pct": 20.0,
            "ci_low_pct": 18.0,
            "ci_high_pct": 22.0,
            "correctness": "pass",
            "quality": "local-smoke pass; FFHQ pending",
        },
        "hypotheses": [
            {
                "mechanism": "Split unlocks affine",
                "change": "Slice -> Split",
                "supporting_evidence": "Analyzer and detail profile identify the routed tail.",
                "expected_delta": "Remove routed Mul without added Transpose.",
                "status": "confirmed",
                "falsifier": "Transpose increases",
            }
        ],
        "experiments": [
            {
                "id": "c1",
                "parent": "baseline",
                "status": "confirmed",
                "change": "route + affine + Exp",
                "correctness": "pass",
                "p50_ms": 8.0,
                "gain_pct": 20.0,
                "ci_low_pct": 18.0,
                "ci_high_pct": 22.0,
                "pairs": 4,
                "graph_delta": {"Slice": -8, "Split": 2, "Mul": -7},
                "trace_delta": {
                    "accelerator_us": -5400,
                    "transpose": 0,
                    "partitions": 0,
                },
                "notes": "paired CI above zero",
                "details": {
                    "commands": ["winml optimize ...", "winml perf ..."],
                    "artifacts": ["experiment.json", "trace.json"],
                },
            }
        ],
        "capability_closure": {
            "reviewer": "Graph Scout",
            "leader_sha256": "b" * 64,
            "registry_evidence": "winml optimize --list-capabilities --verbose",
            "analyzer_evidence": "winml optimize --check-optim",
            "review_summary": "All registered residual opportunities are closed.",
            "coverage_verdict": "NO_MATERIAL_OMISSION",
            "probe_limit": 3,
            "rows": [
                {
                    "capability": "generic-capability",
                    "flag": "--enable-generic-capability",
                    "sources": ["registry", "residual topology"],
                    "residual_anchor": "generic producer/consumer neighborhood",
                    "eligibility": "matched: static attributes",
                    "safety": "established: public I/O preserved",
                    "analyzer": "not reported",
                    "rationale": "new leader exposed the neighborhood",
                    "probe": "winml optimize --enable-generic-capability ...",
                    "expected_delta": "remove residual provider work; no latency claim",
                    "status": "CLOSED_ALREADY_TESTED",
                    "closure_reason": "same leader hash and outcome recorded",
                }
            ],
        },
        "feature_gaps": [
            {
                "name": "nested affine",
                "status": "draft-pr",
                "draft_pr": "https://github.com/example/project/pull/123",
                "reviewer_verdict": "READY_FOR_CHECK_IN",
                "description": "generic CLI capability",
                "complexity_review": {
                    "source": "ponytail",
                    "plugin_version": "4.9.0",
                    "verdict": "Lean already. Ship.",
                    "net_removable_lines": 0,
                    "resolved": [],
                    "waived": [],
                },
            }
        ],
        "conclusion": {
            "stop_reason": "target confirmed",
            "remaining_opportunities": ["final FFHQ gate"],
            "reproduce": ["winml optimize ...", "winml perf ..."],
        },
        "artifacts": {
            "champion_onnx": "champion.onnx",
            "companions": [{"path": "champion_qnn_ctx.bin", "role": "QNN EP context"}],
            "winml_config": "winml_config.json",
            "manifest": "manifest.json",
        },
    }

    sections = [
        (report["baseline"], ("p90_ms", "p99_ms", "throughput_ips")),
        (
            report["evidence"]["execution"],
            (
                "accelerator_pct",
                "host_overhead_pct",
                "partition_count",
                "fallback_nodes",
                "transfers",
            ),
        ),
        (report["evidence"]["analyzer"], ("coverage", "optimizations")),
        (
            report["evidence"]["detail_profile"],
            ("hardware_time_us", "memory_time_us", "ddr_read_bytes", "ddr_write_bytes"),
        ),
    ]
    sections += [(row, ("nodes",)) for row in report["model"]["components"]]
    sections += [
        (row, ("hardware_time_us", "memory_time_us", "dram_bytes"))
        for row in report["baseline"]["hotspots"]
    ]
    for row, fields in sections:
        row["missing_reasons"] = {
            key: "Not collected in this fixture" for key in fields if row.get(key) in (None, "", [])
        }
    return report


def test_template_is_valid_and_report_renders_all_sections(
    report_module: ModuleType,
    tmp_path: Path,
) -> None:
    report_module.validate_report(report_module.report_template())
    output = tmp_path / "report.html"

    report_module.render_report(_report(), output)

    text = output.read_text(encoding="utf-8")
    for heading in (
        "Overview",
        "Baseline Diagnosis",
        "Execution Evidence",
        "Model Structure",
        "Ranked Hypotheses",
        "Experiment Lineage",
        "Capability Closure",
        "Champion Delivery",
        "Feature Gaps",
        "Conclusion",
    ):
        assert heading in text
    assert "Model &lt;unsafe&gt; optimization" in text
    assert "Model <unsafe> optimization" not in text
    assert "Complexity review" in text
    assert "ponytail 4.9.0" in text
    assert "Lean already. Ship." in text
    assert "Experiment gain chart" in text
    assert "The routed output tail dominates accelerator time." in text
    assert "Analyzer and detail profile identify the routed tail." in text
    assert "generic-capability" in text
    assert "CLOSED_ALREADY_TESTED" in text
    assert "NO_MATERIAL_OMISSION" in text
    assert "<details" in text
    assert 'href="champion.onnx"' in text
    assert 'href="winml_config.json"' in text
    assert 'href="manifest.json"' in text
    assert "winml-theme" in text
    assert 'aria-label="Toggle dark mode"' in text
    assert "overflow-x:hidden" in text
    assert ".table-wrap{overflow:auto" in text
    assert "nav{position:sticky" in text


def test_closure_sources_are_html_escaped(
    report_module: ModuleType,
    tmp_path: Path,
) -> None:
    report = _report()
    report["capability_closure"]["rows"][0]["sources"] = [
        'registry<script>alert("unsafe")</script>'
    ]
    output = tmp_path / "report.html"

    report_module.render_report(report, output)

    text = output.read_text(encoding="utf-8")
    assert "registry&lt;script&gt;alert(&quot;unsafe&quot;)&lt;/script&gt;" in text
    assert '<script>alert("unsafe")</script>' not in text


def test_intermediate_report_without_closure_remains_renderable(
    report_module: ModuleType,
    tmp_path: Path,
) -> None:
    report = _report()
    del report["capability_closure"]
    output = tmp_path / "report.html"

    report_module.validate_report(report)
    report_module.render_report(report, output)

    text = output.read_text(encoding="utf-8")
    assert "Capability Closure" in text
    assert "Not reviewed yet" in text


def test_final_report_without_closure_is_rejected(report_module: ModuleType) -> None:
    report = _report()
    del report["capability_closure"]

    with pytest.raises(report_module.ReportError, match="capability_closure"):
        report_module.validate_report(report, final=True)


def test_empty_template_is_not_a_valid_final_report(report_module: ModuleType) -> None:
    with pytest.raises(report_module.ReportError, match="final report"):
        report_module.validate_report(report_module.report_template(), final=True)


def test_complete_final_report_passes_strict_validation(
    report_module: ModuleType,
) -> None:
    report_module.validate_report(_report(), final=True)


def test_final_report_accepts_structured_provisional_quality_gate(
    report_module: ModuleType,
) -> None:
    report = _report()
    report["leader"]["status"] = "confirmed-performance-provisional-quality"
    report["leader"]["quality_gate"] = {
        "task_evaluator": "unavailable",
        "tensor_validation": "pass",
        "evidence_gap": "Task-level evaluator was unavailable.",
    }

    report_module.validate_report(report, final=True)


@pytest.mark.parametrize(
    "quality_gate",
    [
        None,
        {},
        {
            "task_evaluator": "available",
            "tensor_validation": "pass",
            "evidence_gap": "Task-level evaluator was unavailable.",
        },
        {
            "task_evaluator": "unavailable",
            "tensor_validation": "failed",
            "evidence_gap": "Task-level evaluator was unavailable.",
        },
        {
            "task_evaluator": "unavailable",
            "tensor_validation": "pass",
            "evidence_gap": "",
        },
    ],
)
def test_final_report_rejects_invalid_provisional_quality_gate(
    report_module: ModuleType,
    quality_gate: object,
) -> None:
    report = _report()
    report["leader"]["status"] = "confirmed-performance-provisional-quality"
    if quality_gate is not None:
        report["leader"]["quality_gate"] = quality_gate

    with pytest.raises(report_module.ReportError, match="quality_gate"):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize("status", ["PROBE_REQUIRED", "DEFERRED_BUDGET"])
def test_final_report_rejects_omission_free_verdict_with_unresolved_closure(
    report_module: ModuleType,
    status: str,
) -> None:
    report = _report()
    report["capability_closure"]["rows"][0]["status"] = status

    with pytest.raises(report_module.ReportError, match="capability_closure"):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize(
    ("status", "verdict"),
    [
        ("PROBE_REQUIRED", "MATERIAL_OMISSION_FOUND"),
        ("DEFERRED_BUDGET", "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_intermediate_report_accepts_consistent_unresolved_closure(
    report_module: ModuleType,
    status: str,
    verdict: str,
) -> None:
    report = _report()
    report["capability_closure"]["rows"][0]["status"] = status
    report["capability_closure"]["coverage_verdict"] = verdict

    report_module.validate_report(report)


def test_intermediate_report_accepts_probe_and_deferred_with_insufficient_verdict(
    report_module: ModuleType,
) -> None:
    report = _report()
    probe = report["capability_closure"]["rows"][0]
    probe["status"] = "PROBE_REQUIRED"
    deferred = dict(probe)
    deferred["capability"] = "deferred-capability"
    deferred["flag"] = "--enable-deferred-capability"
    deferred["status"] = "DEFERRED_BUDGET"
    report["capability_closure"]["rows"].append(deferred)
    report["capability_closure"]["coverage_verdict"] = "INSUFFICIENT_EVIDENCE"

    report_module.validate_report(report)


def test_deferred_budget_takes_verdict_precedence_over_open_probes(
    report_module: ModuleType,
) -> None:
    report = _report()
    probe = report["capability_closure"]["rows"][0]
    probe["status"] = "PROBE_REQUIRED"
    deferred = dict(probe)
    deferred["capability"] = "deferred-capability"
    deferred["flag"] = "--enable-deferred-capability"
    deferred["status"] = "DEFERRED_BUDGET"
    report["capability_closure"]["rows"].append(deferred)
    report["capability_closure"]["coverage_verdict"] = "MATERIAL_OMISSION_FOUND"

    with pytest.raises(report_module.ReportError, match="INSUFFICIENT_EVIDENCE"):
        report_module.validate_report(report)


def test_intermediate_report_enforces_active_probe_limit(
    report_module: ModuleType,
) -> None:
    report = _report()
    template = report["capability_closure"]["rows"][0]
    report["capability_closure"]["rows"] = []
    for index in range(4):
        row = dict(template)
        row["capability"] = f"capability-{index}"
        row["flag"] = f"--enable-capability-{index}"
        row["status"] = "PROBE_REQUIRED"
        report["capability_closure"]["rows"].append(row)
    report["capability_closure"]["coverage_verdict"] = "MATERIAL_OMISSION_FOUND"

    with pytest.raises(report_module.ReportError, match="probe_limit"):
        report_module.validate_report(report)


@pytest.mark.parametrize("sources", ["registry", ["registry", ""], ["registry", 1]])
def test_closure_sources_must_be_nonempty_string_list(
    report_module: ModuleType,
    sources: Any,
) -> None:
    report = _report()
    report["capability_closure"]["rows"][0]["sources"] = sources

    with pytest.raises(report_module.ReportError, match="sources"):
        report_module.validate_report(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer", ""),
        ("leader_sha256", ""),
        ("registry_evidence", ""),
        ("analyzer_evidence", ""),
        ("review_summary", ""),
        ("coverage_verdict", ""),
        ("probe_limit", 4),
        ("rows", "not-a-list"),
    ],
)
def test_final_report_requires_capability_closure_evidence(
    report_module: ModuleType,
    field: str,
    value: Any,
) -> None:
    report = _report()
    report["capability_closure"][field] = value

    with pytest.raises(report_module.ReportError, match="capability_closure"):
        report_module.validate_report(report, final=True)


def test_final_report_allows_closed_review_without_candidate_rows(
    report_module: ModuleType,
) -> None:
    report = _report()
    report["capability_closure"]["probe_limit"] = 0
    report["capability_closure"]["rows"] = []
    report["capability_closure"]["review_summary"] = (
        "No structurally plausible registered capability remained."
    )

    report_module.validate_report(report, final=True)


@pytest.mark.parametrize(
    "field",
    [
        "capability",
        "flag",
        "sources",
        "residual_anchor",
        "eligibility",
        "safety",
        "analyzer",
        "rationale",
        "probe",
        "expected_delta",
        "status",
        "closure_reason",
    ],
)
def test_final_report_requires_complete_generic_closure_rows(
    report_module: ModuleType,
    field: str,
) -> None:
    report = _report()
    del report["capability_closure"]["rows"][0][field]

    with pytest.raises(report_module.ReportError, match=field):
        report_module.validate_report(report, final=True)


def test_draft_pr_requires_checkin_reviewer_verdict(report_module: ModuleType) -> None:
    report = _report()
    report["feature_gaps"][0]["reviewer_verdict"] = ""

    with pytest.raises(report_module.ReportError, match="reviewer_verdict"):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "path", "pending"),
        ("model", "sha256", "pending"),
        ("model", "opset", "pending"),
        ("model", "node_count", 0),
        ("model", "inputs", []),
        ("model", "outputs", []),
        ("model", "op_counts", {}),
        ("model", "components", []),
        ("target", "ep", "pending"),
        ("target", "device", "pending"),
        ("leader", "id", ""),
        ("leader", "status", "pending"),
        ("leader", "model_path", "pending"),
        ("leader", "p50_ms", None),
        ("leader", "gain_pct", None),
        ("leader", "quality", "pending"),
    ],
)
def test_final_report_requires_complete_identity_and_leader(
    report_module: ModuleType,
    section: str,
    field: str,
    value: Any,
) -> None:
    report = _report()
    report[section][field] = value

    with pytest.raises(report_module.ReportError, match=rf"{section}\.{field}"):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize("missing_key", ["transpose", "partitions"])
def test_final_report_requires_layout_and_partition_evidence(
    report_module: ModuleType,
    missing_key: str,
) -> None:
    report = _report()
    del report["experiments"][0]["trace_delta"][missing_key]

    with pytest.raises(report_module.ReportError, match=missing_key):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize("missing_field", ["draft_pr", "reviewer_verdict"])
def test_pr_related_feature_gap_requires_url_and_review(
    report_module: ModuleType,
    missing_field: str,
) -> None:
    report = _report()
    report["feature_gaps"][0][missing_field] = ""

    with pytest.raises(report_module.ReportError, match=missing_field):
        report_module.validate_report(report, final=True)


@pytest.mark.parametrize(
    "missing_field",
    [
        "source",
        "plugin_version",
        "verdict",
        "net_removable_lines",
        "resolved",
        "waived",
    ],
)
def test_pr_related_feature_gap_requires_complexity_review(
    report_module: ModuleType,
    missing_field: str,
) -> None:
    report = _report()
    del report["feature_gaps"][0]["complexity_review"][missing_field]

    with pytest.raises(report_module.ReportError, match=missing_field):
        report_module.validate_report(report, final=True)


def test_missing_required_section_is_rejected(report_module: ModuleType) -> None:
    report = _report()
    del report["experiments"]

    with pytest.raises(report_module.ReportError, match="experiments"):
        report_module.validate_report(report)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evidence", "diagnosis", ""),
        ("evidence", "execution", {}),
        ("evidence", "analyzer", {}),
        ("evidence", "detail_profile", {}),
        ("evidence", "ranked_levers", []),
        ("artifacts", "champion_onnx", ""),
        ("artifacts", "winml_config", ""),
        ("artifacts", "manifest", ""),
    ],
)
def test_final_report_requires_diagnosis_and_delivery_artifacts(
    report_module: ModuleType,
    section: str,
    field: str,
    value: Any,
) -> None:
    report = _report()
    report[section][field] = value

    with pytest.raises(report_module.ReportError, match=rf"{section}\.{field}"):
        report_module.validate_report(report, final=True)


def test_final_report_rejects_unexplained_display_gap(report_module):
    report = _report()
    report["evidence"]["execution"].pop("partition_count")
    with pytest.raises(report_module.ReportError, match="partition_count"):
        report_module.validate_report(report, final=True)


def test_cycles_and_missing_reason_render(report_module, tmp_path):
    report = _report()
    report["baseline"]["hotspots"][0]["cycles"] = 123456
    report["baseline"]["p90_ms"] = None
    report["baseline"]["missing_reasons"] = {"p90_ms": "Not aggregated: session percentiles only"}
    out = tmp_path / "report.html"
    report_module.render_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert "123456" in text
    assert "Not aggregated: session percentiles only" in text


def test_missing_metric_explanation_is_small(report_module):
    rendered = report_module._metric("p90", "No raw timing samples recorded", " ms")
    assert "<strong>N/A</strong>" in rendered
    assert "<small>No raw timing samples recorded</small>" in rendered
