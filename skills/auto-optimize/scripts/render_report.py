#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validate report facts and render a self-contained optimization report."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_SECTIONS = {
    "schema_version",
    "title",
    "updated_at",
    "model",
    "target",
    "baseline",
    "evidence",
    "leader",
    "hypotheses",
    "experiments",
    "feature_gaps",
    "conclusion",
    "artifacts",
}


class ReportError(ValueError):
    """Raised when report facts do not satisfy the renderer contract."""


def report_template() -> dict[str, Any]:
    """Return the complete report v2 fact template."""
    return {
        "schema_version": 2,
        "title": "ONNX optimization report",
        "updated_at": "not-started",
        "model": {
            "path": "pending",
            "sha256": "pending",
            "opset": "pending",
            "node_count": 0,
            "inputs": [],
            "outputs": [],
            "op_counts": {},
            "components": [],
        },
        "target": {
            "ep": "pending",
            "device": "pending",
            "latency_target_ms": None,
            "provider_options": {},
        },
        "baseline": {
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "throughput_ips": None,
            "protocol": "pending",
            "hotspots": [],
            "trace": {},
        },
        "evidence": {
            "diagnosis": "pending",
            "execution": {},
            "analyzer": {},
            "detail_profile": {},
            "ranked_levers": [],
            "gaps": [],
        },
        "leader": {
            "id": "baseline",
            "status": "pending",
            "model_path": "pending",
            "p50_ms": None,
            "gain_pct": 0.0,
            "ci_low_pct": None,
            "ci_high_pct": None,
            "correctness": "pending",
            "quality": "pending",
        },
        "hypotheses": [],
        "experiments": [],
        "capability_closure": {
            "reviewer": "pending",
            "leader_sha256": "pending",
            "registry_evidence": "pending",
            "analyzer_evidence": "pending",
            "review_summary": "pending",
            "coverage_verdict": "INSUFFICIENT_EVIDENCE",
            "probe_limit": 3,
            "rows": [],
        },
        "feature_gaps": [],
        "conclusion": {
            "stop_reason": "running",
            "remaining_opportunities": [],
            "reproduce": [],
        },
        "artifacts": {
            "champion_onnx": "pending",
            "companions": [],
            "winml_config": "pending",
            "manifest": "pending",
        },
    }


def _missing(value: Any) -> bool:
    return value in (None, "", "pending")


def _relative_artifact(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and ":" not in value


def _unreviewed_closure() -> dict[str, Any]:
    return {
        "reviewer": "Not reviewed yet",
        "leader_sha256": "pending",
        "registry_evidence": "pending",
        "analyzer_evidence": "pending",
        "review_summary": "Not reviewed yet",
        "coverage_verdict": "INSUFFICIENT_EVIDENCE",
        "probe_limit": 0,
        "rows": [],
    }


def validate_quality_gate(leader: Any) -> None:
    """Require explicit evidence when task-level quality remains provisional."""
    if not isinstance(leader, dict):
        raise ReportError("leader must be an object")
    if str(leader.get("status", "")).lower() != ("confirmed-performance-provisional-quality"):
        return
    gate = leader.get("quality_gate")
    if not isinstance(gate, dict):
        raise ReportError("leader.quality_gate must be an object")
    if gate.get("task_evaluator") != "unavailable":
        raise ReportError("leader.quality_gate.task_evaluator must be unavailable")
    if gate.get("tensor_validation") != "pass":
        raise ReportError("leader.quality_gate.tensor_validation must be pass")
    if not isinstance(gate.get("evidence_gap"), str) or not gate["evidence_gap"]:
        raise ReportError("leader.quality_gate.evidence_gap is required")


def _display_sections(report):
    yield "baseline", report["baseline"], ("p90_ms", "p99_ms", "throughput_ips")
    yield (
        "execution",
        report["evidence"].get("execution", {}),
        ("accelerator_pct", "host_overhead_pct", "partition_count", "fallback_nodes", "transfers"),
    )
    yield "analyzer", report["evidence"].get("analyzer", {}), ("coverage", "optimizations")
    yield (
        "detail_profile",
        report["evidence"].get("detail_profile", {}),
        ("hardware_time_us", "memory_time_us", "ddr_read_bytes", "ddr_write_bytes"),
    )
    for i, row in enumerate(report["model"].get("components", [])):
        yield f"components[{i}]", row, ("nodes",)
    for i, row in enumerate(report["baseline"].get("hotspots", [])):
        yield f"hotspots[{i}]", row, ("hardware_time_us", "memory_time_us", "dram_bytes")


def _display_value(row, key):
    value = row.get(key)
    if value is None or value == [] or value == "":
        return row.get("missing_reasons", {}).get(key, "Not recorded")
    return value


def validate_report(report: Any, *, final: bool = False) -> dict[str, Any]:
    """Validate the stable report v2 fact contract."""
    if not isinstance(report, dict):
        raise ReportError("report must be a JSON object")
    missing = sorted(REQUIRED_SECTIONS - report.keys())
    if missing:
        raise ReportError(f"missing report sections: {', '.join(missing)}")
    if report["schema_version"] != 2:
        raise ReportError("schema_version must be 2")
    for field in (
        "model",
        "target",
        "baseline",
        "evidence",
        "leader",
        "conclusion",
        "artifacts",
    ):
        if not isinstance(report[field], dict):
            raise ReportError(f"{field} must be an object")
    for field in ("hypotheses", "experiments", "feature_gaps"):
        if not isinstance(report[field], list):
            raise ReportError(f"{field} must be a list")
    if final and "capability_closure" not in report:
        raise ReportError("final report incomplete: capability_closure")
    closure = report.get("capability_closure", _unreviewed_closure())
    if not isinstance(closure, dict):
        raise ReportError("capability_closure must be an object")
    probe_limit = closure.get("probe_limit")
    if not isinstance(probe_limit, int) or not 0 <= probe_limit <= 3:
        raise ReportError("capability_closure.probe_limit")
    rows = closure.get("rows")
    if not isinstance(rows, list):
        raise ReportError("capability_closure.rows must be a list")

    valid_statuses = {
        "PROBE_REQUIRED",
        "CLOSED_INELIGIBLE",
        "CLOSED_ALREADY_TESTED",
        "CLOSED_REGISTRY_ABSENT",
        "DEFERRED_BUDGET",
    }
    required_row_fields = (
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
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReportError(f"capability_closure.rows[{index}] must be an object")
        for field in required_row_fields:
            if field not in row or row[field] in (None, "", []):
                raise ReportError(f"capability_closure.rows[{index}].{field}")
        sources = row["sources"]
        if not isinstance(sources, list) or not all(
            isinstance(source, str) and source for source in sources
        ):
            raise ReportError(f"capability_closure.rows[{index}].sources")
        if row["status"] not in valid_statuses:
            raise ReportError(f"capability_closure.rows[{index}].status")

    statuses = {row["status"] for row in rows}
    active_probes = sum(row["status"] == "PROBE_REQUIRED" for row in rows)
    if active_probes > probe_limit:
        raise ReportError("capability_closure PROBE_REQUIRED exceeds probe_limit")
    verdict = closure.get("coverage_verdict")
    if "DEFERRED_BUDGET" in statuses and verdict != "INSUFFICIENT_EVIDENCE":
        raise ReportError("capability_closure DEFERRED_BUDGET requires INSUFFICIENT_EVIDENCE")
    if (
        "DEFERRED_BUDGET" not in statuses
        and "PROBE_REQUIRED" in statuses
        and verdict != "MATERIAL_OMISSION_FOUND"
    ):
        raise ReportError("capability_closure PROBE_REQUIRED requires MATERIAL_OMISSION_FOUND")
    if not final:
        return report

    errors: list[str] = []
    for section, row, fields in _display_sections(report):
        reasons = row.get("missing_reasons", {})
        if not isinstance(reasons, dict):
            errors.append(f"{section}.missing_reasons must be an object")
            continue
        errors.extend(
            f"{section}.{field}: value or missing reason required"
            for field in fields
            if row.get(field) in (None, "", [])
            and not (isinstance(reasons.get(field), str) and reasons[field].strip())
        )

    for section in ("baseline", "leader"):
        metrics = report[section]
        for field in (
            "p50_ms",
            "p90_ms",
            "p99_ms",
            "throughput_ips",
            "gain_pct",
            "ci_low_pct",
            "ci_high_pct",
        ):
            value = metrics.get(field)
            if value is None and field != "p50_ms":
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                errors.append(f"{section}.{field} must be finite numeric evidence")
            elif (field.endswith("_ms") or field == "throughput_ips") and value <= 0:
                errors.append(f"{section}.{field} must be positive")
        low, high = metrics.get("ci_low_pct"), metrics.get("ci_high_pct")
        if (low is None) != (high is None):
            errors.append(f"{section} confidence bounds must be supplied together")
        elif type(low) in (int, float) and type(high) in (int, float) and low > high:
            errors.append(f"{section} confidence bounds are reversed")
    model = report["model"]
    target = report["target"]
    baseline = report["baseline"]
    evidence = report["evidence"]
    leader = report["leader"]
    validate_quality_gate(leader)
    conclusion = report["conclusion"]
    artifacts = report["artifacts"]
    closure = report.get("capability_closure", _unreviewed_closure())

    errors.extend(
        [f"model.{field}" for field in ("path", "sha256", "opset") if _missing(model.get(field))]
    )
    if not isinstance(model.get("node_count"), int) or model["node_count"] <= 0:
        errors.append("model.node_count")
    errors.extend(
        [
            f"model.{field}"
            for field in ("inputs", "outputs", "op_counts", "components")
            if not model.get(field)
        ]
    )
    errors.extend([f"target.{field}" for field in ("ep", "device") if _missing(target.get(field))])
    if not isinstance(baseline.get("p50_ms"), (int, float)):
        errors.append("baseline.p50_ms")
    if not baseline.get("hotspots") or not baseline.get("trace"):
        errors.append("baseline hotspots/trace")

    if _missing(evidence.get("diagnosis")):
        errors.append("evidence.diagnosis")
    errors.extend(
        [
            f"evidence.{field}"
            for field in ("execution", "analyzer", "detail_profile")
            if not isinstance(evidence.get(field), dict) or not evidence[field]
        ]
    )
    if not evidence.get("ranked_levers"):
        errors.append("evidence.ranked_levers")
    if not report["hypotheses"]:
        errors.append("hypotheses")
    if not report["experiments"]:
        errors.append("experiments")

    errors.extend(
        [
            f"capability_closure.{field}"
            for field in (
                "reviewer",
                "leader_sha256",
                "registry_evidence",
                "analyzer_evidence",
                "review_summary",
                "coverage_verdict",
            )
            if _missing(closure.get(field))
        ]
    )
    unresolved = {
        row["status"]
        for row in closure.get("rows", [])
        if row["status"] in {"PROBE_REQUIRED", "DEFERRED_BUDGET"}
    }
    if unresolved or closure.get("coverage_verdict") != "NO_MATERIAL_OMISSION":
        errors.append("capability_closure unresolved or omission verdict")

    errors.extend(
        [
            f"leader.{field}"
            for field in ("id", "status", "model_path", "correctness", "quality")
            if _missing(leader.get(field))
        ]
    )
    errors.extend(
        [
            f"leader.{field}"
            for field in ("p50_ms", "gain_pct")
            if not isinstance(leader.get(field), (int, float))
        ]
    )
    for experiment in report["experiments"]:
        identifier = experiment.get("id")
        if not experiment.get("graph_delta") or not experiment.get("trace_delta"):
            errors.append(f"experiment {identifier} graph/trace delta")
            continue
        errors.extend(
            [
                f"experiment {identifier} trace_delta.{field}"
                for field in ("transpose", "partitions")
                if field not in experiment["trace_delta"]
            ]
        )

    for feature in report["feature_gaps"]:
        status = str(feature.get("status", "")).lower()
        if feature.get("draft_pr") or "pr" in status:
            errors.extend(
                [
                    f"feature gap {feature.get('name')} {field}"
                    for field in ("draft_pr", "reviewer_verdict")
                    if not feature.get(field)
                ]
            )
            complexity = feature.get("complexity_review")
            if not isinstance(complexity, dict):
                errors.append(f"feature gap {feature.get('name')} complexity_review")
            else:
                errors.extend(
                    [
                        f"feature gap {feature.get('name')} complexity_review.{field}"
                        for field in (
                            "source",
                            "plugin_version",
                            "verdict",
                            "net_removable_lines",
                            "resolved",
                            "waived",
                        )
                        if field not in complexity or complexity[field] in (None, "")
                    ]
                )

    if _missing(conclusion.get("stop_reason")) or conclusion["stop_reason"] == "running":
        errors.append("conclusion.stop_reason")
    if not conclusion.get("reproduce"):
        errors.append("conclusion.reproduce")
    errors.extend(
        [
            f"artifacts.{field}"
            for field in ("champion_onnx", "winml_config", "manifest")
            if not _relative_artifact(artifacts.get(field))
        ]
    )
    companions = artifacts.get("companions")
    if not isinstance(companions, list):
        errors.append("artifacts.companions")
    else:
        for companion in companions:
            if not isinstance(companion, dict) or not _relative_artifact(companion.get("path")):
                errors.append("artifacts.companions.path")
                break
    if errors:
        raise ReportError(f"final report incomplete: {', '.join(errors)}")
    return report


def _escape(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        value = f"{value:.3f}"
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _href(value: Any) -> str:
    return _escape(value) if _relative_artifact(value) else "#"


def _rows(values: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not values:
        return f'<tr><td colspan="{len(columns)}" class="empty">No evidence yet</td></tr>'
    rendered = []
    for value in values:
        cells = "".join(f"<td>{_escape(_display_value(value, key))}</td>" for key, _ in columns)
        rendered.append(f"<tr>{cells}</tr>")
    return "".join(rendered)


def _evidence_table(row, key, columns):
    return _table(row[key], columns) if row.get(key) else _escape(_display_value(row, key))


def _table(values: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{_escape(label)}</th>" for _, label in columns)
    return (
        f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>'
        f"{_rows(values, columns)}</tbody></table></div>"
    )


def _io_rows(values: list[dict[str, Any]]) -> str:
    return _table(values, [("name", "Name"), ("shape", "Shape"), ("dtype", "Dtype")])


def _status_class(value: Any) -> str:
    lowered = str(value).lower()
    if any(token in lowered for token in ("pass", "confirm", "ready", "keep", "high")):
        return "pass"
    if any(token in lowered for token in ("fail", "reject", "regress", "block")):
        return "fail"
    return "pending"


def _metric(label: str, value: Any, suffix: str = "") -> str:
    if isinstance(value, str) and (bool(suffix) or len(value) > 40):
        content = f"<strong>N/A</strong><small>{_escape(value)}</small>"
    elif value is None:
        content = "<strong>N/A</strong><small>Not recorded</small>"
    else:
        unit = suffix if isinstance(value, (int, float)) else ""
        content = f"<strong>{_escape(value)}{unit}</strong>"
    return (
        "<div class="
        + chr(34)
        + "metric"
        + chr(34)
        + "><span>"
        + _escape(label)
        + "</span>"
        + content
        + "</div>"
    )


def _hypothesis_rows(items: list[dict[str, Any]]) -> str:
    rows = [
        (
            "<tr>"
            f'<td><span class="status {_status_class(item.get("status"))}">'
            f"{_escape(item.get('status'))}</span></td>"
            f"<td><strong>{_escape(item.get('mechanism'))}</strong>"
            f'<div class="muted">{_escape(item.get("change"))}</div></td>'
            f"<td>{_escape(item.get('supporting_evidence'))}</td>"
            f"<td>{_escape(item.get('expected_delta'))}</td>"
            f"<td>{_escape(item.get('falsifier'))}</td></tr>"
        )
        for item in items
    ]
    return "".join(rows) or '<tr><td colspan="5" class="empty">No hypotheses yet</td></tr>'


def _experiment_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        ci = f"{_escape(item.get('ci_low_pct'))}% to {_escape(item.get('ci_high_pct'))}%"
        rows.append(
            f'<tr data-experiment="{_escape(item.get("id"))}">'
            f"<td><strong>{_escape(item.get('id'))}</strong>"
            f'<div class="muted">from {_escape(item.get("parent"))}</div></td>'
            f'<td><span class="status {_status_class(item.get("status"))}">'
            f"{_escape(item.get('status'))}</span></td>"
            f"<td>{_escape(item.get('change'))}</td><td>{_escape(item.get('p50_ms'))} ms</td>"
            f'<td>{_escape(item.get("gain_pct"))}%<div class="muted">CI {ci}</div></td>'
            f"<td>{_escape(item.get('correctness'))}</td>"
            "<td><details><summary>Evidence</summary><dl>"
            f"<dt>Graph delta</dt><dd><code>{_json(item.get('graph_delta', {}))}</code></dd>"
            f"<dt>Trace delta</dt><dd><code>{_json(item.get('trace_delta', {}))}</code></dd>"
            f"<dt>Notes</dt><dd>{_escape(item.get('notes'))}</dd>"
            "<dt>Artifacts and commands</dt><dd><code>"
            f"{_json(item.get('details', {}))}</code></dd>"
            "</dl></details></td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="7" class="empty">No experiments yet</td></tr>'


def _feature_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        complexity = item.get("complexity_review") or {}
        summary = (
            f"{complexity.get('source', '-')} {complexity.get('plugin_version', '-')} | "
            f"{complexity.get('verdict', '-')} | net "
            f"{complexity.get('net_removable_lines', '-')} lines"
        )
        rows.append(
            "<tr>"
            f"<td><strong>{_escape(item.get('name'))}</strong>"
            f'<div class="muted">{_escape(item.get("description"))}</div></td>'
            f'<td><span class="status {_status_class(item.get("status"))}">'
            f"{_escape(item.get('status'))}</span></td>"
            f"<td>{_escape(item.get('draft_pr'))}</td>"
            f"<td>{_escape(item.get('reviewer_verdict'))}</td>"
            "<td><details><summary>Complexity review</summary>"
            f"<p>{_escape(summary)}</p><code>{_json(complexity)}</code>"
            "</details></td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="5" class="empty">No feature gap</td></tr>'


def _closure_rows(items: list[dict[str, Any]]) -> str:
    rows = [
        (
            "<tr>"
            f"<td><strong>{_escape(item.get('capability'))}</strong>"
            f'<div class="muted"><code>{_escape(item.get("flag"))}</code></div></td>'
            f'<td><span class="status {_status_class(item.get("status"))}">'
            f"{_escape(item.get('status'))}</span>"
            f'<div class="muted">{_escape(item.get("closure_reason"))}</div></td>'
            f"<td>{_escape(item.get('residual_anchor'))}"
            '<div class="muted">Sources: '
            f"{_escape(', '.join(str(source) for source in item.get('sources', [])))}</div></td>"
            f"<td>{_escape(item.get('eligibility'))}"
            f'<div class="muted">Safety: {_escape(item.get("safety"))}</div></td>'
            f"<td>{_escape(item.get('analyzer'))}"
            f'<div class="muted">{_escape(item.get("rationale"))}</div></td>'
            "<td><details><summary>Probe and expected evidence</summary>"
            f"<p><code>{_escape(item.get('probe'))}</code></p>"
            f"<p>{_escape(item.get('expected_delta'))}</p></details></td>"
            "</tr>"
        )
        for item in items
    ]
    return "".join(rows) or '<tr><td colspan="6" class="empty">No closure candidate</td></tr>'


def _artifact_links(artifacts: dict[str, Any]) -> str:
    items = [
        ("Champion ONNX", artifacts.get("champion_onnx"), "Runnable confirmed leader"),
        (
            "WinML config",
            artifacts.get("winml_config"),
            "Resolved replay configuration",
        ),
        ("Manifest", artifacts.get("manifest"), "Hashes and dependencies"),
    ]
    items.extend(
        (item.get("role", "Companion"), item.get("path"), "Required by champion")
        for item in artifacts.get("companions", [])
        if isinstance(item, dict)
    )
    return "".join(
        (
            f'<a class="artifact" href="{_href(path)}"><span>{_escape(label)}</span>'
            f"<strong>{_escape(path)}</strong><small>{_escape(note)}</small></a>"
        )
        for label, path, note in items
    )


def _gain_chart(experiments: list[dict[str, Any]]) -> str:
    gains = [abs(float(item.get("gain_pct") or 0)) for item in experiments]
    maximum = max(gains, default=1.0) or 1.0
    rows = []
    for item in experiments:
        gain = float(item.get("gain_pct") or 0)
        width = min(100.0, abs(gain) / maximum * 100.0)
        css = "negative" if gain < 0 else "positive"
        rows.append(
            f'<div class="gain-row"><span>{_escape(item.get("id"))}</span>'
            f'<span class="bar-track"><i class="{css}" style="width:{width:.1f}%"></i>'
            f"</span><strong>{_escape(gain)}</strong></div>"
        )
    return "".join(rows) or '<div class="empty">No measured experiment</div>'


def render_report(report: dict[str, Any], output: Path) -> None:
    """Render one deterministic, self-contained report."""
    report = validate_report(report)
    model = report["model"]
    target = report["target"]
    baseline = report["baseline"]
    evidence = report["evidence"]
    execution = evidence.get("execution", {})
    analyzer = evidence.get("analyzer", {})
    detail = evidence.get("detail_profile", {})
    closure = report.get("capability_closure", _unreviewed_closure())
    leader = report["leader"]
    conclusion = report["conclusion"]
    artifacts = report["artifacts"]

    levers = (
        "".join(
            (
                f"<li><span>{_escape(item.get('rank'))}</span><div>"
                f"<strong>{_escape(item.get('lever'))}</strong>"
                f"<p>{_escape(item.get('evidence'))}</p></div>"
                f'<em class="status {_status_class(item.get("confidence"))}">'
                f"{_escape(item.get('confidence'))}</em></li>"
            )
            for item in evidence.get("ranked_levers", [])
        )
        or '<li class="empty">No ranked lever</li>'
    )
    gaps = (
        "".join(f"<li>{_escape(item)}</li>" for item in evidence.get("gaps", []))
        or "<li>None recorded</li>"
    )
    opportunities = (
        "".join(
            f"<li>{_escape(item)}</li>" for item in conclusion.get("remaining_opportunities", [])
        )
        or "<li>None recorded</li>"
    )
    commands = (
        "".join(
            f"<li><code>{_escape(item)}</code></li>" for item in conclusion.get("reproduce", [])
        )
        or "<li>No command recorded</li>"
    )
    coverage_columns = [("classification", "Classification"), ("count", "Count")]
    optimization_columns = [
        ("name", "Opportunity"),
        ("status", "Status"),
        ("instances", "Instances"),
    ]
    hotspot_columns = [
        ("cycles", "Cycles"),
        ("name", "Operation"),
        ("share_pct", "Share %"),
        ("hardware_time_us", "Hardware us"),
        ("memory_time_us", "Memory us"),
        ("dram_bytes", "DRAM bytes"),
    ]
    component_columns = [
        ("name", "Component"),
        ("nodes", "Nodes"),
        ("note", "Details"),
    ]

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">\
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(report["title"])}</title><style>
:root{{--bg:#faf9f8;--bg2:#f5f5f5;--panel:#fff;--border:#e0e0e0;--border2:#d1d1d1;--text:#242424;--dim:#616161;--accent:#0f6cbd;--accent2:#115ea3;--soft:#eff6fc;--pass:#0e700e;--passbg:#f1faf1;--warn:#835b00;--warnbg:#fff4ce;--fail:#b10e1c;--failbg:#fdf3f4;--radius:4px}}
body.dark{{--bg:#141414;--bg2:#1f1f1f;--panel:#292929;--border:#404040;--border2:#525252;--text:#f5f5f5;--dim:#adadad;--accent:#62abf5;--accent2:#8fc5ff;--soft:rgba(40,134,222,.16);--pass:#5ec75e;--passbg:rgba(94,199,94,.13);--warn:#e6c84a;--warnbg:rgba(230,200,74,.13);--fail:#f4868f;--failbg:rgba(244,134,143,.13)}}
*{{box-sizing:border-box}}\
html{{scroll-behavior:smooth;overflow-x:hidden}}\
body{{margin:0;overflow-x:hidden;background:var(--bg);color:var(--text);\
font:16px 'Segoe UI Variable','Segoe UI',system-ui,sans-serif;\
-webkit-font-smoothing:antialiased}}\
a{{color:var(--accent2)}}\
header{{background:var(--panel);border-bottom:1px solid var(--border);\
padding:18px 28px 16px}}\
.header-top{{display:flex;align-items:center;gap:12px}}\
h1{{font-size:28px;font-weight:600;letter-spacing:0;margin:0}}\
.subtitle,.muted,small{{color:var(--dim);font-size:13px}}\
.theme-toggle{{margin-left:auto;width:36px;height:36px;\
border:1px solid var(--border2);border-radius:var(--radius);\
background:var(--panel);color:var(--text);font-size:18px;cursor:pointer}}\
.theme-toggle .sun,body.dark .theme-toggle .moon{{display:none}}\
body.dark .theme-toggle .sun{{display:inline}}\
.metrics{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));\
gap:8px;margin-top:16px}}\
.metric{{min-height:72px;background:var(--bg2);border:1px solid var(--border);\
border-radius:var(--radius);padding:10px 12px}}\
.metric span{{display:block;color:var(--dim);font-size:13px}}\
.metric strong{{display:block;font-size:22px;font-weight:600;margin-top:5px;\
overflow-wrap:anywhere}}\
nav{{position:sticky;top:0;z-index:5;display:flex;gap:4px;overflow:auto;\
padding:8px 28px;background:var(--bg2);border-bottom:1px solid var(--border)}}\
nav a{{flex:0 0 auto;color:var(--text);text-decoration:none;font-size:13px;\
padding:6px 9px;border-radius:var(--radius)}}\
nav a:hover{{background:var(--soft);color:var(--accent2)}}\
main{{max-width:1440px;margin:0 auto}}\
section{{padding:24px 28px;border-bottom:1px solid var(--border);\
scroll-margin-top:52px}}\
section:nth-child(even){{background:var(--panel)}}\
h2{{font-size:20px;font-weight:600;letter-spacing:0;margin:0 0 14px}}\
h3{{font-size:15px;margin:20px 0 8px}}\
.diagnosis{{border-left:4px solid var(--accent);background:var(--soft);\
padding:14px 16px;margin:0 0 16px}}\
.diagnosis strong{{display:block;margin-bottom:5px}}\
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}\
.three-col{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}\
.table-wrap{{overflow:auto;border:1px solid var(--border);\
border-radius:var(--radius)}}\
table{{width:100%;border-collapse:collapse;font-size:13px}}\
th,td{{padding:9px 10px;border-bottom:1px solid var(--border);text-align:left;\
vertical-align:top}}\
th{{position:sticky;top:0;background:var(--bg2);color:var(--dim);font-weight:600}}\
tbody tr:hover{{background:var(--soft)}}\
code{{font-family:Consolas,'Cascadia Code',monospace;white-space:pre-wrap;\
overflow-wrap:anywhere}}\
.status{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;\
font-weight:700;text-transform:uppercase}}\
.status.pass{{color:var(--pass);background:var(--passbg)}}\
.status.pending{{color:var(--warn);background:var(--warnbg)}}\
.status.fail{{color:var(--fail);background:var(--failbg)}}\
.composition{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}}\
.composition div{{padding:12px;background:var(--bg2);\
border:1px solid var(--border);border-radius:var(--radius)}}\
.composition span{{display:block;color:var(--dim);font-size:12px}}\
.composition strong{{font-size:18px}}\
.lever-list{{list-style:none;padding:0;margin:0}}\
.lever-list li{{display:grid;grid-template-columns:28px 1fr auto;gap:10px;\
align-items:start;padding:10px 0;border-bottom:1px solid var(--border)}}\
.lever-list li>span{{display:grid;place-items:center;width:24px;height:24px;\
background:var(--accent);color:#fff;border-radius:50%;font-weight:700;\
font-size:12px}}\
.lever-list p{{margin:4px 0;color:var(--dim)}}\
.artifact-grid{{display:grid;\
grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}}\
.artifact{{display:flex;flex-direction:column;min-height:92px;padding:12px;\
border:1px solid var(--border);border-radius:var(--radius);\
background:var(--bg2);text-decoration:none;color:var(--text)}}\
.artifact:hover{{border-color:var(--accent)}}\
.artifact span,.artifact small{{color:var(--dim)}}\
.artifact strong{{margin:5px 0;overflow-wrap:anywhere;color:var(--accent2)}}\
.gain-chart{{border-left:1px solid var(--border2);padding:4px 0}}\
.gain-row{{display:grid;grid-template-columns:80px 1fr 72px;gap:8px;\
align-items:center;margin:8px 0}}\
.gain-row>strong{{text-align:right}}\
.bar-track{{display:block;height:12px;background:var(--bg2);\
border-radius:var(--radius);overflow:hidden}}\
.bar-track i{{display:block;height:100%;background:var(--accent)}}\
.bar-track i.negative{{background:var(--fail)}}\
details{{max-width:520px}}\
summary{{cursor:pointer;color:var(--accent2)}}\
dl{{margin:8px 0}}\
dt{{font-weight:600;margin-top:8px}}\
dd{{margin:3px 0}}\
.empty{{color:var(--dim);text-align:center}}\
footer{{padding:18px 28px;color:var(--dim);font-size:12px}}
@media(max-width:980px){{.metrics{{grid-template-columns:repeat(3,1fr)}}.two-col,.three-col{{grid-template-columns:1fr}}.composition{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:600px){{\
header,section,footer{{padding-left:16px;padding-right:16px}}\
nav{{padding-left:12px;padding-right:12px}}\
h1{{font-size:23px}}\
.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}\
.metric strong{{font-size:18px}}\
.composition{{grid-template-columns:1fr}}\
.gain-row{{grid-template-columns:54px minmax(90px,1fr) 58px}}}}
</style></head><body><script>try{{if(localStorage.getItem('winml-theme')==='dark')document.body.classList.add('dark')}}catch(e){{}}</script>
<header><div class="header-top"><div><h1>\
{_escape(report["title"])}\
</h1><div class="subtitle">\
{_escape(target.get("ep"))} / \
{_escape(target.get("device"))} | updated \
{_escape(report["updated_at"])}\
</div></div><button class="theme-toggle" type="button" onclick="toggleTheme()" \
title="Toggle dark mode" aria-label="Toggle dark mode">\
<span class="moon" aria-hidden="true">&#9790;</span>\
<span class="sun" aria-hidden="true">&#9728;</span></button></div>\
<div class="metrics">\
{_metric("Baseline p50", baseline.get("p50_ms"), " ms")}\
{_metric("Champion p50", leader.get("p50_ms"), " ms")}\
{_metric("Paired gain", leader.get("gain_pct"), "%")}\
{_metric("95% CI low", leader.get("ci_low_pct"), "%")}\
{_metric("95% CI high", leader.get("ci_high_pct"), "%")}\
{_metric("Correctness", leader.get("correctness"))}</div></header>
<nav aria-label="Report sections">\
<a href="#overview">Overview</a>\
<a href="#diagnosis">Diagnosis</a>\
<a href="#execution">Execution</a>\
<a href="#structure">Structure</a>\
<a href="#hypotheses">Hypotheses</a>\
<a href="#closure">Closure</a>\
<a href="#experiments">Experiments</a>\
<a href="#champion">Champion</a>\
<a href="#gaps">Feature gaps</a>\
<a href="#conclusion">Conclusion</a></nav><main>
<section id="overview"><h2>Overview</h2><div class="artifact-grid">\
{_artifact_links(artifacts)}</div><div class="two-col">\
<div><h3>Model</h3><p><strong>\
{_escape(model.get("path"))}\
</strong><br><span class="muted">SHA-256 \
{_escape(model.get("sha256"))}\
</span></p></div><div><h3>Target</h3><p>\
{_escape(target.get("ep"))} / \
{_escape(target.get("device"))} | target \
{_escape(target.get("latency_target_ms"))} ms<br><code>\
{_json(target.get("provider_options", {}))}\
</code></p></div></div></section>
<section id="diagnosis"><h2>Baseline Diagnosis</h2>\
<div class="diagnosis"><strong>Evidence-backed diagnosis</strong>\
{_escape(evidence.get("diagnosis"))}\
</div><div class="metrics">\
{_metric("p50", baseline.get("p50_ms"), " ms")}\
{_metric("p90", _display_value(baseline, "p90_ms"), " ms")}\
{_metric("p99", _display_value(baseline, "p99_ms"), " ms")}\
{_metric("Throughput", _display_value(baseline, "throughput_ips"), " inf/s")}\
</div><div class="two-col"><div><h3>Ranked levers</h3>\
<ol class="lever-list">{levers}</ol></div>\
<div><h3>Evidence gaps</h3><ul>{gaps}</ul><h3>Protocol</h3><p>\
{_escape(baseline.get("protocol"))}\
</p></div></div></section>
<section id="execution"><h2>Execution Evidence</h2><div class="composition">\
<div><span>Accelerator (%)</span><strong>\
{_escape(_display_value(execution, "accelerator_pct"))}</strong></div>\
<div><span>Host overhead (%)</span><strong>\
{_escape(_display_value(execution, "host_overhead_pct"))}</strong></div>\
<div><span>Partitions</span><strong>\
{_escape(_display_value(execution, "partition_count"))}</strong></div>\
<div><span>Fallback nodes</span><strong>\
{_escape(_display_value(execution, "fallback_nodes"))}</strong></div>\
<div><span>Transfers</span><strong>\
{_escape(_display_value(execution, "transfers"))}</strong></div></div>\
<div class="three-col"><div><h3>Analyzer coverage</h3>\
{_evidence_table(analyzer, "coverage", coverage_columns)}\
</div><div><h3>Optimization opportunities</h3>\
{_evidence_table(analyzer, "optimizations", optimization_columns)}\
</div><div><h3>Detail profile</h3><p>\
<span class="status {_status_class(detail.get("status"))}">\
{_escape(detail.get("status"))}</span></p><dl>\
<dt>Hardware time</dt><dd>\
{_escape(_display_value(detail, "hardware_time_us"))} us</dd>\
<dt>Memory time</dt><dd>\
{_escape(_display_value(detail, "memory_time_us"))} us</dd>\
<dt>DDR read / write</dt><dd>\
{_escape(_display_value(detail, "ddr_read_bytes"))} / \
{_escape(_display_value(detail, "ddr_write_bytes"))} bytes</dd>\
<dt>Artifacts</dt><dd>{_json(detail.get("artifacts", []))}</dd>\
</dl></div></div><h3>Hotspots</h3>\
{_table(baseline.get("hotspots", []), hotspot_columns)}</section>
<section id="structure"><h2>Model Structure</h2><div class="two-col">\
<div><h3>Inputs</h3>{_io_rows(model.get("inputs", []))}</div>\
<div><h3>Outputs</h3>{_io_rows(model.get("outputs", []))}</div></div>\
<div class="two-col"><div><h3>Components</h3>\
{_table(model.get("components", []), component_columns)}\
</div><div><h3>Operator counts</h3><code>\
{_json(model.get("op_counts", {}))}\
</code><p class="muted">Opset {_escape(model.get("opset"))} | \
{_escape(model.get("node_count"))} nodes</p></div></div></section>
<section id="hypotheses"><h2>Ranked Hypotheses</h2>\
<div class="table-wrap"><table><thead><tr><th>Status</th>\
<th>Mechanism / change</th><th>Supporting analyzer/profile evidence</th>\
<th>Expected graph/trace delta</th><th>Cheapest falsifier</th>\
</tr></thead><tbody>\
{_hypothesis_rows(report["hypotheses"])}\
</tbody></table></div></section>
<section id="closure"><h2>Capability Closure</h2><div class="metrics">\
{_metric("Reviewer", closure.get("reviewer"))}\
{_metric("Coverage verdict", closure.get("coverage_verdict"))}\
{_metric("Probe limit", closure.get("probe_limit"))}</div>\
<div class="diagnosis"><strong>LLM closure summary</strong>\
{_escape(closure.get("review_summary"))}</div>\
<p class="muted">Leader {_escape(closure.get("leader_sha256"))} | registry: \
{_escape(closure.get("registry_evidence"))} | analyzer: \
{_escape(closure.get("analyzer_evidence"))}</p>\
<div class="table-wrap"><table><thead><tr><th>Capability</th><th>Status</th>\
<th>Residual anchor</th><th>Eligibility / safety</th>\
<th>Analyzer / rationale</th><th>Probe</th></tr></thead><tbody>\
{_closure_rows(closure.get("rows", []))}\
</tbody></table></div></section>
<section id="experiments"><h2>Experiment Lineage</h2>\
<h3>Experiment gain chart</h3><div class="gain-chart">\
{_gain_chart(report["experiments"])}\
</div><h3>Complete experiment evidence</h3>\
<div class="table-wrap"><table><thead><tr><th>ID</th><th>Status</th><th>Change</th>\
<th>p50</th><th>Gain / CI</th><th>Correctness</th><th>Details</th>\
</tr></thead><tbody>{_experiment_rows(report["experiments"])}</tbody>\
</table></div></section>
<section id="champion"><h2>Champion Delivery</h2><div class="metrics">\
{_metric("Candidate", leader.get("id"))}\
{_metric("Status", leader.get("status"))}\
{_metric("p50", leader.get("p50_ms"), " ms")}\
{_metric("Gain", leader.get("gain_pct"), "%")}</div>\
<div class="two-col"><div><h3>Validation</h3><p><strong>Correctness:</strong> \
{_escape(leader.get("correctness"))}\
<br><strong>Quality:</strong> {_escape(leader.get("quality"))}</p></div>\
<div><h3>Resolved WinML delivery</h3><p><a href="\
{_href(artifacts.get("champion_onnx"))}\
">{_escape(artifacts.get("champion_onnx"))}</a><br><a href="\
{_href(artifacts.get("winml_config"))}\
">{_escape(artifacts.get("winml_config"))}</a><br><a href="\
{_href(artifacts.get("manifest"))}\
">{_escape(artifacts.get("manifest"))}</a></p></div></div></section>
<section id="gaps"><h2>Feature Gaps</h2><div class="table-wrap">\
<table><thead><tr><th>Feature</th><th>Status</th><th>Draft PR</th>\
<th>Reviewer</th><th>Review evidence</th></tr></thead><tbody>\
{_feature_rows(report["feature_gaps"])}\
</tbody></table></div></section>
<section id="conclusion"><h2>Conclusion</h2><p><strong>Stop reason:</strong> \
{_escape(conclusion.get("stop_reason"))}</p><div class="two-col">\
<div><h3>Remaining opportunities</h3><ul>{opportunities}</ul></div>\
<div><h3>Reproduction</h3><ul>{commands}</ul></div></div></section>
</main><footer>Generated deterministically from report.json. \
Exact run evidence remains local to this output bundle.</footer>\
<script>function toggleTheme(){{\
const dark=document.body.classList.toggle('dark');\
try{{localStorage.setItem('winml-theme',dark?'dark':'light')}}\
catch(e){{}}\
}}</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main(argv: list[str]) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    if args.write_template:
        args.input.write_text(json.dumps(report_template(), indent=2) + "\n", encoding="utf-8")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    validate_report(report, final=args.final)
    render_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
