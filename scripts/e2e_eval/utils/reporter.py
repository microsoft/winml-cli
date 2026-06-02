# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Result building, summary generation, and report output (JSON, Markdown, HTML).

Works with the unified eval_result.json format:
  result["perf"]     — perf phase facts (always present when perf ran)
  result["accuracy"] — accuracy phase facts (present when accuracy ran, else None)
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .classifier import classify_failure


if TYPE_CHECKING:
    from collections.abc import Callable

    from .registry import ModelEntry


# ---------------------------------------------------------------------------
# Result dict construction
# ---------------------------------------------------------------------------


def build_eval_result(
    entry: ModelEntry,
    perf_proc: dict | None,
    device: str,
    eval_types_run: list[str],
    accuracy_result: dict | None = None,
    ep: str | None = None,
    onnx_size_bytes: int | None = None,
    sanitize_fn: Callable[[str], str] | None = None,
) -> dict:
    """Build a unified eval_result dict (facts only, no derived fields).

    perf_proc is the raw subprocess result from run_model(), or None when
    eval_types_run is ["accuracy"] (accuracy-only mode, perf phase skipped).
    accuracy_result is the accuracy sub-section dict (or None if not run).
    ep is the explicit execution provider (e.g., "qnn", "dml"), or None when
    not specified (device-to-provider mapping was used).
    onnx_size_bytes is the combined size of the exported ONNX + .data files.
    sanitize_fn, when provided, is applied to stdout/stderr to remove noise.
    """
    perf_section: dict | None = None
    if perf_proc is not None:
        passed = perf_proc["exit_code"] == 0
        raw_stdout = perf_proc["stdout"]
        raw_stderr = perf_proc["stderr"]
        if sanitize_fn is not None:
            stdout = sanitize_fn(raw_stdout)
            stderr = sanitize_fn(raw_stderr)
        else:
            stdout = raw_stdout
            stderr = raw_stderr
        perf_section = {
            "passed": passed,
            "elapsed": perf_proc["elapsed"],
            "exit_code": perf_proc["exit_code"],
            "stdout_output": stdout,
            "stderr_output": stderr,
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
            "timeout": perf_proc["timeout"],
            "command": perf_proc["command"],
            "error": perf_proc.get("error_summary", ""),
        }

    result = {
        "model": entry.hf_id,
        "task": entry.task,
        "device": device,
        "model_type": entry.model_type,
        "group": entry.group,
        "priority": entry.priority,
        "eval_types_run": eval_types_run,
        "run_timestamp": (
            perf_proc.get("timestamp") if perf_proc else datetime.now(timezone.utc).isoformat()
        ),
        "perf": perf_section,
        "accuracy": accuracy_result,
    }
    # Optional fields: only include when explicitly provided by the user.
    if onnx_size_bytes is not None:
        result["onnx_size_bytes"] = onnx_size_bytes
    if ep is not None:
        result["ep"] = ep
    return result


# ---------------------------------------------------------------------------
# Perf failure classification (derived from perf sub-section facts)
# ---------------------------------------------------------------------------


def classify_result(result: dict) -> str | None:
    """Derive failure_classification from result["perf"] facts. Returns None if passed."""
    perf = result.get("perf")
    if perf is None or perf.get("passed"):
        return None
    if perf.get("timeout"):
        return "TIMEOUT"
    combined = perf.get("stdout_output", "") + perf.get("stderr_output", "")
    exit_code = perf.get("exit_code", -1)
    return classify_failure(combined, exit_code).value


def classify_results(results: list[dict]) -> None:
    """Add failure_classification to each result's perf sub-section in-place."""
    for r in results:
        perf = r.get("perf")
        if perf is not None:
            perf["failure_classification"] = classify_result(r)


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------


def write_result_json(result: dict, path: Path) -> None:
    """Write a single model eval_result.json (facts only, no derived fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def load_result_json(path: Path) -> dict:
    """Load a single model eval_result.json."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_results_from_dir(results_dir: Path) -> list[dict]:
    """Walk results_dir/models/*/eval_result.json and load all results.

    Skips malformed JSON files with a warning.
    """
    models_dir = results_dir / "models"
    if not models_dir.exists():
        return []
    results: list[dict] = []
    for result_file in sorted(models_dir.glob("*/eval_result.json")):
        try:
            results.append(load_result_json(result_file))
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  WARNING: skipping {result_file} ({exc})")
    return results


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(results: list[dict], total_elapsed: float) -> dict:
    """Generate combined summary report dict (perf + accuracy)."""
    # Perf stats
    perf_results = [r for r in results if r.get("perf") is not None]
    perf_passed = sum(1 for r in perf_results if r["perf"].get("passed"))
    perf_failed = len(perf_results) - perf_passed

    # Perf failure type distribution
    failure_counts: Counter[str] = Counter()
    for r in perf_results:
        if not r["perf"].get("passed"):
            fc = r["perf"].get("failure_classification") or "UNKNOWN"
            failure_counts[fc] += 1

    # By task (perf)
    task_stats: dict[str, dict] = {}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in perf_results:
        by_task[r["task"] or "(none)"].append(r)
    for task_name, task_results in sorted(by_task.items()):
        t_passed = sum(1 for r in task_results if r["perf"].get("passed"))
        task_stats[task_name] = {
            "total": len(task_results),
            "passed": t_passed,
            "failed": len(task_results) - t_passed,
        }

    summary: dict = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "perf_summary": {
            "passed": perf_passed,
            "failed": perf_failed,
            "total": len(perf_results),
            "elapsed": round(total_elapsed, 1),
        },
        "results": results,
        "by_perf_failure_type": dict(failure_counts.most_common()),
        "by_task": task_stats,
    }

    # Accuracy stats (only if any result has accuracy data)
    acc_results = [r for r in results if r.get("accuracy") is not None]
    if acc_results:
        from .accuracy import generate_accuracy_summary

        acc_summary = generate_accuracy_summary(results)
        summary["accuracy_summary"] = acc_summary

    return summary


def write_summary_json(summary: dict, path: Path) -> None:
    """Write summary report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def format_text_summary(results: list[dict]) -> str:
    """Generate text summary (perf-focused)."""
    lines = [
        "",
        "=" * 60,
        "  E2E EVALUATION SUMMARY",
        f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "=" * 60,
    ]
    perf_results = [r for r in results if r.get("perf") is not None]
    p = sum(1 for r in perf_results if r["perf"].get("passed"))
    f = len(perf_results) - p
    for r in perf_results:
        perf = r["perf"]
        tag = "PASS" if perf.get("passed") else "FAIL"
        label = f"{r['model']} / {r['task']}" if r.get("task") else r["model"]
        fc = perf.get("failure_classification", "")
        fc_tag = f"  [{fc}]" if fc else ""
        err = f"  ({perf.get('error', '')})" if perf.get("error") else ""
        elapsed = perf.get("elapsed", 0)
        lines.append(f"  [{tag}]  {'perf':<16} {label:<42} {elapsed:>6.1f}s{fc_tag}{err}")
    total_time = sum(r["perf"].get("elapsed", 0) for r in perf_results)
    lines += ["", f"  TOTAL: {p} passed, {f} failed ({total_time:.1f}s)", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_summary_md(results: list[dict], summary: dict, path: Path) -> None:
    """Write markdown summary with perf + accuracy sections."""
    ps = summary.get("perf_summary", {})
    total = ps.get("total", 0)
    passed = ps.get("passed", 0)
    failed = ps.get("failed", 0)
    rate = (passed / total * 100) if total else 0

    lines = [
        "# E2E Evaluation Report",
        "",
        f"**Run**: {summary['timestamp']} | **Total**: {total} models",
        "",
        "## Perf Pass Rate",
        "",
        "| | Count | Rate |",
        "|---|---|---|",
        f"| **PASS** | {passed} | {rate:.1f}% |",
        f"| **FAIL** | {failed} | {100 - rate:.1f}% |",
    ]

    # Perf failure distribution
    by_ft = summary.get("by_perf_failure_type", {})
    if by_ft:
        lines += [
            "",
            "## Perf Failure Distribution",
            "",
            "| Classification | Count | % of Failures |",
            "|---|---|---|",
        ]
        for fc, count in sorted(by_ft.items(), key=lambda x: -x[1]):
            pct = (count / failed * 100) if failed else 0
            lines.append(f"| {fc} | {count} | {pct:.1f}% |")

    # By task (perf)
    by_task = summary.get("by_task", {})
    if by_task:
        lines += [
            "",
            "## Perf Results by Task",
            "",
            "| Task | Total | Pass | Fail | Rate |",
            "|---|---|---|---|---|",
        ]
        for task_name, ts in sorted(by_task.items()):
            t_rate = (ts["passed"] / ts["total"] * 100) if ts["total"] else 0
            lines.append(
                f"| {task_name} | {ts['total']} | {ts['passed']} | {ts['failed']} | {t_rate:.1f}% |"
            )

    # Failed perf models
    failed_results = [r for r in results if r.get("perf") and not r["perf"].get("passed")]
    if failed_results:
        lines += [
            "",
            "## Failed Perf Models",
            "",
            "| Model | Task | Type | Classification | Error |",
            "|---|---|---|---|---|",
        ]
        for r in failed_results:
            perf = r["perf"]
            fc = perf.get("failure_classification", "UNKNOWN")
            lines.append(
                f"| {r['model']} | {r.get('task', '')} | {r.get('model_type', '')} "
                f"| {fc} | {perf.get('error', '')} |"
            )

    # Accuracy section (if present)
    acc_summary = summary.get("accuracy_summary")
    if acc_summary:
        from .accuracy import _build_accuracy_md_lines

        lines += ["", "---", "", *_build_accuracy_md_lines(results, acc_summary)]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML report (self-contained, dark theme from models_viewer.html)
# ---------------------------------------------------------------------------


def generate_html_report(
    report_data: dict,
    output_path: Path,
    registry_path: Path | None = None,
) -> None:
    """Generate interactive HTML report with Perf and Accuracy tabs."""
    from .accuracy import format_delta

    results = report_data.get("results", [])

    # Load registry for enrichment
    registry_lookup: dict[tuple[str, str], dict] = {}
    if registry_path and registry_path.exists():
        with registry_path.open(encoding="utf-8") as f:
            for entry in json.load(f):
                key = (entry.get("hf_id", ""), entry.get("task", ""))
                registry_lookup[key] = entry

    # Build viewer-compatible data
    viewer_data = []
    for r in results:
        hf_id = r.get("model", "")
        task = r.get("task", "")
        reg = registry_lookup.get((hf_id, task), {})
        perf = r.get("perf") or {}
        acc = r.get("accuracy")
        viewer_data.append(
            {
                "hf_id": hf_id,
                "task": task,
                "device": r.get("device", ""),
                "ep": r.get("ep"),
                "model_type": r.get("model_type", ""),
                "group": r.get("group", ""),
                "priority": r.get("priority", ""),
                "downloads": reg.get("downloads", 0) or 0,
                "last_update_time": reg.get("last_update_time"),
                "optimum_supported": reg.get("optimum_supported", False),
                # Perf fields
                "passed": perf.get("passed", False),
                "failure_classification": perf.get("failure_classification"),
                "elapsed": perf.get("elapsed", 0),
                "error": perf.get("error", ""),
                # Accuracy fields
                "accuracy_verdict": (
                    (acc.get("verdict") if acc and not acc.get("skipped") else None)
                    if acc is not None
                    else None
                ),
                "delta_display": (format_delta(acc) if acc and not acc.get("skipped") else ""),
                "metric": (
                    {
                        "name": (acc.get("winml_metric") or {}).get("metric"),
                        "baseline": (acc.get("pytorch_baseline_metric") or {}).get("value"),
                        "winml": (acc.get("winml_metric") or {}).get("value"),
                    }
                    if acc and not acc.get("skipped")
                    else None
                ),
            }
        )

    data_json = json.dumps(viewer_data, ensure_ascii=False, separators=(",", ":"))

    # Try to read models_viewer.html template for CSS
    template_dir = Path(__file__).parent.parent
    template_path = template_dir / "models_viewer.html"
    if template_path.exists():
        template_html = template_path.read_text(encoding="utf-8")
        css_match = re.search(r"<style>(.*?)</style>", template_html, re.DOTALL)
        base_css = css_match.group(1) if css_match else ""
    else:
        base_css = ""

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E2E Evaluation Report</title>
<style>
{base_css}
.summary-row {{ flex-wrap: nowrap !important; }}
.summary-card {{ min-width: auto !important; flex: 1; }}
.summary-card .value {{ font-size: 22px !important; }}
.badge-pass {{ background: rgba(78,205,196,0.15); color: #4ecdc4; }}
.badge-fail {{ background: rgba(255,107,157,0.15); color: #ff6b9d; }}
.badge-fc {{ background: rgba(255,217,61,0.12); color: #ffd93d; }}
.badge-acc-pass {{ background: rgba(78,205,196,0.15); color: #4ecdc4; }}
.badge-acc-risk {{ background: rgba(255,217,61,0.15); color: #ffd93d; }}
.badge-acc-reg {{ background: rgba(255,107,157,0.15); color: #ff6b9d; }}
.elapsed {{ color: var(--text2); font-size: 12px; white-space: nowrap; }}
.info-cell {{ color: var(--text2); font-size: 11px; white-space: nowrap; }}
.table-wrap {{ overflow-x: auto; }}
.table-wrap table {{ table-layout: auto; }}
.table-wrap th:first-child,
.table-wrap td:first-child {{
  position: sticky; left: 0; z-index: 2;
  background: var(--surface); min-width: 200px;
}}
.table-wrap th:first-child {{ background: var(--surface2); }}
tr[title] {{ cursor: help; }}
.col-sep {{ border-left: 2px solid var(--surface2) !important; }}
</style>
</head>
<body>

<div class="header">
  <h1>E2E Evaluation Report</h1>
  <div class="header-stats" id="headerStats"></div>
</div>

<div class="main">
  <aside class="sidebar">
    <input class="search-box" id="searchBox" type="text" placeholder="Search models...">
    <div class="filter-section">
      <h3>Status <span class="count" id="statusCount"></span>
        <button class="clear-btn" data-clear="status">clear</button></h3>
      <div class="filter-chips" id="statusChips"></div>
    </div>
    <div class="filter-section">
      <h3>Perf Class <span class="count" id="clsCount"></span>
        <button class="clear-btn" data-clear="cls">clear</button></h3>
      <div class="filter-chips" id="clsChips"></div>
    </div>
    <div class="filter-section">
      <h3>Acc Verdict <span class="count" id="accCount"></span>
        <button class="clear-btn" data-clear="acc">clear</button></h3>
      <div class="filter-chips" id="accChips"></div>
    </div>
    <div class="filter-section">
      <h3>Task <span class="count" id="taskCount"></span>
        <button class="clear-btn" data-clear="task">clear</button></h3>
      <div class="filter-chips" id="taskChips"></div>
    </div>
    <div class="filter-section">
      <h3>Model Type <span class="count" id="typeCount"></span>
        <button class="clear-btn" data-clear="model_type">clear</button></h3>
      <div class="filter-chips" id="typeChips"></div>
    </div>
    <div class="filter-section">
      <h3>Priority <span class="count" id="priorityCount"></span>
        <button class="clear-btn" data-clear="priority">clear</button></h3>
      <div class="filter-chips" id="priorityChips"></div>
    </div>
    <div class="filter-section">
      <h3>Group <span class="count" id="groupCount"></span>
        <button class="clear-btn" data-clear="group">clear</button></h3>
      <div class="filter-chips" id="groupChips"></div>
    </div>
    <div class="filter-section">
      <h3>Optimum <span class="count" id="optimumCount"></span>
        <button class="clear-btn" data-clear="optimum">clear</button></h3>
      <div class="filter-chips" id="optimumChips"></div>
    </div>
  </aside>

  <div class="content">
    <div class="summary-row" id="summaryCards"></div>
    <div class="toolbar">
      <div class="toolbar-left">
        <div class="result-count" id="resultCount"></div>
        <label style="font-size:12px;color:var(--text2)">Group by:</label>
        <select class="group-by-select" id="groupBySelect">
          <option value="none">None</option>
          <option value="task" selected>Task</option>
          <option value="model_type">Model Type</option>
          <option value="group">Group</option>
          <option value="status">Status</option>
          <option value="cls">Classification</option>
        </select>
        <label style="font-size:12px;color:var(--text2)">Sort:</label>
        <select class="sort-select" id="sortSelect">
          <option value="status_asc" selected>Status (FAIL first)</option>
          <option value="status_desc">Status (PASS first)</option>
          <option value="elapsed_desc">Time (Slow first)</option>
          <option value="elapsed_asc">Time (Fast first)</option>
          <option value="hf_id_asc">Name (A-Z)</option>
          <option value="downloads_desc">Downloads (High-Low)</option>
          <option value="updated_desc">Last Updated (Newest)</option>
        </select>
      </div>
    </div>
    <div id="output"></div>
  </div>
</div>

<script>
const DATA = {data_json};

DATA.forEach(d => {{
  d.status = d.passed ? 'PASS' : 'FAIL';
  d.cls = d.failure_classification || (d.passed ? '' : 'UNKNOWN');
  d.optimum = d.optimum_supported ? 'Yes' : 'No';
  d.acc = d.accuracy_verdict || (d.accuracy_verdict === null ? 'N/A' : '');
}});

const hasPerf = DATA.some(d => d.elapsed > 0 || d.passed !== undefined);
const hasAcc  = DATA.some(d => d.accuracy_verdict != null);

function formatNum(n) {{
  if (n == null) return '-';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return n.toLocaleString();
}}
function formatDate(iso) {{ return iso ? new Date(iso).toLocaleDateString('en-CA') : '-'; }}
function formatDelta(display, verdictCls) {{
  if (!display) return '-';
  return `<span class="badge ${{verdictCls}}">${{display}}</span>`;
}}

let filters = {{
  task: new Set(), model_type: new Set(), priority: new Set(),
  group: new Set(), status: new Set(), cls: new Set(), optimum: new Set(), acc: new Set()
}};
let searchQuery = '';
let groupBy = 'task';
let sortKey = 'status_asc';

function getUnique(key) {{
  const c = {{}};
  DATA.forEach(d => {{ const v = d[key] || '(empty)'; c[v] = (c[v]||0)+1; }});
  return Object.entries(c).sort((a,b) => b[1]-a[1]);
}}
function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}
function buildChips(id, key, entries) {{
  document.getElementById(id).innerHTML = entries.map(([v,c]) =>
    `<div class="chip ${{filters[key].has(v)?'active':''}}" data-key="${{key}}" data-val="${{esc(v)}}">
      ${{esc(v)}} <span class="chip-count">${{c}}</span></div>`
  ).join('');
}}
function toggleFilter(k,v) {{ filters[k].has(v)?filters[k].delete(v):filters[k].add(v); render(); }}
function clearFilter(k) {{ filters[k].clear(); render(); }}

function getFiltered() {{
  return DATA.filter(d => {{
    if (searchQuery && !d.hf_id.toLowerCase().includes(searchQuery)
        && !(d.model_type||'').toLowerCase().includes(searchQuery)
        && !(d.task||'').toLowerCase().includes(searchQuery)) return false;
    for (const k of ['task','model_type','priority','group','status','cls','optimum','acc'])
      if (filters[k].size > 0 && !filters[k].has(d[k]||'(empty)')) return false;
    return true;
  }});
}}
function sortData(arr) {{
  const s = [...arr];
  switch(sortKey) {{
    case 'status_asc': return s.sort((a,b) => a.passed-b.passed || b.elapsed-a.elapsed);
    case 'status_desc': return s.sort((a,b) => b.passed-a.passed || b.elapsed-a.elapsed);
    case 'elapsed_desc': return s.sort((a,b) => b.elapsed-a.elapsed);
    case 'elapsed_asc': return s.sort((a,b) => a.elapsed-b.elapsed);
    case 'hf_id_asc': return s.sort((a,b) => a.hf_id.localeCompare(b.hf_id));
    case 'downloads_desc': return s.sort((a,b) => (b.downloads||0)-(a.downloads||0));
    case 'updated_desc': return s.sort((a,b) => (b.last_update_time||'').localeCompare(a.last_update_time||''));
  }}
  return s;
}}

function renderSummary(f) {{
  const p = f.filter(d=>d.passed).length, fail = f.length-p;
  const rate = f.length ? (p/f.length*100).toFixed(0) : 0;
  const uIds = new Set(f.map(d=>d.hf_id)).size;
  const accPass = f.filter(d=>d.accuracy_verdict==='ACCURACY_PASS').length;
  const accTotal = f.filter(d=>d.accuracy_verdict!=null).length;
  document.getElementById('summaryCards').innerHTML = `
    <div class="summary-card"><div class="label">Perf Pass Rate</div>
      <div class="value c1">${{rate}}%</div></div>
    <div class="summary-card"><div class="label">Perf Pass / Fail / Total</div>
      <div class="value"><span style="color:var(--accent2)">${{p}}</span><span style="color:var(--text2)"> / </span><span style="color:var(--accent3)">${{fail}}</span><span style="color:var(--text2)"> / ${{f.length}}</span></div></div>
    <div class="summary-card"><div class="label">Unique Models</div>
      <div class="value c1">${{uIds}}</div></div>
    ${{accTotal > 0 ? `<div class="summary-card"><div class="label">Accuracy Pass Rate</div>
      <div class="value c1">${{accTotal ? (accPass/accTotal*100).toFixed(0) : 0}}%</div></div>` : ''}}`;
  document.getElementById('headerStats').innerHTML =
    `<div>Generated: <span>{generated_at}</span></div>
     <div>Showing: <span>${{f.length}}</span> / ${{DATA.length}}</div>`;
}}

function toggleSort(col) {{
  sortKey = (sortKey===col+'_desc') ? col+'_asc' : col+'_desc';
  document.getElementById('sortSelect').value = sortKey;
  render();
}}
function sortArrow(col) {{
  if (sortKey===col+'_asc') return ' \\u2191';
  if (sortKey===col+'_desc') return ' \\u2193';
  return '';
}}

function renderTable(items) {{
  const perfHead = hasPerf ? `
      <th onclick="toggleSort('status')" style="cursor:pointer" class="col-sep">Perf${{sortArrow('status')}}</th>
      <th>Classification</th>
      <th onclick="toggleSort('elapsed')" style="cursor:pointer">Time${{sortArrow('elapsed')}}</th>` : '';
  const accHead = hasAcc ? `
      <th class="col-sep">Acc Verdict</th>
      <th>Delta%</th>` : '';
  const header = `<thead><tr>
      <th onclick="toggleSort('hf_id')" style="cursor:pointer">Model ID${{sortArrow('hf_id')}}</th>
      <th>Task</th><th>Type</th>
      ${{perfHead}}${{accHead}}
      <th>Opt</th><th>Priority</th><th>Group</th>
      <th onclick="toggleSort('downloads')" style="cursor:pointer">Info${{sortArrow('downloads')}}</th>
    </tr></thead>`;
  const rows = items.map(d => {{
    const info = [formatDate(d.last_update_time), formatNum(d.downloads)].filter(x=>x!=='-').join(' · ');
    const errTip = d.error ? d.error.replace(/"/g,'&quot;') : '';
    const perfCells = hasPerf ? `
      <td class="col-sep"><span class="badge ${{d.passed?'badge-pass':'badge-fail'}}">${{d.status}}</span></td>
      <td>${{d.cls?`<span class="badge badge-fc">${{d.cls}}</span>`:'-'}}</td>
      <td class="elapsed">${{d.elapsed.toFixed(1)}}s</td>` : '';
    const verdictCls = d.accuracy_verdict==='ACCURACY_PASS' ? 'badge-acc-pass'
      : d.accuracy_verdict==='ACCURACY_AT_RISK' ? 'badge-acc-risk'
      : d.accuracy_verdict ? 'badge-acc-reg' : '';
    const accCells = hasAcc ? `
      <td class="col-sep">${{d.accuracy_verdict
        ? `<span class="badge ${{verdictCls}}">${{d.accuracy_verdict}}</span>`
        : '<span style="color:var(--text2);font-size:11px">N/A</span>'}}</td>
      <td>${{formatDelta(d.delta_display, verdictCls)}}</td>` : '';
    return `<tr${{errTip ? ` title="${{errTip}}"` : ''}}>
      <td><a class="hf-link" href="https://huggingface.co/${{d.hf_id}}" target="_blank">${{d.hf_id}}</a></td>
      <td><span class="badge badge-task">${{d.task||'-'}}</span></td>
      <td><span class="badge badge-type">${{d.model_type||'-'}}</span></td>
      ${{perfCells}}${{accCells}}
      <td><span class="badge ${{d.optimum_supported?'badge-opt-yes':'badge-opt-no'}}">${{d.optimum[0]}}</span></td>
      <td><span class="badge ${{d.priority==='P0'?'badge-p0':'badge-p1'}}">${{d.priority}}</span></td>
      <td><span class="badge badge-group">${{d.group||'-'}}</span></td>
      <td class="info-cell">${{info||'-'}}</td>
    </tr>`;
  }}).join('');
  return `<div class="table-wrap"><table>${{header}}<tbody>${{rows}}</tbody></table></div>`;
}}

function renderGrouped(items, key, tabFn) {{
  const groups = {{}};
  items.forEach(d => {{
    const g = d[key] || '(empty)';
    if (!groups[g]) groups[g] = [];
    groups[g].push(d);
  }});
  return Object.entries(groups).sort((a,b)=>b[1].length-a[1].length).map(([name,models]) => {{
    const p = models.filter(m=>m.passed).length;
    return `<div class="group-section">
      <div class="group-header" onclick="this.querySelector('.arrow').classList.toggle('open');this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'':'none'">
        <span class="arrow open">&#9654;</span>
        <span class="group-name">${{name}}</span>
        <span class="group-count">${{models.length}}</span>
        <span class="group-downloads">${{p}}/${{models.length}} passed</span>
      </div>
      <div class="group-body">${{tabFn(models)}}</div>
    </div>`;
  }}).join('');
}}

function render() {{
  const tasks=getUnique('task'), types=getUnique('model_type'),
    prios=getUnique('priority'), groups=getUnique('group'),
    statuses=getUnique('status'), clses=getUnique('cls'),
    optLabels=getUnique('optimum'), accLabels=getUnique('acc');
  buildChips('taskChips','task',tasks);
  buildChips('typeChips','model_type',types);
  buildChips('priorityChips','priority',prios);
  buildChips('groupChips','group',groups);
  buildChips('statusChips','status',statuses);
  buildChips('clsChips','cls',clses);
  buildChips('optimumChips','optimum',optLabels);
  buildChips('accChips','acc',accLabels);
  document.getElementById('taskCount').textContent=tasks.length;
  document.getElementById('typeCount').textContent=types.length;
  document.getElementById('priorityCount').textContent=prios.length;
  document.getElementById('groupCount').textContent=groups.length;
  document.getElementById('statusCount').textContent=statuses.length;
  document.getElementById('clsCount').textContent=clses.length;
  document.getElementById('optimumCount').textContent=optLabels.length;
  document.getElementById('accCount').textContent=accLabels.length;

  const filtered = getFiltered(), sorted = sortData(filtered);
  renderSummary(filtered);
  document.getElementById('resultCount').innerHTML =
    `Showing <span>${{filtered.length}}</span> of <span>${{DATA.length}}</span>`;

  const out = document.getElementById('output');
  out.innerHTML = groupBy==='none' ? renderTable(sorted) : renderGrouped(sorted, groupBy, renderTable);
}}

document.querySelector('.sidebar').addEventListener('click', e => {{
  const chip = e.target.closest('.chip[data-key][data-val]');
  if (chip) {{ toggleFilter(chip.dataset.key, chip.dataset.val); return; }}
  const clearBtn = e.target.closest('[data-clear]');
  if (clearBtn) clearFilter(clearBtn.dataset.clear);
}});
document.getElementById('searchBox').addEventListener('input', e => {{
  searchQuery = e.target.value.toLowerCase(); render();
}});
document.getElementById('groupBySelect').addEventListener('change', e => {{
  groupBy = e.target.value; render();
}});
document.getElementById('sortSelect').addEventListener('change', e => {{
  sortKey = e.target.value; render();
}});
render();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
