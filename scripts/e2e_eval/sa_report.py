# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SA eval HTML report — pre/post optimizer comparison with detailed breakdowns.

Shows separate pre/post SA level distributions, unknown op info,
EPContext ground-truth accuracy, and per-model drill-down.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def generate_sa_html_report(report_data: dict, output_path: Path) -> None:
    """Generate interactive HTML report for SA eval results."""
    results = report_data.get("results", [])
    pre_opt = report_data.get("pre_optimization", {})
    post_opt = report_data.get("post_optimization", {})
    effectiveness = report_data.get("optimizer_effectiveness", {})
    epctx_gt = report_data.get("epcontext_ground_truth", {})
    epctx_pre_acc = epctx_gt.get("avg_accuracy_pre")
    epctx_post_acc = epctx_gt.get("avg_accuracy_post")
    epctx_pre_n = epctx_gt.get("models_with_pre_gt", 0)
    epctx_post_n = epctx_gt.get("models_with_post_gt", 0)

    viewer_data = []
    for r in results:
        pre_sum = r.get("sa_pre", {}).get("summary", {})
        post_sum = r.get("sa_post", {}).get("summary", {})
        delta = r.get("delta", {})
        optim = r.get("optimization", {})
        viewer_data.append(
            {
                "model": r.get("model", ""),
                "task": r.get("task", ""),
                "model_type": r.get("model_type", ""),
                "elapsed": r.get("elapsed", 0),
                # pre SA
                "pre_supported": pre_sum.get("supported", 0),
                "pre_partial": pre_sum.get("partial", 0),
                "pre_unsupported": pre_sum.get("unsupported", 0),
                "pre_unknown": pre_sum.get("unknown", 0),
                "pre_total": pre_sum.get("total", 0),
                "pre_supported_ratio": pre_sum.get("supported_ratio", 0),
                "pre_partial_patterns": r.get("sa_pre", {}).get("partial_patterns", []),
                "pre_unsupported_patterns": r.get("sa_pre", {}).get("unsupported_patterns", []),
                "pre_unknown_patterns": r.get("sa_pre", {}).get("unknown_patterns", []),
                "pre_info_items": r.get("sa_pre", {}).get("info_items", []),
                # post SA
                "post_supported": post_sum.get("supported", 0),
                "post_partial": post_sum.get("partial", 0),
                "post_unsupported": post_sum.get("unsupported", 0),
                "post_unknown": post_sum.get("unknown", 0),
                "post_total": post_sum.get("total", 0),
                "post_supported_ratio": post_sum.get("supported_ratio", 0),
                "post_partial_patterns": r.get("sa_post", {}).get("partial_patterns", []),
                "post_unsupported_patterns": r.get("sa_post", {}).get("unsupported_patterns", []),
                "post_unknown_patterns": r.get("sa_post", {}).get("unknown_patterns", []),
                "post_info_items": r.get("sa_post", {}).get("info_items", []),
                # delta
                "supported_ratio_delta": delta.get("supported_ratio_delta", 0),
                "improved": delta.get("improved", []),
                "fused_away": delta.get("fused_away", []),
                "regressed": delta.get("regressed", []),
                "unchanged_partial_unsupported": delta.get("unchanged_partial_unsupported", []),
                # optimization
                "optim_config": optim.get("optim_config", {}),
                "optim_flags": list(optim.get("optim_config", {}).keys()),
                # epcontext pre (graph_optimized compiled vs sa_pre)
                "pre_epctx_accuracy": r.get("epcontext_diff_pre", {})
                .get("summary", {})
                .get("accuracy"),
                "pre_epctx_fn_ops": [
                    c["pattern_id"]
                    for c in r.get("epcontext_diff_pre", {}).get("comparison", [])
                    if c["verdict"] == "FN"
                ],
                "pre_epctx_fp_ops": [
                    c["pattern_id"]
                    for c in r.get("epcontext_diff_pre", {}).get("comparison", [])
                    if c["verdict"] == "FP"
                ],
                # epcontext post (sa_optimized compiled vs sa_post)
                "post_epctx_accuracy": r.get("epcontext_diff_post", {})
                .get("summary", {})
                .get("accuracy"),
                "post_epctx_fn_ops": [
                    c["pattern_id"]
                    for c in r.get("epcontext_diff_post", {}).get("comparison", [])
                    if c["verdict"] == "FN"
                ],
                "post_epctx_fp_ops": [
                    c["pattern_id"]
                    for c in r.get("epcontext_diff_post", {}).get("comparison", [])
                    if c["verdict"] == "FP"
                ],
            }
        )

    common_improved = report_data.get("common_improved_patterns", [])
    common_fused = report_data.get("common_fused_away_patterns", [])
    unresolved = report_data.get("unresolved_partial_unsupported_patterns", [])
    common_unknown = report_data.get("common_unknown_patterns", [])

    data_json = json.dumps(viewer_data, ensure_ascii=False, separators=(",", ":"))
    improved_json = json.dumps(common_improved, ensure_ascii=False, separators=(",", ":"))
    fused_json = json.dumps(common_fused, ensure_ascii=False, separators=(",", ":"))
    unresolved_json = json.dumps(unresolved, ensure_ascii=False, separators=(",", ":"))
    unknown_json = json.dumps(common_unknown, ensure_ascii=False, separators=(",", ":"))

    template_dir = Path(__file__).parent.parent
    template_path = template_dir / "models_viewer.html"
    if template_path.exists():
        template_html = template_path.read_text(encoding="utf-8")
        css_match = re.search(r"<style>(.*?)</style>", template_html, re.DOTALL)
        base_css = css_match.group(1) if css_match else _fallback_css()
    else:
        base_css = _fallback_css()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_complete = report_data.get("models_complete", 0)
    n_total = report_data.get("models_total", 0)
    avg_pre = pre_opt.get("avg_supported_ratio", 0)
    avg_post = post_opt.get("avg_supported_ratio", 0)
    avg_delta = effectiveness.get("avg_supported_ratio_delta", 0)
    n_improved = effectiveness.get("models_improved", 0)
    n_regressed = effectiveness.get("models_regressed", 0)
    avg_pre_unknown = pre_opt.get("avg_unknown_count", 0)
    avg_post_unknown = post_opt.get("avg_unknown_count", 0)
    delta_cls = "c-good" if avg_delta > 0 else "c-muted"
    regressed_cls = "c-bad" if n_regressed > 0 else "c-muted"

    def _epctx_card(label: str, acc: float | None, n: int) -> str:
        if not n or acc is None:
            return ""
        cls = "c-good" if acc >= 0.9 else "c-warn"
        return f"""
    <div class="summary-card">
      <div class="label">{label}</div>
      <div class="value {cls}">{acc * 100:.1f}%</div>
      <div style="font-size:11px;color:var(--text2)">{n} models w/ GT</div>
    </div>"""

    epctx_section = _epctx_card("EPCtx Acc (Pre SA)", epctx_pre_acc, epctx_pre_n) + _epctx_card(
        "EPCtx Acc (Post SA)", epctx_post_acc, epctx_post_n
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SA Eval Report</title>
<style>
{base_css}
.summary-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
.summary-card {{
  background: var(--surface, #1a1d27); border: 1px solid var(--border, #2e3142);
  border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 130px;
}}
.summary-card .label {{ font-size: 11px; color: var(--text2, #8b8fa3); text-transform: uppercase; letter-spacing: 0.8px; }}
.summary-card .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
.c-good {{ color: var(--accent2, #4ecdc4); }}
.c-warn {{ color: var(--accent4, #ffd93d); }}
.c-bad  {{ color: var(--accent3, #ff6b9d); }}
.c-info {{ color: var(--accent, #6c8aff); }}
.c-muted {{ color: var(--text2, #8b8fa3); }}

.badge {{ display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
.badge-supported    {{ background: rgba(78,205,196,0.15); color: #4ecdc4; }}
.badge-partial      {{ background: rgba(255,217,61,0.15); color: #ffd93d; }}
.badge-unsupported  {{ background: rgba(255,107,157,0.15); color: #ff6b9d; }}
.badge-unknown  {{ background: rgba(139,143,163,0.15); color: #8b8fa3; }}
.badge-improved {{ background: rgba(78,205,196,0.15); color: #4ecdc4; }}
.badge-regressed {{ background: rgba(255,107,157,0.15); color: #ff6b9d; }}
.badge-task     {{ background: rgba(108,138,255,0.1); color: var(--accent, #6c8aff); }}
.badge-tp {{ background: rgba(78,205,196,0.15); color: #4ecdc4; }}
.badge-tn {{ background: rgba(108,138,255,0.15); color: #6c8aff; }}
.badge-fp {{ background: rgba(255,107,157,0.15); color: #ff6b9d; }}
.badge-fn {{ background: rgba(255,217,61,0.15); color: #ffd93d; }}

.tabs {{ display: flex; gap: 0; border-bottom: 2px solid var(--border, #2e3142); margin-bottom: 20px; }}
.tab {{
  padding: 10px 20px; cursor: pointer; font-size: 13px; font-weight: 600;
  color: var(--text2, #8b8fa3); border-bottom: 2px solid transparent; margin-bottom: -2px;
}}
.tab:hover {{ color: var(--text, #e1e4ed); }}
.tab.active {{ color: var(--accent, #6c8aff); border-bottom-color: var(--accent, #6c8aff); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.table-wrap {{ overflow-x: auto; }}
.table-wrap table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.table-wrap th {{
  text-align: left; padding: 10px 12px; background: var(--surface2, #242734);
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--text2, #8b8fa3); position: sticky; top: 0; z-index: 1; cursor: pointer;
}}
.table-wrap td {{ padding: 8px 12px; border-bottom: 1px solid var(--border, #2e3142); }}
.table-wrap tr:hover td {{ background: rgba(108,138,255,0.04); }}
.table-wrap th:first-child, .table-wrap td:first-child {{
  position: sticky; left: 0; z-index: 2; background: var(--surface, #1a1d27); min-width: 200px;
}}
.table-wrap th:first-child {{ background: var(--surface2, #242734); }}

.detail-panel {{
  background: var(--surface2, #242734); border: 1px solid var(--border, #2e3142);
  border-radius: 8px; padding: 16px; margin: 4px 0 8px 0;
}}
.detail-cols {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.detail-col {{ flex: 1; min-width: 180px; }}
.detail-col h5 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); margin-bottom: 6px; }}

/* Level distribution bar */
.level-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--surface2); gap: 1px; }}
.lb-supported   {{ background: #4ecdc4; }}
.lb-partial     {{ background: #ffd93d; }}
.lb-unsupported {{ background: #ff6b9d; }}
.lb-unk   {{ background: #8b8fa3; }}
.level-legend {{ display: flex; gap: 8px; margin-top: 4px; font-size: 11px; }}
.ll-item {{ display: flex; align-items: center; gap: 3px; }}
.ll-dot  {{ width: 8px; height: 8px; border-radius: 50%; }}

.search-box {{
  width: 100%; padding: 10px 14px; background: var(--surface2, #242734);
  border: 1px solid var(--border, #2e3142); border-radius: 8px;
  color: var(--text, #e1e4ed); font-size: 13px; margin-bottom: 16px; outline: none;
}}
.search-box:focus {{ border-color: var(--accent, #6c8aff); }}
.hf-link {{ color: var(--accent, #6c8aff); text-decoration: none; }}
.hf-link:hover {{ text-decoration: underline; }}

.section-title {{ font-size: 13px; font-weight: 600; color: var(--text2); margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
</style>
</head>
<body>

<div class="header">
  <h1>SA Eval Report &mdash; Pre / Post Optimizer</h1>
  <div class="header-stats">
    <div>Generated: <span>{generated_at}</span></div>
    <div>Models: <span>{n_complete}</span> / {n_total} complete</div>
  </div>
</div>

<div style="padding: 24px 32px; max-width: 1500px; margin: 0 auto;">

  <!-- Summary cards -->
  <div class="summary-row">
    <div class="summary-card">
      <div class="label">Avg SUPPORTED (Pre)</div>
      <div class="value c-info">{avg_pre * 100:.1f}%</div>
      <div style="font-size:11px;color:var(--text2)">Before optimization</div>
    </div>
    <div class="summary-card">
      <div class="label">Avg SUPPORTED (Post)</div>
      <div class="value c-good">{avg_post * 100:.1f}%</div>
      <div style="font-size:11px;color:var(--text2)">After optimization</div>
    </div>
    <div class="summary-card">
      <div class="label">Avg Delta</div>
      <div class="value {delta_cls}">{avg_delta * 100:+.1f}%</div>
    </div>
    <div class="summary-card">
      <div class="label">Improved</div>
      <div class="value c-good">{n_improved}</div>
    </div>
    <div class="summary-card">
      <div class="label">Regressed</div>
      <div class="value {regressed_cls}">{n_regressed}</div>
    </div>
    <div class="summary-card">
      <div class="label">Avg UNKNOWN (Pre→Post)</div>
      <div class="value c-muted" style="font-size:18px">{avg_pre_unknown:.1f} → {avg_post_unknown:.1f}</div>
    </div>
    <div class="summary-card">
      <div class="label">All-SUPPORTED (Post)</div>
      <div class="value c-good">{post_opt.get("models_all_supported", 0)}</div>
    </div>
    {epctx_section}
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" data-tab="models">Per-Model Results</div>
    <div class="tab" data-tab="improved">Improved Patterns</div>
    <div class="tab" data-tab="unresolved">Unresolved PARTIAL/UNSUPPORTED</div>
    <div class="tab" data-tab="unknown">Unknown Ops</div>
  </div>

  <div class="tab-content active" id="tab-models">
    <input class="search-box" id="searchBox" type="text" placeholder="Search models...">
    <div id="modelTable"></div>
  </div>
  <div class="tab-content" id="tab-improved"><div id="improvedList"></div></div>
  <div class="tab-content" id="tab-unresolved"><div id="unresolvedList"></div></div>
  <div class="tab-content" id="tab-unknown"><div id="unknownList"></div></div>
</div>

<script>
const DATA = {data_json};
const COMMON_IMPROVED = {improved_json};
const COMMON_FUSED = {fused_json};
const UNRESOLVED = {unresolved_json};
const COMMON_UNKNOWN = {unknown_json};

// ---- Tabs ----
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

function esc(s) {{ const d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; }}
function pct(v) {{ return (v * 100).toFixed(1) + '%'; }}
function deltaPct(v) {{
  const s = v > 0 ? '+' : '';
  const cls = v > 0.001 ? 'c-good' : v < -0.001 ? 'c-bad' : 'c-muted';
  return `<span class="${{cls}}">${{s}}${{(v*100).toFixed(1)}}%</span>`;
}}

function levelBar(s, p, u_, unk) {{
  const total = s + p + u_ + unk || 1;
  const sp = (s/total*100).toFixed(1), pp = (p/total*100).toFixed(1), up = (u_/total*100).toFixed(1), ukp = (unk/total*100).toFixed(1);
  return `<div>
    <div class="level-bar">
      ${{s>0   ? `<div class="lb-supported"   style="width:${{sp}}%"  title="SUPPORTED ${{s}}"></div>`   : ''}}
      ${{p>0   ? `<div class="lb-partial"     style="width:${{pp}}%"  title="PARTIAL ${{p}}"></div>`     : ''}}
      ${{u_>0  ? `<div class="lb-unsupported" style="width:${{up}}%"  title="UNSUPPORTED ${{u_}}"></div>` : ''}}
      ${{unk>0 ? `<div class="lb-unk"         style="width:${{ukp}}%" title="UNKNOWN ${{unk}}"></div>`   : ''}}
    </div>
    <div class="level-legend">
      ${{s>0   ? '<div class="ll-item"><div class="ll-dot" style="background:#4ecdc4"></div><span>S '+s+'</span></div>' : ''}}
      ${{p>0   ? '<div class="ll-item"><div class="ll-dot" style="background:#ffd93d"></div><span>P '+p+'</span></div>' : ''}}
      ${{u_>0  ? '<div class="ll-item"><div class="ll-dot" style="background:#ff6b9d"></div><span>U '+u_+'</span></div>' : ''}}
      ${{unk>0 ? '<div class="ll-item"><div class="ll-dot" style="background:#8b8fa3"></div><span>? '+unk+'</span></div>' : ''}}
    </div>
  </div>`;
}}

function epctxOps(fnOps, fpOps) {{
  let html = '';
  if (fpOps && fpOps.length) {{
    html += '<div style="margin-bottom:4px"><span style="font-size:11px;font-weight:600;color:#ff6b9d">FP \u2014 SA=SUPPORTED but CPU fallback</span>';
    html += fpOps.map(p => `<div style="font-size:12px;margin:2px 0"><span class="badge badge-fp">${{esc(p)}}</span></div>`).join('');
    html += '</div>';
  }}
  if (fnOps && fnOps.length) {{
    html += '<div style="margin-bottom:4px"><span style="font-size:11px;font-weight:600;color:#ffd93d">FN \u2014 SA=PARTIAL/UNSUPPORTED but NPU compiled</span>';
    html += fnOps.map(p => `<div style="font-size:12px;margin:2px 0"><span class="badge badge-fn">${{esc(p)}}</span></div>`).join('');
    html += '</div>';
  }}
  if (!html) html = '<span style="font-size:12px;color:var(--accent2)">no FP / FN \u2714</span>';
  return html;
}}

function patternList(pids, badgeCls) {{
  if (!pids || !pids.length) return '<span class="c-muted" style="font-size:12px">none</span>';
  return pids.map(p => `<div style="font-size:12px;margin:2px 0"><span class="badge ${{badgeCls}}">${{esc(p)}}</span></div>`).join('');
}}

function infoList(items) {{
  if (!items || !items.length) return '';
  return items.map(it => `<div style="font-size:11px;color:var(--text2);margin:2px 0;padding:4px 8px;background:rgba(0,0,0,0.15);border-radius:4px">
    <strong style="color:var(--text)">${{esc(it.pattern_id)}}</strong><br>
    ${{esc(it.explanation)}}
  </div>`).join('');
}}

// ---- Per-Model Table ----
let sortKey = 'pre_supported_asc';
let searchQuery = '';

function sortData(arr) {{
  const s = [...arr];
  switch(sortKey) {{
    case 'pre_supported_asc':  return s.sort((a,b) => a.pre_supported_ratio - b.pre_supported_ratio);
    case 'pre_supported_desc': return s.sort((a,b) => b.pre_supported_ratio - a.pre_supported_ratio);
    case 'post_supported_asc': return s.sort((a,b) => a.post_supported_ratio - b.post_supported_ratio);
    case 'post_supported_desc':return s.sort((a,b) => b.post_supported_ratio - a.post_supported_ratio);
    case 'delta_desc': return s.sort((a,b) => b.supported_ratio_delta - a.supported_ratio_delta);
    case 'delta_asc':  return s.sort((a,b) => a.supported_ratio_delta - b.supported_ratio_delta);
    case 'model_asc':  return s.sort((a,b) => a.model.localeCompare(b.model));
    case 'unknown_desc': return s.sort((a,b) => (b.pre_unknown||0) - (a.pre_unknown||0));
    case 'pre_epctx_desc': return s.sort((a,b) => (b.pre_epctx_accuracy||0) - (a.pre_epctx_accuracy||0));
    case 'pre_epctx_asc':  return s.sort((a,b) => (a.pre_epctx_accuracy||0) - (b.pre_epctx_accuracy||0));
    case 'post_epctx_desc': return s.sort((a,b) => (b.post_epctx_accuracy||0) - (a.post_epctx_accuracy||0));
    case 'post_epctx_asc':  return s.sort((a,b) => (a.post_epctx_accuracy||0) - (b.post_epctx_accuracy||0));
  }}
  return s;
}}

function renderModelTable() {{
  let filtered = DATA;
  if (searchQuery) filtered = DATA.filter(d => d.model.toLowerCase().includes(searchQuery) || (d.task||'').toLowerCase().includes(searchQuery));
  filtered = sortData(filtered);
  const arrow = col => sortKey===col+'_asc' ? ' \u2191' : sortKey===col+'_desc' ? ' \u2193' : '';

  let html = `<div class="table-wrap"><table><thead><tr>
    <th onclick="toggleSort('model')">Model${{arrow('model')}}</th>
    <th>Task</th>
    <th onclick="toggleSort('pre_supported')">Pre SA${{arrow('pre_supported')}}</th>
    <th onclick="toggleSort('pre_epctx')">Pre EPCtx${{arrow('pre_epctx')}}</th>
    <th>SA Opt Flags</th>
    <th onclick="toggleSort('post_supported')">Post SA${{arrow('post_supported')}}</th>
    <th onclick="toggleSort('post_epctx')">Post EPCtx${{arrow('post_epctx')}}</th>
    <th onclick="toggleSort('delta')">SA Delta${{arrow('delta')}}</th>
    <th onclick="toggleSort('unknown')">Unknown${{arrow('unknown')}}</th>
    <th>Time</th>
  </tr></thead><tbody>`;

  filtered.forEach((d, i) => {{
    function epctxAcc(acc) {{
      if (acc == null) return '<span class="c-muted">—</span>';
      return `<span class="${{acc>=0.9?'c-good':'c-warn'}}" style="font-weight:600">${{pct(acc)}}</span>`;
    }}
    const unknownBadge = d.pre_unknown > 0
      ? `<span class="badge badge-unknown">${{d.pre_unknown}}</span>`
      : '<span class="c-muted">0</span>';
    const flagsCell = (d.optim_flags||[]).length
      ? d.optim_flags.map(f => `<div style="font-size:10px;font-family:monospace;white-space:nowrap;color:var(--accent2)">${{esc(f)}}</div>`).join('')
      : '<span class="c-muted" style="font-size:11px">—</span>';
    html += `<tr onclick="toggleDetail(${{i}})" style="cursor:pointer">
      <td><a class="hf-link" href="https://huggingface.co/${{esc(d.model)}}" target="_blank" onclick="event.stopPropagation()">${{esc(d.model)}}</a></td>
      <td><span class="badge badge-task">${{esc(d.task||'-')}}</span></td>
      <td>${{levelBar(d.pre_supported,d.pre_partial,d.pre_unsupported,d.pre_unknown)}}</td>
      <td>${{epctxAcc(d.pre_epctx_accuracy)}}</td>
      <td>${{flagsCell}}</td>
      <td>${{levelBar(d.post_supported,d.post_partial,d.post_unsupported,d.post_unknown)}}</td>
      <td>${{epctxAcc(d.post_epctx_accuracy)}}</td>
      <td>${{deltaPct(d.supported_ratio_delta)}}</td>
      <td>${{unknownBadge}}</td>
      <td style="color:var(--text2);font-size:12px">${{(d.elapsed||0).toFixed(1)}}s</td>
    </tr>
    <tr id="detail-${{i}}" style="display:none"><td colspan="10">
      <div class="detail-panel">

        <!-- PRE SA ROW -->
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text2);font-weight:600;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--border)">Pre SA — graph_optimized.onnx</div>
        <div class="detail-cols" style="margin-bottom:14px">
          <div class="detail-col">
            <h5>PARTIAL / UNSUPPORTED Patterns</h5>
            ${{patternList(d.pre_partial_patterns, 'badge-partial')}}
            ${{patternList(d.pre_unsupported_patterns, 'badge-unsupported')}}
          </div>
          <div class="detail-col">
            <h5>UNKNOWN Patterns</h5>
            ${{patternList(d.pre_unknown_patterns, 'badge-unknown')}}
          </div>
          <div class="detail-col">
            <h5>EPContext diff (graph_optimized compiled)</h5>
            ${{d.pre_epctx_accuracy != null
              ? epctxOps(d.pre_epctx_fn_ops, d.pre_epctx_fp_ops)
              : '<span class="c-muted" style="font-size:12px">no compiled ONNX</span>'}}
          </div>
          ${{d.pre_info_items.length ? '<div class="detail-col"><h5>SA Explanations</h5>' + infoList(d.pre_info_items) + '</div>' : ''}}
        </div>

        <!-- OPTIMIZATION FLAGS -->
        ${{(d.optim_flags||[]).length ? '<div style="display:flex;align-items:center;gap:10px;padding:6px 0;margin-bottom:10px;border-top:1px solid var(--border);border-bottom:1px solid var(--border)"><span style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text2);font-weight:600;white-space:nowrap">SA Auto-Conf \u2192</span>' + d.optim_flags.map(f => '<span style="font-size:11px;font-family:monospace;padding:2px 8px;background:rgba(78,205,196,0.1);border-radius:4px;color:var(--accent2)">' + esc(f) + '</span>').join(' ') + '</div>' : ''}}

        <!-- POST SA ROW -->
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.8px;color:var(--text2);font-weight:600;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid var(--border)">Post SA — sa_optimized.onnx</div>
        <div class="detail-cols">
          <div class="detail-col">
            <h5>PARTIAL / UNSUPPORTED Patterns</h5>
            ${{patternList(d.post_partial_patterns, 'badge-partial')}}
            ${{patternList(d.post_unsupported_patterns, 'badge-unsupported')}}
          </div>
          <div class="detail-col">
            <h5>UNKNOWN Patterns</h5>
            ${{patternList(d.post_unknown_patterns, 'badge-unknown')}}
          </div>
          <div class="detail-col">
            <h5>EPContext diff (sa_optimized compiled)</h5>
            ${{d.post_epctx_accuracy != null
              ? epctxOps(d.post_epctx_fn_ops, d.post_epctx_fp_ops)
              : '<span class="c-muted" style="font-size:12px">no compiled ONNX</span>'}}
          </div>
          <div class="detail-col">
            <h5>Delta vs Pre</h5>
            ${{d.improved.length ? '<div style="margin-bottom:4px"><span style="font-size:11px;font-weight:600;color:#4ecdc4">IMPROVED</span><br>' + patternList(d.improved, 'badge-improved') + '</div>' : ''}}
            ${{(d.fused_away||[]).length ? '<div style="margin-bottom:4px"><span style="font-size:11px;font-weight:600;color:#4ecdc4">FUSED AWAY</span><br>' + patternList(d.fused_away, 'badge-improved') + '</div>' : ''}}
            ${{d.regressed.length ? '<div style="margin-bottom:4px"><span style="font-size:11px;font-weight:600;color:#ff6b9d">REGRESSED</span><br>' + patternList(d.regressed, 'badge-regressed') + '</div>' : ''}}
            ${{!d.improved.length && !(d.fused_away||[]).length && !d.regressed.length ? '<span class="c-muted" style="font-size:12px">unchanged</span>' : ''}}
          </div>
        </div>

      </div>
    </td></tr>`;
  }});

  html += `</tbody></table></div><div style="font-size:12px;color:var(--text2);margin-top:8px">Showing ${{filtered.length}} of ${{DATA.length}} models</div>`;
  document.getElementById('modelTable').innerHTML = html;
}}

function toggleDetail(i) {{
  const el = document.getElementById('detail-'+i);
  el.style.display = el.style.display === 'none' ? '' : 'none';
}}

function toggleSort(col) {{
  sortKey = sortKey === col+'_desc' ? col+'_asc' : col+'_desc';
  renderModelTable();
}}

document.getElementById('searchBox').addEventListener('input', e => {{
  searchQuery = e.target.value.toLowerCase();
  renderModelTable();
}});

// ---- Pattern list tabs (shared renderer) ----
function renderPatternList(containerId, items, color, emptyMsg, subtitle) {{
  if (!items.length) {{
    document.getElementById(containerId).innerHTML = `<div style="color:var(--text2);padding:20px">${{emptyMsg}}</div>`;
    return;
  }}
  let html = `<div style="font-size:12px;color:var(--text2);margin-bottom:12px">${{subtitle}}</div>`;
  items.forEach(m => {{
    html += `<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--border)">
      <div style="font-size:18px;font-weight:700;color:${{color}};min-width:30px;text-align:center">${{m.count}}</div>
      <div><div style="font-weight:600">${{esc(m.pattern)}}</div>
           <div style="font-size:11px;color:var(--text2)">${{m.count}} model(s)</div></div>
    </div>`;
  }});
  document.getElementById(containerId).innerHTML = html;
}}

function renderUnknown() {{
  if (!COMMON_UNKNOWN.length) {{
    document.getElementById('unknownList').innerHTML =
      '<div style="color:var(--text2);padding:20px">No UNKNOWN ops found — all operators have support data.</div>';
    return;
  }}
  let html = '<div style="font-size:12px;color:var(--text2);margin-bottom:12px">Patterns with no entry in the runtime database. SA cannot classify these — treated as UNKNOWN rather than PARTIAL/UNSUPPORTED. These represent gaps in the runtime database coverage.</div>';
  COMMON_UNKNOWN.forEach(m => {{
    html += `<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--border)">
      <div style="font-size:18px;font-weight:700;color:#8b8fa3;min-width:30px;text-align:center">${{m.count}}</div>
      <div><div style="font-weight:600"><span class="badge badge-unknown">${{esc(m.pattern)}}</span></div>
           <div style="font-size:11px;color:var(--text2)">${{m.count}} model(s) &mdash; not in runtime database</div></div>
    </div>`;
  }});
  document.getElementById('unknownList').innerHTML = html;
}}

renderModelTable();
renderPatternList('improvedList', COMMON_IMPROVED, '#4ecdc4',
  COMMON_FUSED.length ? '' : 'No improvement data yet.',
  'Patterns that moved PARTIAL/UNSUPPORTED \u2192 SUPPORTED (explicit level change). See also "Fused Away" below.');
if (COMMON_FUSED.length) {{
  const fusedTitle = document.createElement('div');
  fusedTitle.className = 'section-title';
  fusedTitle.style.marginTop = '20px';
  fusedTitle.textContent = 'Fused Away (implicit improvement)';
  document.getElementById('improvedList').appendChild(fusedTitle);
  const fusedSub = document.createElement('div');
  fusedSub.style.cssText = 'font-size:12px;color:var(--text2);margin-bottom:12px';
  fusedSub.textContent = 'PARTIAL/UNSUPPORTED patterns eliminated by optimizer fusion (op disappears from graph).';
  document.getElementById('improvedList').appendChild(fusedSub);
  COMMON_FUSED.forEach(m => {{
    const el = document.createElement('div');
    el.style.cssText = 'display:flex;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--border)';
    el.innerHTML = '<div style="font-size:18px;font-weight:700;color:#4ecdc4;min-width:30px;text-align:center">' + m.count + '</div>' +
      '<div><div style="font-weight:600">' + esc(m.pattern) + '</div>' +
      '<div style="font-size:11px;color:var(--text2)">' + m.count + ' model(s) \u2014 fused away by optimizer</div></div>';
    document.getElementById('improvedList').appendChild(el);
  }});
}}
renderPatternList('unresolvedList', UNRESOLVED, '#ffd93d',
  'No unresolved patterns \u2014 all PARTIAL/UNSUPPORTED fixed!', 'Patterns still PARTIAL/UNSUPPORTED after optimization. No optimizer rewrite available or rewrite insufficient.');
renderUnknown();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _fallback_css() -> str:
    return """\
:root {
  --bg: #0f1117; --surface: #1a1d27; --surface2: #242734;
  --border: #2e3142; --text: #e1e4ed; --text2: #8b8fa3;
  --accent: #6c8aff; --accent2: #4ecdc4; --accent3: #ff6b9d; --accent4: #ffd93d;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
.header {
  background: linear-gradient(135deg, #1a1d27 0%, #242840 100%);
  border-bottom: 1px solid var(--border); padding: 24px 32px;
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}
.header h1 {
  font-size: 22px; font-weight: 600;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-stats { display: flex; gap: 20px; font-size: 13px; color: var(--text2); }
.header-stats span { font-weight: 600; color: var(--accent); }
"""
