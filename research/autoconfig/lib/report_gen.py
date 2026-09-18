# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""report_gen.py — Phase 3 HTML report generator for autoconfig.

Reads results.tsv and generates report.html with:
  - Summary bar chart (p50 per hypothesis, colour-coded by status)
  - Experiment table (config / delta_pct / status / CV)
  - Champion config box
"""

from __future__ import annotations

import csv
import html as html_lib
from datetime import datetime
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_tsv(results_tsv: Path) -> list[dict]:
    if not results_tsv.exists():
        return []
    with results_tsv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _status_color(status: str) -> str:
    s = status.lower()
    if "new best" in s or (s.startswith("keep") and "marginal" not in s):
        return "#2e7d32"  # dark green
    if "marginal" in s:
        return "#f57f17"  # amber
    if "discard" in s:
        return "#b0bec5"  # grey
    if "crash" in s or "fail" in s:
        return "#c62828"  # red
    return "#78909c"


def _status_bg(status: str) -> str:
    s = status.lower()
    if "new best" in s or (s.startswith("keep") and "marginal" not in s):
        return "#e8f5e9"
    if "marginal" in s:
        return "#fff8e1"
    if "crash" in s or "fail" in s:
        return "#ffebee"
    return "#f5f5f5"


def _p50_float(val: str | None) -> float | None:
    if not val or val == "N/A" or "UNSTABLE" in str(val):
        return None
    try:
        return float(str(val).replace("ms", "").strip())
    except ValueError:
        return None


# ── bar chart ─────────────────────────────────────────────────────────────────


def _bar_chart_html(rows: list[dict], baseline_p50: float | None) -> str:
    valid = [(r, _p50_float(r.get("median_p50_ms") or r.get("screen_p50_ms"))) for r in rows]
    valid = [(r, v) for r, v in valid if v is not None]
    if not valid:
        return "<p style='color:#888;font-size:12px'>No benchmark data yet.</p>"

    max_val = max(v for _, v in valid) * 1.1
    bars = []
    for r, p50 in valid:
        label = html_lib.escape(r.get("label", "?"))
        status = r.get("status", "")
        color = _status_color(status)
        width_pct = p50 / max_val * 100
        delta = r.get("delta_pct", "")
        baseline_marker = ""
        if baseline_p50:
            bx = baseline_p50 / max_val * 100
            baseline_marker = (
                f'<div style="position:absolute;left:{bx:.1f}%;top:0;bottom:0;'
                f'width:2px;background:#3949ab;opacity:0.4;z-index:2"></div>'
            )
        bars.append(f"""
  <div style="margin-bottom:6px;position:relative">
    {baseline_marker}
    <div style="font-size:10px;color:#556;margin-bottom:2px;white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;max-width:600px">{label}</div>
    <div style="display:flex;align-items:center;gap:8px">
      <div style="flex:1;background:#eee;border-radius:3px;height:16px;position:relative">
        <div style="width:{width_pct:.1f}%;background:{color};height:100%;border-radius:3px;
                    transition:width 0.3s"></div>
      </div>
      <div style="font-size:11px;color:#334;min-width:60px">{p50:.1f}ms
        <span style="color:{color};font-size:10px">{html_lib.escape(delta)}</span>
      </div>
    </div>
  </div>""")

    return (
        '<div style="max-width:700px">\n'
        '  <div style="font-size:10px;color:#3949ab;margin-bottom:6px">'
        "&#8212; baseline (blue line)</div>\n" + "".join(bars) + "\n</div>"
    )


# ── experiment table ──────────────────────────────────────────────────────────


def _table_html(rows: list[dict]) -> str:
    cols = [
        "iter",
        "label",
        "dimension",
        "optim_flags",
        "opset",
        "screen_p50_ms",
        "median_p50_ms",
        "delta_pct",
        "cv",
        "status",
    ]
    hdrs = "".join(
        f'<th style="text-align:left;padding:6px 10px;font-size:10px;'
        f"text-transform:uppercase;letter-spacing:0.6px;color:#778;"
        f'border-bottom:2px solid #dde">{c.replace("_", " ")}</th>'
        for c in cols
    )
    trs = []
    for r in rows:
        status = r.get("status", "")
        bg = _status_bg(status)
        color = _status_color(status)
        cells = []
        for c in cols:
            val = html_lib.escape(str(r.get(c, "")))
            if c == "status":
                cells.append(
                    f'<td style="padding:5px 10px;font-size:11px;'
                    f'color:{color};font-weight:600">{val}</td>'
                )
            else:
                cells.append(f'<td style="padding:5px 10px;font-size:11px;color:#334">{val}</td>')
        trs.append(
            f'<tr style="background:{bg};border-bottom:1px solid #eef">' + "".join(cells) + "</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse">'
        f"<thead><tr>{hdrs}</tr></thead>"
        f"<tbody>{''.join(trs)}</tbody>"
        "</table>"
    )


# ── champion box ─────────────────────────────────────────────────────────────


def _champion_html(rows: list[dict], model_id: str, ep: str) -> str:
    keeps = [r for r in rows if r.get("status", "").lower().startswith("keep")]
    if not keeps:
        return (
            '<div style="background:#fff3e0;border:1.5px solid #ffcc80;border-radius:8px;'
            'padding:14px 18px;font-size:12px;color:#e65100">'
            "No KEEP verdict yet — search in progress.</div>"
        )
    best = min(keeps, key=lambda r: _p50_float(r.get("median_p50_ms")) or 999)
    flags = html_lib.escape(best.get("optim_flags", "(none)"))
    opset = html_lib.escape(str(best.get("opset", 17)))
    p50 = html_lib.escape(best.get("median_p50_ms", "N/A"))
    delta = html_lib.escape(best.get("delta_pct", "N/A"))
    label = html_lib.escape(best.get("label", "?"))
    return f"""
<div style="background:#e8f5e9;border:1.5px solid #a5d6a7;border-radius:8px;
            padding:14px 18px;font-size:12px">
  <div style="font-weight:700;font-size:13px;color:#1b5e20;margin-bottom:8px">
    Champion Config</div>
  <table style="border-collapse:collapse">
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">Model</td>
        <td style="font-family:monospace;font-size:11px">{html_lib.escape(model_id)}</td></tr>
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">EP</td>
        <td style="font-family:monospace;font-size:11px">{html_lib.escape(ep.upper())}</td></tr>
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">Hypothesis</td>
        <td style="font-size:11px">{label}</td></tr>
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">Optim flags</td>
        <td style="font-family:monospace;font-size:11px">{flags}</td></tr>
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">Opset</td>
        <td style="font-family:monospace;font-size:11px">{opset}</td></tr>
    <tr><td style="color:#778;padding:2px 12px 2px 0;font-size:11px">Median p50</td>
        <td style="font-size:11px;color:#2e7d32;font-weight:600">{p50} ms
          ({delta})</td></tr>
  </table>
</div>"""


# ── main entry ────────────────────────────────────────────────────────────────


def generate_report(
    results_tsv: Path,
    work_dir: Path,
    model_id: str,
    ep: str,
    insight_notes: list[str] | None = None,
) -> Path:
    """Generate report.html inside work_dir. Returns the output path."""
    rows = _load_tsv(results_tsv)
    out_path = work_dir / "report.html"

    # Find baseline p50 from h0 row
    baseline_p50: float | None = None
    for r in rows:
        if r.get("iter") == "0" or "baseline" in r.get("label", "").lower():
            baseline_p50 = _p50_float(r.get("median_p50_ms"))
            if baseline_p50:
                break

    chart = _bar_chart_html(rows, baseline_p50)
    table = _table_html(rows)
    champion = _champion_html(rows, model_id, ep)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_done = len(rows)
    n_keep = sum(1 for r in rows if r.get("status", "").lower().startswith("keep"))

    insight_section = ""
    if insight_notes:
        items = "".join(f"<li>{html_lib.escape(n)}</li>" for n in insight_notes)
        insight_section = f"""
<h3 style="font-size:13px;font-weight:700;margin:24px 0 8px">Phase 1 Insight Engine</h3>
<ul style="font-size:11px;color:#556;line-height:1.8;padding-left:18px">{items}</ul>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>autoconfig report — {html_lib.escape(model_id)} ({ep.upper()})</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f4f6f9; padding: 28px 24px; color: #1a1a2e; font-size: 13px; }}
h2 {{ font-size: 16px; font-weight: 700; margin-bottom: 4px; }}
h3 {{ font-size: 13px; font-weight: 700; margin: 24px 0 10px; }}
.meta {{ font-size: 11px; color: #778; margin-bottom: 24px; }}
.card {{ background: #fff; border-radius: 10px; padding: 18px 20px;
         border: 1.5px solid #dde; margin-bottom: 20px; }}
</style>
</head>
<body>

<h2>autoconfig — {html_lib.escape(model_id)}</h2>
<div class="meta">EP: {html_lib.escape(ep.upper())} &nbsp;&middot;&nbsp;
  {n_done} experiments &nbsp;&middot;&nbsp; {n_keep} KEEP &nbsp;&middot;&nbsp;
  Generated: {ts}</div>

<div class="card">
  {champion}
</div>

<div class="card">
  <h3 style="margin-top:0">Benchmark Chart (median p50)</h3>
  {chart}
</div>

{f'<div class="card">{insight_section}</div>' if insight_section else ""}

<div class="card">
  <h3 style="margin-top:0">All Experiments</h3>
  {table}
</div>

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  Report written: {out_path}")
    return out_path
