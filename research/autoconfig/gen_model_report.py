#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Generate per-model HTML optimization reports from autoconfig sweep results."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
CHART_MIN_GAIN = -200.0
CHART_MAX_GAIN = 200.0


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} ms"


def _fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def _status_class(gain_pct: float | None) -> str:
    if gain_pct is None:
        return "neutral"
    if gain_pct > 0:
        return "good"
    if gain_pct < 0:
        return "bad"
    return "neutral"


def _short_label(label: str, max_len: int = 26) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "…"


def _sort_hypothesis_ids(hyp_id: str) -> tuple[int, str]:
    if hyp_id.startswith("h"):
        try:
            return int(hyp_id[1:]), hyp_id
        except ValueError:
            pass
    return 9999, hyp_id


def _get_p50(hyp: dict) -> float | None:
    """Get median p50 from either nested (QNN/CPU) or flat (GPU) schema."""
    if "full" in hyp:
        return hyp["full"].get("median_p50_ms")
    return hyp.get("median_p50_ms") or hyp.get("overall_median_p50_ms")


def _get_runs(hyp: dict) -> list[float]:
    if "full" in hyp:
        return [float(v) for v in hyp.get("all_p50s_ms") or hyp.get("full", {}).get("p50s_ms", [])]
    return [float(v) for v in hyp.get("all_p50s_ms") or hyp.get("full_p50s_ms", [])]


def _get_gain_pct(hyp_id: str, hyp: dict, baseline_p50_ms: float | None) -> float | None:
    if hyp_id == "h0" and baseline_p50_ms is not None:
        return 0.0
    for key in ("overall_gain_pct", "confirm_overall_gain_pct", "gain_vs_baseline_pct"):
        value = hyp.get(key)
        if value is not None:
            return float(value)
    p50 = _get_p50(hyp)
    if baseline_p50_ms and p50:
        return (baseline_p50_ms - p50) / baseline_p50_ms * 100
    return None


def _format_extra_optim(extra_optim: dict | None) -> str:
    if not extra_optim:
        return "autoconf defaults"
    enabled = [key for key, value in extra_optim.items() if value]
    return ", ".join(enabled) if enabled else "autoconf defaults"


def _format_champion_config(hyp: dict) -> str:
    opset = hyp.get("opset")
    flags = _format_extra_optim(hyp.get("extra_optim"))
    if opset is None:
        return flags
    if flags == "autoconf defaults":
        return f"opset {opset} + autoconf defaults"
    return f"opset {opset} + {flags}"


def _confidence_text(hyp_id: str, hyp: dict, baseline_runs: list[float]) -> str:
    status = str(hyp.get("status", ""))
    verdict = str(hyp.get("verdict", ""))

    if status.startswith("BUILD"):
        return "build failed"
    if status == "BENCH_FAIL":
        return "bench failed"
    if status.startswith("SKIPPED"):
        return "guarded skip"
    if hyp.get("confirm_verdict") == "CONFIRMED":
        return "ranges separated"
    if hyp.get("confirm_verdict") == "MARGINAL_UNCONFIRMED":
        return "ranges overlap"
    if verdict == "KEEP_CONFIRMED":
        wins = hyp.get("sessions_above_threshold")
        total = hyp.get("total_sessions")
        if wins is not None and total is not None:
            return f"{wins}/{total} sessions confirm"
        return "confirmation passed"
    if verdict == "MARGINAL_UNCONFIRMED":
        wins = hyp.get("sessions_above_threshold")
        total = hyp.get("total_sessions")
        if wins is not None and total is not None:
            return f"{wins}/{total} sessions confirm"
        return "confirmation incomplete"

    runs = _get_runs(hyp)
    if baseline_runs and runs:
        if max(runs) < min(baseline_runs) or min(runs) > max(baseline_runs):
            return "ranges separated"
        return "ranges overlap"

    if hyp_id == "h0":
        return "baseline reference"
    return "single-point only"


def _table_rows(
    hyps: list[tuple[str, dict]],
    baseline_p50_ms: float | None,
    champion_hyp: str | None,
    predicate,
) -> list[dict]:
    rows: list[dict] = []
    baseline_runs = _get_runs(dict(hyps).get("h0", {}))
    for hyp_id, hyp in hyps:
        gain_pct = _get_gain_pct(hyp_id, hyp, baseline_p50_ms)
        status = str(hyp.get("status", ""))
        verdict = str(hyp.get("verdict") or hyp.get("confirm_verdict") or status or "—")
        row = {
            "hyp_id": hyp_id,
            "label": hyp.get("label", ""),
            "gain_pct": gain_pct,
            "verdict": verdict,
            "confidence": _confidence_text(hyp_id, hyp, baseline_runs),
            "status": status,
            "is_champion": hyp_id == champion_hyp,
        }
        if predicate(row, hyp):
            rows.append(row)
    return rows


def _render_table(title: str, icon: str, rows: list[dict], champion_hyp: str | None) -> str:
    if not rows:
        return ""

    table_rows = []
    for row in rows:
        champion_class = " champion-row" if row["hyp_id"] == champion_hyp else ""
        gain_style = (
            "gain-neg" if row["gain_pct"] is not None and row["gain_pct"] < 0 else "gain-pos"
        )
        table_rows.append(
            f"""
            <tr class="{champion_class.strip()}">
              <td><span class="hyp-pill">{_escape(row["hyp_id"])}</span></td>
              <td>{_escape(row["label"])}</td>
              <td class="{gain_style}">{_fmt_pct(row["gain_pct"])}</td>
              <td>{_escape(row["verdict"])}</td>
              <td>{_escape(row["confidence"])}</td>
            </tr>
            """
        )

    return f"""
    <section class="section-card">
      <div class="section-title">{icon} {title}</div>
      <table class="report-table">
        <thead>
          <tr>
            <th>Hypothesis</th>
            <th>Label</th>
            <th>Gain %</th>
            <th>Verdict</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {"".join(table_rows)}
        </tbody>
      </table>
    </section>
    """


def _render_characteristics(results: dict) -> str:
    rows = [
        ("Model ID", results.get("model_id")),
        ("Task", results.get("task")),
        ("Arch type", results.get("model_type")),
        ("Baseline opset", results.get("baseline_opset")),
        ("EP", results.get("ep")),
        ("Device", results.get("device")),
    ]

    conv_pct = results.get("conv_pct")
    if "npu006_risk" in results:
        conv_text = "N/A" if conv_pct is None else f"{conv_pct:.1f}%"
        rows.append(("Conv%", conv_text))
        rows.append(("npu-006 risk", "HIGH" if results.get("npu006_risk") else "LOW"))

    if "npu001_generalized" in results:
        rows.append(("npu-001 note", results.get("npu001_generalized")))

    cells = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_escape(value if value is not None else '—')}</td></tr>"
        for label, value in rows
    )
    return f"""
    <section class="section-card">
      <div class="section-title">Model Characteristics</div>
      <table class="characteristics-table">
        {cells}
      </table>
    </section>
    """


def _chart_bar_color(gain_pct: float | None) -> str:
    if gain_pct is None:
        return "#90a4ae"
    if gain_pct > 5:
        return "#43a047"
    if gain_pct < -5:
        return "#e53935"
    return "#90a4ae"


def _render_chart(
    hyps: list[tuple[str, dict]], baseline_p50_ms: float | None, champion_hyp: str | None
) -> str:
    row_h = 40
    header_h = 48
    footer_h = 26
    label_w = 150
    bar_w = 520
    value_w = 78
    total_w = label_w + bar_w + value_w
    total_h = header_h + footer_h + len(hyps) * row_h
    center_x = label_w + bar_w / 2

    elements: list[str] = [
        f'<svg class="chart-svg" viewBox="0 0 {total_w} {total_h}" role="img" '
        'aria-label="Hypothesis gain chart">',
        "<defs>",
        '<pattern id="buildFailPattern" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">',
        '<rect width="8" height="8" fill="#cfd8dc"></rect>',
        '<rect width="3" height="8" fill="#90a4ae"></rect>',
        "</pattern>",
        "</defs>",
        '<text x="0" y="20" class="axis-label">Hypothesis</text>',
        f'<text x="{label_w}" y="20" class="axis-label">Gain vs baseline (%)</text>',
        f'<line x1="{center_x:.1f}" y1="{header_h - 8}" x2="{center_x:.1f}" y2="{total_h - footer_h}" class="center-line" />',
    ]

    for tick in (-200, -100, 0, 100, 200):
        x = label_w + ((tick - CHART_MIN_GAIN) / (CHART_MAX_GAIN - CHART_MIN_GAIN)) * bar_w
        elements.append(
            f'<line x1="{x:.1f}" y1="{header_h - 4}" x2="{x:.1f}" y2="{total_h - footer_h}" class="tick-line" />'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{total_h - 6}" text-anchor="middle" class="tick-label">{tick}%</text>'
        )

    for idx, (hyp_id, hyp) in enumerate(hyps):
        y = header_h + idx * row_h
        bar_mid = y + row_h / 2
        bar_top = y + 8
        bar_height = row_h - 16
        gain_pct = _get_gain_pct(hyp_id, hyp, baseline_p50_ms)
        clipped_gain = (
            None if gain_pct is None else max(min(gain_pct, CHART_MAX_GAIN), CHART_MIN_GAIN)
        )
        status = str(hyp.get("status", ""))
        verdict = str(hyp.get("verdict") or hyp.get("confirm_verdict") or "")
        p50 = _get_p50(hyp)
        title = (
            f"{hyp_id}: {hyp.get('label', '')}\n"
            f"status={status or '—'}  verdict={verdict or '—'}\n"
            f"p50={_fmt_ms(p50)}  gain={_fmt_pct(gain_pct)}"
        )

        elements.append(f"<g><title>{_escape(title)}</title>")
        elements.append(
            f'<rect x="0" y="{y:.1f}" width="{total_w}" height="{row_h}" class="row-bg" />'
        )
        elements.append(
            f'<text x="8" y="{y + 16:.1f}" class="hyp-label">{_escape(hyp_id)}</text>'
            f'<text x="8" y="{y + 29:.1f}" class="hyp-sub">{_escape(_short_label(str(hyp.get("label", ""))))}</text>'
        )

        if hyp_id == "h0":
            elements.append(
                f'<line x1="{center_x:.1f}" y1="{bar_top:.1f}" x2="{center_x:.1f}" '
                f'y2="{bar_top + bar_height:.1f}" class="baseline-bar" />'
            )
            elements.append(
                f'<text x="{center_x + 8:.1f}" y="{bar_mid + 4:.1f}" text-anchor="start" class="value-text">0.0%</text>'
            )
        elif status.startswith("BUILD"):
            fail_w = 92
            fail_x = center_x - fail_w / 2
            stroke = "#1e88e5" if hyp_id == champion_hyp else "#78909c"
            stroke_w = 4 if hyp_id == champion_hyp else 1.5
            elements.append(
                f'<rect x="{fail_x:.1f}" y="{bar_top:.1f}" width="{fail_w}" height="{bar_height}" '
                f'fill="url(#buildFailPattern)" stroke="{stroke}" stroke-width="{stroke_w}" rx="4" />'
            )
            elements.append(
                f'<text x="{center_x:.1f}" y="{bar_mid + 4:.1f}" text-anchor="middle" class="build-fail-text">'
                "BUILD_FAIL</text>"
            )
        elif clipped_gain is not None:
            target_x = (
                label_w
                + ((clipped_gain - CHART_MIN_GAIN) / (CHART_MAX_GAIN - CHART_MIN_GAIN)) * bar_w
            )
            x = min(center_x, target_x)
            width = max(abs(target_x - center_x), 2.0)
            stroke = "#1e88e5" if hyp_id == champion_hyp else "none"
            stroke_w = 4 if hyp_id == champion_hyp else 0
            value_x = target_x + 8 if clipped_gain >= 0 else target_x - 8
            anchor = "start" if clipped_gain >= 0 else "end"
            elements.append(
                f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{width:.1f}" height="{bar_height}" '
                f'fill="{_chart_bar_color(gain_pct)}" stroke="{stroke}" stroke-width="{stroke_w}" rx="4" />'
            )
            elements.append(
                f'<text x="{value_x:.1f}" y="{bar_mid + 4:.1f}" text-anchor="{anchor}" class="value-text">'
                f"{_escape(_fmt_pct(gain_pct))}</text>"
            )

        elements.append("</g>")

    elements.append("</svg>")
    return f"""
    <section class="section-card">
      <div class="section-title">Hypothesis Gain Chart</div>
      <div class="chart-wrap">
        {"".join(elements)}
      </div>
    </section>
    """


def _render_all_hypotheses(
    hyps: list[tuple[str, dict]],
    baseline_p50_ms: float | None,
    champion_hyp: str | None,
) -> str:
    """Full hypothesis table with opset, flags, all session p50s, and verdict."""
    baseline_runs = _get_runs(dict(hyps).get("h0", {}))
    rows: list[str] = []

    for hyp_id, hyp in hyps:
        status = str(hyp.get("status", ""))
        verdict = str(hyp.get("verdict") or hyp.get("confirm_verdict") or status or "—")
        label = hyp.get("label", "")
        opset = hyp.get("opset", "—")
        extra_optim = hyp.get("extra_optim")
        gain_pct = _get_gain_pct(hyp_id, hyp, baseline_p50_ms)
        p50 = _get_p50(hyp)
        all_runs = _get_runs(hyp)

        is_champion = hyp_id == champion_hyp
        row_class = "champion-row" if is_champion else ""

        # Format extra_optim flags
        if extra_optim:
            enabled = [k for k, v in extra_optim.items() if v]
            flags_str = (
                ", ".join(f'<span class="flag-pill">{_escape(f)}</span>' for f in enabled)
                if enabled
                else '<em style="color:#aaa">none</em>'
            )
        else:
            # Not stored — parse from label as fallback
            flags_str = '<em style="color:#bbb">not stored</em>'

        # Format all session p50s
        if all_runs:
            runs_html = " · ".join(f"{r:.2f}" for r in all_runs)
            runs_cell = f'<span class="runs-val">[{runs_html}]</span>'
        elif status.startswith("BUILD"):
            runs_cell = f'<span style="color:#c62828;font-weight:700">{_escape(status)}</span>'
        else:
            runs_cell = "—"

        # p50 cell
        p50_cell = _fmt_ms(p50) if p50 else ("—" if not status.startswith("BUILD") else status)

        # gain cell
        if gain_pct is not None:
            gain_class = "gain-pos" if gain_pct > 0 else ("gain-neg" if gain_pct < 0 else "")
            gain_cell = f'<span class="{gain_class}">{_fmt_pct(gain_pct)}</span>'
        else:
            gain_cell = "—"

        # verdict / confidence
        verdict_class = (
            "verdict-keep"
            if "KEEP" in verdict.upper()
            else "verdict-discard"
            if (
                "DISCARD" in verdict.upper()
                or "BUILD" in verdict.upper()
                or "FAIL" in verdict.upper()
            )
            else ""
        )
        conf = _confidence_text(hyp_id, hyp, baseline_runs)
        champion_star = (
            ' <span style="color:#1976d2;font-weight:900">★</span>' if is_champion else ""
        )

        rows.append(f"""
        <tr class="{row_class}">
          <td><span class="hyp-pill">{_escape(hyp_id)}</span>{champion_star}</td>
          <td class="label-cell">{_escape(label)}</td>
          <td class="opset-cell">{_escape(str(opset))}</td>
          <td class="flags-cell">{flags_str}</td>
          <td class="p50-cell">{_escape(p50_cell)}</td>
          <td class="sessions-cell">{runs_cell}</td>
          <td>{gain_cell}</td>
          <td><span class="{verdict_class}">{_escape(verdict)}</span></td>
          <td class="conf-cell">{_escape(conf)}</td>
        </tr>""")

    return f"""
    <section class="section-card">
      <div class="section-title">🔬 All Hypotheses — Full Detail</div>
      <div style="overflow-x:auto">
      <table class="report-table hyp-detail-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Config Label</th>
            <th>Opset</th>
            <th>Extra Flags</th>
            <th>Median p50</th>
            <th>Session p50s (ms)</th>
            <th>Gain %</th>
            <th>Verdict</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
      </div>
      <div style="margin-top:10px;font-size:11px;color:#7b8794">
        ★ = champion hypothesis &nbsp;·&nbsp; Session p50s are individual bench sessions (median used for comparison)
      </div>
    </section>
    """


def _render_feature_gaps(results: dict) -> str:
    feature_gaps = results.get("feature_gaps") or []
    if not feature_gaps:
        return ""

    cards = "".join(f'<div class="gap-card">{_escape(gap)}</div>' for gap in feature_gaps)
    return f"""
    <section class="section-card">
      <div class="section-title">Feature Gaps</div>
      <div class="gap-grid">{cards}</div>
    </section>
    """


def generate_model_report(results: dict, output_path: Path) -> None:
    """Generate a single self-contained HTML report."""
    hypotheses_map = results.get("hypotheses", {})
    hyps = sorted(hypotheses_map.items(), key=lambda item: _sort_hypothesis_ids(item[0]))
    baseline_p50_ms = results.get("baseline_p50_ms")
    champion_hyp = results.get("best_hypothesis")
    champion = hypotheses_map.get(champion_hyp or "", {})
    champion_p50_ms = results.get("best_p50_ms") or _get_p50(champion)
    best_gain_pct = results.get("best_gain_pct")

    keep_rows = _table_rows(
        hyps,
        baseline_p50_ms,
        champion_hyp,
        lambda row, _: (row["gain_pct"] is not None and row["gain_pct"] > 5)
        or row["verdict"] == "KEEP_CONFIRMED",
    )
    discard_rows = _table_rows(
        hyps,
        baseline_p50_ms,
        champion_hyp,
        lambda row, hyp: row["status"].startswith("BUILD")
        or (row["gain_pct"] is not None and row["gain_pct"] < -2),
    )
    neutral_rows = _table_rows(
        hyps,
        baseline_p50_ms,
        champion_hyp,
        lambda row, hyp: row not in keep_rows and row not in discard_rows,
    )

    sweep_ts = results.get("timestamp")
    sweep_date = (
        sweep_ts.split("T", 1)[0] if isinstance(sweep_ts, str) and "T" in sweep_ts else sweep_ts
    )
    header_title = (
        f"{str(results.get('ep', 'unknown')).upper()} {str(results.get('device', 'unknown')).upper()} "
        f"Optimization Report — {results.get('model_id', 'unknown')}"
    )
    subtitle = (
        f"{results.get('model_type', 'unknown')} arch · {sweep_date or 'unknown date'} · "
        f"{len(hyps)} hypotheses tested"
    )
    baseline_delta_ms = None
    if baseline_p50_ms is not None and champion_p50_ms is not None:
        baseline_delta_ms = baseline_p50_ms - champion_p50_ms

    keep_count = len(keep_rows)
    discard_count = len(discard_rows)
    champion_summary = _format_champion_config(champion) if champion else "—"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_escape(header_title)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f9;
      color: #1a1a2e;
      padding: 28px 24px 40px;
      font-size: 13px;
      line-height: 1.5;
    }}
    h1 {{ font-size: 24px; font-weight: 800; margin-bottom: 6px; color: #102a43; }}
    .subtitle {{ color: #5f6c80; font-size: 12px; margin-bottom: 24px; }}
    .section-card {{
      background: #ffffff;
      border: 1.5px solid #dbe5f0;
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(16, 42, 67, 0.06);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .kpi-card {{
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
      border: 1.5px solid #dbe5f0;
      border-radius: 12px;
      padding: 16px;
      min-height: 120px;
    }}
    .kpi-label {{
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: #6b7c93;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 23px;
      font-weight: 800;
      color: #102a43;
      line-height: 1.15;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    .kpi-card.good .kpi-value {{ color: #2e7d32; }}
    .kpi-card.bad .kpi-value {{ color: #c62828; }}
    .kpi-sub {{ color: #6b7c93; font-size: 11px; }}
    .section-title {{
      font-size: 14px;
      font-weight: 800;
      color: #102a43;
      margin-bottom: 14px;
    }}
    .characteristics-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .characteristics-table th,
    .characteristics-table td {{
      padding: 9px 10px;
      border-bottom: 1px solid #ebf1f6;
      text-align: left;
      vertical-align: top;
    }}
    .characteristics-table th {{
      width: 180px;
      color: #5f6c80;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.7px;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .report-table th {{
      text-align: left;
      padding: 9px 10px;
      background: #eef4fb;
      color: #486581;
      border-bottom: 2px solid #d9e2ec;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.7px;
    }}
    .report-table td {{
      padding: 10px;
      border-bottom: 1px solid #ebf1f6;
      vertical-align: top;
    }}
    .report-table tr:hover td {{ background: #f8fbff; }}
    .champion-row td {{ background: #e8f1fd; }}
    .hyp-pill {{
      display: inline-block;
      background: #102a43;
      color: white;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
    }}
    .gain-pos {{ color: #2e7d32; font-weight: 700; }}
    .gain-neg {{ color: #c62828; font-weight: 700; }}
    .chart-wrap {{
      overflow-x: auto;
      border: 1px solid #e6edf5;
      border-radius: 10px;
      background: #fbfdff;
      padding: 10px;
    }}
    .chart-svg {{ width: 100%; min-width: 760px; display: block; }}
    .axis-label {{ fill: #486581; font-size: 11px; font-weight: 700; }}
    .tick-label {{ fill: #7b8794; font-size: 10px; }}
    .tick-line {{ stroke: #d9e2ec; stroke-width: 1; }}
    .center-line {{ stroke: #1e88e5; stroke-width: 2; stroke-dasharray: 4 4; }}
    .row-bg {{ fill: transparent; }}
    .hyp-label {{ fill: #102a43; font-size: 12px; font-weight: 800; }}
    .hyp-sub {{ fill: #7b8794; font-size: 10px; }}
    .baseline-bar {{ stroke: #546e7a; stroke-width: 3; }}
    .value-text {{ fill: #102a43; font-size: 11px; font-weight: 700; }}
    .build-fail-text {{ fill: #37474f; font-size: 10px; font-weight: 800; letter-spacing: 0.5px; }}
    .gap-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .gap-card {{
      background: #fff8e1;
      border: 1.5px solid #ffe082;
      border-radius: 10px;
      padding: 12px 14px;
      color: #7c5b00;
      font-size: 12px;
    }}
    .flag-pill {{
      display: inline-block;
      background: #e3f2fd;
      color: #1565c0;
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      font-weight: 700;
      margin: 1px 2px 1px 0;
    }}
    .runs-val {{
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 10.5px;
      color: #546e7a;
      white-space: nowrap;
    }}
    .hyp-detail-table .label-cell {{ font-size: 11.5px; max-width: 220px; }}
    .hyp-detail-table .opset-cell {{ text-align: center; font-weight: 700; color: #3949ab; font-size: 12px; }}
    .hyp-detail-table .flags-cell {{ min-width: 140px; }}
    .hyp-detail-table .p50-cell {{ font-family: "Cascadia Code","Consolas",monospace; font-size: 12px; white-space: nowrap; }}
    .hyp-detail-table .sessions-cell {{ min-width: 160px; }}
    .hyp-detail-table .conf-cell {{ font-size: 11px; color: #546e7a; }}
    .verdict-keep {{ color: #2e7d32; font-weight: 700; }}
    .verdict-discard {{ color: #c62828; font-weight: 700; }}
    .footer {{
      margin-top: 16px;
      color: #7b8794;
      font-size: 11px;
      text-align: center;
    }}
    @media (max-width: 1200px) {{
      .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .kpi-grid {{ grid-template-columns: 1fr; }}
      body {{ padding: 18px 14px 28px; }}
    }}
  </style>
</head>
<body>
  <h1>{_escape(header_title)}</h1>
  <div class="subtitle">{_escape(subtitle)}</div>

  <section class="kpi-grid">
    <div class="kpi-card {_status_class(best_gain_pct)}">
      <div class="kpi-label">Best Gain %</div>
      <div class="kpi-value">{_fmt_pct(best_gain_pct)}</div>
      <div class="kpi-sub">Champion: {_escape(champion_hyp or "—")}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Baseline → Champion ms</div>
      <div class="kpi-value">{_escape(_fmt_ms(baseline_p50_ms))} → {_escape(_fmt_ms(champion_p50_ms))}</div>
      <div class="kpi-sub">Latency reduction: {_escape(_fmt_ms(baseline_delta_ms))}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">EP + Device</div>
      <div class="kpi-value">{_escape(str(results.get("ep", "unknown")).upper())} / {_escape(str(results.get("device", "unknown")).upper())}</div>
      <div class="kpi-sub">Baseline opset {_escape(results.get("baseline_opset", "—"))}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Champion Config</div>
      <div class="kpi-value">{_escape(champion_hyp or "—")}</div>
      <div class="kpi-sub">{_escape(champion_summary)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total experiments</div>
      <div class="kpi-value">{len(hyps)}</div>
      <div class="kpi-sub">{keep_count} KEEP / {discard_count} DISCARD</div>
    </div>
  </section>

  {_render_characteristics(results)}
  {_render_chart(hyps, baseline_p50_ms, champion_hyp)}
  {_render_all_hypotheses(hyps, baseline_p50_ms, champion_hyp)}
  {_render_table("Effective Optimizations", "✅", keep_rows, champion_hyp)}
  {_render_table("Ineffective or Harmful", "❌", discard_rows, champion_hyp)}
  {_render_table("Neutral / Build Fail", "⚪", neutral_rows, champion_hyp)}
  {_render_feature_gaps(results)}

  <div class="footer">Generated by gen_model_report.py · research/autoconfig</div>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")


def _load_results(results_path: Path) -> dict:
    return json.loads(results_path.read_text(encoding="utf-8"))


def _generate_for_results_file(results_path: Path) -> Path:
    results = _load_results(results_path)
    output_path = results_path.with_name("report.html")
    generate_model_report(results, output_path)
    return output_path


def _generate_for_sweep_dir(sweep_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for results_path in sorted(sweep_dir.rglob("results.json")):
        outputs.append(_generate_for_results_file(results_path))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate per-model autoconfig HTML report(s).")
    parser.add_argument("results_json", nargs="?", help="Path to a single results.json file")
    parser.add_argument(
        "--sweep-dir", help="Sweep directory containing per-model results.json files"
    )
    args = parser.parse_args()

    if bool(args.results_json) == bool(args.sweep_dir):
        parser.error("Provide exactly one of <results_json> or --sweep-dir.")

    if args.sweep_dir:
        sweep_dir = _resolve_path(args.sweep_dir)
        outputs = _generate_for_sweep_dir(sweep_dir)
        for output in outputs:
            print(output)
        return 0

    results_path = _resolve_path(args.results_json)
    output = _generate_for_results_file(results_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
