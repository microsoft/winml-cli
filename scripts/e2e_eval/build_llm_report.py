"""Generate a self-contained HTML report for a CPU LLM benchmark result."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import jsonschema


RESULT_LABEL = "llm_cpu_benchmark.json"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "llm_benchmark.schema.json"


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _number(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):,.{digits}f}"


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100 if mean else None


def _bar_rows(points: list[dict[str, Any]], key: str, digits: int) -> str:
    values = [float(point.get(key) or 0.0) for point in points]
    maximum = max(values, default=1.0) or 1.0
    rows = []
    for point, value in zip(points, values, strict=True):
        width = max(2.0, value / maximum * 100)
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{int(point["context_length_tokens"])} tok</span>'
            f'<span class="bar-track"><span class="bar" style="width:{width:.1f}%"></span></span>'
            f'<strong>{value:,.{digits}f}</strong>'
            "</div>"
        )
    return "".join(rows)


def render_html(result: dict[str, Any], title: str | None = None) -> str:
    """Render one CPU benchmark result as a standalone HTML document."""
    points = result.get("context_sweep") or []
    benchmark = result.get("benchmark") or {}
    environment = result.get("environment") or {}
    run = result.get("run") or {}
    heading = title or f'{result.get("model", "LLM")} CPU Benchmark'

    summary_rows = []
    detail_sections = []
    for point in points:
        raw = point.get("raw") or {}
        decode_samples = [float(value) for value in raw.get("decode_tokens_per_second") or []]
        summary_rows.append(
            "<tr>"
            f'<td>{int(point["context_length_tokens"]):,}</td>'
            f'<td>{int(point["prompt_tokens"]):,}</td>'
            f'<td>{int(point["generated_tokens"]):,}</td>'
            f'<td><strong>{_number(point.get("decode_tokens_per_second"), 2)}</strong></td>'
            f'<td>{_number(point.get("prefill_tokens_per_second"), 2)}</td>'
            f'<td>{_number(point.get("ttft_s"), 3)}</td>'
            f'<td>{_number(point.get("generation_compute_s"), 3)}</td>'
            f'<td>{_number(_cv(decode_samples), 2)}</td>'
            f'<td>{_number(point.get("process_cpu_avg_pct"), 1)}</td>'
            f'<td>{_number(point.get("process_memory_avg_mb"), 0)}</td>'
            "</tr>"
        )
        detail_sections.append(
            '<article class="detail">'
            f'<h3>{int(point["context_length_tokens"]):,}-token context</h3>'
            '<div class="metrics">'
            f'<div><span>Decode samples</span><strong>{_escape(", ".join(f"{value:.2f}" for value in decode_samples))}</strong></div>'
            f'<div><span>TTFT samples (ms)</span><strong>{_escape(", ".join(f"{float(value):.1f}" for value in raw.get("ttft_ms") or []))}</strong></div>'
            f'<div><span>Generation samples (ms)</span><strong>{_escape(", ".join(f"{float(value):.1f}" for value in raw.get("generation_compute_ms") or []))}</strong></div>'
            f'<div><span>Resource samples</span><strong>{int(point.get("resource_sample_count") or 0):,}</strong></div>'
            "</div></article>"
        )

    status = "PASS" if run.get("passed") else "FAIL"
    errors = run.get("errors") or []
    error_html = (
        '<section class="notice fail"><strong>Errors</strong><ul>'
        + "".join(f"<li>{_escape(error)}</li>" for error in errors)
        + "</ul></section>"
        if errors
        else ""
    )
    total_ram_gb = (
        float(environment["total_ram_mb"]) / 1024 if environment.get("total_ram_mb") else None
    )

    css = """
:root { --ink:#20262d; --muted:#68727d; --paper:#f4f5f2; --surface:#fff; --line:#d9ddd8; --accent:#087f8c; --pass:#2d7d46; --fail:#b43d32; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 "Aptos","Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:28px 22px 56px; }
header { border-top:5px solid var(--accent); padding:18px 0; }
h1 { margin:0 0 7px; font:600 30px Georgia,serif; }
.pills { display:flex; flex-wrap:wrap; gap:7px; }
.pill { padding:4px 9px; background:#e5eae5; font-size:12px; }
.pill.status { color:#fff; background:var(--pass); font-weight:700; }
.pill.status.fail { background:var(--fail); }
.grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); background:var(--surface); margin-bottom:18px; }
.fact { padding:13px 15px; border-right:1px solid var(--line); }
.fact:last-child { border-right:0; }
.fact span,.metric span { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }
.fact strong { display:block; margin-top:3px; }
.section { background:var(--surface); border:1px solid var(--line); margin-bottom:18px; }
.section h2 { margin:0; padding:14px 16px; border-bottom:1px solid var(--line); font:600 19px Georgia,serif; }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
th,td { padding:9px 10px; border-bottom:1px solid #e9ebe8; text-align:right; white-space:nowrap; }
th { background:#ecefeb; color:#4e5962; font-size:11px; text-transform:uppercase; }
th:first-child,td:first-child { text-align:left; }
.charts { display:grid; grid-template-columns:1fr 1fr; gap:22px; padding:17px; }
.chart h3 { margin:0 0 12px; font-size:14px; }
.bar-row { display:grid; grid-template-columns:75px 1fr 70px; gap:8px; align-items:center; margin:8px 0; }
.bar-label { font:11px Consolas,monospace; }
.bar-track { height:18px; background:#e7eae6; }
.bar { display:block; height:100%; background:var(--accent); }
.details { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; padding:16px; }
.detail { border:1px solid var(--line); padding:13px; }
.detail h3 { margin:0 0 10px; font-size:14px; }
.metrics { display:grid; gap:8px; }
.metric strong { font-family:Consolas,monospace; font-size:12px; overflow-wrap:anywhere; }
.notes { margin:0; padding:12px 34px 17px; }
.notice { margin-bottom:18px; padding:12px 15px; background:#fff4d6; border-left:4px solid #d9a927; }
.notice.fail { background:#fde9e7; border-color:var(--fail); }
footer { color:var(--muted); text-align:center; font-size:11px; }
@media (max-width:800px) { .grid,.charts { grid-template-columns:1fr; } .fact { border-right:0; border-bottom:1px solid var(--line); } }
"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(heading)}</title><style>{css}</style></head>
<body><main class="wrap">
<header><h1>{_escape(heading)}</h1><div class="pills"><span class="pill">{_escape(result.get("model"))}</span><span class="pill">{_escape(result.get("runtime"))}</span><span class="pill">{_escape(result.get("precision"))}</span><span class="pill">CPU</span><span class="pill status{' fail' if status == 'FAIL' else ''}">{status}</span></div></header>
{error_html}
<section class="grid"><div class="fact"><span>CPU</span><strong>{_escape(environment.get("cpu"))}</strong></div><div class="fact"><span>Logical cores</span><strong>{_escape(environment.get("logical_cores"))}</strong></div><div class="fact"><span>System RAM</span><strong>{_number(total_ram_gb, 1)} GB</strong></div><div class="fact"><span>Sampling</span><strong>{int(benchmark.get("iterations") or 0)} timed / {int(benchmark.get("warmup") or 0)} warmup</strong></div></section>
<section class="section"><h2>Summary</h2><div class="table-wrap"><table><thead><tr><th>Context</th><th>Prompt tok</th><th>Generated</th><th>Decode tok/s</th><th>Prefill tok/s</th><th>TTFT s</th><th>Generation s</th><th>Decode CV %</th><th>Process CPU %</th><th>Process RAM MB</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div></section>
<section class="section"><h2>Scaling</h2><div class="charts"><article class="chart"><h3>Decode throughput (higher is better)</h3>{_bar_rows(points, 'decode_tokens_per_second', 2)}</article><article class="chart"><h3>TTFT (lower is better)</h3>{_bar_rows(points, 'ttft_s', 3)}</article></div></section>
<section class="section"><h2>Raw samples</h2><div class="details">{''.join(detail_sections)}</div></section>
<section class="section"><h2>Method</h2><ul class="notes"><li>Each context uses deterministic repeated filler text tokenized to the exact target length.</li><li>Decode throughput excludes the first token. TTFT includes prefill and first-token compute.</li><li>Generation time excludes model loading, prompt encoding, warmup, detokenization, and report I/O.</li><li>Process CPU and RSS are full perf-process-tree averages. CPU percentage may exceed 100% on multicore systems.</li></ul></section>
<footer>Generated from {_escape(RESULT_LABEL)}</footer>
</main></body></html>"""


def _validate_result(result: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load benchmark result: {exc}")
        return 1
    try:
        _validate_result(result)
    except jsonschema.ValidationError as exc:
        print(f"Invalid benchmark result: {exc.message}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(result, args.title), encoding="utf-8")
    print(f"Report written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
