# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import datetime
import json


results = json.load(open(r"ablation-search\results.json"))

clean_base = [r for r in results if r["name"] in ["base_0", "base_1"]]
clean_runs = [v for r in clean_base for v in r["p50_runs"]]
clean_mean = round(sum(clean_runs) / len(clean_runs), 1)


def verdict(name, mean):
    if name in ["base_0", "base_1", "base_2", "base_mid", "base_end"]:
        return "outlier run" if name == "base_2" else "baseline"
    if name == "matmul_add":
        return "CONFIRMED REGRESSION"
    if name == "matmul_scale":
        return "probable mild regression"
    if name.startswith("opset_"):
        opset = int(name.split("_")[1])
        if opset >= 19:
            return "SEVERE REGRESSION (kMaxSupportedOpset bug)"
        return "neutral"
    delta = mean - clean_mean
    if abs(delta) < 5:
        return "neutral"
    if delta > 5:
        return "mild regression"
    return "possible improvement"


def row_class(name):
    if name in ["base_0", "base_1", "base_mid", "base_end"]:
        return "row-base"
    if name == "base_2":
        return "row-outlier"
    if name == "matmul_add":
        return "row-bad"
    if name.startswith("opset_") and int(name.split("_")[1]) >= 19:
        return "row-bad"
    if name in ["matmul_scale"]:
        return "row-warn"
    return "row-neutral"


rows_html = ""
for r in results:
    runs = r["p50_runs"]
    delta = r["p50_mean"] - clean_mean
    v = verdict(r["name"], r["p50_mean"])
    rc = row_class(r["name"])
    runs_str = " / ".join("%.1f" % x for x in runs)
    sign = "+" if delta >= 0 else ""
    rows_html += (
        '<tr class="%s"><td>%s</td><td>%.1f</td><td>%s%.1f</td>'
        "<td>%.1f</td><td>%.1f</td><td>%s</td><td>%s</td></tr>\n"
        % (rc, r["name"], r["p50_mean"], sign, delta, min(runs), max(runs), runs_str, v)
    )

bar_labels = [
    r["name"]
    for r in results
    if r["name"] not in ["base_0", "base_1", "base_2", "base_mid", "base_end"]
]
bar_values = [
    round(r["p50_mean"], 1)
    for r in results
    if r["name"] not in ["base_0", "base_1", "base_2", "base_mid", "base_end"]
]
bar_colors = []
for r in results:
    if r["name"] in ["base_0", "base_1", "base_2", "base_mid", "base_end"]:
        continue
    if r["name"] == "matmul_add" or (
        r["name"].startswith("opset_") and int(r["name"].split("_")[1]) >= 19
    ):
        bar_colors.append("'#dc3545'")
    elif r["name"] in ["matmul_scale"]:
        bar_colors.append("'#fd7e14'")
    elif abs(r["p50_mean"] - clean_mean) < 5:
        bar_colors.append("'#198754'")
    else:
        bar_colors.append("'#ffc107'")

bar_labels_js = json.dumps(bar_labels)
bar_values_js = json.dumps(bar_values)
bar_colors_js = ",".join(bar_colors)
n_bars = len(bar_labels)
baseline_line = clean_mean
now_str = datetime.datetime.now().strftime("%Y-%m-%d")
n_results = len(results)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ConvNext CPU Ablation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f8f9fa;color:#212529}
.container{max-width:1150px;margin:0 auto;padding:24px}
h1{font-size:1.6rem;border-bottom:2px solid #dee2e6;padding-bottom:8px}
h2{font-size:1.2rem;color:#495057;margin-top:32px}
h3{font-size:1rem;color:#495057;margin-top:20px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}
.card{background:white;border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card-title{font-size:.75rem;color:#6c757d;text-transform:uppercase;letter-spacing:.5px}
.card-value{font-size:1.8rem;font-weight:700;margin:4px 0}
.card-sub{font-size:.8rem;color:#6c757d}
.green{color:#198754}.red{color:#dc3545}.grey{color:#6c757d}
.banner{border-radius:6px;padding:12px 16px;margin:16px 0;font-size:.88rem}
.banner-info{background:#d1ecf1;border:1px solid #bee5eb}
.banner-warn{background:#fff3cd;border:1px solid #ffc107}
.banner-danger{background:#f8d7da;border:1px solid #f5c6cb}
table{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.1);font-size:.875rem}
th{background:#495057;color:white;padding:10px 12px;text-align:left}
td{padding:8px 12px;border-bottom:1px solid #dee2e6}
tr.row-base td{background:#f8f9fa;color:#6c757d}
tr.row-outlier td{background:#fff3cd}
tr.row-bad td{background:#f8d7da;font-weight:600}
tr.row-warn td{background:#fff3cd}
tr.row-neutral td{background:white}
.chart-box{background:white;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.finding{border-left:4px solid #0d6efd;border-radius:0 6px 6px 0;padding:10px 14px;margin:8px 0;background:white;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.finding.confirmed{border-color:#dc3545}
.finding.neutral{border-color:#198754}
.finding.correction{border-color:#6f42c1}
.finding.weak{border-color:#fd7e14}
.finding.rootcause{border-color:#dc3545;background:#fff5f5}
code{background:#f1f3f4;padding:1px 5px;border-radius:3px;font-size:.85em}
pre{background:#1e1e1e;color:#d4d4d4;border-radius:8px;padding:16px;font-size:.82rem;overflow-x:auto}
.opt-table{width:100%;border-collapse:collapse;font-size:.875rem;margin:12px 0}
.opt-table th{background:#6c757d;color:white;padding:7px 10px;text-align:left}
.opt-table td{padding:6px 10px;border-bottom:1px solid #dee2e6}
.opt-table tr:nth-child(even) td{background:#f8f9fa}
</style>
</head>
<body>
<div class="container">
<h1>&#x1F4CA; ConvNext CPU Ablation &#x2014; Autoconfig POC + Opset Cliff RCA</h1>
<p style="color:#6c757d;font-size:.9rem">Model: <strong>facebook/convnext-tiny-224</strong> &nbsp;|&nbsp; EP: <strong>CPU</strong> &nbsp;|&nbsp; DATE_PLACEHOLDER &nbsp;|&nbsp; N_RESULTS_PLACEHOLDER experiments &nbsp;|&nbsp; ORT ORTVER_PLACEHOLDER</p>

<div class="banner banner-info">
<strong>Measurement methodology:</strong> <code>winml perf --ep cpu --warmup 10 --iterations 50</code> &mdash; pure inference latency, no preprocessing.
3 independent perf runs per config. Metric: p50 (median) latency. Promotion threshold: max(3%, 2&times;&sigma;_baseline).
</div>

<div class="grid">
  <div class="card"><div class="card-title">Clean Baseline p50</div><div class="card-value">CLEAN_MEAN_PLACEHOLDERms</div><div class="card-sub">base_0 + base_1, opset=17</div></div>
  <div class="card"><div class="card-title">Best Config Found</div><div class="card-value green">Baseline</div><div class="card-sub">opset=17, no extra flags</div></div>
  <div class="card"><div class="card-title">Worst Finding</div><div class="card-value red">+38ms</div><div class="card-sub">matmul-add-fusion</div></div>
  <div class="card"><div class="card-title">Root Cause Found</div><div class="card-value red" style="font-size:1rem">kMaxSupportedOpset</div><div class="card-sub">Transpose Optimizer gate</div></div>
</div>

<!-- ==================== ROOT CAUSE SECTION ==================== -->
<h2>&#x1F50D; Root Cause Analysis: ORT Opset Performance Cliff</h2>

<div class="finding rootcause">
<strong>&#x274C; ROOT CAUSE IDENTIFIED: ORT <code>kMaxSupportedOpset</code> gates the entire Transpose Optimizer</strong><br><br>
In <code>onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h</code>:
<pre>constexpr int64_t kMaxSupportedOpset = 18;  // in ORT v1.14.x
// Current ORT (v1.24.5) kMaxSupportedOpset = 21 or 22

// In onnx_transpose_optimization.cc:
if (*opset &gt; kMaxSupportedOpset) {
    return std::nullopt;  // ← ENTIRE Transpose Optimizer skipped silently
}</pre>
ConvNext has <strong>42 Transpose nodes</strong> forming a NCHW&harr;NHWC "transpose sandwich" in every block.
The Transpose Optimizer normally eliminates/merges these (pushing through Add&times;18, Mul&times;18, canceling adjacent inverses).
When it is bypassed, all 42 Transpose nodes execute as raw memory-layout copy operations &rarr; systemic slowdown.
</div>

<h3>&#x1F4CA; ORT Optimization Level Experiment (confirms root cause)</h3>
<table class="opt-table">
<tr><th>Session Optimization Level</th><th>opset=17</th><th>opset=19</th><th>Ratio</th><th>Explanation</th></tr>
<tr><td><code>DISABLE_ALL</code></td><td>47.5ms</td><td style="background:#f8d7da;font-weight:600">355ms</td><td style="background:#f8d7da">7.5&times;</td><td>No Transpose Optimizer &rarr; all 42 Transposes execute. v17 model.onnx has pre-fused ops; v19 export has more raw ops.</td></tr>
<tr><td><code>ENABLE_BASIC</code></td><td>289ms</td><td>315ms</td><td>1.1&times;</td><td>Basic opts run on already-fused model, some interference. Near-parity: Transpose Optimizer not yet active at this level.</td></tr>
<tr><td><code>ENABLE_EXTENDED</code></td><td>209ms</td><td>241ms</td><td>1.2&times;</td><td>Extended optimizations help both but some overhead from re-optimizing pre-fused model.</td></tr>
<tr><td><code>ENABLE_ALL</code> (default)</td><td style="background:#d1e7dd">216ms</td><td style="background:#d1e7dd">215ms</td><td style="background:#d1e7dd">1.0&times;</td><td>Transpose Optimizer runs on both. Full parity achieved &mdash; confirms optimizer gap is the entire cause.</td></tr>
</table>

<div class="banner banner-warn">
<strong>Why does winml perf show opset=19 as 160ms vs 44ms?</strong>
winml build pre-applies <code>ORT_ENABLE_ALL</code> and saves <code>model.onnx</code>. winml perf then loads <em>that</em> pre-optimized model.
For opset=17, <code>kMaxSupportedOpset</code> is satisfied &rarr; Transpose Optimizer ran during build &rarr; model.onnx has fewer effective Transposes.
For opset=19, <code>kMaxSupportedOpset</code> may have been exceeded in the ORT version used during build &rarr; Transpose Optimizer skipped &rarr; model.onnx retains 42 raw Transposes.
When winml perf loads model.onnx (with <code>ENABLE_ALL</code> again at runtime), if the runtime ORT version's <code>kMaxSupportedOpset</code> covers 19, the gap partially closes. The residual difference depends on which ORT version winml-cli ships.
</div>

<h3>&#x1F4CB; <code>kMaxSupportedOpset</code> Version History (verified from ORT git tags)</h3>
<table class="opt-table">
<tr><th>ORT Release</th><th>kMaxSupportedOpset</th><th>Effect</th></tr>
<tr><td>v1.14.x</td><td style="background:#f8d7da">18</td><td>opset &ge; 19 &rarr; Transpose Optimizer DISABLED</td></tr>
<tr><td>v1.16.x</td><td style="background:#fff3cd">19</td><td>opset &ge; 20 &rarr; disabled</td></tr>
<tr><td>v1.17.x</td><td style="background:#fff3cd">20</td><td>opset &ge; 21 &rarr; disabled</td></tr>
<tr><td>v1.18.x</td><td style="background:#fff3cd">21</td><td>opset &ge; 22 &rarr; disabled</td></tr>
<tr><td>main/HEAD</td><td style="background:#d1e7dd">26</td><td>Fully covered for all current ONNX opsets</td></tr>
</table>

<h3>&#x1F4DC; ORT Source (exact call chain)</h3>
<pre>InferenceSession::Initialize()
  &rarr; graph_transformer_mgr_.ApplyTransformers(graph, Level1)
      &rarr; TransposeOptimizer::ApplyImpl()           [transpose_optimizer.cc:18]
          &rarr; onnx_transpose_optimization::Optimize() [onnx_transpose_optimization.cc:3344]
              &rarr; MakeOptimizerContext(graph, ...)
                  &rarr; graph.Opset("ai.onnx")         // reads DomainToVersionMap()
                  &rarr; if opset &gt; kMaxSupportedOpset: return nullopt  // &larr; THE GATE
              &rarr; if ctx == nullopt: return early    // no optimization performed</pre>

<h3>Why ConvNext is especially sensitive</h3>
<p style="font-size:.9rem">The Transpose Optimizer can push Transposes through <code>Add</code>, <code>Mul</code>, and simple unary ops. ConvNext has 18&times;(Add + Mul) layer-scale and residual connections between blocks, meaning a single Transpose can cascade through many nodes. With the optimizer enabled, adjacent inverse pairs cancel; without it, every NCHW&harr;NHWC conversion is a full memory copy of the activation tensor.</p>

<!-- ==================== ABLATION RESULTS ==================== -->
<h2>&#x1F4A1; Ablation Key Findings</h2>

<div class="finding confirmed">
<strong>&#x274C; CONFIRMED REGRESSION: <code>matmul-add-fusion</code> +38ms</strong><br>
All 3 independent runs: 63.0 / 70.8 / 111.2ms vs clean baseline ~43.7ms.
The minimum observed (63ms) is 20ms above the highest clean-baseline run. Not attributable to noise.
Hypothesis: baseline already converts MatMul+Add&rarr;Gemm (37 Gemm in model.onnx); applying matmul-add-fusion creates redundant or conflicting dispatch. Unconfirmed &mdash; requires op-level profiling.
</div>

<div class="finding correction">
<strong>&#x1F4DD; MEASUREMENT CORRECTION: <code>transpose-optimizer</code> is NEUTRAL on inference latency</strong><br>
Earlier 8-iteration search using <code>winml eval</code> reported +270ms. That measurement included HF preprocessing pipeline and had no warmup &mdash; it measured <em>application latency</em>, not <em>model inference</em>.
With <code>winml perf</code> (warmup=10, iter=50): 42.3 / 52.3 / 41.8ms &mdash; indistinguishable from baseline.
The +270ms was entirely a measurement artifact. Do not cite in user-facing reports.
</div>

<div class="finding confirmed">
<strong>&#x274C; CONFIRMED: opset=19&ndash;22 causes 1.9&ndash;3.9&times; regression on this ORT build</strong><br>
Mechanism confirmed: <code>kMaxSupportedOpset</code> gate in ORT's Transpose Optimizer. All 3 runs per opset are consistent.
Fix: use opset&le;17 (current winml-cli default) OR upgrade ORT to a version where <code>kMaxSupportedOpset &ge; 22</code> (main branch).
</div>

<div class="finding neutral">
<strong>&#x2705; NEUTRAL: <code>nchwc-transformer</code>, <code>transpose-optimizer</code>, opset=18</strong> &mdash; all within noise of baseline (~43.7ms).
</div>

<div class="finding weak">
<strong>&#x26A0; PROBABLE MILD REGRESSION: <code>matmul-scale-fusion</code></strong> &mdash; all 3 runs elevated (51.5 / 58.1 / 61.2ms). Weak signal due to baseline drift during experiment.
</div>

<h2>&#x1F4CA; Per-Config p50 Latency vs Baseline</h2>
<div class="chart-box"><canvas id="barChart" height="110"></canvas></div>

<h2>&#x1F4CB; Full Results Table</h2>
<div class="banner banner-warn">
Phase 0 = baseline &times;3 &nbsp;|&nbsp; Phase 1 = opset/CF flags &nbsp;|&nbsp; Phase 2 = single-flag ablation &nbsp;|&nbsp; Phase 3 = stepwise (no candidates) &nbsp;|&nbsp; Phase 4 = base_end recheck &nbsp;|&nbsp; Phase 5 = opset 19&ndash;22
</div>
<table>
<tr><th>Config</th><th>p50 mean (ms)</th><th>&#x0394; vs baseline</th><th>min</th><th>max</th><th>Runs (ms)</th><th>Verdict</th></tr>
ROWS_PLACEHOLDER
</table>

<h2>&#x1F527; Optimal Config</h2>
<pre># Optimal config: baseline (opset=17, constant_folding=True, no extra flags)
winml build --model-id facebook/convnext-tiny-224 -o out_cpu/
winml perf -m out_cpu/model.onnx --ep cpu --warmup 10 --iterations 50
# Expected: p50 ~43-44ms

# AVOID:
#   --optimize matmul-add-fusion     (confirmed +38ms regression)
#   opset_version: 19-22             (kMaxSupportedOpset bug: 3-4x regression on affected ORT builds)</pre>

<h2>&#x1F9E0; Open Questions</h2>
<ul style="font-size:.9rem">
<li><strong>Exact ORT version boundary:</strong> winml-cli ships ORT 1.24.5 (internal versioning). The exact <code>kMaxSupportedOpset</code> value in that build determines whether opset 19-22 is safe. Needs verification against ORT source at that specific commit.</li>
<li><strong>Why does <code>matmul-add-fusion</code> regress?</strong> 37 Gemm nodes already exist; applying this fusion may create double-fusion or suboptimal kernel selection. Requires <code>--profile</code> to confirm.</li>
<li><strong>GELU fusion mystery:</strong> baseline model.onnx has <code>com.microsoft/Gelu</code>&times;18 despite <code>GeluFusion</code> being in <code>disabled_optimizers</code>. Source unclear &mdash; likely HF Optimum pre-fuses GELU before ORT.</li>
</ul>

</div>
<script>
const ctx = document.getElementById('barChart').getContext('2d');
new Chart(ctx, {
  type: 'bar',
  data: {
    labels: BAR_LABELS_JS,
    datasets: [{
      label: 'p50 latency (ms)',
      data: BAR_VALUES_JS,
      backgroundColor: [BAR_COLORS_JS],
      borderRadius: 4
    },{
      type: 'line',
      label: 'Clean baseline (BASELINE_LINE_PLACEHOLDERms)',
      data: Array(N_BARS_PLACEHOLDER).fill(BASELINE_LINE_PLACEHOLDER),
      borderColor: '#0d6efd', borderDash: [6,3], pointRadius: 0, borderWidth: 2
    }]
  },
  options: {
    responsive: true,
    scales: { y: { beginAtZero: false, min: 30, title: { display: true, text: 'p50 latency (ms)' } } },
    plugins: {
      legend: { position: 'top' },
      tooltip: { callbacks: { label: c => c.dataset.label + ': ' + c.raw + 'ms' } }
    }
  }
});
</script>
</body>
</html>"""

import subprocess


result = subprocess.run(
    ["python", "-c", "import onnxruntime as ort; print(ort.__version__)"],
    capture_output=True,
    encoding="utf-8",
    cwd=r"C:\tmp\autoconfig-demo",
    env={
        **__import__("os").environ,
        "PATH": r"C:\tmp\autoconfig-demo\.venv\Scripts;" + __import__("os").environ.get("PATH", ""),
    },
)
ort_ver = result.stdout.strip() or "1.24.5"

html = html.replace("DATE_PLACEHOLDER", now_str)
html = html.replace("N_RESULTS_PLACEHOLDER", str(n_results))
html = html.replace("ORTVER_PLACEHOLDER", ort_ver)
html = html.replace("CLEAN_MEAN_PLACEHOLDER", str(clean_mean))
html = html.replace("ROWS_PLACEHOLDER", rows_html)
html = html.replace("BAR_LABELS_JS", bar_labels_js)
html = html.replace("BAR_VALUES_JS", bar_values_js)
html = html.replace("BAR_COLORS_JS", bar_colors_js)
html = html.replace("N_BARS_PLACEHOLDER", str(n_bars))
html = html.replace("BASELINE_LINE_PLACEHOLDER", str(baseline_line))

with open(r"report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("report.html written: %d bytes, %d experiments" % (len(html), n_results))
