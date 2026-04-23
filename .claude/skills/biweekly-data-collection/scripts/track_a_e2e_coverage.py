"""
Track A: E2E Coverage Data Collection
Usage: python track_a_e2e_coverage.py <report_period_folder>
  e.g. python track_a_e2e_coverage.py "bi-weekly report/040826_001"

Fetches combined perf+accuracy eval_report.html files from the static reporting site,
writes per-EP CSVs and ep_coverage_analysis.md into <report_period_folder>/data/.

Site structure (as of 2026-04):
  merged_report.html embeds EP HTML paths (e.g. QNNExecutionProvider_NPU/0327/eval_report.html)
  Each eval_report.html has a combined DATA array with both perf and accuracy fields.
"""
import sys, re, json, csv, urllib.request, os
from collections import defaultdict
from datetime import date

if len(sys.argv) < 2:
    print("Usage: python track_a_e2e_coverage.py <report_period_folder>")
    sys.exit(1)

REPORT_DIR = sys.argv[1].rstrip("/\\")
DATA_DIR   = REPORT_DIR + "/data/"
BASE       = "https://icy-moss-029643d00.6.azurestaticapps.net/e2e_model_coverage_result/"

os.makedirs(DATA_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ModelKit-biweekly-report"})
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", errors="replace")

def normalize_acc(v):
    if v is None: return "N/A"
    if v in ("ACCURACY_PASS", "ACCURACY_AT_RISK"): return "PASS"
    if v == "ACCURACY_REGRESSION": return "REGRESSION"
    return v  # EVAL_ERROR or unknown

def fmt_delta(d):
    if d is None: return ""
    return f"{d * 100:.1f}%"

COLS = ["model_id", "task", "model_type", "status", "elapsed_s",
        "failure_class", "accuracy_verdict", "delta_relative"]

EP_KEY_MAP = {
    "QNNExecutionProvider_NPU":       "qnn",
    "OpenVINOExecutionProvider_NPU":  "ov",
    "VitisAIExecutionProvider_NPU":   "vitisai",
}

# ── A1. Discover EP HTML paths from merged_report.html ─────────────────────────

print("Fetching merged_report.html ...")
merged_html = fetch_text(BASE + "merged_report.html")

# Find all eval_report.html paths referenced for each EP
# Pattern: QNNExecutionProvider_NPU/MMDD/eval_report.html
ep_html_paths = re.findall(
    r'((?:QNN|OpenVINO|VitisAI)ExecutionProvider_NPU/\d{4}/eval_report\.html)',
    merged_html
)
# Deduplicate while preserving order
seen = set()
ep_html_paths = [p for p in ep_html_paths if not (p in seen or seen.add(p))]
print(f"  EP HTML paths found: {ep_html_paths}")

if not ep_html_paths:
    print("ERROR: No EP eval_report.html paths found in merged_report.html")
    sys.exit(1)

# ── A2+A3. Fetch each eval_report.html and write CSV ──────────────────────────

ep_stats       = {}   # ep_key -> {csv_name, ep_name, mmdd, total, pass_count, rows}
ep_rows_by_key = {}   # ep_key -> list of row dicts

for path in ep_html_paths:
    # Derive EP name and key from path prefix
    ep_name_raw = path.split("/")[0]  # e.g. QNNExecutionProvider_NPU
    ep_key = EP_KEY_MAP.get(ep_name_raw, ep_name_raw.lower().replace("executionprovider_npu", ""))

    mmdd_m = re.search(r"/(\d{4})/", path)
    mmdd   = mmdd_m.group(1) if mmdd_m else "unknown"

    csv_name = f"{ep_key}_report_{mmdd}.csv"
    url      = BASE + path
    print(f"Fetching {path} ...")

    try:
        html = fetch_text(url)
        m = re.search(r'const\s+DATA\s*=\s*(\[[\s\S]*?\]);\s*\n', html)
        if not m:
            print(f"  WARNING: const DATA not found in {path}")
            continue
        data = json.loads(m.group(1))
        print(f"  {ep_name_raw}: {len(data)} records")

        rows = []
        for r in data:
            rows.append({
                "model_id":         r.get("hf_id", ""),
                "task":             r.get("task", ""),
                "model_type":       r.get("model_type", ""),
                "status":           "PASS" if r.get("passed") else "FAIL",
                "elapsed_s":        r.get("elapsed", ""),
                "failure_class":    r.get("failure_classification") or "",
                "accuracy_verdict": normalize_acc(r.get("accuracy_verdict")),
                "delta_relative":   fmt_delta(r.get("delta_relative")),
            })

        out_path = DATA_DIR + csv_name
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"  Written: {out_path} ({len(rows)} rows)")

        pass_count = sum(1 for row in rows if row["status"] == "PASS")
        ep_stats[ep_key] = {
            "csv_name":   csv_name,
            "ep_name":    ep_name_raw,
            "mmdd":       mmdd,
            "total":      len(rows),
            "pass_count": pass_count,
            "rows":       rows,
        }
        ep_rows_by_key[ep_key] = rows

    except Exception as e:
        print(f"  ERROR fetching {path}: {e}")

if len(ep_rows_by_key) < 3:
    print(f"\nWARNING: Only {len(ep_rows_by_key)} of 3 EP datasets available. "
          f"Three-EP intersection skipped.")

# ── A4. Cross-EP analysis ──────────────────────────────────────────────────────

def pass_set(rows):
    return set((r["model_id"], r["task"]) for r in rows if r["status"] == "PASS")

sets = {k: pass_set(v) for k, v in ep_rows_by_key.items()}

if len(sets) >= 3:
    vals = list(sets.values())
    all3 = vals[0] & vals[1] & vals[2]
else:
    all3 = set()

by_task = defaultdict(list)
for (m, t) in sorted(all3):
    by_task[t].append(m)

# Index rows for quick lookup
all_rows_by_key = {}
for k, rows in ep_rows_by_key.items():
    all_rows_by_key[k] = {(r["model_id"], r["task"]): r for r in rows}

# Accuracy counts for all-3 models (use QNN accuracy preferentially)
acc_counts = defaultdict(int)
regressions = []
for key in all3:
    row = (all_rows_by_key.get("qnn", {}).get(key)
           or all_rows_by_key.get("ov", {}).get(key)
           or all_rows_by_key.get("vitisai", {}).get(key)
           or {})
    v = row.get("accuracy_verdict", "N/A")
    acc_counts[v] += 1
    if v == "REGRESSION":
        regressions.append((key, row.get("delta_relative", "")))

# Partial coverage (2/3 EPs)
keys_list = list(ep_rows_by_key.keys())
partial = {}
if len(keys_list) >= 3:
    k0, k1, k2 = keys_list[0], keys_list[1], keys_list[2]
    partial[f"{k0}+{k1} only ({k2} bottleneck)"] = (sets[k0] & sets[k1]) - sets[k2]
    partial[f"{k0}+{k2} only ({k1} bottleneck)"] = (sets[k0] & sets[k2]) - sets[k1]
    partial[f"{k1}+{k2} only ({k0} bottleneck)"] = (sets[k1] & sets[k2]) - sets[k0]

# Zero-coverage tasks
all_tasks = set(r["task"] for rows in ep_rows_by_key.values() for r in rows)
zero_all3_tasks = sorted(t for t in all_tasks if not any(task == t for (_, task) in all3))

# ── A5. Write ep_coverage_analysis.md ─────────────────────────────────────────

today = date.today().isoformat()
total_tested = max((s["total"] for s in ep_stats.values()), default=0)

lines = []
lines.append("# E2E Model Coverage Analysis — Three-EP Intersection")
lines.append(f"**Report Date**: {today}")
lines.append("**Data Sources** (combined perf + accuracy per EP):")
for k, s in ep_stats.items():
    lines.append(f"- {k.upper()}: `{s['csv_name']}` ({s['ep_name']}, snapshot {s['mmdd']})")
if len(ep_rows_by_key) < 3:
    lines.append(f"\n> **WARNING**: Only {len(ep_rows_by_key)} of 3 EP datasets available. "
                 "Three-EP intersection not computed.")
lines.append("")

# Section 1
lines.append("## 1. Per-EP Summary Statistics")
lines.append("")
lines.append("| EP | Total | PASS | FAIL | Pass Rate |")
lines.append("|----|-------|------|------|-----------|")
for k, s in ep_stats.items():
    fail = s["total"] - s["pass_count"]
    rate = s["pass_count"] / s["total"] * 100 if s["total"] else 0
    lines.append(f"| {s['ep_name']} | {s['total']} | {s['pass_count']} | {fail} | {rate:.1f}% |")
if all3:
    lines.append(f"| **All Three EPs** | {total_tested} | {len(all3)} | — | {len(all3)/total_tested*100:.1f}% |")
lines.append("")

if ep_stats:
    bottleneck = min(ep_stats.values(), key=lambda s: s["pass_count"] / s["total"] if s["total"] else 0)
    lines.append(f"**Key observation**: Bottleneck EP is {bottleneck['ep_name']} "
                 f"({bottleneck['pass_count']}/{bottleneck['total']} = "
                 f"{bottleneck['pass_count']/bottleneck['total']*100:.1f}% pass rate).")
    lines.append("")

# Section 2
lines.append("## 2. Models Passing All Three EPs")
lines.append("")
if not all3:
    lines.append("_Insufficient data — fewer than 3 EPs available._")
else:
    lines.append(f"**{len(all3)} model-task combinations pass all three EPs.**")
    lines.append("")
    NLP_TASKS = {"text-classification", "token-classification", "fill-mask",
                 "question-answering", "text-generation", "summarization",
                 "translation", "feature-extraction", "sentence-similarity",
                 "zero-shot-classification"}
    nlp_tasks = sorted(t for t in by_task if t in NLP_TASKS)
    cv_tasks  = sorted(t for t in by_task if t not in NLP_TASKS)
    for group_name, tasks in [("NLP Tasks", nlp_tasks), ("Computer Vision Tasks", cv_tasks)]:
        if not tasks: continue
        lines.append(f"### {group_name}")
        lines.append("")
        for task in tasks:
            lines.append(f"#### {task} ({len(by_task[task])} models)")
            lines.append("")
            ep_cols = " | ".join(k.upper() for k in ep_rows_by_key)
            lines.append(f"| model_id | {ep_cols} | Accuracy | Delta |")
            lines.append(f"|----------{'|---' * len(ep_rows_by_key)}|----------|-------|")
            for m in sorted(by_task[task]):
                ep_marks = " | ".join("✅ PASS" for _ in ep_rows_by_key)
                row = (all_rows_by_key.get("qnn", {}).get((m, task))
                       or all_rows_by_key.get(list(ep_rows_by_key.keys())[0], {}).get((m, task))
                       or {})
                acc  = row.get("accuracy_verdict", "N/A")
                delt = row.get("delta_relative", "")
                lines.append(f"| {m} | {ep_marks} | {acc} | {delt} |")
            lines.append("")

# Section 3
lines.append("## 3. Pass Rate by Task Category")
lines.append("")
lines.append("| Task | All-3 Pass | Accuracy PASS | Accuracy REGRESSION | N/A |")
lines.append("|------|-----------|---------------|---------------------|-----|")
for task in sorted(all_tasks):
    a3 = len([k for k in all3 if k[1] == task])
    acc_pass = acc_reg = acc_na = 0
    for key in all3:
        if key[1] != task: continue
        row = (all_rows_by_key.get("qnn", {}).get(key) or {})
        v = row.get("accuracy_verdict", "N/A")
        if v == "PASS": acc_pass += 1
        elif v == "REGRESSION": acc_reg += 1
        else: acc_na += 1
    mark = "" if a3 > 0 else " ❌"
    lines.append(f"| {task}{mark} | {a3} | {acc_pass} | {acc_reg} | {acc_na} |")
lines.append("")

if zero_all3_tasks:
    lines.append(f"**Zero-coverage tasks**: {', '.join(zero_all3_tasks)}")
    lines.append("")

# Section 4
lines.append("## 4. Accuracy Summary (all-3 EP PASS models)")
lines.append("")
lines.append("| Metric | Count |")
lines.append("|--------|-------|")
for v in ("PASS", "REGRESSION", "EVAL_ERROR", "N/A"):
    lines.append(f"| {v} | {acc_counts.get(v, 0)} |")
lines.append("")
if regressions:
    lines.append("### Accuracy Regressions")
    lines.append("")
    lines.append("| Model | Task | Delta |")
    lines.append("|-------|------|-------|")
    for ((m, t), delta) in sorted(regressions, key=lambda x: x[1]):
        lines.append(f"| {m} | {t} | {delta} |")
    lines.append("")

# Section 5
lines.append("## 5. Architecture Pattern Analysis")
lines.append("")
lines.append("_[Encoder-only Transformer / ViT families tend to pass; "
             "decoder / generative models fail due to dynamic shapes and attention op coverage.]_")
lines.append("")

# Section 6
lines.append("## 6. Notable Gaps and Failure Patterns")
lines.append("")
lines.append("_[Fill in from manual review of FAIL rows and zero-coverage tasks above.]_")
lines.append("")

# Section 7
lines.append("## 7. Models Passing Exactly 2 EPs (Partial Coverage)")
lines.append("")
for label, s in partial.items():
    lines.append(f"### {label} ({len(s)} combinations)")
    lines.append("")
    for (m, t) in sorted(s):
        lines.append(f"- {m} / {t}")
    lines.append("")

# Section 8
lines.append("## 8. Implications for Milestone Target")
lines.append("")
lines.append("_[Fill in based on team milestone targets and current pace.]_")
lines.append("")

# Section 9
lines.append("## 9. Summary")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|--------|-------|")
lines.append(f"| Total model-task combinations tested | {total_tested} |")
lines.append(f"| Combinations passing all three EPs (perf) | "
             f"{len(all3)} ({len(all3)/total_tested*100:.1f}% of {total_tested}) |")
lines.append(f"| Of those: accuracy PASS | {acc_counts.get('PASS', 0)} |")
lines.append(f"| Of those: accuracy REGRESSION | {acc_counts.get('REGRESSION', 0)} |")
lines.append(f"| Of those: accuracy N/A | {acc_counts.get('N/A', 0)} |")
if ep_stats:
    lines.append(f"| Primary bottleneck EP | {bottleneck['ep_name']} "
                 f"({bottleneck['pass_count']/bottleneck['total']*100:.1f}%) |")
lines.append("")

out_md = DATA_DIR + "ep_coverage_analysis.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nWritten: {out_md}")
print("Track A complete.")
