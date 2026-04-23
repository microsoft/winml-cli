---
name: biweekly-data-collection
description: >
  Collects, organizes, and analyzes all source data for bi-weekly engineering reports.
  Covers two independent tracks: (1) E2E test coverage data pulled from the ModelKit
  static reporting site and saved as per-EP CSV files + ep_coverage_analysis.md in the
  data/ subfolder; (2) git-based explain files (by-pr, by-module, by-milestone) saved
  in the explain/ subfolder.
  Use this skill whenever the user wants to pull fresh E2E test results, refresh coverage
  data, regenerate explain files, or collect any report source data. Triggers include:
  "pull latest data", "refresh coverage data", "update test results", "re-pull the data",
  "update the data folder", "generate explain files", "collect report data".
  This skill must be run before biweekly-report whenever the data/ or explain/ folders
  are empty or stale.
---

# Bi-Weekly Data Collection

Two independent data collection tracks. Run both in parallel.

**Report period folder example**: `docs/bi-weekly report/032226_001/`

## Scripts

Both tracks have ready-to-run scripts in `scripts/`. Run with the Azure CLI Python:

```bash
PYTHON="/c/Program Files (x86)/Microsoft SDKs/Azure/CLI2/python.exe"
REPORT="docs/bi-weekly report/<MMDDYY_NNN>"

# Track A: fetch E2E coverage data → writes CSVs + ep_coverage_analysis.md
PYTHONIOENCODING=utf-8 "$PYTHON" .claude/skills/biweekly-data-collection/scripts/track_a_e2e_coverage.py "$REPORT"

# Track B (B4 only): fetch GitHub issues → writes issues-raw.md
PYTHONIOENCODING=utf-8 "$PYTHON" .claude/skills/biweekly-data-collection/scripts/track_b_collect_issues.py "$REPORT"
```

After running the scripts, complete Track B manually (B1–B3: git explain files; B5: rewrite milestone md files from issues-raw.md).

---

## Track A: E2E Coverage Data  →  `data/`

### A1. Discover EP Paths from merged_report.html

Base URL: `https://icy-moss-029643d00.6.azurestaticapps.net/e2e_model_coverage_result/`

First fetch the merged report page to discover current per-EP paths:

```
GET https://icy-moss-029643d00.6.azurestaticapps.net/e2e_model_coverage_result/merged_report.html
```

> **Site structure note (updated 2026-04)**: The page previously used `DATA_SOURCES`
> (perf JSON) + `ACC_CONFIGS` (accuracy HTML). It now uses **combined** per-EP HTML files
> that contain both perf and accuracy data in a single `DATA` array. The `EP_CONFIGS`
> variable lists EP labels only; the actual file paths are embedded as string literals
> matching the pattern below.

Search for all `eval_report.html` paths in the HTML:
```python
ep_html_paths = re.findall(
    r'((?:QNN|OpenVINO|VitisAI)ExecutionProvider_NPU/\d{4}/eval_report\.html)',
    merged_html
)
```

**EP HTML path pattern** (all under the same base URL):
```
QNNExecutionProvider_NPU/<MMDD>/eval_report.html
OpenVINOExecutionProvider_NPU/<MMDD>/eval_report.html
VitisAIExecutionProvider_NPU/<MMDD>/eval_report.html
```

Fetch all three HTML files from:
```
GET <base_url>/<ep_path>
```

### A2. Extract Combined DATA Array from Each eval_report.html

Each per-EP HTML contains **both perf and accuracy** in a single embedded JS array:
```js
const DATA = [ ... ];
```

Extract with:
```python
import re, json
m = re.search(r'const\s+DATA\s*=\s*(\[[\s\S]*?\]);\s*\n', html)
data = json.loads(m.group(1))
```

**Combined record schema** (fields in the `DATA` array):

| Field | Type | Notes |
|-------|------|-------|
| `hf_id` | string | HuggingFace model path → maps to `model_id` in CSV |
| `task` | string | HF task name |
| `model_type` | string | Architecture class |
| `group` | string | e.g. `"Top200"`, `"AITK"` |
| `priority` | string | `"P0"`, `"P1"`, `"P2"` |
| `passed` | bool | Perf pass/fail |
| `failure_classification` | string\|null | Failure category → `failure_class` in CSV |
| `elapsed` | float | Seconds → `elapsed_s` in CSV |
| `accuracy_verdict` | string\|null | See verdict table below |
| `delta_relative` | float\|null | Relative accuracy delta (negative = regression) |

No join needed — perf and accuracy are in the same record.

**`accuracy_verdict` raw values → normalized label:**

| Raw value | Normalized label | Meaning |
|-----------|-----------------|---------|
| `"ACCURACY_PASS"` | `PASS` | Within acceptable threshold |
| `"ACCURACY_AT_RISK"` | `PASS` | Slightly degraded but acceptable |
| `"ACCURACY_REGRESSION"` | `REGRESSION` | Exceeds regression threshold |
| `"EVAL_ERROR"` | `EVAL_ERROR` | Evaluation script failed |
| `null` | `N/A` | No accuracy data available |

Note: `ACCURACY_AT_RISK` is treated as `PASS` in rollups.

### A3. Write Per-EP CSV Files

Save to `<report_period>/data/` with naming convention `<ep>_report_MMDD.csv`,
where MMDD is the snapshot date from the perf JSON path (not today's date):

```
qnn_report_MMDD.csv
ov_report_MMDD.csv
vitisai_report_MMDD.csv
```

CSV header (case-sensitive, no spaces) — **8 columns** including accuracy:
```
model_id,task,model_type,status,elapsed_s,failure_class,accuracy_verdict,delta_relative
```

| Source | CSV column | Transformation |
|--------|------------|----------------|
| HTML `hf_id` | `model_id` | as-is |
| HTML `task` | `task` | as-is |
| HTML `model_type` | `model_type` | as-is |
| HTML `passed` | `status` | `true`→`"PASS"`, `false`→`"FAIL"` |
| HTML `elapsed` | `elapsed_s` | as-is |
| HTML `failure_classification` | `failure_class` | empty string if null |
| HTML `accuracy_verdict` | `accuracy_verdict` | normalize to `PASS`/`REGRESSION`/`EVAL_ERROR`/`N/A` |
| HTML `delta_relative` | `delta_relative` | as percentage string (e.g. `"-3.4%"`); empty if null |

No join required — perf and accuracy are combined in the same record.

### A4. Compute Cross-EP Coverage Analysis

Read all three CSVs and compute:

**Per-EP statistics:**
- Total rows, PASS count, FAIL count, pass rate %

**Three-EP intersection** — `(model_id, task)` pairs that are PASS in all three files:
```python
qnn_pass = set((r.model_id, r.task) for r in qnn if r.status == "PASS")
ov_pass  = set((r.model_id, r.task) for r in ov  if r.status == "PASS")
vai_pass = set((r.model_id, r.task) for r in vai if r.status == "PASS")
all_three = qnn_pass & ov_pass & vai_pass
```
Group the intersection by `task`, then by NLP vs. Computer Vision.

**Partial coverage (2/3 EPs):**
- QNN + VitisAI only (OV bottleneck)
- QNN + OV only (VitisAI bottleneck)
- OV + VitisAI only (QNN bottleneck)

**Zero-coverage tasks** — tasks where all_three count is 0; include root cause.

**Bottleneck EP** — the EP with the lowest pass rate; compute pp gap vs. the others.

### A5. Write ep_coverage_analysis.md

Save to `<report_period>/data/ep_coverage_analysis.md`. Structure:

```markdown
# E2E Model Coverage Analysis — Three-EP Intersection
**Report Date**: YYYY-MM-DD
**Data Sources**:
- QNN perf: `qnn_report_MMDD.csv` (QNNExecutionProvider_NPU, YYYY-MM-DD)
- OV perf: `ov_report_MMDD.csv` (OpenVINOExecutionProvider_NPU, YYYY-MM-DD)
- VitisAI perf: `vitisai_report_MMDD.csv` (VitisAIExecutionProvider_NPU, YYYY-MM-DD)
- Accuracy: QNNExecutionProvider_NPU/MMDD/eval_report.html (YYYY-MM-DD)
  [note which EPs have accuracy data; mark missing EPs as "N/A — not yet evaluated"]

## 1. Per-EP Summary Statistics
[table: EP | Total | PASS | FAIL | Pass Rate — including All Three row]
Key observation: [bottleneck EP name and pp gap]

## 2. Models Passing All Three EPs (N combinations)
[grouped by task — NLP first, then Computer Vision]
[each group: task heading + table: model_id | QNN | OV | VitisAI | Accuracy | Delta]
  - QNN/OV/VitisAI columns: ✅ PASS
  - Accuracy column: normalized verdict (PASS / REGRESSION / EVAL_ERROR / N/A)
  - Delta column: delta_relative as percentage (e.g. "-3.4%"), empty if N/A

## 3. Pass Rate by Task Category
[table: Task | All-3 Pass | Accuracy PASS | Accuracy REGRESSION | N/A]
[zero-coverage perf rows marked ❌ with root cause]

## 4. Accuracy Summary (models with perf PASS)
[table: Model | Task | Accuracy Verdict | Delta — sorted by delta_relative ascending]
[Count of REGRESSION models and worst delta_relative values]
[Note: ACCURACY_AT_RISK is counted as PASS]

## 5. Architecture Pattern Analysis
[why encoder-only Transformers / ViT families pass vs. why decoder models fail]

## 6. Notable Gaps and Failure Patterns
[table: failed task | root cause]
[near-miss: models at 2/3 EPs with reason for the missing EP]
[accuracy regressions: model | task | delta — models that pass perf but regress on accuracy]

## 7. Models Passing Exactly 2 EPs (Partial Coverage)
[three sub-sections by EP bottleneck]

## 8. Implications for Milestone Target
[table: priority | action | expected model count gain]
[projection: current pace vs. required pace]

## 9. Summary
[table: key metrics]
| Metric | Value |
| Total model-task combinations tested | N |
| Combinations passing all three EPs (perf) | N (X.X%) |
| Of those: accuracy PASS | N |
| Of those: accuracy REGRESSION | N |
| Of those: accuracy N/A | N |
| Primary bottleneck EP | EP name (X.X% vs Y.Y% others) |
```

---

## Track B: Git Explain Files  →  `explain/`

Explain files are structured technical write-ups generated from git history.
They are organized into three sub-folders:

```
explain/
  by-pr/        — one file per merged PR
  by-module/    — one file per affected module
  by-milestone/ — milestone progress snapshots
```

### B1. Collect Git Data for the Period

```bash
# All commits in the period (excluding merges and reverts)
git log --oneline --since="YYYY-MM-DD" --until="YYYY-MM-DD" --no-merges

# Full metadata per commit
git log --format="%H %h %ae %ad %s" --date=short \
        --since="YYYY-MM-DD" --until="YYYY-MM-DD" --no-merges

# Files changed per commit (run for each hash)
git show --stat <hash>

# Full diff for a commit (run when writing the explain file)
git show <hash>
```

For each commit, extract:
- Short + full hash
- Author email → map to display name (check git log history for established name mapping)
- Date
- PR number (parse from commit message: `(#NNN)` or `#NNN`)
- Subject line → PR title
- Files changed with +/- line counts

### B2. Write explain/by-pr/ Files

One file per PR: `PR-NN-<slug>.md` where NN is sequential and slug is kebab-case title.

```markdown
# PR-NN: <commit subject> (#NNN)

## Commit Metadata
| Field | Value |
| Commit Hash | `<hash>` |
| Date | YYYY-MM-DD |
| Author | <Name> |
| PR Number | #NNN |
| Message | <subject> |
| Files Changed | N |
| Insertions | +N |
| Deletions | -N |

## Files Changed
### Source Code Changes (modelkit/*)
| File | Type | +/- | Summary |
### Test Changes (tests/*)
| File | Type | +/- | Summary |

## Per-File Explanations
[For each changed file: what was there before, what changed, why]

## Summary
[2-4 sentences: root cause + fix + verification method]

## Design Alignment
[table: design doc | status]
```

### B3. Write explain/by-module/ Files

One file per affected module: `mod-<module>.md`.

```markdown
# Module: <module-name>
**Path**: `modelkit/<module>/`

## 1. Module Overview
[1-2 sentences: what this module does + summary of changes this period]

## 2. Files Changed
| File | Status | Lines +/- |

## 3. Cumulative Changes (Net Effect)
[For each file: before/after description — combined view across all PRs]

## 4. New APIs/Classes/Functions
| API | Description |
```

### B4. Collect Issue / Task Tracking Data

Milestone files depend on **issue status**, not just git history. Git tells you what
landed; issues tell you what is planned, in-progress, or blocked.

**Source**: GitHub Issues via REST API. The `gh` CLI is not reliably authenticated in
this environment. Use the GitHub REST API with a token retrieved from the Windows
Credential Manager via `git credential fill`.

#### B4a. Retrieve GitHub Token

```python
import subprocess
result = subprocess.run(
    ['git', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\n\n',
    capture_output=True, text=True
)
token = next((l.split('=',1)[1] for l in result.stdout.splitlines() if l.startswith('password=')), '')
```

#### B4b. Fetch All Issues (Python)

Write a script to `temp/analyze_issues.py` and run with the Azure CLI Python:
```bash
PYTHONIOENCODING=utf-8 "/c/Program Files (x86)/Microsoft SDKs/Azure/CLI2/python.exe" temp/analyze_issues.py
```

The script must:
1. Fetch all open issues: `GET /repos/microsoft/ModelKit/issues?state=open&per_page=100&page=N`
2. Fetch all closed issues: `GET /repos/microsoft/ModelKit/issues?state=closed&per_page=100&page=N`
3. Filter out pull requests (`"pull_request" not in issue`)
4. Fetch open PRs: `GET /repos/microsoft/ModelKit/pulls?state=open&per_page=100&page=N`
5. Fetch closed PRs: `GET /repos/microsoft/ModelKit/pulls?state=closed&per_page=100&page=N`, then filter to merged only (`merged_at` not null)
6. Group issues by `issue["milestone"]["title"]` (or `"(no milestone)"` if null)
7. Write structured markdown to `<report_period>/explain/by-milestone/issues-raw.md`

#### B4c. issues-raw.md Structure

```markdown
# GitHub Issues — microsoft/ModelKit
**Fetched**: YYYY-MM-DD
**Total open**: N
**Total closed**: N

## Milestone Summary
| Milestone | Open | Closed | Total |

## <Milestone Name> (open: N, closed: N)
### Open
| # | Title | Labels | Assignee | Created |
### Closed
| # | Title | Labels | Assignee | Closed |

## P0 Issue Bodies (open)
### #NNN — <title>
**Milestone**: ... | **Labels**: ... | **Assignee**: ...
<first 400 chars of body>
```

#### B4d. Milestone Mapping (microsoft/ModelKit)

| GitHub Milestone | Program Name | Scope |
|-----------------|--------------|-------|
| `202603 Release` | 0315 / March Release | Core framework, 3 EPs baseline |
| `202604 Release` | April 14 Gate | Legal/security deadline + EP/infra depth |
| `202605 Release` | 0501 / May 1 Release | Full feature set, CI/CD, model scale |
| `202606+ Post Build` | Post-Build / June+ | AMD NPU, advanced features, GGUF |
| `(no milestone)` | Unscheduled | Model tracking, ESRP, trade compliance |

#### B4e. Label → Track Mapping

| Label | Milestone File |
|-------|---------------|
| `P0`, `feature scale` | milestone-feature-scale.md |
| `P0`, `EP scale` | milestone-ep-scale.md |
| `P0`, `model / task scale` | milestone-model-scale.md |
| `release`, `P0` | milestone-release-readiness.md |
| `hardware` | milestone-ep-scale.md (hardware blockers section) |

**Owner field**: derive from issue `assignees[*].login`, strip `_microsoft` suffix.
If unassigned, write "Unassigned" — an unassigned P0 task is always a lowlight.

### B5. Write explain/by-milestone/ Files

Five standard files, all rewritten from `issues-raw.md` (never from memory):

| File | Content |
|------|---------|
| `milestone-overview.md` | Milestone mapping table + dashboard (P0 open/closed per milestone) + per-milestone snapshot + risk register |
| `milestone-feature-scale.md` | P0 feature tasks by milestone (202603/604/605), closed vs open, owner, notes |
| `milestone-model-scale.md` | Model count trajectory + model-family tracking issues (#425–#447 pattern) + pace analysis |
| `milestone-ep-scale.md` | EP status table (operational/not started/blocked) + open P0 EP issues + E2E coverage snapshot |
| `milestone-release-readiness.md` | Legal/security/ESRP/trade compliance tasks, deadline timeline, completion % by track |

For every task listed, include the GitHub issue number (`#NNN`) so the report is
traceable back to the issue tracker.

Source priority (highest to lowest):
1. `issues-raw.md` (GitHub Issues REST API) — authoritative for task status and owners
2. `data/ep_coverage_analysis.md` — authoritative for EP pass rates in ep-scale file
3. `plans/release/<milestone>/README.md` — authoritative for targets and deadlines
4. Git log — confirms what actually landed (issue may say "done" before the PR merges)

**Never write milestone files from memory or git log alone.** Always collect
`issues-raw.md` first (B4 above), then derive all task tables from that file.

---

## Output Checklist

Before handing off to the `biweekly-report` skill, verify:

**Track A (data/):**
- [ ] `qnn_report_MMDD.csv`, `ov_report_MMDD.csv`, `vitisai_report_MMDD.csv` all present
- [ ] Each CSV has the 8-column header, correct row count (matches source JSON)
- [ ] `accuracy_verdict` column is populated — `N/A` is acceptable, but empty cells are not
- [ ] `ep_coverage_analysis.md` pass rates are consistent with CSV PASS counts
- [ ] Accuracy summary section present; REGRESSION count matches CSV
- [ ] Zero-coverage tasks explicitly listed

**Track B (explain/):**
- [ ] One `by-pr/PR-NN-*.md` per merged PR (reverted PRs noted, excluded from totals)
- [ ] One `by-module/mod-*.md` per affected module
- [ ] All milestone files present and updated

---

## Data Quality Rules

- **Snapshot dates from source**: use MMDD from the JSON path for CSV naming, not today
- **PASS/FAIL is case-sensitive**: `true`/`false` in JSON → `"PASS"`/`"FAIL"` in CSV
- **Reverted PRs**: document separately; exclude from module and milestone files
- **Model count ≠ model-task count**: one model with 2 tasks = 2 rows; always label
  counts as "model-task combinations" not "models"
- **Partial data warning**: if only 2 of 3 EP JSONs are available, do not compute the
  three-EP intersection — flag this in `ep_coverage_analysis.md` header
- **Accuracy is per-EP**: each EP has its own `eval_report.html`; if only QNN has accuracy
  data, label OV/VitisAI accuracy as "N/A — not yet evaluated" (not missing data)
- **`ACCURACY_AT_RISK` = PASS in rollups**: normalize to `PASS` in the CSV and all counts;
  keep the raw label visible only in the detailed accuracy table
- **delta_relative is signed**: negative means regression (output accuracy dropped);
  always display as percentage, e.g. `-3.4%` not `-0.034`
