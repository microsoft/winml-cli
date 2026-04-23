---
name: biweekly-report
description: >
  Generates a professional bi-weekly engineering status report for an engineering manager
  or tech lead to share with leadership. Use this skill whenever the user asks to write,
  generate, or create a bi-weekly report, sprint summary, fortnightly update, periodic
  engineering status update, or similar. Also trigger when the user says things like
  "summarize the last two weeks", "what did we ship this period", "generate a status update
  for my boss", "write the biweekly", or "produce an engineering update". The skill covers
  the full workflow: collecting data from git, milestone plans, and test/coverage files,
  then producing a structured markdown report with four sections: high-level goals,
  highlights/lowlights, data analysis, and task status tables.
---

# Bi-Weekly Engineering Report

You are writing a professional bi-weekly report for an engineering manager or tech lead
to share with leadership. The report must be numbers-driven, concise, and immediately
scannable — no filler prose. Every claim needs a reference (PR, commit, or data file).
Every risk needs a severity, deadline, and owner.

---

## Step 1: Clarify Scope Before Collecting Data

Ask (or infer from context) the following before starting:

1. **Period**: Start date to end date (e.g., 2026-03-13 to 2026-03-22)
2. **Report folder**: Where to save the output (e.g., `docs/bi-weekly report/032226_001/`)
3. **Core dimensions**: What are the 2-4 top-level goals being tracked? (e.g., EP Scale /
   Model Scale / Feature Scale). If unclear, check the milestone/release plan files.
4. **Milestone targets**: What are the numerical targets and deadlines? (e.g., 50 models
   by May 1)

If the user just says "write the bi-weekly report", look for existing report folders to
infer structure, and read the latest milestone plan to discover dimensions and targets.

---

## Step 2: Collect Data

Run these in parallel — do not wait for one before starting the next.

### 2a. Git Log for the Period

```bash
git log --oneline --since="YYYY-MM-DD" --until="YYYY-MM-DD"
git log --format="%h %ae %s" --since="YYYY-MM-DD" --until="YYYY-MM-DD"
```

For each commit, capture: short hash, author email (to derive owner name), subject line,
and PR number if present. Group commits by module area (e.g., commands, models, config,
pattern, static-analyzer, utils).

### 2b. Milestone / Release Plan Files

Look in the project for files like:
- `plans/release/<milestone>/README.md` — targets, task list, owner column
- `plans/release/<milestone>/release.md` — open-source / legal gate tasks
- `docs/bi-weekly report/<prev_period>/explain/by-milestone/milestone-overview.md`

Extract: per-dimension targets, current task status (done / in-progress / not started),
owners, blockers, deadlines.

### 2c. Test / Coverage Data (if available)

If `data/ep_coverage_analysis.md` already exists in the report period folder, read it
directly — it is the pre-computed output of the `biweekly-data-collection` skill and
contains all per-EP statistics and the three-EP intersection.

If the `data/` folder is empty or stale, invoke the `biweekly-data-collection` skill
first to pull and compute the latest data, then return to this step.

For each execution provider (EP) or test dimension, extract from the analysis file:
- Total test combinations
- Pass count and pass rate
- Failure class breakdown if available

**Cross-dimension intersection** (e.g., "models passing all EPs"):
Use the intersection set already computed in `ep_coverage_analysis.md` — Section 2
lists all passing (model, task) pairs grouped by task category.

### 2d. Backlog / Task Tracking

Check `docs/backlogs/`, `plans/`, or any Jira/ADO export. Note which P0 tasks have no
owner assigned — that is always a lowlight.

---

## Step 3: Write the Report

Save to `<report_folder>/biweekly-report.md`. Use this exact structure:

---

### Header

```markdown
# [Project Name] Bi-Weekly Report
**Period:** YYYY-MM-DD to YYYY-MM-DD
**Report ID:** MMDDYY_NNN
**Generated:** YYYY-MM-DD
**Distribution:** Engineering Leadership
```

---

### Section 1: High-Level Goals — Core Metrics Dashboard

A single summary table — one row per dimension, one sentence per cell. Do not expand
into per-dimension sub-tables. EP Scale tracks enablement count only, not coverage rates.

```markdown
| Dimension     | Now | Target (May 1) | Gap | Status |
|---------------|-----|----------------|-----|--------|
| EP Scale      | 5 EPs enabled | 7 EPs | -2 (AMD NPU + QNN Adreno) | Blocked — no hardware, no owner |
| Model Scale   | ~24 built-in models | 50 models | -26 models | At Risk — pace 0.4/day vs need 0.65/day |
| Feature Scale | 3/14 P0 tasks done | 14/14 P0 tasks | -11 tasks | At Risk — GGUF, Debug, CI/CD not started |
```

Use status emoji (done / at-risk / critical) in the Status cell only. Keep each Gap cell
as a signed number plus the single most important blocking fact.

---

### Section 2: Highlights & Lowlights

**Highlights — organized by module**, not a flat list. Group related PRs under a module
heading with a one-line description of what the module group does. End each bullet with
the owner(s) in italics.

```markdown
**Module Name** — *one-line description of what this group of changes does*
- PR title and key detail — what changed and why it matters. *(Owner)*
- PR title and key detail — what changed and why it matters. *(Owner)*
```

Example:
```markdown
**Models / Export** — *IOConfig fixes to unblock model export failures*
- Segformer support added (#423) — SegformerIOConfig + class mapping; unblocks nvidia/segformer family. *(Qiong Wu)*
- ZoeDepth, MPNet, nougat-base fixes (#411, #415, #449) — 3 new IOConfig entries resolving export failures. *(Chao Zhang, Qiong Wu)*
```

**Lowlights table** — one row per risk:

| # | Risk | Severity | Deadline | Owner |
|---|------|----------|----------|-------|

Severity: Critical / High / Medium (use emoji: red circle, orange circle, yellow circle).
Never leave Deadline or Owner blank — use "TBD" and flag it as needing assignment.

---

### Section 3: Data Analysis

Use ASCII charts and tables — not prose paragraphs. Organize around:

1. **Coverage/pass-rate chart**: horizontal bar per EP or test dimension, showing pass rate
   and raw counts. Include an "all dimensions combined" row.

2. **Category breakdown table**: for cross-dimensional intersection data (e.g., models
   passing all EPs), group by category and show count. Highlight the zero-coverage rows
   as they reveal architectural gaps.

3. **Trajectory table**: checkpoint to metric value, with target and required pace.
   Compute "current pace" vs "required pace" explicitly.

4. **Bottleneck analysis**: 2-3 bullet points identifying the single largest gap, why it
   exists (architecture class, missing operator, hardware block), and what fixing it would
   unlock.

---

### Section 4: Task Status — Completed & In-Progress

All three sub-sections are organized by module group, not as flat lists.

**4.1 Completed This Period** — group PRs by module. Each group has a heading with a
one-line description of what the module does, followed by a table:

```markdown
**Module Name** — *one-line description of what this module does*

| PR | Description | Owner |
|----|-------------|-------|
| #NNN / <hash> | What it does and why | Name |
```

The Description column explains the specific change — not just its title. One sentence
is enough. Include the owner derived from git author.

**4.2 In-Progress Tasks** — group by module with a Description column:

```markdown
**Module Name**

| Task | Description | Progress | Owner | Blocker |
|------|-------------|----------|-------|---------|
```

Express progress as a rough percentage (e.g., ~40%). If unknown, write "unknown".

**4.3 Not Started — P0 Items Needing Owner Assignment** — group by track (e.g.,
Release Gate, EP Scale, Feature Scale, Model Scale). Surface hard deadlines as separate
groups. Each group has a Description column:

```markdown
**Track Name (deadline if applicable)**

| Task | Description | Deadline | Priority |
|------|-------------|----------|----------|
```

Red = assign owner immediately. Orange = assign this sprint.

---

### Footer: Action Items for This Week

Always end with this table:

| # | Action | Owner | Due |
|---|--------|-------|-----|

Pull action items from the lowlights and the not-started table. Be specific: "Assign DRI
for L-01 through L-07 legal tracks" not "work on legal stuff".

---

### Footer: Data Sources

One line citing the data used:

```
*Data sources: git log <hash>..<hash>, <test-data-file> (EP, date), <milestone-plan-file>.*
```

---

## Writing Style Rules

These rules exist because this report goes to leadership — every word is read critically:

- **Section 1 is a scoreboard, not a deep-dive**: one row per dimension, one number per
  cell. Coverage rates and sub-metrics belong in Section 3.
- **Lead with numbers**: "42.1% pass rate (91/216)" not "good coverage"
- **Name the gap explicitly**: "-26 models in 40 days" not "behind on models"
- **Required vs actual pace**: if a target needs X units/week and you are doing Y, say both
- **No passive voice on risks**: "AMD NPU has no owner — escalate to PM" not "this is TBD"
- **Module grouping over flat lists**: both Highlights and Section 4 must be grouped by
  module — a flat numbered list of PRs is a changelog, not a status report
- **Description column is mandatory**: every task table needs a Description column that
  explains what the task is, not just its name
- **Owner on every completed PR**: derive from git log author email; never leave blank

---

## Common Patterns from Past Reports

- **Section 1 creep**: resist expanding Section 1 into per-dimension sub-tables or adding
  coverage metrics. Keep it as a single 3-row summary. Details go in Section 3.
- **Legal deadlines slip silently**: always check if legal/security gate tasks have owners.
  If they do not and the deadline is less than 30 days away, it is a Critical lowlight.
- **New model does not equal passing model**: a model added to the codebase does not count
  toward a "models supported" target until it passes E2E tests on the target EPs.
- **Encoder-only models cross EPs more reliably** than decoder/generative models — when
  doing gap analysis on which models to add, recommend encoder-only architectures first.
- **The bottleneck EP is the ceiling**: for "models passing all EPs", the EP with the
  lowest individual pass rate limits the intersection. Always highlight which EP is the
  bottleneck and by how many percentage points it lags.
- **Unowned P0 tasks**: scan every P0 task in the milestone plan for TBD owners. Each
  unowned P0 task is a separate lowlight entry, not a footnote.
