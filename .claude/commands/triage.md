# Issue Triage

Triage open issues in microsoft/WinML-ModelKit in three phases:

1. **Phase 0 — Screen new issues**: find open issues without "need triage" label, decide if each is a Bug or Feature/Task, mark bugs for triage.
2. **Phase 1 — Triage**: for each issue with "need triage" label, set priority in the GitHub Project, classify component, assign owners, post a triage comment.
3. **Phase 2 — HTML report**: generate/overwrite `temp/triage-report.html` with the current state of all open "need triage" issues.

---

## Key IDs (do not change)

- **Project**: `PVT_kwDOAF3p4s4BTTUF` (ModelKit)
- **Priority field**: `PVTSSF_lADOAF3p4s4BTTUFzhAlRdY`
  - P0 → `79628723`
  - P1 → `0a877460`
  - P2 → `da944a9c`

- **Milestones** (repo: microsoft/WinML-ModelKit)
  - `202603 Release` → milestone number `1`
  - `202604 Release` → milestone number `2`
  - `202605 Release` → milestone number `3`  ← **May Release (current)**
  - `202606+ Post Build` → milestone number `4`

---

## Owner map (alias → GitHub handle)

> **Source of truth**: `docs/context/team.json` — load it at the start of any task that needs alias↔GitHub resolution or assignee lookup.
>
> ```
> cat docs/context/team.json
> ```

| Name | Alias | GitHub handle |
|------|-------|--------------|
| Lu Han | luhan | luhan2017 |
| Hualiang Xie | hualxie | xieofxie |
| Zheng Te | zhengte | tezheng |
| Yi Ren | reny | vortex-captain |
| Zhipeng Wang | zhiwang | timenick |
| Yue Sun | yuesu | KayMKM |
| Chao Zhang | zhangchao | chinazhangchao |
| Fangyang Ci | fangyangci | Fangyangci |
| Ziyuan Guo | ziyuanguo | ziyuanguo1998 |
| Shiyi Zhen | shizhen | ssss141414 |
| Qiong Wu | qiowu | dingmaomaobjtu |
| Zhenchao Ni | zhenni | zhenchaoni |
| Brenda Bai | yiba | hi-brenda |

## Component → owners

> **Source of truth**: `docs/context/assignments.json` — load it for component→owner lookups.
>
> ```
> cat docs/context/assignments.json
> ```

## Component detection keywords

- **Load & Export**: export, exporter, onnx export, hf export, optimum, torch.onnx, pixel_mask, OnnxConfig, model config, architecture
- **Analyzer**: analyze, static analyzer, SA, op coverage, runtime check, unknown op, QDQ support, EP support, IHV, dynamic input, opset
- **Optimizer**: optimizer, graph optimizer, graph-optimizer, fusion, rewrite, pattern, RTR, QLinear, attention fusion, gelu
- **Eval**: eval, evaluator, evaluation, metric, dataset, accuracy, label alignment, fill-mask, text-classification eval
- **Compile**: compile, compiler, EPContext, WinMLSession, provider_options, EP context, AOT, TarWriter
- **Sys**: sysinfo, sys, hardware detection, device detection, machine architecture, system info
- **Config**: config, --device, --ep flag, device resolution, wmk config, winml config
- **Build**: build, winml build, build pipeline, build stage, full pipeline, calibration dataset mismatch
- **Quantize**: quantize, quantization, QDQ, calibration, quant, qdq
- **Perf**: perf, performance, throughput, latency, op-tracing, profil, benchmark
- **Catalog**: catalog, hub, model list, model registry, WinML Hub, WinML Catalog
- **Inspect**: inspect, resolver, MUST-rule
- **Repository**: CI, CD, GitHub Actions, CodeQL, CHANGELOG, README, CONTRIBUTING, release, PyPI, ruff, import, dependency
- **Other**: anything that doesn't fit above

## Priority criteria

| Priority | When to use |
|----------|-------------|
| **P0** | Silent fallback (user gets wrong result without knowing); silent wrong behavior (wrong EP, wrong device, dropped options); broken pipeline (different results between equivalent commands); data corruption or silent data loss; **feature broken with a clear (non-silent) error and no workaround** |
| **P1** | Crash / exception / process exit (P0 if severe or frequent; P1 if edge-case or rare); **customer-reported bugs** (`customer report` label) that do not meet P0 conditions; misleading error messages that block workflow but are not silent |
| **P2** | Non-blocking bugs with workarounds; UX improvements; output quality; inconsistent behavior; perf issues; most feature requests |

Crashes and customer-reported bugs always land at **P0 or P1 — never P2**. When in doubt between P1 and P2, use P2.

---

## Phase 0 — Screen new issues

Find open issues that do NOT yet have the `need triage` label:
```
gh issue list --repo microsoft/WinML-ModelKit --state open --json number,title,body,labels,issueType --limit 200 \
  | python3 -c "import json,sys; issues=json.load(sys.stdin); [print(i['number'],'|',i['title']) for i in issues if not any(l['name']=='need triage' for l in i['labels'])]"
```

For each such issue, read the title and body carefully and decide:

**Is it a Bug?**  Broken behavior, crash, wrong output, silent failure, regression, unexpected error → **Bug**
**Is it a Feature?**  New capability, enhancement, "add support for", "would be nice if" → **Feature — skip**, do not add `need triage`
**Is it a Task?**  Refactor, chore, CI, docs, dependency upgrade → **Task — skip**, do not add `need triage`

For issues classified as **Bug**:

### 0a. Set Issue Type to Bug (GraphQL)

First query available issue types for the repo:
```
gh api graphql -f query='{ repository(owner:"microsoft",name:"WinML-ModelKit") { issueTypes(first:10) { nodes { id name } } } }'
```
Find the `id` where `name == "Bug"`.

Then set it:
```
gh api graphql -f query='mutation { updateIssue(input:{id:"<issue_node_id>",issueTypeId:"<bug_type_id>"}) { issue { number } } }'
```

### 0b. Add labels and "need triage"
```
gh issue edit <number> --repo microsoft/WinML-ModelKit --add-label "bug,need triage"
```

After Phase 0, log which issues were marked for triage and which were skipped (with reason).

---

## Phase 1 — Triage

For EACH issue with the `need triage` label (process one at a time):

### 1. Fetch the issue
```
gh issue view <number> --repo microsoft/WinML-ModelKit --json number,title,body,labels,assignees
```

### 2. Classify
Read title and body. Determine:
- **Component** (use keyword matching above; pick the single best fit)
- **Priority** (P0/P1/P2 per criteria above)
- **Labels to ADD** (only from this list — never create new ones):
  `documentation`, `duplicate`, `good first issue`, `help wanted`, `invalid`, `question`, `wontfix`,
  `refactor`, `testing`, `infrastructure`, `static-analyzer`, `graph-optimizer`, `QDQ`, `accuracy`,
  `hardware`, `NPU`, `GPU`, `release`, `to followup / verify`, `EP scale`, `feature scale`,
  `model / task scale`, `dependencies`, `python`, `dev experience`, `bug`, `customer report`,
  `0430 bugbash`, `need triage`
- **Owners** (1–2 GitHub handles from the component table above)

### 3. Add to project and set priority

Get issue node ID:
```
gh api graphql -f query='{ repository(owner:"microsoft",name:"WinML-ModelKit") { issue(number:<N>) { id } } }'
```

Add to project and get item ID:
```
gh api graphql -f query='mutation { addProjectV2ItemById(input:{projectId:"PVT_kwDOAF3p4s4BTTUF",contentId:"<issue_node_id>"}) { item { id } } }'
```

Set priority:
```
gh api graphql -f query='mutation { updateProjectV2ItemFieldValue(input:{projectId:"PVT_kwDOAF3p4s4BTTUF",itemId:"<item_id>",fieldId:"PVTSSF_lADOAF3p4s4BTTUFzhAlRdY",value:{singleSelectOptionId:"<priority_option_id>"}}) { projectV2Item { id } } }'
```

### 4. Set milestone to May Release
```
gh api repos/microsoft/WinML-ModelKit/issues/<number> -X PATCH -f milestone=3
```

### 5. Add labels
```
gh issue edit <number> --repo microsoft/WinML-ModelKit --add-label "<label1>,<label2>"
```

### 6. Assign owners

Look up the issue's component in `docs/context/assignments.json`. Use the **`github` field** (not the `aliases` field) to get the GitHub handles to assign. The `github` field lists the correct handles in priority order — assign the first 1–2.

**UX routing rule**: If the issue has the `dev experience` label, assign `hi-brenda` (yiba, PM) for confirmation — do NOT assign an engineering owner yet.

Otherwise, assign the engineering handles from `assignments.json` → `github` for that component:
```
gh issue edit <number> --repo microsoft/WinML-ModelKit --add-assignee "<handle1>,<handle2>"
```

**Important**: If the issue already has assignees, check whether the correct component owners are already assigned. If not, ADD the missing component owners even though the issue is already assigned to someone. Only skip assignment if all correct component owners are already present.

### 7. Post triage comment

Do NOT mention assignees in the comment body.

```
gh issue comment <number> --repo microsoft/WinML-ModelKit --body "..."
```

Comment template:
```
**Triage summary**

- **Component**: <component>
- **Priority**: <P0/P1/P2>

<1–2 sentence summary of what the issue is about and suggested next step>
```

---

## Phase 2 — Generate HTML report

After Phase 1 completes, fetch the current real state of all open "need triage" issues and write `quality-status/bug-fixing/<YYYYMMDD>/triage-report.html` where `<YYYYMMDD>` is today's date (e.g. `20260507`). Create the directory if it does not exist.

### Fetch data

Get all open "need triage" issues with assignees:
```
gh issue list --repo microsoft/WinML-ModelKit --label "need triage" --state open --json number,title,labels,assignees --limit 200
```

Get each issue's priority from the project (use the `addProjectV2ItemById` mutation which is idempotent and returns field values, or query project items directly).

### Report structure

Write a self-contained HTML file to `quality-status/bug-fixing/<YYYYMMDD>/triage-report.html` with:

**Header**: "WinML-ModelKit — Issue Triage Report", generation date, issue count.

**Summary cards** (4): Total open, P0 count, P1 count, P2 count.

**Charts section** (2 columns):
- Left: horizontal bar chart by component (sorted descending by count), each bar clickable to filter the table to that component.
- Right: priority breakdown list (P0/P1/P2 with count and brief description).

**Interactive issue table**:
- Default view: **grouped by component** (colored section headers per component, rows sorted P0→P1→P2 within each group).
- Toggle: "Group by: Priority | Component" switches between modes.
- Filter buttons update to match the active mode (All/P0/P1/P2 in priority mode; All/Compile/Perf/… in component mode).
- Each row: clickable issue number linking to GitHub, title, component tag, priority badge, labels, assignees.
- Priority badge colors: P0=red, P1=orange, P2=yellow.
- Component colors (consistent across bar chart, dots, filter buttons, group headers):
  Compile=blue, Perf=green, Load&Export=light-blue, Analyzer=gold, Build=dark-green,
  Quantize=purple, Sys=red, Config=orange, Others=gray.

Write the complete file (HTML + inline CSS + inline JS, no external dependencies).

---

## Phase 3 — Generate Teams table

After the HTML report, print a **Teams-pasteable summary** directly to the console output (do not write to file).

Teams renders markdown tables. Output this exact format:

```
## WinML-ModelKit Bug Status — <YYYY-MM-DD>

**Total open: N** | 🔴 P0: N | 🟠 P1: N | 🟡 P2: N

| # | Title | Component | Priority | Owner |
|---|-------|-----------|----------|-------|
| [#428](url) | winml compile crashes … | Compile | 🔴 P0 | @zhenchaoni |
| [#436](url) | Investigation: Perf gap … | Perf | 🔴 P0 | @tezheng |
…
```

Rules:
- Sort rows: P0 first, then P1, then P2; within same priority sort by issue number ascending.
- **Priority emoji**: P0 → 🔴, P1 → 🟠, P2 → 🟡
- **Title**: truncate to 55 characters max, add `…` if truncated.
- **Owner**: first assignee only (skip `hi-brenda` if there is another assignee; show `hi-brenda` only if she is the sole assignee).
- Issue number links to `https://github.com/microsoft/WinML-ModelKit/issues/<N>`.
- After the table, add one line per component that has P0 issues, listing those issue numbers: `> 🔴 Compile: #428, #430, #429, #434, #240, #186`

### Constraints

- Only use **existing labels** — never create new ones
- Do not close issues
- Do not modify issue title or body

- Only use **existing labels** — never create new ones
- Do not close issues
- Do not modify issue title or body
- If an issue already has a priority set, keep the existing priority (do not downgrade)
- If an issue already has assignees, ADD the correct component owners from `docs/context/assignments.json` → `github` if they are not already among the assignees; do NOT leave incorrect or missing owners just because someone else is already assigned
- Process issues one at a time (not in parallel) to avoid rate limits
