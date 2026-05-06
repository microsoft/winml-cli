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

---

## Owner map (alias → GitHub handle)

| Alias | GitHub handle |
|-------|--------------|
| luhan | luhan2017 |
| hualxie | xieofxie |
| zhengte | tezheng |
| reny | vortex-captain |
| zhiwang | timenick |
| yuesu | KayMKM |
| zhangchao | chinazhangchao |
| fangyangci | Fangyangci |
| ziyuanguo | ziyuanguo1998 |
| shizhen | ssss141414 |
| qiowu | dingmaomaobjtu |
| zhenni | zhenchaoni |
| yiba | hi-brenda |

## Component → owners

| Component | Aliases | GitHub handles |
|-----------|---------|----------------|
| Load & Export | reny, zhangchao | vortex-captain, chinazhangchao |
| Analyzer | zhangchao, fangyangci, qiowu | chinazhangchao, Fangyangci, dingmaomaobjtu |
| Optimizer | qiowu, reny | dingmaomaobjtu, vortex-captain |
| Eval | zhenni | zhenchaoni |
| Sys | zhiwang | timenick |
| Config | zhangchao | chinazhangchao |
| Compile | zhenni | zhenchaoni |
| Quantize | zhenni | zhenchaoni |
| Perf | zhiwang, zhengte | timenick, tezheng |
| Build | zhangchao, zhengte | chinazhangchao, tezheng |
| Catalog | qiowu | dingmaomaobjtu |
| Inspect | zhengte | tezheng |
| Repository | zhiwang, yuesu | timenick, KayMKM |
| User Experience | yiba | hi-brenda |
| Other | zhengte | tezheng |

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
| **P0** | Crash / exception / process exit during normal use; silent fallback (user gets wrong result without knowing); silent wrong behavior (wrong EP, wrong device, dropped options); broken pipeline (different results between equivalent commands); data corruption or silent data loss; **all customer-reported bugs** (`customer report` label) |
| **P1** | Major feature broken for specific EP/model/platform with a clear (non-silent) error; misleading error messages that block workflow but are not silent |
| **P2** | Non-blocking bugs with workarounds; UX improvements; output quality; inconsistent behavior; perf issues; most feature requests |

When in doubt between P1 and P2, use P2 (P1 is reserved for clear non-silent breakage without workaround).

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

### 4. Add labels
```
gh issue edit <number> --repo microsoft/WinML-ModelKit --add-label "<label1>,<label2>"
```

### 5. Assign owners

**UX routing rule**: If the issue has the `dev experience` label, assign `hi-brenda` (yiba, PM) for confirmation — do NOT assign an engineering owner yet.

Otherwise, assign 1–2 engineering handles from the component table:
```
gh issue edit <number> --repo microsoft/WinML-ModelKit --add-assignee "<handle1>,<handle2>"
```

### 6. Post triage comment

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

After Phase 1 completes, fetch the current real state of all open "need triage" issues and write `temp/triage-report.html`.

### Fetch data

Get all open "need triage" issues with assignees:
```
gh issue list --repo microsoft/WinML-ModelKit --label "need triage" --state open --json number,title,labels,assignees --limit 200
```

Get each issue's priority from the project (use the `addProjectV2ItemById` mutation which is idempotent and returns field values, or query project items directly).

### Report structure

Write a self-contained HTML file to `temp/triage-report.html` with:

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

### Constraints

- Only use **existing labels** — never create new ones
- Do not close issues
- Do not modify issue title or body
- If an issue is already assigned or has a priority set, keep existing values and only add missing ones
- Process issues one at a time (not in parallel) to avoid rate limits
