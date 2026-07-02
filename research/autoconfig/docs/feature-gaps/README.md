# Feature Gap Issues — WinML autoconfig Research

Each issue is a separate JSON file in this directory. Filed issues have `issue_number` set;
pending issues have `issue_number: null`.

## JSON Schema

```json
{
  "issue_number": 921,            // null if not yet filed
  "github_url": "https://...",    // null if pending
  "title": "...",
  "status": "OPEN | CLOSED | PENDING",
  "labels": ["..."],
  "filed_date": "YYYY-MM-DD",     // null if pending
  "category": "analyze | build | optimize | perf | ...",
  "source_findings": ["npu-010"], // KB finding IDs that motivated this issue
  "affected_eps": ["qnn_npu"],
  "affected_arch": ["mobilevit"],
  "summary": "One paragraph",
  "root_cause": "Detailed explanation",
  "measured_impact": [
    {
      "model": "apple/mobilevit-small",
      "ep": "qnn_npu",
      "hypothesis": "h9",
      "baseline_ms": 26.6,
      "result_ms": 31.8,
      "gain_pct": -19.5,
      "verdict": "DISCARD",
      "protocol": "3x500 iters",
      "date": "YYYY-MM-DD"
    }
  ],
  "fix_needed": {
    "file": "analyze_insight.py",
    "function": "...",
    "description": "...",
    "code_sketch": "..."  // optional
  },
  "discriminator": "How to detect this case at analysis time",
  "related_issues": [180],
  "notes": "..."
}
```

## Index

| File | Issue | Status | Category | Source Findings |
|---|---|---|---|---|
| `921-analyze-highdimRTR-hybrid-unfold.json` | [#921](https://github.com/microsoft/winml-cli/issues/921) | OPEN | analyze | npu-010, gpu-008 |
| `pending-cpu001-opset-regression-warning.json` | pending | PENDING | build | cpu-001 |
| `pending-cpu008-layer-norm-fusion-guard.json` | pending | PENDING | optimize | cpu-008 |
| `pending-npu006-fusedconv-unfuse.json` | pending | PENDING | optimize | npu-006 |
| `pending-npu007-dvfs-protocol-flag.json` | pending | PENDING | perf | npu-007 |

## How to file a pending issue

```bash
gh issue create --repo microsoft/winml-cli \
  --title "<title from json>" \
  --body "$(cat pending-<name>.json | python -c 'import json,sys; d=json.load(sys.stdin); print(d[\"summary\"] + \"\\n\\n\" + d[\"root_cause\"])')" \
  --label "P2,triaged"

# Then update the JSON:
# - Set issue_number, github_url, status = "OPEN", filed_date
# - Rename file from pending-* to <number>-<slug>.json
```
