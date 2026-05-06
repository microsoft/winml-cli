# Issue Dashboard

Generate a daily issue health dashboard for microsoft/WinML-ModelKit.
Shows: volume (opened/closed), closed-by-person breakdown, and risk flags.

## Usage

- `/issue-dashboard` → last 7 days
- `/issue-dashboard 14` → last N days
- `/issue-dashboard 2026-04-28` → since a specific date

## Key IDs (do not change)

- **Project**: `PVT_kwDOAF3p4s4BTTUF` (ModelKit)
- **Priority field**: `PVTSSF_lADOAF3p4s4BTTUFzhAlRdY`
  - P0 → `79628723`
  - P1 → `0a877460`
  - P2 → `da944a9c`

## Steps

### 1. Determine date range

Parse the argument:
- No argument → `since = today - 7 days`
- Integer N → `since = today - N days`
- Date string → `since = that date`

Compute `since_iso` (YYYY-MM-DD) and `today_iso`.

### 2. Fetch all issues

```bash
gh issue list --repo microsoft/WinML-ModelKit \
  --state all --limit 500 \
  --json number,title,state,createdAt,closedAt,updatedAt,assignees,labels
```

### 3. Fetch project priorities

Get priority for each open issue from the Project board:

```bash
gh api graphql -f query='
{
  node(id: "PVT_kwDOAF3p4s4BTTUF") {
    ... on ProjectV2 {
      items(first: 100) {
        nodes {
          content { ... on Issue { number } }
          fieldValues(first: 10) {
            nodes {
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field { ... on ProjectV2SingleSelectField { id } }
              }
            }
          }
        }
      }
    }
  }
}'
```

Build a map: `issue_number → priority (P0/P1/P2/none)`.

If the project returns more than 100 items, paginate using `after` cursor until all items are fetched.

### 4. Compute metrics

Use Python to process the JSON. Write a script to temp/ and run with `uv run python`.

```python
import json, sys
from datetime import datetime, timezone

issues   = json.loads(open("temp/_dash_issues.json").read())
prio_map = json.loads(open("temp/_dash_prio.json").read())   # {number: "P0"/"P1"/"P2"/None}
since    = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=timezone.utc)
today    = datetime.now(timezone.utc)

def parse(s): return datetime.fromisoformat(s.replace("Z","+00:00")) if s else None

# ── Panel 1: Volume ──────────────────────────────────────────
opened   = [i for i in issues if parse(i["createdAt"]) >= since]
closed   = [i for i in issues if i["state"]=="CLOSED" and parse(i["closedAt"]) and parse(i["closedAt"]) >= since]
open_all = [i for i in issues if i["state"]=="OPEN"]
open_p0  = [i for i in open_all if prio_map.get(i["number"]) == "P0"]

# ── Panel 2: Closed by person ────────────────────────────────
from collections import Counter
closer_counts = Counter()
for i in closed:
    assignees = [a["login"] for a in i.get("assignees", [])]
    if assignees:
        for a in assignees:
            closer_counts[a] += 1
    else:
        closer_counts["(unassigned)"] += 1

# ── Panel 3: Risk flags ──────────────────────────────────────
risks = []
STALE_P0_DAYS = 5
STALE_P1_DAYS = 14
UNASSIGNED_DAYS = 2

for i in open_all:
    p = prio_map.get(i["number"])
    updated = parse(i["updatedAt"])
    age_days = (today - updated).days if updated else 999
    no_assignee = len(i.get("assignees", [])) == 0
    created_days = (today - parse(i["createdAt"])).days if parse(i["createdAt"]) else 0

    if p == "P0" and age_days >= STALE_P0_DAYS:
        risks.append(("P0", i["number"], i["title"][:60], f"open {age_days}d, no update"))
    elif p == "P1" and age_days >= STALE_P1_DAYS:
        risks.append(("P1", i["number"], i["title"][:60], f"stale {age_days}d"))
    elif no_assignee and created_days >= UNASSIGNED_DAYS:
        risks.append(("UNASSIGNED", i["number"], i["title"][:60], f"unassigned {created_days}d"))

# ── Output ───────────────────────────────────────────────────
print(json.dumps({
    "period": f"{since.date()} → {today.date()}",
    "opened": len(opened),
    "closed": len(closed),
    "open_total": len(open_all),
    "open_p0": len(open_p0),
    "closed_by": closer_counts.most_common(),
    "risks": risks,
}))
```

### 5. Output the dashboard

Print a formatted report. Use this exact layout:

```
## Issue Dashboard · <period>

### Volume
| Metric            | Count |
|-------------------|-------|
| Opened            | N     |
| Closed            | N     |
| Total open        | N     |
| P0 open           | N     |

### Closed by Person
| Assignee      | Closed |
|---------------|--------|
| @handle       | N      |
| ...           | ...    |

### Risk Flags
| Level       | Issue | Summary                          | Detail               |
|-------------|-------|----------------------------------|----------------------|
| 🔴 P0       | #NNN  | title...                         | open Nd, no update   |
| 🟡 P1       | #NNN  | title...                         | stale Nd             |
| ⚪ UNASSIGNED | #NNN | title...                         | unassigned Nd        |
```

If there are no risk flags, print: `✅ No risk flags.`

After the tables, print a one-line summary:
```
**Summary**: N issues opened, N closed in the last N days. N P0s open. [N risks flagged / No risks.]
```

### 6. Clean up temp files

```bash
rm -f temp/_dash_issues.json temp/_dash_prio.json temp/_dash_compute.py
```

## Thresholds

| Flag         | Trigger                                      |
|--------------|----------------------------------------------|
| P0 stale     | Open P0 with no update in ≥ 5 days           |
| P1 stale     | Open P1 with no update in ≥ 14 days          |
| Unassigned   | Open issue with no assignee for ≥ 2 days     |

## Notes

- "Closed by person" uses the issue's **assignees at time of close** as a proxy for who fixed it.
  If an issue has multiple assignees, each gets credit.
- Priority comes from the GitHub Project board field, not labels.
- Issues with no priority set in the board are listed as `none` and excluded from P0/P1 risk checks.
