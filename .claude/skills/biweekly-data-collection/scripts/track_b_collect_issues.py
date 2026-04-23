"""
Track B (B4): GitHub Issues + PRs Collection
Usage: python track_b_collect_issues.py <report_period_folder>
  e.g. python track_b_collect_issues.py "bi-weekly report/040826_001"

Fetches all issues and merged PRs from microsoft/ModelKit via GitHub REST API,
writes issues-raw.md to <report_period_folder>/explain/by-milestone/.

Auth: retrieves token from Windows Credential Manager via git credential fill.
"""
import json, sys, subprocess, urllib.request
from collections import defaultdict
from datetime import date

REPO = "microsoft/ModelKit"

if len(sys.argv) < 2:
    print("Usage: python track_b_collect_issues.py <report_period_folder>")
    sys.exit(1)

REPORT_DIR  = sys.argv[1].rstrip("/\\")
OUT_DIR     = REPORT_DIR + "/explain/by-milestone/"
OUT_FILE    = OUT_DIR + "issues-raw.md"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Auth ───────────────────────────────────────────────────────────────────────

result = subprocess.run(
    ['git', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\n\n',
    capture_output=True, text=True
)
token = next(
    (line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('password=')),
    ''
)
if not token:
    print("ERROR: Could not retrieve GitHub token from credential store.")
    print("  Run: git credential fill  — and ensure github.com credentials are stored.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "ModelKit-biweekly-report",
}

# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read()), r.headers.get("Link", "")

def fetch_all(state):
    items, page = [], 1
    while True:
        data, link = fetch(
            f"https://api.github.com/repos/{REPO}/issues"
            f"?state={state}&per_page=100&page={page}"
        )
        issues = [i for i in data if "pull_request" not in i]
        items.extend(issues)
        print(f"  {state} page {page}: {len(issues)} issues (total: {len(items)})")
        if not data or 'rel="next"' not in link:
            break
        page += 1
    return items

def fetch_all_prs(state):
    """Fetch PRs from /pulls endpoint (richer PR metadata than /issues)."""
    items, page = [], 1
    while True:
        data, link = fetch(
            f"https://api.github.com/repos/{REPO}/pulls"
            f"?state={state}&per_page=100&page={page}&sort=updated&direction=desc"
        )
        items.extend(data)
        print(f"  PR {state} page {page}: {len(data)} PRs (total: {len(items)})")
        if not data or 'rel="next"' not in link:
            break
        page += 1
    return items

print("Fetching open issues...")
open_issues   = fetch_all("open")
print(f"  Total open: {len(open_issues)}")

print("Fetching closed issues...")
closed_issues = fetch_all("closed")
print(f"  Total closed: {len(closed_issues)}")

print("Fetching open PRs...")
open_prs   = fetch_all_prs("open")
print(f"  Total open PRs: {len(open_prs)}")

print("Fetching merged/closed PRs...")
closed_prs = fetch_all_prs("closed")
merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]
print(f"  Total closed PRs: {len(closed_prs)}, merged: {len(merged_prs)}")

all_issues = open_issues + closed_issues

# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_labels(labels):
    return ", ".join(l["name"] for l in labels) if labels else ""

def fmt_assignees(assignees):
    if not assignees:
        return "Unassigned"
    return ", ".join(a["login"].replace("_microsoft", "") for a in assignees)

def fmt_ms(ms):
    return ms["title"] if ms else "(no milestone)"

def fmt_date(d):
    return d[:10] if d else ""

def fmt_body(body):
    if not body:
        return ""
    return body.replace("\r\n", " ").replace("\n", " ").strip()[:400]

# ── Group by milestone ─────────────────────────────────────────────────────────

ms_open   = defaultdict(list)
ms_closed = defaultdict(list)

for i in open_issues:
    ms_open[fmt_ms(i["milestone"])].append(i)
for i in closed_issues:
    ms_closed[fmt_ms(i["milestone"])].append(i)

all_ms = sorted(set(fmt_ms(i["milestone"]) for i in all_issues))

# ── Write markdown ─────────────────────────────────────────────────────────────

lines = []
today = date.today().isoformat()

lines.append(f"# GitHub Issues — {REPO}")
lines.append(f"**Fetched**: {today}")
lines.append(f"**Total open**: {len(open_issues)}")
lines.append(f"**Total closed**: {len(closed_issues)}")
lines.append("")

# Milestone summary table
lines.append("## Milestone Summary")
lines.append("")
lines.append("| Milestone | Open | Closed | Total |")
lines.append("|-----------|------|--------|-------|")
for ms in all_ms:
    o = len(ms_open.get(ms, []))
    c = len(ms_closed.get(ms, []))
    lines.append(f"| {ms} | {o} | {c} | {o+c} |")
lines.append("")

# Per-milestone detailed tables (known milestones first, then rest)
MILESTONE_ORDER = [
    "202603 Release",
    "202604 Release",
    "202605 Release",
    "202606+ Post Build",
    "(no milestone)",
]
ordered_ms = MILESTONE_ORDER + [ms for ms in all_ms if ms not in MILESTONE_ORDER]

for ms in ordered_ms:
    if ms not in all_ms:
        continue
    open_list   = sorted(ms_open.get(ms, []),   key=lambda x: x["number"])
    closed_list = sorted(ms_closed.get(ms, []), key=lambda x: x["number"])

    lines.append("---")
    lines.append("")
    lines.append(f"## {ms} (open: {len(open_list)}, closed: {len(closed_list)})")
    lines.append("")

    if open_list:
        lines.append("### Open")
        lines.append("")
        lines.append("| # | Title | Labels | Assignee | Created |")
        lines.append("|---|-------|--------|----------|---------|")
        for i in open_list:
            title = i["title"].replace("|", "｜")
            lines.append(
                f"| #{i['number']} | {title} | {fmt_labels(i['labels'])} "
                f"| {fmt_assignees(i['assignees'])} | {fmt_date(i['created_at'])} |"
            )
        lines.append("")

    if closed_list:
        lines.append("### Closed")
        lines.append("")
        lines.append("| # | Title | Labels | Assignee | Closed |")
        lines.append("|---|-------|--------|----------|--------|")
        for i in closed_list:
            title = i["title"].replace("|", "｜")
            lines.append(
                f"| #{i['number']} | {title} | {fmt_labels(i['labels'])} "
                f"| {fmt_assignees(i['assignees'])} | {fmt_date(i['closed_at'])} |"
            )
        lines.append("")

# P0 open issue bodies
lines.append("---")
lines.append("")
lines.append("## P0 Issue Bodies (open)")
lines.append("")
p0_open = [i for i in open_issues if any(l["name"] == "P0" for l in i["labels"])]
for i in sorted(p0_open, key=lambda x: x["number"]):
    lines.append(f"### #{i['number']} — {i['title']}")
    lines.append(
        f"**Milestone**: {fmt_ms(i['milestone'])}  |  "
        f"**Labels**: {fmt_labels(i['labels'])}  |  "
        f"**Assignee**: {fmt_assignees(i['assignees'])}"
    )
    lines.append("")
    body = fmt_body(i["body"])
    if body:
        lines.append(body)
    lines.append("")

# ── PR Section ────────────────────────────────────────────────────────────────

lines.append("---")
lines.append("")
lines.append("## Pull Requests")
lines.append("")
lines.append(f"**Open**: {len(open_prs)}  |  **Merged (all time)**: {len(merged_prs)}")
lines.append("")

if open_prs:
    lines.append("### Open PRs")
    lines.append("")
    lines.append("| # | Title | Author | Labels | Created | Branch |")
    lines.append("|---|-------|--------|--------|---------|--------|")
    for pr in sorted(open_prs, key=lambda x: x["number"]):
        title  = pr["title"].replace("|", "｜")
        author = pr["user"]["login"].replace("_microsoft", "") if pr.get("user") else ""
        labels = fmt_labels(pr.get("labels", []))
        branch = pr.get("head", {}).get("ref", "")
        lines.append(
            f"| #{pr['number']} | {title} | {author} | {labels} "
            f"| {fmt_date(pr['created_at'])} | `{branch}` |"
        )
    lines.append("")

if merged_prs:
    lines.append("### Merged PRs (most recent 50)")
    lines.append("")
    lines.append("| # | Title | Author | Labels | Merged | Branch |")
    lines.append("|---|-------|--------|--------|--------|--------|")
    for pr in sorted(merged_prs, key=lambda x: x["merged_at"] or "", reverse=True)[:50]:
        title  = pr["title"].replace("|", "｜")
        author = pr["user"]["login"].replace("_microsoft", "") if pr.get("user") else ""
        labels = fmt_labels(pr.get("labels", []))
        branch = pr.get("head", {}).get("ref", "")
        lines.append(
            f"| #{pr['number']} | {title} | {author} | {labels} "
            f"| {fmt_date(pr['merged_at'])} | `{branch}` |"
        )
    lines.append("")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nWritten {len(lines)} lines to {OUT_FILE}")
print("\nMilestone counts:")
for ms in ordered_ms:
    if ms not in all_ms:
        continue
    o = len(ms_open.get(ms, []))
    c = len(ms_closed.get(ms, []))
    print(f"  {ms}: {o} open, {c} closed")
