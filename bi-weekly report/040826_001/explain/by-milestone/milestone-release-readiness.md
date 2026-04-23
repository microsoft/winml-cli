# Release Readiness — microsoft/ModelKit
**Period**: 2026-03-23 to 2026-04-08
**Generated**: 2026-04-08

Release readiness tracks the legal, security, compliance, CI/CD, and community issues required before the repository can be made public. The April 14 Gate (`202604 Release`) is the immediate legal/security deadline — now 6 days away.

---

## Release Issue Summary

| # | ID | Title | Milestone | Owner | Status |
|---|----|-------|-----------|-------|--------|
| #164 | P0-REL-L | Legal & License Review — MIT License + Dependency Audit | `202604 Release` | Zhipeng Wang | Closed 2026-04-08 |
| #165 | P0-REL-S | Security Review — Static Analysis, Secret Scan, Vulnerability Scan, SBOM | `202604 Release` | Zhipeng Wang | Closed 2026-04-08 |
| #168 | P0-REL-R | Repository & CI/CD Setup — Branch Protection, GitHub Actions, PyPI Workflow | `202604 Release` | Zhipeng Wang | Open |
| #169 | P0-REL-O | Open-Source Community Readiness — CODE_OF_CONDUCT, Versioning, Roadmap | `202604 Release` | Zhipeng Wang | Open |
| #170 | P0-REL-E | Release Execution — Tag, PyPI Publish, Announce, Internal Docs Archive | `202604 Release` | Zhipeng Wang | Open |
| #166 | P0-REL-C | Code Quality & Public API — Ruff Clean, `__all__`, Internal Code Cleanup | `202605 Release` | Te Zheng, Zhipeng Wang, Qiong Wu | Open |
| #167 | P0-REL-D | Documentation — README, CONTRIBUTING, CHANGELOG, User Guide, CLI Reference | `202605 Release` | Te Zheng, Brenda | Open |
| #171 | ESRP-03 | Validate signed artifacts — verify signatures and test install | `(no milestone)` | Zhipeng Wang | Open |
| #172 | TR-01 | Complete Export Control Classification (ECCN) | `(no milestone)` | Zhipeng Wang | Open |
| #173 | TR-02 | Submit Trade Compliance Review via internal Trade portal | `(no milestone)` | Zhipeng Wang | Open |
| #174 | TR-03 | Obtain Trade sign-off for public open-source distribution | `(no milestone)` | Zhipeng Wang | Open |
| #90 | — | Release a wheel for more user to bug bash | `202603 Release` | Te Zheng | Closed 2026-04-08 |
| #112 | P0-TEST-003 | Analyzer Validation Tests | `(various)` | Qiong Wu | Closed 2026-04-08 |

---

## Track A — April 14 Gate (202604 Release)

Two of five gate issues are now closed. Three remain open. All are owned by Zhipeng Wang.

### #164 — P0-REL-L: Legal & License Review (CLOSED 2026-04-08)

- **Scope**: Confirm MIT License file is present and correct; audit all third-party dependencies for license compatibility; document attributions.
- **Dependencies**: None
- **Status**: Closed 2026-04-08. License review is complete. Related work includes PR #227 (license header CI check, merged 2026-04-02) and the earlier Apache 2.0 attribution for HuggingFace code (#10, closed 2026-03-26).

### #165 — P0-REL-S: Security Review (CLOSED 2026-04-08)

- **Scope**: Run static security analysis (e.g., CodeQL), scan for secrets/credentials in history, perform vulnerability scan on dependencies, generate SBOM.
- **Dependencies**: Depends on #164 (license audit should precede SBOM)
- **Status**: Closed 2026-04-08. Security review is complete. Related closed work this period: PR #227 added license header CI check; PR #254 moved network tests to integration (merged 2026-04-07); PR #252 fixed a CI hang (merged 2026-04-06).

### #168 — P0-REL-R: Repository & CI/CD Setup

- **Scope**: Configure branch protection rules on `main`; set up GitHub Actions workflows; implement PyPI publish workflow.
- **Dependencies**: Depends on #165 (security must clear before automating publish). #165 is now closed, unblocking this item.
- **Status**: Open — CI work is in progress (license header CI merged via #227; test infrastructure improved via #254 and #252). Full branch protection rules and PyPI publish workflow are not yet in place.

### #169 — P0-REL-O: Open-Source Community Readiness

- **Scope**: Add CODE_OF_CONDUCT.md, establish versioning policy (SemVer), publish public Roadmap.
- **Dependencies**: Can proceed in parallel with #164/#165, both of which are now closed.
- **Status**: Open — no PR activity observed for community readiness files during this period.

### #170 — P0-REL-E: Release Execution

- **Scope**: Tag v0.x.0 release, publish to PyPI, send announcement, archive internal documentation.
- **Dependencies**: Blocked on #168 and #169 being closed (and downstream on #166 and #167 for quality/docs). #164 and #165 have cleared.
- **Status**: Open — cannot close until all upstream release items (#168, #169) are resolved, and quality/docs work (#166, #167) is complete.

---

## Track B — May 1 Release (202605 Release)

### #166 — P0-REL-C: Code Quality & Public API

- **Scope**: Run `ruff` clean across all modules; ensure `__all__` is defined and correct for all public packages; internal code cleanup and dead code removal.
- **Owners**: Te Zheng, Zhipeng Wang, Qiong Wu
- **Status**: Open — import cleanup (phases completed in prior period) addressed some of this scope. Remaining work includes full `__all__` audit and ruff-clean verification across all modules.

### #167 — P0-REL-D: Documentation

- **Scope**: README, CONTRIBUTING guide, CHANGELOG, User Guide, CLI reference documentation.
- **Owners**: Te Zheng, Brenda
- **Status**: Open — no documentation PRs observed during the period. The CLI rename to `winml` (completed 2026-04-01) means CLI reference documentation must be written against the new command name.

---

## Track C — ESRP & Trade Compliance (no milestone)

These items are required for public open-source distribution under Microsoft's export control and release process.

| # | ID | Title | Owner | Status | Notes |
|---|----|-------|-------|--------|-------|
| #171 | ESRP-03 | Validate signed artifacts — verify signatures and test install | Zhipeng Wang | Open | Artifact signing validation; requires build artifacts to exist first |
| #172 | TR-01 | Complete Export Control Classification (ECCN) | Zhipeng Wang | Open | ECCN classification required before public distribution |
| #173 | TR-02 | Submit Trade Compliance Review via internal Trade portal | Zhipeng Wang | Open | Blocked on ECCN (#172) |
| #174 | TR-03 | Obtain Trade sign-off for public open-source distribution | Zhipeng Wang | Open | Final sign-off; blocked on #173 |

All four ESRP/trade items are sequential: #172 → #173 → #174, with #171 (artifact signing) proceeding in parallel once build artifacts are ready. None have been started.

---

## Completion by Track

| Track | Total Issues | Closed | Open | Completion |
|-------|-------------|--------|------|------------|
| Track A — April 14 Gate (#164, #165, #168, #169, #170) | 5 | 2 | 3 | 40% |
| Track B — May 1 Release (#166, #167) | 2 | 0 | 2 | 0% |
| Track C — ESRP/Trade (#171–#174) | 4 | 0 | 4 | 0% |
| **Total** | **11** | **2** | **9** | **18%** |

---

## Related Work Completed This Period (Not Counted Above)

The following closed issues and merged PRs represent progress toward release readiness, even though the primary tracking issues remain open:

| # | Item | Type | Closed/Merged | Relevance |
|---|------|------|---------------|-----------|
| #90 | Release a wheel for more user to bug bash | Issue | 2026-04-08 | Prerequisite for community adoption; contributes to #169 |
| #112 | P0-TEST-003: Analyzer Validation Tests | Issue | 2026-04-08 | Contributes to code quality (#166) |
| PR #227 | License header CI check | PR | 2026-04-02 | Supported #165 (security/compliance CI) |
| PR #254 | Move network tests to integration | PR | 2026-04-07 | Supported #168 (CI/CD setup) |
| PR #252 | Fix CI hang | PR | 2026-04-06 | Supported #168 (CI/CD setup) |
| #96 | OV NPU subgraph pattern testing | Issue | 2026-04-01 | Test coverage improvement; contributes to #166 |
| #117 | wmk hub CLI | Issue | 2026-04-01 | CLI feature work completed before rename to winml |

---

## Risk Assessment

- **April 14 Gate is 6 days away** (from report date 2026-04-08). Two of five Track A issues are now closed (#164, #165), which is meaningful progress. Three gate issues remain open (#168, #169, #170), and #170 (Release Execution) cannot close until the other two are done.
- **#168 (Repository & CI/CD Setup) is the next critical path item**. With #165 now closed, this is unblocked. However, branch protection and PyPI publish workflows are not yet in place, and the remaining time before April 14 is tight.
- **Release Execution (#170) is still end-to-end blocked** on #168 and #169. Even after those close, #170 requires additional work (tagging, PyPI publish, announcement).
- **Zhipeng Wang carries the entire Track A and Track C workload** — single-threaded ownership across five open issues in six days creates significant schedule risk.
- **ESRP/Trade (#171–#174) are completely unstarted** and have no milestone. They are likely not gate conditions for April 14 but will be required before any public announcement or broad distribution.
- **Documentation (#167) has no observed PR activity**; with the CLI rename to `winml` complete, documentation must be written against the new command name.
