# Milestone Overview — microsoft/ModelKit
**Period**: 2026-03-23 to 2026-04-08
**Generated**: 2026-04-08

---

## Milestone Mapping

| Milestone | Open | Closed | Total |
|---|---|---|---|
| 202603 Release | 0 | 2 | 2 |
| 202604 Release | 33 | 7 | 40 |
| 202605 Release | 30 | 0 | 30 |
| 202606+ Post Build | 20 | 0 | 20 |
| (no milestone) | 61 | 33 | 94 |
| **Total** | **144** | **42** | **186** |

---

## P0 Issue Dashboard

| Milestone | P0 Open | P0 Closed | Total P0 |
|---|---|---|---|
| 202603 Release | 0 | 1 | 1 |
| 202604 Release | 19 | 4 | 23 |
| 202605 Release | 0 | 0 | 0 |
| 202606+ Post Build | 2 | 0 | 2 |
| (no milestone) | 0 | 0 | 0 |
| **Total** | **21** | **5** | **26** |

Notes:
- 202603 Release P0 closed: #110 (QNN hardware access, closed 2026-04-08)
- 202604 Release P0 open: #52, #94, #97, #98, #99, #101, #102, #103, #104, #105, #107, #108, #109, #111, #113, #114, #168, #169, #170
- 202604 Release P0 closed: #96 (OV NPU subgraph pattern testing, 2026-04-01), #112 (Analyzer Validation Tests, 2026-04-08), #164 (Legal Review, 2026-04-08), #165 (Security Review, 2026-04-08)
- 202606+ Post Build P0 open: #100, #106

---

## Per-Milestone Snapshot

### 202603 Release — Fully Resolved (Open: 0, Closed: 2)

This milestone is now fully closed as of 2026-04-08. Both remaining issues were closed today: #90 (wheel release) and #110 (QNN hardware access). Three issues that had been tracked under this milestone — #68 (ESRGAN), #86 (slow init), and #87 (final model naming) — were moved to the 202604 milestone prior to closure. No open items remain.

### 202604 Release — April 14 Gate (Open: 33, Closed: 7)

The most active milestone and the team's immediate deadline. Seven of 40 issues are now closed, up from 4 at the previous snapshot. The three newly closed items this period are #112 (Analyzer Validation Tests), #164 (Legal Review), and #165 (Security Review), all closed 2026-04-08. The closure of legal and security gate reviews represents meaningful progress toward the April 14 release gate. However, 33 issues remain open, including critical infrastructure items: CI/CD pipeline setup (#168), community readiness files (#169), and release execution (#170). All EP scale, feature scale, and infrastructure work is still in progress. The three release execution items (#168, #169, #170) are all assigned to Zhipeng Wang and remain on the critical path.

### 202605 Release — May 1 (Open: 30, Closed: 0)

No closures this period. All 30 issues remain open. No active movement observed; work on this milestone is likely gated on completion of the 202604 gate items.

### 202606+ Post Build — Post-Build / June+ (Open: 20, Closed: 0)

No closures this period. All 20 issues remain open. This milestone covers longer-horizon work including AMD NPU enablement, OpenVINO GPU/CPU EP coverage, TensorRT GPU, GGUF format support, and advanced graph optimizations. Two P0 issues (#100, #106) remain open here with no progress signal observed.

### (no milestone) — Unscheduled (Open: 61, Closed: 33)

The largest and most active bucket. The 33 closed issues during the period represent a significant burst of activity concentrated in import cleanup, bug bash fixes, CI infrastructure, and test stabilization. The 61 open issues include all 23 model-family tracking issues (#118–#140), ESRP and trade compliance items (#171–#174), and a large set of bug bash items filed in the 2026-04-01 and 2026-04-07 waves. All 23 model-family tracking issues are unstarted.

---

## Risk Register

| Risk | Milestone | Owner | Status |
|---|---|---|---|
| OV NPU hardware (Meteor Lake / Lunar Lake) not available — blocks OV E2E validation — #111 | 202604 Release | Yue Sun (KayMKM) | Open |
| CI/CD pipeline not in place — branch protection, GitHub Actions, PyPI workflow — #168, deadline April 14 | 202604 Release | Zhipeng Wang | Open |
| Community readiness files not added — #169, deadline April 14 | 202604 Release | Zhipeng Wang | Open |
| Release execution (#170) blocked on #168 and #169 | 202604 Release | Zhipeng Wang | Open |
| VitisAI NPU pass rate 50.5% (dropped from 55.6%) — now lowest EP | 202604 Release | — | Degrading |
| 23/23 model-family tracking issues unstarted (#118–#140) | (no milestone) | Various / Unassigned | Not started |
| ESRP / Trade compliance not started (#171–#174) | (no milestone) | Zhipeng Wang | Not started |
