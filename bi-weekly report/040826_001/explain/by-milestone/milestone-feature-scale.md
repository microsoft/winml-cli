# Feature Scale — microsoft/ModelKit
**Period**: 2026-03-23 to 2026-04-08
**Generated**: 2026-04-08

---

## P0 Feature & Infrastructure Issues

### 202604 Release (April 14 Gate) — [12 open, 2 closed]

#### Open

| # | Title | Owner | Notes |
|---|-------|-------|-------|
| #98 | P0-INFRA-001: Runtime Tester - QDQ Support | Te Zheng (xieofxie) | No progress signal this period |
| #99 | P0-INFRA-002: QNN EP Op Coverage Opset Version Scaling | Fangyang Ci (fangyangci) | No progress signal this period |
| #101 | P0-INFRA-005: Runtime Tester INT4 Support for LLM | Te Zheng (xieofxie) | No progress signal this period |
| #102 | P0-INFRA-010: Subgraph Test Framework | Qiong Wu (DingmaomaoBJTU) | No progress signal this period |
| #103 | P0-INFRA-011: Pattern Validation | Qiong Wu (DingmaomaoBJTU) | QNN pattern rules added via #210 (2026-04-07) — likely related |
| #104 | P0-INFRA-006: Doc Parser QNN Checker Functions | Qiong Wu (DingmaomaoBJTU) | No progress signal this period |
| #105 | P0-FEATURE-009: Graph Optimizer Support Change Attributes | Te Zheng (tezheng) | No progress signal this period |
| #107 | P0-INFRA-013: Graph Optimizer Normalize Input ONNX Models | Te Zheng (tezheng) | Input constraints normalization landed via #23 (recent) |
| #108 | P0-FEATURE-006: Profiling | Te Zheng (tezheng), Te Zheng (xieofxie) | No progress observed this period |
| #109 | P0-FEATURE-011: Data Driven Auto-fix with Agent | Fangyang Ci (fangyangci) | No progress observed this period |
| #113 | P0-TEST-004: QDQ Validation Tests | Te Zheng (xieofxie) | QDQ config updated for P1 models via #236 (2026-04-08) — adjacent activity |
| #114 | P0-TEST-005: Graph Optimizer Validation | Qiong Wu (DingmaomaoBJTU) | No progress signal this period |

#### Closed This Period

| # | Title | Owner | Closed | Notes |
|---|-------|-------|--------|-------|
| #96 | P0-EP-009: OV NPU Subgraph Pattern Testing | Charles Zhang (chinazhangchao) | 2026-04-01 | Completed within period |
| #112 | P0-TEST-003: Analyzer Validation Tests | Qiong Wu (DingmaomaoBJTU) | 2026-04-08 | First test-track P0 closed; SA E2E eval framework (PR #222) and feature extraction eval (PR #190) contributed to this track |

---

### 202605 Release (May 1) — P1 Feature Items

| # | Title | Owner | Status | Notes |
|---|-------|-------|--------|-------|
| #153 | P1-FEATURE-DEBUG-001: Debug Command Interactive Mode | Te Zheng (tezheng) | Open | Not yet started |
| #154 | P1-FEATURE-007: Graph Optimizer QLinear * Rewrite | Te Zheng (tezheng) | Open | Not yet started |
| #155 | P1-FEATURE-002: wmk perf Command Improve Existing | Te Zheng (tezheng) | Open | Not yet started; CLI renamed wmk to winml via #205 |
| #156 | P1-FEATURE-010: Build Report Stage-by-Stage | Te Zheng (tezheng) | Open | Not yet started |
| #158 | P1-FEATURE-013: Profiling Integration IHV Tools | Te Zheng (tezheng), Te Zheng (xieofxie) | Open | Blocked on #108 (Profiling) |
| #159 | P1-INFRA-003: CI/CD Pipeline ADO + GitHub Actions + Self-hosted | Zhipeng Wang (timenick), Yue Sun (KayMKM) | Open | Not yet started |

---

### 202606+ Post Build — Feature Items

| # | Title | Owner | Priority | Notes |
|---|-------|-------|----------|-------|
| #100 | P0-INFRA-003: Runtime Tester High Priority Op Types | Unassigned | P0 | Unowned; no triage action observed |
| #106 | P0-FEATURE-010: Graph Optimizer Advanced Optimizations | Unassigned | P0 | Unowned; no triage action observed |
| #152 | P1-FEATURE-GGUF: GGUF Format Load & Convert to ONNX | Te Zheng (tezheng) | P1 | Deferred post-build |

---

## Feature Progress Summary

| Track | Open | Closed | Completion |
|-------|------|--------|------------|
| 202604 P0 Feature & Infra (INFRA + FEATURE) | 10 | 0 | 0% |
| 202604 P0 Test | 2 | 1 | 33% |
| 202604 P0 EP | 0 | 1 | 100% |
| **202604 Total** | **12** | **2** | **14.3%** |
| 202605 P1 Feature | 6 | 0 | 0% |
| 202606+ Post Build | 3 | 0 | 0% |

---

## Key Observations

- The 202604 release gate (April 14) has 12 open P0 items with 6 days remaining; at 14.3% completion the release is at significant risk, with no progress signal on high-complexity items like #108 (Profiling) and #109 (Data Driven Auto-fix with Agent).
- Two P0 post-build items (#100, #106) remain unassigned with no triage action, creating owner gaps that will carry forward into the 202606+ planning cycle.
- Te Zheng (tezheng / xieofxie) is the sole or primary owner on 8 of the 12 open 202604 P0 items plus 5 of 6 open 202605 P1 items, representing a concentration risk if any items require parallel execution before the gate.
- The two closures this period (#96 OV NPU Subgraph Pattern Testing, #112 Analyzer Validation Tests) both fall in the EP and test tracks; no core INFRA or FEATURE P0 items have closed yet, which is the dominant gap against the April 14 gate.
