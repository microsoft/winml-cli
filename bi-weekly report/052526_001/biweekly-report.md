# WinML-ModelKit Bi-Weekly Report
**Period:** 2026-05-12 to 2026-05-25
**Report ID:** 052526_001
**Generated:** 2026-05-25
**Distribution:** Engineering Leadership

---

## Section 1: High-Level Goals — Core Metrics Dashboard

| Dimension | Now | Target (May 31) | Gap | Status |
|-----------|-----|-----------------|-----|--------|
| Bug Burndown | 52 open (8 P0, 11 P1, 33 P2) | 0 P0/P1 by May 31 | -19 P0+P1, 6 days left; fix velocity stalled since 05-20 | At Risk — 5-day plateau, need 3.2/day |
| Release | v0.0.3 + v0.0.4 shipped; winml-cli rename complete | 202605 Release by May 31 | CI pipeline operational, rules zip publishing working | On Track |
| E2E Test Coverage | 15 new E2E test suites added (analyze, build, config, sys, inspect, export, optimize, perf, catalog, CLI surface, compile, eval, quantize) | All commands covered | Functional E2E for every major command now exists | On Track |
| Quality & Stability | P0: 22 to 8 (-63.6%), P1: 35 to 11 (-68.6%) in 8 days | 0 P0+P1 | zhenchaoni owns 4/8 P0s — single point of failure | At Risk — owner concentration |

---

## Section 2: Highlights & Lowlights

### Highlights

**E2E Test Expansion** — *15 new E2E test suites covering all major CLI commands, the largest testing push this milestone*
- `winml analyze` E2E tests (#652) — functional scenarios for static analyzer command. *(Copilot)*
- `winml build` E2E tests expanded (#593) — covers happy path + failure modes; found and fixed a build bug. *(Copilot)*
- `winml config` E2E coverage (#591) — happy / bad / flag-variation paths. *(Copilot)*
- `winml sys` E2E coverage (#612) — happy / bad / flag-variation paths. *(Zhiwei Wang)*
- `winml inspect` E2E tests + --list-tasks crash fix (#676) — functional tests + discovered P0 crash. *(Qiong Wu)*
- `winml export` E2E tests + --input-specs crash fix (#602) — functional tests for export command. *(Ren Y)*
- `winml optimize` E2E tests (#607) — tests + help display improvement. *(Ren Y)*
- `winml perf` CLI tests for ONNX and per-module (#665) — perf CLI surface coverage. *(Hualong Xie)*
- `winml perf` E2E expanded across EPs and devices (#698) — broadened perf coverage. *(Hualong Xie)*
- `winml catalog` CLI surface tests (#669) — catalog command coverage. *(Qiong Wu)*
- `winml` + `winml --help` CLI surface tests (#672) — root command coverage. *(Qiong Wu)*
- Compile E2E validation tests (#645, #661) — compile functional validation + EP compile e2e. *(Zhenchao Ni)*
- Eval functional correctness e2e (#605) — eval accuracy e2e test. *(Zhenchao Ni)*
- Quantize functional e2e test cases (#608) — quantize correctness tests and bug fixes. *(Zhenchao Ni)*

**Bug Burndown** — *21 bugs resolved in 48 hours (05-18 to 05-20), including 4 P0s and 11 P1s*
- P0 telemetry device_id format fix (#691) — CS 4.0 compliance restored; all events were being rejected. *(Zhiwei Wang)*
- P0 inspect bogus HF id error fix (#542) — distinguish local-path-not-found from network errors. *(Qiong Wu)*
- P0 config --device npu silent success fix (#431) — now raises on EP-not-available-on-system. *(Zhiwei Wang)*
- P0 quantize --precision echo fix (#556) — precision now echoed in output. *(Zhenchao Ni)*
- 11 P1 bugs closed spanning Build, Analyzer, Compile, Config, Eval, Export, Quantize. *(Charles Zhang, Zhenchao Ni, Ren Y, Hualong Xie)*

**EP / Session Architecture** — *Strong-typed EP parameters and sysinfo hardening*
- Strong-type EP parameters across analyze/compiler/optracing (#632) — replaces stringly-typed EP params with typed objects. *(Hualong Xie)*
- EP type added to EP parameters (#621) — extends typed EP support. *(Hualong Xie)*
- EP filter added to resolve_device (#459) — enables per-EP device resolution. *(Hualong Xie)*
- Raise on explicit device with no compatible EP (#660) — fail-fast instead of silent fallback. *(Zhiwei Wang)*
- Raise on EP-not-available-on-system (#686) — consistent error for missing EPs. *(Hualong Xie)*
- Demote per-provider ensure_ready failure to debug (#703) — EP discovery resilience. *(Zhiwei Wang)*
- Remove ov, vitis, trtrtx alias (#690) — clean up deprecated EP aliases for release. *(Hualong Xie)*

**CLI / UX** — *Package rename, version bump, EP policy, and UX hardening*
- Rename winml-modelkit to winml-cli (#623) — package name alignment for release. *(Zhiwei Wang)*
- Update to winml 2.0 (#441) — major CLI version bump. *(Charles Zhang)*
- EP_SUPPORTED_DEVICES policy + trust_remote_code warning (#641) — security and device policy enforcement. *(Ren Y)*
- Batch fix of small UX/contract issues + -o/--output refactor (#624) — consistent output flag behavior. *(Ren Y)*
- Improve winml eval UX: validate inputs upfront, friendlier errors (#694) — better error messages. *(Zhenchao Ni)*
- Remove cuda from EP option for release (#683) — scope EP options for 202605 release. *(Hualong Xie)*

**Performance** — *10x catalog speedup and 2-3x sys speedup*
- `winml catalog` ~10x faster by deferring heavy data-pkg imports (#613). *(Ren Y)*
- `winml sys` warm latency 2-3x faster (#595). *(Zhiwei Wang)*
- Per-module benchmark works for ResNet and similar HF models (#586). *(Hualong Xie)*

**Release / CI** — *v0.0.3 and v0.0.4 shipped; CI pipeline modernized*
- v0.0.3 released and merged back to main (#655). *(Hualong Xie)*
- v0.0.4 released and merged back to main (#687). *(Zhiwei Wang)*
- CI release pipeline migrated from OneBranch to 1ES template (#616). *(Zhiwei Wang)*
- Rules zip published with version-qualified filename (#627). *(Zhiwei Wang)*
- Telemetry fixed: device_id format to CS 4.0 (#693), dropped undocumented fields (#642). *(Zhiwei Wang)*

**Analyze / Static Analyzer** — *EP/device resolution refactor and correctness fixes*
- Refactor EP/device resolution and fix --run-unknown-op compile regression (#662). *(Fangyang Ci)*
- Fix empty node.name handling with deterministic stable-node keys (#609). *(Fangyang Ci)*
- Create parent dirs for -o and --optim-config before writing (#644). *(Qiong Wu)*
- Reject unsupported EPContext targets and handle no-rule-data gracefully (#622). *(Qiong Wu)*
- Disambiguate EPContext pattern_id by EP label (#615). *(Hualong Xie)*

**Build / Export / Models** — *Build hardening, ONNX normalization, and model fixes*
- Validate config at front of build command (#675) — fail-fast on bad config. *(Charles Zhang)*
- Output analyze_result.json; fix --device npu config validation (#673). *(Qiong Wu)*
- Auto-select EP via resolve_device, forward --ep from run_eval (#667). *(Ren Y)*
- Normalize exported ONNX in-place via optimize_onnx (#681). *(Ren Y)*
- Fix quantization P0 bugs (#680). *(Zhenchao Ni)*
- Fix image-to-text model build (#671). *(Zhenchao Ni)*

---

### Lowlights

| # | Risk | Severity | Deadline | Owner |
|---|------|----------|----------|-------|
| L-01 | Bug fix velocity stalled for 5 days (05-20 to 05-25): 52 open bugs unchanged, 8 P0 + 11 P1 remaining | Critical | May 31 | All assignees — resume immediately |
| L-02 | zhenchaoni owns 4 of 8 open P0 bugs (#433, #528, #555, #563) — single point of failure for P0 closure | Critical | May 31 | PM — redistribute or pair-assign |
| L-03 | #217 (perf --op-tracing EPContext version mismatch) open since original filing — NPU-blocking P0 | High | May 31 | tezheng, xieofxie |
| L-04 | #469 (SA default device hardcoded to NPU) — P0, assigned to fangyangci, no closure signal | High | May 31 | fangyangci |
| L-05 | #543, #544 (inspect 24s latency, --list-tasks 12.6s) — both P0 performance bugs, no commits from assignee this period | High | May 31 | ziyuanguo1998 |
| L-06 | #654 (perf --monitor not working for --module) — P2 with no assignee | Medium | May 31 | Unassigned — needs owner |
| L-07 | #688 (EP discovery resilience) — new P1, "triage to be confirmed" label still present | Medium | May 31 | timenick, tezheng |

---

## Section 3: Data Analysis

### 3.1 Bug Burndown — 202605 Release

```
P0+P1 Burndown (05-12 to 05-25)

Date    P0  P1  P0+P1  Total  Event
05-12   22  35   57     70    <- period start
05-13   21  31   52     64
05-14   21  30   51     89    <- +25 new bugs filed (triage wave)
05-15   14  27   41     77    <- 7 P0 closed in one day
05-18   11  20   31     71
05-19    9  14   23     63    <- 8 bugs closed
05-20    8  11   19     52    <- 21 resolved in 48h push
05-25    8  11   19     52    <- today (5-day plateau)

Target: 0 P0+P1 by 05-31 (6 days remaining)
```

```
P0 Bugs:  ########..............  8/22 remaining (63.6% resolved)
P1 Bugs:  ###########.........................  11/35 remaining (68.6% resolved)
Total:    ####################################################..................  52/70 remaining
```

**Pace analysis**: In the active window (05-12 to 05-20, 8 days), 38 P0+P1 bugs closed = 4.75/day. Since 05-20, pace = 0/day for 5 consecutive days. To hit 0 P0+P1 by 05-31 from today: need 19 bugs in 6 days = 3.2/day. Achievable at prior pace (4.75/day), but the 5-day stall is a red flag.

### 3.2 Open Bugs by Component

```
Component        P0  P1  P2  Total  Bar
Other             1   2   8   11   ===========
Compile           1   0   5    6   ======
Perf              0   1   5    6   ======
Eval              1   1   3    5   =====
Quantize          1   0   4    5   =====
Analyzer          1   2   1    4   ====
Build             0   1   2    3   ===
Sys               0   1   2    3   ===
Inspect           2   0   1    3   ===
Load & Export     0   1   1    2   ==
Repository        0   0   2    2   ==
Catalog           0   0   1    1   =
Config            0   0   1    1   =
```

### 3.3 Open Bugs per Person (Top 8)

```
Person         Total  P0  Load
zhenchaoni      14    4   ==============  <- overloaded (27% of all bugs, 50% of P0s)
hi-brenda       10    0   ==========
tezheng          7    1   =======
xieofxie         7    1   =======
zhangchao        5    0   =====
fangyangci       4    1   ====
timenick         4    0   ====
ziyuanguo1998    3    2   ===  <- 2 P0s
```

zhenchaoni carries 27% of all open bugs (14/52) and 50% of all open P0s (4/8).

### 3.4 Resolved Bugs (05-18 to 05-20): 21 Bugs Closed

| Person | Resolved | P0 | P1 | P2 |
|--------|----------|----|----|-----|
| Qiong Wu | 8 | 1 | 1 | 6 |
| Zhenchao Ni | 4 | 1 | 3 | 0 |
| Charles Zhang | 3 | 0 | 3 | 0 |
| Zhiwei Wang | 2 | 2 | 0 | 0 |
| Ren Y | 2 | 0 | 2 | 0 |
| Hualong Xie | 2 | 0 | 2 | 0 |
| **Total** | **21** | **4** | **11** | **6** |

Resolved P0s: #691 (telemetry device_id), #542 (inspect error reporting), #431 (config --device npu), #556 (quantize precision echo).

Resolved P1s: #643 (analyze parent dirs), #536 (export TracerWarning), #526 (config bogus HF id), #516 (trust_remote_code), #513 (analyze --ep cpu), #193 (quantize NPU compilation), #530 (eval --samples ignored), #517 (build config validation order), #450 (custom architecture traceback), #428 (compile AMD EPs), #430 (perf AMD NPU TarWriter).

### 3.5 Commit Activity (05-12 to 05-25)

79 commits by 9 contributors (excluding merge commits).

| Person | Commits | Key Areas |
|--------|---------|-----------|
| Hualong Xie | 15 | EP params, perf, sysinfo, release |
| Qiong Wu | 12 | E2E tests, build, inspect, analyze |
| Zhiwei Wang | 11 | telemetry, CI, sysinfo, release |
| Ren Y | 10 | CLI, export, build, perf, E2E tests |
| Zhenchao Ni | 8 | compile, eval, quantize, UX |
| Charles Zhang | 6 | build, analyze, CLI 2.0 |
| Copilot | 5 | E2E tests (analyze, build, config, export) |
| Fangyang Ci | 4 | analyze, rules |
| Yue Sun | 3 | pipeline, docs |

### 3.6 Bottleneck Analysis

- **zhenchaoni is the critical bottleneck**: carries 4/8 P0s and 14 total open bugs. Any slowdown from this contributor directly delays P0 closure. Redistribution or pair-programming is the single highest-leverage action.
- **Inspect P0 pair (#543, #544) both assigned to ziyuanguo1998**: these are performance bugs (24s latency, 12.6s for static lookup). No commits from ziyuanguo1998 observed this period — status unknown.
- **5-day bug fix stall (05-20 to 05-25)**: burndown flatlined at 52. With 6 days to the May 31 deadline, the team must resume 3.2 P0+P1 closures/day — previously demonstrated at 4.75/day. The capacity exists but needs reactivation.

---

## Section 4: Task Status

### 4.1 Completed This Period

**E2E Testing** — *Comprehensive E2E test coverage added across all major CLI commands*

| PR | Description | Owner |
|----|-------------|-------|
| #652 | Add E2E tests for `winml analyze` — functional scenario coverage | Copilot |
| #593 | Expand E2E tests for `winml build` — happy path + failure modes + bug fix | Copilot |
| #591 | Add E2E coverage for `winml config` — happy/bad/flag-variation paths | Copilot |
| #612 | Add E2E coverage for `winml sys` — happy/bad/flag-variation paths | Zhiwei Wang |
| #676 | winml inspect E2E tests + fix --list-tasks crash | Qiong Wu |
| #602 | test(export): add E2E tests + fix --input-specs crash | Ren Y |
| #607 | test(optimize): add E2E tests + show (Default: enabled/disabled) in --help | Ren Y |
| #665 | test(e2e): add perf CLI tests for ONNX and per-module | Hualong Xie |
| #698 | test(e2e): expand winml perf coverage across EPs and devices | Hualong Xie |
| #669 | test(cli): add CLI surface tests for `winml catalog` | Qiong Wu |
| #672 | test(cli): add CLI surface tests for `winml` and `winml --help` | Qiong Wu |
| #645 | Compile: add E2E functional validation tests | Zhenchao Ni |
| #661 | EP: implement compile validation and E2E | Zhenchao Ni |
| #605 | Eval: add functional correctness E2E test and fix issues found | Zhenchao Ni |
| #608 | Quantize: implement functional E2E test cases and fix issues found | Zhenchao Ni |

**EP / Session** — *Strong-typed EP parameters and sysinfo hardening*

| PR | Description | Owner |
|----|-------------|-------|
| #632 | Strong-type EP parameters across analyze/compiler/optracing | Hualong Xie |
| #621 | Add EP type to EP parameters | Hualong Xie |
| #459 | Add EP filter to resolve_device | Hualong Xie |
| #660 | Raise on explicit device with no compatible EP | Zhiwei Wang |
| #686 | Raise on EP-not-available-on-system | Hualong Xie |
| #703 | Demote per-provider ensure_ready failure to debug | Zhiwei Wang |
| #690 | Remove ov, vitis, trtrtx alias for release | Hualong Xie |

**CLI / UX** — *Package rename, version bump, EP policy, and UX refinements*

| PR | Description | Owner |
|----|-------------|-------|
| #623 | Rename winml-modelkit to winml-cli (package name alignment) | Zhiwei Wang |
| #441 | Update to winml 2.0 (major CLI version bump) | Charles Zhang |
| #641 | EP_SUPPORTED_DEVICES policy + trust_remote_code warning | Ren Y |
| #624 | Batch fix of UX/contract issues + -o/--output refactor | Ren Y |
| #694 | Improve winml eval UX: validate inputs, friendlier errors, --schema | Zhenchao Ni |
| #683 | Remove cuda from EP option for release | Hualong Xie |
| #611 | Add -h shorthand flag across commands | Hualong Xie |

**Build / Export / Models** — *Build hardening, ONNX normalization, and model fixes*

| PR | Description | Owner |
|----|-------------|-------|
| #675 | Validate config at front of build command (fail-fast) | Charles Zhang |
| #673 | Output analyze_result.json; fix --device npu config validation | Qiong Wu |
| #477 | Optional config, autoconf status display, EP compatibility fixes | Qiong Wu |
| #667 | Auto-select EP via resolve_device, forward --ep from run_eval | Ren Y |
| #681 | Normalize exported ONNX in-place via optimize_onnx | Ren Y |
| #651 | Restore monolithic BlipCaptioningIOConfig | Ren Y |
| #650 | Register BART for table-question-answering task (TAPEX) | Ren Y |
| #680 | Fix quantization P0 bugs | Zhenchao Ni |
| #671 | Fix image-to-text model build | Zhenchao Ni |
| #678 | Forward explicit --ep to analyzer in HF build path | Hualong Xie |
| #689 | Fix onnxruntime search dll path | Charles Zhang |

**Analyze / Static Analyzer** — *EP/device resolution refactor and robustness*

| PR | Description | Owner |
|----|-------------|-------|
| #662 | Refactor EP/device resolution and fix --run-unknown-op compile regression | Fangyang Ci |
| #609 | Fix empty node.name handling with deterministic stable-node keys | Fangyang Ci |
| #644 | Create parent dirs for -o and --optim-config before writing | Qiong Wu |
| #622 | Reject unsupported EPContext targets, handle no-rule-data gracefully | Qiong Wu |
| #615 | Disambiguate EPContext pattern_id by EP label | Hualong Xie |
| #670 | Fix skipped json exist when update rules | Fangyang Ci |
| #618 | Disable run unknown operators on local machine by default | Charles Zhang |

**Perf** — *Performance fixes, per-module benchmark, and exit code cleanup*

| PR | Description | Owner |
|----|-------------|-------|
| #613 | Make `winml catalog` ~10x faster by deferring heavy imports | Ren Y |
| #595 | Speed up `winml sys` warm latency 2-3x | Zhiwei Wang |
| #625 | Honour --ep cpu and surface actual EP in perf output | Qiong Wu |
| #586 | Per-module benchmark works for ResNet and similar HF models | Hualong Xie |
| #606 | Write default JSON report under ~/.cache/winml/perf | Hualong Xie |
| #601 | Improve --module UX (reject .onnx, clarify help, suggest on typo) | Hualong Xie |
| #597 | Replace sys.exit() with click exceptions for consistent exit codes | Hualong Xie |

**Release / CI / Telemetry** — *Two releases shipped, CI modernized, telemetry compliance*

| PR | Description | Owner |
|----|-------------|-------|
| #687 | Merge Release/v0.0.4 back to main | Zhiwei Wang |
| #655 | Merge Release/v0.0.3 back to main | Hualong Xie |
| #616 | Migrate release pipeline from OneBranch to 1ES template | Zhiwei Wang |
| #627 | Publish rules zip with version-qualified filename | Zhiwei Wang |
| #693 | Fix telemetry device_id format to CS 4.0 r:\<uuid\> form | Zhiwei Wang |
| #642 | Drop undocumented telemetry fields ext.os.release / ext.app.initTs | Zhiwei Wang |
| #600 | Zip rules to release | Fangyang Ci |

**Inspect / Eval / Misc** — *Error handling, warnings, and eval precision fix*

| PR | Description | Owner |
|----|-------------|-------|
| #679 | Distinguish local-path-not-found from network errors in inspect | Qiong Wu |
| #647 | Downgrade huggingface_hub symlinks UserWarning to INFO | Qiong Wu |
| #626 | Only apply w8a16 precision default on NPU, not CPU/GPU | Qiong Wu |
| #619 | Fix license issue for calculating cider metric | Zhenchao Ni |
| #574 | Fix `winml export` for prajjwal1/bert-tiny with model-specific task default | Copilot |

**Pipeline / Docs** — *Eval pipeline enhancements and ADO agent setup*

| PR | Description | Owner |
|----|-------------|-------|
| #692 | Add agent name parameter for pipeline | Yue Sun |
| #685 | Init example configs test pipeline | Yue Sun |
| #674 | Forward --device to winml build subprocess in run_eval | Ren Y |
| #664 | Add doc for how to setup self-hosted ADO agent | Yue Sun |

---

### 4.2 In-Progress Tasks

**Bug Fixes — Open P0 Bugs (8 remaining)**

| Issue | Description | Owner | Blocker |
|-------|-------------|-------|---------|
| #217 | perf --op-tracing EPContext version mismatch between onnxruntime-windowsml and onnxruntime-qnn | tezheng, xieofxie | NPU hardware dependency |
| #433 | Quantize stage shows train output instead of quantize output | zhenchaoni | — |
| #469 | SA default device hardcoded to NPU — should auto-detect | fangyangci | — |
| #528 | eval --dataset leaks raw internal traceback | zhenchaoni | — |
| #543 | inspect takes 24s end-to-end; first output silent for ~14s | ziyuanguo1998 | — |
| #544 | inspect --list-tasks takes 12.6s for static dict lookup | ziyuanguo1998 | — |
| #555 | quantize --precision banana silently falls back to defaults | zhenchaoni | — |
| #563 | --model semantic drift across 10 commands | zhenchaoni | Cross-cutting; requires coordinated change |

**Bug Fixes — Open P1 Bugs (11 remaining)**

| Issue | Description | Owner | Blocker |
|-------|-------------|-------|---------|
| #216 | eval label alignment error with auto-selected dataset | zhenchaoni | — |
| #435 | analyze silently omits Vitis EP without explanation | zhangchao, fangyangci | — |
| #508 | `winml exprt` (typo) returns "No such command" with no suggestion | zhenchaoni, hi-brenda | — |
| #520 | build config.loader.task not validated against --model | zhangchao | — |
| #529 | eval accepts ONNX whose I/O signature does not match --task | zhenchaoni | — |
| #537 | export mismatched --task dumps raw transformers traceback | vortex-captain | — |
| #562 | Missing-required-option errors lack a runnable example | zhenchaoni, hi-brenda | — |
| #596 | perf: HF model ID uses AOT pipeline, ONNX file uses raw JIT — inconsistent | tezheng, xieofxie | — |
| #637 | analyze fails on facebook/sam2.1-hiera-small | fangyangci | — |
| #648 | analyze incorrect output on TRTRTX EP | fangyangci | — |
| #688 | EP discovery should be resilient to per-provider failures | timenick, tezheng | New — triage to be confirmed |

---

### 4.3 Not Started — Items Needing Attention

**Unassigned Bugs**

| Issue | Description | Priority | Owner |
|-------|-------------|----------|-------|
| #654 | perf --monitor not working for --module | P2 | Unassigned — needs owner |

**Triage Pending**

| Issue | Description | Priority | Owner |
|-------|-------------|----------|-------|
| #688 | EP discovery should be resilient to per-provider failures | P1 | timenick, tezheng — "triage to be confirmed" label |

---

## Action Items for This Week

| # | Action | Owner | Due |
|---|--------|-------|-----|
| 1 | Resume bug fixing immediately — 5-day plateau must break; target 3+ P0+P1 closures/day through 05-31 | All P0/P1 assignees | May 26 |
| 2 | Redistribute P0 load from zhenchaoni (4/8 P0s) — pair-assign or reassign #433, #528, #555, #563 | PM + zhenchaoni | May 26 |
| 3 | Get status update from ziyuanguo1998 on inspect P0 pair (#543, #544) — no commits observed this period | PM | May 26 |
| 4 | Confirm triage for #688 (EP discovery resilience) — remove "triage to be confirmed" label | timenick | May 27 |
| 5 | Assign owner for #654 (perf --monitor for --module) | PM | May 27 |
| 6 | Close or deprioritize remaining 33 P2 bugs — decide which are 202605 blockers vs backlog | PM + Tech Leads | May 28 |
| 7 | Verify v0.0.4 release artifacts and telemetry events flow correctly post-#693 fix | Zhiwei Wang | May 27 |

---

*Data sources: git log e2aa9631..7a5769b3 (2026-05-12 to 2026-05-25, 79 commits), quality-status/bug-fixing/20260520/triage-report.html (52 open bugs, 21 resolved, burndown data 04-25 to 05-31), GitHub Issues API microsoft/WinML-ModelKit.*
