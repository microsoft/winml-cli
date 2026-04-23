# ModelKit Bi-Weekly Report
**Period:** 2026-03-23 to 2026-04-08
**Report ID:** 040826_001
**Generated:** 2026-04-08 (updated with 0403 E2E snapshot)
**Distribution:** Engineering Leadership

---

## Section 1: High-Level Goals — Core Metrics Dashboard

| Dimension | Now | Target (April 14 / May 1) | Gap | Status |
|-----------|-----|--------------------------|-----|--------|
| Release Gate | 2/5 April 14 gate items closed (#164, #165 today) | 5/5 by April 14 | −3 items (#168 CI/CD, #169 community, #170 execution), T-6 days | At Risk — single owner (Zhipeng Wang), 3 items in 6 days |
| EP Scale | 3 NPU EPs operational; 2 P0 coverage tasks open | Close #52 + #94 by April 14 | −2 P0 tasks; OV hardware blocked (#111) | At Risk — OV NPU hardware not available |
| Feature Scale | 4/23 P0s closed this milestone (202604); 19 open | 23/23 by April 14 | −19 tasks; 0 core INFRA/FEATURE P0s closed | Critical — no core feature P0 has closed yet |
| Model Scale | 0/23 model families closed; 98/216 all-EP pass (45.4%) | 23/23 families pass winml perf (May 1) | −23 families; 15/23 unassigned; VitisAI regressed to 50.5% | Critical — pace 0 families/week vs need 1+/day |

---

## Section 2: Highlights & Lowlights

### Highlights

**Release Gate** — *Legal and security reviews completed, unblocking CI/CD automation*
- Legal & License Review closed (#164) — MIT license confirmed, dependency audit complete; unblocks SBOM generation for #165. *(Zhipeng Wang)*
- Security Review closed (#165) — static analysis, secret scan, vulnerability scan done; unblocks PyPI workflow automation in #168. *(Zhipeng Wang)*
- Analyzer Validation Tests closed (#112) — P0 test coverage for static analyzer confirmed passing. *(Qiong Wu)*

**CLI / Commands** — *Major rename and Windows compatibility fixes across all command files*
- CLI renamed from `wmk` to `winml` (#205) — 100+ references updated across commands, docs, and test fixtures; closes #203. *(Zhipeng Wang)*
- 6 bug bash issues resolved in one PR (#246, covers #228–#233) — JSON output format, --format flag ignored, ANSI color leak, PowerShell stderr, stdout debug print, hub truncation on cp1252. *(Qiong Wu)*
- Charmap codec errors fixed across remaining command files (#208) and analyze/inspect (#200) — Windows cp1252 terminals no longer crash on emoji output. *(Zhipeng Wang, Qiong Wu)*
- `--device` flag made case-insensitive (#264, closes #215) — `--device npu` now works alongside `--device NPU`; consistent with other commands. *(Qiong Wu)*
- `winml hub` command added (#196) — queries built-in model registry from CLI; closes #117. *(Qiong Wu)*

**E2E Evaluation** — *End-to-end evaluation pipeline built from scratch for QNN and feature-extraction tasks*
- QNN NPU E2E eval pipeline added (#242) — end-to-end evaluation for QNN NPU models, enabling automated pass/fail tracking per model-EP pair. *(Yue Sun)*
- Static analyzer E2E eval framework added (#222, #221) — framework to run SA evaluation end-to-end; adds reproducible regression testing for the analyzer. *(Qiong Wu)*
- Feature extraction evaluation implemented (#190) — adds accuracy evaluation for feature-extraction task; first accuracy measurement for embedding models. *(Zhenchao Ni)*
- E2E test pipeline initialized (#48, #27, #25) — initial Modelkit E2E test pipeline and perf/accuracy test harness committed. *(Yue Sun, Qiong Wu)*

**Static Analyzer / Pattern Rules** — *QNN GPU pattern rules and SAM2 E2E fix*
- QNN pattern rules added (#210) — new pattern rules for QNN EP, expanding subgraph pattern coverage for QNN GPU. *(Charles Zhang)*
- QNN GPU rules updated (#234) — additional QNN GPU-specific rules; improves pattern matching for GPU workloads. *(Charles Zhang)*
- SAM2 registration fix (#235) — fixes `winml` registration for SAM2 model family. *(Charles Zhang)*
- facebook/sam2.1-hiera-small fixed (#212) — corrects E2E pipeline failure for SAM2.1 small variant. *(Charles Zhang)*
- Slice derive_properties crash fixed (#244) — crash on symbolic dynamic axes in Slice op resolved. *(vortex-captain)*

**QDQ / Static Analyzer Config** — *P1 model coverage and input constraint normalization*
- QDQ config updated for P1 models (#204, #236) — expands QDQ parameter config to cover P1-priority models; replaces `support_weight`/`support_activation` with `qdq_types`. *(xieofxie)*
- Input constraint normalization (#23) — normalizes `input_constraints` before computing case signature, fixing signature collisions across model variants. *(xieofxie)*

**CI / Test Infrastructure** — *License enforcement, network test isolation, and CI stability*
- License header CI check added (#227) — automated enforcement of license headers on all source files; lint errors fixed. *(Zhipeng Wang)*
- Network-dependent tests moved to integration (#254) — removes flaky 504 timeout failures from unit test suite. *(Zhipeng Wang)*
- test_runtime_checker.py hang fixed (#252) — hardware probing mocked in CI; prevents hang on machines without NPU hardware. *(Copilot)*
- Runtime check rule zips moved to external repo (#213, closes #219) — reduces repo size; download_rules.py updated with `--account` flag (#251). *(Zhipeng Wang)*

**Import Cleanup** — *10-phase public API standardization across all packages*
- Import cleanup Phases 0–9 completed (#39–#49) — all 10 packages now have correct `__init__.py` exports and enforced import boundaries; closes #29–#38. *(Zhipeng Wang)*

---

### Lowlights

| # | Risk | Severity | Deadline | Owner |
|---|------|----------|----------|-------|
| L-01 | VitisAI pass rate regressed from 55.6% to 50.5% (−5.1pp, lost 11 model-task combinations in 0327→0403 snapshot) | High | — | Unassigned — root cause unknown |
| L-02 | CI/CD pipeline (#168) not started — branch protection, GitHub Actions, PyPI workflow all missing; unblocked by #165 today | Critical | April 14 | Zhipeng Wang |
| L-03 | Community readiness (#169) not started — CODE_OF_CONDUCT.md, versioning policy, public Roadmap all absent | Critical | April 14 | Zhipeng Wang |
| L-04 | Release Execution (#170) blocked on #168 and #169; cannot tag or publish until both close | Critical | April 14 | Zhipeng Wang |
| L-05 | 15/23 model-family tracking issues (#118–#140) have no assignee; model scale pace = 0 families/week | Critical | May 1 | PM — assign owners this week |
| L-06 | OV NPU hardware (Meteor Lake / Lunar Lake) not available (#111) — blocks OV E2E validation and #94 (OV op coverage) | High | April 14 | Yue Sun |
| L-07 | ESRP-03 (#171) and Trade Compliance (#172–#174) not started; required for public distribution | High | TBD | Zhipeng Wang — no schedule set |
| L-08 | P0-FEATURE-011 Data Driven Auto-fix with Agent (#109) — no PR activity, no progress signal this period | High | April 14 | Fangyang Ci |
| L-09 | P0-INFRA-003 (#100) and P0-FEATURE-010 (#106) in 202606+ both unassigned; both P0 | Medium | Post-Build | PM — assign owners |
| L-10 | 202605 Release: 30/30 issues open, no closures; all May 1 work gated behind April 14 completion | Medium | May 1 | Te Zheng / Zhipeng Wang |
| L-11 | Bug bash wave (#211–#261) generated 50+ issues on 2026-04-01 and 2026-04-07; 40+ still unassigned | Medium | TBD | PM — triage required |

---

## Section 3: Data Analysis

### 3.1 E2E Pass Rate by EP (Snapshot 0403)

```
EP                              Pass    Fail   Pass Rate   Bar
QNNExecutionProvider_NPU        137/216   79    63.4%  ████████████████░░░░░░░░░
OpenVINOExecutionProvider_NPU   128/216   88    59.3%  ██████████████░░░░░░░░░░░
VitisAIExecutionProvider_NPU    109/216  107    50.5%  ████████████░░░░░░░░░░░░░  ▼ REGRESSED
─────────────────────────────────────────────────────────────────────────────────
All Three EPs                    98/216   —     45.4%  ███████████░░░░░░░░░░░░░░
```

Delta vs. previous snapshot (0327):

| EP | 0327 | 0403 | Delta |
|----|------|------|-------|
| QNN | 59.7% (129/216) | 63.4% (137/216) | +3.7pp (+8) |
| OV | 59.7% (129/216) | 59.3% (128/216) | −0.4pp (−1) |
| VitisAI | 55.6% (120/216) | **50.5% (109/216)** | **−5.1pp (−11)** |
| All Three | 44.4% (96/216) | 45.4% (98/216) | +1.0pp (+2) |

VitisAI is the bottleneck EP at 50.5%, now 12.9pp below QNN. The 5.1pp regression is unexplained.

### 3.2 All-Three-EP Pass Rate by Task Category

| Task | All-3 Pass | Notes |
|------|-----------|-------|
| fill-mask | 10 | BERT, RoBERTa, XLM-RoBERTa, DistilBERT, MPNet variants |
| feature-extraction | 9 | BGE, CLIP, sentence-transformers |
| image-segmentation | 9 | Segformer family (b0–b5 variants) |
| question-answering | 9 | DistilBERT, BERT, RoBERTa, Electra, mDeBERTa |
| image-classification | 8 | ViT, ResNet, MobileViT variants |
| text-classification | 8 | DeBERTa, cross-encoder, RoBERTa |
| sentence-similarity | 7 | sentence-transformers, BGE, E5 |
| token-classification | 7 | BERT, RoBERTa, CamemBERT variants |
| zero-shot-classification | 7 | DeBERTa-v3 family |
| image-feature-extraction | 7 | DINOv2, ViT, rad-dino |
| image-to-text | 7 | TrOCR, BLIP captioning, ViT-GPT2 |
| object-detection | 5 | DETR, YOLOS, table-transformer |
| zero-shot-image-classification | 3 | SigLIP, Marqo fashionSigLIP |
| depth-estimation | 1 | Intel/dpt-hybrid-midas only |
| masked-lm | 1 | bert-base-multilingual-cased |
| document-question-answering | **0** | ❌ LayoutLM, BERT-based DQA — zero coverage |
| mask-generation | **0** | ❌ SAM2 passes QNN+VitisAI only (OV bottleneck) |
| summarization | **0** | ❌ T5, BART — decoder architecture; all EPs fail |
| text-generation | **0** | ❌ GPT-2, Qwen — autoregressive decoder; all EPs fail |
| translation | **0** | ❌ Marian, M2M-100 — seq2seq decoder; all EPs fail |
| visual-question-answering | **0** | ❌ BLIP-2, InternLM2 — multimodal decoder; all EPs fail |

6 task categories at zero all-EP coverage. Decoder/generative architectures fail due to dynamic shape and attention op gaps across all 3 EPs.

### 3.3 Near-Miss: Partial Coverage (2/3 EPs)

| EP Bottleneck | Count | Notable Models |
|---------------|-------|----------------|
| VitisAI bottleneck (QNN+OV only) | 28 combinations | BGE-small/base, CLIP-large variants, BEiT, rtdetr, splinter |
| OV bottleneck (QNN+VitisAI only) | 6 combinations | DPT-large, ZoeDepth, SAM2 hiera (3 variants), CLIP-ViT-H-14 |
| QNN bottleneck (OV+VitisAI only) | 1 combination | swin-large |

28 combinations are VitisAI-blocked. Resolving the VitisAI regression would be the single highest-leverage action — these 28 should become all-3 passers with no code changes.

### 3.4 Model Scale Trajectory

| Checkpoint | Model Families Closed | All-3 Pass Combinations |
|------------|----------------------|------------------------|
| 2026-03-31 (snapshot 0327) | 0/23 | 96/216 (44.4%) |
| 2026-04-08 (snapshot 0403) | 0/23 | 98/216 (45.4%) |
| Target (May 1, T-23 days) | 23/23 | — |

**Required pace**: 1.0 model family/day.
**Actual pace this period**: 0.0 model families/day.

### 3.5 Bottleneck Analysis

- **VitisAI is the ceiling** at 50.5%, 12.9pp below QNN. The −5.1pp regression from 55.6% (−11 combinations in 0327→0403) is unexplained. Until root-caused, VitisAI limits all-EP intersection improvement regardless of other work.
- **Decoder architecture gap is structural**: text-generation, summarization, translation, VQA, and mask-generation have zero all-EP coverage. These require EP-level attention/KV-cache op support — not a ModelKit tooling fix. Encoder-only models should be prioritized for the May 1 model scale target.
- **28 encoder-only near-misses are the near-term lever**: all 28 QNN+OV-only combinations are encoder-only models (BERT variants, BGE, CLIP text encoders). Once the VitisAI regression is identified and fixed, these convert to all-3 passers without any model code changes.

---

## Section 4: Task Status

### 4.1 Completed This Period

**Release Gate** — *Legal and security gate items cleared; 202603 milestone fully closed*

| PR / Issue | Description | Owner |
|-----------|-------------|-------|
| #164 closed | Legal & License Review complete — MIT license verified, dependency audit done, unblocks SBOM | Zhipeng Wang |
| #165 closed | Security Review complete — static analysis, secret scan, vulnerability scan, SBOM generated | Zhipeng Wang |
| #112 closed | P0-TEST-003 Analyzer Validation Tests — SA test coverage confirmed passing | Qiong Wu |
| #90 closed | Wheel release for bug bash — 202603 milestone closed | Te Zheng |
| #110 closed | QNN hardware access (Snapdragon X Elite) obtained — 202603 milestone closed | Yue Sun |

**CLI / Commands** — *Rename, Windows compatibility, and new hub command*

| PR | Description | Owner |
|----|-------------|-------|
| #205 / 2500b9b | Rename CLI from `wmk` to `winml` across all commands and fixtures | Zhipeng Wang |
| #246 / 7123665 | Fix 6 bug bash issues (#228–#233): JSON format, --format ignored, ANSI leak, PowerShell stderr, debug print, hub truncation | Qiong Wu |
| #264 / e48ad29 | Make `--device` flag case-insensitive; fixes #215 | Qiong Wu |
| #208 / 903ce0a | Fix charmap codec errors in remaining command files on Windows | Zhipeng Wang |
| #200 / 92325bf | Fix charmap codec errors in wmk analyze and wmk inspect on Windows | Qiong Wu |
| #201 / 9bcf4ad | Suppress ep_registry INFO log leaking into wmk sys CLI output | Qiong Wu |
| #196 / 78442c4 | Add `winml hub` command to query built-in model registry | Qiong Wu |

**E2E Evaluation** — *QNN E2E pipeline, SA eval framework, feature extraction eval*

| PR | Description | Owner |
|----|-------------|-------|
| #242 / 6116e2c | Add E2E eval pipeline for QNN NPU — automated model pass/fail per EP | Yue Sun |
| #222 / b8d97c8 | Add SA E2E eval framework — reproducible end-to-end testing for static analyzer | Qiong Wu |
| #221 / 199d3c5 | Fix SA eval framework bugs | Qiong Wu |
| #190 / f92b313 | Implement feature extraction evaluation — first accuracy measurement for embedding models | Zhenchao Ni |
| #48 / 605cd7c | Initial Modelkit E2E test pipeline commit | Yue Sun |
| #25 / 18ba5d4 | Add E2E tests for perf and accuracy | Qiong Wu |

**Static Analyzer / Pattern Rules** — *QNN patterns, SAM2 fix, Slice crash*

| PR | Description | Owner |
|----|-------------|-------|
| #210 / 6469aa6 | Add QNN pattern rules for subgraph pattern coverage | Charles Zhang |
| #234 / fd43a2f | Update QNN GPU rules — additional GPU-specific patterns | Charles Zhang |
| #235 / 4285d18 | Fix winml registration issue for SAM2 | Charles Zhang |
| #212 / e1087e0 | Fix facebook/sam2.1-hiera-small E2E pipeline failure | Charles Zhang |
| #244 / f6f9f82 | Fix Slice derive_properties crash on symbolic dynamic axes | vortex-captain |
| #19 / 166a453 | Overwrite op results with subgraph results in graph optimizer | Charles Zhang |

**QDQ / Config** — *P1 model QDQ coverage and input constraint normalization*

| PR | Description | Owner |
|----|-------------|-------|
| #236 / 7f6e9e5 | Update QDQ config for P1 models (expanded coverage) | xieofxie |
| #204 / b606007 | Update QDQ config for P1 models (initial batch) | xieofxie |
| #23 / d436597 | Normalize input_constraints before computing case signature — fixes signature collisions | xieofxie |
| #17 / 247698e | Replace `support_weight`/`support_activation` with unified `qdq_types` field | xieofxie |
| #22 / 603309d | Pick up P1 model coverage (QNN op coverage expansion) | Fangyang Ci |

**CI / Test Infrastructure** — *License enforcement, test isolation, CI stability*

| PR | Description | Owner |
|----|-------------|-------|
| #227 / b4db8bb | Add license header CI check; fix lint errors across codebase | Zhipeng Wang |
| #254 / 7512c96 | Move remaining network-dependent tests to integration — eliminates 504 flakes from unit suite | Zhipeng Wang |
| #252 / ba39203 | Fix test_runtime_checker.py hang on CI by mocking hardware probing | Copilot |
| #213 / c9c3a88 | Move runtime check rule zips to external repo (ModelKitArtifacts) | Zhipeng Wang |
| #251 / ce7d591 | Improve download_rules.py with --account flag and better error messages | Zhipeng Wang |
| #14 / 97df6de | Add CI workflows for ModelKit (GitHub Actions) | Zhipeng Wang |
| #241 / e361726 | Fix test transient failure | Charles Zhang |
| #237 / d16c00b | Clean .onnx.data temp files to prevent disk pollution | Yue Sun |

**Import Cleanup / Refactor** — *10-phase public API standardization*

| PR | Description | Owner |
|----|-------------|-------|
| #39–#49 / d1336b5–a23f160 | Import cleanup Phases 0–9 — all packages (onnx, datasets, quant, export, compiler, eval, optracing, loader, models, optim, session, analyze, pattern) enforce `__init__.py` boundaries | Zhipeng Wang |
| #95 / 17b40ab | Move misplaced integration/e2e tests out of tests/unit/ | Zhipeng Wang |
| #28 / e120e67 | Standardize naming conventions and reorganize test directory structure | Zhipeng Wang |
| #188 / 5c5b873 | Add NvTensorRTRTXExecutionProvider to EP device map; fixes #187 | Zhipeng Wang |
| #18, #24 / 17f1b52, 2e5af47 | Mock WinML EP init in session tests and root conftest for CI | Zhipeng Wang |
| #15 / 922b5d3 | Update codebase with latest changes | Zhipeng Wang |

*Reverted: #20 pickup p1_coverage (Fangyang Ci) — reverted via #21 on 2026-03-30; excluded from module totals.*

---

### 4.2 In-Progress Tasks

**Release Gate (202604)**

| Task | Description | Progress | Owner | Blocker |
|------|-------------|----------|-------|---------|
| #168 P0-REL-R | CI/CD setup — branch protection, GitHub Actions, PyPI workflow | ~0% | Zhipeng Wang | #165 just closed (unblocked today) |
| #169 P0-REL-O | Community readiness — CODE_OF_CONDUCT, versioning policy, Roadmap | ~0% | Zhipeng Wang | None |
| #170 P0-REL-E | Release execution — tag, PyPI publish, announcement | ~0% | Zhipeng Wang | Blocked on #168, #169 |

**EP Scale (202604)**

| Task | Description | Progress | Owner | Blocker |
|------|-------------|----------|-------|---------|
| #52 P0-EP-002 | QNN EP subgraph pattern testing | unknown | Qiong Wu | — |
| #94 P0-EP-008 | OV NPU op coverage collection | unknown | Fangyang Ci | #111 hardware |
| #111 P0-HW-002 | OV NPU hardware access (Meteor Lake / Lunar Lake) | unknown | Yue Sun | External — procurement |

**Feature / Infra Scale (202604)**

| Task | Description | Progress | Owner | Blocker |
|------|-------------|----------|-------|---------|
| #98 P0-INFRA-001 | Runtime Tester QDQ support | unknown | xieofxie | — |
| #99 P0-INFRA-002 | QNN EP op coverage opset version scaling | unknown | Fangyang Ci | — |
| #101 P0-INFRA-005 | Runtime Tester INT4 support for LLM | unknown | xieofxie | — |
| #102 P0-INFRA-010 | Subgraph test framework | unknown | Qiong Wu | — |
| #103 P0-INFRA-011 | Pattern validation | unknown | Qiong Wu | — |
| #104 P0-INFRA-006 | Doc Parser QNN checker functions | unknown | Qiong Wu | — |
| #105 P0-FEATURE-009 | Graph Optimizer: support change attributes | unknown | Te Zheng | — |
| #107 P0-INFRA-013 | Graph Optimizer: normalize input ONNX models | unknown | Te Zheng | — |
| #108 P0-FEATURE-006 | Profiling | unknown | Te Zheng / xieofxie | — |
| #109 P0-FEATURE-011 | Data Driven Auto-fix with Agent | ~0% | Fangyang Ci | No PR activity observed |
| #113 P0-TEST-004 | QDQ validation tests | unknown | xieofxie | — |
| #114 P0-TEST-005 | Graph Optimizer validation | unknown | Qiong Wu | — |

---

### 4.3 Not Started — P0 Items Needing Owner Assignment

**Release Gate — April 14 (T-6 days)**

| Task | Description | Deadline | Priority |
|------|-------------|----------|----------|
| #168 P0-REL-R | Set up branch protection on main, configure GitHub Actions, implement PyPI publish workflow | April 14 | Red — start immediately |
| #169 P0-REL-O | Add CODE_OF_CONDUCT.md, establish SemVer policy, publish public Roadmap | April 14 | Red — start immediately |
| #170 P0-REL-E | Tag v0.x.0, publish to PyPI, send announcement, archive internal docs | April 14 | Red — blocked on #168, #169 |

**ESRP / Trade Compliance (no milestone, no schedule)**

| Task | Description | Deadline | Priority |
|------|-------------|----------|----------|
| #171 ESRP-03 | Validate signed artifacts — verify signatures and test install | TBD | Red — required for public release |
| #172 TR-01 | Complete ECCN classification | TBD | Red — required for public distribution |
| #173 TR-02 | Submit Trade Compliance Review via internal portal | TBD | Blocked on #172 |
| #174 TR-03 | Obtain Trade sign-off for public open-source distribution | TBD | Blocked on #173 |

**Model Scale — Unassigned Model Families**

| Task | Description | Deadline | Priority |
|------|-------------|----------|----------|
| #119 | t5 / summarization + translation — zero all-EP coverage | May 1 | Red — unassigned, zero coverage |
| #121 | marian + m2m_100 / translation — zero coverage | May 1 | Red — unassigned, zero coverage |
| #125 | qwen2 + qwen3 / text-generation — zero coverage | May 1 | Red — unassigned, zero coverage |
| #130 | gpt2 / text-generation — zero coverage | May 1 | Red — unassigned, zero coverage |
| #120 | bart + mbart / summarization + text-classification + zero-shot-classification | May 1 | Orange — unassigned |
| #123 | depth_anything + dpt + zoedepth / depth-estimation | May 1 | Orange — unassigned |
| #128 | deberta + deberta-v2 / text-classification + zero-shot-classification | May 1 | Orange — unassigned |
| #129 | swin / image-classification | May 1 | Orange — unassigned |
| #131 | blip + blip-2 / visual-question-answering | May 1 | Orange — unassigned |
| #132 | pix2struct + vilt / visual-question-answering | May 1 | Orange — unassigned |
| #134 | layoutlm + layoutlmv3 / document-question-answering | May 1 | Orange — unassigned |
| #135 | distilbert + camembert / question-answering + token-classification | May 1 | Orange — unassigned |
| #136–#138 | siglip, siglip_vision_model, dinov2 families | May 1 | Orange — unassigned |
| #140 | internlm2 + phi4mm / visual-question-answering | May 1 | Orange — unassigned |

**Post-Build P0s (202606+) — Unassigned**

| Task | Description | Deadline | Priority |
|------|-------------|----------|----------|
| #100 P0-INFRA-003 | Runtime Tester high-priority op types | Post-Build | Orange — P0 with no owner |
| #106 P0-FEATURE-010 | Graph Optimizer advanced optimizations | Post-Build | Orange — P0 with no owner |

---

## Action Items for This Week

| # | Action | Owner | Due |
|---|--------|-------|-----|
| 1 | Start #168 (CI/CD setup): configure branch protection, GitHub Actions, and PyPI workflow now that #165 is cleared | Zhipeng Wang | April 12 |
| 2 | Start #169 (community readiness): commit CODE_OF_CONDUCT.md, SemVer policy doc, and Roadmap stub | Zhipeng Wang | April 12 |
| 3 | Root-cause VitisAI pass rate drop (−5.1pp, −11 combinations, 0327→0403); file bug with EP owner | Unassigned — escalate to PM | April 10 |
| 4 | Assign DRI for 15 unowned model-family tracking issues (#119, #120, #121, #123, #128–#132, #134–#138, #140) | PM | April 10 |
| 5 | Assign DRI for ESRP/Trade items (#171–#174); agree on schedule with Zhipeng Wang | PM | April 10 |
| 6 | Provide progress signal for #109 (Data Driven Auto-fix) — PR or written status update | Fangyang Ci | April 10 |
| 7 | Triage ~40 unassigned bug bash issues (#211–#261 range) — assign owners or defer to backlog | PM | April 11 |
| 8 | Prepare tag and publish runbook for #170 (Release Execution) so it can close immediately once #168 and #169 merge | Zhipeng Wang | April 13 |
| 9 | Provide written status on OV NPU hardware procurement (#111) — ETA or escalation path | Yue Sun | April 10 |
| 10 | Identify which 202605 (May 1) tasks can begin in parallel with April gate work | Te Zheng / Zhipeng Wang | April 11 |

---

*Data sources: git log 4f60333..6116e2c (2026-03-26 to 2026-04-08), qnn_report_0403.csv / ov_report_0403.csv / vitisai_report_0403.csv (snapshot 0403, all EPs), ep_coverage_analysis.md, GitHub Issues API microsoft/ModelKit (fetched 2026-04-08), milestone files in explain/by-milestone/.*
