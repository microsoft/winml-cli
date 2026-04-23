# EP Scale — microsoft/ModelKit
**Period**: 2026-03-23 to 2026-04-08
**Generated**: 2026-04-08
**E2E Snapshot**: 0403 (QNN), 0403 (OV), 0403 (VitisAI)

---

## EP Status Table

| EP | Status | Pass Rate | Snapshot | Notes |
|----|--------|-----------|----------|-------|
| QNNExecutionProvider_NPU | Operational | 63.4% | 0403 | Up from 59.7% (+3.7pp) |
| OpenVINOExecutionProvider_NPU | Operational | 59.3% | 0403 | Down from 59.7% (-0.4pp) |
| VitisAIExecutionProvider_NPU | Operational — declining | 50.5% | 0403 | Down from 55.6% (-5.1pp); current bottleneck |
| DirectMLExecutionProvider | Not started | — | — | No milestone assigned |
| AMD NPU (Ryzen AI) | Not started | — | — | Blocked on hardware (#145, #146) |
| OpenVINO GPU | Not started | — | — | Scheduled 202606+ |
| OpenVINO CPU | Not started | — | — | Scheduled 202606+ |
| TensorRT GPU | Not started | — | — | Scheduled 202606+ |
| QNN Adreno GPU | Not started | — | — | Scheduled 202606+ |

---

## E2E Coverage Snapshot (0403)

| EP | PASS | FAIL | Total | Pass Rate | vs. Previous |
|----|------|------|-------|-----------|--------------|
| QNNExecutionProvider_NPU | 137 | 79 | 216 | 63.4% | +3.7pp (was 59.7%) |
| OpenVINOExecutionProvider_NPU | 128 | 88 | 216 | 59.3% | -0.4pp (was 59.7%) |
| VitisAIExecutionProvider_NPU | 109 | 107 | 216 | 50.5% | -5.1pp (was 55.6%) |
| All Three EPs | 98 | 118 | 216 | 45.4% | +1.0pp (was 44.4%) |

**Key observation**: VitisAI pass rate dropped 5.1pp between snapshot 0327 and 0403, making it the primary bottleneck. It now trails QNN by 12.9pp. The all-three-EP intersection improved slightly despite the VitisAI decline, suggesting QNN and OV gains covered some of the loss in the overlap set.

## Zero-Coverage Tasks

The following task types have zero passing combinations across all three EPs:

- document-question-answering
- mask-generation
- summarization
- text-generation
- translation
- visual-question-answering

## Partial Coverage (2 of 3 EPs)

### QNN + OV only (VitisAI bottleneck)

28 model-task combinations pass on QNN and OpenVINO but fail on VitisAI. VitisAI is the sole blocker for these combinations reaching full three-EP coverage.

### QNN + VitisAI only (OV bottleneck)

6 model-task combinations pass on QNN and VitisAI but fail on OpenVINO:

- Intel/dpt-large
- Intel/zoedepth-nyu-kitti
- SAM2 variant 1
- SAM2 variant 2
- SAM2 variant 3
- CLIP-ViT-H-14

### OV + VitisAI only (QNN bottleneck)

1 model-task combination passes on OpenVINO and VitisAI but fails on QNN:

- swin-large

---

## Open P0 EP Issues (202604 Release)

| Issue | ID | Title | Target | Owner | Status |
|-------|----|-------|--------|-------|--------|
| #52 | P0-EP-002 | QNN EP Subgraph Pattern Testing | 202604 | Qiong Wu (DingmaomaoBJTU) | Open |
| #94 | P0-EP-008 | OpenVINO NPU Op Coverage Collection | 202604 | Fangyang Ci (fangyangci) | Open |
| #111 | P0-HARDWARE-002 | OpenVINO NPU Hardware Access | 202604 | Yue Sun (KayMKM) | Open |

## Closed EP Issues This Period

| Issue | ID | Title | Closed |
|-------|----|-------|--------|
| #96 | P0-EP-009 | OpenVINO NPU Subgraph Pattern Testing | 2026-04-01 |
| #110 | P0-HARDWARE-001 | QNN Hardware Access Snapdragon X Elite | 2026-04-08 |

---

## Post-Build EP Items (202606+)

| Issue | ID | Title | Target | Owner |
|-------|----|-------|--------|-------|
| #142 | P1-EP-001 | OpenVINO GPU | 202606+ | Fangyang Ci (fangyangci) |
| #143 | P1-EP-003 | OpenVINO CPU | 202606+ | Fangyang Ci (fangyangci) |
| #144 | P1-EP-005 | TensorRT GPU | 202606+ | Charles Zhang (chinazhangchao) |
| #145 | P1-EP-AMD-NPU-001 | AMD NPU Ryzen AI | 202606+ | Charles Zhang (chinazhangchao) |
| #146 | P1-EP-AMD-NPU-002 | AMD NPU Subgraph Pattern Testing | 202606+ | Charles Zhang (chinazhangchao) |
| #147 | P1-EP-AMD-GPU | AMD GPU MIGraphX | 202606+ | Charles Zhang (chinazhangchao) |
| #148 | P1-EP-QNN-ADRENO | QNN Adreno GPU | 202606+ | Charles Zhang (chinazhangchao) |

All post-build items are open and not yet started.

---

## Summary

Three EPs are operational for the 202604 release target: QNN NPU (63.4%), OpenVINO NPU (59.3%), and VitisAI NPU (50.5%). QNN showed the strongest improvement this period (+3.7pp). VitisAI declined 5.1pp and is now the bottleneck for three-EP intersection coverage, sitting 12.9pp behind QNN.

Two P0 issues closed this period: QNN hardware access (#110, closed 2026-04-08) and OpenVINO NPU subgraph pattern testing (#96, closed 2026-04-01). Three P0 issues remain open: QNN subgraph pattern testing (#52), OpenVINO NPU op coverage collection (#94), and OpenVINO NPU hardware access (#111).

Six task types have zero coverage across all EPs and represent a structural gap for the 202604 release. All post-build EP items (202606+) are not yet started; AMD NPU work is additionally blocked on hardware availability.
