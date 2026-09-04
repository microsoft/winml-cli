# EP Knowledge Base — Critical Review

> Date: 2026-06-16
> Reviewer: internal audit
> Scope: `ep_knowledge/qnn_npu.json` findings npu-001 through npu-007
>
> This document records issues found in the original KB entries and the
> reasoning behind corrections applied in the June 2026 update.

---

## Summary of Issues Found

| Finding | Status Before Review | Issue | Corrected Status |
|---------|---------------------|-------|-----------------|
| npu-001 | `mechanism_confirmed: true` | ORT version used has kMaxSupportedOpset ≥ 22 — bypass mechanism does not apply; ResNet-18 data is noise | `mechanism_confirmed: false`, mechanism UNKNOWN |
| npu-002 | scope: "General / most vision models" | Tested on 1 model only (ConvNext) | scope narrowed to ConvNext |
| npu-003 | scope: "General / all QNN NPU" | Tested on 1 model only (ConvNext) | scope narrowed to ConvNext |
| npu-004 | confidence: "medium" | No recorded data; experiment aborted before measurements saved | confidence: "very_low / anecdote" |
| npu-005 | confidence: "medium" | Compares ORT QNN EP vs qairt native stack — different compilation pipeline entirely | added fairness caveat |
| npu-006 | `mechanism_confirmed: false` | Observation is solid (3-session consistent). Mechanism is unconfirmed but regression is unambiguous | no change to confirmed status; added session evidence |
| npu-007 | `mechanism_confirmed: true` | Solid, confirmed across all 8 models | no change |

---

## Detailed Analysis

### npu-001 — opset 21 speedup

#### ORT version issue (critical)

The catalog sweep used `onnxruntime-windowsml==1.24.5`. The npu-001 mechanism
explanation relies on ORT's `kMaxSupportedOpset` gate:

> "On older ORT where kMaxSupportedOpset < 21, opset 21 models bypass the
> NCHW→NHWC layout transformer entirely."

But the `kMaxSupportedOpset` version table (from `cpu.json`) shows:

| ORT version | kMaxSupportedOpset |
|-------------|-------------------|
| v1.14.x | 18 |
| v1.16.x | 19 |
| v1.17.x | 20 |
| v1.18.x | 21 |
| main_HEAD | 26 |

At ORT 1.24.x, `kMaxSupportedOpset` is almost certainly ≥ 22. This means BOTH
opset 17 and opset 21 models go through the NHWC layout transform in the ORT
version actually used in the sweep. **The "bypass" mechanism does not apply.**

Consequence: `mechanism_confirmed` must be `false`. The speedup for DINOv2 and
MobileViT is empirically real but the cause is **unknown**. The ORT source code
investigation confirmed the bypass mechanism for *older* ORT versions, not for
the ORT version actually used.

Possible alternative mechanisms (uninvestigated):
1. PyTorch ONNX exporter produces a structurally different graph at opset 21
   (different op decompositions, fewer reshape/squeeze nodes)
2. QNN EP's graph partitioner behaves differently with opset 21 operator
   semantics even when the NHWC transform fires
3. Quantization calibration path differs between opset export versions
4. The NHWC transform at opset 21 still inserts fewer Transposes for some reason
   despite firing (investigation needed via optimized graph dump)

#### ResNet-18 data is noise-dominated

ResNet-18 baseline p50 is ~1ms. At this latency, the 3×500-iter protocol
produces per-session p50s that vary 4x between sessions:

```
h1 (opset17): sessions = [0.990, 4.003, 2.716] ms  ← 4x range
h3 (opset21): sessions = [1.054, 2.175, 4.107] ms  ← 4x range
```

The two distributions fully overlap. Declaring a "+20.2% speedup" from comparing
medians (2.716 vs 2.175ms) is not statistically valid. This data point is
**removed** from `validated_models.benefits_from_opset21`.

To get reliable data for ResNet-18, a minimum of ~3000 iterations per session
and ≥ 5 sessions would be needed.

#### MobileViT DVFS spike in h1

h1 (opset17) sessions: [10.557, 11.721, **27.436**] ms

The third session at 27.4ms is a clear DVFS thermal event (2.4x spike). The
median (11.721ms) is upward-biased by this session. The "true" opset17 p50 is
likely ~11ms, making the "+26.5%" speedup calculation overstated. A more
conservative estimate is ~20-22%.

However, h3 (opset21) sessions [10.814, 8.625, 8.449] show two highly consistent
low-latency sessions. The speedup is real, magnitude uncertain (~20-26%).

#### DINOv2 — most reliable evidence for npu-001

h1 (opset17): [7.176, 6.392, 9.436] ms — range 6.4–9.4ms
h3 (opset21): [4.977, 4.876, 6.884] ms — range 4.9–6.9ms

The two distributions barely overlap only at extremes (h3 max 6.884 ≈ h1 min
6.392). h3 sessions 1 and 2 (4.977, 4.876ms) are tightly clustered at ~4.9ms,
well below the h1 range. The speedup appears real (≥24% vs h1's non-spiked
sessions, up to 31% vs h1 median).

DINOv2-small's benefit is notable because it is primarily a Vision Transformer —
it has a patch embedding Conv layer but attention-dominant compute. Why opset21
helps DINOv2 but NOT ViT-base is unknown. This architecture distinction needs
investigation.

#### Updated empirical claim for npu-001

**Observable fact**: For DINOv2-small and MobileViT-small on QNN NPU (ORT 1.24.5,
Snapdragon X Elite), using opset 21 export instead of opset 17 produces a
consistent latency reduction of ~20-31% across 3-session benchmarks.

**What is NOT known**: Why this occurs in ORT 1.24.x where the kMaxSupportedOpset
bypass should not apply.

**What needs investigation**:
1. Dump optimized.onnx for both opset17 and opset21 DINOv2, count Transpose nodes
   — if opset21 has fewer Transposes, explains speedup via a different mechanism
2. Verify ORT 1.24.x kMaxSupportedOpset value from compiled binary
3. Test 3+ additional Conv+residual models: EfficientNet-B0, MobileNet-V3,
   ConvNeXt-tiny (already done for CPU; needs QNN NPU validation)

---

### npu-002 — W8A16 speedup over FP32

**Issue**: Scope states "General (applies to most vision models on QNN NPU)".
Evidence base: 1 model (ConvNext), 1 device.

The 1.9x speedup is plausible from HTP architecture (INT8 weight path), but
the magnitude varies by model: a model with few weight-heavy ops (e.g., pure
attention) may see less speedup than a Conv-heavy model. "Most vision models"
is over-claimed.

**Correction**: Scope narrowed to "ConvNext — single model validation". The
catalog sweep provides indirect evidence (all 8 models used W8A16 and ran
faster than FP32 would on HTP) but no direct FP32 comparison baseline for
those models.

---

### npu-003 — compile speedup

**Issue**: Scope states "General (applies to all QNN NPU deployments)". Evidence
base: 1 model (ConvNext), 1 device.

The compile (EPContext) mechanism is well-understood and applies generally, but
the 1.7x magnitude is model-specific. Models with simpler graphs may see less
benefit; models with many ops may see more.

**Correction**: Scope narrowed. The mechanism claim ("eliminates JIT partitioning")
is generally correct; the magnitude claim (1.7x) is ConvNext-specific.

---

### npu-004 — W8A8 accuracy collapse

**Issue**: The observation is "Exact numbers not recorded — aborted early." This
is an anecdote, not a finding. The confidence of "medium" is unjustified without
data.

The claim may well be correct (W8A8 on LN+GELU is problematic), but without
recorded accuracy numbers it cannot be treated as a KB finding.

**Correction**: Confidence downgraded to "very_low". The finding is relabeled
as an unrecorded anecdote pending a proper experiment with recorded numbers.

---

### npu-006 — conv fusions catastrophic regression

This finding is the **most statistically solid** in the entire KB:

ResNet-18 h4 sessions: [132.3, 134.97, 130.669] ms — CV = 0.016 (extremely stable)
ResNet-18 h1 sessions: [0.990, 4.003, 2.716] ms — median 2.716ms

Even using the best h1 session (0.990ms) vs worst h4 session (134.97ms), the
regression is 136x. The 3-session consistency of h4 (~130-135ms) with near-zero
variance is unusual for QNN NPU (all other hypotheses show high CV). This
suggests the fused ops cause a deterministic CPU fallback with no DVFS noise —
consistent with the mechanism hypothesis.

The only issue is "mechanism_confirmed: false" — the CPU fallback has not been
verified via EP partition dump. The regression is unambiguous; the mechanism is
a strong hypothesis.

**No changes needed** except documenting the 3-session evidence more explicitly.

---

## Additional Models Needed for Validation

### For npu-001 (opset21 benefit for Conv+residual)

| Model | Why useful | Predicted result |
|-------|-----------|-----------------|
| `microsoft/efficientnet-b0` | Conv-dominant, no residual-add structure | uncertain |
| `microsoft/mobilenet-v3-small` | Conv-dominant + SE blocks | likely benefits |
| `timm/convnextv2-nano` | ConvNext variant, already confirmed for ConvNext | should benefit |
| `facebook/deit-small-patch16-224` | Pure ViT (no Conv), similar to ViT-base | should be neutral |
| `timm/regnetx-002` | ResNet-like but with group Conv | uncertain |

Goal: determine whether the benefit is "Conv+residual" or something more specific
to the DINOv2/MobileViT architectures (e.g., hybrid Conv+attention).

### For npu-006 (conv fusions)

| Model | Why useful | Predicted result |
|-------|-----------|-----------------|
| `microsoft/efficientnet-b0` | Conv+BN heavy (many fuseable patterns) | should regress |
| `google/mobilenet-v2-1.0-224` | Depthwise Conv dominant | should regress |
| `timm/vgg16` | Pure Conv-BN | should regress |
| `microsoft/beit-base-patch16-224` | Pure transformer | should be neutral |

Goal: confirm that the regression generalizes to all Conv-dominant models, not
just ResNet-18.

### For npu-002/003 (W8A16 and compile)

Run FP32 vs W8A16 and W8A16 vs W8A16+compile on at least:
- `apple/mobilevit-small` (already benchmarked W8A16; need FP32 baseline)
- `microsoft/resnet-18` (same)
- `facebook/dinov2-small` (same)

This would promote npu-002 and npu-003 from "1-model observations" to
"catalog-validated" findings.

---

## Minimum Experiment Protocol for Validation

For any new model added to the KB:

1. Run 3 independent sessions × 500 iters with 30s cool-down (npu-007 protocol)
2. Record raw per-session p50s, not just the median
3. Verify session-to-session range is < 50% of the median before reporting a gain
4. For sub-2ms models: increase to 3 sessions × 2000 iters minimum
5. Always dump the optimized graph (`--save-optimized-model`) for opset comparison
6. Record ORT version (`winml --version`) at experiment time in the finding

---

*This review document should be re-run after any ORT or QNN SDK version update.*
