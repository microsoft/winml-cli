# `QuantSpec` — Per-Variant Quantization Specification Attached to EPDevice

**Date:** 2026-05-14
**Status:** **DRAFT — design captured, decision pending**
**Branch:** `feat/op-tracing-refactor` (this PR may or may not include the implementation)
**Predecessor:** `2026-05-13-ep-device-spec-design.md` (the EPDeviceSpec catalog)
**Driver:** Two gaps surfaced during the EPDeviceSpec PR verification:

1. **Verification gap.** The CLI matrix verified `winml perf --ep qnn --device npu` with the **FP32** ResNet-50 export, not the canonical QDQ-quantized variant. The "QDQ → QNN HTP" canonical NPU workflow was never exercised in the matrix.
2. **Design gap.** `config/precision.py:62-69` carries an explicit TODO acknowledging that `_BITS_TO_WEIGHT_TYPE` / `_BITS_TO_ACTIVATION_TYPE` need an "EP-specific override layer". Today both tables are global; QNN-NPU's quant requirements (uint8 weights, **uint16** activations, per-tensor) are baked in globally and cannot be expressed per-variant.

## 1. Context — what was rejected, what is being proposed

| Earlier proposal | Status |
|---|---|
| Add `default_precision: Literal["fp32","fp16","int8","int16"]` to `EPDeviceSpec` | **REJECTED.** Rationale: every NPU variant wants `int8`; the field would be redundant with the device-keyed `_AUTO_PRECISION` dict in `config/precision.py:41-45`. No per-variant variance. |
| **THIS DOC:** Add a structured `QuantSpec` type capturing precision + weight type + activation type + scheme; attach to `EPDeviceSpec.default_quant` and resolve into `EPDevice.quant` | **DRAFT.** Per-variant variance is real once the type is structured (QNN-HTP wants `uint8/uint16` unsigned; VitisAI-NPU wants `int8/int8` signed; DML-GPU wants `fp16`). The TODO in `precision.py:64` is the explicit signal. |

The difference is **structural richness**, not the placement. A single precision string has no per-variant content. A multi-field spec does.

## 2. Decision (proposed)

**Add two things:**

1. A new frozen dataclass `QuantSpec` in `session/ep_device.py`.
2. A new optional field `default_quant: QuantSpec | None` on `EPDeviceSpec`, and a corresponding optional field `quant: QuantSpec | None` on `EPDevice` (the runtime instance).

**Migrate two things:**

1. `config/precision.py:_BITS_TO_WEIGHT_TYPE` / `_BITS_TO_ACTIVATION_TYPE` — values move into per-variant catalog entries; precision string parser stays for user input.
2. `commands/quantize.py` — reads `ep_device.quant` instead of consulting the global bit-width tables.

**Do not touch:**

- `WinMLSession` core API stays the same (it takes `EPDevice`; `quant` is a new optional field, not a constructor change).
- `resolve_device()` signature stays compatible (new keyword-only `quant=None` override).
- All non-quantize commands (`perf`, `compile`, `eval`, `config`, `inspect`, `build`) continue to work without behavior changes.

## 3. Type relationship — same Spec→Instance pattern as `EPDeviceSpec`

```
EP_DEVICE_SPECS  (catalog, machine-independent)
  ┌──────────────────────────────────────────────┐
  │ EPDeviceSpec                                 │
  │   ep, device                                 │
  │   default_provider_options                   │
  │   default_quant: QuantSpec | None  ◄── NEW   │
  └──────────────────┬───────────────────────────┘
                     │ resolve_device(ep, device, quant=None)
                     │   reads default_quant from catalog
                     │   (or user override via quant=)
                     ▼
  ┌──────────────────────────────────────────────┐
  │ EPDevice (runtime, machine-specific)         │
  │   ep, device, vendor_id, device_id, vendor   │
  │   quant: QuantSpec | None  ◄── NEW           │
  └──────────────────────────────────────────────┘
```

The flow is identical to how `default_provider_options` resolves onto `WinMLSession` — a per-variant default lives in the catalog; the runtime instance carries the resolved value; user code can override at resolve time.

## 4. Concrete types

```python
# session/ep_device.py

@dataclass(frozen=True, kw_only=True, slots=True)
class QuantSpec:
    """Quantization specification for one binding target.

    Distinct from a "precision string" — captures the full scheme:
    weight type, activation type, signed vs unsigned, per-channel vs per-tensor.
    Different (EP, device) variants need different schemes:
      - QNN-HTP    : uint8 weights, uint16 activations, per-tensor, asymmetric
      - VitisAI-NPU: int8 weights, int8 activations, per-channel, symmetric
      - DML-GPU    : fp16 (no quantization scheme; informational)

    A `None` quant means "any precision is acceptable" (typical of CPU,
    informational for variants we haven't verified yet).
    """
    precision: Literal["fp32", "fp16", "int8", "int16", "w8a16"] = "fp32"
    weight_type:     Literal["fp32", "fp16", "uint8", "int8", "uint16", "int16"] | None = None
    activation_type: Literal["fp32", "fp16", "uint8", "int8", "uint16", "int16"] | None = None
    symmetric: bool = False
    per_channel_weights: bool = True

    def __post_init__(self) -> None:
        # If weight_type / activation_type unspecified, derive from precision
        # via the same defaults the precision parser uses today.
        if self.weight_type is None:
            object.__setattr__(self, "weight_type",
                               _DEFAULT_WEIGHT_FOR_PRECISION.get(self.precision))
        if self.activation_type is None:
            object.__setattr__(self, "activation_type",
                               _DEFAULT_ACTIVATION_FOR_PRECISION.get(self.precision))


@dataclass(frozen=True, kw_only=True, slots=True)
class EPDeviceSpec:
    ep: str
    device: str
    default_provider_options: Mapping[str, str] = field(default_factory=dict)
    default_quant: QuantSpec | None = None              # NEW


@dataclass(frozen=True)
class EPDevice:
    ep: str
    device: str
    vendor_id: int
    device_id: int
    vendor: str = ""
    quant: QuantSpec | None = None                      # NEW
```

## 5. Catalog impact

Catalog entries gain a `default_quant=` argument. Only **verified** variants get a non-`None` value; others stay `None` (= "no opinion") until measured.

```python
EP_DEVICE_SPECS: tuple[EPDeviceSpec, ...] = (
    # ---- Verified (have real numbers) ----
    EPDeviceSpec(
        ep="QNNExecutionProvider", device="npu",
        default_provider_options={
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
        default_quant=QuantSpec(
            precision="int8",
            weight_type="uint8",
            activation_type="uint16",     # HTP's unusual choice
            symmetric=False,
            per_channel_weights=False,    # HTP requires per-tensor
        ),
    ),
    EPDeviceSpec(ep="DmlExecutionProvider", device="gpu",
                 default_quant=QuantSpec(precision="fp16")),
    EPDeviceSpec(ep="CPUExecutionProvider", device="cpu",
                 default_quant=None),     # CPU runs any precision; no opinion

    # ---- Plausible defaults (not yet measured) ----
    EPDeviceSpec(ep="VitisAIExecutionProvider", device="npu",
                 default_quant=QuantSpec(
                     precision="int8",
                     weight_type="int8",          # AMD wants signed
                     activation_type="int8",
                     symmetric=True,
                     per_channel_weights=True,
                 )),
    # ... other variants: default_quant=None (no claim until verified)
)
```

**Catalog ordering still encodes deduction preference** — unchanged. `default_quant` is per-entry metadata, not a tiebreaker.

## 6. Consumer impact

### `config/precision.py`

| Today | After |
|---|---|
| `_AUTO_PRECISION: dict[str, str]` (device → precision) | Stays (it's user-facing, drives `--precision auto`). Optionally seed from catalog. |
| `_WEIGHT_TYPE`, `_ACTIVATION_TYPE` (precision → type, single-valued globals) | Stays for **user-supplied precision strings** (`--precision int8` still needs `("uint8", "uint16")`). |
| `_BITS_TO_WEIGHT_TYPE`, `_BITS_TO_ACTIVATION_TYPE` (bit-width → type, with TODO at line 64) | **Move into per-variant catalog entries.** TODO at line 64 resolved. |
| `resolve_quant_types(precision)` (current public function, no EP awareness) | Stays for user-string path. **New** `resolve_quant_types_for_target(ep_device)` reads `ep_device.quant` directly. |

The two paths coexist: user-precision-string → global tables (unchanged), EP-device-target → catalog. This preserves backward compatibility and matches how today's `default_provider_options` coexists with user-supplied provider options.

### `commands/quantize.py`

Today: hard-coded bit-width tables driven by `--precision`. After: if a target EP device is known (`--ep`, `--device`), pull `ep_device.quant` and use it; otherwise fall back to the user-string path. Net result: `winml quantize -m model.onnx --ep qnn --device npu` automatically produces the right HTP-shaped QDQ without the user needing to know HTP wants uint16 activations.

### `commands/compile.py`, `commands/perf.py`

Optional, low-priority: read `ep_device.quant` for **validation** ("model is fp32 but target is int8 — runtime conversion will happen") and **diagnostics**. No behavior change required in this PR.

### `WinMLSession`

No API change. `WinMLSession(ep_device, ep_config)` still works as today. `ep_device.quant` is available if a future consumer wants it.

## 7. What this enables vs. what stays as-is

| Use case | Today | After |
|---|---|---|
| `winml quantize -m m.onnx --ep qnn --device npu` produces HTP-correct QDQ | Yes (the global tables happen to match HTP) | Yes, **and** the correctness is **declared** rather than incidental |
| `winml quantize -m m.onnx --ep vitisai --device npu` produces AMD-correct QDQ | **NO** — globals are HTP-shaped | Yes, catalog declares VitisAI wants `int8/int8/symmetric/per_channel` |
| Adding a new EP variant (e.g., MIGraphX-NPU) needs custom quant | Touches `precision.py` (the TODO would have to be implemented) | One catalog line. No `precision.py` changes. |
| `winml compile` warns on model-vs-target precision mismatch | Not implemented | Possible (post-hoc, optional follow-up) |
| User passes `--precision int8` explicitly | Works (current global tables) | Same — user-string path unchanged |

## 8. Trade-offs

**Pros**

- Per-variant quant requirements live next to per-variant provider options — symmetric design.
- Resolves the `precision.py:64` TODO.
- Makes the `winml quantize` + downstream EP variants actually correct for non-QNN EPs.
- Keeps the catalog as single-source-of-truth (the same invariant the EPDeviceSpec PR established).

**Cons / risks**

- **Field set is opinionated.** Real quantize libraries have more knobs (calibration method, bias signed, exclude_ops, quant_format MQAT/QDQ/QOperator…). The proposed 5 fields cover *current* per-variant variance but not the full quantize-library surface. Mitigation: start minimal; extend later if measured variance demands it.
- **Coupling.** Adds a build-pipeline concept (quantization) into a session module (`session/ep_device.py`). The alternative is a separate `quant/spec.py` module that `EPDeviceSpec` references — cleaner separation but two-file complexity. **Tentative pick: keep in `session/ep_device.py`** since the spec catalog is small and the dependency direction is `quantize.py → session/`, which we already have.
- **Catalog entries with `default_quant=None` are silent on what they need.** Documentation-only mitigation — comment which variants are unverified vs. "no opinion intended".
- **Serialization.** `EPDevice` adding a nested dataclass field means `to_dict` / `from_dict` (if any) need updating. Low effort; standard dataclass machinery handles it.

## 9. Scope options for the rollout

This is **substantial new design work**, not a cleanup. Three landing options:

| Option | Scope | LOC est. | Risk | When to pick |
|---|---|---|---|---|
| **(a) DEFER** | Capture this doc + close current PR | 0 (this file) | None | If reviewer of the EPDeviceSpec PR wants minimal additional surface area |
| **(b) MINIMAL — types only** | Add `QuantSpec` + extend `EPDeviceSpec` with `default_quant` field. No consumer migration. No catalog entries beyond `default_quant=None` everywhere. | ~50 | Low (additive; no behavior change) | If we want the type slot in place so future work can plug in without churning catalog |
| **(c) FULL — types + migration + tests** | (b) + populate catalog defaults for verified variants + migrate `precision.py:_BITS_TO_*` + wire into `quantize.py` + tests | ~300-500 | Moderate (touches quantize pipeline; tests required) | If the precision.py TODO is felt as active pain in this PR's scope |

**My lean: (a) DEFER**, file as follow-up. Reasons:

- This PR's stated goal was EP/device taxonomy consolidation. QuantSpec adds a new dimension (quantization) that wasn't in the original scope.
- Reviewers reading the current PR's diff are not primed for quant-pipeline changes.
- A dedicated follow-up PR can have its own design doc finalization, plan (v1/v2/v3 pattern that worked here), and verification matrix.

**If (c) is wanted in-PR**, it's tractable but expands the diff by ~500 LOC and adds quantize-pipeline review surface.

## 10. Open questions (must answer before promoting from DRAFT)

1. **Field set finalization** — are the 5 fields (`precision`, `weight_type`, `activation_type`, `symmetric`, `per_channel_weights`) sufficient for **all current verified variants**? Need to enumerate what QNN-HTP, DML-GPU, VitisAI-NPU, OpenVINO-NPU each *actually* require and confirm coverage.
2. **`None` semantics** — does `default_quant=None` mean "no opinion" or "any precision OK"? Pick one and document. (Proposal: "no opinion" — quant pipeline falls back to user-string path.)
3. **`quant=` override behavior** — when user passes `resolve_device(ep, device, quant=user_spec)`, does `user_spec` *override* the catalog default entirely, or *merge* field-by-field? (Proposal: full override — simpler. Field-merge is a future extension.)
4. **Validation policy** — should `compile`/`perf` warn / error / silently accept when model's QDQ scheme disagrees with `ep_device.quant`? (Proposal: warning only; never error. Runtime can still convert.)
5. **Naming** — `QuantSpec` vs `QuantizationSpec` vs `QuantScheme`? (Proposal: `QuantSpec` — mirrors `EPDeviceSpec` brevity.)

## 11. Decision needed (from user)

- [ ] **Direction approved?** (proceed to plan + impl) — or DRAFT to be revised
- [ ] **Landing option:** (a) defer / (b) minimal / (c) full
- [ ] **If (a) defer:** file follow-up issue with this doc as the seed
- [ ] **If (b) or (c):** open questions in §10 must be answered first

## Appendix A — verification gap (NPU + FP32 vs NPU + QDQ)

The EPDeviceSpec PR's CLI verification matrix exercised:

| Command | EP | Device | Model used |
|---|---|---|---|
| `winml perf` | qnn | cpu | FP32 ONNX |
| `winml perf` | qnn | gpu | FP32 ONNX |
| `winml perf` | qnn | npu | **FP32 ONNX** ← canonical NPU workflow is QDQ; this gap is the verification motivation |
| `winml compile` | qnn | cpu | FP32 ONNX |
| `winml compile` | qnn | gpu | FP32 ONNX |
| `winml compile` | qnn | npu | FP32 ONNX |

The FP32-on-NPU result (2.01ms, 498 samples/s on ResNet-50) is a valid number — it reflects ORT/QNN's runtime FP32-to-int8 conversion path. But it is **not** the canonical QDQ-direct-to-HTP path that production workflows use.

### Verification — ResNet-50 on QNN-NPU (2026-05-15, 50 iter / 5 warmup, Snapdragon X-Elite)

| Path | Model artifact | Avg latency | P50 | P99 | Throughput |
|---|---|---:|---:|---:|---:|
| FP32 → NPU (runtime conv) | `_export.onnx` (102 MB FP32) | 2.01 ms | — | — | 498 /s |
| **QDQ → NPU (runtime AOT compile)** | `_quantized.onnx` (25 MB QDQ) | **0.73 ms** | 0.72 | 0.83 | **1368 /s** |
| **QDQ → NPU (pre-compiled ctx)** | `_quantized_npu_ctx.onnx` + `_qnn.bin` (~26 MB) | **0.80 ms** | 0.73 | 3.04* | **1247 /s** |

\* P99=3.04ms is a single cold-call spike on the pre-compiled path; P50/P90 match the runtime-AOT path (0.73/0.84ms). Steady-state cost is identical between AOT-compile and pre-compiled paths once warmup amortizes.

**Findings:**

- QDQ-direct-to-HTP path is **2.75× faster** than FP32-runtime-converted (1368/s vs 498/s).
- Both QDQ paths exercise the same QNN HTP compile stages (Graph Preparation, Graph Optimizations, Graph Sequencing, VTCM Allocation, Parallelization Optimization, Finalizing). The runtime-AOT path pays this cost on first session creation; the pre-compiled `.ctx.onnx` path skips it but pays I/O cost for the larger `.qnn.bin` instead.
- The benign warning `"Some nodes were not assigned to the preferred execution providers"` appears on both paths — ORT routes shape-related ops to CPU, which is expected and not a regression.
- The EPDeviceSpec refactor does **not** break the canonical QDQ → QNN HTP NPU path (the only meaningful difference between this PR and `main` for this workload is the `htp_performance_mode='burst'` default, which is what's driving the 2.75× speedup).

**Conclusion:** Canonical NPU workflow verified end-to-end. The earlier matrix gap is closed.

## Appendix B — relationship to `2026-05-13-ep-device-spec-design.md`

That doc established the spec→instance pattern with two fields (`ep`, `device`) plus `default_provider_options`. This doc extends *the same pattern* with one more field (`default_quant`). No invariants from the predecessor doc are broken:

- ✅ Single source of truth: `EP_DEVICE_SPECS` remains the only place declaring (EP, device) variants.
- ✅ Catalog ordering still encodes deduction preference (`default_quant` is metadata, not a tiebreaker).
- ✅ Frozen dataclass, `slots=True`, `kw_only=True` — same discipline.
- ✅ Machine-independent (Spec) vs machine-specific (Instance) split preserved.

If the design is approved, this is a strict extension, not a redesign.
