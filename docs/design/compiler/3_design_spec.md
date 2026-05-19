# `WinMLCompileSpec` — Per-Target Compile Capability Catalog

**Version**: 1.0
**Date**: 2026-05-19
**Status**: Draft
**Module**: compiler
**Companion-To**: [`../session/3_design_ep.md`](../session/3_design_ep.md) — Stage 2 device handle selection (EPDeviceSpec catalog is the upstream)
**Depends-On**: `session/ep_device.py:EPDeviceSpec`, `session/ep_device.py:EP_DEVICE_SPECS`

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Scope](#2-scope)
- [3. Why a Separate Catalog](#3-why-a-separate-catalog)
- [4. Design](#4-design)
- [5. Catalog Contents](#5-catalog-contents)
- [6. Consumer Changes](#6-consumer-changes)
- [7. Migration](#7-migration)
- [8. Testing](#8-testing)
- [9. Open Questions](#9-open-questions)
- [10. Appendix](#10-appendix)

---

## 1. Executive Summary

`WinMLSession.compile()` today calls `ort.ModelCompiler.compile_to_file()` for every (EP, device) and silently falls back to the original model on failure with a `logger.warning` (see `session/session.py:347-349`). Users running `winml compile --ep cpu` get a "compile succeeded" exit even though no compile artifact was produced — only a copied-through original model. The eight per-EP factory methods on `WinMLCompileConfig` (`for_qnn`, `for_cpu`, ...) encode one bit of data per EP — whether `enable_ep_context` defaults `True` or `False` — across 100+ lines of repeated boilerplate.

This spec introduces a typed **`WinMLCompileSpec`** dataclass plus a per-(EP, device) catalog **`COMPILE_SPECS`** under `compiler/spec.py`, parallel to `session/ep_device.py:EP_DEVICE_SPECS`. The catalog declares three v1 capability flags per target:

- **`supports_ep_context: bool`** — does this (EP, device) actually produce an EPContext artifact via `ort.ModelCompiler`?
- **`supports_dynamic_shapes: bool`** — does this (EP, device) tolerate dynamic-shape inputs (`False` for NPU targets that require static-shape baking)?
- **`quant_format: Literal["qdq", "qlinear"] | None`** — declared quantization format; `None` means no quant support.

`winml compile` consults `supports_ep_context` up-front:
- `True`: invoke `ort.ModelCompiler.compile_to_file()` (current behavior).
- `False`: skip the compile call, copy the input to the output dir, log "EP `<X>` does not support compile; original model used as-is."

The per-EP `for_xxx` factories on `WinMLCompileConfig` and `for_provider(str)` are deleted in the same change — the typed `for_ep_device(EPDevice)` factory becomes the single entry point, consulting the new catalog for `enable_ep_context` defaults.

## 2. Scope

**In scope:**
- New `WinMLCompileSpec` dataclass + `COMPILE_SPECS` tuple in `compiler/spec.py`.
- New `get_compile_spec(ep_device_spec: EPDeviceSpec) -> WinMLCompileSpec` lookup.
- Re-export of the above from `winml.modelkit.compiler` package.
- Deletion of 8 per-EP factories (`for_qnn`, ..., `for_migraphx`) and `for_provider(str)` from `WinMLCompileConfig`.
- Migration of 3 internal callers in `config/build.py` + 2 test sites in `tests/unit/models/auto/test_config.py`.
- Behavior change in `WinMLSession.compile()`: gate `ort.ModelCompiler` invocation on `supports_ep_context`. Permissive UX — pass-through with clear log for non-EPContext EPs.
- Architecture test enforcing 1:1 (ep, device) key parity between `EP_DEVICE_SPECS` and `COMPILE_SPECS`.

**Out of scope:**
- Additional capability flags beyond the three (e.g., `supports_fp16`, `requires_calibration`, `compile_artifact_format`). Add when concrete consumers need them.
- Alternative compile mechanisms (TensorRT engine plan emission, OpenVINO IR generation). Today only `ort.ModelCompiler` + EPContext is the compile path.
- Quant-format auto-selection based on `WinMLCompileSpec.quant_format`. The flag exists but downstream wiring into `WinMLQuantizationConfig.mode` is a follow-up (the field is currently caller-set).
- Static-shape baking driven by `supports_dynamic_shapes`. The flag exists but the shape-resolution step that consumes it is a follow-up.

## 3. Why a Separate Catalog

The natural alternative is to extend `EPDeviceSpec` with the three new fields directly. We rejected this because:

1. **Layering integrity.** `session/ep_device.py` is the session-layer catalog. Its existing fields (`ep`, `device`, `default_provider_options`) are session/EP-routing concerns — what does ORT need to bind this EP to this device. Adding compile/quant capability flags pulls compile-pipeline knowledge into the session module. The existing codebase enforces module boundaries (`session/`, `compiler/`, `quant/` are distinct concerns).

2. **`EPDeviceSpec` stays focused.** Tests for `session/ep_device.py` shouldn't need to construct compile-capability values to instantiate fixtures. A separate `WinMLCompileSpec` keeps the session module's surface area unchanged.

3. **Growth axis.** Future compile/quant capability fields (`supports_fp16`, `requires_calibration`, `compile_artifact`, etc.) accumulate on `WinMLCompileSpec`, not on `EPDeviceSpec`. The session catalog doesn't grow wide as the compile catalog matures.

4. **Drift risk is bounded.** The concern with two catalogs is forgetting to add a row when a new EP lands. One architecture test (`{(s.ep, s.device) for s in EP_DEVICE_SPECS} == {(s.ep, s.device) for s in COMPILE_SPECS}`) catches drift loudly.

The cost is one duplicated bit per row — both catalogs carry `(ep, device)` strings. The architecture test makes the duplication self-correcting.

## 4. Design

### 4.1 The dataclass

```python
# src/winml/modelkit/compiler/spec.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from ..session import EPDeviceSpec


@dataclass(frozen=True)
class WinMLCompileSpec:
    """Per-(EP, device) compile/quant capability descriptor.

    Sibling to ``EPDeviceSpec`` (session layer). Where ``EPDeviceSpec``
    captures what ORT needs to bind an EP to a hardware device, this
    captures what the compile/quant pipeline can do with that target.

    Fields:
        ep: Canonical full EP name (matches EPDeviceSpec.ep).
        device: Device category — "npu" | "gpu" | "cpu".
        supports_ep_context: True iff ``ort.ModelCompiler.compile_to_file()``
            produces a meaningful EPContext artifact for this target. False
            means ``winml compile`` should skip the compile call and emit
            the original model unchanged with a clear message.
        supports_dynamic_shapes: False iff the target requires static
            shapes (NPUs typically; OpenVINO NPU requires reshape_input,
            QNN NPU bakes shapes into the HTP context binary).
        quant_format: Declared quantization format the target accepts.
            None means no quant support — quantization should not run for
            this target.
    """

    ep: str
    device: str
    supports_ep_context: bool = False
    supports_dynamic_shapes: bool = True
    quant_format: Literal["qdq", "qlinear"] | None = None
```

### 4.2 The catalog

```python
# src/winml/modelkit/compiler/spec.py (continued)

COMPILE_SPECS: Final[tuple[WinMLCompileSpec, ...]] = (
    # Primary per-device (matches EP_DEVICE_SPECS order)
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="npu",
        supports_ep_context=True,
        supports_dynamic_shapes=False,  # HTP context bakes shapes
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="DmlExecutionProvider", device="gpu",
        supports_ep_context=False,  # DML doesn't emit EPContext today
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="CPUExecutionProvider", device="cpu",
        supports_ep_context=False,  # CPU doesn't compile
        supports_dynamic_shapes=True,
        quant_format="qlinear",  # traditional QOperator path
    ),
    # QNN secondary
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="gpu",
        supports_ep_context=True,
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="cpu",
        supports_ep_context=False,  # QNN-CPU reference backend, no EPContext
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    # OpenVINO family
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="npu",
        supports_ep_context=True,
        supports_dynamic_shapes=False,  # reshape_input required
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="gpu",
        supports_ep_context=True,
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="cpu",
        supports_ep_context=True,
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    # Single-device EPs
    WinMLCompileSpec(
        ep="VitisAIExecutionProvider", device="npu",
        supports_ep_context=False,  # VitisAI uses its own caching, not EPContext
        supports_dynamic_shapes=False,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="MIGraphXExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="TensorrtExecutionProvider", device="gpu",
        supports_ep_context=True,  # TensorRT engine wrapped in EPContext (ORT 1.24+)
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="CUDAExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="NvTensorRtRtxExecutionProvider", device="gpu",
        supports_ep_context=True,
        supports_dynamic_shapes=True,
        quant_format="qdq",
    ),
)
```

The 13 entries are 1:1 with `EP_DEVICE_SPECS` rows. Some `supports_ep_context` values are conservative best-guesses pending empirical verification on real hardware — see §9 Open Questions.

### 4.3 The lookup

```python
# src/winml/modelkit/compiler/spec.py (continued)

def get_compile_spec(ep_device_spec: EPDeviceSpec) -> WinMLCompileSpec:
    """Look up compile capabilities for a session-layer EPDeviceSpec entry.

    Args:
        ep_device_spec: A row from ``EP_DEVICE_SPECS`` (typically obtained
            via ``lookup_device_spec(ep, device)`` from the session module).

    Returns:
        The matching WinMLCompileSpec row from COMPILE_SPECS, or a
        conservative no-capability fallback for uncatalogued (ep, device)
        pairs (supports_ep_context=False, supports_dynamic_shapes=True,
        quant_format=None). The fallback is intentionally noisy at
        consumer code paths — uncatalogued EPs fail at compile / quant
        gates rather than producing wrong output silently.
    """
    for spec in COMPILE_SPECS:
        if spec.ep == ep_device_spec.ep and spec.device == ep_device_spec.device:
            return spec
    return WinMLCompileSpec(ep=ep_device_spec.ep, device=ep_device_spec.device)
```

### 4.4 Package re-exports

Per CLAUDE.md import convention (`src/` uses package-level imports through `__init__.py`):

```python
# src/winml/modelkit/compiler/__init__.py

from .spec import COMPILE_SPECS, WinMLCompileSpec, get_compile_spec

__all__ = [
    "COMPILE_SPECS",
    "WinMLCompileSpec",
    "get_compile_spec",
    # ... existing exports
]
```

Consumers always import from `winml.modelkit.compiler`, never from `winml.modelkit.compiler.spec`.

## 5. Catalog Contents

The 13 (ep, device) target rows mirror `EP_DEVICE_SPECS` exactly. Verification levels per cell:

| EP / device | `supports_ep_context` | `supports_dynamic_shapes` | `quant_format` |
|---|---|---|---|
| QNN / npu | ✅ verified (CI) | ✅ verified (HTP) | ✅ verified |
| QNN / gpu | ❓ conservative-true | ❓ unverified | ❓ unverified |
| QNN / cpu | ❓ conservative-false | ❓ unverified | ❓ unverified |
| DML / gpu | ❓ false (DML has no EPContext today) | ✅ verified | ❓ unverified |
| CPU / cpu | ✅ verified false | ✅ verified | ❓ unverified |
| OpenVINO / npu | ✅ verified (this branch) | ✅ verified (reshape_input req) | ✅ verified |
| OpenVINO / gpu | ✅ verified (this branch) | ✅ verified | ❓ unverified |
| OpenVINO / cpu | ✅ verified (this branch) | ✅ verified | ❓ unverified |
| VitisAI / npu | ❓ unverified (no host) | ❓ unverified | ❓ unverified |
| MIGraphX / gpu | ❓ unverified (no host) | ❓ unverified | ❓ unverified |
| Tensorrt / gpu | ❓ unverified (no host) | ✅ doc | ❓ unverified |
| CUDA / gpu | ❓ unverified (no host) | ✅ doc | ❓ unverified |
| NvTensorRtRtx / gpu | ❓ unverified (no host) | ❓ unverified | ❓ unverified |

Unverified cells are conservative defaults. Updating them is a low-risk follow-up as hardware coverage expands. The architecture test pins the keyset; the values can be tightened later.

## 6. Consumer Changes

### 6.1 `WinMLCompileConfig.for_ep_device` — reads `supports_ep_context`

Today (post-prior-refactor in `compiler/configs.py`):
```python
@classmethod
def for_ep_device(cls, ep_device: EPDevice) -> WinMLCompileConfig:
    from ..session import short_ep_name
    provider = short_ep_name(ep_device.ep)
    base = cls.for_provider(provider) or cls(ep_config=EPConfig(provider=provider))
    base.ep_device = ep_device
    return base
```

After:
```python
@classmethod
def for_ep_device(cls, ep_device: EPDevice) -> WinMLCompileConfig:
    """Build a compile config from a resolved EPDevice."""
    from ..session import lookup_device_spec, short_ep_name
    from . import get_compile_spec

    ep_device_spec = lookup_device_spec(ep_device.ep, ep_device.device)
    compile_spec = get_compile_spec(ep_device_spec)
    return cls(
        ep_config=EPConfig(
            provider=short_ep_name(ep_device.ep),
            enable_ep_context=compile_spec.supports_ep_context,
        ),
        ep_device=ep_device,
    )
```

### 6.2 `WinMLSession.compile()` — permissive pass-through on non-EPContext EPs

Today (`session/session.py:288-349`): always attempts `ort.ModelCompiler`, falls back on exception with a `WARNING`. Output: silently the original model for non-EPContext EPs.

After: consult `WinMLCompileSpec.supports_ep_context` up-front.

```python
def compile(self) -> None:
    # ... existing idempotency + cache check ...

    from ..compiler import get_compile_spec
    from ..session import lookup_device_spec

    ep_device_spec = lookup_device_spec(self._ep_device.ep, self._ep_device.device)
    compile_spec = get_compile_spec(ep_device_spec)

    if not compile_spec.supports_ep_context:
        logger.info(
            "%s on %s does not support EPContext compilation; "
            "using original model unchanged.",
            self._ep_device.ep, self._ep_device.device,
        )
        # Fall through to InferenceSession creation against the original model.
        model_path = self._onnx_path
    else:
        # ... existing ModelCompiler.compile_to_file() flow ...
        model_path = ctx_path  # or self._onnx_path on fallback
```

### 6.3 Deletion: 8 per-EP factories + `for_provider`

The `_EP_CONTEXT_DEFAULTS` table introduced earlier in this branch is also superseded — the data lives in the catalog now. Delete:

- `WinMLCompileConfig.for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_nv_tensorrt_rtx`, `for_openvino`, `for_vitisai`, `for_migraphx`
- `WinMLCompileConfig.for_provider(provider: str | None)`
- The local `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]]` (in-flight, not yet committed)

After deletion, `for_ep_device` is the sole factory.

### 6.4 Migration of callers in `config/build.py`

Three sites currently use `WinMLCompileConfig.for_provider(short_str)`:

- `config/build.py:307` — `resolve_quant_compile_config`
- `config/build.py:604` — `generate_hf_build_config` policy path
- `config/build.py:616` — `generate_hf_build_config` hw-detected path

Each migrates to the typed pattern:

```python
# Before
compile_config = WinMLCompileConfig.for_provider(policy.compile_provider)

# After
from ..session import resolve_device
ep_device = resolve_device(ep=policy.compile_provider, device=policy.device)
compile_config = WinMLCompileConfig.for_ep_device(ep_device)
```

**Side effect:** `resolve_device` registers the EP plugin DLL as a side effect. This means config generation now requires the target EP to be installed locally. The pre-this-branch workflow of "generate a QNN config on a non-QNN host for cross-host deployment" no longer works without manually constructing an EPDevice. This is consistent with the branch's typed-everywhere stance and acceptable per the explicit no-back-compat directive.

### 6.5 `winml compile` CLI — permissive output

`commands/compile.py` calls `WinMLSession.compile()` and produces a user-visible artifact in `--output-dir`. With v1, the artifact depends on `supports_ep_context`:

- **`supports_ep_context=True`**: existing flow. `*_ctx.onnx` (+ optional `*.bin` sidecar) written to output dir.
- **`supports_ep_context=False`**: copy the input ONNX to the output dir under the input filename. Print a clear message: `"<EP> on <device> does not support EPContext compilation; copied original model to <output_path>."` Exit 0.

This matches the user-facing "permissive" UX: every `winml compile <model> --ep <X> --device <Y>` invocation succeeds and produces a model file at the output path. Users targeting non-EPContext EPs get the original model back with explicit messaging — they know nothing was compiled.

Implementation site: `commands/compile.py` after the `WinMLSession.compile()` call returns. The compile spec is already in hand via the resolved `EPDevice` at the CLI boundary.

### 6.6 Quant-format consumption (future)

`WinMLCompileSpec.quant_format` is set in v1 but not yet consumed by `WinMLQuantizationConfig`. A follow-up will:

```python
spec = get_compile_spec(lookup_device_spec(ep, device))
if spec.quant_format is None:
    raise ValueError(f"{ep}/{device} does not support quantization")
quant_config = WinMLQuantizationConfig(mode=spec.quant_format)
```

This isn't in v1's scope — `WinMLQuantizationConfig.mode` remains caller-set.

### 6.7 Dynamic-shape baking (future)

`supports_dynamic_shapes=False` rows (QNN-NPU, OpenVINO-NPU, VitisAI-NPU) need a shape-baking step before compile. The hook is the flag; the shape-resolution code that consumes it is a follow-up.

## 7. Migration

### 7.1 Files added

- `src/winml/modelkit/compiler/spec.py` — new module (~150 LOC including catalog rows).
- `tests/unit/compiler/test_spec.py` — new file. Architecture-test + per-row sanity.

### 7.2 Files modified

- `src/winml/modelkit/compiler/configs.py` — delete 8 per-EP factories + `for_provider` (~140 LOC removed). Update `for_ep_device` body (~10 LOC changed). Update class docstring examples.
- `src/winml/modelkit/compiler/__init__.py` — re-export `WinMLCompileSpec`, `COMPILE_SPECS`, `get_compile_spec`.
- `src/winml/modelkit/compiler/compiler.py` — update docstring examples (lines 49, 186, 189, 192).
- `src/winml/modelkit/session/session.py` — modify `compile()` body to consult `supports_ep_context`.
- `src/winml/modelkit/config/build.py` — migrate 3 caller sites.
- `src/winml/modelkit/commands/compile.py` — add the permissive `supports_ep_context=False` output-file path (copy input to output dir + clear log).

### 7.3 Tests modified

- `tests/unit/compiler/test_compiler_configs.py` — delete the test class(es) for per-EP factories and `for_provider` (~40 LOC). Keep `for_ep_device` tests; expand to cover the new `supports_ep_context`-driven path.
- `tests/unit/models/auto/test_config.py:169,172` — replace `for_qnn()`/`for_cpu()` with `for_ep_device(EPDevice(...))`.

### 7.4 Total blast radius

- ~150 LOC added (new spec.py + tests)
- ~180 LOC deleted (factories + tests for them)
- 6 files modified
- Net delta: roughly -30 LOC, much cleaner public surface

## 8. Testing

### 8.1 Architecture test (load-bearing)

```python
# tests/unit/architecture/test_compile_spec_coverage.py
from winml.modelkit.session import EP_DEVICE_SPECS
from winml.modelkit.compiler import COMPILE_SPECS

def test_compile_specs_cover_all_ep_device_specs():
    """Every (ep, device) in EP_DEVICE_SPECS has exactly one row in COMPILE_SPECS."""
    ep_device_keys = {(s.ep, s.device) for s in EP_DEVICE_SPECS}
    compile_keys = {(s.ep, s.device) for s in COMPILE_SPECS}
    assert ep_device_keys == compile_keys, (
        f"Missing in COMPILE_SPECS: {ep_device_keys - compile_keys}\n"
        f"Extra in COMPILE_SPECS: {compile_keys - ep_device_keys}"
    )

def test_compile_specs_have_unique_keys():
    """No duplicate (ep, device) rows in COMPILE_SPECS."""
    keys = [(s.ep, s.device) for s in COMPILE_SPECS]
    assert len(keys) == len(set(keys)), f"Duplicates: {keys}"
```

### 8.2 Per-row sanity

```python
def test_quant_format_consistency():
    """If quant_format is set, it must be 'qdq' or 'qlinear'."""
    for spec in COMPILE_SPECS:
        if spec.quant_format is not None:
            assert spec.quant_format in {"qdq", "qlinear"}

def test_npu_targets_require_static_shapes():
    """All NPU targets we ship have supports_dynamic_shapes=False (regression guard)."""
    for spec in COMPILE_SPECS:
        if spec.device == "npu" and spec.ep in {"QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider"}:
            assert not spec.supports_dynamic_shapes, (
                f"{spec.ep}/{spec.device}: NPUs need static shapes baked in"
            )
```

### 8.3 Lookup fallback

```python
def test_get_compile_spec_returns_conservative_default_for_uncatalogued():
    """Unknown (ep, device) pair returns a no-capability fallback, not an error."""
    fake = EPDeviceSpec(ep="FakeEP", device="npu")
    result = get_compile_spec(fake)
    assert result.ep == "FakeEP"
    assert result.device == "npu"
    assert result.supports_ep_context is False
    assert result.supports_dynamic_shapes is True
    assert result.quant_format is None
```

### 8.4 Permissive-compile UX test

```python
def test_winml_session_compile_logs_and_passes_through_on_non_epcontext_ep():
    """Compile on a no-EPContext EP emits a clear log + returns the original model."""
    # Build a session targeting CPU (supports_ep_context=False).
    # Call session.compile(). Assert:
    # - No ort.ModelCompiler invocation occurred (mock)
    # - Log message contains "does not support EPContext"
    # - The InferenceSession uses self._onnx_path, not a *_ctx.onnx
```

## 9. Open Questions

### OQ1 — Hardware-verification gaps

The catalog rows marked ❓ in §5 are conservative defaults that need empirical verification:

- **QNN-GPU / QNN-CPU**: do these support EPContext? Reference path on QNN-CPU and Adreno on QNN-GPU may not emit context binaries.
- **DML-GPU**: `supports_ep_context=False` is the documented current state, but ORT 1.24+ has experimental DML EPContext support — verify.
- **TensorRT-GPU / CUDA-GPU / NvTensorRtRtx-GPU**: EPContext for TensorRT engine plans is recent (ORT 1.24); behavior on the dual-NV-EP host needs verification.
- **VitisAI-NPU / MIGraphX-GPU**: no AMD silicon in CI today. Marked conservative until verified.

**Recommendation:** ship v1 with conservative defaults. Open a follow-up issue to verify and tighten cells as hardware coverage expands. The arch test pins the keyset; values can move without API change.

### OQ2 — Should `quant_format` be a list, not a single literal?

DML and CPU EPs accept BOTH QDQ and QLinear formats in modern ORT. Picking one as "the format" is a soft choice. v1 picks one per row; if users surface a need for multi-format support, switch to `frozenset[Literal["qdq", "qlinear"]]`.

### OQ3 — `WinMLCompileConfig.ep_config.enable_ep_context` interaction

If a user manually sets `config.ep_config.enable_ep_context = True` after `for_ep_device()` populated it from the catalog as `False` — what wins? Today the user override wins (we just set the attribute, no validation). This is intentional — caller intent overrides catalog defaults. The catalog provides a sensible default, not a hard constraint.

### OQ4 — Should `get_compile_spec` raise on uncatalogued targets?

v1 returns a conservative default (`supports_ep_context=False`). The argument for raising: catches typos/drift earlier. The argument for default: caller code is more uniform (no exception handling in consumer paths). v1 chooses default + arch-test-enforced 1:1 coverage to catch drift centrally.

### OQ5 — Future capability flags

When concrete consumers need them, add to `WinMLCompileSpec`:
- `supports_fp16: bool`, `supports_int8: bool`, `supports_int16: bool` (precision support)
- `requires_calibration: bool` (static quant gate)
- `compile_artifact: Literal["ep_context", "engine_plan", "ir", "none"]` (when we add non-EPContext compile paths)
- `supports_per_channel_quant: bool` (DML doesn't; QNN does)

Not in v1 — add when the consuming code is ready.

## 10. Appendix

### 10.1 Glossary

| Term | Meaning |
|---|---|
| **EPDevice** | Project-defined plain-data descriptor (frozen dataclass) at `session/ep_device.py`. The resolved (EP, device) descriptor. |
| **EPDeviceSpec** | The session-layer catalog row — static template `(ep, device, default_provider_options)`. One per (EP, device) target. |
| **EP_DEVICE_SPECS** | Tuple of 13 EPDeviceSpec rows in `session/ep_device.py`. Order encodes preference. |
| **WinMLCompileSpec** | This spec's contribution — the compile-layer catalog row. `(ep, device, supports_ep_context, supports_dynamic_shapes, quant_format)`. |
| **COMPILE_SPECS** | Tuple of 13 WinMLCompileSpec rows in `compiler/spec.py`. 1:1 with EP_DEVICE_SPECS keys. |
| **EPContext** | ORT's container format wrapping a pre-compiled EP-specific binary blob. Used by `ort.ModelCompiler.compile_to_file()`. |
| **QDQ** | Quantize-Dequantize quantization format (Q/DQ node pairs around floats). |
| **QLinear** | QOperator/QLinearOps quantization format (single quantized ops). |

### 10.2 References

- [`docs/design/session/3_design_ep.md`](../session/3_design_ep.md) — Stage 1+2 EP registration and device handle selection (the EPDevice / EPDeviceSpec upstream).
- [`docs/design/session/2026-05-13-ep-device-spec-design.md`](../session/2026-05-13-ep-device-spec-design.md) — EPDeviceSpec catalog design.
- `src/winml/modelkit/session/ep_device.py` — EPDeviceSpec definition, EP_DEVICE_SPECS catalog.
- `src/winml/modelkit/session/session.py:288-349` — current `WinMLSession.compile()` body.
- `src/winml/modelkit/compiler/configs.py` — current `WinMLCompileConfig` with the 8 per-EP factories to be deleted.
- ONNX Runtime ModelCompiler docs (EPContext format).

### 10.3 Document History

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-19 | Initial draft. Captures v1 capability flags (`supports_ep_context`, `supports_dynamic_shapes`, `quant_format`), `WinMLCompileSpec` catalog under `compiler/spec.py`, deletion of 8 per-EP factories + `for_provider(str)`, permissive-compile UX, architecture test for key parity. |

### 10.4 Code Reference Map

| Concern | File | Symbol |
|---|---|---|
| New: capability dataclass | `src/winml/modelkit/compiler/spec.py` | `WinMLCompileSpec` |
| New: catalog | `src/winml/modelkit/compiler/spec.py` | `COMPILE_SPECS` |
| New: lookup | `src/winml/modelkit/compiler/spec.py` | `get_compile_spec(ep_device_spec)` |
| New: re-exports | `src/winml/modelkit/compiler/__init__.py` | `WinMLCompileSpec`, `COMPILE_SPECS`, `get_compile_spec` |
| Modified: typed-only factory | `src/winml/modelkit/compiler/configs.py` | `WinMLCompileConfig.for_ep_device(ep_device)` |
| Modified: permissive compile | `src/winml/modelkit/session/session.py` | `WinMLSession.compile()` |
| Modified: caller sites | `src/winml/modelkit/config/build.py:307,604,616` | `resolve_quant_compile_config`, `generate_hf_build_config` |
| New: architecture test | `tests/unit/architecture/test_compile_spec_coverage.py` | key-parity assertion |
| New: catalog tests | `tests/unit/compiler/test_spec.py` | per-row sanity, fallback semantics |
| Modified: caller migration | `tests/unit/models/auto/test_config.py:169,172` | replace `for_qnn`/`for_cpu` with `for_ep_device` |
| Modified: factory-test cleanup | `tests/unit/compiler/test_compiler_configs.py` | delete per-EP factory + `for_provider` test classes |
