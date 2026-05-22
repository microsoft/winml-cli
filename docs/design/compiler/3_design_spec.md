# `WinMLCompileSpec` — Per-Target Compile Capability Catalog

**Version**: 1.3
**Date**: 2026-05-19
**Status**: Draft (Option B + uniform `{stem}_{device}.onnx` + NPU-only EPContext policy)
**Module**: compiler + session (lazy-init refactor)
**Companion-To**: [`../session/3_design_ep.md`](../session/3_design_ep.md) — Stage 2 device handle selection (WinMLEPDeviceSpec catalog is the upstream)
**Depends-On**: `session/ep_device.py:WinMLEPDeviceSpec`, `session/ep_device.py:EP_DEVICE_SPECS`

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

`WinMLSession.compile()` today calls `ort.ModelCompiler.compile_to_file()` for every (EP, device) and silently falls back to the original model on failure with a `logger.warning` (see `session/session.py:347-349`). Users running `winml compile --ep cpu` get a "compile succeeded" exit even though no compile artifact was produced — only a copied-through original model. The eight per-EP factory methods on `WinMLCompileConfig` (`for_qnn`, `for_cpu`, ...) encode one bit of data per EP — `enable_ep_context` defaulting `True` or `False` — across 100+ lines of repeated boilerplate. Meanwhile, `WinMLSession` carries a dual-path `_persist_jit` flag (eager session creation when `False`, deferred when `True`) that makes `compile()` partially dead-code for non-EPContext targets.

This spec introduces a typed **`WinMLCompileSpec`** dataclass plus a per-(EP, device) catalog **`COMPILE_SPECS`** under `compiler/spec.py`, parallel to `session/ep_device.py:EP_DEVICE_SPECS`. The catalog declares three v1 capability flags per target:

- **`supports_ep_context: bool`** — does this (EP, device) actually produce an EPContext artifact via `ort.ModelCompiler`?
- **`supports_dynamic_shapes: bool`** — does this (EP, device) tolerate dynamic-shape inputs (`False` for NPU targets that require static-shape baking)?
- **`quant_format: Literal["qdq", "qlinear"] | None`** — declared quantization format; `None` means no quant support.

**`CompileStage` is the single spec-aware layer.** It consults `WinMLCompileSpec.supports_ep_context` directly and branches:
- `True`: call `WinMLSession.compile()` → returns `Path` → relocate to `{output_dir}/{stem}_{device}.onnx`.
- `False`: skip `compile()` entirely; copy the input ONNX to `{output_dir}/{stem}_{device}.onnx`. INFO log: `"EP <X> on <device> does not support EPContext compilation; original model copied to <output_path>"`.

**Output filename is `{stem}_{device}.onnx` for both branches.** No `_ctx` suffix. The filename signals the targeted device, not the internal format. Whether the output happens to be an EPContext model is recorded by ONNX metadata, not by filename.

The per-EP `for_xxx` factories and `for_provider(str)` are deleted. **The `enable_ep_context` field on `EPConfig` is also deleted** — it was a derived copy of `supports_ep_context`. With the catalog as the single source of truth, the derived bit is dead weight. `EPConfig` shrinks to pure ORT/QNN runtime settings (`provider`, `provider_options`, `embed_context`, `compiler`, `qnn_sdk_root`).

**Three supporting refactors land in the same change** because they're load-bearing for the spec's behavior to actually fire:

1. **`WinMLSession` lazy-init refactor.** Drop `_persist_jit`. All session creation funnels through a private `_ensure_session()`. `__init__` becomes pure assignment. `compile()` returns `Path` (the written file) or **raises** — no silent fallback. The session is pure mechanism: it compiles when asked; the spec-aware caller (`CompileStage`) decides whether to ask.
2. **`CompileStage` slim.** Replace `_finalize_output`'s three-way filename search with `_relocate_ctx(ctx_path, output_dir, device)` consuming the exact Path from `compile()`. Drop the EP-string fallback at line 73-75. Other responsibilities retained: qairt/ort selection, validation, `.bin` sidecar rename, `ep_cache_context` patch.
3. **`deduce_ep_device()` companion to `resolve_device()`.** Returns an WinMLEPDevice from the catalog's deduction phase without registering the EP plugin. Required so `winml config` on cross-host workflows (e.g., generating a QNN config on x64 dev box) still works post-typed-everywhere migration.

## 2. Scope

**In scope:**
- New `WinMLCompileSpec` dataclass + `COMPILE_SPECS` tuple in `compiler/spec.py`.
- New `get_compile_spec(ep_device_spec: WinMLEPDeviceSpec) -> WinMLCompileSpec` lookup (**raises `KeyError` on uncatalogued (ep, device) pairs** — arch test guarantees no miss for known rows).
- Re-export of the above from `winml.modelkit.compiler` package.
- Deletion of 8 per-EP factories (`for_qnn`, ..., `for_migraphx`) and `for_provider(str)` from `WinMLCompileConfig`.
- **Deletion of `enable_ep_context` field from `EPConfig`** — superseded by `WinMLCompileSpec.supports_ep_context` consulted at `CompileStage`. `EPConfig` shrinks to runtime-only fields.
- Migration of 3 internal callers in `config/build.py` (target-host build, uses `resolve_device`) + `commands/config.py` (cross-host config-gen, uses new `deduce_ep_device`) + 2 test sites in `tests/unit/models/auto/test_config.py`.
- **`WinMLSession` lazy-init refactor**: drop `_persist_jit`, single `_ensure_session()` funnel, `compile()` returns `Path` (raises on failure — no permissive fallback).
- **`CompileStage` spec integration**: consults `WinMLCompileSpec.supports_ep_context` directly. Branches before calling `WinMLSession.compile()`. Owns the passthrough copy + filename naming. Replaces `_finalize_output`'s three-way filename search with `_relocate_ctx` consuming the exact `compile()` return Path. Drops EP-string fallback.
- **New `deduce_ep_device(ep, device) -> WinMLEPDevice`** in `session/ep_device.py` — pure deduction-phase variant of `resolve_device`, no EP-plugin registration. Returns WinMLEPDevice with placeholder `vendor_id=0, device_id=0`.
- **`resolve_device` if/elif refactor** — sequential if-guards → mutually-exclusive if/elif/else.
- **`commands/compile.py:244` simplification** — drop the `enable_ep_context and not result.output_path` guard. `result.output_path` is always set by `CompileStage` (compiled artifact or copied original). The user-facing message keys off the path suffix.
- Architecture test enforcing 1:1 (ep, device) key parity between `EP_DEVICE_SPECS` and `COMPILE_SPECS`, plus uniqueness within `COMPILE_SPECS`.

**Out of scope:**
- Additional capability flags beyond the three (e.g., `supports_fp16`, `requires_calibration`, `compile_artifact_format`). Add when concrete consumers need them.
- Alternative compile mechanisms (TensorRT engine plan emission, OpenVINO IR generation). Today only `ort.ModelCompiler` + EPContext is the compile path.
- Quant-format auto-selection based on `WinMLCompileSpec.quant_format`. The flag exists but downstream wiring into `WinMLQuantizationConfig.mode` is a follow-up (the field is currently caller-set).
- Static-shape baking driven by `supports_dynamic_shapes`. The flag exists but the shape-resolution step that consumes it is a follow-up.
- Two-format quant support (`frozenset[Literal["qdq", "qlinear"]]`). v1 picks one format per (EP, device); switch to multi-set when CPU/DML callers surface a concrete need.

## 3. Why a Separate Catalog

The natural alternative is to extend `WinMLEPDeviceSpec` with the three new fields directly. We rejected this because:

1. **Layering integrity.** `session/ep_device.py` is the session-layer catalog. Its existing fields (`ep`, `device`, `default_provider_options`) are session/EP-routing concerns — what does ORT need to bind this EP to this device. Adding compile/quant capability flags pulls compile-pipeline knowledge into the session module. The existing codebase enforces module boundaries (`session/`, `compiler/`, `quant/` are distinct concerns).

2. **`WinMLEPDeviceSpec` stays focused.** Tests for `session/ep_device.py` shouldn't need to construct compile-capability values to instantiate fixtures. A separate `WinMLCompileSpec` keeps the session module's surface area unchanged.

3. **Growth axis.** Future compile/quant capability fields (`supports_fp16`, `requires_calibration`, `compile_artifact`, etc.) accumulate on `WinMLCompileSpec`, not on `WinMLEPDeviceSpec`. The session catalog doesn't grow wide as the compile catalog matures.

4. **Drift risk is bounded.** The concern with two catalogs is forgetting to add a row when a new EP lands. One architecture test (`{(s.ep, s.device) for s in EP_DEVICE_SPECS} == {(s.ep, s.device) for s in COMPILE_SPECS}`) catches drift loudly.

The cost is one duplicated bit per row — both catalogs carry `(ep, device)` strings. The architecture test makes the duplication self-correcting.

## 4. Design

### 4.1 The dataclass

```python
# src/winml/modelkit/compiler/spec.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from ..session import WinMLEPDeviceSpec


@dataclass(frozen=True)
class WinMLCompileSpec:
    """Per-(EP, device) compile/quant capability descriptor.

    Sibling to ``WinMLEPDeviceSpec`` (session layer). Where ``WinMLEPDeviceSpec``
    captures what ORT needs to bind an EP to a hardware device, this
    captures what the compile/quant pipeline can do with that target.

    Fields:
        ep: Canonical full EP name (matches WinMLEPDeviceSpec.ep).
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
    # v1.3 default policy: NPU → EPContext + QDQ; everything else → no EPContext + QLinear.
    # Rationale: NPUs require pre-compiled binaries (HTP context, OpenVINO blob, VitisAI cache)
    # and accept QDQ format universally. GPU/CPU EPs run from generic ONNX with QLinear ops.
    # Individual cells can be tightened later (e.g., TensorRT EPContext, DML INT8) without
    # API change — the arch test pins keys, not values.

    # NPU targets (supports_ep_context=True, quant_format="qdq")
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="npu",
        supports_ep_context=True,
        supports_dynamic_shapes=False,  # HTP context bakes shapes
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="npu",
        supports_ep_context=True,
        supports_dynamic_shapes=False,  # reshape_input required
        quant_format="qdq",
    ),
    WinMLCompileSpec(
        ep="VitisAIExecutionProvider", device="npu",
        supports_ep_context=True,
        supports_dynamic_shapes=False,
        quant_format="qdq",
    ),

    # GPU targets (supports_ep_context=False, quant_format="qlinear")
    WinMLCompileSpec(
        ep="DmlExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="MIGraphXExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="TensorrtExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="CUDAExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="NvTensorRtRtxExecutionProvider", device="gpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),

    # CPU targets (supports_ep_context=False, quant_format="qlinear")
    WinMLCompileSpec(
        ep="CPUExecutionProvider", device="cpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="OpenVINOExecutionProvider", device="cpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
    WinMLCompileSpec(
        ep="QNNExecutionProvider", device="cpu",
        supports_ep_context=False,
        supports_dynamic_shapes=True,
        quant_format="qlinear",
    ),
)
```

The 13 entries are 1:1 with `EP_DEVICE_SPECS` rows. The default policy (`NPU → EPContext+QDQ`, everything else → `no-EPContext+QLinear`) reflects the architectural truth that NPUs require pre-compiled binaries while GPU/CPU EPs can JIT from generic ONNX. Cells that could be tightened on verification (e.g., TensorRT EPContext is real, OpenVINO-GPU supports EPContext) are intentionally left at the conservative default — see §9 OQ1.

### 4.3 The lookup

```python
# src/winml/modelkit/compiler/spec.py (continued)

def get_compile_spec(ep_device_spec: WinMLEPDeviceSpec) -> WinMLCompileSpec:
    """Look up compile capabilities for a session-layer WinMLEPDeviceSpec entry.

    Args:
        ep_device_spec: A row from ``EP_DEVICE_SPECS`` (typically obtained
            via ``lookup_device_spec(ep, device)`` from the session module).

    Returns:
        The matching WinMLCompileSpec row from COMPILE_SPECS.

    Raises:
        KeyError: if (ep, device) is uncatalogued. The architecture test
            (§8.1) guarantees 1:1 coverage with ``EP_DEVICE_SPECS``, so this
            should never fire for any (ep, device) pair that ``resolve_device``
            could legitimately produce. A KeyError at runtime indicates
            either catalog drift or a synthetically constructed WinMLEPDeviceSpec
            outside the canonical catalog.
    """
    for spec in COMPILE_SPECS:
        if spec.ep == ep_device_spec.ep and spec.device == ep_device_spec.device:
            return spec
    raise KeyError(
        f"No WinMLCompileSpec entry for ({ep_device_spec.ep!r}, "
        f"{ep_device_spec.device!r}). The architecture test should have "
        f"caught this — check COMPILE_SPECS vs EP_DEVICE_SPECS parity."
    )
```

**Rationale.** v1.0 returned a conservative no-capability fallback. v1.1 raises instead: the arch test guarantees no legitimate caller hits an uncatalogued (ep, device) pair, so silent fallback only masked drift. Raising surfaces drift loudly at the first consumer touch.

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

### 4.5 `deduce_ep_device()` — cross-host config-gen companion

`resolve_device(ep, device)` has a hidden side effect: it calls `register_ep(ep_full)` (line 394) which loads the EP plugin DLL. On hosts where the target EP isn't installed (e.g., generating a QNN config on an x64 dev box), `register_ep` raises before the function can return.

Pre-this-branch, `WinMLCompileConfig.for_provider(str)` accepted a string and avoided this — config-gen worked cross-host. Post-typed-everywhere, every config consumer expects an `WinMLEPDevice`. We need a pure-deduction variant of `resolve_device` that returns an `WinMLEPDevice` without registering the plugin:

```python
# src/winml/modelkit/session/ep_device.py

def deduce_ep_device(
    ep: str | None = None,
    device: str | None = None,
) -> WinMLEPDevice:
    """Deduce an WinMLEPDevice from (ep, device) hints without registering the EP plugin.

    Same input-handling logic as ``resolve_device`` (auto-detect when both
    None, deduce device from EP via the catalog, deduce EP from device via
    fixed priority), but skips the ``register_ep`` + ``OrtEpDevice`` resolution
    step. Returns an ``WinMLEPDevice`` with placeholder ``vendor_id=0, device_id=0``.

    Intended for cross-host config-generation workflows where the target EP
    is not installed on the dev host (e.g., generating a QNN config on x64,
    or a CUDA config on a non-NVIDIA box). Callers that need a real bound
    OrtEpDevice handle (perf/compile/build pipelines) must use ``resolve_device``.

    Args:
        ep: Short or full EP name (e.g., "qnn", "QNNExecutionProvider").
        device: Device category ("npu", "gpu", "cpu", or "auto"/None).

    Returns:
        WinMLEPDevice with the deduced (ep, device) and placeholder hardware IDs.

    Raises:
        ValueError: same conditions as ``resolve_device`` (invalid name, no
            default device for the EP, etc.).
    """
```

The implementation shares the deduction phase with `resolve_device` (§4.8 refactor folds the shared logic into a private `_deduce_ep_device(ep, device) -> tuple[str, str]` helper that both call).

### 4.6 `WinMLSession` lazy-init refactor

**Today** (`session/session.py:259-286`):

```python
self._persist_jit = ep_config.enable_ep_context if ep_config else False
# ... lots of setup ...
if not self._persist_jit:
    self._session = ort.InferenceSession(...)  # eager
else:
    self._session = None  # deferred — compile() will populate
```

**Problem.** Two-path session creation. For non-EPContext EPs (e.g., DML, CPU), `_persist_jit=False`, so the session is eagerly created in `__init__`. Then `compile()`'s line 301 `if self._session is not None: return` short-circuits — any later branching inside `compile()` is dead. The current `try/except` at line 347-349 silently swallows compile failures.

**After.** Drop `_persist_jit`. All session creation funnels through a single private `_ensure_session()`. `compile()` becomes pure mechanism: it always attempts compile; it does not consult the spec or branch on it. The spec-aware caller (`CompileStage`) decides whether to call `compile()` at all.

```python
class WinMLSession:
    _session: ort.InferenceSession | None
    _model_path: Path  # the path currently bound to _session (or to be bound)

    def __init__(
        self,
        onnx_path: Path,
        ep_device: WinMLEPDevice,
        ep_config: EPConfig | None = None,
        *,
        base_session_options: ort.SessionOptions | None = None,
    ):
        # Pure assignment. No ORT calls.
        self._onnx_path = onnx_path
        self._ep_device = ep_device
        self._ep_config = ep_config or EPConfig(provider=short_ep_name(ep_device.ep))
        self._base_session_options = base_session_options
        self._device: str = ep_device.device
        self._embed_context: bool = self._ep_config.embed_context
        self._session = None
        self._model_path = onnx_path  # default; compile() may rebind

    def _ensure_session(self) -> ort.InferenceSession:
        """Lazily create the InferenceSession bound to self._model_path."""
        if self._session is None:
            session_options = _build_session_options(
                self._ep_device, self._ep_config, None, self._base_session_options,
            )
            self._session = ort.InferenceSession(str(self._model_path), sess_options=session_options)
        return self._session

    def compile(self) -> Path:
        """Compile the model to an EPContext artifact. Pure mechanism.

        Output filename: ``{onnx_stem}_{device}.onnx`` next to ``self._onnx_path``.
        v1.3 drops the ``_ctx`` suffix used in pre-v1.3 code — the filename
        signals the targeted device, not the internal format.

        Returns:
            Path to the written ``*_{device}.onnx``, OR the path to a fresh
            cache hit, OR the input path if it was already an EPContext model.
            Always a valid Path on success.

        Raises:
            ort.OrtException / RuntimeError: any compile failure propagates.
            Callers (``CompileStage``) decide whether to invoke this based on
            ``WinMLCompileSpec.supports_ep_context``; no permissive fallback here.
        """
        out_path = self._onnx_path.parent / f"{self._onnx_path.stem}_{self._device}.onnx"

        # Cache hit: existing fresh artifact next to input.
        if out_path.exists() and out_path.stat().st_mtime >= self._onnx_path.stat().st_mtime:
            logger.info("Using cached compiled model: %s", out_path)
            self._model_path = out_path
            return out_path

        # Input is already an EPContext model — no work to do; use as-is.
        if is_compiled_onnx(self._onnx_path):
            logger.info("Model already compiled (EPContext): %s", self._onnx_path)
            self._model_path = self._onnx_path
            return self._onnx_path

        # AOT compile via ort.ModelCompiler. Raises on failure (no swallow).
        so = _build_session_options(self._ep_device, self._ep_config, None, self._base_session_options)
        model_compiler = ort.ModelCompiler(
            so, str(self._onnx_path), embed_compiled_data_into_model=self._embed_context,
        )
        compile_log = self._onnx_path.parent / "compile.log"
        with _suppress_native_output(compile_log):
            model_compiler.compile_to_file(str(out_path))

        self._model_path = out_path
        return out_path

    def run(self, *args, **kwargs):
        return self._ensure_session().run(*args, **kwargs)
```

**Subclass contract — `WinMLQairtSession.compile() -> Path`.** The QAIRT subclass at `session/qairt/qairt_session.py:75` currently returns `None`. Under v1.3, it MUST return the `Path` to the EPContext model it produces (the `.onnx` wrapper around the compiled `.bin`). Today the subclass writes to `self._ctx_path`; the refactor adds `return self._ctx_path` at the end and changes the signature to `-> Path`. The subclass also retains the same lazy-init contract — no eager session creation in `__init__`, all session access through `_ensure_session()`.

**Known limitation: input directory must be writable.** `WinMLSession.compile()` writes the compiled artifact next to `self._onnx_path` (the input ONNX). If the input lives in a read-only directory (system path, mounted read-only volume, network share without write access), `ort.ModelCompiler.compile_to_file` raises `PermissionError` / `OrtException`, which propagates per the "raises on failure" contract. v1.3 does NOT add a tempdir fallback — the input directory is expected to be writable. Document this in the user-facing `winml compile --help` text. If a fallback becomes necessary later, add it as an opt-in `--scratch-dir <path>` CLI flag, not as silent magic.

**What this fixes:**
- Dead-code bug in `compile()` for non-EPContext EPs (the early-out short-circuit is gone).
- Two-path session creation collapses to one — easier to reason about.
- `compile()` is pure mechanism — no spec lookup, no policy. Callers decide whether to call.
- Failures in EPContext compilation are no longer silently swallowed — they raise.

**Pure callsite impact:** `WinMLSession.run()`, `WinMLSession.get_inputs()`, and any other delegation to the underlying session now route through `_ensure_session()` instead of `self._session.<method>`. There's one funnel; if there's no session yet (e.g., compile() was never called), it's created on first use.

### 4.7 `CompileStage` spec integration

`compiler/stages/compile.py:CompileStage` becomes the single spec-aware layer. It consults `WinMLCompileSpec.supports_ep_context` from the catalog and branches before calling `WinMLSession.compile()`. `WinMLSession` has no knowledge of the catalog; the stage owns the policy decision.

**Retained responsibilities** (each verified absent elsewhere):
- **qairt-vs-ort selection** (line 30-33) — chooses backend based on `ep_config.compiler`.
- **Validation** — post-action model validation via `_validate_model(session)`.
- **Output relocation** — moves the compiled artifact (or copies the input) into the user's `--output-dir`.
- **`.bin` sidecar handling** — when `embed_context=False`, ORT writes a `.bin` next to `.onnx`. Move both.
- **`ep_cache_context` patch** — rewrites EPContext metadata to a relative path for portability.

**New / restructured logic:**

```python
def process(self, context: CompileContext) -> CompileContext:
    ep_device = self._resolve_ep_device(context)
    compile_spec = get_compile_spec(lookup_device_spec(ep_device.ep, ep_device.device))

    output_dir = self._get_output_dir(context)
    model_path = self._ensure_model_file(context)
    compile_cfg = WinMLCompileConfig.from_dict(context.config)
    ep_config = compile_cfg.ep_config
    session_cls = COMPILER_SESSION_MAPPING[ep_config.compiler]

    winml_session = session_cls(onnx_path=model_path, ep_device=ep_device, ep_config=ep_config)

    if compile_spec.supports_ep_context:
        ctx_path = winml_session.compile()                             # raises on failure
        if context.validate:
            session = winml_session._ensure_session()                  # loads from ctx_path
            self._validate_model(session, context)
            self._collect_model_info(session, context)
        output_path = self._relocate_ctx(ctx_path, output_dir, ep_device.device)
    else:
        logger.info(
            "%s on %s does not support EPContext compilation; "
            "original model copied to output dir.",
            ep_device.ep, ep_device.device,
        )
        if context.validate:
            session = winml_session._ensure_session()                  # loads original input
            self._validate_model(session, context)
            self._collect_model_info(session, context)
        output_path = output_dir / f"{model_path.stem}_{ep_device.device}.onnx"
        shutil.copy2(model_path, output_path)

    context.output_path = output_path
    return context
```

Validation runs **before** relocation in both branches. `_ensure_session()` reads from `self._model_path` (= `ctx_path` after compile, or = original input for passthrough). `_relocate_ctx` then copies the validated artifact to `output_dir`.

**Filename convention:** `{stem}_{device}.onnx` for both branches. No `_ctx` suffix anywhere (intermediate or final). The compiled and passthrough cases produce identically-named output files; whether a given file happens to be an EPContext model is recorded in ONNX metadata, not in the filename. This is a v1.3 simplification from the previous `{stem}_{device}_ctx.onnx` (EPContext) vs `{stem}_{device}.onnx` (passthrough) split.

**`_relocate_ctx`** replaces `_finalize_output`. Let `dest_stem = f"{model_path.stem}_{device}"` where `model_path` is `CompileStage`'s input ONNX (note: this is the input to the stage, not the path returned by `compile()` — they differ in the already-EPContext branch). Let `src_path` be the path returned by `compile()`. Then `_relocate_ctx`:

1. **Copies** `src_path` → `output_dir / f"{dest_stem}.onnx"`.
2. If a sidecar `src_path.with_suffix(".onnx.bin")` (i.e., `{src_path.stem}.onnx.bin` next to `src_path`) exists, copies it to `output_dir / f"{dest_stem}.onnx.bin"`. The sidecar always carries the destination stem, never the source stem.
3. Patches the destination ONNX's `ep_cache_context` attribute to the relative path `f"{dest_stem}.onnx.bin"` — i.e., the new sidecar filename, NOT the source sidecar filename. Critical when the already-EPContext input path differs from the canonical `{dest_stem}` (e.g., input `model.onnx` already-compiled, device `npu` → src has stem `model`, dest has stem `model_npu`).
4. Returns the destination Path `output_dir / f"{dest_stem}.onnx"`.

**Sequencing: validate before relocation.** `process()` runs `_ensure_session()` (for validation) BEFORE `_relocate_ctx`. This means `self._model_path` (the path the InferenceSession will be loaded from) is the freshly compiled `ctx_path` next to the input ONNX, and the relocation step is purely a file-copy operation that doesn't disturb session state. Alternative considered: move-then-rebind `_model_path = output_path`, but copy-then-validate is simpler and matches today's `shutil.copy2` semantics in `_finalize_output`.

No three-way filename search. No EP-string fallback. The stage is a small dispatcher; the heavy lifting is in `_relocate_ctx` and `WinMLSession.compile()`.

### 4.8 `resolve_device` if/elif refactor

**Today** (`session/ep_device.py:resolve_device`): four sequential `if` guards on (ep, device) state combinations. Sequential ifs allow accidental fall-through if a branch's guard isn't tight; the if/elif refactor makes mutual exclusivity load-bearing:

```python
def resolve_device(ep: str | None, device: str | None = None) -> WinMLEPDevice:
    # Normalize "auto" sentinel to None.
    if device is not None and device.lower() == "auto":
        device = None

    # Branch on the four possible (ep, device) state combinations.
    if ep is None and device is None:
        # No hints: pick the system's best device on this host.
        # auto_detect_device returns str (device only); ep deduction falls
        # through to the next branch.
        device = auto_detect_device()

    if ep is not None and device is None:
        # EP given, device missing — infer from catalog.
        ep_full = expand_ep_name(ep)
        deduced = default_device_for_ep(ep_full)
        if deduced is None:
            raise ValueError(f"No default device for {ep_full}; specify --device.")
        device = deduced
    elif ep is None and device is not None:
        # Device given, EP missing — pick default EP for this device.
        ep_full = default_ep_for_device(device.lower())
    else:  # both given
        ep_full = expand_ep_name(ep)

    # Resolution phase: register EP, find matching OrtEpDevice, return.
    WinMLEPRegistry.get_instance().register_ep(ep_full)
    return _resolve_to_ep_device(ep_full, device)
```

**Companion split.** The deduction phase (lines 1-15 above) and the resolution phase (lines 16-18) are separated; `deduce_ep_device` (§4.5) calls only the deduction phase + returns an `WinMLEPDevice` with placeholder hardware IDs.

## 5. Catalog Contents

The 13 (ep, device) target rows mirror `EP_DEVICE_SPECS` exactly. v1.3 applies a uniform default policy by device category:

| EP / device | `supports_ep_context` | `supports_dynamic_shapes` | `quant_format` | Notes |
|---|---|---|---|---|
| QNN / npu | ✅ True | False (HTP bakes shapes) | qdq | verified CI |
| OpenVINO / npu | ✅ True | False (reshape_input req) | qdq | verified this branch |
| VitisAI / npu | ✅ True | False | qdq | no host; policy default |
| DML / gpu | False | True | qlinear | DML has no EPContext today |
| OpenVINO / gpu | False (policy) | True | qlinear | could be tightened to True — verified locally; left conservative |
| QNN / gpu | False | True | qlinear | Adreno path, no EPContext |
| MIGraphX / gpu | False | True | qlinear | no host; policy default |
| Tensorrt / gpu | False (policy) | True | qlinear | could be tightened to True (ORT 1.24 supports EPContext for TRT) |
| CUDA / gpu | False | True | qlinear | CUDA never compiles |
| NvTensorRtRtx / gpu | False (policy) | True | qlinear | could be tightened to True; left conservative |
| CPU / cpu | False | True | qlinear | CPU never compiles |
| OpenVINO / cpu | False | True | qlinear | OV-CPU JITs from ONNX |
| QNN / cpu | False | True | qlinear | QNN-CPU reference backend |

**Default policy:** NPU → `supports_ep_context=True, quant_format="qdq"`. GPU/CPU → `supports_ep_context=False, quant_format="qlinear"`. Three GPU cells (OpenVINO, Tensorrt, NvTensorRtRtx) could be tightened to `True` based on documented EPContext support; v1.3 leaves them False as a conservative starting point. The arch test pins keys, not values — tightening a cell is a one-line change in a follow-up PR with no API impact.

## 6. Consumer Changes

### 6.1 `WinMLCompileConfig.for_ep_device` — trivial, no spec lookup

Today (post-prior-refactor in `compiler/configs.py`):
```python
@classmethod
def for_ep_device(cls, ep_device: WinMLEPDevice) -> WinMLCompileConfig:
    from ..session import short_ep_name
    provider = short_ep_name(ep_device.ep)
    base = cls.for_provider(provider) or cls(ep_config=EPConfig(provider=provider))
    base.ep_device = ep_device
    return base
```

After:
```python
@classmethod
def for_ep_device(cls, ep_device: WinMLEPDevice) -> WinMLCompileConfig:
    """Build a compile config from a resolved WinMLEPDevice. No spec lookup;
    CompileStage consults WinMLCompileSpec at run time."""
    from ..session import short_ep_name
    return cls(
        ep_config=EPConfig(provider=short_ep_name(ep_device.ep)),
        ep_device=ep_device,
    )
```

The spec is **not** consulted at config-build time. `CompileStage` reads it directly from the catalog when it needs to decide whether to attempt compile (§4.7).

### 6.2 `WinMLSession.compile()` — `Path` return, pure mechanism

See §4.6 for the full lazy-init refactor. The signature is `compile() -> Path` — always a valid path on success, raises on failure. The session does not consult `WinMLCompileSpec`; the spec-aware caller (`CompileStage`) decides whether to invoke `compile()` at all.

- AOT compile succeeds → returns the written `{stem}_{device}.onnx`.
- Cache hit on a fresh existing `{stem}_{device}.onnx` → returns the cached path.
- Input is already an EPContext (`is_compiled_onnx`) → returns `self._onnx_path` unchanged.
- Compile raises → propagates (no swallow).

### 6.3 Deletion: 8 per-EP factories + `for_provider` + `EPConfig.enable_ep_context`

The `_EP_CONTEXT_DEFAULTS` table introduced earlier in this branch and the `enable_ep_context` field on `EPConfig` are both superseded — the catalog is the single source of truth. Delete:

- `WinMLCompileConfig.for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_nv_tensorrt_rtx`, `for_openvino`, `for_vitisai`, `for_migraphx`
- `WinMLCompileConfig.for_provider(provider: str | None)`
- The local `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]]` (already committed in `configs.py:32`)
- **`EPConfig.enable_ep_context: bool` field** — derived data, now read from `WinMLCompileSpec.supports_ep_context` at `CompileStage`.

After deletion, `for_ep_device` is the sole factory and `EPConfig` shrinks to `provider`, `provider_options`, `embed_context`, `compiler`, `qnn_sdk_root`.

**`EPConfig.to_dict` / `from_dict`** also drop `enable_ep_context`. Existing on-disk configs containing the field are read with the unknown field silently ignored (the `from_dict` already ignores legacy fields).

### 6.4 Migration of caller sites

`WinMLCompileConfig.for_provider(short_str)` callers split into two groups by host-installation requirement:

**Group A — target-host (build pipeline). Uses `resolve_device`.** Three sites in `config/build.py` run on the target host (the build pipeline assumes the target EP is installed):

- `config/build.py:307` — `resolve_quant_compile_config`
- `config/build.py:604` — `generate_hf_build_config` policy path
- `config/build.py:616` — `generate_hf_build_config` hw-detected path

```python
# After
from ..session import resolve_device
ep_device = resolve_device(ep=policy.compile_provider, device=policy.device)
compile_config = WinMLCompileConfig.for_ep_device(ep_device)
```

**Group B — cross-host (config-generation CLI). Uses `deduce_ep_device`.** `commands/config.py:570-578` runs on a dev host that may not have the target EP installed (e.g., generating a QNN config on x64). Using `resolve_device` here would call `register_ep` and fail. Use the deduction-only companion (§4.5):

```python
# commands/config.py — after
from ..session import deduce_ep_device
ep_device = deduce_ep_device(ep=args.ep, device=args.device)
compile_config = WinMLCompileConfig.for_ep_device(ep_device)
```

The resulting `WinMLEPDevice` has placeholder `vendor_id=0, device_id=0` — fine for config serialization (the target host re-resolves at build time).

### 6.5 `winml compile` CLI — simplified result handling

The CLI is unchanged in shape: it produces a user-visible artifact in `--output-dir`. `CompileStage` (§4.7) is the only compile callsite — `commands/compile.py` consumes the pipeline result and reports it. With Option B integration:

- **`supports_ep_context=True`**: `CompileStage` writes `{stem}_{device}.onnx` (+ optional `.onnx.bin` sidecar) to `output_dir` via `_relocate_ctx`.
- **`supports_ep_context=False`**: `CompileStage` copies the input ONNX to `{output_dir}/{stem}_{device}.onnx`. INFO log: `"<EP> on <device> does not support EPContext compilation; original model copied to <output_path>"`. Exit 0.

**Unified filename `{stem}_{device}.onnx`** for both branches. The filename signals the targeted device; it does not claim a format. EPContext-ness is recorded in ONNX metadata (the `ep_cache_context` node-attribute) and can be checked at load time via `is_compiled_onnx(path)`.

**`commands/compile.py:244` simplification.** Today the line reads:
```python
if config.ep_config.enable_ep_context and not result.output_path:
    raise click.ClickException("No output file produced. ...")
```

After:
```python
if not result.output_path:
    raise click.ClickException(
        "Compilation succeeded but no output file was written. "
        "This is a pipeline bug — please file an issue."
    )
```

`CompileStage` always populates `result.output_path` (compiled artifact or passthrough copy). A missing `output_path` now indicates a genuine pipeline bug, not a UX-permissive EP. The `enable_ep_context` check disappears with the field.

The user-facing message branches on which path the pipeline took, not on the filename (filenames are uniform). `compile_onnx`'s `CompileResult` exposes a `compile_invoked: bool` field (set by `CompileStage`) — `True` means the EPContext branch ran, `False` means passthrough. `commands/compile.py` keys the success message off this flag:
- `compile_invoked=True`: "Compiled model written to {output_path}"
- `compile_invoked=False`: "Original model copied to {output_path} (EP does not support EPContext)"

This matches the user-facing "permissive" UX: every `winml compile <model> --ep <X> --device <Y>` invocation succeeds and produces a model file at the output path. Users targeting non-EPContext EPs get the original model back with explicit messaging — they know nothing was compiled.

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

### 6.8 Other callers of `WinMLSession.compile()` and `EPConfig.enable_ep_context`

The agent review surfaced four caller sites the v1.1 migration list missed. All are in scope for v1.2.

**`commands/perf.py:425` and `commands/perf.py:1109`** — direct callers of `WinMLSession.compile()`. Both use the call for its side effect of populating `self._session` (the original comment says "Compile session early so session.device is resolved for display"). After v1.2:
- For EPContext EPs (QNN NPU, OpenVINO, etc.): `compile()` still produces the EPContext artifact, and perf benchmarks the compiled session — same as today.
- For non-EPContext EPs (CPU, DML, etc.): today the eager-session path makes `compile()` a no-op; in v1.2 with the lazy refactor, calling `compile()` would attempt AOT compile and **raise** (or produce unwanted artifacts).

**Migration**: gate the perf compile call on `supports_ep_context` exactly as `CompileStage` does:
```python
spec = get_compile_spec(lookup_device_spec(ep_device.ep, ep_device.device))
if spec.supports_ep_context:
    session.compile()
# either way, _ensure_session() warms up the InferenceSession on first .run() call.
```

Both `perf.py:425` and `perf.py:1109` get the same wrapper. The pattern is repetitive enough to consider extracting `winml_session.compile_if_supported()` — a thin spec-aware helper on `WinMLSession`. **Open question (OQ6)**: extract helper, or repeat the 4-line guard at each callsite? See §9.

**`compiler/cli.py:110,135,172`** — older CLI entrypoint (`python -m winml.modelkit.compiler`) exposes a `--ep-context/--no-ep-context` flag that wires directly into `EPConfig(enable_ep_context=...)`. With the field deletion, this CLI breaks at import time.

**Migration**: three coordinated deletions in `compiler/cli.py`:
1. Delete the `@click.option("--ep-context/--no-ep-context", "enable_ep_context", ...)` decorator (line 110).
2. Delete the `enable_ep_context: bool` parameter from the function signature (line 135).
3. Delete the `enable_ep_context=enable_ep_context` kwarg from the `EPConfig(...)` constructor call (line 172).

Skipping any of the three causes either a runtime `TypeError` (Click passes the param but function signature doesn't accept it) or a stale parameter that does nothing. The catalog is the source of truth — no user override at the CLI for now. If user override resurfaces as a need, add it later as `--force-compile`/`--no-force-compile` at the spec-aware layer (`CompileStage`), not as a per-EPConfig field.

**`compiler/context.py:91-93`** — `CompileContext.enable_ep_context` property reads `self.config.get("enable_ep_context", True)`. After the refactor, no stage writes `"enable_ep_context"` into the context config dict. The property becomes dead code and would silently default to `True` for any caller that reads it.

**Migration**: delete the property at `context.py:91-93`. Update `tests/unit/compiler/test_compiler_stages.py:199` which asserts `context.enable_ep_context is True`.

**`compiler/configs.py:307` (`from_dict` default)** — today, `from_dict` defaults `enable_ep_context=True` when the field is missing from the dict. After deletion, any persisted config that omits the field simply silently loses the (now-removed) bit. The behavior change is: pre-v1.2 callers that constructed `EPConfig` from a dict without `enable_ep_context` got `True` (compile enabled); post-v1.2, the catalog decides per (ep, device). For QNN/OV this is the same outcome; for CPU/DML it changes "attempt compile (which today fails silently)" → "do not attempt compile (clean passthrough)". This is an improvement, not a regression — but note it explicitly in release notes.

## 7. Migration

### 7.1 Files added

- `src/winml/modelkit/compiler/spec.py` — new module (~150 LOC including catalog rows).
- `tests/unit/compiler/test_spec.py` — new file. Architecture-test + per-row sanity + raise-on-miss.
- `tests/unit/session/test_deduce_ep_device.py` — new file. Deduction-only contract; placeholder-IDs invariant; parity-with-resolve assertions.

### 7.2 Files modified

**Compiler module:**
- `src/winml/modelkit/compiler/configs.py` — delete 8 per-EP factories + `for_provider` (~140 LOC removed). Trim `for_ep_device` body to trivial constructor. **Drop `EPConfig.enable_ep_context` field** + remove from `to_dict`/`from_dict`. Drop `_EP_CONTEXT_DEFAULTS` constant. Update class docstring examples.
- `src/winml/modelkit/compiler/__init__.py` — re-export `WinMLCompileSpec`, `COMPILE_SPECS`, `get_compile_spec` (line 22 area).
- `src/winml/modelkit/compiler/compiler.py` — update docstring examples (lines 49, 186, 189, 192) to drop `for_provider`/`for_qnn` references.
- `src/winml/modelkit/compiler/stages/compile.py` — `CompileStage` spec integration (§4.7): consult `WinMLCompileSpec.supports_ep_context`; branch before `WinMLSession.compile()`; passthrough copies input as `{stem}_{device}.onnx`. Replace `_finalize_output` with `_relocate_ctx(ctx_path, output_dir, device)`. Drop EP-string fallback at line 73-75. Validation moved before relocation. ~80 LOC removed, ~30 LOC added.
- `src/winml/modelkit/compiler/context.py:91-93` — **delete `CompileContext.enable_ep_context` property**. Dead after EPConfig field removal.
- `src/winml/modelkit/compiler/cli.py:110,135,172` — **delete `--ep-context/--no-ep-context` flag and the `enable_ep_context` parameter**. Catalog is the source of truth.

**Session module:**
- `src/winml/modelkit/session/session.py` — `WinMLSession` lazy-init refactor (§4.6): drop `_persist_jit`, introduce `_ensure_session()` private funnel, `__init__` becomes pure assignment, `compile()` returns `Path` (raises on failure — no silent fallback, no spec lookup). Update `run()`, `get_inputs()`, etc. to route through `_ensure_session()`. ~60 LOC net rewrite.
- `src/winml/modelkit/session/qairt/qairt_session.py:75` — **update `WinMLQairtSession.compile()` signature to `-> Path`** (currently `-> None`). Return `self._ctx_path` at end. Same lazy-init contract as parent (no eager session in `__init__`).
- `src/winml/modelkit/session/ep_device.py` — `resolve_device` if/elif refactor (§4.8). Add `deduce_ep_device(ep, device)` and shared private `_deduce_ep_device(ep, device) -> tuple[str, str]` helper. ~30 LOC added.
- `src/winml/modelkit/session/__init__.py` — export `deduce_ep_device`.

**Config / CLI:**
- `src/winml/modelkit/config/build.py:307,604,616` — migrate 3 sites to `resolve_device` + `for_ep_device` (target-host workflow). Drop any `enable_ep_context=...` arg passthrough.
- `src/winml/modelkit/commands/config.py:570-578` — migrate to `deduce_ep_device` + `for_ep_device` (cross-host config-gen workflow).
- `src/winml/modelkit/commands/compile.py:244` — simplify the post-pipeline check (no `enable_ep_context` reference). `result.output_path` is always set; key user message off `result.compile_invoked` (new `CompileResult` field).
- `src/winml/modelkit/compiler/result.py:15` — add `compile_invoked: bool` to `CompileResult` dataclass; `CompileStage` sets it true on the EPContext branch, false on passthrough.
- `src/winml/modelkit/commands/perf.py:425,1109` — **gate `session.compile()` on `WinMLCompileSpec.supports_ep_context`** (avoid attempting AOT compile for non-EPContext EPs). See §6.8 / OQ6 for helper-extraction question.

### 7.3 Tests modified

- `tests/unit/compiler/test_compiler_configs.py` — delete the test class(es) for per-EP factories and `for_provider` (~40 LOC). Keep `for_ep_device` tests; expand to cover the new spec-driven path. Update any `enable_ep_context` field assertions (field is deleted).
- `tests/unit/models/auto/test_config.py:169,172` — replace `for_qnn()`/`for_cpu()` with `for_ep_device(WinMLEPDevice(...))`.
- `tests/unit/session/test_winml_session.py` — add lazy-compile tests: `_ensure_session` not called in `__init__`; `compile()` returns `Path` on EPContext EP; `compile()` raises on ORT failure (no silent fallback). Note: `compile()` is never called for non-EPContext EPs — `CompileStage` gates the call.
- `tests/unit/config/test_build.py:1857-1858` — update comments referencing `for_provider`.
- `tests/unit/compiler/test_compile_stage.py` (or equivalent) — replace `_finalize_output` tests with `_relocate_ctx` tests; add spec-aware branching tests (compile branch vs passthrough branch).
- `tests/unit/compiler/test_compiler_stages.py:199` — **delete** the `assert context.enable_ep_context is True` assertion (property deleted).

### 7.4 Total blast radius

- ~250 LOC added (new spec.py + deduce_ep_device + new tests)
- ~280 LOC deleted (factories + their tests + CompileStage filename search + `_persist_jit` paths)
- ~11 files modified
- Net delta: roughly -30 LOC, much cleaner public surface, two structural bugs fixed (silent compile fallback, dead-code permissive path)

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

### 8.3 Lookup raises on miss

```python
def test_get_compile_spec_raises_on_uncatalogued_pair():
    """Unknown (ep, device) pair raises KeyError (arch test should prevent this in practice)."""
    fake = WinMLEPDeviceSpec(ep="FakeEP", device="npu")
    with pytest.raises(KeyError, match="No WinMLCompileSpec entry"):
        get_compile_spec(fake)
```

### 8.4 Lazy-session + Path-return tests

```python
def test_winml_session_init_does_not_create_inference_session():
    """__init__ is pure assignment; first _ensure_session() call creates the ORT session."""
    session = WinMLSession(model_path, ep_device=...)
    assert session._session is None
    _ = session.get_inputs()  # routes through _ensure_session
    assert session._session is not None

def test_winml_session_compile_returns_path_on_epcontext_ep(tmp_path):
    """compile() returns the Path to {stem}_{device}.onnx (or input path for an
    already-compiled input)."""
    session = WinMLSession(model_path, ep_device=WinMLEPDevice(ep="QNNExecutionProvider", device="npu", ...))
    result = session.compile()
    assert result.suffix == ".onnx"
    # Fresh-compile or cache-hit: stem == "{onnx_stem}_{device}".
    # Already-compiled input: result == self._onnx_path (unchanged).
    assert result.stem == f"{model_path.stem}_{session._device}" or result == session._onnx_path

def test_winml_session_compile_propagates_failure():
    """compile() does not silently fallback — failures raise."""
    # Mock ort.ModelCompiler.compile_to_file to raise. Assert the exception escapes.
    with pytest.raises(RuntimeError):
        session.compile()

def test_winml_session_compile_returns_cached_path_on_fresh_cache(tmp_path):
    """If a fresh {stem}_{device}.onnx exists next to input, compile() returns it without invoking ORT."""
    # Touch a fresh cached ctx file. Assert ort.ModelCompiler is not invoked.
```

### 8.5 CompileStage spec-aware branching

```python
def test_compile_stage_skips_compile_for_non_epcontext_ep(tmp_path, caplog):
    """For supports_ep_context=False EPs, CompileStage does not invoke session.compile()."""
    # CPU/cpu has supports_ep_context=False.
    # Assert WinMLSession.compile mock is NOT called.
    # Assert output_path = output_dir / "{stem}_cpu.onnx" exists and matches input bytes.
    # Assert INFO log message contains "does not support EPContext".

def test_compile_stage_invokes_compile_for_epcontext_ep(tmp_path):
    """For supports_ep_context=True EPs, CompileStage invokes session.compile() and relocates."""
    # QNN/npu has supports_ep_context=True.
    # Assert output_path = output_dir / "{stem}_npu.onnx" exists.
    # Assert ep_cache_context patched to relative path.

def test_compile_stage_relocates_bin_sidecar_when_embed_context_false(tmp_path):
    """When embed_context=False, _relocate_ctx moves the .bin alongside the .onnx."""
```

### 8.6 `deduce_ep_device` contract

```python
def test_deduce_ep_device_does_not_register_ep(monkeypatch):
    """deduce_ep_device never invokes the EP registry."""
    # EP registration happens via WinMLEPRegistry.get_instance().register_ep(...)
    # — patch the registry class, not a free function (which doesn't exist).
    called = []

    class FakeRegistry:
        @classmethod
        def get_instance(cls):
            return cls()

        def register_ep(self, ep_full):
            called.append(ep_full)

    monkeypatch.setattr("winml.modelkit.session.ep_device.WinMLEPRegistry", FakeRegistry)
    result = deduce_ep_device(ep="qnn", device="npu")
    assert called == []
    assert result.ep == "QNNExecutionProvider"
    assert result.device == "npu"
    assert result.vendor_id == 0
    assert result.device_id == 0

def test_deduce_and_resolve_agree_on_deduction(monkeypatch):
    """deduce_ep_device and resolve_device produce identical (ep, device) for the same inputs."""
    # Mock register_ep + OrtEpDevice lookup so resolve_device can complete in test env.
    # Iterate over a fixed input grid: (None, None), (None, "npu"), ("qnn", None), ("qnn", "npu").
    # Assert deduced.ep == resolved.ep and deduced.device == resolved.device.
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

### OQ3 — `enable_ep_context` user-override path **[RESOLVED v1.2: field deleted]**

v1.0/v1.1 carried `enable_ep_context` on `EPConfig` as a user-overridable field. Resolved in v1.2 by deleting the field entirely: the catalog is the single source of truth, and `CompileStage` consults it directly. If a future use case needs per-call override of `supports_ep_context`, it should arrive as an explicit `force_compile: bool | None` parameter at `CompileStage`/CLI, not as a derived copy on `EPConfig`.

### OQ4 — Should `get_compile_spec` raise on uncatalogued targets? **[RESOLVED v1.1: raises]**

v1.0 returned a conservative default. Resolved in v1.1 to raise `KeyError` — the arch test guarantees no legitimate caller hits a miss, and silent fallback only masked drift. Drift now surfaces at the first consumer touch.

### OQ6 — Extract `WinMLSession.compile_if_supported()` helper?

`commands/perf.py:425` and `commands/perf.py:1109` both need the same 4-line guard around `session.compile()`:
```python
spec = get_compile_spec(lookup_device_spec(ep_device.ep, ep_device.device))
if spec.supports_ep_context:
    session.compile()
```

Two callsites is the minimum threshold for considering extraction. Options:
- **(a) Repeat the 4 lines** — explicit and grep-able. Cost: duplication. Benefit: each site clearly shows what it's doing.
- **(b) Add `WinMLSession.compile_if_supported() -> Path | None`** as a thin spec-aware helper. Cost: introduces a method that *does* know about the catalog, partially undoing the "session is pure mechanism" goal of v1.2. Benefit: less duplication.

v1.2 chooses **(a)** — the duplication is two callsites, and keeping `WinMLSession` spec-unaware preserves the architectural cleanliness from §4.6. If more callsites appear later, revisit.

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
| **WinMLEPDevice** | Project-defined plain-data descriptor (frozen dataclass) at `session/ep_device.py`. The resolved (EP, device) descriptor. |
| **WinMLEPDeviceSpec** | The session-layer catalog row — static template `(ep, device, default_provider_options)`. One per (EP, device) target. |
| **EP_DEVICE_SPECS** | Tuple of 13 WinMLEPDeviceSpec rows in `session/ep_device.py`. Order encodes preference. |
| **WinMLCompileSpec** | This spec's contribution — the compile-layer catalog row. `(ep, device, supports_ep_context, supports_dynamic_shapes, quant_format)`. |
| **COMPILE_SPECS** | Tuple of 13 WinMLCompileSpec rows in `compiler/spec.py`. 1:1 with EP_DEVICE_SPECS keys. |
| **EPContext** | ORT's container format wrapping a pre-compiled EP-specific binary blob. Used by `ort.ModelCompiler.compile_to_file()`. |
| **QDQ** | Quantize-Dequantize quantization format (Q/DQ node pairs around floats). |
| **QLinear** | QOperator/QLinearOps quantization format (single quantized ops). |

### 10.2 References

- [`docs/design/session/3_design_ep.md`](../session/3_design_ep.md) — Stage 1+2 EP registration and device handle selection (the WinMLEPDevice / WinMLEPDeviceSpec upstream).
- [`docs/design/session/2026-05-13-ep-device-spec-design.md`](../session/2026-05-13-ep-device-spec-design.md) — WinMLEPDeviceSpec catalog design.
- `src/winml/modelkit/session/ep_device.py` — WinMLEPDeviceSpec definition, EP_DEVICE_SPECS catalog.
- `src/winml/modelkit/session/session.py:288-349` — current `WinMLSession.compile()` body.
- `src/winml/modelkit/compiler/configs.py` — current `WinMLCompileConfig` with the 8 per-EP factories to be deleted.
- ONNX Runtime ModelCompiler docs (EPContext format).

### 10.3 Document History

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-19 | Initial draft. Captures v1 capability flags (`supports_ep_context`, `supports_dynamic_shapes`, `quant_format`), `WinMLCompileSpec` catalog under `compiler/spec.py`, deletion of 8 per-EP factories + `for_provider(str)`, permissive-compile UX, architecture test for key parity. |
| 1.1 | 2026-05-19 | Amended after agent review. Integrated four supporting refactors required for the spec to actually fire: (a) `get_compile_spec` raises `KeyError` on miss (was: conservative default — resolves OQ4); (b) `deduce_ep_device(ep, device)` cross-host config-gen companion to `resolve_device`; (c) `WinMLSession` lazy-init refactor — drop `_persist_jit`, `_ensure_session()` funnel, `compile()` returns `Path \| None` (fixes dead-code bug for non-EPContext EPs and silent-fallback bug for compile failures); (d) `CompileStage` slim — collapse `_finalize_output` three-way filename search, drop EP-string fallback. Expanded §7 migration list to 11 files including session refactor surface, expanded §8 with lazy-compile and deduce_ep_device tests. Downgraded OV-CPU `supports_ep_context` to ❓ pending verification. |
| 1.2 | 2026-05-19 | **Option B integration.** `CompileStage` becomes the single spec-aware layer; spec consultation moved out of `WinMLSession.compile()` and out of `WinMLCompileConfig.for_ep_device`. **Deleted `EPConfig.enable_ep_context` field** — derived data, superseded by `WinMLCompileSpec.supports_ep_context` (resolves OQ3). `WinMLSession.compile()` now returns `Path` (not `Path \| None`) — pure mechanism, raises on failure; the spec-aware caller decides whether to invoke. `_finalize_output` replaced by `_relocate_ctx(ctx_path, output_dir, device)`. Filename convention: `{stem}_{device}_ctx.onnx` (compiled) vs `{stem}_{device}.onnx` (passthrough). `commands/compile.py:244` post-pipeline guard simplified — no `enable_ep_context` reference. Added v1.2.1 corrections after second agent review: (1) `__init__` snippet missing `_base_session_options`, `_device`, `_embed_context` fields; (2) `resolve_device` pseudocode bugs (`auto_detect_device` returns str, not tuple; function is `default_ep_for_device`); (3) added §6.8 covering 4 missed migration sites — `perf.py:425,1109`, `compiler/cli.py:110-172` (--ep-context flag), `compiler/context.py:91-93` (dead property), and `WinMLQairtSession.compile()` signature (must return `Path`). Validation moved before relocation in `CompileStage`. Added OQ6 (helper-extraction question for perf.py duplication). |
| 1.3 | 2026-05-19 | **User directives applied:** (a) `--ep-context/--no-ep-context` flag deletion from `compiler/cli.py` confirmed (was option in §6.8, now decided). (b) **Drop `_ctx` suffix from all output filenames** — unified `{stem}_{device}.onnx` for both compiled and passthrough cases. Internal cache also uses `{stem}_{device}.onnx` next to input. CLI success message keys off new `CompileResult.compile_invoked: bool` field, not off the path stem. (c) **NPU-uniform policy in `COMPILE_SPECS`**: all NPU rows (`QNNExecutionProvider/npu`, `OpenVINOExecutionProvider/npu`, `VitisAIExecutionProvider/npu`) → `supports_ep_context=True, quant_format="qdq"`. All GPU/CPU rows → `supports_ep_context=False, quant_format="qlinear"`. Three GPU cells (OpenVINO, Tensorrt, NvTensorRtRtx) that could be tightened to `True` are intentionally left conservative — tightening is a one-line follow-up. (d) **Single PR, this branch** — no phased rollout. |

### 10.4 Code Reference Map

| Concern | File | Symbol |
|---|---|---|
| New: capability dataclass | `src/winml/modelkit/compiler/spec.py` | `WinMLCompileSpec` |
| New: catalog | `src/winml/modelkit/compiler/spec.py` | `COMPILE_SPECS` |
| New: lookup (raises on miss) | `src/winml/modelkit/compiler/spec.py` | `get_compile_spec(ep_device_spec)` |
| New: cross-host deduction | `src/winml/modelkit/session/ep_device.py` | `deduce_ep_device(ep, device)` |
| New: shared deduction helper | `src/winml/modelkit/session/ep_device.py` | `_deduce_ep_device(ep, device)` |
| New: re-exports (compiler) | `src/winml/modelkit/compiler/__init__.py` | `WinMLCompileSpec`, `COMPILE_SPECS`, `get_compile_spec` |
| New: re-exports (session) | `src/winml/modelkit/session/__init__.py` | `deduce_ep_device` |
| Modified: trivial typed factory | `src/winml/modelkit/compiler/configs.py` | `WinMLCompileConfig.for_ep_device(ep_device)` |
| Deleted: derived field | `src/winml/modelkit/compiler/configs.py` | `EPConfig.enable_ep_context` |
| Modified: lazy session + `Path` return | `src/winml/modelkit/session/session.py` | `WinMLSession.__init__`, `WinMLSession.compile()`, `WinMLSession._ensure_session()` |
| Modified: if/elif refactor | `src/winml/modelkit/session/ep_device.py` | `resolve_device(ep, device)` |
| Modified: spec-aware stage | `src/winml/modelkit/compiler/stages/compile.py` | `CompileStage.process` (consults `get_compile_spec`), `_relocate_ctx` (replaces `_finalize_output`) |
| Modified: QAIRT subclass `Path` return | `src/winml/modelkit/session/qairt/qairt_session.py:75` | `WinMLQairtSession.compile() -> Path` (was `-> None`) |
| Deleted: dead property | `src/winml/modelkit/compiler/context.py:91-93` | `CompileContext.enable_ep_context` |
| Deleted: legacy CLI flag | `src/winml/modelkit/compiler/cli.py:110,135,172` | `--ep-context/--no-ep-context` and `enable_ep_context` parameter |
| Modified: perf compile guard | `src/winml/modelkit/commands/perf.py:425,1109` | gate `session.compile()` on `WinMLCompileSpec.supports_ep_context` |
| Modified: target-host caller sites | `src/winml/modelkit/config/build.py:307,604,616` | `resolve_quant_compile_config`, `generate_hf_build_config` |
| Modified: cross-host caller site | `src/winml/modelkit/commands/config.py:570-578` | uses `deduce_ep_device` instead of `for_provider(str)` |
| Modified: direct CLI compile callsite | `src/winml/modelkit/commands/compile.py:244` | simplified post-pipeline guard; no `enable_ep_context` reference; `result.output_path` always set by `CompileStage` |
| New: architecture test | `tests/unit/architecture/test_compile_spec_coverage.py` | key-parity assertion |
| New: catalog tests | `tests/unit/compiler/test_spec.py` | per-row sanity, raise-on-miss |
| New: deduction tests | `tests/unit/session/test_deduce_ep_device.py` | no-register invariant, parity with `resolve_device` |
| New: lazy-session tests | `tests/unit/session/test_winml_session.py` | `__init__` purity, `compile() -> Path` contract |
| Modified: caller migration | `tests/unit/models/auto/test_config.py:169,172` | replace `for_qnn`/`for_cpu` with `for_ep_device` |
| Modified: factory-test cleanup | `tests/unit/compiler/test_compiler_configs.py` | delete per-EP factory + `for_provider` test classes |
| Modified: build comments | `tests/unit/config/test_build.py:1857-1858` | drop `for_provider` references |
