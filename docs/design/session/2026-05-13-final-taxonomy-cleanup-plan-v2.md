# Final Taxonomy Cleanup Plan — v2 (2026-05-14)

> **v1**: `2026-05-13-final-taxonomy-cleanup-plan.md` — executed in commits
> `6ce5aa3d`, `720a4ed4`, `eee42e7f`, `8fc6e30b`.  All v1 decisions are landed.
>
> **v2** (this doc): a comprehensive re-audit found 1 BLOCKER + 4 IMPORTANT +
> 7 NICE-TO-HAVE items that v1's scope didn't cover.  v2 closes the audit
> trail definitively — anything not in v2's "out of scope" section is either
> done, queued in v2, or documented as a deliberate carve-out.

After the 4-commit cleanup (`680b232c..8fc6e30b`), audit every remaining
(EP, device) mapping and verify usage of the new EPDeviceSpec catalog
+ helpers across the full codebase.

## Status snapshot

- HEAD: `8fc6e30b`
- 8 commits ahead of `gh/main`
- `EP_DEVICE_SPECS` catalog: 13 entries (confirmed — see §2 Finding A-5)
- Session facade `__init__.py`: fully re-exports all new helpers; `get_provider_for_device` and `_VALID_DEVICES` are gone
- All major cleanup items from `2026-05-13-final-taxonomy-cleanup-plan.md` (D1–D7) are now **executed**

---

## 1. Decisions (what to fix/clean up)

### BLOCKERS (fix before next push)

- **B1** — `check_ops.py` and `winml.py` use wrong EP casing `"NvTensorRTRTXExecutionProvider"` (all-caps RTX) while catalog defines `"NvTensorRtRtxExecutionProvider"`. The `canonicalize_ep_name` alias exists to fix this casing, but these call sites bypass canonicalization and pass the wrong name directly to `EPChecker.__init__` and ORT. The catalog lookup `lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` returns `None`. This is a latent runtime bug. **Severity: BLOCKER** (silent wrong lookup). Fact-check widened scope: **5 occurrences** in `check_ops.py` at lines 264, 267, 289, 335, 343 (line 264 is the `raise ValueError(...)` site, missed in original plan).

### IMPORTANT (fix in this PR)

- **I1** — `analyze/analyzer.py:667–671` has a hardcoded 3-EP list `["QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider"]` in the "analyze all supported EPs" branch. This is a duplicate of an NPU-subset policy and will not include future EPs. **Severity: IMPORTANT** (diverges from catalog on next EP addition).
- **I2** — `analyze/pattern/check_patterns.py:331` has `choices=["QNNExecutionProvider", "OpenVINOExecutionProvider"]` as a hardcoded 2-EP argparse list. This is a subprocess tool boundary, but the values are hardcoded and bypass the catalog. **Severity: IMPORTANT** (curated subset, but no catalog derivation).
- **I3** — `utils/optimum_loader.py:68` hardcodes `"CPUExecutionProvider"` and `"CUDAExecutionProvider"` in a ternary. The CPU branch is fine (CPU EP is always canonical). The GPU branch uses `"CUDAExecutionProvider"` regardless of the catalog default for GPU (`"DmlExecutionProvider"` on Windows). **Severity: IMPORTANT** — this is an Optimum-specific codepath that intentionally uses CUDA (cross-platform HF loader), but it should be documented as an intentional carve-out rather than a silent divergence.
- ~~**I4**~~ — **DROPPED (false finding)**. Fact-check confirmed the catalog has 13 entries; `CPUExecutionProvider/cpu` IS present at index 2 (line 183). The §1 claim was an error caught and corrected in §2 Finding A-5. No code change needed.

### NICE-TO-HAVE (follow-up PR)

- **N1** — `analyze/analyzer.py:667–671` "all EPs" list does not derive from `eps_for_device("npu")` — it is a static 3-element list. Could be replaced with a helper call but is lower priority than B1/I1.
- **N2** — `test_precision.py:230` imports `VALID_EPS` from `winml.modelkit.config.precision` rather than the intended `winml.modelkit.session`. Works at runtime (Python re-exports imported names as attributes) but violates the facade contract and is fragile. **Severity: NICE-TO-HAVE** (cosmetic, non-breaking).
- **N3** — Four command files (`commands/compile.py:55`, `commands/config.py:117`, `commands/eval.py:56`, `commands/perf.py:1243`) hardcode `["auto", "npu", "gpu", "cpu"]` device choice lists instead of `["auto"] + sorted(VALID_DEVICES)`. Values are stable and currently correct, but not derived from the catalog. **Severity: NICE-TO-HAVE**.
- **N4** — `vendor_id=0x4D4F` (Qualcomm) appears in **24+ occurrences across 5 files** (`test_ep_device.py`, `test_build_session_options.py`, `test_qairt_session.py`, `conftest.py`, `test_ep_registry.py` — the last file was missed in the original plan). A shared `QNN_VENDOR_ID = 0x4D4F` constant in `tests/unit/session/conftest.py` would deduplicate. Fact-check confirmed the original "5×" count understated scope significantly. **Severity: NICE-TO-HAVE**.
- **N5** — Architecture test `test_ep_device_import_rule.py` does not detect inline literal EP/device mapping dicts (e.g., `NPU_EPS = {"qnn", "vitisai"}`). The detector is AST-based and scans imports only. **Severity: NICE-TO-HAVE** (gap in coverage).
- **N6** — `winml.py:149` docstring says `"NvTensorRTRTXExecutionProvider"` (wrong casing). After fixing B1, this docstring also needs updating. **Severity: NICE-TO-HAVE** (documentation only).
- **N7** — `analyze/runtime_checker/check_ops.py:284–291` `ep_name_to_checker` dict has `"NvTensorRTRTXExecutionProvider"` as a key. Once B1 is fixed, this key must also be corrected to `"NvTensorRtRtxExecutionProvider"`. **Severity: part of B1 fix**.

---

## 2. Audit findings — Investigation A: remaining mapping data

All mapping-direction literal patterns (`"qnn":"npu"`, `"npu":"qnn"`, `"dml":"gpu"`, `"gpu":"dml"`, `"cpu":None`, etc.) returned **0 hits** in `src/` and `tests/`. All `frozenset(...qnn...)`, `NPU_EPS`, `GPU_EPS`, `_NPU_PROVIDERS`, `_GPU_PROVIDERS` patterns returned **0 hits**. All removed-name patterns (`_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `_EP_DEVICE_MAP`, `_DEVICE_EP_MAP`, `_VALID_DEVICES`, `get_provider_for_device`, `_compile_provider`, `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `_NPU_EPS`) returned **0 production hits** (only docstring comments and the architecture test's own fixture strings).

| File:Line | Pattern | Verdict | Action |
|---|---|---|---|
| `analyze/runtime_checker/check_ops.py:264,267,289,335,343` | `"NvTensorRTRTXExecutionProvider"` (wrong casing) | **STALE STRING** | Fix to `"NvTensorRtRtxExecutionProvider"` (see B1) |
| `winml.py:149` | `"NvTensorRTRTXExecutionProvider"` in docstring | **STALE STRING** (docstring) | Update docstring after B1 fix |
| `analyze/analyzer.py:667–671` | `["QNNExecutionProvider","OpenVINOExecutionProvider","VitisAIExecutionProvider"]` inline list | **INLINE LITERAL** | Replace with catalog-derived set (see I1) |
| `analyze/pattern/check_patterns.py:331` | `choices=["QNNExecutionProvider","OpenVINOExecutionProvider"]` | **CARVE-OUT** | Subprocess boundary; document as intentional |
| `utils/optimum_loader.py:68` | `"CUDAExecutionProvider"` (GPU branch) | **CARVE-OUT** | Intentional for Optimum/HF cross-platform; add comment |
| `optim/pipes/graph.py:572` | `providers=["CPUExecutionProvider"]` | **CARVE-OUT** | ORT direct API call for optimization pass; CPU-only is correct |
| `conftest.py:92,136,158` | `providers=["CPUExecutionProvider"]` | **CARVE-OUT** | ORT direct API calls in test infra; correct |
| `unit/session/conftest.py:252` + 3 other test files | `vendor_id=0x4D4F` repeated 5× | **CARVE-OUT** | Legitimate test fixtures; shared constant NICE-TO-HAVE |
| `EP_DEVICE_SPECS` tuple (12 entries vs design doc 13) | Missing `CPUExecutionProvider/cpu` entry | **INLINE LITERAL** (catalog gap) | Verify actual tuple length — see I4 |
| `analyze/runtime_checker/check_ops.py:284–291` | `ep_name_to_checker` dict with `"NvTensorRTRTXExecutionProvider"` key | **STALE STRING** | Fix as part of B1 |

### Finding A-5 detail — catalog entry count

The `EP_DEVICE_SPECS` tuple in the current `ep_device.py` ends at line 198 with 12 `EPDeviceSpec(...)` entries (QNN/npu, DML/gpu, CPU/cpu, QNN/gpu, QNN/cpu, OpenVINO/npu, OpenVINO/gpu, OpenVINO/cpu, VitisAI/npu, MIGraphX/gpu, TensorRT/gpu, CUDA/gpu, NvTensorRtRtx/gpu). That is **13 entries** — the design doc count is correct. The CPUExecutionProvider/cpu entry IS present at position 2 in the catalog.

**Correction**: Full count is confirmed 13. The status snapshot above was correct. `CPUExecutionProvider/cpu` is at line 183. Initial concern is resolved.

---

## 3. Audit findings — Investigation B: ep_device usage

| File:Line | Usage | Verdict | Action |
|---|---|---|---|
| `session/ep_device.py:465` | `EPDevice(ep=..., device=..., vendor_id=..., ...)` | OK | Only construction site in src/; legitimate inside resolve_device |
| `session/monitor/qnn_monitor.py:497,640` | `ep="QNNExecutionProvider"` in `OpTraceResult(...)` | OK | Monitor is QNN-specific; hardcoded canonical name is correct |
| `commands/perf.py:771` | `resolve_device("cpu", "cpu")` | SUBOPTIMAL | Passes both ep AND device as positional strings where ep is a short name. Passes `"cpu"` as both `ep` and `device` which works because `expand_ep_name("cpu") == "CPUExecutionProvider"` and device is `"cpu"`. Functional but unusual calling convention — should use keyword args: `resolve_device(ep="cpu", device="cpu")` |
| `config/precision.py:21–27` | Imports `VALID_DEVICES, VALID_EPS, default_ep_for_device, ep_to_device, short_ep_name` from session facade | OK | Clean facade import |
| `config/build.py:606–610` | Uses `default_ep_for_device` + `short_ep_name` pattern | OK | Correct, matches design doc After code |
| `commands/build.py:368–379` | Uses `resolve_device` + `short_ep_name` | OK | Correct — auto-detection via catalog |
| `utils/cli.py:11,16,19` | `from ..session import VALID_DEVICES, VALID_EPS` | OK | Correct facade import |
| `compiler/cli.py:13,54` | `from ..session import VALID_EPS` + `sorted(VALID_EPS)` | OK | Correct, this was the fix from D6 |
| `compiler/configs.py:111–113` | `from ..session import short_ep_name` then `short_ep_name(ep_device.ep)` | OK | Clean usage |
| `compiler/stages/compile.py:75` | `resolve_device(ep=ep_str)` (ep-only form) | OK | Correct; device deduced from catalog |
| `session/ep_registry.py` | `from .ep_device import EPNotDiscovered, EPRegistrationFailed` | OK | Within-session sibling import, allowed |
| `session/session.py:22–28` | `from .ep_device import ...` | OK | Within-session sibling import, allowed |
| `models/winml/base.py:33` | `from ...session.session import WinMLSession` | BYPASSING (session.session, not ep_device) | Out of scope for this ticket — architecture test does not guard `session.session`; separate concern |
| `test_precision.py:230` | `from winml.modelkit.config.precision import VALID_EPS` | SUBOPTIMAL | `VALID_EPS` is imported into `precision.py` from session but not declared in its `__all__`; test should import directly from `winml.modelkit.session import VALID_EPS` |
| `session/qairt/qairt_session.py:237` | `from ..session import _build_session_options` | **BYPASSING** | `_build_session_options` is a private function; importing it from `..session` (which re-exports only public API) will fail at runtime unless it is in `__init__.py`. Verify this path. |
| `analyze/runtime_checker/check_ops.py:267,289` | `"NvTensorRTRTXExecutionProvider"` as key/arg | **BROKEN** | Wrong casing bypasses catalog lookup — see B1 |

### Investigation B — `canonicalize_ep_name` usage

`canonicalize_ep_name` is only used inside `ep_device.py` itself (the alias normalization stub). External callers do not call it directly — they use `expand_ep_name` or pass strings to `resolve_device`. This is correct: `canonicalize_ep_name` is meant to be an implementation detail called within `expand_ep_name`.

### Investigation B — `EPDevice` direct construction

`EPDevice(...)` is constructed directly in 16 test files (fixtures and parametrize entries). All follow the pattern `EPDevice(ep="FullEPName", device="device", vendor_id=..., device_id=...)` which is the correct and only way to create a test fixture without calling `resolve_device` (which requires ORT). This is legitimate — tests cannot and should not call `resolve_device` in unit tests. No action needed.

---

## 4. Architecture-test gaps

The current `test_ep_device_import_rule.py` guards only against `session.ep_device` direct imports. It does NOT detect:

### Gap 1 — Inline EP/device mapping literals

A constant like `NPU_EPS = {"qnn", "vitisai"}` defined outside `ep_device.py` duplicates catalog data but is invisible to the AST import scanner. Detection would require a semantic pattern matcher, not just import analysis.

**Proposed test case (conceptual, no implementation):**
```python
# Scan src/**/*.py for the pattern:
#   frozenset({"qnn", "vitisai", ...})  # EP short names as literals
#   frozenset({"npu", "gpu", "cpu"})    # device strings as literals
# outside session/ep_device.py and utils/cli.py
# Flag as violations when the set is a known sub-set of VALID_EPS or VALID_DEVICES
```

This is hard to implement generically without false positives (any frozenset containing "cpu" would trigger). Deferred.

### Gap 2 — Deleted name sentinels

The architecture test's parametrize list already includes:
- `"from winml.modelkit.session.ep_device import _EP_TO_DEVICE"` ✓
- `"from winml.modelkit.session.ep_device import get_provider_for_device"` ✓

**Missing sentinels to add** (import forms that should be detected if someone re-adds them):
```python
"from winml.modelkit.session.ep_device import _DEVICE_TO_PROVIDER",
"from winml.modelkit.session.ep_device import _VALID_DEVICES",   # was the old private name
"from winml.modelkit.session.ep_device import _compile_provider",
"from winml.modelkit.session import get_provider_for_device",   # removed from facade too
"from winml.modelkit.session import _VALID_DEVICES",            # removed from facade
```

**Proposed implementation:** Add these 5 strings to `test_detector_catches_forbidden_forms` parametrize list. No change to `_collect_violations` scanner needed — those are already caught by the import-from-ep_device pattern for the ep_device variants; the facade variants need a second scan targeting `session/__init__.py` exports (currently not scanned).

### Gap 3 — `session.session` direct imports from outside session/

`models/winml/base.py:33` imports `from ...session.session import WinMLSession` rather than `from ...session import WinMLSession`. The architecture test only guards `session.ep_device`, not `session.session`. This is a minor facade violation but for a different sub-module.

**Proposed test case:**
```python
def test_no_direct_session_session_imports_in_src():
    """Source files outside session/ should not import directly from session/session.py."""
    # Similar structure to test_no_direct_ep_device_imports_in_src
    # Pattern: "session.session" in module path
```

---

## 5. Per-file before/after diffs

### File 1: `src/winml/modelkit/analyze/runtime_checker/check_ops.py`

**Finding B1 — Wrong casing for NvTensorRtRtx (BLOCKER)**

**Before:**
```python
# Line 260–268:
class RTXChecker(EPChecker):
    """NVIDIA TensorRT RTX execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        if device_type != ort.OrtHardwareDeviceType.GPU:
            raise ValueError("NvTensorRTRTXExecutionProvider only supports GPU device type")
        """Initialize RTX checker."""
        super().__init__(
            ep_name="NvTensorRTRTXExecutionProvider", device_type=ort.OrtHardwareDeviceType.GPU
        )

# Line 284–291:
    ep_name_to_checker = {
        "QNNExecutionProvider": QNNNPUChecker,
        "OpenVINOExecutionProvider": OpenVINONPUChecker,
        "VitisAIExecutionProvider": VitisAIChecker,
        "MIGraphXExecutionProvider": MIGraphXChecker,
        "NvTensorRTRTXExecutionProvider": RTXChecker,
        # Add other EPChecker subclasses here as needed
    }

# Line 330–335:
    choices=[
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "VitisAIExecutionProvider",
        "MIGraphXExecutionProvider",
        "NvTensorRTRTXExecutionProvider",
    ],
```

**After:**
```python
# Line 260–268:
class RTXChecker(EPChecker):
    """NVIDIA TensorRT RTX execution provider checker wrapper for pytest compatibility."""

    def __init__(self, device_type: ort.OrtHardwareDeviceType) -> None:
        if device_type != ort.OrtHardwareDeviceType.GPU:
            raise ValueError("NvTensorRtRtxExecutionProvider only supports GPU device type")
        """Initialize RTX checker."""
        super().__init__(
            ep_name="NvTensorRtRtxExecutionProvider", device_type=ort.OrtHardwareDeviceType.GPU
        )

# Line 284–291:
    ep_name_to_checker = {
        "QNNExecutionProvider": QNNNPUChecker,
        "OpenVINOExecutionProvider": OpenVINONPUChecker,
        "VitisAIExecutionProvider": VitisAIChecker,
        "MIGraphXExecutionProvider": MIGraphXChecker,
        "NvTensorRtRtxExecutionProvider": RTXChecker,  # canonical casing from catalog
        # Add other EPChecker subclasses here as needed
    }

# Line 330–335:
    choices=[
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "VitisAIExecutionProvider",
        "MIGraphXExecutionProvider",
        "NvTensorRtRtxExecutionProvider",  # canonical casing from catalog
    ],
```

Rationale: `_EP_NAME_ALIASES` in `ep_device.py` maps `"nvtensorrtrtxexecutionprovider"` → `"NvTensorRtRtxExecutionProvider"` (casing fix). The analyzer's `check_ops.py` bypasses this alias by hardcoding the old wrong casing. A `lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` would return `None` because the catalog key uses the correct casing.

---

### File 2: `src/winml/modelkit/winml.py`

**Finding N6 — Docstring casing (NICE-TO-HAVE, part of B1 fix)**

**Before (line 149):**
```python
        - "NvTensorRTRTXExecutionProvider"
```

**After:**
```python
        - "NvTensorRtRtxExecutionProvider"
```

Rationale: Docstring should reflect the canonical catalog spelling.

---

### File 3: `src/winml/modelkit/analyze/analyzer.py`

**Finding I1 — Hardcoded 3-EP list (IMPORTANT)**

**Before (lines 665–672):**
```python
        if ep_normalized is None:
            # Analyze all supported EPs
            eps_to_analyze = [
                "QNNExecutionProvider",
                "OpenVINOExecutionProvider",
                "VitisAIExecutionProvider",
            ]
            logger.info("No EP specified, analyzing all supported EPs: %s", eps_to_analyze)
```

**After:**
```python
        if ep_normalized is None:
            # Analyze all NPU-capable EPs from catalog, in catalog order.
            # eps_for_device returns a frozenset; sort by position for determinism.
            from ..session import EP_DEVICE_SPECS, eps_for_device
            _npu_canonical = frozenset(s.ep for s in EP_DEVICE_SPECS if s.device == "npu")
            eps_to_analyze = sorted(
                _npu_canonical,
                key=lambda e: next(i for i, s in enumerate(EP_DEVICE_SPECS) if s.ep == e and s.device == "npu"),
            )
            logger.info("No EP specified, analyzing all NPU-capable EPs: %s", eps_to_analyze)
```

Rationale: The inline list hardcodes the NPU EP subset and will miss future catalog additions (e.g., if a new NPU EP is added to `EP_DEVICE_SPECS`). Sorting by catalog position preserves the existing priority order (QNN first, then OpenVINO, then VitisAI).

**Note:** The `analyze/` module may have a deliberate policy to limit "all EPs" to a curated NPU subset. If this is intentional (not all catalog NPU EPs are supported by the static analyzer), a comment should document that and the list can remain hardcoded. Discuss with team before applying this diff.

---

### File 4: `src/winml/modelkit/utils/optimum_loader.py`

**Finding I3 — CUDAExecutionProvider carve-out (IMPORTANT — add comment only)**

**Before (line 68):**
```python
                provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
```

**After:**
```python
                # Intentional: Optimum's ORTModel uses CUDA as the generic non-CPU GPU EP.
                # This is a cross-platform HF Optimum codepath and does NOT use the
                # Windows-ML catalog's default GPU EP (DmlExecutionProvider).
                # See: https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort
                provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
```

Rationale: Prevents future auditors from flagging this as an unintended catalog bypass.

---

### File 5: `tests/unit/architecture/test_ep_device_import_rule.py`

**Finding Gap 2 — Missing deleted-name sentinels (NICE-TO-HAVE)**

**Before (parametrize list at line 136–147):**
```python
    [
        "from winml.modelkit.session.ep_device import EPDevice",
        "from winml.modelkit.session.ep_device import EPDevice, resolve_device",
        "from winml.modelkit.session.ep_device import _EP_TO_DEVICE",
        "from winml.modelkit.session.ep_device import get_provider_for_device",
        "import winml.modelkit.session.ep_device",
        "import winml.modelkit.session.ep_device as epd",
        "from .ep_device import EPDevice",
    ],
```

**After (add 3 new deleted-name sentinels):**
```python
    [
        "from winml.modelkit.session.ep_device import EPDevice",
        "from winml.modelkit.session.ep_device import EPDevice, resolve_device",
        "from winml.modelkit.session.ep_device import _EP_TO_DEVICE",
        "from winml.modelkit.session.ep_device import _DEVICE_TO_PROVIDER",
        "from winml.modelkit.session.ep_device import _VALID_DEVICES",
        "from winml.modelkit.session.ep_device import _compile_provider",
        "from winml.modelkit.session.ep_device import get_provider_for_device",
        "import winml.modelkit.session.ep_device",
        "import winml.modelkit.session.ep_device as epd",
        "from .ep_device import EPDevice",
    ],
```

Rationale: Adds coverage for the other names deleted in the cleanup commits.

---

### File 6: `tests/unit/config/test_precision.py`

**Finding N2 — VALID_EPS import from wrong module (NICE-TO-HAVE)**

**Before (line 230):**
```python
        from winml.modelkit.config.precision import VALID_EPS
```

**After:**
```python
        from winml.modelkit.session import VALID_EPS
```

Rationale: `VALID_EPS` is a session-facade symbol. Importing it via `config.precision` (where it is a private implementation-import, not a re-exported public symbol) is fragile and violates the intended API boundary.

---

## 6. New helpers needed (if any)

None required. All needed helpers (`eps_for_device`, `VALID_DEVICES`, `VALID_EPS`, `default_ep_for_device`, `short_ep_name`) are already present and exported.

---

## 7. Migration plan

Steps in execution order:

1. **Fix B1 (BLOCKER)** — Update `check_ops.py` lines 264, 267, 289, 335, 343 (5 occurrences) from `NvTensorRTRTXExecutionProvider` → `NvTensorRtRtxExecutionProvider`. Update `winml.py:149` docstring simultaneously.
2. **Fix I1 (IMPORTANT)** — Discuss with team whether `analyze/analyzer.py:667–671` "all EPs" list is intentionally curated or should derive from `eps_for_device("npu")`. If catalog-derived: apply File 3 diff. If intentionally curated: add a comment explaining the curation policy.
3. **Fix I2 (IMPORTANT)** — Add a docstring comment to `check_patterns.py:331` argparse `choices=` list explaining it is an intentional subprocess-boundary subset. No code change.
4. **Fix I3 (IMPORTANT)** — Add explanatory comment to `optimum_loader.py:68` (File 4 diff). No code change.
5. **Fix N2 (NICE-TO-HAVE)** — Update `test_precision.py:230` to import `VALID_EPS` from `winml.modelkit.session`.
6. **Fix N3 (NICE-TO-HAVE)** — Replace hardcoded `["auto","npu","gpu","cpu"]` in 4 command files with `["auto"] + sorted(VALID_DEVICES)`. Add `from ..session import VALID_DEVICES` (or `from ..utils.cli import _DEVICE_CHOICES` if the shared helper is preferred).
7. **Fix N4 (NICE-TO-HAVE)** — Add `QNN_VENDOR_ID: int = 0x4D4F` constant to `tests/unit/session/conftest.py` and replace the 24+ inline occurrences across 5 files (`test_ep_device.py`, `test_build_session_options.py`, `test_qairt_session.py`, `conftest.py`, `test_ep_registry.py`).
8. **Fix arch test (NICE-TO-HAVE)** — Add missing deleted-name sentinels to `test_ep_device_import_rule.py` (File 5 diff).
9. **Run verification gate** (see §8).
10. **Commit** — All changes above as a single atomic commit with tag `fix(session+analyze): NvTensorRtRtx casing fix + catalog boundary comments`.

---

## 8. Verification gate

```bash
# 1. Confirm NvTensorRtRtx casing is consistent across all files
# Expected: 0 hits for wrong casing (outside ep_device.py alias table)
# PowerShell: Select-String -Path src\**\*.py -Pattern "NvTensorRTRTX" -Recurse
# Expected: 0 hits (allow: ep_device.py alias table entry, which is the fix map)

# 2. Lint
uv run ruff check --fix src/ tests/

# 3. Full unit suite
uv run pytest tests/unit/ --tb=short -q

# 4. Architecture regression
uv run pytest tests/unit/architecture/ -v --tb=short

# 5. Catalog sanity smoke test
uv run python -c "
from winml.modelkit.session import (
    EP_DEVICE_SPECS, VALID_DEVICES, VALID_EPS,
    default_ep_for_device, default_device_for_ep, eps_for_device, short_ep_name,
    lookup_device_spec,
)
print('Total variants:           ', len(EP_DEVICE_SPECS))
print('VALID_DEVICES:            ', sorted(VALID_DEVICES))
print('VALID_EPS:                ', sorted(VALID_EPS))
print('default_ep_for_device(npu):', default_ep_for_device('npu'))  # QNNExecutionProvider
print('default_ep_for_device(gpu):', default_ep_for_device('gpu'))  # DmlExecutionProvider
print('default_ep_for_device(cpu):', default_ep_for_device('cpu'))  # CPUExecutionProvider
print('eps_for_device(npu):      ', sorted(eps_for_device('npu')))  # 3 entries
print('lookup NvTensorRtRtx/gpu: ', lookup_device_spec('NvTensorRtRtxExecutionProvider', 'gpu'))
print('lookup NvTensorRTRTX/gpu: ', lookup_device_spec('NvTensorRTRTXExecutionProvider', 'gpu'))  # must be None
"
# Expected:
#   Total variants:            13
#   default_ep_for_device(gpu): DmlExecutionProvider
#   default_ep_for_device(cpu): CPUExecutionProvider
#   lookup NvTensorRtRtx/gpu:  EPDeviceSpec(ep='NvTensorRtRtxExecutionProvider', device='gpu', ...)
#   lookup NvTensorRTRTX/gpu:  None

# 6. Session + commands suites
uv run pytest tests/unit/session/ tests/unit/commands/ tests/unit/config/ -v --tb=short -q
```

---

## 9. Risks

### R1 — `check_ops.py` casing fix breaks existing callers

The `ep_name_to_checker` dict key change from `"NvTensorRTRTXExecutionProvider"` to `"NvTensorRtRtxExecutionProvider"` will break any caller that passes the old wrong casing. But callers should be passing canonicalized names (from ORT or from `expand_ep_name`); ORT itself returns `"NvTensorRtRtxExecutionProvider"` (correct casing). So this should be a net fix. **Mitigation:** Grep all callers of `get_ep_checker` in `check_ops.py` before applying.

### R2 — `analyze/analyzer.py` catalog-derive change alters EP ordering

Replacing the hardcoded 3-EP list with a catalog-derived sorted list could change iteration order if catalog order ever changes. Current catalog order (QNN, then OpenVINO, then VitisAI for NPU) matches the hardcoded list, so there is no immediate behavior change. **Mitigation:** Only apply this change after confirming intent with team.

### R3 — `VALID_EPS` import fix in tests could shadow a bug

If `VALID_EPS` in `config.precision` ever diverges from `session.VALID_EPS` (e.g., due to a PR that modifies precision without updating session), the test currently catches it via the cross-module import. Fixing the test to import directly from `session` removes this accidental cross-check. **Mitigation:** Low risk — VALID_EPS is a derived frozenset with no independent definition in `precision.py`.

---

## 10. Out of scope

- Replacing the 4 hardcoded `["auto","npu","gpu","cpu"]` device choice lists in commands files — deferred as N3 unless the team wants it in this PR.
- Adding `is_compile_target: bool` or `supports_op_tracing: bool` fields to `EPDeviceSpec` — deferred per design doc §11.
- `models/winml/base.py:33` importing `from ...session.session import WinMLSession` (bypasses session facade, different sub-module, separate concern).
- Fixing `analyze/runtime_checker/ep_checker.py:31` `EPS_REQUIRING_FILE_PATH = {"VitisAIExecutionProvider"}` — intentional curated set for VitisAI provider configuration, not a taxonomy duplicate.
- `feat/update-pkg-deps` integration — the `canonicalize_ep_name` stub in `ep_device.py` explicitly defers to that PR.
- Speculative `default_provider_options` for unverified variants (OpenVINO, TensorRT, CUDA, etc.).
