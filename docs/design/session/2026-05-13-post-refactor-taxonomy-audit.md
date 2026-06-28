# Post-Refactor Taxonomy Duplication Audit — 2026-05-13

After commit `680b232c` introduced `EPDeviceSpec` as the single source of truth,
this audit re-runs the patterns from the 2026-05-12 sweep and surfaces anything
that should not coexist with the new catalog.

## Verdict

**PASS-WITH-EXCEPTIONS** — `_EP_TO_DEVICE` and `_DEVICE_TO_PROVIDER` are fully
removed; `EP_DEVICE_SPECS` is the sole source of truth for EP/device (ep, device)
pairs. Three surviving exceptions are documented below; two are justified carve-outs
and one is a pre-existing unfixed item from the sweep.

## Summary

- 30+ grep patterns from the 2026-05-12 sweep re-run across `src/` and `tests/`
- 6 findings total (categorized by severity below)
- 2 backward-compat wrappers (`get_provider_for_device`, `_VALID_DEVICES` in sysinfo)
- Multiple documentation references in design docs — all reference `EP_DEVICE_SPECS`
  as the source; inline tables in historical docs are fine as documentation

---

## Section 1 — Removed names (must be 0 hits)

`_EP_TO_DEVICE`: **0 production hits** (only survives as a string literal in the
architecture test `tests/unit/architecture/test_ep_device_import_rule.py:140` —
this is the test's own detector fixture, not a real import).

`_DEVICE_TO_PROVIDER`: **0 production hits** (only in docstring references inside
`ep_device.py` explaining what the new helpers replace — these are comments).

Status: **CLEAN**

---

## Section 2 — Backward-compat wrappers

### 2a. `get_provider_for_device` — **EXCEPTION, severity: MEDIUM**

**Implementation** (`src/winml/modelkit/session/ep_device.py:265-291`):
The function was NOT deleted. It exists as a named function containing a
**new independent `_compile_provider` dict** hardcoded as `{"npu": "qnn", "gpu": "dml", "cpu": None}`.
This is a new parallel data structure — it is not derived from `EP_DEVICE_SPECS`.
The docstring explains the rationale: this is a "compile-provider" subset that
intentionally preserves the old `_DEVICE_TO_PROVIDER` behavior (`"gpu" → "dml"`)
rather than returning the catalog's first GPU EP (`OpenVINOExecutionProvider`).

**Callers:**
- `src/winml/modelkit/config/precision.py:25,270` — production (2 sites)
- `src/winml/modelkit/config/build.py:606,608` — production (2 sites, via
  `from .precision import get_provider_for_device`)
- `src/winml/modelkit/session/__init__.py:23,67` — re-export in facade

**0 test callers** — no tests assert on `get_provider_for_device` directly.

**Design doc requirement** (§8): "DELETE `get_provider_for_device` — replaced by
`default_ep_for_device`. Callers must update to `short_ep_name(default_ep_for_device(device))`."

**Assessment:** The function embeds a compile-specific policy (`gpu → dml` not
`gpu → openvino`) that differs from the catalog's `default_ep_for_device` ordering.
This is a **justified carve-out** IF the compile path intentionally prefers DML
over OpenVINO for Windows GPU compilation targets. However, the current
implementation creates a hidden third mapping that is not visibly connected to
`EP_DEVICE_SPECS`, which defeats the single-source goal. The design doc did not
anticipate this distinction — it assumed the new `default_ep_for_device` could
simply replace `_DEVICE_TO_PROVIDER`.

**Recommendation:** Either (a) migrate the 2 production callers to
`short_ep_name(default_ep_for_device(device))` and reorder the catalog so DML
precedes OpenVINO for GPU, or (b) add a `is_compile_target: bool = False` field
to `EPDeviceSpec` and promote `get_provider_for_device` to a catalog-derived
helper. Option (b) preserves the semantic distinction while keeping EP_DEVICE_SPECS
as the authoritative source.

### 2b. `_VALID_DEVICES` in `sysinfo/device.py` — **EXCEPTION, severity: LOW**

**File:** `src/winml/modelkit/sysinfo/device.py:61`
```python
_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})
```

This is a **duplicate** of `_VALID_DEVICES` derived from `EP_DEVICE_SPECS` in
`session/__init__.py`. The `sysinfo/device.py` module also retains `_EP_DEVICE_MAP`
(7 entries, canonical-keyed) and its derived `_DEVICE_EP_MAP` — these predate the
new catalog and encode slightly different data (OpenVINO mapped to `"npu/gpu/cpu"`
as a multi-value string, not as separate entries).

The 2026-05-12 sweep identified this as "MOVE to ep_device.py; DELETE after step 1."
It was not addressed by commit `680b232c`.

**Assessment:** The `_VALID_DEVICES` duplicate is cosmetically inconsistent but
functionally harmless — both sets have the same three values. The `_EP_DEVICE_MAP`
and `_DEVICE_EP_MAP` in `sysinfo/device.py` serve a different purpose: they are
used by `resolve_device_category` (WMI-based host-hardware discovery) to find
**available** EPs on the current machine — a different problem from the catalog's
EP/device taxonomy. The multi-device `"npu/gpu/cpu"` entry for OpenVINO is
sysinfo-specific. These are not pure duplicates of `EP_DEVICE_SPECS`; they are a
related but distinct concept (runtime availability vs. supported variants).

**Recommendation (LOW):** Delete the duplicate `_VALID_DEVICES` line and import
from the session facade. Leave `_EP_DEVICE_MAP` and `_DEVICE_EP_MAP` in
`sysinfo/device.py` with a comment explaining they serve availability detection,
not taxonomy.

---

## Section 3 — Inline literals in production code

### 3a. `get_provider_for_device` embedded dict — **EXCEPTION (see Section 2a)**

`src/winml/modelkit/session/ep_device.py:286-290`:
```python
_compile_provider: dict[str, str | None] = {
    "npu": "qnn",
    "gpu": "dml",
    "cpu": None,
}
```
This is a new parallel dict not derived from `EP_DEVICE_SPECS`. Severity: MEDIUM
(documented in Section 2a).

### 3b. `commands/build.py:369-373` — **EXCEPTION, severity: LOW**

```python
candidate_eps = [
    "QNNExecutionProvider",
    "OpenVINOExecutionProvider",
    "VitisAIExecutionProvider",
]
```
Hardcoded list for auto-EP selection during build. This is the same item flagged
in the 2026-05-12 sweep (Section 2, `commands/build.py:370-372` → "NORMALIZE:
derive from consolidated table"). Not addressed by `680b232c`.

**Assessment:** This is an intentional policy list (NPU-capable EPs) rather than
a general taxonomy enumeration. The sweep noted "the 'auto-select NPU EP' logic
is heuristic and may need explicit curation regardless." A future `EP_DEVICE_SPECS`
field (e.g., `is_npu_capable: bool`) could replace this, but that is out of scope
per design doc §11.

### 3c. `compiler/cli.py:53` — **EXCEPTION, severity: LOW**

```python
type=click.Choice(["qnn", "cpu", "cuda", "dml"]),
```
Hardcoded EP short-name list. Identified in the 2026-05-12 sweep as the only
inconsistent `click.Choice` (missing `vitisai`/`migraphx`/`openvino`, adds `cuda`).
Not addressed by `680b232c`.

**Assessment:** `compiler/cli.py` is a separate legacy CLI entry point from
`commands/compile.py`. The inconsistency predates this PR. `commands/compile.py:61`
correctly uses `sorted(VALID_EPS)`. Fix: replace with `sorted(VALID_EPS)`.

### 3d. `analyze/runtime_checker/check_ops.py:330-335` — **EXISTING, severity: LOW**

```python
choices=[
    "QNNExecutionProvider",
    "OpenVINOExecutionProvider",
    "VitisAIExecutionProvider",
    "MIGraphXExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
]
```
Internal `argparse` choices for a subprocess entry point. Identified in the
2026-05-12 sweep. Not addressed by `680b232c`. Acceptable as a subprocess tool
boundary where the EP list is intentionally curated for runtime-checker support.

---

## Section 4 — Test fixtures

All test fixtures that hardcode `(ep, device)` pairs are legitimate:

- `tests/e2e/test_session.py:57-94` — cross-platform parametrize table including
  `rocm`, `cuda`, `coreml` which are intentionally NOT in the production catalog.
  Legitimate: the E2E suite must cover EPs beyond the Windows-ML catalog.

- `tests/unit/session/conftest.py:45-55` — `EP_NAME_MAP` with 9 entries including
  cross-platform EPs. Same justification as E2E table.

- `tests/unit/config/test_precision.py:187,208,239,461` — assertions on
  `compile_provider == "dml"` and `compile_provider == "qnn"`. These test the
  behavior of `get_provider_for_device` (the compile-provider mapping). They are
  correct test-of-contract assertions, not taxonomy duplicates.

- `tests/unit/architecture/test_ep_device_import_rule.py:140` — `_EP_TO_DEVICE`
  appears only as a string literal in a negative-test fixture (verifying the
  detector catches imports of deleted names). Correct behavior.

Status: **All test fixtures are legitimate.**

---

## Section 5 — CLI Choice lists

| File | Option | Values | Status |
|---|---|---|---|
| `commands/compile.py:55` | `--device` | `["auto","npu","gpu","cpu"]` | Hardcoded — consistent with all other device choices |
| `commands/compile.py:61` | `--ep` | `sorted(VALID_EPS)` | **Derived from catalog — correct** |
| `commands/config.py:117` | `--device` | `["auto","npu","gpu","cpu"]` | Hardcoded — consistent |
| `commands/eval.py:56` | `--device` | `["auto","cpu","gpu","npu"]` | Hardcoded — consistent |
| `commands/perf.py:1244` | `--device` | `["auto","cpu","gpu","npu"]` | Hardcoded — consistent |
| `utils/cli.py:16,19` | both | `sorted(_VALID_DEVICES)`, `sorted(VALID_EPS)` | **Derived from catalog — correct** |
| `compiler/cli.py:53` | `--ep` | `["qnn","cpu","cuda","dml"]` | **HARDCODED — stale (Section 3c)** |

Note: Device choices `["auto","npu","gpu","cpu"]` appear hardcoded in 4 commands
files. These are not taxonomy duplicates in the problematic sense — the set is
stable and `["auto","npu","gpu","cpu"]` is a policy value (includes "auto"). However,
the values are not derived from `_VALID_DEVICES`. A future improvement would be
`["auto"] + sorted(_VALID_DEVICES)`.

---

## Section 6 — Documentation references

All documentation in `docs/design/session/` that discusses EP/device taxonomy
refers to `EP_DEVICE_SPECS` as the source of truth
(`2026-05-13-ep-device-spec-design.md` is the canonical design doc). No doc was
found that contradicts or duplicates the catalog inline without attribution.

Historical docs (`2026-05-12-ep-taxonomy-sweep.md`, `2026-05-11-ep-device-refactor.md`)
contain inline EP/device tables as part of their archival documentation of the
pre-refactor state — this is expected and correct.

---

## Section 7 — Catalog structure verification

**Entry count:** 13 entries ✓ (matches design doc §3)

**Burst defaults:** QNN-NPU entry has `htp_performance_mode: "burst"` and
`htp_graph_finalization_optimization_mode: "3"` ✓

**`_BY_KEY` derivation:** `_BY_KEY` at line 206 is `{(s.ep, s.device): s for s in EP_DEVICE_SPECS}` — fully derived from catalog ✓

**`VALID_EPS` derivation:** `frozenset({short_ep_name(s.ep) for s in EP_DEVICE_SPECS})` — derived ✓

**`_VALID_DEVICES` derivation:** `frozenset({s.device for s in EP_DEVICE_SPECS})` — derived ✓

**Catalog order vs. design doc — ONE DISCREPANCY (informational):**
The design doc §3 comment states "cpu-first: QNNExecutionProvider". The
implementation instead places `OpenVINOExecutionProvider/cpu` at position 4,
before `QNNExecutionProvider/cpu` at position 12. This means
`default_ep_for_device("cpu") == "OpenVINOExecutionProvider"`, not `"QNNExecutionProvider"`.
The implementation's own comment (`OpenVINO CPU comes before QNN CPU so
default_ep_for_device("cpu") == OpenVINO`) indicates this was an intentional
override at implementation time. The design doc §3 text was not updated to reflect
this decision. **Severity: informational** — the comment in the code is
self-consistent; the design doc just needs a one-line update.

---

## Section 8 — Recommendations

| # | Severity | Finding | Recommended Action |
|---|---|---|---|
| R1 | MEDIUM | `get_provider_for_device` embeds a standalone compile-provider dict not derived from `EP_DEVICE_SPECS` | Either (a) add `is_primary_compile_target: bool` field to `EPDeviceSpec` and derive the mapping from the catalog, or (b) migrate 2 production callers and delete the function (requires DML to precede OpenVINO in GPU catalog slot, or callers tolerate OpenVINO as compile target) |
| R2 | LOW | `sysinfo/device.py:61` duplicates `_VALID_DEVICES` | Delete the line; `resolve_device_category` already validates against its own local `_VALID_DEVICES` — switching to the session facade import would be correct |
| R3 | LOW | `commands/build.py:369-373` hardcodes NPU-capable EP list | Accept as a curated policy list, or add `is_npu_capable: bool = False` to `EPDeviceSpec` and derive (requires hardware measurement to set the flag correctly) |
| R4 | LOW | `compiler/cli.py:53` uses stale hardcoded `["qnn","cpu","cuda","dml"]` | Replace with `sorted(VALID_EPS)` — 1 line fix |
| R5 | LOW | Device `click.Choice` lists hardcoded in 4 command files | Replace with `["auto"] + sorted(_VALID_DEVICES)` for consistency — 4 line fix |
| R6 | INFO | Design doc §3 says "cpu-first: QNNExecutionProvider" but catalog has OpenVINO first for cpu | Update the design doc comment to reflect the intentional override |

**Items resolved by `680b232c` (verified clean):**
- `_EP_TO_DEVICE` — deleted ✓
- `_DEVICE_TO_PROVIDER` — deleted ✓
- Three inline `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` dicts in `perf.py`/`evaluate.py` — deleted ✓
- `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES` in `utils/constants.py` — deleted ✓
- `SUPPORTED_DEVICES` uppercase constant — deleted ✓
- `utils/cli.py` EP/device choices — now derived from catalog ✓
- `config/precision.py` — no longer embeds its own EP/device mapping ✓
- `_EP_DEVICE_MAP` / `_DEVICE_EP_MAP` in `sysinfo/device.py` — NOT moved (different
  responsibility: runtime availability, not taxonomy); acceptable as-is per Section 2b
- Legacy `WinMLSession._build_session_options` instance method — replaced by free
  function ✓
- `_ep_defaults` in `session.py` — delegates to `lookup_device_spec` ✓
