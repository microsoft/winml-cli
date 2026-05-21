# v2 Plan Fact-Check — 2026-05-14

Verification of every claim in §1 of the v2 plan against the actual source
at HEAD `8fc6e30b`. Each claim labeled VERIFIED / FALSE / PARTIAL / CONTRADICTING
with evidence.

## TL;DR

- 10 claims VERIFIED (legit, can execute as planned)
- 0 claims FALSE
- 3 claims PARTIAL (mostly right, but sub-claim details are wrong)
- 1 claim CONTRADICTING (doc self-contradicts; resolved in §2 of the same doc)

## Definitive ground truth

| Question | Answer (verified) |
|---|---|
| `EP_DEVICE_SPECS` entry count | **13 entries** (indices 0–12, verified by Python import + enumerate) |
| `lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` | **None** (confirmed at runtime) |
| `lookup_device_spec("NvTensorRtRtxExecutionProvider", "gpu")` | **EPDeviceSpec(ep='NvTensorRtRtxExecutionProvider', device='gpu', default_provider_options={})** |
| `default_ep_for_device("gpu")` | **DmlExecutionProvider** |
| `default_ep_for_device("npu")` | **QNNExecutionProvider** |
| `default_ep_for_device("cpu")` | **CPUExecutionProvider** |
| `eps_for_device("npu")` | **{'OpenVINOExecutionProvider', 'QNNExecutionProvider', 'VitisAIExecutionProvider'}** |
| Wrong-casing occurrences in `check_ops.py` | **5 occurrences** at lines 264, 267, 289, 335, 343 |
| Wrong-casing occurrences in `winml.py` | **1 occurrence** at line 149 |

---

## Per-claim verdicts

### B1 — PARTIAL

**Claim**: `check_ops.py` uses `"NvTensorRTRTXExecutionProvider"` (wrong casing) at lines
267, 289, 335, 343, and the dict key at 284–291. `winml.py:149` also has wrong casing.
`lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` returns `None`.

**Evidence — confirmed lines:**
```
check_ops.py:264  raise ValueError("NvTensorRTRTXExecutionProvider only supports GPU device type")
check_ops.py:267      ep_name="NvTensorRTRTXExecutionProvider", device_type=ort.OrtHardwareDeviceType.GPU
check_ops.py:289      "NvTensorRTRTXExecutionProvider": RTXChecker,
check_ops.py:335          "NvTensorRTRTXExecutionProvider",
check_ops.py:343          "NvTensorRTRTXExecutionProvider"
winml.py:149          - "NvTensorRTRTXExecutionProvider"
```

**Runtime confirmation:**
```
lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu") → None
lookup_device_spec("NvTensorRtRtxExecutionProvider", "gpu") → EPDeviceSpec(ep='NvTensorRtRtxExecutionProvider', device='gpu', ...)
```

**Verdict**: PARTIAL — The bug is real and the catalog lookup does return `None` for the wrong
casing (BLOCKER confirmed). However, the cited line list is **incomplete**: the plan cites lines
267, 289, 335, 343 but **omits line 264**, which also contains `"NvTensorRTRTXExecutionProvider"`
in the `raise ValueError(...)` inside `RTXChecker.__init__`. That is a **5th occurrence** that the
migration step (§7 step 1) must also fix.

**Notes**: The dict at lines 284–291 is cited correctly (the key is at line 289, within that range).
The B1 fix scope should be widened to include line 264.

---

### I1 — VERIFIED

**Claim**: `analyze/analyzer.py:667–671` has a hardcoded 3-EP list
`["QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider"]`
in the "analyze all supported EPs" branch. Not derived from the catalog.

**Evidence (actual source at lines 665–672):**
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

**Verdict**: VERIFIED — Line numbers match exactly (list body at 667–671). The list is inline,
hardcoded, and has no import from or reference to `EP_DEVICE_SPECS` or `eps_for_device`.
This is confirmed to be a duplicate of the NPU-subset policy, with no catalog derivation.

---

### I2 — PARTIAL

**Claim**: `analyze/pattern/check_patterns.py:331` has
`choices=["QNNExecutionProvider", "OpenVINOExecutionProvider"]` as a hardcoded 2-EP argparse list.

**Evidence (actual source):**
```python
    parser.add_argument(
        "--ep",
        type=str,
        required=True,
        choices=["QNNExecutionProvider", "OpenVINOExecutionProvider"],   # line 331
```

**Verdict**: PARTIAL — The claim content is correct (the string is at line 331, it is a
`choices=` list, values match, no catalog derivation). However, the claim describes it as an
"argparse-style" list, but the actual code uses `argparse.ArgumentParser` directly, not
Click's `@click.option`. This is a documentation precision issue, not a code error. The fix
description (add a comment as intentional subset) is appropriate. All code-level details are
accurate.

---

### I3 — VERIFIED

**Claim**: `utils/optimum_loader.py:68` hardcodes `"CPUExecutionProvider"` and
`"CUDAExecutionProvider"` in a ternary. CPU branch is fine; GPU branch uses CUDA regardless
of Windows catalog default (`DmlExecutionProvider`). Intentional cross-platform HF Optimum path.

**Evidence (actual source at line 68):**
```python
                provider="CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
```

**Verdict**: VERIFIED — Line 68 matches exactly. The ternary form, both EP strings, and the
context (inside `ort_model_class.from_pretrained(...)`) are all correct. The fix is comment-only
(no code change), which is appropriate.

---

### I4 — CONTRADICTING

**Claim (§1 text)**: `EP_DEVICE_SPECS` tuple has 12 entries, not 13. Entry
`CPUExecutionProvider/cpu` was dropped from the current code.

**Evidence (Python runtime):**
```
EP_DEVICE_SPECS count: 13
  [0] ep='QNNExecutionProvider',          device='npu'
  [1] ep='DmlExecutionProvider',          device='gpu'
  [2] ep='CPUExecutionProvider',          device='cpu'   <-- present
  [3] ep='QNNExecutionProvider',          device='gpu'
  [4] ep='QNNExecutionProvider',          device='cpu'
  [5] ep='OpenVINOExecutionProvider',     device='npu'
  [6] ep='OpenVINOExecutionProvider',     device='gpu'
  [7] ep='OpenVINOExecutionProvider',     device='cpu'
  [8] ep='VitisAIExecutionProvider',      device='npu'
  [9] ep='MIGraphXExecutionProvider',     device='gpu'
  [10] ep='TensorrtExecutionProvider',    device='gpu'
  [11] ep='CUDAExecutionProvider',        device='gpu'
  [12] ep='NvTensorRtRtxExecutionProvider', device='gpu'
```

**Verdict**: CONTRADICTING — §1 says 12 entries, but §2 Finding A-5 of the same document
self-corrects to 13. The actual code has **13 entries**. `CPUExecutionProvider/cpu` IS present
at index 2 (source line 183). The §1 claim is **false**; §2 is **correct**. This finding
(I4) should be **dropped entirely** from §1 — there is no catalog gap and no code discrepancy.

---

### N1 — VERIFIED

**Claim**: Same code as I1 — `analyze/analyzer.py:667–671` static 3-element list (duplicate
finding). Severity: NICE-TO-HAVE.

**Evidence**: See I1. The list is at lines 667–671 and is identical to what I1 describes.

**Verdict**: VERIFIED — This is indeed a duplicate of I1's finding. The plan correctly notes
it as lower priority. The code evidence is the same as I1.

---

### N2 — VERIFIED

**Claim**: `tests/unit/config/test_precision.py:230` imports `VALID_EPS` from
`winml.modelkit.config.precision` rather than `winml.modelkit.session`.

**Evidence (actual source at line 230):**
```python
        from winml.modelkit.config.precision import VALID_EPS
```

**Verdict**: VERIFIED — Line 230 matches exactly. The import is from `config.precision`, not
`session`. This is fragile because `VALID_EPS` is only available in `precision` as a
module-level name leakage from its `from ..session import VALID_EPS` statement (not declared
in `__all__`). The fix (import from `winml.modelkit.session` directly) is appropriate.

---

### N3 — PARTIAL

**Claim**: Four command files hardcode `["auto", "npu", "gpu", "cpu"]` as device choices:
`commands/compile.py:55`, `commands/config.py:117`, `commands/eval.py:56`, `commands/perf.py:1243`.

**Evidence (actual source):**
```python
# compile.py:55
    type=click.Choice(["auto", "npu", "gpu", "cpu"], case_sensitive=False),

# config.py:117
    type=click.Choice(["auto", "npu", "gpu", "cpu"], case_sensitive=False),

# eval.py:56
    type=click.Choice(["auto", "cpu", "gpu", "npu"], case_sensitive=False),

# perf.py:1243
    type=click.Choice(["auto", "cpu", "gpu", "npu"], case_sensitive=False),
```

**Verdict**: PARTIAL — All four files do contain hardcoded device choice lists at the cited
line numbers. The claim is essentially correct. However, the **order differs** between files:
`compile.py` and `config.py` use `["auto", "npu", "gpu", "cpu"]` while `eval.py` and
`perf.py` use `["auto", "cpu", "gpu", "npu"]`. The plan's claim says all four use
`["auto", "npu", "gpu", "cpu"]` which implies a single canonical order — in reality there are
two different orderings. Minor inconsistency, does not affect the NICE-TO-HAVE verdict.

---

### N4 — PARTIAL

**Claim**: `vendor_id=0x4D4F` (Qualcomm) appears in multiple test fixtures. Specifically
at `test_ep_device.py:28, 46`, `test_build_session_options.py:25`, `test_qairt_session.py:30`,
`conftest.py:252`. Plan states "5×" total occurrences.

**Evidence (grep result across tests/):**
```
test_build_session_options.py:25     vendor_id=0x4D4F,
test_build_session_options.py:104    chosen = _ort_dev("NPU", 0x4D4F, ...)
test_build_session_options.py:105    sibling = _ort_dev("GPU", 0x4D4F, ...)
test_build_session_options.py:126    chosen = _ort_dev("NPU", 0x4D4F, ...)
test_build_session_options.py:146    only_gpu = _ort_dev("GPU", 0x4D4F, ...)
test_build_session_options.py:158    a = _ort_dev("NPU", 0x4D4F, ...)
test_build_session_options.py:159    b = _ort_dev("NPU", 0x4D4F, ...)
test_build_session_options.py:198    chosen = _ort_dev("NPU", 0x4D4F, ...)
test_ep_device.py:28                 vendor_id=0x4D4F,
test_ep_device.py:36                 assert rehydrated.vendor_id == 0x4D4F
test_ep_device.py:46                 vendor_id=0x4D4F,
test_ep_device.py:88,89,90,97,105,106,117,127,128 (via _fake_ort_dev helper)
test_ep_registry.py:32               d.device.vendor_id = 0x4D4F
conftest.py:252                      vendor_id=0x4D4F,
conftest.py:262                      d.device.vendor_id = 0x4D4F   <-- unreported
test_qairt_session.py:30             vendor_id=0x4D4F,
test_qairt_session.py:50             fake_ort_npu.device.vendor_id = 0x4D4F
```

**Verdict**: PARTIAL — The four specific cited lines (test_ep_device.py:28, 46;
test_build_session_options.py:25; test_qairt_session.py:30; conftest.py:252) are all
VERIFIED accurate. However, the plan significantly **understates** the scope: it says "5×" total
occurrences, but there are **at least 24 occurrences** of `0x4D4F` across 5 files (including
`test_ep_registry.py`, which is not cited at all). The plan's "5×" figure and the 4-file
scope are both wrong. This understates the benefit of the N4 fix — a shared constant would
eliminate far more duplication than described.

---

### N5 — VERIFIED

**Claim**: `test_ep_device_import_rule.py` does not detect inline literal EP/device mapping
dicts. The detector is AST-based and scans imports only.

**Evidence (from reading the test file):**
The `_is_direct_ep_device_import()` function checks only `ast.ImportFrom` and `ast.Import`
node types. The `_collect_violations()` walker calls `_is_direct_ep_device_import(node)` for
every node in the AST — it matches only import statements (lines 58–70). There is no
`ast.Assign`, `ast.Dict`, `ast.Set`, or `ast.Call` check. A constant like
`NPU_EPS = {"qnn", "vitisai"}` defined outside `ep_device.py` would produce no AST import
node and would not be flagged.

**Verdict**: VERIFIED — The architecture test genuinely cannot detect inline EP/device literal
mappings. Only import nodes are checked. The gap described in §4 is real.

---

### N6 — VERIFIED

**Claim**: `winml.py:149` docstring has wrong casing `"NvTensorRTRTXExecutionProvider"`.

**Evidence (actual source at line 149):**
```python
        - "NvTensorRTRTXExecutionProvider"
```

**Verdict**: VERIFIED — The wrong-casing string is at line 149 in a docstring. Confirmed by
both direct Read and by the grep search.

---

### N7 — VERIFIED

**Claim**: `analyze/runtime_checker/check_ops.py:284–291` `ep_name_to_checker` dict has
`"NvTensorRTRTXExecutionProvider"` as a key.

**Evidence (actual source at lines 284–291):**
```python
    ep_name_to_checker = {
        "QNNExecutionProvider": QNNNPUChecker,
        "OpenVINOExecutionProvider": OpenVINONPUChecker,
        "VitisAIExecutionProvider": VitisAIChecker,
        "MIGraphXExecutionProvider": MIGraphXChecker,
        "NvTensorRTRTXExecutionProvider": RTXChecker,   # line 289
        # Add other EPChecker subclasses here as needed
    }
```

**Verdict**: VERIFIED — The dict key with wrong casing is at line 289, within the 284–291
range cited. This is a real bug: any caller passing `"NvTensorRtRtxExecutionProvider"` (correct
casing, as returned by ORT or `expand_ep_name`) would not find a matching checker and would
raise `ValueError`.

---

## Doc self-contradictions

### I4 vs Finding A-5

**Issue**: The §1 decision item I4 states that `EP_DEVICE_SPECS` has 12 entries (not 13) and
that `CPUExecutionProvider/cpu` was dropped. However, §2 Finding A-5 of the same document
explicitly corrects itself: "Full count is confirmed 13. The CPUExecutionProvider/cpu entry IS
present at line 183."

**Ground truth**: 13 entries confirmed by Python runtime. `CPUExecutionProvider/cpu` is at
index 2 in the tuple (source line 183).

**Resolution**: I4 is a **false problem statement**. The §1 claim was an error made during
drafting that was caught and corrected in §2 before the plan was finalized. The §1 text
was never updated to remove the corrected item.

**What should be done**: Drop I4 entirely from §1. It describes a non-existent discrepancy
between the design doc and the code. The catalog is correct; there is nothing to fix.

---

## Summary of doc errors requiring edits

If this fact-check is used to revise the v2 plan before execution:

1. **§1 I4** — Drop the entire I4 bullet. It is a false finding; the catalog has 13 entries
   and `CPUExecutionProvider/cpu` is present. §2 already acknowledges this but §1 was never
   cleaned up.

2. **§1 B1** and **§7 step 1** — Widen the line list for `check_ops.py` to include
   line 264 (the `raise ValueError(...)` inside `RTXChecker.__init__`). The fix currently
   describes 4 occurrences (267, 289, 335, 343) but there are 5.

3. **§1 N4** and **§7 step 7** — The "5×" occurrence count is wrong. There are at least 24
   occurrences of `0x4D4F` across 5 files (including `test_ep_registry.py`, which is unmentioned).
   The plan should either enumerate the actual scope or describe it as "many occurrences across
   the session test suite." The cited lines (28, 46 in test_ep_device; :25 in
   test_build_session_options; :30 in test_qairt_session; :252 in conftest) are all correct,
   but the scope is understated.

4. **Status snapshot line 19** — The parenthetical "note: was 13 in design doc; actual tuple is
   12 entries — see §2 Finding A-5" is backwards. The actual tuple is 13, not 12. This line
   should be updated to: "`EP_DEVICE_SPECS` catalog: 13 entries (confirmed — see §2 Finding A-5)".

No other claims require edits. All remaining findings (B1 core, I1, I2, I3, N1, N2, N3, N5,
N6, N7) are accurate enough to execute as written.
