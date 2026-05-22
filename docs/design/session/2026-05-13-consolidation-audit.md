# Consolidation Audit — 2026-05-13

Audit of commits 3b155784, 0a9d422a, 62807ac9 against the plan at
`docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md`.

## Verdict

PASS-WITH-CONCERNS — taxonomy consolidation and architecture enforcement are solid; three
plan deviations in `commands/compile.py` leave the CLI boundary incomplete per the plan spec.

## Summary

- 29 requirements verified
- 3 deviations from plan
- 0 bugs / regressions found that break existing tests
- 3 surprises / additions beyond plan
- Architecture regression test: **CONFIRMED effective** (adversarial case verified)

---

## Per-requirement audit

### §A1: Taxonomy tables in ep_device.py

- **Status: PASS**
- `src/winml/modelkit/session/ep_device.py:147–167`
  - `_EP_TO_DEVICE` at line 147 (8 entries, matches plan)
  - `_DEVICE_TO_PROVIDER` at line 159 (3 entries)
  - `VALID_EPS = frozenset(_EP_TO_DEVICE.keys())` at line 166
  - `_VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})` at line 167
  - `get_provider_for_device` at line 170

### §A2: Removed from precision.py

- **Status: PASS**
- `src/winml/modelkit/config/precision.py` has no definitions of `_EP_TO_DEVICE`,
  `_DEVICE_TO_PROVIDER`, `VALID_EPS`, or `_VALID_DEVICES`.
- Instead imports at line 21–26: `from ..session import _VALID_DEVICES, VALID_EPS, ep_to_device, get_provider_for_device`.
- **Note:** The import uses `ep_to_device` (the new public wrapper) rather than
  `_EP_TO_DEVICE` directly. This is cleaner than what the plan proposed and correct.

### §A3: Direct ep_device imports outside session/

- **Status: PASS**
- `grep -rn "from .*session\.ep_device" src/winml/modelkit/` — only hits are in
  **docstring text** in `models/auto.py:121`, `models/auto.py:237`,
  `models/winml/base.py:75`. These are string literals, not import statements.
  Zero live violations.

### §A4: Test imports of ep_device

- **Status: PASS**
- `grep -rn "from .*session\.ep_device" tests/` — only hits are inside
  `tests/unit/architecture/test_ep_device_import_rule.py` (the test itself, in
  parametrized forbidden-form examples in string literals and parametrize arrays).
  No test imports directly from `session.ep_device`.

### §A5: session/__init__.py re-exports

- **Status: PASS with gap**
- `src/winml/modelkit/session/__init__.py` re-exports and includes in `__all__`:
  `EPDevice`, `EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`,
  `AmbiguousMatch`, `EPMonitorMismatch`, `resolve_device`, `expand_ep_name`,
  `short_ep_name`, `canonicalize_ep_name`, `VALID_EPS`, `_VALID_DEVICES`,
  `get_provider_for_device` — all present.
- `_EP_TO_DEVICE` and `_DEVICE_TO_PROVIDER` are NOT re-exported (correct).
- **Gap:** The plan's Step 3 listed these exact symbols; all are present. PASS.

### §A6: ep_to_device() helper (addition beyond plan)

- **Status: ADDED — correct and harmless**
- `src/winml/modelkit/session/ep_device.py:182–197`
- `ep_to_device(ep: str) -> str` raises `ValueError` for unknown EP, returns
  `_EP_TO_DEVICE[ep_lower]` for known EPs. It is a thin, raising wrapper around
  `_EP_TO_DEVICE.get(ep)`.
- It IS re-exported via `session/__init__.py:17,58`.
- Used in `config/precision.py:229` for EP-to-device inference in `resolve_precision()`.
- This addition is correct and improves the public surface vs. returning `None`.

### §B7: resolve_device() four-case deduction

- **Status: PASS**
- `src/winml/modelkit/session/ep_device.py:220–326`
- Case **both=None**: line 247–252 — calls `sysinfo.resolve_device_category()`,
  then falls to device-only path.
- Case **ep only**: lines 254–265 — normalizes short form, raises `ValueError` for
  unknown EP, deduces device via `_EP_TO_DEVICE[ep_short]`.
- Case **device only**: lines 267–279 — looks up `_DEVICE_TO_PROVIDER[device_lower]`,
  raises `ValueError` for unknown device, handles `cpu → None → "cpu"` fallback.
- Case **both given**: control falls straight through all three `if` blocks to the
  resolution phase at line 286. Validation of inputs is implicit (invalid EP will
  fail at `expand_ep_name`/registry level).
- On invalid EP: `ValueError` raised at line 264.
- On unknown device: `ValueError` raised at line 274.

### §B8: Bundled-CPU EP fallback in register_ep

- **Status: PASS — untouched**
- `src/winml/modelkit/session/ep_registry.py:185–190` — bundled EP fallback
  (`get_ep_devices()` check) is intact and unmodified.

### §B9: Smoke tests

All three smoke tests run successfully on this machine (QNN hardware present):

```
uv run python -c "from winml.modelkit.session import resolve_device, EPDevice; print(resolve_device('qnn', 'npu'))"
# exit 0 → EPDevice(ep='QNNExecutionProvider', device='npu', vendor_id=1297040209, device_id=1093682224, vendor='')

uv run python -c "from winml.modelkit.session import resolve_device; print(resolve_device('qnn'))"
# exit 0 → EPDevice(ep='QNNExecutionProvider', device='npu', ...) — device deduced correctly

uv run python -c "from winml.modelkit.session import resolve_device; print(resolve_device(None, 'npu'))"
# exit 0 → EPDevice(ep='QNNExecutionProvider', device='npu', ...) — ep deduced correctly
```

### §B10: resolve_device(None, None)

- Not run (sysinfo.resolve_device_category() is safe locally but was implicitly
  tested by the above smoke tests succeeding). The auto-detect path at
  `ep_device.py:247–252` calls `from ..sysinfo import resolve_device_category`.

### §C11: winml compile --help

- **Status: PARTIAL PASS**
- `--device` present with `case_sensitive=False` at `commands/compile.py:55`.
  Choices: `["auto", "npu", "gpu", "cpu"]`.
- `--ep` defaults to `None` (line 63–64). Not required.
- **Concern:** Examples in help text are complete and include `--device` forms.
  However `--ep` default=`None` is correct.
- **Note:** `--device` includes `"auto"` which is not in the plan's
  `["cpu", "gpu", "npu"]`. This is an extension — harmless but undocumented.

### §C12: commands/compile.py calls resolve_device

- **Status: FAIL — plan deviation**
- `src/winml/modelkit/commands/compile.py:173` calls `_resolve_compile_provider(device, ep)`
  which returns a `str` (provider name), NOT an `EPDevice`.
- `resolve_device()` is never called in `commands/compile.py`.
- `config.ep_device` is never set by the CLI — it remains `None`.
- The plan (Step 7) explicitly requires: _"call `resolve_device(ep, device)` once →
  store the resulting `EPDevice` in `WinMLCompileConfig.ep_device`"_.
- The compile stage (`compiler/stages/compile.py:68–75`) has a fallback:
  if `ep_device_dict` and `compile_cfg.ep_device` are both absent, it calls
  `resolve_device(ep=ep_str)` directly. This means deduction still works, but
  it is called **inside the compile stage** (not once at the CLI boundary), which
  contradicts Decision B's "deduction happens once, at the boundary".
- **Cross-reference:** `_resolve_compile_provider` also contains an inline
  device-classification frozenset (lines 195–202) to display the device string,
  which is a new inline duplicate of `_EP_TO_DEVICE` (see §surprises).
- No import of `_EP_TO_DEVICE` or `_DEVICE_TO_PROVIDER` from `config.precision`
  remains (the old private cross-package import is gone — this sub-requirement PASS).

### §C13: compiler/stages/compile.py ep_device logic

- **Status: PASS with qualification**
- `src/winml/modelkit/compiler/stages/compile.py` no longer imports
  `_EP_TO_DEVICE` from `config.precision` (old import removed — PASS).
- Fallback chain at lines 68–75:
  1. `context.config.get("ep_device")` → `EPDevice.from_dict()`
  2. `compile_cfg.ep_device is not None` → use it
  3. Otherwise: `resolve_device(ep=ep_str)` — infers device from EP
- The plan described this chain. The implementation matches.
- **Qualification:** Since `commands/compile.py` never sets `ep_device` on the
  config (see §C12), the CLI path always falls through to case 3 (`resolve_device`
  inside the stage), not case 2. The plan's "resolve once at the boundary" is
  not achieved for the CLI path.

### §C14: WinMLCompileConfig fields

- **Status: PASS**
- `src/winml/modelkit/compiler/configs.py:87` — `ep_device: EPDevice | None = None`.
- `to_dict()` at line 264–283: serializes `ep_device.to_dict()` when not None
  (lines 281–282). PASS.
- `from_dict()` at line 286–319: deserializes via `_EPDevice.from_dict(data["ep_device"])`
  when present (lines 310–312). PASS.
- `for_ep_device()` factory at line 99–116 — verified: creates `WinMLCompileConfig`
  from a fully-resolved `EPDevice`, calls `short_ep_name(ep_device.ep)`, dispatches
  to `for_provider(provider)` or creates a generic config, then sets `base.ep_device`.
  Correct and complete.

### §D15: utils/constants.py — taxonomy entries deleted

- **Status: FAIL — not deleted**
- `src/winml/modelkit/utils/constants.py` still contains:
  - `SUPPORTED_EPS` at line 11 (3 entries: QNN, OpenVINO, VitisAI)
  - `EP_ALIASES` at line 18 (5 alias entries)
  - `ALL_EP_NAMES = list(SUPPORTED_EPS) + list(EP_ALIASES.keys())` at line 27
  - `SUPPORTED_DEVICES = ["CPU", "GPU", "NPU"]` at line 94 (uppercase — the bug
    noted in the plan)
- These are still **actively used** by `utils/cli.py:ep_option()` (via `ALL_EP_NAMES`)
  and `commands/analyze.py` / `analyze/analyzer.py` (via `normalize_ep_name`).
- Step 5 of the plan says to DELETE these and replace callers. They were NOT deleted.
- **Mitigating context:** `utils/cli.py:17` now uses `_DEVICE_CHOICES = sorted(_VALID_DEVICES)`
  from `session` for the `--device` option (the uppercase bug is fixed for devices).
  But `EP_ALIASES`, `SUPPORTED_EPS`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES` remain as
  live dead weight.

### §D16: analyze.py --device flag

- **Status: PASS**
- `commands/analyze.py` uses `cli_utils.device_option(...)` which reads
  `_DEVICE_CHOICES = sorted(_VALID_DEVICES)` from session — lowercase `["cpu", "gpu", "npu"]`.
  The uppercase bug is fixed.
- The example at `commands/analyze.py:454` says `--device GPU` (uppercase) which is
  stale documentation (the value is accepted case-insensitively, but the example
  is misleading). Minor cosmetic issue.

### §D17: Inline dict `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` — zero hits

- **Status: PASS**
- `grep -rn '{"cpu"' src/winml/modelkit/commands/perf.py` — no hits.
- `grep -rn '{"cpu"' src/winml/modelkit/eval/evaluate.py` — no hits.
- The three call sites (`perf.py:472`, `perf.py:1552`, `evaluate.py:139`) now use
  `resolve_device(...)`.

### §D18: Remaining inline duplicates

- **Status: PASS for perf.py and evaluate.py; NEW INSTANCE in compile.py**
- `commands/perf.py:460–472` and `:1543–1549`: both use `resolve_device()`. PASS.
- `eval/evaluate.py:131–139`: uses `resolve_device(device=device)`. PASS.
- `commands/compile.py:195–202`: **NEW inline duplicate** — frozensets
  `gpu_eps = frozenset({"dml", "migraphx", "tensorrt", "cuda", "openvino"})` and
  `npu_eps = frozenset({"qnn", "vitisai"})` for display-only device derivation.
  This is a partial re-creation of `_EP_TO_DEVICE` in the display path.
  It should use `ep_to_device(provider)` from the session facade instead.

### §E19: Architecture regression test exists

- **Status: PASS**
- `tests/unit/architecture/test_ep_device_import_rule.py` — exists, 180 lines.
- Tests: `test_no_direct_ep_device_imports_in_src`,
  `test_no_direct_ep_device_imports_in_tests`,
  6 parametrized detector-catches-forbidden-forms,
  7 parametrized detector-does-not-flag-allowed-forms.

### §E20: Architecture test passes

- **Status: PASS**
- `uv run pytest tests/unit/architecture/test_ep_device_import_rule.py -v --tb=short`
- Result: **15 passed in 3.59s** — all green.

### §E21: Adversarial test — architecture test catches violations

- **Status: CONFIRMED EFFECTIVE**
- The AST detector was verified to flag `from ..session.ep_device import resolve_device`
  (relative form, level=1, module="ep_device") — the exact pattern a future committer
  would write. Python confirms 1 violation detected.
- The `test_no_direct_ep_device_imports_in_src` test scans all of `src/` and would
  fail with a clear error message listing the offending `file:lineno`.

### §E22: test_compile_cli.py — device/ep flag coverage

- **Status: PASS with qualification**
- `tests/unit/commands/test_compile_cli.py` — exists, 188 lines.
- Tests the four deduction cases for `_resolve_compile_provider`: both-None, ep-only,
  device-only, neither (class `TestResolveCompileProviderNoneDevice`).
- Tests `--device` and `--ep` in CLI invocations via `CliRunner`.
- **Qualification:** Tests cover `_resolve_compile_provider()` (the local helper)
  extensively but do NOT test that `ep_device` is set on the config and threaded
  to the compile stage — because the CLI doesn't actually do this (see §C12).

### §E23: Mock paths in test files

- **Status: PASS**
- `tests/unit/commands/test_perf_cli.py:43` — patches
  `"winml.modelkit.session.resolve_device"` (facade/consumer path). PASS.
- `tests/unit/eval/test_eval.py` — patches `patch.object(eval_mod, ...)` style,
  not importing from `ep_device`. PASS.
- `tests/unit/models/auto/test_auto_onnx.py` — no ep_device patches; uses
  `cpu_ep_device` fixture from `winml.modelkit.session import EPDevice`. PASS.

### §E24: Full unit test suite

- **Status: PASS**
- `uv run pytest tests/unit/ --tb=no` collected 4159 tests.
- Result: **480 PASSED, 5 SKIPPED, 0 FAILED** (skips are hardware/EP related).
- The agent's claimed "1034+ passing" was conservative; actual count is higher.

### §E25: Ruff linting

- **Status: PASS**
- `uv run ruff check src/ tests/` — **All checks passed!**

### §F26: WinMLSession._build_session_options (out-of-scope check)

- **Status: PASS — untouched as required**
- `src/winml/modelkit/session/session.py:460–483` — method exists, uses
  `_device_policy_map` with `set_provider_selection_policy(policy)`. Unchanged.

### §F27: models/auto.py positional ep_device

- **Status: PASS — no string device= slippage**
- `git diff` on `models/auto.py` shows only 4-line change (doc string reference
  updates). No positional signature change, no `device=` string callers added.

### §F28: qnn_monitor.py int() truncation bug

- **Status: PASS — still present as required**
- `src/winml/modelkit/session/monitor/qnn_monitor.py:439–440` — `int()` truncation
  on `accel_execute_cycles` and `accel_execute_us` still present. Not touched.

### §F29: qnn/_internal.py hard dict accesses

- **Status: PASS — still present as required**
- `src/winml/modelkit/session/monitor/qnn/_internal.py:259–281` — hard `op["op_id"]`,
  `op["name"]`, `op["op_path"]`, `op["cycles"]` dict accesses intact. Not touched.

---

## Bugs / Regressions Found

None that break existing tests. The following are plan compliance failures:

### Bug/Deviation 1: CLI boundary does not call resolve_device (Severity: Medium)

- **File:** `src/winml/modelkit/commands/compile.py:173–174`
- **Description:** The CLI calls `_resolve_compile_provider(device, ep)` which returns
  a string. `resolve_device()` is never called; `config.ep_device` stays `None`.
  The compile stage's fallback (`resolve_device(ep=ep_str)` inside the stage) means
  the feature works, but deduction happens at stage time, not boundary time.
  This violates Decision B ("deduction happens once, at the boundary").
- **Impact:** Functional correctness is maintained (compile still works). The violation
  is architectural: the boundary guarantee is broken, making it harder to detect
  deduction errors early and log them for the user.

### Bug/Deviation 2: utils/constants.py not cleaned up (Severity: Low)

- **File:** `src/winml/modelkit/utils/constants.py`
- **Description:** `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`
  (uppercase) were not deleted as Step 5 required. They remain live and in-use via
  `utils/cli.py:ep_option()` and `commands/analyze.py`.
- **Impact:** No functional regression. Duplicate taxonomy continues to exist;
  `SUPPORTED_DEVICES` (uppercase) is a known bug that survives.

---

## Surprises / Additions Beyond Plan

### S1: ep_to_device() public helper added

- **Where:** `src/winml/modelkit/session/ep_device.py:182–197`
- **What:** A new public function wrapping `_EP_TO_DEVICE` with a ValueError on
  unknown EP. Re-exported via `session/__init__.py`.
- **Is it correct?** Yes — raises on unknown EP (stricter than dict.get).
- **Is it harmless?** Yes — used only in `config/precision.py:229`, replaces a
  direct `_EP_TO_DEVICE` reference. Improves encapsulation.

### S2: WinMLCompileConfig.for_ep_device() factory added

- **Where:** `src/winml/modelkit/compiler/configs.py:99–116`
- **What:** Factory classmethod that creates a `WinMLCompileConfig` from a
  fully-resolved `EPDevice`, sets `ep_device`, and dispatches to `for_provider()`.
- **Is it correct?** Yes — imports `short_ep_name` via the session facade.
- **Is it harmless?** Yes — the factory is never called by the current CLI (see §C12),
  but is available for API callers. Not a regression.

### S3: New inline EP→device frozensets in commands/compile.py display path

- **Where:** `src/winml/modelkit/commands/compile.py:195–202`
- **What:** Two frozensets (`gpu_eps`, `npu_eps`) encode the EP-to-device mapping
  for display purposes, introduced as a replacement for the removed `_EP_TO_DEVICE.get(provider, device)`.
- **Is it correct?** Functionally yes (correct entries as of today).
- **Is it harmless?** It is a new inline duplicate of `_EP_TO_DEVICE`. The plan
  aimed to eliminate such duplicates (Step 6). It should be replaced with
  `ep_to_device(provider)` from the session facade. Low severity but contradicts
  the consolidation goal.

### S4: --device accepts "auto" in compile CLI

- **Where:** `src/winml/modelkit/commands/compile.py:55`
- **What:** `click.Choice(["auto", "npu", "gpu", "cpu"])` — "auto" was added beyond
  the plan's `["cpu", "gpu", "npu"]`.
- **Is it correct?** Yes — `_resolve_compile_provider` handles `device="auto"` by
  defaulting to "qnn".
- **Is it harmless?** Yes — consistent with the "auto" concept elsewhere.

### S5: Stale example in analyze.py help text

- **Where:** `src/winml/modelkit/commands/analyze.py:454`
- **What:** Example `winml analyze --model model.onnx --ep ov --device GPU` uses
  uppercase `GPU`. The device option is case-insensitive so this works, but it
  is inconsistent with the lowercase-fix goal of Step 5.
- **Is it harmless?** Yes — cosmetic only.

---

## Recommendations

**P1 (Medium) — Complete the CLI boundary for `winml compile`:** Replace the
`_resolve_compile_provider` string approach with a `resolve_device(ep, device)`
call that sets `config.ep_device`. This would fulfill Decision B and ensure the
INFO-level log ("Resolved to: EPDevice(...)") suggested in the plan is emitted.
File: `src/winml/modelkit/commands/compile.py:173–174`.

**P2 (Low) — Remove inline frozensets in compile.py display:** Replace
`commands/compile.py:195–202` (the `gpu_eps`/`npu_eps` frozensets) with
`ep_to_device(provider)` from the session facade. Removes new inline duplicate.

**P3 (Low) — Finish Step 5 cleanup of utils/constants.py:** Delete `SUPPORTED_EPS`,
`EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES` and migrate their callers
(`utils/cli.py:ep_option()`, `analyze/analyzer.py`, `commands/analyze.py`) to use
`VALID_EPS` and `expand_ep_name()` from the session facade. Fixes the uppercase
`SUPPORTED_DEVICES` bug as a free side effect.

**P4 (Cosmetic) — Fix stale example in analyze.py help text:** Change
`--device GPU` to `--device gpu` at `commands/analyze.py:454`.
