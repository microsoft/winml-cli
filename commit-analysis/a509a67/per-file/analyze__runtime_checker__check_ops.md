# src/winml/modelkit/analyze/runtime_checker/check_ops.py

## TL;DR
A pure casing-fix migration: every occurrence of the EP literal `NvTensorRTRTXExecutionProvider`
(four spots — one error message, one `super().__init__` ep_name, one dict key in
`get_ep_checker`, two argparse `choices`/`help` strings) is renamed to
`NvTensorRtRtxExecutionProvider` to match `ort.get_all_providers()` casing and the new
session catalog. No structural / behavioural changes; the runtime EP-checker classes still
construct `ort.InferenceSession` directly via `EPChecker`, not `WinMLSession`.

## Diff metrics
- 5 insertions / 5 deletions (net 0), all single-token casing changes.
- Touches one class (`RTXChecker`) and one function (`get_ep_checker`) plus one parser
  (`build_parser`).
- No new imports, no new functions, no signature changes.

## Role before vs after
Before: the file used the casing `NvTensorRTRTX...` (all-caps RTX), which did not match what
`ort.get_all_providers()` actually returns. As the commit body documents under "Sysinfo +
taxonomy cleanup": *"NvTensorRtRtx casing bug fixed (verified via ort.get_all_providers())."*
Any code path that compared EP names case-sensitively would have failed to find this EP.

After: all five literals use the corrected `NvTensorRtRtx...` casing. The
`get_ep_checker` lookup table key, the `RTXChecker` `super().__init__` ep_name, and the
argparse `choices` list now match the canonical ORT name and the catalog's `VALID_EPS`.

## Symbol-level changes
- `RTXChecker.__init__` (lines 259–268):
  - Error message string `"NvTensorRTRTXExecutionProvider only supports GPU device type"` →
    `"NvTensorRtRtxExecutionProvider only supports GPU device type"`.
  - `super().__init__(ep_name="NvTensorRTRTXExecutionProvider", ...)` →
    `super().__init__(ep_name="NvTensorRtRtxExecutionProvider", ...)`.
- `get_ep_checker` (lines 271–297):
  - Dict key `"NvTensorRTRTXExecutionProvider": RTXChecker` → `"NvTensorRtRtxExecutionProvider":
    RTXChecker`. Callers passing the old casing now raise the dict's "Unsupported execution
    provider" `ValueError`.
- `build_parser` (lines 300+):
  - `argparse choices=[..., "NvTensorRTRTXExecutionProvider"]` → `[...,
    "NvTensorRtRtxExecutionProvider"]`.
  - Corresponding `help=` string substring updated to match.

No other symbol — `EPChecker`, `OpenVINONPUChecker`, `QNNNPUChecker`, `VitisAIChecker`,
`MIGraphXChecker`, `check_ops`, the module-level `winml.register_execution_providers(ort=True)`
call (line 41) — is touched.

## Behavior / contract changes
- **Breaking CLI input contract**: any external caller invoking the subprocess
  `python -m winml.modelkit.analyze.runtime_checker.check_ops --ep
  NvTensorRTRTXExecutionProvider ...` now fails argparse with an "invalid choice" error.
  The new accepted spelling is `NvTensorRtRtxExecutionProvider`. This is an
  intentional hard-break (the commit's Option A no-compat-shims posture).
- **Breaking programmatic key**: `get_ep_checker("NvTensorRTRTXExecutionProvider", "GPU")`
  now raises `ValueError`. Callers should pass `NvTensorRtRtxExecutionProvider`.
- The error message inside `RTXChecker` is informational only; the casing change there is
  for consistency.
- `EPChecker` itself was not modified: it still builds an `ort.InferenceSession` directly in
  `check_run` (line 108 of `ep_checker.py`) using `winml.add_ep_for_device()` for EP wiring.
  It does **not** route through `WinMLSession` or any session/monitor pipeline added in
  this commit.

## Cross-file impact
- The pre-commit casing `NvTensorRTRTX...` survives in zero callers (sweep across the repo
  confirms only this file referenced it). The renaming therefore needs no further
  ripple-fix beyond this file.
- The new spelling matches:
  - `ort.get_all_providers()` output (per commit message verification).
  - `_SHORT_TO_FULL` and `VALID_EPS` in `session/ep_device.py` (which is now the single
    source of truth for EP-name canonicalization).
  - The error spelling produced by `expand_ep_name("NvTensorRtRtx")` etc.
- The file does **not** import any new session symbol, does **not** use `EPDevice`,
  `resolve_device`, or `eps_for_device`. The argparse `choices` list is still a hardcoded
  5-tuple of EP literals — unlike `analyzer.py` which adopted the catalog. This is a real
  taxonomy violation by the project's "No Hardcoded Logic" rule, but the commit chose to
  preserve the explicit allowlist (consistent with `check_patterns.py`'s carve-out
  rationale, though here without a comment explaining why).

## Risks / subtleties
- **Silent dependency on `winml.register_execution_providers(ort=True)`** at line 41
  (module top). Per `docs/design/session/2026-05-13-t6-analyze-crash-diagnostic.md`, this is
  exactly the second-call path that crashed the analyze loop with `STATUS_DLL_NOT_FOUND` in
  the perf HF pipeline. The fix landed in `winml.py:WinML.register_execution_providers`
  (symmetric defensive guard), not here. So this file *depends* on the patched guard
  remaining in place; if that guard regresses, importing this module in a process that has
  already called `WinMLEPRegistry.register_ep` (the new perf/eval path) will native-crash
  again. The file has no defensive registration check of its own.
- **Hardcoded 5-EP `choices` list** in `build_parser` is a known taxonomy carve-out
  (matches the pattern in `check_patterns.py`). Future contributors adding a 6th NPU EP to
  the catalog will not see this site update automatically — an explicit-allowlist trap
  with no comment to flag the carve-out (unlike `check_patterns.py` which got the
  CARVE-OUT comment in this commit). Consider replicating that comment here.
- **Subprocess-spawn caveat from the diagnostic doc**: spawned children (via
  `ResilientRunner`) freshly import this module, run `winml.register_execution_providers`,
  and then build an `ort.SessionOptions` via `EPChecker._get_sess_options`. If the QNN/RTX
  DLL is not discoverable in the child, `winml.add_ep_for_device` "silently does nothing"
  (per the diagnostic), and `ort.ModelCompiler` then exits 127. The parent recovers via
  `BrokenProcessPool`. The casing fix does not change this; it remains a known recovery
  path.

## Open questions / TODOs surfaced
- The `choices` allowlist duplicates the catalog. Consider deriving it from
  `sorted(VALID_EPS - {"CPUExecutionProvider", "DmlExecutionProvider"})` or
  `sorted(eps_for_device("npu") | eps_for_device("gpu"))` — paralleling the analyzer's
  migration. If a carve-out is intentional, add the same `CARVE-OUT:` comment style used
  in `check_patterns.py` (the absence of that comment here is the most fragile part of
  this file post-commit).
- The wrapper classes (`QNNNPUChecker`, `OpenVINONPUChecker`, `VitisAIChecker`,
  `MIGraphXChecker`, `RTXChecker`) exist purely "as there is a bug with pytest in subprocess"
  — comment at line 221. The bug is not described elsewhere; it's not clear whether the
  EPDevice refactor obviates the need for these wrappers.
- The pre-condition guards in each wrapper's `__init__` (`device_type != GPU/NPU`) still
  encode EP→device routing as Python `if` statements rather than as `EPDeviceSpec` lookups.
  This is a duplicate of catalog logic and a candidate for follow-on simplification once
  the analyze surface accepts an `EPDevice` directly.
