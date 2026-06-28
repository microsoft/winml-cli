# src/winml/modelkit/analyze/analyzer.py

## TL;DR
A single-hunk surgical change inside `ONNXStaticAnalyzer.analyze_from_proto` replaces a hardcoded
3-element list of NPU EPs (QNN / OpenVINO / VitisAI) with a structural lookup
`sorted(eps_for_device("npu"))` from the new `session.ep_device` catalog. This is a downstream
migration: the analyzer now derives its "all NPU EPs" universe from the single source of truth
(`EP_DEVICE_SPECS`) rather than carrying its own duplicate list. No session-construction code in
the analyzer was touched, and no private session symbols are imported.

## Diff metrics
- 13 insertions / 12 deletions (net +1), all in one hunk near line 663.
- Touches one method only: `ONNXStaticAnalyzer.analyze_from_proto` (the no-`ep` branch).
- Adds one lazy import: `from ..session import eps_for_device` (function-scoped, not module-top).

## Role before vs after
Before: the analyzer hardcoded `["QNNExecutionProvider", "OpenVINOExecutionProvider",
"VitisAIExecutionProvider"]` whenever the caller did not pin an EP — a per-call literal whose
membership had to be kept in sync with the EP taxonomy by hand. This violates the project's
Cardinal Rule "No Hardcoded Logic" for model/EP names.

After: the same default fan-out is derived structurally — every catalog entry tagged
`device == "npu"` is included automatically, in deterministic `sorted()` order. Adding a new
NPU EP to `EP_DEVICE_SPECS` (the commit's new single source of truth) now propagates into the
default analyze loop with no edit here.

## Symbol-level changes
- `ONNXStaticAnalyzer.analyze_from_proto` (only method changed):
  - The branch `if ep_normalized is None` no longer instantiates a literal list; it imports
    `eps_for_device` lazily and assigns `eps_to_analyze = sorted(eps_for_device("npu"))`.
  - The `logger.info` message string changed from "all supported EPs" to "all NPU-capable EPs"
    — slight semantic narrowing (was always NPU-only anyway, but now self-documenting).
- No other symbol added, removed, renamed, or re-signed. `normalize_ep_name`, `RuntimeChecker`
  construction, `InformationEngine` wiring, and `AnalyzeResult` / `AnalysisResult` dataclasses
  are unchanged.

## Behavior / contract changes
- Default EP set is now `sorted(eps_for_device("npu"))` — currently the same three EPs
  (QNN, OpenVINO, VitisAI) but in sorted (alphabetic) order. The pre-commit literal order was
  QNN-first; post-commit the order is `OpenVINOExecutionProvider, QNNExecutionProvider,
  VitisAIExecutionProvider`. Any test or downstream consumer that depends on iteration order
  (e.g. which EP populates `check_op_results` first, or `dict` key order in
  `AnalysisOutput.aggregate`) sees a different ordering.
- Default device unchanged (`"NPU"`).
- The special-case `if current_ep == "VitisAIExecutionProvider": run_unknown_op_for_ep = False`
  at lines 716–717 is untouched — VitisAI is still string-matched here, a residual hardcoding
  flagged by the project rules. The accompanying TODO comment ("add VitisAIExecutionProvider
  back once non-QDQ data is ready") was not addressed by this commit.

## Cross-file impact
- New dependency edge: `analyze/analyzer.py` → `modelkit.session` (public re-export of
  `eps_for_device` from `session/__init__.py` lines 22, 68). Verified to be a public symbol
  in `__all__`; no private boundary is crossed.
- The session subsystem is the only new import. No `EPDevice`, `WinMLSession`, `resolve_device`,
  or any monitor symbol is referenced — the analyzer does not yet participate in the new
  EPDevice flow at the construction layer (it still passes raw `ep` / `device` strings into
  `RuntimeChecker`).
- The fan-out depends on the QNN-NPU / OpenVINO-NPU / VitisAI-NPU entries existing in
  `EP_DEVICE_SPECS`. If a future cleanup removes one (or repoints it to a non-NPU device),
  the analyzer's default scope changes silently — no test asserts the equivalence to the old
  literal.

## Risks / subtleties
- **Lazy import in a hot loop method.** `from ..session import eps_for_device` lives inside
  `analyze_from_proto`, executed every call. The session package's `__init__.py` is heavy
  (it pulls `WinMLSession`, all four monitor modules, qairt session, the EP registry, the
  full ep_device catalog). First analyze call after process start now pays a much larger
  one-time import cost than before, and that cost is paid even when the caller supplies
  `ep=` (because the import is inside the `if ep_normalized is None` branch — actually
  *only* paid in the no-EP branch, which mitigates this; but session is already imported
  elsewhere in any realistic CLI run, so the effective cost is near-zero second time).
- **Order change is observable.** Prior order put QNN first (the production NPU EP on
  Snapdragon). Now alphabetic order puts OpenVINO first. Logs, JSON output dict keys, and
  any "first EP" heuristic downstream will differ. The `AnalysisOutput.aggregate` call
  on line 748 receives `check_op_results` as a `dict` — insertion-ordered in Python 3.7+,
  so consumers iterating it will see OpenVINO first now.
- **Crash bug is elsewhere.** Per
  `docs/design/session/2026-05-13-t6-analyze-crash-diagnostic.md`, the analyze-loop crash
  fixed by this commit was an `ort.register_execution_provider_library` double-registration
  in `winml.py:WinML` vs `WinMLEPRegistry`. The fix is the *symmetric defensive guard* in
  `winml.py` and `session/ep_registry.py` — **not** in `analyzer.py`. This file's edit is
  an unrelated taxonomy cleanup.
- **Hardcoding rule still partially violated.** Line 716 still does
  `if current_ep == "VitisAIExecutionProvider"` — a hardcoded EP name. The commit chose
  to address only the literal list in the `ep is None` branch; the VitisAI run_unknown_op
  carve-out remains a string compare.

## Open questions / TODOs surfaced
- The existing TODO at line 713–714 ("add VitisAIExecutionProvider back once non-QDQ data is
  ready, and run_unknown_op is supported for QDQ ops") survives unchanged. With the analyze
  default now structurally tied to the catalog, the VitisAI string-compare carve-out is the
  last remaining hardcoded EP literal in this file — worth lifting onto an EP-capability
  attribute on `EPDeviceSpec` if catalog drift becomes a maintenance concern.
- Should the analyzer accept an `EPDevice` (or `Iterable[EPDevice]`) instead of paired
  `ep: str | None, device: str | None` strings? The rest of the commit migrated
  CLI boundaries (`commands/perf.py`, `eval/evaluate.py`, `compiler/stages/compile.py`,
  `models/auto.py`) to `EPDevice` but the analyze surface still takes loose strings.
  Not addressed here — explicit follow-on opportunity for an end-to-end EPDevice flow.
- Default-device behaviour is still a string literal `"NPU"` on line 676 — could also be
  derived from the catalog (e.g. `default_device_for_ep` or the first NPU spec) once the
  surface accepts `EPDevice`.
