# src/winml/modelkit/analyze/analyzer.py

## TL;DR
A single-hunk surgical change inside `ONNXStaticAnalyzer.analyze_from_proto` replaces a hardcoded
3-element list of NPU EPs (QNN / OpenVINO / VitisAI) with a structural lookup
`sorted(eps_for_device("npu"))` from the `session.ep_device` catalog. The analyzer's default
fan-out is now derived from the single source of truth (`EP_DEVICE_SPECS`) rather than carrying
its own duplicate literal. No session-construction code in the analyzer was touched, and no
private session symbols are imported — the public `eps_for_device` re-export is consumed via
`from ..session import eps_for_device` (function-scoped, lazy).

## Diff metrics
- 7 insertions / 6 deletions (net +1), all in one hunk near line 667.
- Touches one method only: `ONNXStaticAnalyzer.analyze_from_proto` (the `ep_normalized is None` branch).
- Adds one lazy import: `from ..session import eps_for_device` (function-scoped).
- This hunk is byte-identical to the same hunk in commit `a509a67` — the v2.9 squash carries it forward unchanged.

## Role before vs after
Before: the analyzer hardcoded `["QNNExecutionProvider", "OpenVINOExecutionProvider",
"VitisAIExecutionProvider"]` whenever the caller did not pin an EP — a per-call literal whose
membership had to be kept in sync with the EP taxonomy by hand. This violates the project's
Cardinal Rule "No Hardcoded Logic" for EP names.

After: the same default fan-out is derived structurally — every catalog entry tagged
`device == "npu"` is included automatically, in deterministic `sorted()` order. Adding a new NPU
EP to `EP_DEVICE_SPECS` (the v2.9 unified-source catalog) now propagates into the default
analyze loop with no edit here.

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
- Default EP set is now `sorted(eps_for_device("npu"))` — currently the same three EPs (QNN,
  OpenVINO, VitisAI) but in alphabetic order. The pre-commit literal order was QNN-first;
  post-commit the order is `OpenVINOExecutionProvider, QNNExecutionProvider,
  VitisAIExecutionProvider`. Anything that depends on iteration order (e.g. which EP populates
  `check_op_results` first, or `dict` key order in `AnalysisOutput.aggregate`) sees a different
  ordering.
- Default device unchanged (`"NPU"`).
- The special-case `if current_ep == "VitisAIExecutionProvider": run_unknown_op_for_ep = False`
  carve-out further down (around line 716–717) is untouched — VitisAI is still string-matched
  here, a residual hardcoded literal flagged by the project rules.

## Cross-file impact
- New dependency edge: `analyze/analyzer.py` → `modelkit.session` (public re-export of
  `eps_for_device`, listed in `session/__init__.py` `__all__`). No private boundary crossed.
- The session subsystem is the only new import. No `EPDevice`, `WinMLSession`, `resolve_device`,
  or monitor symbol is referenced — the analyzer still passes raw `ep` / `device` strings into
  `RuntimeChecker` and does not yet participate in the new EPDevice flow at the construction
  layer.
- The fan-out depends on the QNN-NPU / OpenVINO-NPU / VitisAI-NPU entries existing in
  `EP_DEVICE_SPECS`. If a future cleanup removes one (or repoints it to a non-NPU device), the
  analyzer's default scope changes silently — no test asserts the equivalence to the old literal.

## Risks / subtleties
- **Lazy import in a hot-loop method.** `from ..session import eps_for_device` lives inside
  `analyze_from_proto`. The session package's `__init__.py` is heavy (it pulls `WinMLSession`,
  all monitor modules, qairt session, the EP registry, and the full ep_device catalog). The
  first analyze call after process start pays a one-time import cost; subsequent calls benefit
  from module caching.
- **Order change is observable.** Prior order put QNN first (the production NPU EP on
  Snapdragon). Now alphabetic order puts OpenVINO first. Logs and JSON output dict keys
  insertion-ordered by Python 3.7+ semantics will differ.
- **Hardcoded VitisAI carve-out still present** at line 716. The commit only addressed the
  literal list in the `ep is None` branch.

## Open questions / TODOs surfaced
- Should the analyzer accept an `EPDevice` (or `Iterable[EPDevice]`) instead of paired
  `ep: str | None, device: str | None` strings? The rest of the v2.9 refactor migrated CLI
  boundaries (`commands/perf.py`, `commands/build.py`, `commands/compile.py`) toward
  `EPDeviceTarget`-driven flows, but the analyze surface still takes loose strings.
- Default-device behaviour is still a string literal `"NPU"` on line 687 — could also be
  derived from the catalog (e.g. via `default_device_for_ep`) once the surface accepts an
  `EPDeviceTarget`.

## Simplification opportunities
- The lazy import inside `analyze_from_proto` could move to module top since the session
  package is already imported by virtually every realistic CLI entry point (perf, build,
  compile). The "function-scoped lazy import" pattern here is defensive without a clear
  payoff — module-top would be one line shorter and clearer.
- The `if ep_normalized is None / else` split could collapse to
  `eps_to_analyze = [ep_normalized] if ep_normalized else sorted(eps_for_device("npu"))` with
  the `logger.info` branched on the same condition.
- The `device_to_use = device if device is not None else "NPU"` fallback duplicates the
  catalog's default-device knowledge; consider `default_device_for_ep(eps_to_analyze[0])` so
  the analyzer's defaults are also catalog-derived.
- The lingering `if current_ep == "VitisAIExecutionProvider"` carve-out at line 716 is the
  last hardcoded EP literal in this file and a candidate for an `EPDeviceSpec` capability
  flag (e.g. `supports_unknown_op: bool`).
