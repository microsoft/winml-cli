# src/winml/modelkit/analyze/pattern/check_patterns.py

## TL;DR
A docs-only change: a 4-line comment block added inside the `argparse` `--ep` definition of the
CLI subcommand `parse_and_check`. The comment is a "CARVE-OUT" notice telling future maintainers
**not** to migrate this site to `eps_for_device("npu")` — the curated allowlist
`["QNNExecutionProvider", "OpenVINOExecutionProvider"]` is intentional. Zero behavioural change.

## Diff metrics
- 4 insertions / 0 deletions, all comment lines, inserted between `required=True,` and
  `choices=[...]` at line 331–334.
- Touches one function only: `parse_and_check()` at line 301.
- No code, runtime, parser semantics, or test surface affected.

## Role before vs after
Before: the subprocess CLI tool's `--ep` argument enforced a hand-picked allowlist of two EPs
(QNN, OpenVINO). The reason was implicit — readers might assume it was an oversight or
historical accident.

After: the same allowlist is now explicitly justified as a *carve-out*. The comment establishes
that this is a tested-validation boundary, not a candidate for catalog-derived auto-expansion.
This is the inverse partner of the analyzer change: `analyzer.py` adopted the catalog,
`check_patterns.py` deliberately rejected adopting the catalog and recorded that decision in
code where the next refactor will see it.

## Symbol-level changes
None — the parser's `--ep` argument keeps the same `choices=["QNNExecutionProvider",
"OpenVINOExecutionProvider"]`, same `required=True`, same `help=` string. No
import, class, function, or signature changed.

## Behavior / contract changes
Nothing executable changed. CLI behaviour, exit codes, validation, and test outcomes are
byte-identical to the pre-commit version. The contract for `python -m
winml.modelkit.analyze.pattern.check_patterns --ep ...` is unchanged.

## Cross-file impact
- Documentation contract only: the comment now ties future maintenance to the session
  catalog. A future contributor running a sweep against `eps_for_device("npu")` or
  `EP_DEVICE_SPECS` knows to skip this site.
- The file still does `from ..runtime_checker.ep_checker import EPChecker` (line 35) and
  `winml.register_execution_providers(ort=True)` (line 38) at module-import — the module
  remains a subprocess entrypoint and does not touch `WinMLSession` or any private session
  symbol.

## Risks / subtleties
- **Comment is a contract, not enforcement.** Nothing in the codebase prevents a future
  contributor from running a mechanical refactor and replacing the literal list with
  `sorted(eps_for_device("npu"))` — the only guard is the comment itself. If the carve-out
  is load-bearing (validation hasn't been run against VitisAI), the safer fix is an
  assertion or test that the allowlist matches the documented set.
- **Drift between this file and `runtime_checker/check_ops.py`.** The sibling tool
  `check_ops.py` accepts 5 EPs (QNN, OpenVINO, VitisAI, MIGraphX, NvTensorRtRtx) in its
  `--ep` parser, and was updated in this same commit for the `NvTensorRtRtx` casing fix.
  `check_patterns.py` keeps a 2-EP allowlist. The "untested against VitisAI" rationale
  in the comment matches `check_ops.py`'s broader allowlist only inconsistently — pattern
  testing has stricter validation requirements than op testing.
- **No test asserts the comment's claim.** The carve-out reason ("not validated against
  VitisAI / future NPU EPs") is an assertion in prose, not an executable check. If the
  pattern-test suite is later run on VitisAI and passes, the comment becomes stale silently.

## Open questions / TODOs surfaced
- Should the carve-out be encoded structurally? E.g. an attribute on `EPDeviceSpec` like
  `pattern_tested: bool` that this CLI would filter on, so the rationale is testable rather
  than commentary. Until then, the comment-as-contract pattern is fragile.
- The comment cites `eps_for_device("npu")` and `EP_DEVICE_SPECS` by name — a forward
  reference to the new session catalog. If those symbols are ever renamed, this comment
  silently goes stale (`grep` for the names is the only safety net).
- The pattern-test path was not part of the `t6-analyze-crash-diagnostic` chain (that crash
  was in the runtime_checker query path), so this file did not need behavioural fixing — the
  comment is purely an EP-taxonomy-sweep follow-up to document a non-migration decision.
