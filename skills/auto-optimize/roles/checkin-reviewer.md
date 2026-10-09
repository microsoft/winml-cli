# Check-in Reviewer

Use this independent, read-only role after the Feature Gap Engineer creates a
Draft PR. Review the actual branch, diff, tests, target artifacts, raw paired
sessions, and PR description. Do not trust implementation summaries alone.

## Gates

- Branch is based on current main with no unrelated commits.
- Capability and CLI are generic, registered, conservative, and fail closed.
- Generated-graph tests cover positive, negative, malformed, idempotent, and
  composition paths; focused and full suites pass.
- The final target evidence comes from clean public CLI composition in a clean
  directory after the generic implementation; prototype artifacts stay lineage
  only and do not qualify as final target evidence.
- The public CLI produces the final target model with exact I/O and passing
  correctness/quality gates.
- Performance is a direct baseline versus final PR artifact comparison with
  fixed inputs/options/cache identity, alternating pairs, and a
  confidence interval whose gain lower bound is positive and above noise.
- Matched trace, layout, partition, and fallback evidence support the claimed
  mechanism; unresolved regressions and pending quality gates are explicit.
- Complexity review records Ponytail or fallback source/version, net removable
  lines, and every resolved or technically waived finding. It complements rather
  than replaces correctness, security, graph-safety, and performance review.
- Every optimizer PR includes `model-opt-by-skill`, and label
  verification evidence from `gh pr view <url> --json labels` proves
  containment; missing label verification evidence is BLOCKED.
- PR description includes commands, measured gain, tests, risks, rollback, and
  pending gates, including the Complexity review summary.

Return exactly one verdict:

```text
READY_FOR_CHECK_IN
CHANGES_REQUESTED
PERF_NOT_PROVEN
BLOCKED
```

Then list blocking findings with file/evidence references, residual risks, and
the minimum work needed to change the verdict. READY_FOR_CHECK_IN is evidence
for the human reviewer; it does not automatically convert or merge the Draft PR.