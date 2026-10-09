# Optional Ponytail Integration

Upstream: https://github.com/dietrichgebert/ponytail

Ponytail minimizes implementation complexity; it does not create PRs or review
correctness, security, or performance. Those remain auto-optimize gates.

## Detect

Run `copilot plugin list`. If Ponytail is installed, record its plugin version
and use its namespaced skills. Do not install, update, or trust a new plugin
automatically. If absent or invocation fails, record `ponytail: unavailable` and
use the fallback below; Draft PR creation is not blocked.

## Before implementation

After you understand the root cause and real control flow, invoke
`/ponytail:ponytail full` for the Feature Gap task. Stop at the first applicable
rung:

1. Does the feature need to exist?
2. Does the WinML CLI codebase already provide the helper, registry path, or
   capability?
3. Does Python/ONNX stdlib behavior cover it?
4. Does the native WinML/ORT platform cover it?
5. Does an installed dependency cover it?
6. Only then write the minimum generic capability.

Never simplify away trust-boundary validation, graph safety, error handling,
security, correctness tests, target-model quality, cache identity, or paired
performance evidence.

## Before Draft PR

After normal review and all tests/perf gates pass, invoke
`/ponytail:ponytail-review` on `origin/main...HEAD`. It reviews complexity only
and may return `delete`, `stdlib`, `native`, `yagni`, or `shrink` findings.
Technically evaluate each finding; Ponytail does not understand ONNX semantics
or provider evidence automatically. After an accepted change, rerun affected
tests and any correctness/perf gate whose graph or runtime behavior changed.

Write `complexity-review.md` containing:

```text
source: ponytail | fallback
plugin version: version | unavailable
verdict: Lean already. Ship. | findings
net removable lines:
resolved findings:
waived findings and technical reasons:
```

## Fallback

Run the same seven-rung check yourself and review the diff for unnecessary
files, dependencies, wrappers, one-implementation abstractions, speculative
configuration, and duplicated helpers. Preserve all safety and evidence gates.