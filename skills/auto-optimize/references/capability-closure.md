# LLM Capability Closure

Use this review after each material leader and before stopping. Its purpose is
to find staged, inverse, and interaction opportunities that analyzer discovery
can miss. Do not enumerate capability combinations.

## Inputs

Read the leader ONNX and hash, residual producer/consumer graph facts,
baseline-to-leader graph and matched trace deltas, `winml optimize
--list-capabilities --verbose`, current `--check-optim` output, at most three
selected knowledge cases, and the prior closure ledger with tested outcomes.
Do not use a reference model or desired verdict.

## Review

Compare residual topology and unexplained provider work with registered
capability descriptions. Consider analyzer-reported matches, topology-compatible
but unreported capabilities, inverses of applied representation changes,
knowledge anchors, and interactions exposed by the new leader. Analyzer non-reporting is detector evidence only, never proof of inapplicability.

Rank by evidence, expected trace impact, and falsification cost. Promote at most
three entries to `PROBE_REQUIRED`; mark further plausible entries
`DEFERRED_BUDGET`. Do not infer a latency gain from graph shape.

For each entry return:

```text
capability and exact flag:
sources: analyzer | registry | residual topology | knowledge | prior delta
generic residual anchor:
eligibility: matched | contradicted | unknown; supporting facts
safety: established | unresolved; supporting facts
analyzer: reported | not reported
staging or interaction rationale:
cheapest explicit probe:
expected graph and provider-trace delta:
status: PROBE_REQUIRED | CLOSED_INELIGIBLE | CLOSED_ALREADY_TESTED | CLOSED_REGISTRY_ABSENT | DEFERRED_BUDGET
closure reason:
```

Use `PROBE_REQUIRED` when a registered capability is structurally plausible and
untested, even if `--check-optim` is silent. Unknown required facts trigger the
cheapest read-only inspection first. The main agent builds a fresh explicit
probe, verifies observed graph delta and public I/O, then applies correctness
and paired-performance gates. A weak enabler remains eligible when it may expose
a larger interaction.

`DEFERRED_BUDGET` takes precedence and requires `INSUFFICIENT_EVIDENCE`.
Otherwise `PROBE_REQUIRED` requires `MATERIAL_OMISSION_FOUND`. Return
`NO_MATERIAL_OMISSION` only when all entries are closed. Persist the
LLM-produced ledger in the run journal and report.