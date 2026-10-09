# Resume at the first unverified gate

Distinguish a new optimization search from validation of an existing candidate.
On resume, read the handoff and match model/input hashes, graph changes, provider
options, toolchain and cache identity to the current run. Reuse only completed
gates with matching evidence. Claims without evidence leave the gate unverified.

The planning router selects experiments. It is not a prerequisite to assessing
an already-supplied candidate: when asked to validate its correctness, run that
gate first. Do not invoke plan_hotspot.py merely because it appears earlier in
the skill, or manufacture hotspot evidence to satisfy it. Missing attribution
blocks a hotspot/performance claim, not a numerical correctness check.

| Entry state | Next action |
|---|---|
| New search or new hypotheses needed | Baseline evidence, then planning router |
| Candidate with structural/IO checks complete | Correctness before performance |
| Correctness passed, paired evidence incomplete | Paired performance and Perf Arbiter when needed |
| Candidate rejected by correctness | Retain failure; no performance, replay or promotion for it |
| All candidate gates passed | Clean replay, publication, validation, then promotion |
| Replay failed | Stop publication; retain logs, never substitute stale artifacts |

Changed inputs, model, options or toolchain invalidate affected downstream
evidence. Resume at the earliest invalidated gate. Return to planning only when
a new search/repair experiment is required, and respect the requested boundary:
a validation-only request does not authorize another optimization loop.

Retain model identities, source paths, node/tensor names, build commands and
artifacts only in run-local evidence, not bundled knowledge. Inspect actual CLI
help at the relevant gate; never invent flags.
