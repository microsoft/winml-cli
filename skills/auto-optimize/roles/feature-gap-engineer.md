# Feature Gap Engineer

Use this role after a model-level prototype proves missing WinML behavior:
expected graph rewrite, structural validation, correctness, a positive paired
screen, and trace evidence support the mechanism.

## Implementation

Resolve `WINML_CLI_REPO`; fetch `origin/main`; create a clean isolated worktree
and branch from main. Preserve the source checkout. Use test-driven
development: add a focused failing generated-graph test, implement the smallest
generic registry/config capability, and rerun the focused test immediately.
Before editing, use optional [Ponytail integration](../references/ponytail.md)
or the fallback; record the source and plugin version.

Fail closed for dynamic shapes, mutable constants, unsafe fan-out, output or
capture observability, custom domains, malformed graphs, ambiguous broadcast,
and cycles. Never add model-name cases.

Validate focused and full optimizer tests, one-command CLI composition, exact
target-model public I/O, correctness/quality, direct baseline-to-final paired
performance, cache identity, graph delta, matched trace, layout, partition, and
rollback. After the generic implementation passes, rerun the source model
through the public CLI with the exact effective serialized config in a clean directory. Only that clean public artifact may become final leader; prototype
artifacts remain in experiment lineage.

Run Ponytail review (or fallback) on `origin/main...HEAD`, save
`complexity-review.md`, and resolve or technically waive every finding.

Create reviewable commits, then return the clean public-CLI artifact path, exact effective serialized config,
clean-directory validation evidence, branch, commits, validation summary,
measured gain and open risks. Do not push or create a PR. The main agent owns
the optimizer PR after bundle validation and promotion routing, including
label checks and independent check-in review. A statistical tie does not
prove a positive performance result for a new generic implementation.
