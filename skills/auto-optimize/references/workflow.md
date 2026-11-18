# Bounded workflow commands

Resolve scripts/workflow.py against the installed skill root. No new runtime
or dependency is required. Use its JSON diagnostics; nonzero exit means blocked.

1. Run `workflow.py status --state <run/state.json>` before resuming. Missing or
changed evidence returns the earliest affected gate. For new searches, execute
baseline and planning before candidate gates as described in SKILL.md.
2. After actually passing each gate, run `workflow.py record --state <state>
--gate <correctness|performance|review|replay|config_validation> --evidence <file>`; repeat evidence
for every dependency. Include model plus companions, input data, configuration,
versions and actual command/review outputs at the earliest applicable gate.
Updating a gate invalidates all downstream records. Never record a failed gate.
3. Run `workflow.py prepare --report <narrative.json> --baseline <perf.json>
--candidate <perf.json> --analyzer <analysis.json> --output <new-report.json>`.
Repeat performance arguments for compatible sessions of one artifact. Raw
samples drive percentiles; analyzer data populates coverage and opportunities.
The output includes source hashes. Human/LLM diagnosis, profile interpretation
and independent review verdicts are never synthesized by this helper.
4. After successful clean replay, run `validate_config.py` against the exact
final `rebuild_config`, source, confirmed champion, and frozen correctness
inputs in a new work directory. Declare `--min-cosine` before running it. The
helper rebuilds through public `winml build`, checks ONNX public I/O, and runs
`winml eval --mode compare` on the target. Record the passing receipt and all
bound inputs in the `config_validation` gate. Never hand-author the receipt.
5. After final report preparation, supply a JSON delivery config with absolute
paths: report, champion, winml_config, companions (list), rebuild_config,
repro_script, repro_lock, repro_assets (list), config_validation,
promotion_context. Existing finalizer contracts govern their contents. The
report and all delivery inputs must be present in recorded gate evidence.
6. Run `workflow.py deliver --state <state> --config <delivery.json>
--output <new-version-directory>`. It requires matching evidence, validates
the final report, finalizes bundle/, validates it, creates the handoff and
validates that before writing receipt.json. Existing output is never replaced.
If later delivery fails, partial output is retained for diagnosis, WITHOUT a
success receipt. Correct the issue and choose a new version directory.

Cleanup is downstream of validation, not an exemption from it. Never delete a
file recorded in workflow state, including the prepared source bound by final
config validation. After cleanup, rerun `workflow.py status`; completion requires
`next_gate=deliver` with every recorded hash still resolvable.

Boundary: record is an explicit attestation, not proof of correctness or reviewer
identity. Hashes detect changes, not fabricated evidence. Operators must verify
gates and source identity. Receipt separates artifact validation, recorded model
evidence, recorded independent review, recorded replay and unperformed visual
review. The separately executed final-config receipt proves only the clean
rebuild, public I/O, target load, and configured correctness threshold it names.
Do not describe recorded attestations as independently executed checks.
State updates assume one writer per run. Simultaneous agents must use separate
run directories. The wrapper does not execute hardware or create GitHub PRs.

A validated receipt is required for a final completion claim. Separately run
the bundle wrapper with -ValidateOnly in the intended environment and inspect
the actual rendered output; record unavailable browser inspection honestly.
