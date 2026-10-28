# Evals for `use-winml-cli`

Three suites restore the evaluation coverage that accompanied this skill before
only its body was published in PR [#733](https://github.com/microsoft/winml-cli/pull/733).
They follow the sibling skills' `evals/` layout and use ungraded assertions:
`passed: null` and empty `evidence` are placeholders, not passing results.

| Suite | File | Coverage |
|---|---|---|
| Trigger | [trigger_eval.json](./trigger_eval.json) | 30 queries: 14 should activate, 16 should not |
| Response | [evals.json](./evals.json) | 11 cases with 60 concept assertions |
| E2E | [e2e_eval.json](./e2e_eval.json) | 6 cases, default 3 independent trials per eligible case |

These are evaluation inputs and instructions, not a bundled agent runner or an
automatic CI job. An agent host and, optionally, an externally installed
`skill-creator` are authoring dependencies. Do not install them as runtime
dependencies of this skill.

## Provenance and scope

The inputs were recovered from the original three-pillar framework introduced
in commit `5820645fd751ae08fcfbcbcb8602e7c7122c652a`, using the later snapshot
`db80c8bc609fa0d7a385fe88368ee56ecaeaaad3` on the former
`feat/winml-modelkit-skill` branch.

The 30 trigger prompts, their labels, and the 11 response prompts are retained.
Skill names, stale explanatory text, and assertion wording are updated for this
layout. Command validity requires actual CLI-help evidence, not merely
plausible-looking flags. Installation assertions reflect the current package's
Python 3.11 requirement. CI repeatability does not imply byte-identical model
artifacts without controlling inputs and dependencies.

The six E2E scenarios are retained with explicit CPU targeting, named benchmark
outputs, and trace/artifact checks rather than success inferred from prose or a
compiled-model filename. Historical runs, scores, transcripts, model binaries,
machine-specific Python runners, and Claude Code slash commands are deliberately
not imported.

Scope assertions evaluate the contract of [SKILL.md](../SKILL.md), not the full
CLI capability matrix. In particular, the shipped skill excludes generative
models even if the current CLI supports additional paths. A scope-policy
response must not claim that the CLI has no such capability. Revisit these
fixtures when the skill's declared scope changes; do not make unsupported
late-2026 roadmap claims.

These scenarios are behavioral regressions, not independent engineering golden
truth. The old skill and its historical grades cannot establish that the current
skill or current CLI is correct.

## 1. Trigger evaluation

Use fresh sessions with normal skill discovery. Supply only each `query` as the
user request; keep `should_trigger`, `rationale`, and the other fixtures hidden
from the agent. Record whether `use-winml-cli` was actually invoked successfully,
using the host's tool trace. Mentioning its name or claiming to have loaded it is
not evidence of activation. Other skills may legitimately activate.

Report under-trigger and over-trigger separately, and distinguish execution
errors from selection failures. Retain failed trials. A description-only judge
or an external `skill-creator` description optimizer can help author the
description, but its predictions are not a measurement of native activation.

## 2. Response evaluation

Follow the same with-skill versus baseline workflow as the sibling
`adding-model-support` evals:

1. Spawn fresh, matched sessions for each prompt: `with_skill` reads
   [SKILL.md](../SKILL.md); `baseline` has no access to this skill. Do not expose
   assertions or expected labels to either response writer.
2. Keep these trials advice-only: no builds, model downloads, repository edits,
   or publication. Read-only CLI-help and hardware-discovery tools may be used
   equally by both configurations. Supply identical versioned help and hardware
   evidence if the evaluation host cannot run them.
3. Use a fresh independent grader that sees the response, assertions, relevant
   help evidence, and tool trace, but not the skill body or historical grades.
   Record a boolean decision and specific evidence for each assertion.
4. For every concrete command quoted, check subcommands, flags, and argument
   shape against the evaluated CLI version's help. Invented flags fail even if
   the response also says "check --help". A help reference alone does not prove
   the agent consulted it. Missing help evidence leaves command validity
   unverified, not passed.
5. Aggregate per-case concept results and static command validity separately.
   Report the with-skill minus baseline delta; do not reuse historical baseline
   outputs after changing prompts, the CLI, model, tools, or host configuration.

The JSON uses the sibling skill's `skill_name` / `evals` structure, `files: []`,
and assertions with unset judgments. Preserve assertion IDs for comparisons.
An external `skill-creator` grader and `aggregate_benchmark` can consume the
resulting run directories; no such tooling is bundled here.

## 3. E2E evaluation

E2E trials execute commands and may download models. Run them only in an
explicitly approved, disposable evaluation directory with the necessary network
access and hardware. Do not reuse the development checkout as a writable trial
workspace. Install the CLI independently and record its version.

1. Record the skill hash, agent/model/host versions, OS, hardware, available EPs,
   and CLI version. Keep expected outcomes and grading metadata out of the
   agent's prompt.
2. Check both device availability and registered EPs before scheduling trials.
   `required_ep: null` means no NPU requirement, not permission to invent
   hardware. A registered provider alone does not prove that the target NPU is
   usable. Mark unavailable hardware cases SKIP with a reason; never substitute
   CPU for a requested NPU and count it as a pass.
3. Give each eligible case three fresh trials (`default_runs`), each with an
   empty output directory. Capture commands, final response, tool count,
   elapsed time, stdout/stderr, and output artifact paths.
4. Grade the actual outputs and tool trace against `expected_outcomes`.
   Globs are recursive relative to that trial's output directory; they are
   candidate discovery only, not proof of success. `artifact_keys` are required
   top-level keys, and `latency_ms.mean` must be a positive finite number that
   agrees with the reported latency at the reported precision. Verify actual
   execution on the requested device/EP, not merely the requested CLI flags.
5. For builds, load the newly produced ONNX and verify its external data exists.
   For QNN, verify the model's EPContext identifies the requested provider and
   all referenced context sidecars exist (or the context is embedded).
   An ONNX filename or a success sentence cannot prove compilation.
6. For the refusal case, `should_complete_task: false` is intentional. PASS
   requires the stated skill-scope boundary and no executed forbidden pipeline
   command for that model. Grade `forbidden_commands` from actual tool calls,
   not a regex over the final response.

`max_tool_uses` and `max_duration_s` are per-trial budgets; exceeding either
fails a completed trial. `final_message_must_match` is a case-insensitive
presentation check, not an outcome oracle. Preserve runtime/tool failures as
ERROR, unavailable hardware as SKIP, assertion failures as FAIL, and fully
evidenced trials as PASS. Report the passing-trial count out of three for each
case, together with all FAIL/ERROR/SKIP outcomes. This is not a statistical
estimate of pass@k or an accuracy evaluation of the resulting models.

Keep all results outside the shipped skill. Review logs before sharing: they
may contain local paths or host configuration. No new live-agent or hardware
results are asserted by these restored fixtures.

Recovered May 2026 scores and CPU metric evidence are archived separately in
[historical evaluation results](../../../docs/skill-evals/use-winml-cli/README.md).
They do not certify these revised fixtures; NPU execution evidence was not
found in the searched sources, and the recovered CPU run records NPU cases as SKIP.
