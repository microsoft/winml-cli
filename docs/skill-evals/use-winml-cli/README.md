# Historical `use-winml-cli` evaluation results

This is an archive of recovered May 2026 records, **not a new evaluation of the
current skill or CLI**. The inputs restored in PR
[#1497](https://github.com/microsoft/winml-cli/pull/1497) have since been adapted;
do not apply these historical passing grades to those revised assertions.
The [machine-readable index](./index.json) distinguishes recorded results,
evidence types, and missing hardware coverage.

## Recovered results

| Evaluation | Historical run | Recorded result | What the evidence establishes |
|---|---|---|---|
| Description-only trigger judge | [20260520-133832](./trigger/20260520-133832/results.json) | 29/30 correct; 0 false positives, 1 false negative | A single judge's selection predictions, not native skill invocation |
| Advice-only response | [20260520-133849](./response/20260520-133849/benchmark.json) | With skill 56/56; without skill 29/56; delta +48.2 percentage points | Archived concept/static-command grades for 9 cases |
| Advice-only response | [20260524-133637](./response/20260524-133637/benchmark.md) | With skill 71/71; without skill 35/71; delta +50.7 percentage points | Archived concept/static-command grades for 11 cases |
| CPU E2E | [20260520-134922_cpu-only](./e2e/20260520-134922_cpu-only/meta.json) | 3 eligible cases, 3 trials each; 9/9 trials recorded as passing | CPU benchmark/build and scope-refusal evidence under the old grader |
| QNN/VitisAI NPU E2E | Same CPU run's `cases_skipped` | QNN benchmark, QNN compiled build, and VitisAI benchmark all SKIP | No NPU execution result in this recovered run |

The trigger input has **14 positive and 16 negative** labels. The original
cross-pillar prose reported a 15/15 split; the archived query rows, not that
prose, determine the actual split. The single false negative is the
version-controlled config/CI-build query (row ID 13).

The response totals include a static CLI-command check per case. The later
71 assertions are 60 concept assertions plus 11 static checks, not 71 concept
assertions. Unchanged baseline grades were reused historically; the two runs
are not independent replications.

### Actual CPU latency records

All three metric files identify `microsoft/resnet-50`, image classification,
CPU execution, batch size 1, 10 warmups, and 100 measured iterations.

| Trial | Recorded mean latency | Raw metric output |
|---|---|---|
| 1 | 22.326 ms | [resnet50_perf.json](./e2e/20260520-134922_cpu-only/cases/cpu-benchmark-resnet/trial-1/outputs/resnet50_perf.json) |
| 2 | 27.276 ms | [resnet50_perf.json](./e2e/20260520-134922_cpu-only/cases/cpu-benchmark-resnet/trial-2/outputs/resnet50_perf.json) |
| 3 | 34.251 ms | [resnet50_perf.json](./e2e/20260520-134922_cpu-only/cases/cpu-benchmark-resnet/trial-3/outputs/resnet50_perf.json) |

The sample counts and rounded sample means agree with the recorded metrics.
These are separate historical trials, not controlled cross-device performance
comparisons. The exact CPU model and complete runtime environment were not
recovered. Do not treat this range as a performance claim for other machines.

The other two CPU-eligible cases are a CPU build and a generative-model
scope refusal. A refusal passing its assertions is not model inference or
an NPU benchmark. Their trial gradings and per-case aggregates are archived
alongside the benchmark case.

## Where are the NPU results?

No separate NPU-machine execution record was found in the searched sources:
reachable local branches, reflog-only history, 59 unreachable commits, the
surviving development workspace, or the inspected local scratch/research
directories and older clone. The original PR discussion did not supply a
hardware-run link either. This is a bounded recovery finding, **not proof that
no NPU trial ever ran on another machine**.

There are NPU-named response cases, including Snapdragon builds, Ryzen AI
benchmarking, and NPU-versus-CPU comparisons. They are advice-only responses
graded for content. Their passing grades must not be presented as QNN,
OpenVINO NPU, or VitisAI hardware execution.

In the recovered CPU run, the registered EPs were CPU, DirectML, and OpenVINO.
Provider registration alone does not establish an Intel NPU. The recorded
QNN and VitisAI cases were explicitly skipped because those providers were
unavailable. The historical E2E suite did not contain an Intel NPU case.

Two earlier directories, `20260513-220000_cpu-only` and
`20260513-230000_cpu-only-dogfood`, were also found in commit `638c6f09`.
Their metadata uses `passk-demo` / `pre-commit-verify` placeholders and reuses
identical trial telemetry. They are not imported as independent real trials.

## Evidence and provenance

Tracked evidence comes from snapshot
`db80c8bc609fa0d7a385fe88368ee56ecaeaaad3` of the former
`feat/winml-modelkit-skill` branch:

| Archive directory | Original path below `.claude/skills/winml-cli-workspace/` | Included evidence |
|---|---|---|
| `trigger/20260520-133832/` | `trigger_eval/runs/20260520-133832/` | Description, query labels, judge decisions, and result rows |
| `response/20260520-133849/` | `response_eval/runs/20260520-133849/` | Original benchmark JSON and 18 per-case/configuration grading files |
| `response/20260524-133637/` | `response_eval/runs/20260524-133637/` | Original benchmark Markdown and 22 per-case/configuration grading files |
| `e2e/20260520-134922_cpu-only/` | `e2e_eval/runs/20260520-134922_cpu-only/` | Run metadata, 3 case aggregates, and 9 trial grading files |

The three `outputs/resnet50_perf.json` files were recovered from surviving
local-only output files at the corresponding original E2E run paths. They
were not tracked in the source Git snapshot. Their integrity is recorded in
the provenance manifest using SHA-256, which verifies the copied bytes, not
the authenticity of a complete execution trace.

All imported source records are byte-preserving copies except `meta.json`,
whose private host name was removed. Source Git blob IDs for every tracked
record and SHA-256 hashes for the local metrics are recorded in
[source-files.json](./source-files.json). No scores were regraded during
recovery. Aggregate totals were recomputed from the archived grades as a
consistency check. Scoped Git attributes preserve evidence line endings so
checkout does not invalidate the recorded hashes.

Full response outputs/transcripts, machine-specific scripts, build configs,
model binaries, external contexts, and private machine paths are not included.
Short response quotations already embedded in grading evidence are retained.
The old benchmark Markdown's recommendations are historical statements, not
instructions to ship or promote the current skill.

## Interpretation limits

- The response grader was the parent assistant from the same model family as
  the response writers, not the fresh independent grader now required by the
  restored evaluation instructions. Its evidence is historical and not an
  independent quality certification.
- Static checks are archived claims against the old CLI, not a new verification
  that every quoted command works today. Historical installation assertions and
  scope/roadmap statements may now be stale.
- CPU metadata records `winml, version 0.0.2` and a `skill_commit`, but the
  original cross-pillar report says skill edits were in flight. A complete
  immutable tested skill snapshot, agent/model configuration, and raw tool
  traces were not recovered. The recorded commit is not an exact tested hash.
- The old E2E grader checked artifact existence/key presence, final-message
  regexes, and tool/time budgets. It did not establish the stronger model-load,
  numeric/device agreement, native activation, or context-sidecar requirements
  in the restored fixtures. No current-contract pass rate is claimed.
- Legacy `pass_at_k` fields store counts of passing trials. They are not
  statistical pass@k estimates or additional independent trials.
- Archived FAIL judgments and the trigger false negative are preserved.
  SKIP remains missing coverage, never PASS. No new model execution or
  live-agent evaluation was performed for this archive.
