# Evals for `adding-model-support`

Use fresh with-skill and baseline agents, then grade their outputs independently
with an installed `skill-creator` harness. The agent workflow itself is unchanged.

## What each case measures

The 11 response-suite cases declare a `mode`:

| Cases | Mode | Required evidence |
|---|---|---|
| 0-2 | `implementation` | Code changes and executed regressions against the pinned pre-fix source. |
| 4 | `implementation` | A supplied pre-fix checkout, candidate recipe and compatible data; actual evaluator and recipe-semantic tests. |
| 3 | `response` | A written summary of the supplied recorded handoffs, not a new model run or live PR. |
| 5-9 | `response` | Review, validation, rerun, cleanup and PR decisions for the stated scenarios; no external operations. |
| 10 | `response` | The existing four custom-code authorization/isolation decisions; no execution or publication. |

Implementation cases still fail for incorrect routing, broken explicit-task or
sibling behavior, orphan decoders, silent ambiguity, or missing consumer/build
tests. Accept equivalent shared helpers and authoritative metadata policies;
do not require the exact names or internal layout of a historical patch.

Response cases test decisions and communication. Do not penalize a correct
plan because it did not actually rebase, delete files, create a PR or measure
hardware. Conversely, invented execution or measurements are failures.
These cases do not establish end-to-end operational or hardware correctness.

Case 7 covers no-impact reuse, bounded main/patch changes, and full-impact
conditions in one response. It follows the existing planner contract: bounded
patch drift may require a partial rerun; unknown dependency reachability or
broad impact requires a full rerun. Case 9 covers creation, recovery and failed
metadata checks; correctly recovering an existing PR must not be penalized for
avoiding a duplicate.

## Inputs and reference truth

Cases 0-2 reference independent, pre-skill human engineering changes:

| Case | Reference | Baseline |
|---|---|---|
| 0 | [#790](https://github.com/microsoft/winml-cli/pull/790), `eab2c48e` | Parent of the library-routing change. |
| 1 | [#807](https://github.com/microsoft/winml-cli/pull/807), `cf25bfd4` | Parent of unified modality-aware task detection. |
| 2 | [#850](https://github.com/microsoft/winml-cli/pull/850), `507c2696` | Parent of no-task composite resolution. |

Pin the source and compatible environment for both conditions. Do not supply
the already-fixed checkout or reveal the later patch/assertions to producers.
Do not use skill-produced or recipe-only registration PRs as engineering goldens.
Policy cases use the current skill contracts, not a presumed baseline failure.
`baseline_expectation` is a hypothesis, not a grading result.

Case 3 uses [recorded handoffs](fixtures/pr_description_handoffs.json) from an
earlier code-generated CPU fixture. Its timings are recorded input data, not
new benchmark results; provider classifications are explicitly synthetic.
No PyTorch/ONNX execution is needed for this formatting case. No target-model
recipe is supplied, so the response must disclose that reproduction gap rather
than fabricate a command or public artifact. The original source digest and
historical generator reference are retained in the fixture.

The [trigger suite](trigger_eval.json) contains 22 queries: 11 positive and 11
negative, including existing-model analysis versus missing-capability repairs.
Use the external harness for backend selection and optional neighboring-skill
catalogs; a target-only trigger probe is not a full workflow-routing benchmark.

## Running and grading

Freeze the skill version, case inputs, environment and mode before dispatch.
Implementation work requires approved isolated checkouts and supplied inputs;
missing prerequisites are harness setup errors, not evidence that the skill
failed. The custom-code authorization gate still applies.

The external harness owns cache/process setup, provenance, outcome schemas,
aggregation and viewing. Use its assessment-v2 distinction between actual
failure, explicitly unselected scope and infrastructure error. Missing required
participant output is a failure, not an exemption. Keep assertions ungraded in
`evals.json`, preserve per-check evidence, and publish coverage beside pass rate.

Protocol 3 retains IDs 0-10, removes the extra simulator scenarios and separates
implementation from response expectations. Do not rescore old artifacts or
compare its aggregate directly with the earlier read-only/execution protocols.
A positive policy response is not proof of actual independent-agent execution.

Lightweight input checks, from the repository root with run-owned tool state:

```powershell
uv run --no-sync pytest skills\adding-model-support\tests `
  -p no:cacheprovider --no-cov --basetemp <new-run-owned-directory>
```

These checks validate inputs and existing contracts, not the skill's benchmark
performance. Use a fresh `--basetemp`; pytest may remove its previous contents.
