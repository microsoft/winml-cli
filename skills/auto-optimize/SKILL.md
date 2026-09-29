---
name: auto-optimize
description: 'Use when optimizing ONNX latency with WinML for a target EP/device, including QNN NPU profiling and graph interactions.'
---

## Entry

Read [workflow commands](./references/workflow.md). Use workflow.py status before resume and workflow.py deliver as the sole final delivery entry. Lower-level scripts remain diagnostic helpers, not alternative completion paths.

Record the resolved skill path and SHA-256 in run-local evidence. Resolve model, EP/device, goal, workdir, and `WINML_CLI_REPO` from context; ask only for missing values. Read [resume](./references/resume.md) before choosing any measurement.

| Entry state | Next gate | Boundary |
| --- | --- | --- |
| New search | Baseline below, then planning router | No invented attribution |
| Supplied candidate / validation-only | First unverified structural, correctness or performance gate | No planner or new search |
| Resume | Match hashes, options, versions and cache identity | Invalidate affected evidence |
| Failed replay | Diagnose replay | No publication |

Hash model, inputs, env, versions, and options. Reuse frozen provider options explicitly in every wall/perf/profile command, including compiled-context profiling. Resolve scripts relative to this skill, then invoke absolute paths from the run directory.

## Baseline - new search only

Inspect CLI help. Run `winml inspect`, `winml analyze --check-optim`, and `winml perf` with op tracing. Collect hotspots, partitions, fallback, layout, transfers. Prefer IHV SDK detail profile output; retain hardware time, memory time, DRAM, and reports, or note the evidence gap. Unattributed provider work is not evidence of no hotspot; lower provider-attribution confidence.

Resume at the [first unverified gate](./references/resume.md): validate supplied candidates; plan only new experiments. Write an evidence brief on bottlenecks, provider work, gaps before hypotheses.

## Planning router - evaluate before loading cases or proposing hypotheses

Read [`knowledge/qnn-npu.md`](./knowledge/qnn-npu.md).

When a provider-attributed detail trace assigns 70 percent or more accelerator time to a dominant region and fallback, partitioning, and transfers are not larger explanations, the fast lane is priority only, schedules at most two probes, does not prune other candidates, and keeps the normal correctness and paired performance gates. For quantized graphs, the second bounded probe is qdq-boundary placement.

Write `hotspot_evidence.json`, resolve [plan_hotspot.py](./scripts/plan_hotspot.py), and run `python ./scripts/plan_hotspot.py hotspot_evidence.json --output hotspot_plan.json`.
Resolve that linked file path; do not infer a workspace-root `scripts/` directory.
Adopt the helper result as the current plan only after exit code 0, stdout parses as JSON, and stdout bytes equal the `--output` file bytes.
Never synthesize or rewrite the plan JSON.
If mode is `dominant-hotspot-fast-lane`, execute only its steps and exit instruction before loading cases or proposing normal-loop hypotheses.
If mode is `normal-hypothesis-loop`, continue normally.

Read [`knowledge/index.json`](./knowledge/index.json), match EP/device anchors,
and load at most three cases.

Read [Graph Scout](./roles/graph-scout.md) before its first invocation. Initialize `report.json`/`report.html` with its independent baseline review. Supply immutable facts, not the main agent's hypotheses. Repeat after each material leader and before normal completion. If an independent role is unavailable, retain an unverified review; never impersonate an independent reviewer.

After fast-lane probes, record outcomes in evidence and invoke the planner in the next planning step. A bounded planning step is not completion of the optimization request. Resume the normal loop unless the user requested only that bounded step.

### Normal hypothesis loop - only when the hard gate is inactive or exited

Maintain at most three active hypotheses. Before each experiment write mechanism, change or bundle, supporting evidence, graph delta, trace delta, safety, and the cheapest falsifier.

Choose the cheapest discriminating experiment. Test interactions when one representation enables another.

Per candidate: use a fresh directory keyed by model hash, graph changes, provider options, runtime/SDK versions, and profiling mode; never reuse an incompatible compiled context for that cache identity. Record commands and hashes; validate ONNX, shapes, graph delta, and I/O. Enforce correctness before performance. Screen with alternating A/B and B/A runs; confirm leaders with paired evidence. Compare graph and trace delta with removed provider work, layout, partitions, fallback, and accelerator time; reject worse lowering. Preserve failures.

Use [Perf Arbiter](./roles/perf-arbiter.md) for order disagreement, zero-crossing intervals, drift, cache suspicion, or wall/trace mismatch. Confirmed against baseline, a candidate may lead on a lower point estimate within noise; label a statistical tie and do not claim superiority.

After each material leader, Graph Scout runs LLM [capability closure review](./references/capability-closure.md) over the leader, verbose registry, analyzer output, residual topology, cases, and ledger. Probe at most three `PROBE_REQUIRED` capabilities; route graph-changing probes through candidate gates. `DEFERRED_BUDGET` blocks `NO_MATERIAL_OMISSION`.

After structural, correctness, paired-evidence, and trace gates, the [Feature Gap Engineer](./roles/feature-gap-engineer.md) may modify `WINML_CLI_REPO` in an isolated current-main worktree. Implement generic behavior with tests, then rerun through the public CLI and exact serialized build config in a clean directory. Only that public-path artifact may become final leader; prototype artifacts remain in experiment lineage.

For stable replay requests, read [reproduction](./references/reproduction.md) and require independent clean builds. Map analyzer coverage and opportunities into report fields; retain native trace units (cycles are not microseconds). Every missing display metric or empty evidence table needs a field-specific `missing_reasons` explanation. Refresh diagnosis after closure and distinguish unmeasured from inapplicable. Never substitute handwritten HTML. Run [`render_report.py`](./scripts/render_report.py). Run full replay from a fresh temporary directory, then invoke workflow.py deliver (which calls [`finalize_output.py`](./scripts/finalize_output.py)) to publish `champion.onnx`, companion files, `winml_config.json`, `report.json`, `report.html`, hash-bound `manifest.json`, and reproduction assets. New runs require `rebuild_config.json`, replay body `repro-run.ps1`, generated wrapper `repro.ps1` with `-ValidateOnly`, `repro.lock.json`, `perf_input.npz`, `eval_inputs.npz`, and `inputs_manifest.json`; use `requires-unmerged-pr` honestly. Keep `winml_config.json` for the built champion and `rebuild_config.json` semantically separate. Validate the published bundle.

No generic source change means no optimizer PR or optimizer label requirement. Feature Gap Engineer returns implementation evidence; the main agent alone owns PR creation after publication. After bundle validation, the delivery wrapper runs `promotion.py create` once for `promotion_handoff.json`; do not run it again; follow [PR Routing](./references/pr-routing.md). For an eligible optimizer route only, Auto-optimize owns the optimizer PR. Run `gh label list`; the target repo must contain `model-opt-by-skill`, and the skill must not create the label automatically. Only when PR Routing selects an eligible optimizer change, create the Draft PR once with `gh pr create --draft --label model-opt-by-skill`, then verify with `gh pr view <url> --json labels`. Missing or unavailable label blocks handoff, and missing post-create label verification blocks handoff. For that optimizer PR, use [Ponytail](./references/ponytail.md) or fallback, then invoke [Check-in Reviewer](./roles/checkin-reviewer.md), record its ready for check-in verdict; never merge or convert the Draft.

Bundled knowledge is model-agnostic. Artifacts stay run-local.

Persist reusable tested outcomes. Run [`save_case.py`](./scripts/save_case.py) so the case and SHA-256-bound index are atomic. Before writing, Graph Scout must return `GENERIC_CASE_APPROVED` for the `--content-digest` digest and bind it in `generic_review`; otherwise keep it run-local.

## Stop

Stop adding experiments for confirmed target, exhausted hypotheses, or budget. Review closure before normal completion; `DEFERRED_BUDGET` remains insufficient evidence. Finish replay and publication only with a deliverable leader and sufficient budget. Otherwise retain run-local partial results and list unverified gates, without claiming final delivery. A user stop request ends additional experiments, reviews, and publication immediately. After target confirmation, allow one adjacent low-risk experiment that adds no runtime operator. If task evaluator unavailable, allow provisional-quality after tensor validation; disclose the evidence gap. Retain run-local evidence. Source changes require generic behavior and tests. Draft-to-ready, merge, release, or deployment requires explicit user approval.
## Report delivery gate

Read [report delivery](./references/report-delivery.md) before authoring report facts.
Raw measurements take precedence over missing-data explanations. The final
report is generated by the bundled renderer, never by hand-written HTML.
