---
name: adding-model-support
description: >
  Use this skill whenever contributing model support to the `winml` CLI: an unsupported
  Hugging Face model or architecture, an existing family that fails in one or more commands,
  a missing recipe, exporter, resolver, task, dataset adapter, evaluator, or promotion handoff.
  It diagnoses the gap, implements the narrowest reusable fix, validates the result, opens a
  draft model-support PR, and captures reusable findings. Trigger even when the user asks to
  repair support rather than explicitly saying "add model support". Do not use it merely to
  run or tune an already-supported model, optimize an existing recipe, or add a new execution
  provider backend.
compatibility: Requires a microsoft/winml-cli checkout, Python 3.11 with uv, GitHub CLI with push access, and an agent runtime that can delegate fresh subagents.
---

# Adding model support

Use this skill to take one unsupported or partially supported model from current-main diagnosis through a validated draft contribution. Diagnose first, freeze the Effort/Goal/Outcome target from evidence, implement the narrowest reusable fix, validate independently, and capture durable knowledge.

This is a dispatcher, not the complete procedure. Load the owning role contract under [agents/](./agents/) at every transition. Do not substitute this overview for those instructions.

## Repository scope

This skill lives in `skills/adding-model-support/` in [`microsoft/winml-cli`](https://github.com/microsoft/winml-cli). Resolve `src/...`, `tests/...`, and `examples/...` from the repository root. Keep methodology changes under `skills/adding-model-support/**` in a separate skill-only branch and PR; never mix them into a model-support PR.

## Prerequisites and optional integrations

- Run from a `microsoft/winml-cli` checkout with Python 3.11, `uv`, authenticated `gh`, and push access. The orchestrator verifies these before implementation.
- The agent runtime must support fresh subagent delegation. Each role loads its file under `agents/`; if delegation is unavailable, stop as BLOCKED rather than collapsing producer, tester, and reviewer into one agent.
- `model-breakdown` is optional. When discoverable as an installed skill, require schema `1.2+`; otherwise the planner builds the same profile from pinned source, config, and model-card evidence.
- `auto-optimize` is optional and needed only for `--promotion-handoff`. Discover its `scripts/promotion.py` through the installed skill location. If it is absent or incompatible, promotion mode stops as BLOCKED while normal model-support work remains available. Follow the [orchestrator promotion handoff entry](./agents/orchestrator.md#promotion-handoff-entry).

## Pipeline

Run one model at a time. The orchestrator delegates every role to a fresh agent and drives all loop-backs to `APPROVE`, `REJECT`, or `BLOCKED`; `REQUEST_CHANGES` is not terminal.

| Role | Owns | Contract |
|---|---|---|
| Orchestrator | preflight, delegation, loop continuation, isolation, cleanup | [orchestrator.md](./agents/orchestrator.md) |
| Planner | current-main diagnosis, model profile, baseline, Effort/Goal/Outcome charter | [planner.md](./agents/planner.md) |
| Producer | minimal code and exact-evidence recipe candidates | [producer.md](./agents/producer.md) |
| Tester | repair loop and Goal-ladder verdicts | [tester.md](./agents/tester.md) |
| Learner | model knowledge, methodology findings, blocker diligence | [learner.md](./agents/learner.md) |
| Explainer | existing/new draft PR, body, label, replies, shipment | [explainer.md](./agents/explainer.md) |
| Reviewer | independent final-head verification and one opinion comment | [reviewer.md](./agents/reviewer.md) |

```text
planner -> producer -> tester -> learner -> explainer -> reviewer
   ^           ^                                      |
   |           +---- artifact REQUEST_CHANGES --------+
   +---------------- scope/tier contradiction --------+
```

## Non-negotiable boundaries

- Planner freezes the target only after current-main baseline evidence. If reality contradicts it, return to planner.
- Producer, tester, explainer, and reviewer remain separate agents. The reviewer starts from a fresh context after the draft PR exists.
- Every model-support PR targets `main`, remains draft, and carries `model-scale-by-skill`. Dependency ancestry never changes the live PR base.
- The reviewer writes only one normal PR comment containing `APPROVE`, `REQUEST_CHANGES`, or `REJECT`; these are not GitHub Review states.
- Every required `(EP, device, precision)` tuple retains exact PASS, blocked, or exhausted-failure evidence. Never infer one tuple from another.
- Run one bounded final-SHA FP32 CPU functional smoke Eval; do not present it as representative accuracy. Keep per-tuple performance and both component-level and op-level Analyze evidence.
- Treat a recipe as authoritative during recipe validation. Semantic CLI overrides are diagnostic evidence, not proof that the recipe works.
- Prefer narrow metadata/config/architecture-derived shared fixes. Preserve existing public defaults, signatures, recipe semantics, and sibling-model behavior with regression tests.
- Never edit `examples/recipes/README.md` for recipes produced by this workflow.
- Cleanup is orchestrator-owned, manifest-driven, dry-run-first, and limited to marked run-owned regenerable state via [cleanup-run-cache.ps1](./scripts/cleanup-run-cache.ps1).

## Knowledge and evaluation resources

- Read [model knowledge](./model_knowledge/) before planning and update it only with scoped evidence after testing.
- Record reusable workflow lessons in [skill methodology findings](./skill_meta/); preserve historical provenance.
- Use [response and trigger evals](./evals/) when changing this skill's behavior or description.

Begin execution by loading the [orchestrator contract](./agents/orchestrator.md).
