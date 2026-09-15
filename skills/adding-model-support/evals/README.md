# Evals for `adding-model-support`

Two eval suites, following the `skill-creator` methodology (trigger optimization + response benchmarking with `with_skill` vs `baseline` subagents, a grader, `aggregate_benchmark`, and the eval-viewer).

## 1. Trigger eval — [`trigger_eval.json`](./trigger_eval.json)

18 realistic queries: 9 **should-trigger**, 9 **should-not**. The negatives are deliberately *near-misses* against adjacent workflows this skill must not poach from:

| Adjacent workflow | Example negative |
|---|---|
| `check-model-feasibility` | "which EP should I pick for dinov2 on my Snapdragon laptop?" |
| `adding-ep-support` | "onboard a new NPU vendor's runtime as a new execution provider backend" |
| `autoconfig` | "my vit-base recipe already builds but W8A8 cosine is 0.85, tune the calibration" |

Run it through an installed `skill-creator` description optimizer to tighten the `description` frontmatter. The optimizer is an external authoring dependency and is not bundled with this runtime skill.

## 2. Response eval — [`evals.json`](./evals.json)

### Golden truth must be human-authored

The skill's own shipped PRs (#951 vilt, #952 mgp-str) **cannot** grade the skill that produced them — that is circular. **Recipe-only / catalog-registration PRs (#785, #854) are also excluded**: they add zero source and only register configs, so they can't test the engineering judgment the skill exists to encode. Golden truth is **independent, pre-skill, human-authored source-engineering** PRs that each added support for a model FAMILY, mined from repo history. Three were chosen to span distinct engineering axes:

| Case | Golden PR | Author | Effort | Axis | What it stresses |
|---|---|---|---|---|---|
| `golden-790-library-routing-code` | [#790](https://github.com/microsoft/winml-cli/pull/790) `eab2c48e` | `vortex-captain` | **L1+/L2 shared-layer** | library routing (wrapped-library `model_type`) | architecture-agnostic routing (Cardinal Rule 1 — task derived from Optimum, **not** hardcoded), all-call-site routing, pytest presence |
| `golden-807-modality-task-detection` | [#807](https://github.com/microsoft/winml-cli/pull/807) `cf25bfd4` | `Zhipeng Wang` | **L1/L2 shared-layer** | modality-aware task detection unified across commands | one detector every command routes through, **data-driven** modality table (extend-the-table-not-the-code), upgrade only on the surfaced task, cross-resolver consistency test |
| `golden-850-seq2seq-composite` | [#850](https://github.com/microsoft/winml-cli/pull/850) `507c2696` | `Zhipeng Wang` | **L1+/L2 config-resolution** | encoder-decoder composite build for no-task seq2seq | diagnose the non-runnable decoder-only half, no-task path uses the **same** `detect_task` as inspect, composite registry lookup **not** model-name branching |

The three are deliberately complementary: #790 is "a model loads under a different Optimum library"; #807 is "every command must agree on the task, and modality comes from config fields not model names"; #850 is "a wrong build still emits an ONNX file, so the failure is a non-runnable half the Goal ladder won't catch without a runtime check."

### The measurement is the with-skill − baseline delta

Assertions check **structural decisions**, never byte-exact reproduction (an agent will pick different helper names, different table layout, different test names — all fine). The skill only *earns its keep* where the **baseline (no-skill) run fails the assertion and the with-skill run passes it**. Each case records a `baseline_expectation` describing the failure mode the skill is supposed to prevent (hardcoding the task for #790; per-command/per-model patching for #807; accepting the decoder-only half or model-name branching for #850).

### Why these assertions are falsifiable

- #790: "no `if model_type == 'timm_wrapper'` task assignment anywhere" — grep-checkable, and it's the single most important property the human PR got right ("the task is not hardcoded; the branch imports optimum… to populate Optimum's registry").
- #807: "disambiguates modality from a data-driven table, not `if model_type == 'dinov2'`" — grep-checkable; and "does NOT fire for CLIP (nested `image_size`)" is a concrete negative the human PR explicitly preserved.
- #850: "leaves a sequence-classification BART as a single model" — a checkable negative that separates a real fix from an over-eager `if model_type in {t5,bart,marian}` that would wrongly composite everything.

## Running the harness

```bash
# from the skill-creator directory
# 1. spawn with_skill + baseline subagents for every eval (same turn)
# 2. grade each run's outputs against the assertions array
# 3. aggregate
python -m scripts.aggregate_benchmark <workspace>/iteration-N --skill-name adding-model-support
# 4. view
python <skill-creator>/eval-viewer/generate_review.py <workspace>/iteration-N \
  --skill-name adding-model-support --benchmark <workspace>/iteration-N/benchmark.json
```

Put results under `adding-model-support-workspace/iteration-N/eval-<name>/{with_skill,baseline}/`.

## Adding more golden cases

Mine additional human, pre-skill PRs: `git log --oneline | Select-String "example:|examples:|feat\(timm|model"`. Good next candidates to broaden coverage of the Effort axis: an L0★ encoder-decoder/no-template case (so the catalog-gate + no-template path is exercised) and an L1-light vendor-subclass case. Keep the rule absolute: **never use a skill-produced PR as golden truth.**
