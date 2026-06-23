---
name: adding-model-support
description: >
  Use this skill when contributing support for a new Hugging Face model (or new
  architecture family) to `winml-cli`. The skill is organized around three orthogonal
  axes you commit to up-front: **Effort** (L0 config-only → L1 per-architecture code →
  L2 deeper changes), **Goal** (L0 build passes → L1 perf passes → L2 numerical delta
  vs. PyTorch → L3 task-metric accuracy), and **Outcome** (L0 recipe + artifacts → L1
  add code + feature-request issues for gaps + report). Covers diagnosing the gap with
  `winml inspect`, copying the closest recipe under `examples/recipes/`, writing the
  `{export, optim, quant, compile, loader, eval}` config sections (loader required; export/quant/compile may be `null`; eval is optional), and — at session end —
  appending what you learned to `model_knowledge/<family>.json` so the next
  contributor (human or agent) starts from your findings rather than from scratch.
  Trigger phrases: "I want to add support for model X", "winml says this model type
  is unsupported", "how do I write a recipe for a new architecture", "Qwen3 / Phi-4 /
  [new family] isn't recognized", "where do I add a new exporter", "the loader can't
  find my model_type", "what does a winml recipe look like". Skip for: end-user model
  selection (use `check-model-feasibility`); hand-tuning an already-supported model's
  optimization config (use `autoconfig`); adding a brand-new execution provider
  backend (use `adding-ep-support`).
---

# adding-model-support

You're here because `winml inspect` came back blank — or a build crashed because the toolkit doesn't have a recipe for this architecture. This skill walks the contributor path: **commit to an Effort/Goal/Outcome target up-front, diagnose the gap, do the work, validate against your target, and capture what you learned** so the next attempt on a related model is cheaper.

## When to use

- "I want to add support for Qwen3 / Phi-4 / DINOv3 / [new HF model]"
- "`winml inspect` shows no loader / exporter / inference class for my model"
- "How do I write a recipe config for a new model family?"
- A new HF release of an existing family (e.g. ViT-22B) needs an extra recipe variant
- A user filed an issue that ends with "and it would be great if winml supported X"

## Step 0 — Commit to a target on each axis (do this first)

Before touching anything, pick one cell per axis. Writing this down avoids the most common failure mode: rolling effort up from L0 → L2 mid-session without ever achieving a verified Goal.

### Effort axis — how much work do you expect

| Tier | Scope of change | Examples |
|---|---|---|
| **L0 — Config only** | New recipe file under [examples/recipes/](../../examples/recipes/), no source edits, **and** a copy-able recipe template exists in repo for the same export pattern | New variant of an already-supported architecture (`dinov2-large` next to `dinov2-small`); same family + new task or precision |
| **L0★ — Config only, no template** | Same as L0, but **no checked-in recipe of the same export pattern exists** — contributor writes the first reference recipe, typically by running `winml config` and refining | Code is registered (either via `@register_onnx_overwrite` in `models/hf/`, **or natively in Optimum**) but the export pattern is new to `examples/recipes/`. Today this hits every encoder-decoder model (bart, marian, t5, mu2, vision-encoder-decoder, **m2m_100**, **pix2struct**). Owes a published template + a finding in `model_knowledge/` that promotes the next L0★ in this pattern to plain L0 |
| **L1-light — Subclass a vendor OnnxConfig** | New file under [src/winml/modelkit/models/hf/](../../src/winml/modelkit/models/hf/) that **subclasses Optimum's existing `OnnxConfig`** and overrides one method (`outputs`, `generate_dummy_inputs`, `inputs`), registered with `@register_onnx_overwrite` to either flip overwrite or add a missing task on a `model_type` Optimum already covers partially | mgp-str: Optimum covers `feature-extraction` only; add `image-to-text` task with 3-head outputs by subclassing `MgpstrOnnxConfig`. Marian/bart KV-cache overrides (Optimum has the task; winml replaces the partial for HTP-friendly cache shape) |
| **L1 — Per-architecture code from scratch** | New file under `models/hf/` that writes an `OnnxConfig` against the HF `transformers` source (no vendor base to subclass), plus optionally `@register_composite_model` | `vilt`: not registered anywhere — write `VILTOnnxConfig` from the HF `VILTModel` source. Any `model_type` truly absent from `TasksManager._SUPPORTED_MODEL_TYPE[...]['onnx']` |
| **L2 — Deeper / structural** | Touching [src/winml/modelkit/models/winml/](../../src/winml/modelkit/models/winml/) shared infra, calibration plumbing, custom op handling, or things outside the per-model surface | New `WinMLCompositeModel` sub-pattern (e.g. first true VQA decoder model); architecture needs a non-standard shared `DummyInputGenerator`; tokenization or pre/post-processing not expressible via the existing `InferenceEngine` task spec |

If you find yourself drifting from L0 → L0★, L0★ → L1-light, L1-light → L1, or L1 → L2 mid-session, **stop and re-pick**. Each escalation changes the review surface and the Outcome you owe. **L0★ in particular is the trap**: a contributor commits to L0 ("just a recipe"), discovers no template exists, writes one from scratch, and now also owes a template-publication finding — that's the L0★ contract. **L1-light vs L1 is the other common mis-estimate** — see Step 1 below; many `model_type`s look unregistered to winml's eyes but are already covered by Optimum natively, dropping the work from "write `OnnxConfig` from scratch" to "subclass and override one method" or even to L0★.

> **Batch-mode contract** (see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-010`). If your contribution covers N ≥ 3 models, your PR description MUST include a pre-build N-row tier table classifying every candidate as exactly one of:
> - **RUN** — you committed to building this model in this contribution.
> - **BLOCKED-UPSTREAM** — `winml config` or `winml build` cannot proceed today; cite the error and the upstream gap.
> - **OUT-OF-SCOPE-FOR-TURN** — recognised but explicitly deferred (model size, requires L1+ effort beyond the budget, etc.). Cite why.
>
> A batch contribution that only builds the easy subset without explicit classification of every unbuilt row is the `_meta-007` producer self-grading failure mode at the batch level, and the reviewer agent will REQUEST_CHANGES.

### Goal axis — how will you prove it works

| Tier | What you verify | Pass criterion | Command (run `--help` to confirm flags) |
|---|---|---|---|
| **L0 — Build / config passes + structural validation** | `winml build` produces a valid artifact end-to-end from the recipe, **and** the artifact passes structural validation: loadable via `onnx.load`, IR/opset/input-output names and shapes match the recipe, and (for composite models or new checkpoints in a known family) shapes match a previously-validated sibling checkpoint. Vocab/embedding sizes auto-fill per checkpoint and should be sanity-checked against HF `config.json`. **For artifacts that emit external data** (typical above ~500 MB; default `use_external_data=True` in build/onnx.py) verify the layout: `Get-ChildItem <out>` shows `model.onnx` + UUID-named `.data` files in the SAME directory, NOT scattered in CWD ([`_meta-023`](./skill_meta/findings.json)). | Build prints `✅ Build complete`; `model.onnx` exists in the output dir; `onnx.load` succeeds; printed `(name, shape, dtype)` matches recipe + sibling-checkpoint contract; vocab size = HF config value; `.data` files (if any) sit next to `model.onnx`. | `winml build -c <recipe>.json -m <hf-id> -o <out>/` then verify with `python -c "import onnx; m=onnx.load('<out>/model.onnx', load_external_data=False); print(m.ir_version, [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"`. **Do NOT use `winml inspect` on a built `.onnx`** — `inspect` is HF-model-ID only today (tracked in [skill_meta/findings.json](./skill_meta/findings.json) `_meta-005`); use `winml config -m <artifact>.onnx` if you need a config dump of the artifact. **For recipes whose filename includes `_fp16_`**: also grep emitted initializers for FLOAT16 to confirm the filename isn't lying — recipes with `quant: null` ship fp32 weights regardless of filename ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-014`). |
| **L1 — `winml perf` passes on at least one EP** | Artifact runs on at least one target EP without crashing or massive CPU fallback. **Probe host EP availability first** via `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` before claiming an EP failed — registered-but-broken EPs (DML on hosts without working driver) abort natively with `0xC0000409` STATUS_STACK_BUFFER_OVERRUN and look like recipe bugs but aren't ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-016`). **CPU PASS is the only honest universal floor**; any per-EP L1 claim above CPU MUST attach (a) `get_available_providers()` snapshot, (b) per-EP perf log, (c) classification of failure as host / packaging / recipe. **Special case — special-token-pooling / positional-index models** (NLI heads, BartForSequenceClassification, anything whose forward() does `input_ids.eq(<special_id>).nonzero()[-1]`): `winml perf` uses RANDOM dummy inputs and IGNORES the recipe's `export.input_tensors[*].value_range`, so models that build cleanly may still crash at perf with `Gather indices=-1` ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-017`). Workaround: write a custom Python perf script with real tokenized inputs (template: [temp/bart_mnli_perf.py](../../temp/bart_mnli_perf.py)) — reviewers accept this as L1 evidence in lieu of `winml perf` CLI output. **Big-model L1 obligations**: (a) `winml perf --memory` is default-on per PR#861; capture RAM + (when applicable) VRAM phase deltas alongside latency ([`_meta-024`](./skill_meta/findings.json)); (b) for NPU/GPU FAIL retry with `--ep-options KEY=VALUE` (e.g. QNN `htp_performance_mode=burst`) BEFORE declaring L1 FAIL ([`_meta-026`](./skill_meta/findings.json)); (c) composite models exercise a separate sub-model pathway in `winml perf` per PR#866 — run perf on the composite (`-m <hf-id> --task <task>`) at least once in addition to per-component artifact perf, to validate the composite path itself. | Latency reported (Avg / P50 / P90 / P99 / Throughput) **+ RAM/VRAM phase deltas** for artifacts > 500 MB; no fatal errors; partition coverage acceptable; failed EPs explicitly classified; `--ep-options` retry attempted for NPU/GPU FAIL before downgrading. | `winml perf -m <artifact>.onnx --device <target> --ep <ep> --iterations <N> --warmup <K>` (memory captured by default). Add `--ep-options htp_performance_mode=burst` etc. as needed. Default iterations=100/warmup=10; for big graphs (>200 MB) drop to 20–30 / 3–5 to keep wall-time bounded. For special-token-pooling models: hand-written script via `onnxruntime.InferenceSession` + `AutoTokenizer`. |
| **L2 — Delta vs. original PyTorch (ad-hoc; CLI gap)** | Cosine / SQNR / max-abs delta of ONNX output against the HF PyTorch reference on a fixed input. **For composite seq2seq / decoder-with-past graphs**: a single-step decoder smoke-test with zero-filled KV is NOT apples-to-apples vs. PT prefill — feed identical KV state on both sides or compare full generate loops. Encoder-side L2 is straightforward and gives a clean numerical-correctness signal even when decoder L2 needs more harness work. **Hand-written L2 scripts that re-export ONNX** (rare but possible) must resolve output paths to absolute before calling `torch.onnx.export` for >2GB models — PR#853 fixed this inside `HTPExporter` but external scripts can still leak UUID `.data` files into CWD ([`_meta-023`](./skill_meta/findings.json)). | FP16 cosine ≥ 0.99 · W8A16 ≥ 0.95 · W8A8 ≥ 0.90. **Encoder cosine ≈ 1.0 + max-abs ≤ 1e-3 vs PyTorch is sufficient to prove the export is numerically correct** when decoder L2 is harness-blocked. | *(Pending CLI support — see "Feature gap: PyTorch-reference compare" below.)* Reference template: [temp/fr_en_l2_compare.py](../../temp/fr_en_l2_compare.py) (transformers + onnxruntime, ad-hoc script per recipe, save log next to the script). |
| **L3 — Task-metric accuracy** | Top-1 / F1 / mAP / BLEU / chrF / similarity within acceptable drop from FP32 reference on a real dataset. **First check `winml eval --schema --task <your-task>`** — if the task is not in the supported list (16 entries as of 2026-06-22, **none of them generative text-to-text**), L3 is structurally CLI-blocked for this recipe and reviewers MUST NOT penalize the contribution for missing L3 evidence ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-015`). **L3 has three verdict states, not two**, per [`_meta-029`](./skill_meta/findings.json): `PASS` (within tolerance), `FAIL-correctness` (accuracy drop exceeds spec — investigate quant / calibration / op fallback), `TIMEOUT-at-scale` (eval times out on this EP for this big model — drop a `<model>/<task>_eval_result.timeout` empty marker, file as data not regression; xlm-roberta-large fill-mask on DML is the canonical case). | Within spec, documented in PR. If CLI-blocked: cite the unsupported-task error verbatim and file a feature-gap issue against the TASK_REGISTRY. If TIMEOUT: cite EP + wall-time cap + `.timeout` marker path. | `winml eval -m <artifact>.onnx --model-id <id> --task <task>`. Probe first: `winml eval --schema --task <task>` returns the supported-task list on failure. |

Goal tiers are **cumulative in intent but independently verified**: each row can be checked without the row above it. Pick the highest tier you can honestly commit to before you start; downgrade publicly if blocked rather than silently skipping. **The honest ceiling is whatever the host + CLI lets you reach** — for some recipes that's only `(L0, L1-CPU)`, and that's a complete contribution; claiming more without per-tier evidence is the [`_meta-007`](./skill_meta/findings.json) self-grading failure mode at the Goal level.

> **March rule — the Goal ladder is a contract, not a menu** ([`_meta-018`](./skill_meta/findings.json)). Once a Goal ceiling is committed at session start, the producer MUST attempt **every tier from L0 up to that ceiling in a single uninterrupted pass**, and MUST emit a per-tier verdict for each — exactly one of `PASS` (with numbers), `CLI-BLOCKED` (with the unsupported-task / unsupported-flag error verbatim + feature-gap filing), `HOST-BLOCKED` (with `get_available_providers()` snapshot + classification of the failure as host/packaging — `_meta-016`), or `FAIL → downgrade Goal ceiling to Lk` (with the failing artifact + a follow-up finding). **Stopping mid-ladder to ask the user "should I continue to Lk+1?" is itself the failure mode** — it produces the same silent under-claim as `_meta-007` and `_meta-006`, just in the producer→user direction instead of the producer→reviewer direction. The only acceptable mid-ladder pause is when the producer's tool budget is genuinely exhausted (long-running build over its time cap, missing host hardware) — and even then the pause MUST be a *report* with explicit `BLOCKED` verdict, not a *question*.
>
> **Short-circuit rule — `FAIL` halts the march, `BLOCKED` does not** ([`_meta-018`](./skill_meta/findings.json)). If tier `Lk` returns a hard `FAIL` (build crashed, perf segfaulted, cosine < threshold, eval accuracy collapsed) the producer MUST stop the march, downgrade the Goal ceiling to `L(k-1)`, and emit a follow-up finding explaining the failure. Tiers above `Lk` are NOT attempted: their evidence would be meaningless without the lower foundation (an L2 cosine number on a model whose L1 perf crashes proves nothing about the artifact's real-world correctness; an L3 eval metric on a model whose L0 build silently shipped fp32 weights despite an `_fp16_` filename is actively misleading). `BLOCKED` verdicts are different — they reflect environment limits, not artifact failure — and do NOT halt the march: a recipe whose L3 is `CLI-BLOCKED` (task not in `TASK_REGISTRY`) can still legitimately ship L2 evidence from an ad-hoc script, because the artifact itself is sound. Concretely: `L0 PASS → L1 FAIL → STOP` (downgrade ceiling to L0, finding documents the L1 crash); `L0 PASS → L1 PASS → L2 BLOCKED → L3 PASS` is fine (L2 blocked by harness, not artifact; L3 still meaningful). Recording an `L_{k+1} PASS` after an `Lk FAIL` is the same self-grading dishonesty as `_meta-007`.

> **L2 — feature gap.** Direct PyTorch-vs-ONNX numerical compare is not a first-class `winml` mode today; `winml eval --mode compare` compares ONNX-to-ONNX (e.g. quantized vs. FP32 ONNX). Until a `--reference pytorch` mode exists, **L2 is best-effort** — either (a) approximate by comparing your quantized ONNX to your own FP32 ONNX export (which folds export error into the baseline, masking it), or (b) write a one-off comparison script in `temp/` and report numbers in the PR. Either way, **file the gap** as part of the L1 Outcome (below).

> **L3 — task-registry coverage is a structural gate.** `winml eval`'s TASK_REGISTRY as of 2026-06-22 covers 16 tasks (mostly classification + extractive); generative text-to-text tasks (`translation`, `summarization`, `text2text-generation`) are NOT registered. Every seq2seq translation / summarization recipe is L3-CLI-blocked no matter how good the recipe is. Probe via `winml eval --schema --task <task>` BEFORE planning L3 evidence; if blocked, downgrade publicly and file the gap.

### Outcome axis — what you ship

Every tier ships **both** a code/recipe deliverable **and** a structured contribution report. The report IS the PR description — there is no "PR description vs. report" split. A contribution that produced artifacts but no PR-description-shaped report is half-shipped (the next reader can't verify the claim without re-running everything).

| Tier | Code/recipe deliverable | Contribution report (= PR description) |
|---|---|---|
| **L0 — Recipe + artifacts** | New recipe JSON under `examples/recipes/<org>_<model>/`; **row added to [examples/recipes/README.md](../../examples/recipes/README.md) index table** (a recipe nobody can find via the index is half-shipped, see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-006`); built artifacts under a stable output dir | **A PR description carrying all 9 hand-off items from Step 6** (recipe path / README row / build dir / build log / appended findings / Optimum probe / claimed (E,G,O) / Goal-ladder verdict table / methodology-evolution declaration). Numbers pasted, not paraphrased. Failing to include the PR description = REQUEST_CHANGES on hand-off, regardless of how good the artifact is. |
| **L1 — L0 + code + gap issues** | Everything in L0, plus: source-code changes under `models/hf/`, one filed feature-request issue per gap you hit (missing op coverage, missing PyTorch-compare mode, missing calibration shape, etc.) | L0 report, plus: per-finding entry in `model_knowledge/<family>.json` (Step 4); each `feature_gaps_filed[]` entry either a real issue URL or a `FILE:` TODO surfaced in the PR description |
| **L2 — L1 + new task family** | Everything in L1, plus: a new `TASK_REGISTRY` entry (or task variant), possibly a new shared-infra file under `models/winml/<task>.py` | L1 report, plus: a finding in `skill_meta/` documenting the new task-family pattern so the next "first model in this task" contributor doesn't redesign from scratch. The first VQA / first audio-LM / first speech-translation contribution is L2. |

Mapping:

- Effort L0 or L0★ ⇒ Outcome L0 (L0★ additionally owes a `recipe_template` finding update in `model_knowledge/`)
- Effort L1 ⇒ Outcome L1 always — if you touched code, you owe the feature-request issues and the knowledge-base append
- Effort L2 ⇒ Outcome L2 — structural changes always come with a task-family or pattern-family finding

> **One-PR-per-composite rule** ([`_meta-020`](./skill_meta/findings.json)): encoder + decoder of a composite recipe pair (translation, image-to-text, summarization, …) ship as a **single PR with a single report** covering both halves in one Goal-ladder verdict table. Splitting enc/dec into two PRs is REQUEST_CHANGES — the composite contract treats them as one shippable unit. The verdict-matrix rows expand per-half inside the single report.

> **Report location**: PR descriptions on GitHub are ephemeral. For local/offline work or for skill-evolution audits, also drop a mirror copy under `research/adding-model-support/iter<N>_reports/PR_<org>_<model>.md` so future contributors can read the report without GitHub access. The mirror copy and the PR description must be byte-identical at hand-off.

> **The PR is shipped by the producer, not by the user** ([`_meta-033`](./skill_meta/findings.json)): Outcome at every tier ⇒ an actual git PR opened against `microsoft/WinML-ModelKit`, not a local mirror in `iter<N>_reports/` alone. See Step 7 for the shipment workflow (branch-per-PR, scope rules, push, `gh pr create`). A contribution that produced artifacts + a local mirror but no real PR is half-shipped.

## Where the code lives

| Concern | Path |
|---|---|
| **Per-architecture ONNX export config** | [src/winml/modelkit/models/hf/](../../src/winml/modelkit/models/hf/) — one file per HF `model_type` (`bart.py`, `marian.py`, `depth_pro.py`, `vision_encoder_decoder.py`, …); each registers via `@register_onnx_overwrite(model_type, task, library_name="transformers")` |
| **Composite-model registration** | Same per-architecture files use `@register_composite_model(model_type, task)` to bind user-facing tasks (`translation`, `summarization`, `image-to-text`, …) to a multi-component pipeline (encoder + decoder, prefill + gen). `winml config` emits one recipe per component. |
| **Shared per-task / per-pattern infra** | [src/winml/modelkit/models/winml/](../../src/winml/modelkit/models/winml/) — `encoder_decoder.py`, `decoder_only.py`, `composite_model.py`, `kv_cache.py`, `image_classification.py`, etc. Only touch this layer when no existing pattern fits (Effort L2). |
| **Generic export plumbing** | [src/winml/modelkit/export/](../../src/winml/modelkit/export/) (`pytorch.py`, `io.py`, `value_range.py`) — architecture-agnostic ONNX export. **You almost never edit this for new model support**; the per-architecture work goes in `models/hf/`. |
| **Recipe configs** | [examples/recipes/](../../examples/recipes/) (`<org>_<model>/<task>_<precision>_config.json`) |
| **Loader / task / inference registries** | [src/winml/modelkit/loader/task.py](../../src/winml/modelkit/loader/task.py) (`KNOWN_TASKS`, `TASK_SYNONYM_EXTENSIONS`), [src/winml/modelkit/inference/tasks.py](../../src/winml/modelkit/inference/tasks.py) (`TASK_REGISTRY`) — touched when adding a new task family, not a new model |
| **Self-learning knowledge base** — per-model | [research/adding-model-support/model_knowledge/](./model_knowledge/) — one JSON per HF `model_type`; read before starting, append at the end |
| **Self-learning knowledge base** — about this skill | [research/adding-model-support/skill_meta/](./skill_meta/) — findings about the methodology itself (path drift, missing template patterns, task-family asymmetries). Separate from per-model so the dialectical record of "the skill was wrong about X" doesn't pollute model lookups. |

## Step 1 — Read prior knowledge, then diagnose

**Read first**: open [model_knowledge/](./model_knowledge/) and look for a file matching your architecture family (`vit.json`, `bert.json`, `dinov2.json`, …). If one exists, it tells you which recipes have already been tried, which gotchas hit other contributors, and which `nodes_to_exclude` entries are common for this family. **Treat findings as observational hypotheses, not ground truth** — the same dialectical rule that governs `autoconfig/ep_knowledge/` applies here (see [research/autoconfig/ep_knowledge/README.md](../autoconfig/ep_knowledge/README.md)).

**Then scan repo PRs related to model scale** ([`_meta-019`](./skill_meta/findings.json)). Methodology evolves through merged PRs; SKILL.md may cite removed APIs or pre-refactor behavior. Before relying on a SKILL section, sanity-check it against recent commits:

```powershell
# From repo root. Adjust the alternation pattern for your concern area.
git log --all --oneline -300 |
  Select-String -Pattern "composite|encoder.decoder|external.data|task.resolution|memory|ep.options|scale"
```

Areas to scan if your model is "large or composite" (>500 MB single graph, encoder-decoder, decoder-with-past, dual-encoder, depth/detection heads):

| Area | Representative PRs (as of 2026-06-23) | What changed |
|---|---|---|
| Composite auto-expansion gate | #850 / #862 | `winml config` no-task composite expansion is gated on `WinMLEncoderDecoderModel` subclass AND task ∈ {text2text-generation, image-to-text}, NOT `config.is_encoder_decoder` (BLIP exception). See [`_meta-020`](./skill_meta/findings.json) |
| Optimum task-label correction | #851 | `_upgrade_fill_mask_for_seq2seq` corrects Optimum's `*ForConditionalGeneration → fill-mask` mislabel to `text2text-generation`. See [`_meta-021`](./skill_meta/findings.json) |
| `inspect` / `config` / `build` task agreement | #841 + `tests/integration/test_task_consistency.py` | Architecture-head-aware disambiguation; disagreement = winml bug, not workflow choice. See [`_meta-028`](./skill_meta/findings.json) |
| Task-detection unification | #878 | `detect_task` / `_detect_task_and_class_from_config` / `resolve_task_and_model_class` REMOVED. Single source of truth: `resolve_task(config, *, task=None, model_class=None) -> TaskResolution` in `src/winml/modelkit/loader/resolution.py` (post-merge path — on branches predating #878 the equivalent is `detect_task` / `_detect_task_and_class_from_config` in [src/winml/modelkit/loader/task.py](../../src/winml/modelkit/loader/task.py)). 5-stage pipeline (user override → detection → model class → modality upgrade → composite tag) + `TaskSource` enum + `TaskResolution.composite`. Modality from `main_input_name`, not config field names. See [`_meta-022`](./skill_meta/findings.json) + [`_meta-030`](./skill_meta/findings.json) (branch-state caveat) |
| Composite inspect rendering | #2f688a0a | `winml inspect --format json` gained `pipeline_tasks` (e.g. `['summarization', 'translation']`) + `composite` (component breakdown) for auto-detected composites. See [`_meta-027`](./skill_meta/findings.json) |
| Composite perf pathway | #866 | `winml perf` has a sub-model pathway for composites (duck-typed on `sub_models`); per-component `BenchmarkResult` + `components` JSON output |
| Composite encoder output naming | #863 | `WinMLEncoderDecoderModel` consumes encoder output as `last_hidden_state`; alias-injection in `feature_extraction.py` covers encoders that emit a different name. Hand-written recipes with custom encoder output names are still fragile. See [`_meta-025`](./skill_meta/findings.json) |
| External-data layout for >2GB models | #853 | `torch.onnx.export` for >2GB writes UUID `.data` files RELATIVE to export path; absolute-path fix in `HTPExporter._convert_model_to_onnx`. Hand-written L2/L3 scripts that re-export must call `output_path.resolve()`. See [`_meta-023`](./skill_meta/findings.json) |
| Memory measurement at perf time | #861 | `winml perf --memory` (default-on) reports RAM + VRAM phase deltas. Big-model L1 evidence should include memory. See [`_meta-024`](./skill_meta/findings.json) |
| Runtime EP options | #865 / #889 | `winml perf --ep-options KEY=VALUE` (repeatable) for runtime EP tuning (e.g. QNN `htp_performance_mode=burst`); independent from build-time quant. Try options before declaring L1 FAIL on NPU/GPU. See [`_meta-026`](./skill_meta/findings.json) |
| Eval-time TIMEOUT as data | commit 5e4a9b0a | `<model>/<task>_eval_result.timeout` empty marker files coexist with `*_eval_result.json` PASS files. Big-model TIMEOUT is a tracked third verdict tier. See [`_meta-029`](./skill_meta/findings.json) |

If you find a PR that contradicts SKILL.md or supersedes a `_meta-NNN` finding, **file a new `_meta-NNN+1` in [skill_meta/findings.json](./skill_meta/findings.json) and update the relevant SKILL section in the same PR**.

Then run the **Optimum-coverage probe** — this is the single most important diagnostic and was missing from the first version of this skill (see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-004`). It tells you whether the work is **VENDOR-ONLY** (no winml code needed, L0★ at most), **VENDOR + WINML-OVERRIDE** (winml replaces vendor for HTP-friendliness), **WINML-ONLY** (winml added the task that vendor doesn't have), or truly **UNREGISTERED** (L1 from scratch):

```python
# Run from repo root: uv run python -c "<paste this>"
import optimum.exporters.onnx.model_configs  # force vendor registrations
from optimum.exporters.tasks import TasksManager
from winml.modelkit.export.io import ensure_hf_models_registered

mt = "<your model_type from HF config.json, e.g. 'bart', 'mgp-str', 'm2m_100'>"
vendor = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
ensure_hf_models_registered()
after = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
print({"vendor": vendor, "after_winml": after, "added_by_winml": sorted(set(after) - set(vendor))})
```

**Always probe BOTH the hyphenated and underscored variants** of the `model_type` — Optimum stores `mgp-str` (hyphen) while the underscore-only winml convention may miss it. The same goes for `m2m-100` vs. `m2m_100`.

**Then cross-check the probe's task LABEL against the checkpoint's architecture head** ([`_meta-021`](./skill_meta/findings.json)). The probe answers "does vendor cover (model_type, task)?", NOT "is the task label semantically correct for this checkpoint?". Optimum has known mislabels — `BartForConditionalGeneration` is registered as `fill-mask` (semantically wrong; it's seq2seq generation). WinML's `resolve_task` has a correction layer (`_upgrade_fill_mask_for_seq2seq` from PR#851) that fires only when `config.is_encoder_decoder == True`. To verify the label:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("<your-hf-id>")
print({"architectures": cfg.architectures, "model_type": cfg.model_type, "is_encoder_decoder": getattr(cfg, "is_encoder_decoder", False)})
# Flag if architectures[0].endswith("ForConditionalGeneration") AND probe says "fill-mask"
```

| Probe result for `(model_type, your_target_task)` | Effort tier implication |
|---|---|
| Task in `vendor` and task in `added_by_winml` | impossible (keys can't both be vendor-only and added) — re-run |
| Task in `vendor`, not in `added_by_winml` | **L0★** — Optimum covers it natively. If `models/hf/<model_type>.py` exists and overrides this task, you're getting winml's class instead (keyset-only diff can't show this — check the file directly). Either way, no new export code needed |
| Task in `added_by_winml` | **L0★** — winml registered it. Recipe template may still be missing → owe a `recipe_template` publication |
| Task in neither, but `vendor` covers some other tasks on this `model_type` | **L1-light** — subclass the vendor `OnnxConfig` and override `outputs` / `inputs` / `generate_dummy_inputs` for the new task |
| `vendor == []` and `after_winml == []` | **L1 from scratch** or **L2** if the architecture needs new shared infra |

> **The probe is necessary, not sufficient** (see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-008`). A VENDOR-ONLY verdict only means "the `OnnxConfig` exists" — it does NOT mean the paired `DummyInputGenerator` produces inputs that survive a checkpoint-specific assertion. Two recent counter-examples: `facebook/bart-large-mnli` (`BartForSequenceClassification` pools at last `eos_token_id`; random int32 dummy lacks eos → `index -1` at export); `breezedeus/pix2text-mfr` (vision-encoder-decoder per the probe, but the HF repo lacks `pytorch_model.bin` / `model.safetensors` → loader can't even fetch weights). **Always escalate from probe to a real `winml build` attempt before declaring L0★.**
>
> **The probe is also gated on `winml config` actually emitting a draft** (see `_meta-009`). For image-task models with variable input shapes (pix2struct, donut variants, fuyu) `winml config` may error with "Preprocessors for X need to be available for the ONNX export to infer input static shapes. Got: None" BEFORE any recipe exists. The L0★ path is then closed; downgrade to L1-light effort (hand-write the recipe + thread processor parameters) and capture this as the finding.

Then inspect the model directly:

```bash
winml inspect -m <org/model-id> --format json
```

| `inspect` output | Effort tier implication |
|---|---|
| `loader`, `exporter`, `winml_inference_class` all populated | **L0** or **L0★** depending on whether a recipe template exists for this export pattern |
| `loader` populated, `exporter` empty | **L1-light** (if Optimum covers a sibling task on this `model_type`) or **L1** (if `vendor == []`) |
| All blank, or "unsupported model_type" | **L1** minimum, possibly **L2** if processor/pre-post is non-standard |

For seq2seq / composite models the JSON additionally carries `pipeline_tasks` (e.g. `["summarization", "translation"]`) and `composite` (component breakdown) per [`_meta-027`](./skill_meta/findings.json). Two notes when reading inspect output post-PR#878:

- `task.source` is now a `TaskSource` enum value (`tasks-manager`, `sentinel-default`, `model-id-default`, `wrapped-library`, `hf-task-default`, `user-task`, `user-class`) — not the legacy `TasksManager` / `HF_MODEL_CLASS_MAPPING` strings.
- **Invariant**: `winml inspect -m X`, `winml config -m X`, and `winml build -c <recipe> -m X` MUST resolve the same task for the same input ([`_meta-028`](./skill_meta/findings.json), enforced by `tests/integration/test_task_consistency.py`). If you see disagreement, file it as a bug — DO NOT try to work around it in the recipe.

Save the JSON; cite it in the PR and quote it in your knowledge-base append.

## Step 2 — Add or extend the per-architecture file (Effort ≥ L1 only)

1. **Find the closest existing file** in [src/winml/modelkit/models/hf/](../../src/winml/modelkit/models/hf/). Same family is best (a new ViT variant → start from an existing ViT file); otherwise match by **export pattern** rather than modality:
   - Encoder-only classifier/feature-extractor → [bert.py](../../src/winml/modelkit/models/hf/bert.py) or [convnext.py](../../src/winml/modelkit/models/hf/convnext.py)
   - Vision encoder → [depth_pro.py](../../src/winml/modelkit/models/hf/depth_pro.py) or [convnext.py](../../src/winml/modelkit/models/hf/convnext.py)
   - Text encoder-decoder (seq2seq) → [marian.py](../../src/winml/modelkit/models/hf/marian.py), [bart.py](../../src/winml/modelkit/models/hf/bart.py), or [t5.py](../../src/winml/modelkit/models/hf/t5.py)
   - Vision + text encoder-decoder → [vision_encoder_decoder.py](../../src/winml/modelkit/models/hf/vision_encoder_decoder.py) (covers any HF `VisionEncoderDecoderModel` polymorphically via `PATCHING_SPECS`)
   - Decoder-only LM → [qwen.py](../../src/winml/modelkit/models/hf/qwen.py)
2. **Read it end-to-end** before copying — these files encode subtle assumptions about KV-cache shape (full buffer vs. new-token only), `position_id` vs. `cache_position` ONNX input naming, and which HF model class to wrap. Encoder-decoder files in particular bundle trace-time fixes in `PATCHING_SPECS` that look incidental but are load-bearing.
3. **Implement** the new file:
   - One `OnnxConfig` subclass per (model_type, task) registered with `@register_onnx_overwrite(model_type, task, library_name="transformers")`. Declare `inputs` and `outputs` as `dict[str, dict[int, str]]` with named dynamic axes.
   - For composite models (encoder + decoder, prefill + gen), additionally subclass `WinMLEncoderDecoderModel` / `WinMLCompositeModel` and register with `@register_composite_model(model_type, user_facing_task)`.
   - For shape-driving config, use `NormalizedConfig.with_args(...)` or a custom `NormalizedConfig` subclass (see `_DepthProNormalizedConfig` for the computed-property pattern).
4. **Force the import** so the decorator runs — `models/hf/__init__.py` already wires this; if you add a new file, append it there.
5. **Verify** with `winml inspect -m <model-id> --format json`: `loader`, `exporter`, and `winml_inference_class` should all populate.

Per CLAUDE.md, **no hardcoded model names or per-architecture branching** in shared code paths. New architecture support belongs in a new file under `models/hf/` registered through the decorator, not in `if model_type == "..."` checks scattered across the pipeline.

## Step 3 — Write the recipe

Find the closest recipe in [examples/recipes/](../../examples/recipes/) (same family + same task is best). Copy and adjust.

```json
{
  "export": {
    "opset_version": 17,
    "batch_size": 1,
    "input_tensors": [
      { "name": "pixel_values", "dtype": "float32", "shape": [1, 3, 224, 224], "value_range": [0, 1] }
    ],
    "output_tensors": [ { "name": "last_hidden_state" } ]
  },
  "optim": {},
  "quant": {
    "mode": "qdq",
    "samples": 10,
    "calibration_method": "minmax",
    "weight_type": "uint8",
    "activation_type": "uint16",
    "per_channel": false,
    "symmetric": false,
    "task": "image-feature-extraction",
    "model_name": "<org/model-id>"
  },
  "loader": {
    "task": "image-feature-extraction",
    "model_class": "AutoModel",
    "model_type": "<hf model_type>"
  },
  "eval": {
    "task": "image-feature-extraction",
    "dataset": { "path": "<hf dataset>", "split": "test", "samples": 1000 }
  }
}
```

> **Real schema, not a sketch.** Recipes are `WinMLBuildConfig` instances ([src/winml/modelkit/config/build.py](../../src/winml/modelkit/config/build.py)). Top-level keys: `loader` (required), `export` (object or `null`), `optim` (object, defaults filled by autoconf), `quant` (object or `null`), `compile` (object or `null`), `eval` (object or omitted). Both `compile` and `eval` were historically undocumented here — `winml config` emits `compile` and omits `eval` by default; existing recipes vary. The reviewer for marian-003 flagged a previous version of this row that listed only `{export, optim, quant, loader, eval}` as wrong, leading to spurious schema-violation reports — see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-012`.

Conventions:

- Path: `examples/recipes/<org>_<model>/<task>_<precision>_config.json`
- Precision suffix follows the `quant` section: `fp16` / `w8a16` / `w8a8`. Ship at least `w8a16` if the model can quantize; ship `fp16` as well if it's NPU-targeted.
- Keep `samples` low (10–32) in the checked-in recipe. Full calibration is a user concern.
- **Composite models (encoder-decoder / prefill+gen) emit TWO recipe files per `winml config` call** — one per sub-component, e.g. `translation_fp16_encoder_config.json` + `translation_fp16_decoder_config.json`. Today no encoder-decoder recipe ships under `examples/recipes/` (every recipe there is encoder-only); the first seq2seq contributor pays the template-creation cost, and that cost should be captured as a finding in `model_knowledge/` so the second contributor can copy.
- **Composite-expansion gate**, per [`_meta-020`](./skill_meta/findings.json): `winml config` (no `--task`) auto-emits TWO recipes ONLY when both conditions hold: (a) the resolved class is a `WinMLEncoderDecoderModel` subclass; (b) the resolved task ∈ `{text2text-generation, image-to-text}`. A non-generation head on a seq2seq architecture (e.g. BartForSequenceClassification) is single-recipe. **BLIP is the BLIP exception** — `config.is_encoder_decoder == False` but the model IS composite — so do not rely on `is_encoder_decoder` as the discriminator. Explicit `--task` ALWAYS bypasses auto-detection.
- **Composite encoder output naming contract**, per [`_meta-025`](./skill_meta/findings.json): an encoder whose recipe `output_tensors[*].name` is NOT `last_hidden_state` relies on the alias-injection in `src/winml/modelkit/models/winml/feature_extraction.py` (added PR#863) to be consumable by the encoder-decoder loop. **Safest choice: name the encoder output `last_hidden_state`** in the recipe. Otherwise, verify the alias path covers your chosen name before declaring the recipe done; a runtime `KeyError: last_hidden_state` from the composite class means the alias didn't catch and the recipe needs renaming.
- **Custom-shape models (e.g. DepthPro min 1536², Pix2Struct flattened patches)** — the recipe's `export.input_tensors` must satisfy the architecture's minimum, not the default 224². A recipe that builds with a too-small shape will fail at first inference.

## Step 4 — Capture what you learned (Outcome L1 obligation)

This step is the autoconfig-inspired self-learning loop. After every contribution that produced new information — a recipe that worked, a recipe that didn't, an op that fell back, a precision that broke at Goal L2 — append a finding to the family JSON under [model_knowledge/](./model_knowledge/).

### Mine the build artifacts BEFORE you write the finding

A `winml build` run drops three structured JSONs in the output directory that contain model-specific knowledge you cannot reconstruct from the recipe alone. Read all three (see [skill_meta/findings.json](./skill_meta/findings.json) `_meta-006` for why this is mandatory):

| Artifact | What to extract |
|---|---|
| `<out>/analyze_result.json` | `metadata.operator_counts` (op-type histogram), `metadata.total_operators`, `metadata.unique_operator_types`, and `results[].classification` per EP. **Sanity-check which EP(s) actually ran**: if `runtime_support: false` and every op is `unknown`, the EP wasn't available on the host — the file looks like coverage data but isn't. Re-run with `winml analyze --ep <available-ep>` before drawing conclusions. |
| `<out>/export_htp_metadata.json` | `model.total_parameters` (true param count), `model.total_modules` + `tracing.modules_traced` (trace coverage ratio), `modules.children` (module hierarchy — reveals composite architectures, e.g. "3 independent DINOv2 backbones" is invisible from the `.onnx` alone). |
| `<out>/winml_build_config.json` | Diff against your input recipe — reveals what autoconf filled in (e.g. `optim: {}` becomes `optim: {gelu_fusion: true, matmul_add_fusion: true}`). Anything autoconf chose is the implicit default for this architecture and worth recording. |

These feed the `observation`, `gotchas`, and `recipe_template` fields of your finding with concrete numbers, not paraphrased recollections.

### File layout

```
research/adding-model-support/model_knowledge/
├── README.md                # epistemics warning + schema
├── _template.json           # blank finding skeleton
├── <family>.json            # one per HF model family (vit, bert, dinov2, qwen3, …)
└── ...
```

Filename = lowercase HF `model_type` (`config.json["model_type"]`). One file per architecture family, not per individual checkpoint — checkpoints become entries within the family file.

### Finding schema (mirrors `ep_knowledge`)

```json
{
  "_meta": {
    "family": "dinov2",
    "hf_model_type": "dinov2",
    "models_tested": ["facebook/dinov2-small", "facebook/dinov2-base"],
    "last_updated": "YYYY-MM-DD",
    "epistemics_warning": "Observational findings, not ground truth. Re-validate on new checkpoints / ORT versions / EPs."
  },
  "findings": [
    {
      "id": "dinov2-001",
      "title": "Short, falsifiable claim",
      "observation": "What you ran, what you saw, with concrete numbers and commit/version context.",
      "scope": {
        "validated_on":      ["<org/model-id @ precision @ ep>", "..."],
        "falsified_on":      [],
        "not_yet_tested_on": []
      },
      "effort_tier_required": "L0 | L1 | L2",
      "goal_tier_reached":    "L0 | L1 | L2 | L3",
      "recipe_template":      "examples/recipes/<org>_<model>/<task>_<precision>_config.json",
      "gotchas": [
        "nodes_to_exclude needed for X because Y",
        "calibration sample count below N produces low cosine"
      ],
      "feature_gaps_filed":   ["#1234 — winml eval --reference pytorch"],
      "mechanism_confirmed":  false,
      "mechanism_notes":      "Hypothesis, not proof. What would falsify it.",
      "last_updated": "YYYY-MM-DD"
    }
  ]
}
```

### Rules of engagement (dialectical, like `ep_knowledge`)

1. **Append, don't rewrite.** A new model that contradicts an earlier finding goes into `scope.falsified_on` of the old finding *and* gets a new finding documenting the counter-example. Never delete a refuted finding silently — its existence is evidence about an ORT/SDK era.
2. **One finding per claim.** "DINOv2 needs nodes_to_exclude for LayerNorm at W8A8" and "DINOv2 hits perf parity with FP16 on QNN NPU" are two findings, not one.
3. **Confidence ≠ generality.** A finding can be high-confidence on the one model you tested and still not generalize. Encode reach in `scope`, not in prose.
4. **Cite the artifact.** `observation` must include enough context (model id, recipe path, precision, EP, ORT version where relevant) that another agent can reproduce or refute.
5. **Auto-bootstrap next time.** Step 1 of this skill instructs reading the family file *first*. The whole point is that contributor N+1 starts from contributor N's findings.

### When to create a new family file

- HF `model_type` you've never seen → new file. Use `_template.json` as the starting structure.
- Architecture variant within an existing family (e.g. ViT-22B under `vit`) → new finding inside the existing file.

## Step 4b — Capture methodology learnings (skill-evolution obligation)

`model_knowledge/<family>.json` (Step 4) records what you learned about **the model**. This step records what you learned about **the methodology itself** — SKILL.md, REVIEW.md, the verdict vocabulary, the recipe schema, the CLI surface. Without it, every user lands on the same trap; the skill never gets smarter than its first author.

> **Iteration rule** ([`_meta-031`](./skill_meta/findings.json)): a contribution that produces a working artifact but never edits SKILL.md / REVIEW.md / `skill_meta/findings.json` is presumed to have hit zero methodology friction. The reviewer checks this presumption (see [REVIEW.md](./REVIEW.md) "Methodology-evolution audit"); a producer who silently absorbed a CLI surprise, a doc-code drift, or a new verdict shape is in the same `_meta-007` self-grading failure mode as one who silently skipped Goal-L2.

### Triggers — if any of these fired during your run, you OWE a `_meta-NNN` (and the corresponding SKILL.md / REVIEW.md edit in the same PR)

| # | Trigger | What you ship in addition to the model artifact |
|---|---|---|
| 1 | **CLI surprise** — a command in SKILL.md (or your own muscle memory) failed and you had to discover the correct flag via `--help` or an error message (e.g. `--dataset-config` → `--dataset-name`) | New `_meta-NNN` documenting the wrong-flag → right-flag pair + SKILL.md/REVIEW.md edit to cite the correct flag |
| 2 | **Doc-code drift** — SKILL.md (or REVIEW.md) cites a file path, function, decorator, or output field that no longer exists or has been renamed | New `_meta-NNN` with branch-state classification per [`_meta-030`](./skill_meta/findings.json) + SKILL.md edit to dual-cite pre/post or update to the current name |
| 3 | **Silent-failure mode** — build succeeded, perf succeeded, but the output was subtly wrong (wrong precision, wrong tensor alias, zero-fed cross-attention, mislabeled task) | New `_meta-NNN` documenting the symptom + diagnostic + fix + REVIEW.md row that catches this class going forward |
| 4 | **New verdict shape** — a Goal tier outcome didn't fit `{PASS, CLI-BLOCKED, HOST-BLOCKED, FAIL}` from [`_meta-018`](./skill_meta/findings.json) (e.g. `TIMEOUT-at-scale`, `DEFERRED-HARNESS`) | New `_meta-NNN` extending the verdict vocabulary + SKILL.md Step 0 Goal table edit + REVIEW.md row to validate the new verdict's evidence requirements |
| 5 | **Reviewer found gap** — the reviewer agent flagged a check that REVIEW.md doesn't currently encode | New `_meta-NNN` capturing the missed check + REVIEW.md checklist row added |
| 6 | **Effort mis-estimate** — you committed to L0/L0★/L1-light and ended at L0★/L1-light/L1 (or vice-versa) because the Optimum-coverage probe or the actual code surface contradicted Step 0's classification | New `_meta-NNN` documenting the misclassification signal + SKILL.md Step 0 Effort table edit (add the disambiguator that would have caught it earlier) |
| 7 | **PR-mining discovery** — you read a recent winml PR (per Step 1's PR-mining substep) and found a behavior or check that SKILL.md doesn't yet cite | New `_meta-NNN` per PR + SKILL.md / REVIEW.md edit citing the PR with branch-state classification per [`_meta-030`](./skill_meta/findings.json) |

### Anti-trigger — do NOT bloat findings.json

If NONE of triggers 1–7 fired, you do NOT owe a `_meta-NNN`. A no-friction contribution is a positive signal that the skill is currently calibrated for your tier. Just ship the model artifact + `model_knowledge/<family>.json` finding (Step 4) and hand off. The reviewer will explicitly confirm "no methodology friction observed" rather than `REQUEST_CHANGES`.

### Schema for `_meta-NNN` (mirrors `model_knowledge/`)

Use the same finding schema as Step 4 with these required fields tightened:

- `id`: `_meta-NNN` where `NNN` = `(max existing id) + 1`. Currently next id = **`_meta-031` (post-iter-6, 2026-06-23)** — grep `findings.json` for the actual max before assigning.
- `scope.validated_on`: cite the exact run that surfaced the friction (model id, command, error message or wrong-output diff).
- `scope.refines` / `scope.falsified_on`: if your finding supersedes an existing `_meta-NNN`, name it here. Append, don't rewrite (same rule as Step 4).
- `mechanism_confirmed`: `true` only if you re-ran with the fix and confirmed the friction is gone. Otherwise `false` with hypothesis in `mechanism_notes`.
- `resolution`: name the SKILL.md / REVIEW.md edit you made in the same PR. "To be addressed in a follow-up" is REQUEST_CHANGES at reviewer time — the methodology edit MUST land with the producing PR.

## Step 5 — Common pitfalls (still apply, regardless of tier)

- **New op type not in coverage rules** — run `winml analyze --model <exported>.onnx --ep all --format json` early. If new ops appear unsupported, either it's a coverage data gap (file an issue → counts toward Outcome L1) or you need `nodes_to_exclude` in `quant`.
- **Attention variant (GQA / MQA / MLA)** — validate Goal L2/L3 separately per precision; if cosine drops sharply, add the attention nodes to `nodes_to_exclude` and document why in the knowledge base.
- **Dynamic shapes** — most models want fixed `batch_size: 1`; if dynamic axes are genuinely needed, declare them explicitly in `export.input_tensors`.
- **Non-standard tokenizer / processor** — preprocessing drift is silent and only surfaces at Goal L3.
- **Calibration data quality** — `samples: 10` in the checked-in recipe is a smoke-test default; your own L2/L3 verification should use ≥ 128 representative samples. Don't ship a Goal L2 number measured against 10 samples.

## Step 6 — Hand off to a reviewer agent (do not self-grade)

A contribution is **not done** when the producer thinks it's done. It's done when a **separate reviewer agent** has verified the deliverables against [REVIEW.md](./REVIEW.md). This is structural — not optional politeness.

The two failure modes that motivate this separation are documented in [skill_meta/findings.json](./skill_meta/findings.json):

- `_meta-005`: the producer's first run cited a verification command that didn't actually work; the producer never noticed because they wrote both the command and the report.
- `_meta-006`: the producer's first knowledge-capture only recorded "build succeeded" and missed three structured build artifacts; the producer corrected this only after being externally challenged.

A single-agent loop produces these errors. A two-agent loop (producer + reviewer) catches them by design.

### Producer's hand-off package

Before invoking the reviewer, the producer ensures the PR or workspace contains:

1. The recipe file under `examples/recipes/<org>_<model>/`.
2. The updated row in [examples/recipes/README.md](../../examples/recipes/README.md).
3. The build output directory path (so the reviewer can read `analyze_result.json`, `export_htp_metadata.json`, `winml_build_config.json` directly).
4. The build log (stdout from `winml build`, since exit code is unreliable).
5. The appended finding(s) in `model_knowledge/<family>.json` (and `skill_meta/findings.json` if SKILL.md was edited).
6. The Optimum-coverage probe output (verdict per `_meta-004`).
7. An explicit declaration of claimed `(Effort, Goal, Outcome)` tier in the PR description.
8. **Goal-ladder verdict table** ([`_meta-018`](./skill_meta/findings.json)) — one row per Goal tier from `L0` up to the claimed ceiling, each row carrying exactly one verdict (`PASS` with numbers / `CLI-BLOCKED` with cited error / `HOST-BLOCKED` with `get_available_providers()` snapshot / `FAIL → downgrade Goal ceiling to Lk` with follow-up finding). A hand-off that lists `L0 ✓ L1 ✓` and silently omits L2 and L3 when the ceiling was L3 is REQUEST_CHANGES on hand-off.
9. **Methodology-evolution declaration** ([`_meta-031`](./skill_meta/findings.json)) — a one-line statement in the PR description of either: (a) **"Methodology friction observed: `_meta-NNN..NNN` added"** with the new findings + the SKILL.md/REVIEW.md edits attached to the same PR (per Step 4b triggers 1–7), OR (b) **"No methodology friction observed"** as an affirmative declaration that the producer reflected on triggers 1–7 and none fired. Silence is not acceptable — the reviewer reads silence as "producer skipped Step 4b" and REQUEST_CHANGES.

### Reviewer's contract

The reviewer is bound by [REVIEW.md](./REVIEW.md) and must:

- **Re-run at least one command** from the producer's PR. Verdicts without a re-run are paperwork.
- **Read the 3 build artifacts directly**, not take the producer's summary at face value.
- **Cross-check the claimed Effort tier** against a fresh Optimum-coverage probe.
- **Fail closed**: if a check can't be verified, the answer is REQUEST_CHANGES, not "probably fine".

The reviewer issues APPROVE / REQUEST_CHANGES / REJECT. Only APPROVE closes the contribution.

## Step 7 — Ship the PR (do not wait to be asked)

The Outcome contract ([`_meta-032`](./skill_meta/findings.json)) treats a real GitHub PR as part of the deliverable, not as an optional follow-up the user must request. Producers default to opening the PR; user explicitly says *"don't push yet"* to opt out.

### Two shipment lanes (decide which one applies)

**Lane A — Skill-only updates** (SKILL.md / REVIEW.md / `skill_meta/findings.json` / `research/adding-model-support/iter<N>_reports/`):

- Push directly to the **current working branch** (the producer's skills/research branch — e.g. `shzhen/skills_poc`). No new branch. No separate PR per skill edit.
- **Do NOT run `gh pr create` against `main` for Lane A changes.** The working branch IS the target; the skill content lives on that branch indefinitely and is not staged for merge to `main` unless the user explicitly says so. A producer who opens a Lane A → `main` PR is in `_meta-033` REQUEST_CHANGES — close the PR and leave the push.
- Rationale: methodology evolution is iterative and cross-cuts many contributions. Forcing one PR per `_meta-NNN` finding would shred the dialectical record into unreviewable fragments. Bundling them into a single "snapshot" PR (the failure mode that surfaced this rule, PR #935 closed 2026-06-23) dumps 19+ research files on `main` reviewers who haven't opted in to the methodology debate.
- Reviewer reads the cumulative branch state at the next model-PR hand-off, not via a PR diff on `main`.

**Lane B — New model support** (anything under `examples/recipes/<org>_<model>/` or `src/winml/modelkit/models/hf/<model_type>.py`):

- **Always a new branch off `origin/main`**, naming convention `<author>/add-<org>-<model>-recipe` (or `-codegen` if code was touched). NEVER reuse the working skills branch.
- **Scope = exactly what the contribution needed, nothing more.** Match Effort tier to file set:
  - **L0 / L0★** (recipe-only): `examples/recipes/<org>_<model>/*.json` + the README row. Nothing else. The matching `research/adding-model-support/model_knowledge/<family>.json` append stays on Lane A (working skills branch) UNTIL `research/adding-model-support/` has been accepted to `main` as a separate skill-infra PR. Check `git ls-tree origin/main -- research/adding-model-support/` before staging a knowledge file in a model PR; empty output ⇒ knowledge file goes to Lane A only.
  - **L1** (recipe + per-arch code): all of L0, plus `src/winml/modelkit/models/hf/<model_type>.py` (or edits to an existing one), plus any pytest under `tests/` that exercises the new code path, plus a filed feature-gap-issue URL per gap in the PR description.
  - **L2** (new task family): all of L1, plus `src/winml/modelkit/inference/tasks.py` `TASK_REGISTRY` entry, plus possibly a new `src/winml/modelkit/models/winml/<task>.py`, plus a new `_task-<task-name>-NNN` finding in `skill_meta/`.
- **Composite recipes ship as ONE PR** per [`_meta-020`](./skill_meta/findings.json) — encoder + decoder of the same composite (translation / image-to-text / summarization) share a branch and a PR description; the Goal-ladder verdict table expands per-half inside the single PR.
- **Do NOT include skill-level edits** (SKILL.md / REVIEW.md / `skill_meta/findings.json` outside the per-model `model_knowledge/<family>.json` finding) in a model PR. Those go to Lane A. Mixing the lanes pollutes the diff and forces reviewers to context-switch between code-review and methodology-review modes.

### Shipment commands (Lane B)

From the workspace root, with `gh` authenticated:

```powershell
# 1. Branch off a clean main
git fetch origin main
git checkout -b <author>/add-<org>-<model>-recipe origin/main

# 2. Stage ONLY the scope-relevant files (use explicit paths, never `git add -A`)
git add examples/recipes/<org>_<model>/ examples/recipes/README.md `
        research/adding-model-support/model_knowledge/<family>.json
# add code paths here if L1+; do not catch unrelated edits from the working tree

# 3. Commit with a one-line conventional message
git commit -m "recipe(<model>): add <task> recipe (Goal-Lk PASS on CPU)"

# 4. Push to origin (this repo accepts contributor branches directly; no fork needed)
git push -u origin <author>/add-<org>-<model>-recipe

# 5. Open the PR with the mirror report as body
gh pr create --base main --head <author>/add-<org>-<model>-recipe `
             --title "recipe(<model>): <task> recipe" `
             --body-file research/adding-model-support/iter<N>_reports/PR_<org>_<model>.md
```

### Push-failure escalation

If `git push` is rejected by Microsoft Enterprise SSO / token-lifetime policy (90-day classic-token rule), report the exact stderr to the user and ask: (a) refresh the token, (b) push from a different remote, OR (c) hand the producer-prepared branch over for the user to push manually. Do **not** silently fall back to a local mirror — the Outcome contract is not satisfied until the PR exists.

### Self-check before claiming "done"

- [ ] PR URL returned by `gh pr create` pasted into the producer's hand-off message?
- [ ] PR description = the iter<N>_reports/ mirror file byte-identical at hand-off time (post-review edits are OK to diverge, but Step 6 hand-off requires sync)?
- [ ] PR diff contains exactly the scope-rule files for the claimed Effort tier (no leakage of unrelated working-tree files)?
- [ ] Reviewer agent (Step 6) was given the PR URL, not just the branch?

A producer who declares "done" without a PR URL is in `_meta-007` self-grading failure mode — the user shouldn't have to ask "where's the PR?" any more than they should have to ask "where's the report?".

## Cross-references

- Unsupported ops on the target EP → [check-model-feasibility/SKILL.md](../check-model-feasibility/SKILL.md)
- After Goal L1+ passes on at least one EP → `skills/ship-to-winapp/SKILL.md` (planned)
- Pipeline mental model and `--help`-first discipline → [skills/use-winml-cli/SKILL.md](../../skills/use-winml-cli/SKILL.md)
- Per-model optimization tuning → `skills/autoconfig/SKILL.md` (planned) and [research/autoconfig/](../autoconfig/) for the prior art on dialectical knowledge accumulation
- Meta-rules on writing this SKILL.md → `skills/contributing-a-skill/SKILL.md` (planned)
