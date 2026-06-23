# Reviewing an `adding-model-support` contribution

You are a **reviewer agent**. A separate producer agent has just completed a model-support contribution following [SKILL.md](./SKILL.md). Your job is to independently verify the deliverables match the producer's claimed Effort/Goal/Outcome tier — and **reject the work if they don't**.

## Why an independent reviewer

The methodology has a documented self-grading problem captured in [skill_meta/findings.json](./skill_meta/findings.json):

- `_meta-005` — the producer's first end-to-end run cited a verification command (`winml inspect <artifact>.onnx`) that doesn't actually work; the producer never noticed because they wrote both the command and the verification report.
- `_meta-006` — the producer's first knowledge-capture only recorded "build succeeded" and missed three structured build artifacts containing model-specific knowledge; the producer corrected this only after being challenged.

A separate agent catches these failures because it doesn't share the producer's mental shortcuts. **Fail-closed**: if you can't verify a check from evidence in the workspace or by re-running a command, the answer is "REQUEST_CHANGES", not "probably fine".

## Inputs you receive

1. The PR diff or workspace state at HEAD.
2. The producer's **claimed (Effort, Goal, Outcome) tier** — pull this from the PR description or from the appended `model_knowledge/<family>.json` finding.
3. The model id under contribution.
4. The build output directory (the producer should have referenced this in the PR; if not, that's the first failure).

## Step 0 — Check out the actual PR branch (Lane B reviews)

For a real GitHub PR (`https://github.com/microsoft/WinML-ModelKit/pull/<N>`), the producer's working tree is NOT what the maintainer will merge. The maintainer merges the PR branch into `origin/main`. The reviewer MUST verify from that same state, not from the producer's possibly-dirty working branch.

Protocol (Windows PowerShell, `gh` authenticated):

```powershell
# 1. Capture the producer's working state so it's not lost when switching refs.
git status --short                              # NOTE: any "M" / "??" entries before checkout
git stash push -u -m "reviewer-checkout-pr<N>"  # ONLY if working tree is dirty; skip on clean tree

# 2. Fetch + check out the PR head directly.
gh pr checkout <N>                              # OR: git fetch origin <branch>; git checkout <branch>

# 3. Confirm you are on the PR head and main is the merge base.
git log --oneline -1                            # must match PR's top commit hash
git merge-base HEAD origin/main                 # base for diff scope check below
```

**Diff-scope sanity** (per [`_meta-033`](./skill_meta/findings.json) Lane B rules):

```powershell
git diff --stat origin/main...HEAD              # expect only Effort-tier-matching paths
git diff --name-only origin/main...HEAD         # cite this list in the verdict
```

L0 / L0★ PRs MUST show exactly: `examples/recipes/<org>_<model>/*.json` + `examples/recipes/README.md`. Anything under `src/winml/modelkit/models/hf/`, `tests/`, `research/`, `SKILL.md`, or `REVIEW.md` in an L0/L0★ PR diff is scope leakage = REJECT (or REQUEST_CHANGES if it's a research/ skill file that should have stayed on the producer's Lane A working branch).

**Artifact reuse rule**: if the producer cited a build output dir (e.g. `temp/verify_bart_build/`) built BEFORE the PR branch was cut, the reviewer MUST verify the recipe JSON at PR HEAD is byte-identical to the recipe the producer built from. Run `git diff <cached-build-commit>..HEAD -- examples/recipes/<org>_<model>/`. If non-empty, the cached artifacts are stale and you MUST re-build from the PR-head recipe before scoring L0..L3. If empty, the cached artifacts are valid evidence and full re-build is optional.

**Restore on completion**:

```powershell
git checkout <producer-working-branch>          # e.g. shzhen/skills_poc
git stash pop                                   # if you stashed in step 1
```

A reviewer who scored a PR without running Step 0 cannot distinguish "the recipe works on main + the PR diff" from "the recipe works on the producer's local working tree with 6 months of unrelated edits" — those are different claims.

## Checklist (evidence-based, fail-closed)

Each box requires **a one-line citation**: a file path + line number, a command + observed output, or a commit hash. "Looks fine" is not evidence.

### Outcome-L0 (always required)

- [ ] **PR description (= contribution report) is structured per SKILL.md Step 6 hand-off package (all 9 items)**: recipe path / README row / build output dir / build log / appended findings / Optimum-coverage probe / claimed (E,G,O) tier / Goal-ladder verdict table / methodology-evolution declaration. A PR whose description is a free-form paragraph without these 9 items present (even as "N/A — see ...") is REQUEST_CHANGES at hand-off — the next reader cannot verify the claim without re-running every step. For local/offline PR-less workflows (research turns, internal Q&A), the mirror copy at `research/adding-model-support/iter<N>_reports/PR_<org>_<model>.md` must be byte-identical to what would have been the PR description. Composite contributions (per [`_meta-020`](./skill_meta/findings.json)) ship ONE report covering both halves; splitting into two reports is REQUEST_CHANGES.
- [ ] **Real GitHub PR exists** ([`_meta-033`](./skill_meta/findings.json)) — the producer pasted a `https://github.com/microsoft/WinML-ModelKit/pull/<N>` URL in the hand-off message, not just an `iter<N>_reports/PR_<org>_<model>.md` mirror path. A producer who shipped a local mirror but no PR is in `_meta-007` self-grading at the Outcome contract; REQUEST_CHANGES with "run Step 7 shipment commands and re-hand-off with the PR URL". The only exception: user explicitly said "don't push yet" — in which case the hand-off must surface that opt-out verbatim. Push-failure (Microsoft Enterprise SSO / 90-day token rule) is escalated to user, NOT silently downgraded to mirror-only.
- [ ] **PR scope matches Effort tier** ([`_meta-033`](./skill_meta/findings.json) Lane B scope rules): diff contains exactly what the Effort tier requires. L0/L0★ = recipe JSON + README row + `model_knowledge/<family>.json` only (NO `src/winml/modelkit/models/hf/*.py` edits). L1 = above + per-arch `.py` file + pytest + feature-gap issue URLs. L2 = above + `TASK_REGISTRY` entry + new `models/winml/<task>.py` if applicable + `_task-...` finding. Scope leakage in EITHER direction is REJECT (not REQUEST_CHANGES) — L0★ claim with code edits = dishonest grading; L1 claim with no code = wrong tier. Skill-level edits (SKILL.md / REVIEW.md / `skill_meta/findings.json` outside the per-model knowledge file) in a model PR = REQUEST_CHANGES with "revert these into Lane A on the working skills branch".
- [ ] **Recipe file exists** at `examples/recipes/<org>_<model>/<task>_<precision>_config.json` and follows naming. Cite the path.
- [ ] **Recipe schema correct**: top-level keys are a subset of `{loader, export, optim, quant, compile, eval}` (loader is required; `export`/`quant`/`compile` may be `null`; `eval` may be omitted). Precision suffix in filename matches the `quant` section (e.g. `fp16` ⇒ `quant: null`, `w8a16` ⇒ `quant.activation_type: "uint16"` + `quant.weight_type: "uint8"`). Compare against `winml config` output for the same `(model, task)` if in doubt — the producer should ship what `winml config` emits, then refine.
- [ ] **README index updated**: [examples/recipes/README.md](../../examples/recipes/README.md) table contains a row for this `<model> | <task>`. (Failure mode this catches: producer ships an unfindable recipe — `_meta-006`.)
- [ ] **Build re-runs cleanly** OR build log is committed and shows `✅ Build complete`. If you have the host, re-run `winml build -c <recipe> -m <model-id> -o temp/review_build/` yourself. Do not trust `$LASTEXITCODE` — parse stdout (see `_meta-005`).
- [ ] **Artifact structurally validated**: `python -c "import onnx; m=onnx.load('<out>/model.onnx', load_external_data=False); print(m.ir_version, m.opset_import, [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input], [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])"`. Confirm IR/opset/I/O shapes match the recipe declaration. Do NOT accept `winml inspect <artifact>.onnx` as evidence — that command doesn't support `.onnx` files today.

### Goal-tier verification (whatever the producer claimed)

- [ ] **Goal-L0**: covered by artifact validation above. Additionally: for any recipe whose filename includes `_fp16_`, grep the emitted ONNX initializers for FLOAT16 (`data_type == 10`). Recipes with `quant: null` ship fp32 regardless of filename ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-014`); if filename promises fp16 and 0 FLOAT16 initializers are present, REQUEST_CHANGES (rename file or add quant block).
- [ ] **Goal-L1**: producer pasted `winml perf -m <artifact>.onnx --device <target> --ep <ep>` numbers in PR with per-EP latency. **Minimum honest L1 = pass on at least one EP, normally CPU.** For any EP above CPU the producer claims passed, confirm they attached: (a) `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` output from their host, (b) the actual perf log, (c) classification of each failed EP as **host** / **packaging** / **recipe**. Native crashes (`0xC0000409`) on registered-but-broken EPs are host issues, not recipe issues ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-016`); do not penalize the recipe. If you have the host, re-run on the same EP and confirm within ±20% on a cold cache. **Special-token-pooling models** (NLI heads, BartForSequenceClassification, any forward() that does `input_ids.eq(<special_id>).nonzero()[-1]`): `winml perf` ignores recipe `value_range` and uses random ints, so these models crash at perf even when the recipe builds ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-017`). Accept a custom Python perf script with real tokenized inputs (template: [temp/bart_mnli_perf.py](../../temp/bart_mnli_perf.py)) as valid L1 evidence in lieu of `winml perf` CLI output — do NOT REQUEST_CHANGES for missing CLI perf in this case.
- [ ] **Goal-L2**: PyTorch-vs-ONNX cosine/SQNR pasted with the script that produced them (currently a `temp/` one-off — see SKILL.md). Run the script. **Encoder cosine ≈ 1.0 + max-abs ≤ 1e-3** is sufficient even when the decoder cannot be apples-to-apples compared (decoder-with-past graphs need full generate-loop harness, not single-step zero-KV smoke). Do not REQUEST_CHANGES for missing decoder L2 if encoder L2 passes and the producer cited the harness limitation.
- [ ] **Goal-L3**: task-metric numbers pasted from `winml eval -m <artifact>.onnx --model-id <id> --task <task>`. Re-run, confirm within tolerance. **Probe CLI coverage first**: `winml eval --schema --task <task>` — if the task is not in the supported list (`translation`, `summarization`, `text2text-generation` are NOT registered as of 2026-06-22), L3 is structurally CLI-blocked and the producer is required to cite the unsupported-task error verbatim + file a TASK_REGISTRY feature gap. Missing L3 evidence under this condition is NOT a REQUEST_CHANGES trigger ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-015`).
- [ ] **Goal-ladder coverage** ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-018`): the producer's hand-off MUST contain a per-tier verdict row for **every** tier from `L0` up to their claimed Goal ceiling, each carrying exactly one of `PASS` (with numbers) / `CLI-BLOCKED` (with cited error) / `HOST-BLOCKED` (with `get_available_providers()` snapshot + host/packaging classification) / `FAIL → downgrade ceiling to Lk` (with follow-up finding). A hand-off that reports a subset of tiers and silently omits the rest is REQUEST_CHANGES regardless of how strong the reported tiers look. Equally, a hand-off that ends with the producer *asking* "should I continue to Lk+1?" instead of *reporting* a verdict for Lk+1 is the same failure mode — REQUEST_CHANGES, the producer must run the next tier or attach the explicit `BLOCKED` justification before re-handing off.
- [ ] **Short-circuit honored** ([skill_meta/findings.json](./skill_meta/findings.json) `_meta-018`): if any tier `Lk` in the ladder carries a hard `FAIL` verdict, NO tier above `Lk` may carry a `PASS` verdict — higher-tier evidence on top of a broken lower tier is meaningless or actively misleading (L2 cosine on an artifact whose L1 perf crashes; L3 accuracy on an artifact whose L0 silently shipped wrong-precision weights). The producer's ceiling MUST be downgraded to `L(k-1)` and the finding MUST document the failure. Conversely, `CLI-BLOCKED` / `HOST-BLOCKED` verdicts do NOT trigger short-circuit — a recipe with `L3 CLI-BLOCKED` can still legitimately ship `L2 PASS` from an ad-hoc script. If the table shows `Lk FAIL → L(k+1) PASS`, REJECT.
- [ ] **External-data layout** ([`_meta-023`](./skill_meta/findings.json)): for any artifact above ~500 MB, `Get-ChildItem <out>` MUST show `model.onnx` plus the UUID-named `.data` files in the SAME directory. If `.data` files exist in CWD (or elsewhere) instead of next to the model, the export wrote them wrong (pre-PR#853 bug or a hand-written L2 script that didn't `.resolve()` the output path) — REQUEST_CHANGES.
- [ ] **Memory evidence for big-model L1** ([`_meta-024`](./skill_meta/findings.json)): for artifacts > 500 MB, the L1 perf log MUST include the `--memory` lines (RAM phase deltas, plus VRAM if the device has dedicated memory). `winml perf --memory` is default-on per PR#861; an L1 log that suppresses memory (e.g. via `--no-memory`) for a big model is REQUEST_CHANGES unless the producer explains why.
- [ ] **`--ep-options` retry before NPU/GPU L1 FAIL** ([`_meta-026`](./skill_meta/findings.json)): if the producer reports L1 FAIL on QNN / OpenVINO / DML for a big model, verify they attempted at least one documented runtime option (e.g. QNN `--ep-options htp_performance_mode=burst`) BEFORE declaring FAIL. A "FAIL with default options only" verdict on NPU/GPU is REQUEST_CHANGES — the producer must retry with tuned options or document why retry is structurally impossible.
- [ ] **Composite gate consistency** ([`_meta-020`](./skill_meta/findings.json)): for any seq2seq / encoder-decoder recipe, verify `winml config` (no `--task`) auto-emitted the expected recipe count for the (class, task) pair. Two recipes (encoder + decoder) should appear ONLY when the resolved class is a `WinMLEncoderDecoderModel` subclass AND task ∈ {text2text-generation, image-to-text}. BartForSequenceClassification on text-classification → single recipe. BLIP captioning → composite despite `config.is_encoder_decoder == False`. A producer who manually hand-stitched two recipes for a non-composite-expansion case has either chosen the wrong task tag or worked around a real bug — investigate.
- [ ] **Composite encoder output naming** ([`_meta-025`](./skill_meta/findings.json)): for composite encoder recipes, the `output_tensors[*].name` should be `last_hidden_state` OR the producer should have verified the alias-injection in `feature_extraction.py` covers their chosen name. A recipe declaring a custom encoder output name with no alias-path verification is REQUEST_CHANGES — runtime composite loop will break with `KeyError: last_hidden_state` and that bug won't surface in single-component perf logs.
- [ ] **Task-consistency invariant** ([`_meta-028`](./skill_meta/findings.json)): for the same `(model-id, optional --task, optional --model-type)` tuple, `winml inspect`, `winml config`, and `winml build` MUST resolve the same task (post-PR#878, enforced by `tests/integration/test_task_consistency.py`). If the producer's evidence shows the three disagreeing, that's a winml bug — the producer should NOT have shipped a workaround. REQUEST_CHANGES with "file the inconsistency as a bug; do not paper over it in the recipe".
- [ ] **L3 TIMEOUT verdict** ([`_meta-029`](./skill_meta/findings.json)): if a big-model L3 result is `<model>/<task>_eval_result.timeout` (empty marker file), accept this as a third L3 verdict tier (not FAIL). Confirm the producer attached: (a) the EP on which timeout occurred, (b) the wall-time cap that was exceeded, (c) per-EP differentiation (xlm-roberta-large fill-mask is PASS on QNN GPU, TIMEOUT on DML GPU — the marker is EP-specific). Missing any of these → REQUEST_CHANGES.

### Outcome-L1 add-ons (only if code was touched)

- [ ] Code lives in [src/winml/modelkit/models/hf/](../../src/winml/modelkit/models/hf/)`<model_type>.py`, **NOT** under [src/winml/modelkit/export/](../../src/winml/modelkit/export/). Cite the file path.
- [ ] `@register_onnx_overwrite` (and `@register_composite_model` for composite) is present. The decorator runs: `models/hf/__init__.py` imports the new module. Verify by `grep <module_name> src/winml/modelkit/models/hf/__init__.py`.
- [ ] **No hardcoded model branching** anywhere in shared code paths. `grep -rn 'if model_type ==' src/winml/modelkit/` should show no new entries outside the per-arch file.
- [ ] **Per CLAUDE.md**: pytest covers the new code; no `@pytest.mark.skip` / `xfail` added except for hardware/EP gates. Run the affected pytest scope and paste exit code.
- [ ] **Feature-gap issues filed**: every entry in the finding's `feature_gaps_filed[]` array has either an issue URL or a "FILE:" prefix indicating a TODO. If only "FILE:" entries are present, this is REQUEST_CHANGES.

### Outcome-L2 add-ons (only if a new task family was added)

- [ ] New `TASK_REGISTRY` entry in [src/winml/modelkit/inference/tasks.py](../../src/winml/modelkit/inference/tasks.py).
- [ ] Shared infra under `src/winml/modelkit/models/winml/<task>.py` if the architecture introduces a new export pattern.
- [ ] [skill_meta/findings.json](./skill_meta/findings.json) has a finding documenting the new task-family pattern (id like `_meta-NNN` or `_task-<task-name>-001`).

### Knowledge-capture audit (where producers fail — `_meta-006`)

This section is the hardest to fake and most often skipped. Treat it as load-bearing.

- [ ] **`scope.validated_on` is populated** with at least one entry of the form `<model-id> @ <precision> @ <ep>`. If it's empty, the finding is diagnostic-only and the producer hasn't done what they claimed.
- [ ] **Finding cites `analyze_result.json`**: at minimum `metadata.total_operators` and the top 3 op types by count. Also: which EP(s) the analyze actually ran against. If every op is `unknown` and `runtime_support: false`, the analyze data is useless — REQUEST_CHANGES with "re-run analyze against an available EP".
- [ ] **Finding cites `export_htp_metadata.json`**: `model.total_parameters`, `tracing.modules_traced / model.total_modules` (trace coverage ratio), and at least the top-level module hierarchy (composite architectures are invisible from the `.onnx` alone).
- [ ] **Finding cites `winml_build_config.json` autoconf diff**: what `optim` passes did autoconf choose vs what the producer wrote? Anything autoconf filled in is implicit default knowledge worth recording.
- [ ] **Effort tier in the finding matches the Optimum-coverage probe result.** Re-run the probe from SKILL.md Step 1:

  ```python
  import optimum.exporters.onnx.model_configs
  from optimum.exporters.tasks import TasksManager
  from winml.modelkit.export.io import ensure_hf_models_registered
  mt = "<model_type from HF config.json>"
  vendor = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
  ensure_hf_models_registered()
  after = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
  print({"vendor": vendor, "after_winml": after, "added_by_winml": sorted(set(after) - set(vendor))})
  ```

  Cross-check against SKILL.md's verdict table. If the producer claimed L1 but the probe shows VENDOR-ONLY (L0★), REJECT — they did unnecessary work. If they claimed L0★ but the probe shows UNREGISTERED, REJECT — they shipped a recipe without the code that makes it work.

### Methodology-trap audit (specific failures `_meta-001` through `_meta-013` caught)

- [ ] Did the producer **run the Optimum-coverage probe** (Step 1)? Verify by asking for the probe output, OR by re-running it yourself and confirming the verdict drives the claimed Effort tier.
- [ ] If the PR cites `winml inspect <artifact>.onnx` as evidence, **REJECT** — that command refuses ONNX files (`_meta-005`). The producer didn't actually run their own verification.
- [ ] If the PR claims success based on `$LASTEXITCODE = 0`, **REJECT** — exit code is unreliable due to benign EP DLL load failures (`_meta-005`). The producer must parse stdout for `✅ Build complete`.
- [ ] If the producer's finding contains the phrase "build succeeded" with no concrete numbers from the 3 artifacts, **REQUEST_CHANGES** with reference to `_meta-006`.
- [ ] **L0★ → build-failure trap** (`_meta-008`): if the producer claims L0★ based on the Optimum-coverage probe, the reviewer MUST re-run `winml build` end-to-end. Probe coverage ≠ build success. Recent counter-examples: bart-large-mnli (eos-pooling assertion), pix2text-mfr (non-standard checkpoint repo).
- [ ] **`winml config` dead-end trap** (`_meta-009`): if the producer's finding says `winml config` refused to emit a draft, the reviewer MUST confirm whether the producer attempted any workaround (hand-written recipe, `--shape-config`, alternative task). A negative finding without ANY workaround attempt or explicit downgrade to "blocked pending upstream feature" is REQUEST_CHANGES.
- [ ] **Known-broken recipe convention** (`_meta-013`): if the producer ships a recipe that is intentionally known-broken (regression coverage), it MUST be marked. Currently accepted markers: top-level `"_status": "BROKEN — ..."` field in the JSON (silently ignored by `WinMLBuildConfig.from_dict`), OR location under `examples/recipes/_broken/`. A broken recipe with no marker is REQUEST_CHANGES.
- [ ] **Batch-mode contract** (`_meta-010`): if the contribution covers N ≥ 3 models, the PR description MUST contain a pre-build N-row tier table classifying every candidate as one of {RUN, BLOCKED-UPSTREAM, OUT-OF-SCOPE-FOR-TURN}. A batch contribution that only built the easy subset without explicit classification of the unbuilt rows is REQUEST_CHANGES.
- [ ] **Reviewer tool budget** (`_meta-011`): if you (the reviewer) lack terminal-execution capability, you CANNOT satisfy the "re-run at least one command" rule above. State this limitation explicitly in the verdict; the producer should escalate to a reviewer agent with terminal access OR commit the build log + artifact stat snapshot the reviewer can verify by reading files.
- [ ] **Analyze parquet rules available** (`_meta-012`): if your verdict depends on re-running `winml analyze`, confirm `src/winml/modelkit/analyze/rules/runtime_check_rules/*.parquet` is non-empty on the host. If the directory contains only `README.md`, the analyze step cannot be re-run today on external hosts; downgrade to "verified from producer's checked-in `analyze_result.json` only" and file the host-onboarding gap.

### `skill_meta/` review (only if SKILL.md itself was edited)

- [ ] Any change to SKILL.md is accompanied by a corresponding `_meta-NNN` finding in [skill_meta/findings.json](./skill_meta/findings.json) explaining what was wrong and what's now resolved. Dialectical record per the autoconfig pattern.
- [ ] The new SKILL.md content was **exercised at least once end-to-end** — paper edits without a real run are how the methodology grades itself. Demand the build log.

### Methodology-evolution audit ([`_meta-031`](./skill_meta/findings.json))

This is the load-bearing check that turns one-off model contributions into skill-level evolution. A producer who shipped a working model without editing SKILL.md / REVIEW.md / `skill_meta/findings.json` is presumed friction-free; a producer who hit friction and silently absorbed it is in `_meta-007` self-grading failure mode.

- [ ] **PR description carries a methodology-evolution declaration** (per SKILL.md Step 6 hand-off item #9): either (a) `"Methodology friction observed: _meta-NNN..NNN added"` with the new findings + matching SKILL.md/REVIEW.md edits in the same PR, OR (b) `"No methodology friction observed"` as an affirmative declaration. Silence → REQUEST_CHANGES, the producer must reflect on Step 4b triggers 1–7 and answer one way or the other.
- [ ] **If declaration (a)**, audit each new `_meta-NNN` against the Step 4b trigger taxonomy: is it trigger #1 CLI-surprise / #2 doc-code-drift / #3 silent-failure / #4 new-verdict / #5 reviewer-found-gap / #6 effort-mis-estimate / #7 PR-mining? A `_meta-NNN` that maps to none of the seven triggers is either off-topic (belongs in `model_knowledge/<family>.json` instead) or genuinely opens an 8th trigger — in the latter case the producer owes a SKILL.md Step 4b table-edit adding the new trigger row. Cite which trigger each finding satisfies.
- [ ] **If declaration (a)**, verify the SKILL.md / REVIEW.md edits actually landed in the same PR. "Methodology fix will follow in a separate PR" is REQUEST_CHANGES — the methodology evolution MUST be PR-bundled with the contribution that surfaced the friction; otherwise the next user steps on the same trap before the fix lands.
- [ ] **If declaration (b)**, sanity-check that no friction signals leaked into the build log / chat transcript / PR commit history: producer running `winml ... --help` mid-PR, producer writing custom Python wrappers around `winml perf`/`winml eval`, producer hand-stitching recipes that `winml config` should have auto-emitted, producer's reviewer-handoff package missing 1+ Step 6 items. If any of these are present, the declaration is dishonest — REQUEST_CHANGES with citation.
- [ ] **Dead-link check** (per [`_meta-030`](./skill_meta/findings.json)): pick 3 random `[...](path)` links from any SKILL.md / REVIEW.md / `findings.json` edit the producer made. `Test-Path <path>` (PowerShell) or `[ -f <path> ]` (bash) on the producer's branch. Any dead link without an explicit AHEAD-ON-MAIN / IN-BRANCH / HISTORIC classification in the surrounding text is REQUEST_CHANGES.

## Verdict format

Produce one of:

- **APPROVE**: every applicable box ticked with one-line evidence. Sign-off includes the build log path or commit hash you re-ran.
- **REQUEST_CHANGES**: bullet list of every unticked box with the producer-actionable fix. Include the file/line where the missing evidence should land.
- **REJECT**: structural failure — wrong effort tier, fabricated numbers (verification command that can't actually run), or entire deliverable missing. Cite the SKILL.md row or `_meta-NNN` finding that established the requirement.

## What this reviewer does not check

- Code style / formatting (lint catches it).
- Subjective architecture preferences (the existing `models/hf/<file>.py` is the prior art; if the new file deviates substantially, raise it but don't reject on it alone).
- Performance vs. competitors (out of scope; Goal-L1 only requires "passes on one EP", not "fastest").

## Self-check before issuing a verdict

- Did you re-run **any** command from the producer's PR yourself? If not, your verdict is paperwork, not review.
- Did you read the build artifacts (`analyze_result.json`, `export_htp_metadata.json`, `winml_build_config.json`) directly, or only take the producer's word for what's in them? Reading is the bar.
- If you found nothing wrong, do you know what you would have looked for if you had? If the answer is "I would have looked at the build artifacts but didn't", upgrade your verdict to REQUEST_CHANGES.
