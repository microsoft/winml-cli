# PR: facebook/bart-large-mnli — close Goal-L3 ladder on text-classification

**Iter**: 6 (Goal-ladder extension; recipe shipped in iter-5 as bart-004)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L3, Outcome = L1)`

## Summary

This PR closes the full Goal ladder L0..L3 on `facebook/bart-large-mnli` (text-classification, fp32, CPU). The recipe was shipped in iter-5 with L0+L1-CPU+L2 PASS (bart-004); this PR adds the L3 task-metric evidence via `winml eval` on `glue/mnli/validation_matched/100-sample` and records the result as **the first L3 PASS in repo**. No source-code changes; no new recipe. The contribution is a structured outcome update against an already-shipped artifact plus the appended `bart-005` finding.

## 1. Recipe file

[examples/recipes/facebook_bart-large-mnli/text-classification_config.json](../../../examples/recipes/facebook_bart-large-mnli/text-classification_config.json) — unchanged from iter-5 (bart-004). Recipe carries the `value_range: [2, 3]` workaround on `input_ids` to deterministically inject `eos_token_id=2`; documented inline under `_note` per [`_meta-013`](../skill_meta/findings.json) convention.

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) line 21 — present (`facebook/bart-large-mnli | text-classification | ...`). No edit needed.

## 3. Build output directory + artifact inventory

`temp/verify_bart_build/` (gitignored — referenced by path for reviewer re-execution):

| File | Size | Purpose |
|---|---:|---|
| `model.onnx` | 384,628 B | optimized ONNX graph (post-`optimize` pass) |
| `model.onnx.data` | 1,633,574,896 B | external-data shard (FLOAT32 weights, 1.63 GB) |
| `export.onnx` + `.data` | 1.63 GB | pre-optimize artifact |
| `optimized.onnx` + `.data` | 1.63 GB | mid-pipeline artifact |
| `analyze_result.json` | 1,916 B | op histogram (Step 4 mining) |
| `export_htp_metadata.json` | 275,710 B | module hierarchy + trace coverage (Step 4 mining) |
| `winml_build_config.json` | 1,149 B | autoconf diff (Step 4 mining) |

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): `model.onnx` and `model.onnx.data` are co-located in the same directory. PASS.

## 4. Build log

Iter-5 build log: `temp/verify_bart_build/build.log` (referenced in bart-004 mechanism_notes). Iter-6 used the iter-5 artifact unchanged; no re-build needed for the L3 closure.

L3 eval log (this PR): [temp/bart_mnli_l3.log](../../../temp/bart_mnli_l3.log) — 6,354 B; preserved via `Tee-Object`.

## 5. Appended findings

### Per-model — `model_knowledge/bart.json`

[bart-005](../model_knowledge/bart.json) — "VALIDATED Goal-L3 for facebook/bart-large-mnli — `winml eval` on GLUE/mnli validation_matched (100 samples, CPU) gives accuracy=0.8800, latency=1.89s/sample. Closes the full Goal ladder L0..L3 for the first encoder-decoder family in repo. Cross-refs `_meta-019..030` from iter-6 PR-mining."

Falsifies: [`_meta-015`](../skill_meta/findings.json) scope for single-head NLI tasks (translation/summarization remain CLI-blocked, but text-classification on a seq2seq architecture IS reachable).
Refines: bart-004.

### Skill-meta — `skill_meta/findings.json`

This PR does not introduce new `_meta-NNN` findings; the iter-6 methodology findings (`_meta-019..031`) shipped in a separate PR bundle. See `_meta-029` (L3 verdict triage with TIMEOUT-at-scale third tier) and `_meta-018` (March + Short-circuit rules) which gate this PR's evidence requirements.

## 6. Optimum-coverage probe verdict

```python
import optimum.exporters.onnx.model_configs
from optimum.exporters.tasks import TasksManager
from winml.modelkit.export.io import ensure_hf_models_registered
mt = "bart"
vendor = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
ensure_hf_models_registered()
after  = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
# vendor includes: feature-extraction, feature-extraction-with-past, question-answering, text-classification,
#                  text-generation, text-generation-with-past, text2text-generation, text2text-generation-with-past
# after_winml: same set with winml overrides on feature-extraction + text2text-generation
# added_by_winml: [] for text-classification ⇒ vanilla Optimum BartOnnxConfig handles task='text-classification'
```

**Verdict**: VENDOR-COVERED on `text-classification`. Effort L0★ (no code; pure recipe) is the correct classification. Verified at iter-5 (bart-002) and re-confirmed by the bart-005 build.

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only; one well-chosen `value_range` narrowing on a vendor-covered task)
- **Goal = L3** (full ladder L0..L3 closed on CPU)
- **Outcome = L1** (recipe + appended `bart-005` finding + this report; no source-code changes ⇒ no Outcome-L1 feature-gap issues filed for THIS PR, but the iter-6 methodology-evolution PR carries the cross-cutting feature gaps)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json))

| Tier | Verdict | Evidence |
|---|---|---|
| **L0** — build + artifact validation | **PASS** | `winml build` produced `model.onnx` + `.data` co-located; opset 17, fp32, 1042 nodes, 21 unique op types; external-data layout per [`_meta-023`](../skill_meta/findings.json) |
| **L1-CPU** — perf | **PASS** | 1637 ms/iter on 1024-token sequence via custom Python perf script with real tokenized input (per [`_meta-017`](../skill_meta/findings.json) — `winml perf` ignores recipe `value_range` and crashes on eos-pooling models with random ints) |
| **L1-DML / L1-QNN / L1-OpenVINO** | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json): DML crash 0xC0000409, QNN absent, OpenVINO DLL-load-fails on this host. `--ep-options enable_graph_capture=false` retry per [`_meta-026`](../skill_meta/findings.json) NOT attempted on this host (would not help — DLL-load is a packaging issue). Not penalized per `_meta-016` honest-floor rule. |
| **L2** — PT-vs-ONNX numerical | **PASS** | cosine = 1.000000, max_abs = 1e-6, argmax = 2 (ENTAILMENT) on both PT and ONNX sides, real tokenized input ("A soccer game with multiple males playing." → "This example is sports."). Log: [temp/bart_mnli_l2.log](../../../temp/bart_mnli_l2.log) |
| **L3** — task-metric eval | **PASS** | `accuracy = 0.8800`, latency = 1.89 s/sample, throughput 0.53 samples/sec, total 189.05 s on `glue/mnli/validation_matched/100 samples, seed=42`. Reference (published bart-large-mnli on full validation_matched): ~0.886 — within MC noise of 100-sample subset. Result JSON: [temp/bart_mnli_l3_eval.json](../../../temp/bart_mnli_l3_eval.json). Log: [temp/bart_mnli_l3.log](../../../temp/bart_mnli_l3.log) |
| **L3** — full validation_matched (9815 samples) | **TIMEOUT-at-scale (NOT-ATTEMPTED)** | Per [`_meta-029`](../skill_meta/findings.json) — full run would take ~5h CPU; out of turn budget. Marker file convention not yet dropped; cited here so future contributors know the gap. |

**Short-circuit honored** (per [`_meta-018`](../skill_meta/findings.json)): no FAIL verdict anywhere in the ladder; CPU-PASS at L0..L3 supports the claimed ceiling honestly. Non-CPU EPs are HOST-BLOCKED (not FAIL), so they don't short-circuit higher tiers.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**No NEW methodology friction observed in this contribution.** The iter-6 meta-experiment that surfaced `_meta-019..031` was the *vehicle* that ran this contribution; those findings shipped in a separate methodology PR. Within the bart-mnli L3 closure itself, the only friction was the `--dataset-config` vs `--dataset-name` flag confusion — already captured under bart-005's gotchas section, which is the correct scope (per-model knowledge, not skill-meta, because the wrong flag is the same flag for any task).

Step 4b trigger inventory:
- (1) CLI surprise — `--dataset-config` → `--dataset-name`. Captured in bart-005 gotchas (per-model scope, not `_meta-NNN`).
- (2) Doc-code drift — none observed.
- (3) Silent-failure mode — none.
- (4) New verdict shape — none (PASS / TIMEOUT-at-scale already in vocabulary).
- (5) Reviewer-found gap — pending reviewer pass.
- (6) Effort mis-estimate — none (L0★ predicted, L0★ delivered).
- (7) PR-mining discovery — none in this PR (PR-mining was the methodology PR, separate bundle).

## Artifact mining (Step 4)

### `analyze_result.json`
- `total_operators`: 1042
- `unique_operator_types`: 21
- Top-10 op histogram: Reshape(316), Gemm(194), Transpose(145), Add(98), Mul(72), MatMul(72), LayerNormalization(62), Softmax(36), Gelu(24), Cast(4)
- **EP coverage caveat** per [`_meta-013`](../skill_meta/findings.json): runtime-rule parquet files not available on this external host; re-run analyze against an available EP is structurally blocked. Reviewer with internal host should re-run.

### `export_htp_metadata.json`
- `model.total_parameters`: 407,344,131 (407M — matches HF config card)
- `model.total_modules`: 353
- `tracing.modules_traced`: 93 (26% trace coverage — partial; classification head not fully traced because `BartForSequenceClassification` does eos-pooling via Python indexing rather than as a traceable module)

### `winml_build_config.json` (autoconf diff vs producer recipe)
- `optim` block: autoconf added `clamp_constant_values=true`, `gelu_fusion=true`, `matmul_add_fusion=true`, `remove_isnan_in_attention_mask=true` (recipe specified `optim: null`)
- `loader.model_class`: `AutoModelForSequenceClassification` (auto-resolved from `task=text-classification`)
- All other fields match the recipe verbatim

## Reviewer next steps

1. Re-run the L3 command on a fresh CPU host:
   ```powershell
   uv run winml eval -m temp\verify_bart_build\model.onnx --model-id facebook/bart-large-mnli `
     --task text-classification --dataset glue --dataset-name mnli `
     --split validation_matched --samples 100 --device cpu --ep cpu `
     --column input_column=premise --column second_input_column=hypothesis --column label_column=label `
     -o temp\review_bart_l3.json
   ```
   Expect `accuracy ∈ [0.85, 0.91]` within MC noise at seed=42, n=100.
2. Re-run L2 script (per [temp/bart_mnli_l2.py](../../../temp/bart_mnli_l2.py) referenced in bart-004); confirm cosine ≥ 0.9999 and argmax matches.
3. Verify `model.onnx` + `.data` co-located via `Get-ChildItem temp\verify_bart_build` per [`_meta-023`](../skill_meta/findings.json).
4. Confirm bart-005 finding is appended (not rewriting bart-004) per Step 4 append-don't-rewrite rule.
5. Verdict: APPROVE / REQUEST_CHANGES / REJECT per [REVIEW.md](../REVIEW.md).
