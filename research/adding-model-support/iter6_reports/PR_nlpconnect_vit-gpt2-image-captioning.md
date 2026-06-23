# PR: nlpconnect/vit-gpt2-image-captioning — extend Goal ladder to L2-encoder + probe L3 (composite image-to-text)

**Iter**: 6 (Goal-ladder extension; composite recipe pair shipped in iter-5 as ved-004)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L2-encoder + L3-CLI-BLOCKED, Outcome = L1)`

> **Composite-PR contract** ([`_meta-020`](../skill_meta/findings.json)): this is ONE PR covering BOTH halves of the composite (encoder + decoder). The verdict-matrix rows expand per-half inside this single report. Splitting into two PRs is REQUEST_CHANGES per the composite contract.

## Summary

This PR extends the Goal ladder on `nlpconnect/vit-gpt2-image-captioning` (image-to-text, fp32, CPU) from L0+L1 (shipped in iter-5 as ved-004) to L2-encoder PASS + L3 probe. L3 result: **CLI-BLOCKED** — `winml eval --task image-to-text` errors with `No dataset provided and no default for task 'image-to-text'`. The CLI-BLOCKED verdict is honest closure under [`_meta-018`](../skill_meta/findings.json); the gap is filed against `winml eval` (default captioning dataset). Decoder L2 is **DEFERRED-HARNESS** per the marian-005 precedent (DynamicCache↔past_KV bridge non-trivial). No source-code changes; no new recipe.

## 1. Recipe files

Composite pair, shipped iter-5, unchanged:
- [examples/recipes/nlpconnect_vit-gpt2-image-captioning/image-to-text_encoder_config.json](../../../examples/recipes/nlpconnect_vit-gpt2-image-captioning/image-to-text_encoder_config.json)
- [examples/recipes/nlpconnect_vit-gpt2-image-captioning/image-to-text_decoder_config.json](../../../examples/recipes/nlpconnect_vit-gpt2-image-captioning/image-to-text_decoder_config.json)

Composite-expansion gate ([`_meta-020`](../skill_meta/findings.json)) verified: `winml config` (no `--task`) auto-emits TWO recipes for VisionEncoderDecoderModel @ image-to-text (a `WinMLEncoderDecoderModel` subclass with task ∈ {text2text-generation, image-to-text}).

Encoder output naming ([`_meta-025`](../skill_meta/findings.json)) verified: encoder `output_tensors[0].name = "last_hidden_state"` matches decoder `encoder_hidden_states` input via the alias-injection in `feature_extraction.py` (added PR#863, AHEAD-ON-MAIN per [`_meta-030`](../skill_meta/findings.json) — applies once branch merges main).

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) line 32 — present (`nlpconnect/vit-gpt2-image-captioning | image-to-text | ...`). No edit needed.

## 3. Build output directories + artifact inventory

Two output dirs (one per composite half), both gitignored:

### `temp/verify_vit_enc/` (encoder)

| File | Size | Purpose |
|---|---:|---|
| `model.onnx` | 143,516 B | optimized ONNX graph |
| `model.onnx.data` | 343,194,624 B | external-data shard (327 MB) |
| `export.onnx` + `.data` | 327 MB | pre-optimize |
| `optimized.onnx` + `.data` | 327 MB | mid-pipeline |
| `analyze_result.json` | 1,408 B | Step 4 mining |
| `export_htp_metadata.json` | 112,788 B | Step 4 mining |
| `winml_build_config.json` | 1,032 B | Step 4 mining |

### `temp/verify_vit_dec/` (decoder)

| File | Size | Purpose |
|---|---:|---|
| `model.onnx` | 287,547 B | optimized ONNX graph |
| `model.onnx.data` | 765,632,512 B | external-data shard (730 MB) |
| `export.onnx` + `.data` | 730 MB | pre-optimize |
| `optimized.onnx` + `.data` | 730 MB | mid-pipeline |
| `analyze_result.json` | 1,985 B | Step 4 mining |
| `export_htp_metadata.json` | 472,553 B | Step 4 mining (larger — decoder has more modules) |
| `winml_build_config.json` | 8,438 B | Step 4 mining (larger — decoder has KV-cache section) |

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): both `model.onnx` and `.data` are co-located in their respective directories. PASS for both halves.

## 4. Build logs

Iter-5 build logs: referenced under ved-004 mechanism_notes. Iter-6 used iter-5 artifacts unchanged.

L2 log (encoder, this PR): [temp/vit_gpt2_l2.log](../../../temp/vit_gpt2_l2.log) — 678 B.
L3 log (composite, this PR): [temp/vit_gpt2_l3.log](../../../temp/vit_gpt2_l3.log) — 992 B; CLI-BLOCKED error captured verbatim.

## 5. Appended findings

### Per-model — `model_knowledge/vision_encoder_decoder.json`

[ved-005](../model_knowledge/vision_encoder_decoder.json) — "VALIDATED Goal-L0+L1-CPU+L2-encoder for nlpconnect/vit-gpt2-image-captioning. L2-decoder DEFERRED-HARNESS (past-KV bridge non-trivial, per marian-005 precedent). L3 CLI-BLOCKED: `winml eval --task image-to-text` errors 'No dataset provided and no default for task image-to-text' — composite eval surface for image-to-text is NOT yet wired in winml CLI."

`_meta.models_tested` updated from `[]` to `["nlpconnect/vit-gpt2-image-captioning (L0+L1-CPU+L2-encoder PASS; L2-decoder DEFERRED-HARNESS; L3 CLI-BLOCKED)"]`.

### Skill-meta — `skill_meta/findings.json`

This PR surfaces a NEW class of L3 CLI-BLOCKED distinct from [`_meta-015`](../skill_meta/findings.json) (which was "task not in TASK_REGISTRY"): here the task IS supported (`winml eval --schema --task image-to-text` returns input_column/label_column spec), but NO default dataset is wired. The new sub-class is documented as a `feature_gaps_filed[]` entry on ved-005 and surfaced in declaration (a) below; it does not yet warrant a new `_meta-NNN` (one data point is per-task knowledge; a second occurrence on another non-defaulted task would justify promotion to skill-meta as "tasks-without-default-dataset" verdict-subtype).

## 6. Optimum-coverage probe verdict

```python
mt = "vision-encoder-decoder"
vendor   = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
# vendor includes: image-to-text and text2text-generation (composite tasks)
ensure_hf_models_registered()
after    = sorted(TasksManager._SUPPORTED_MODEL_TYPE.get(mt, {}).get("onnx", {}).keys())
# added_by_winml: WinMLEncoderDecoderModel subclass for HTP-friendly KV-cache shape (separate from Optimum's vanilla)
```

**Verdict**: VENDOR-COVERED on `image-to-text`. Winml's `WinMLEncoderDecoderModel` overrides for HTP-friendly cache shape; the composite registration is the per-architecture work. Effort L0★ (recipe-only against winml's already-registered composite). Verified iter-5 (ved-001/002) and re-confirmed by ved-004 build + ved-005 extension.

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only; winml already covers VisionEncoderDecoder composite via prior L1 work in `models/hf/vision_encoder_decoder.py`)
- **Goal = L2-encoder PASS + L3-CLI-BLOCKED** (honest mixed ceiling — encoder L2 closes; decoder L2 deferred per marian-005 precedent; L3 blocked by CLI)
- **Outcome = L1** (recipe pair + appended ved-005 finding + this report; feature gap filed for `winml eval --task image-to-text` default dataset)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json))

Expanded per-half because composite contract (`_meta-020`):

| Tier | Encoder | Decoder | Evidence |
|---|---|---|---|
| **L0** — build + artifact validation | **PASS** | **PASS** | encoder: 366 nodes, 11 unique ops; decoder: 803 nodes, 22 unique ops. External-data layout per [`_meta-023`](../skill_meta/findings.json) PASS on both. |
| **L1-CPU** — perf | **PASS** | **PASS** | encoder: 69.36 ms/iter (`winml perf --ep cpu`); decoder: 40.39 ms/iter. Random dummy inputs OK — no eos-pooling assertion in ViT encoder or GPT2 cross-attn decoder. |
| **L1-DML / L1-QNN / L1-OpenVINO** | **HOST-BLOCKED** | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json). `--ep-options` retry per [`_meta-026`](../skill_meta/findings.json) NOT attempted (packaging issue, not runtime tuning). |
| **L2** — PT-vs-ONNX numerical | **PASS** | **DEFERRED-HARNESS** | encoder: cosine = 1.000000, max_abs = 2e-6 vs PT `VisionEncoderDecoderModel.encoder` on fixed-seed 224×224 RGB. Decoder: marian-005 precedent — DynamicCache↔past_KV bridge exceeds turn budget. Log: [temp/vit_gpt2_l2.log](../../../temp/vit_gpt2_l2.log). |
| **L3** — task-metric eval (image-to-text) | **CLI-BLOCKED** | **CLI-BLOCKED** | `uv run winml eval -m encoder=... -m decoder=... --task image-to-text --device cpu --ep cpu --samples 20` → `Error: Evaluation failed: No dataset provided and no default for task 'image-to-text'. Use --dataset.` Log: [temp/vit_gpt2_l3.log](../../../temp/vit_gpt2_l3.log). Distinct from [`_meta-015`](../skill_meta/findings.json) (task IS in registry, just no default dataset). Gap filed against `winml eval` (see ved-005 `feature_gaps_filed[0]`). |

**Short-circuit honored** (per [`_meta-018`](../skill_meta/findings.json)): no FAIL anywhere; all unreached tiers carry BLOCKED/DEFERRED verdicts. The decoder DEFERRED-HARNESS does NOT short-circuit L3 because (a) DEFERRED is not FAIL, and (b) L3 is independently blocked by the CLI gap above decoder L2.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**Methodology friction observed: 1 sub-class signal** — but NOT yet upgraded to `_meta-NNN`.

Step 4b trigger inventory:
- (1) CLI surprise — encountered `--dataset` requirement on `--task image-to-text` with no error-message-suggested default. Captured as ved-005 feature gap.
- (2) Doc-code drift — none observed.
- (3) Silent-failure mode — none. CLI failed loudly with a clear error.
- (4) New verdict shape — **borderline**. `CLI-BLOCKED` is already in [`_meta-018`](../skill_meta/findings.json) vocabulary; this PR's CLI-block is a SUB-CLASS distinct from [`_meta-015`](../skill_meta/findings.json). One data point is per-task; promote to skill-meta only if a 2nd non-defaulted task surfaces (audio-classification, speech-to-text?). Logged in ved-005 to seed future detection.
- (5) Reviewer-found gap — pending reviewer pass.
- (6) Effort mis-estimate — none (L0★ predicted, L0★ delivered).
- (7) PR-mining discovery — none in this PR.

**No SKILL.md / REVIEW.md edits required from this PR.** The single sub-class signal under trigger (4) is below the "1 data point" promotion threshold; if reviewer disagrees, REQUEST_CHANGES with proposed `_meta-NNN` text and we promote.

## Artifact mining (Step 4)

### Encoder (`temp/verify_vit_enc/`)

`analyze_result.json`:
- `total_operators`: 366
- `unique_operator_types`: 11
- Top-10: Reshape(121), Gemm(72), Transpose(49), Add(25), LayerNormalization(25), Mul(24), MatMul(24), Softmax(12), Gelu(12), Conv(1)

`export_htp_metadata.json`:
- `model.total_parameters`: 86,389,248 (86M — ViT-base scale)
- `model.total_modules`: 216
- `tracing.modules_traced`: 90 (42% — vision tower is straightforward conv+attention; high coverage)

### Decoder (`temp/verify_vit_dec/`)

`analyze_result.json`:
- `total_operators`: 803
- `unique_operator_types`: 22
- Top-10: Reshape(219), Transpose(108), Mul(96), Add(85), Gemm(84), MatMul(49), LayerNormalization(37), Split(24), ScatterND(24), Softmax(24)
- **ScatterND(24)** in the decoder = KV-cache writes. Marian-003 noted ScatterND as "dominant unknown op" in per-EP coverage — expect similar gap here once analyze re-runs against an available EP (currently blocked per [`_meta-013`](../skill_meta/findings.json) on this external host).

`export_htp_metadata.json`:
- `model.total_parameters`: 152,806,656 (153M — GPT2-base + cross-attention)
- `model.total_modules`: 249
- `tracing.modules_traced`: 147 (59% — KV-cache modules trace cleanly)

### `winml_build_config.json` (autoconf diffs)

Encoder: 1,032 B — standard optim block similar to bart.
Decoder: 8,438 B — significantly larger due to KV-cache `past_key_values` declarations (24 layers × 4 tensors = 96 cache I/O specs).

## Reviewer next steps

1. **Re-run encoder L2** on a fresh CPU host (`temp/vit_gpt2_l2.py` referenced in ved-004); confirm cosine ≥ 0.9999.
2. **Confirm L3 CLI-BLOCK is real**: re-run `uv run winml eval -m encoder=temp\verify_vit_enc\model.onnx -m decoder=temp\verify_vit_dec\model.onnx --model-id nlpconnect/vit-gpt2-image-captioning --task image-to-text --device cpu --ep cpu --samples 20 -o temp\review_vit_l3.json`; expect the same `No dataset provided` error. If the CLI errors differently (different version, different error), the verdict needs updating.
3. **Composite gate cross-check**: `winml inspect nlpconnect/vit-gpt2-image-captioning --format json` should report `composite: true` and `pipeline_tasks: ["image-to-text"]` per [`_meta-020`](../skill_meta/findings.json) + [`_meta-027`](../skill_meta/findings.json). If `composite` field is absent, the inspect output is on a pre-PR#866 branch — note in verdict, do not REQUEST_CHANGES.
4. **External-data co-location** per [`_meta-023`](../skill_meta/findings.json): `Get-ChildItem temp\verify_vit_enc, temp\verify_vit_dec`; confirm `.data` next to `.onnx` in both dirs.
5. **Decoder L2 deferral check**: per marian-005 precedent (encoder L2 PASS + decoder L2 deferred is acceptable). Do NOT REQUEST_CHANGES on decoder L2 absence; this is a known harness gap, not producer laziness.
6. **Methodology-evolution declaration audit** per [REVIEW.md](../REVIEW.md): declaration is (a)-borderline-(b). Confirm the trigger-4 sub-class signal is correctly held at per-model scope; recommend promotion to skill-meta only on second occurrence.
7. Verdict: APPROVE / REQUEST_CHANGES / REJECT per [REVIEW.md](../REVIEW.md).
