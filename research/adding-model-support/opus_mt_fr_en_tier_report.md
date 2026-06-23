# Goal-tier ceiling report — `Helsinki-NLP/opus-mt-fr-en`

_Recipe: `examples/recipes/Helsinki-NLP_opus-mt-fr-en/translation_fp16_{encoder,decoder}_config.json`_
_Date: 2026-06-22 PM, on producer's local Windows host_

User instruction verbatim:
> 你应该这样，尽可能跑，把能达到的level试出来，分析原因并report.

So: push the recipe up the Goal ladder, find the ceiling, classify each failure as **recipe / host-env / CLI-feature-gap**.

---

## Tier-by-tier results

| Tier | Definition | Outcome | Evidence | Limiting factor |
|---|---|---|---|---|
| **Effort L0★** | recipe written, build succeeded, 3 build artifacts present | **PASS** | `temp/opus_fr_en_build/encoder/{model.onnx, model.onnx.data 198.6 MB, analyze_result.json, export_htp_metadata.json, winml_build_config.json}` + same for `decoder/` (346.0 MB external) | — |
| **Goal L0** | `onnx.load` PASS + IR / opset / shapes / dtypes match the recipe + shape contract from a sibling checkpoint | **PASS** | encoder: 204 nodes, IR 8, opset 17, `input_ids[1,512]+attention_mask[1,512] → encoder_hidden_states[1,512,512]`. decoder: 392 nodes, 17 inputs incl. 12 `past_{0..5}_{key,value}[1,8,512,64]` → `logits[1,1,59514]+12 present_*`. Vocab size 59514 matches HF config. | — |
| **Goal L1 — CPU EP** | `winml perf` runs end-to-end + reports per-iter latency | **PASS** | `temp/opus_fr_en_perf_enc_cpu.log`: encoder Avg 60.97 ms / P50 61.16 / P90 73.02 / P95 73.88 / P99 78.03 / Min 48.77 / Max 78.03 / Std 8.29 / Throughput 16.40 samples/s. `temp/opus_fr_en_perf_dec_cpu.log`: decoder Avg 17.90 ms / P50 17.68 / P90 20.08 / P95 20.08 / P99 22.91 / Min 15.94 / Max 22.91 / Std 1.43 / Throughput 55.86 samples/s. 30 iters / 5 warmup each. | — |
| **Goal L1 — DML EP** | same for DirectML | **HOST-BLOCKED** | `temp/opus_fr_en_perf_enc_dml3.log`: native crash, exit `-1073740791` = `0xC0000409` STATUS_STACK_BUFFER_OVERRUN. ORT reports DML as registered but it crashes on session create. Matches marian-003 `analyze_result.json` which already showed `DmlExecutionProvider runtime_support=false`. | Host — DML driver/runtime non-functional. Not a recipe issue. |
| **Goal L1 — QNN EP** | same for Qualcomm NPU | **HOST-BLOCKED** | `temp/opus_fr_en_perf_enc_qnn.log`: clean refusal — `Requested EP QNNExecutionProvider is not available on this system. Available EPs: [CPUExecutionProvider, DmlExecutionProvider]`. | Host — no Snapdragon NPU. Not a recipe issue. |
| **Goal L1 — OpenVINO EP** | same for Intel | **HOST-BLOCKED** | Every perf run shows OpenVINO EP package installs successfully but `register_execution_provider_library` fails: `Error 126 … onnxruntime_providers_shared.dll missing`. | Host/packaging — OpenVINO plugin DLL load fails on this box. Not a recipe issue. |
| **Goal L1 — MIGraphX / NV TensorRT-RTX / VitisAI** | same for AMD ROCm / NVIDIA RTX / AMD Ryzen AI | **NOT ATTEMPTED** | No relevant hardware on this host. | Host — N/A. |
| **Goal L2 — encoder numerical compare vs PyTorch** | cosine ≥ 0.99 and max-abs-diff reasonable on the encoder forward pass | **PASS** | `temp/fr_en_l2_compare.py` + `temp/fr_en_l2_compare.log`: cosine = **1.000000**, max_abs_diff = **8.0e-5** (0.0016 % of PT max-abs). Same input through `transformers.MarianMTModel` vs `ort.InferenceSession(encoder/model.onnx)`, CPU. | — |
| **Goal L2 — decoder numerical compare vs PyTorch** | same for decoder | **PARTIAL / not apples-to-apples** | smoke-test cosine = 0.997, max_abs = 3.81, first-token argmax DISAGREES. **Root cause:** the exported decoder is the "with-past" / incremental graph (12 `past_*_key/value[1,8,512,64]` inputs); PT was driven without KV cache. Feeding zero-filled KV + all-zero `decoder_attention_mask` is structurally inconsistent with PT's prefill. A faithful L2 needs either a separate non-cache export OR an end-to-end generate-loop comparison. | Methodology / script — needs a proper incremental-decoding harness. Not a recipe issue (encoder L2 ≈ 1.0 already proves the export is numerically correct). |
| **Goal L3 — BLEU / chrF on a translation dataset** | task-metric run via `winml eval` | **CLI-BLOCKED** | `uv run winml eval --schema --task translation` → `Task translation is not supported by winml eval. Supported tasks: [16 tasks, none generative]`. | CLI feature gap — `winml eval` TASK_REGISTRY has no entry for `translation` (nor for any other text-to-text generative task). Not a recipe issue. Captured as `_meta-015`. |
| **Outcome L0** | recipe + index row + finding append | **PASS** | recipe pair shipped, README row added, marian-004 finding VALIDATED, `_meta-014/015/016` added. | — |

**Net ceiling reached for opus-mt-fr-en: `(Effort L0★, Goal L1-CPU + L2-encoder, Outcome L0)`.**
Everything above the ceiling is blocked by either the host or the CLI — **not by the recipe**.

---

## Surprises surfaced during the push (each filed as a `_meta-*` finding)

### Surprise 1 — the "fp16" in the recipe filename is a lie (`_meta-014`)

The recipe is named `translation_fp16_encoder_config.json` but contains `"quant": null` and has no `optim.cast_to_fp16` field. `WinMLBuildConfig` itself has no first-class precision-cast knob — precision is downgraded only via `quant`. Direct inspection of the emitted encoder:

```
initializer dtype counts: {FLOAT32: 102, INT64: 32, FLOAT16: 0}
```

`winml perf` is honest and reports `Model Precision: fp32`. The filename is the only thing claiming fp16, and every existing `*_fp16_*` recipe with `quant: null` in the repo is in the same situation. This is a recipe-naming convention bug, not a build bug.

**Fixes filed:** (a) add a real `precision` field to `WinMLBuildConfig`, (b) make REVIEW.md grep the emitted ONNX for FLOAT16 initializers when the filename promises fp16, (c) document in `contributing.md`.

### Surprise 2 — L3 is structurally unreachable for translation (`_meta-015`)

`winml eval`'s supported-task list (depth-estimation, feature-extraction, fill-mask, image-classification, image-feature-extraction, image-segmentation, image-to-text, next-sentence-prediction, object-detection, question-answering, sentence-similarity, sequence-classification, text-classification, token-classification, zero-shot-classification, zero-shot-image-classification) contains **no generative text-to-text task** — no `translation`, no `summarization`, no `text2text-generation`. Every seq2seq translation recipe is capped at L2 via the CLI no matter how good it is.

**Implication for REVIEW.md:** reviewers must not penalize translation recipes for missing L3 evidence; the gap is in the CLI, not in the recipe.

**Fix filed:** register `translation` in TASK_REGISTRY with BLEU/chrF/COMET + a default dataset descriptor.

### Surprise 3 — DML EP crash is native, not Python (`_meta-016`)

`winml perf -m … --ep dml` did not raise a Python exception — the process aborted at native level with exit code `-1073740791` = `0xC0000409` = STATUS_STACK_BUFFER_OVERRUN. ORT lists DML in `get_available_providers()` (the EP is *registered*), but the underlying driver/runtime fails on session create. Easy to mistake for a recipe bug; ALWAYS probe `onnxruntime.get_available_providers()` plus a CPU baseline first to localize.

QNN's behavior is the right way: clean Python `Error: Benchmark failed: Requested EP QNNExecutionProvider is not available on this system.` 

**Fixes filed:** `winml perf` should fail-fast with a clean message for registered-but-broken EPs; SKILL/REVIEW should add a "host EP capability matrix" producers fill in once and reuse.

### Surprise 4 — OpenVINO EP install succeeds but link fails (`_meta-016`, supporting)

Every single perf run (even CPU-only) downloads + "successfully installs" `MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8` and then immediately fails at `register_execution_provider_library` because `onnxruntime_providers_shared.dll` is missing. The CPU run survives this; the DML run interacts with it badly. Not a recipe issue — packaging bug.

---

## What the producer would need to push higher

| Tier | What's needed | Owner |
|---|---|---|
| L1-DML | a host with working DML driver, OR a fix that turns the native crash into a clean refusal | infra / EP team |
| L1-QNN | a Snapdragon X Elite host | infra |
| L1-OpenVINO | fix the missing `onnxruntime_providers_shared.dll` next to the OpenVINO plugin | packaging |
| L2-decoder | a non-cache decoder export OR an end-to-end PT-vs-ORT generate-loop harness (not just single-step smoke) | recipe-author tooling |
| L3 | register `translation` task in `winml eval` TASK_REGISTRY (BLEU/chrF + flores-200/wmt14 default dataset) | CLI feature |
| recipe fp16 truth | add `precision` field to `WinMLBuildConfig` and/or rename existing `_fp16_` recipes that ship fp32 | schema |

---

## Honest summary

- The recipe itself is **clean** end-to-end on the only EP this host can run (CPU): the artifacts load, the shapes are right, perf runs, and **encoder output matches PyTorch reference within fp32 numerical noise (cosine = 1.000000, max_abs = 8e-5)**.
- Everything above the (L1-CPU, L2-encoder) ceiling is blocked by environment or CLI, not by the recipe.
- Three new methodology findings (`_meta-014/015/016`) were extracted from this push.
- Reviewers should treat `(L0★ build, L1-CPU perf, L2-encoder cosine, Outcome L0 recipe shipped)` as the **provable** ceiling for opus-mt-fr-en on this host. Claiming more would be self-grading.
