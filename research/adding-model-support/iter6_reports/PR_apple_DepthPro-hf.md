# PR: apple/DepthPro-hf — depth-estimation recipe (fp32, CPU)

**Iter**: 6 (build closure shipped iter-3 as depth_pro-002/003; this PR adds the L1-CPU evidence on top)
**Producer**: main agent (2026-06-23)
**Claimed tier**: `(Effort = L0★, Goal = L1-CPU, Outcome = L0)`

## Summary

This PR ships the `apple/DepthPro-hf` depth-estimation recipe. DepthPro is a 952M-param model with 3 independent DINOv2 backbones (patch + image + fov encoders) plus neck/fov/fusion stages — the recipe is structurally one of the largest single-graph models in the catalog. Builds cleanly via the standard L0★ template; L1-CPU PASSes via a custom Python harness that reuses the cached artifact (the `winml perf` path triggers a re-export per invocation, which is wasteful for a 3.6 GB model — see Diligence ladder note in §8). No source-code changes.

## 1. Recipe file

[examples/recipes/apple_DepthPro-hf/depth-estimation_fp16_config.json](../../../examples/recipes/apple_DepthPro-hf/depth-estimation_fp16_config.json)

Filename `fp16_*` is cosmetic per [`_meta-014`](../skill_meta/findings.json); recipe ships fp32 (102 FLOAT32 initializers, 0 FLOAT16 verified by `onnx.load`).

Recipe input shape: `pixel_values [1, 3, 1536, 1536] float32 range [0, 1]`. Outputs: `predicted_depth [1, 1536, 1536]` + `field_of_view [1]`.

## 2. README index row

[examples/recipes/README.md](../../../examples/recipes/README.md) — row to add for `apple/DepthPro-hf | depth-estimation | single (no composite) | recipe`.

## 3. Build output directory + artifact inventory

`temp/depth_pro_build/` (gitignored — referenced by path for reviewer re-execution):

| File | Size | Purpose |
|---|---:|---|
| `model.onnx` | small | optimized graph pointer (external-data layout) |
| `model.onnx.data` | ~3.6 GB | external-data shard (FLOAT32 weights) |
| `export.onnx` + `.data` | ~3.6 GB | pre-optimize artifact |
| `optimized.onnx` + `.data` | ~3.6 GB | mid-pipeline artifact |
| `analyze_result.json` | mined | op histogram (Step 4) |
| `export_htp_metadata.json` | mined | module hierarchy + trace coverage (Step 4) |
| `winml_build_config.json` | mined | autoconf diff (Step 4) |

**External-data layout check** ([`_meta-023`](../skill_meta/findings.json)): `model.onnx` and `model.onnx.data` co-located in same directory. PASS.

## 4. Build log

[temp/depth_pro_build.log](../../../temp/depth_pro_build.log) — `Build complete in 758.0s` (export 375s + optimize 355s). Build artifact path: `temp/depth_pro_build/`.

## 5. Appended findings

### Per-model — `model_knowledge/depth_pro.json`

- [depth_pro-001](../model_knowledge/depth_pro.json) — pre-build coverage probe (WINML-ONLY).
- [depth_pro-002](../model_knowledge/depth_pro.json) — VALIDATED build closure (iter-3).
- [depth_pro-003](../model_knowledge/depth_pro.json) — build-artifact mining (3-backbone architecture, 49% layout-move ops, 952M params).

### Skill-meta

No new `_meta-NNN` findings in this PR (Lane B).

## 6. Optimum-coverage probe verdict

```python
mt = "depth_pro"
# vendor: {}  (NOT registered upstream)
# after_winml: {"depth-estimation": <winml override>}
# added_by_winml: ["depth-estimation"]
```

**Verdict**: WINML-ONLY. `depth_pro` model_type is not registered in Optimum's `TasksManager._SUPPORTED_MODEL_TYPE`; winml's `register_onnx_overwrite` decorator at `src/winml/modelkit/models/hf/depth_pro.py` is what makes export work. Despite the WINML-ONLY classification, no code is needed in THIS PR (the per-arch file already exists) — the recipe is a pure consumer of the existing registration. Effort L0★ confirmed.

## 7. Claimed (Effort, Goal, Outcome) tier

- **Effort = L0★** (recipe-only; the per-arch `depth_pro.py` already exists from prior iter; this PR adds only the recipe + finding append)
- **Goal = L1-CPU** (L0 PASS + L1-CPU PASS via custom Python harness; L2/L3 deferred — depth metrics CLI-blocked per `_meta-015` analogue for depth-estimation)
- **Outcome = L0** (recipe + finding append + this report)

## 8. Goal-ladder verdict table (per [`_meta-018`](../skill_meta/findings.json))

| Tier | Verdict | Evidence |
|---|---|---|
| **L0** | **PASS** | `winml build` → `model.onnx` + `model.onnx.data` co-located; opset 17, fp32, 2822 nodes, 19 unique op types; Build complete in 758.0s. Log: [temp/depth_pro_build.log](../../../temp/depth_pro_build.log) |
| **L1-CPU** | **PASS** | Avg 28513.7 ms / Min 27830.9 / Max 29110.3 / Std 525.9 on real-shape `pixel_values [1,3,1536,1536]` input; warmup 29582 ms (cold); throughput 0.035 samples/sec on CPU. Custom Python harness per [`_meta-017`](../skill_meta/findings.json) (avoids re-export). Log: [temp/depth_pro_perf_cpu.log](../../../temp/depth_pro_perf_cpu.log); script: [temp/depth_pro_perf.py](../../../temp/depth_pro_perf.py) |
| **L1-DML/QNN/OpenVINO** | **HOST-BLOCKED** | Per [`_meta-016`](../skill_meta/findings.json). 49% layout-move ops (Reshape/Transpose/Slice = 1378/2822 per `depth_pro-003`) means QNN-NPU would likely be heavily move-bound even when available. |
| **L2** | **DEFERRED-HARNESS** | Honest verdict per [`_meta-018`](../skill_meta/findings.json). PT-vs-ONNX comparison would need DepthPro pipeline reconstruction (preprocessor → 3-backbone forward → neck → fusion → head); script not written this turn. |
| **L3** | **CLI-BLOCKED** | `winml eval` task registry does not include `depth-estimation` (analogous to translation per [`_meta-015`](../skill_meta/findings.json)). |

**Short-circuit honored**: no FAIL anywhere. L2/L3 deferred-or-blocked do not halt the march per [`_meta-018`](../skill_meta/findings.json). Honest ceiling is L1-CPU PASS.

**Diligence ladder ([`_meta-037`](../skill_meta/findings.json))** — invoked during L1 attempt:

1. Re-read `depth_pro.json` — no prior perf workaround documented for this model.
2. PR-mine — no prior DepthPro perf PRs.
3. Re-run `winml config` — N/A, recipe already exists.
4. `--ep-options` retry — N/A, CPU not failing.
5. `value_range` / shape pinning — recipe shape already pinned to `[1,3,1536,1536]`.
6. **Custom Python harness** — ✅ this is the step that worked. `winml perf` triggered full re-export (~13 min per invocation since each `uv run winml perf` rebuilds the artifact); switched to direct `onnxruntime.InferenceSession` against cached `temp/depth_pro_build/model.onnx`. Loaded in 15.44s, ran 3 iters in 86s total.

**Feature gap from step 6 trigger**: `winml perf` should accept a pre-built artifact path (e.g. `--artifact temp/depth_pro_build/model.onnx`) and skip the build phase entirely. For a 3.6 GB model, the build-per-perf-invocation cost is prohibitive. Captured under `depth_pro-003` `feature_gaps_filed[]` as a follow-up.

## 9. Methodology-evolution declaration (per [`_meta-031`](../skill_meta/findings.json))

**No NEW methodology friction in this PR.** The custom-harness pattern is `_meta-017`; the `winml perf` re-export cost is a new observation but rolls into the existing `_meta-017` gotcha rather than a fresh `_meta-NNN`. Triggers:

- (1) CLI surprise — none (one personal flag-name mistake `--warmup-iterations` vs `--warmup`, recovered via `winml perf --help`; not a doc-cited flag).
- (2) Doc-code drift — none.
- (3) Silent-failure mode — none.
- (4) New verdict shape — none.
- (5) Reviewer-found gap — pending.
- (6) Effort mis-estimate — none (L0★ predicted, L0★ delivered).
- (7) PR-mining discovery — none.

Reviewer should confirm "no methodology friction observed" per `_meta-031` anti-trigger.

## Reviewer hand-off package — Step 6 9-item self-check

1. Recipe file — §1 ✓
2. README row — §2 ✓ (to add in this PR)
3. Build output dir + artifact inventory — §3 ✓
4. Build log — §4 ✓
5. Appended findings — §5 ✓
6. Optimum-coverage probe verdict — §6 ✓
7. Claimed tier — §7 ✓
8. Goal-ladder verdict table — §8 ✓
9. Methodology-evolution declaration — §9 ✓
