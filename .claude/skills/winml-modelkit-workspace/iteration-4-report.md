# Iteration 4 — Unified eval report

**Date:** 2026-05-13
**Skill version:** SKILL.md at iter-3 state (no body changes this round)
**Purpose:** First run of the eval framework laid out in `winml-cli-skill-design.md` — three pillars (Trigger / Response / Prescription), measured end-to-end.

---

## TL;DR

All three pillars green on the dimensions runnable today:

| Pillar | Coverage | Pass rate | Status |
|---|---|---|---|
| 1. Trigger correctness | 20/20 queries | **100%** | ✓ baseline established |
| 2. Response correctness | 6 cases × concept + static checks (37 assertions) | **97.3%** | ✓ delta +35pp vs no-skill baseline |
| 3. Prescription correctness (CPU-doable cases only) | 3 cases × outcome assertions | **100% (14/14)** | ✓ E2E framework validated; NPU cases pending hardware |

No skill changes triggered this round. The eval framework itself is now in place and reproducible.

---

## Pillar 1 — Trigger correctness

**Setup.** 20 user queries written in real-sounding voice (`evals/trigger_eval/queries.json`), 10 should-trigger / 10 should-not-trigger. Negative cases targeted near-misses that share keywords or domain but fall outside the skill's intent.

**Method.** A judge subagent was given only the skill's `description` field (no body) plus all 20 queries; it returned YES/NO for each. Compared against ground truth.

**Result.**

| Metric | Value |
|---|---|
| Accuracy | 20/20 = **100%** |
| Recall on should-trigger | 100% (10/10) |
| Specificity on should-not-trigger | 100% (10/10) |
| False positives (over-trigger) | 0 |
| False negatives (under-trigger) | 0 |

**Notable negative cases that correctly didn't trigger:**
- "how do I call windows ml from a c# app?" → SDK programming, not CLI driving
- "我想训练一个 bert classifier 在 azure ml 上" → training, not deployment
- "how do i get llama 3 8b running locally" → LLM, out of scope
- "我的 Foundry Local 部署一直起不来" → different product (adjacent name)
- "what's the best way to read onnx model metadata in python?" → general ONNX, not winml CLI

**No iteration needed** — description is well-targeted on this query set.

**Artifacts:** `trigger_eval/queries.json`, `trigger_eval/judge_responses.json`, `trigger_eval/results.json`

---

## Pillar 2 — Response correctness (carried from iter-3)

**Setup.** 6 representative cases (`iteration-3/eval-*/`). Each case has 4-7 concept assertions + a static command check against `winml <cmd> --help`. Run with-skill vs without-skill (baseline).

**Result.**

| Case | with-skill | baseline |
|---|---|---|
| eval-snapdragon-resnet-build | 7/7 | 6/7 |
| eval-ryzen-ai-quick-benchmark | 5/6 | 3/6 |
| eval-llm-out-of-scope | 5/5 | 4/5 |
| eval-npu-vs-cpu-comparison | 7/7 | 3/7 |
| eval-is-model-supported | 6/6 | 3/6 |
| eval-optimize-failure-recovery | 6/6 | 4/6 |
| **TOTAL** | **36/37 (97.3%)** | **23/37 (62.2%)** |
| Delta | **+35.1pp** | — |

The single fail (eval-ryzen with-skill, on the "inspect-as-sanity-check" assertion) is a legitimate case where the agent skipped `winml inspect` because it knew ConvNeXT was in scope from the skill itself. Not strictly wrong; reflects a tension between the skill's golden rule and its "quick benchmark" pattern. Worth refining in a future iteration but not blocking.

**Artifacts:** `iteration-3/benchmark.json`, `iteration-3/cli_verification.md`, `iteration-3/review.html`

---

## Pillar 3 — Prescription correctness (E2E, CPU subset)

**Setup.** 3 cases that the current CPU-only dev box can validate end-to-end. Each case spawns a real agent (Claude Code subagent with Bash + winml tool access), gives it the user prompt, lets it run for real. After completion, outcome assertions check artifacts produced + final message + efficiency.

**Result.**

| Case | Tool calls | Wall time | Assertions | Status |
|---|---|---|---|---|
| `cpu-benchmark-resnet` (single `perf` invocation) | 7 | 172.3s | 5/5 | ✓ |
| `llm-refusal-phi3` (out-of-scope refusal) | 1 | 16.3s | 5/5 | ✓ |
| `cpu-full-build-resnet` (config + build pipeline) | 9 | 220.1s | 4/4 | ✓ |
| **TOTAL** | — | — | **14/14** | **✓ 100%** |

**Outcome evidence:**

- **cpu-benchmark-resnet** — agent ran `winml inspect` + `winml sys --list-ep` + `winml perf -m microsoft/resnet-50 --device cpu -o ...`. Produced a real `resnet50_cpu_perf.json` with `latency_ms`, `model_info`, `throughput` keys. Final message: "20.17 ms average latency (P50 18.54 ms, P95 39.56 ms)".
- **llm-refusal-phi3** — agent read skill, recognized Phi-3 as out-of-scope, refused without invoking any winml build commands. Pointed user at OpenVINO GenAI / ORT GenAI / DirectML alternatives. Only 1 tool call (Read on SKILL.md).
- **cpu-full-build-resnet** — agent ran `inspect → config → build`, produced 97.5 MB optimized ONNX model + metadata files. Build completed in 23.5s on CPU EP.

**No iteration needed.**

**Artifacts:** `e2e_eval/cases.json`, `e2e_eval/scratch/<case>/{agent_summary.md, grading.json, *.json, build/*}`

---

## Cross-pillar observations

1. **The skill is doing real work.** Response baseline (62%) → with-skill (97%) shows +35pp of value. E2E shows the recommended workflow actually delivers artifacts when followed.
2. **Loose coupling held up.** Across all three pillars, no command in any with-skill response/run referenced a fabricated flag. Static check found 0 violations on the current iteration; E2E execution succeeded for every command the agent emitted.
3. **Refusal works at all three layers.** Phi-3 was correctly refused in chat-mode response eval (Pillar 2), in trigger eval the LLM queries didn't trigger (Pillar 1 specificity), and in agent-mode E2E the agent refused without running anything wasteful (Pillar 3).
4. **Cost profile matches design.** Pillars 1 + 2 ran in seconds and minutes respectively. Pillar 3 took 3-6 min per case (3 cases ran in parallel, ~3.7 min real time for the longest).

---

## What's still pending

Per design doc's "Known limitation":

- **NPU-specific E2E cases.** The 3 cases in Pillar 3 are all CPU-doable. The QNN / VitisAI build + compile + perf paths cannot be validated on this dev box — silent-EP-fallback would produce the wrong artifact without failing visibly. These need a Snapdragon X Elite / Ryzen AI machine.
- **Cross-runtime validation.** Eval today is driven through Claude Code's subagent API. Behavior on Copilot (the FoundryTK ship target), Cursor, etc. is not validated programmatically. Manual smoke testing recommended pre-broad-release.
- **Pass@k for E2E.** Currently each case ran once. Agent-mode is non-deterministic; for reliability claims we should run each case 3-5× and report pass rate. Skipped this round to keep MVP scope tight.

---

## What we changed this round

Nothing in `SKILL.md`. The framework is the new artifact:
- `trigger_eval/` (Pillar 1)
- `e2e_eval/` (Pillar 3)
- `verify_commands.py` (Pillar 2 static check, was already in place)
- This report consolidates results across all three.

The next iteration (5+) should be driven by: (a) telemetry once the skill is in real users' hands, or (b) NPU hardware availability to extend Pillar 3 coverage.
