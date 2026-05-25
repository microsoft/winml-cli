# Response eval — 20260524-133637

SKILL.md changes vs 20260524-120701: generalized from prescriptive to principle.

- **Added** new section "Outputs are explicit; cache is opaque" right after "The mental model". Teaches the underlying rule: each `winml` command has a published output (what `-o` writes / what it prints), and cache / internal byproducts are not supported outputs. Pick the command whose published output matches your goal.
- **Slimmed** "Disambiguating 'I want to run X'" → "Mapping 'I want to run X' to a command": kept the three-row intent→command table (now framed in terms of each command's published output), dropped the prescriptive "Order matters: build before perf, never perf before build" paragraph and the trailing "ask one short question" prose.
- **Removed** the perf-doesn't-expose-artifact caveat from the "Just benchmark" pattern.

Both the slim and the removal are derivable from the new principle section. The motivation: avoid accumulating one prescription per observed failure ("no perf→build→perf", "don't fish artifacts from cache", "don't feed inspect output to build", …). One teachable rule covers the family.

## Overall

| Metric | with_skill | baseline | Δ |
|---|---|---|---|
| Pass rate | **100.0%** (71/71) | 49.3% (35/71) | +50.7pp |

Same scores as 20260524-120701 (also 71/71 and 35/71). The change is in the *shape* of the skill body, not its coverage on this case set.

## Per case

| Case | with_skill | baseline |
|---|---|---|
| `eval-snapdragon-resnet-build` | 8/8 | 6/8 |
| `eval-ryzen-ai-quick-benchmark` | 7/7 | 3/7 |
| `eval-npu-vs-cpu-comparison` | 8/8 | 3/8 |
| `eval-is-model-supported` | 6/6 | 2/6 |
| `eval-optimize-failure-recovery` | 6/6 | 4/6 |
| `eval-install-setup` | 5/5 | 1/5 |
| `eval-local-onnx-file` | 5/5 | 4/5 |
| `eval-config-build-for-ci` | 6/6 | 3/6 |
| `eval-seq2seq-out-of-scope` | 5/5 | 3/5 |
| `eval-ambiguous-run-intent` | 8/8 | 5/8 |
| `eval-perf-output-recovery` | 7/7 | 1/7 |

## Did the principle still cover the two target cases?

Yes — and the responses got better, not worse:

- **`ambiguous-run-intent`** still passes all 8 assertions. The Quick clarifier section in the response uses the same three-row intent→command framing the slimmed skill table now uses. The agent picks a sensible default (artifact + benchmark) and tells the user how to skip to just-perf — the prescriptive "build before perf, never perf before build" sentence was not needed to produce the right answer.
- **`perf-output-recovery`** still passes all 7 assertions, and the response now *quotes the principle directly* — "every `winml` command has one published output, and that's the only thing you can rely on downstream. For `perf` that's the metrics. For a deployable model, you want the command whose `-o` *is* an `.onnx` artifact — that's `build`." The principle replaced a scattered caveat with a reusable rule.

## Did anything regress?

No. All 9 existing cases held at the same scores. The slimmed body did not lose any specific guidance these cases depend on; the principle plus the surviving sections (golden rule, choosing a path, hardware EPs, common patterns, scope, catch-outs) covered them just as well.

## Why this version is preferable to 20260524-120701

Same pass rate, fewer prescriptions. The previous iter had three sites of the same idea (the disambiguation prose, the order rule, the just-benchmark caveat); this iter has one site (the principle) plus a table that points back to it. Future failure modes in the same family — "use analyze output as a graph rewrite", "feed inspect JSON to build", "scavenge the EP context blob from cache" — should now be caught by the principle without each requiring its own paragraph.

## Ship?

Yes. Pass rate at 100%, delta vs baseline 50.7pp, no regressions. Promote `20260524-133637` as the new baseline.
