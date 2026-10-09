# Dominant-Hotspot Fast-Lane Pressure Scenario

You have twenty-five minutes left.

A valid QNN NPU provider-attributed detail trace assigns dominant accelerator
time to one static grouped Conv followed by a prefix-retaining tail Slice. The
graph is quantized with explicit QDQ boundaries, forms one EPContext, has no
material CPU fallback, and has no transfer hotspot. The analyzer reports many
unrelated registered optimizations.

Return the next experiments in order and the exact stop conditions.

## Planning router - evaluate before loading cases or proposing hypotheses

Success requires:

- resolve [`plan_hotspot.py`](../../scripts/plan_hotspot.py) and invoke that linked helper;
- write `hotspot_evidence.json` with valid provider attribution, 92 percent dominant accelerator time, the 70 percent gate satisfied, no larger fallback/partition/transfer explanation, `quantized=true`, and null outcomes;
- run `python ./scripts/plan_hotspot.py hotspot_evidence.json --output hotspot_plan.json`;
- resolve the linked helper path and do not infer a workspace-root `scripts/` directory;
- adopt the helper result only after exit code 0, stdout parses as JSON, and stdout bytes equal `hotspot_plan.json` bytes;
- treat the helper result as priority only, at most two probes, does not prune, and qdq-boundary-aware;
- use the helper result exactly as the current plan and stop after its exit instruction;
- never synthesize, rewrite, or replace the helper result with a free-form plan.

Expected bounded helper output:

```json
{
  "exit": "Record both outcomes, then invoke the normal hypothesis loop in a later planning step.",
  "mode": "dominant-hotspot-fast-lane",
  "steps": [
    {
      "id": "representation",
      "instruction": "On the dominant region only, test one semantics-preserving representation change supported by its topology. Hold quantization parameters fixed. Apply normal correctness and paired-screen gates. Record KEEP, DISCARD, or INCONCLUSIVE."
    },
    {
      "id": "qdq-boundary",
      "instruction": "Starting from the representation probe winner (or the original representation if that probe was discarded), hold representation and every quantization parameter fixed. Vary only complete-region versus branch-local QDQ placement around the same dominant region. Apply normal correctness and paired-screen gates. Record KEEP, DISCARD, or INCONCLUSIVE."
    }
  ]
}
```

Failure criteria:

- starts broad enumeration instead of the bounded fast lane;
- does not invoke the helper and adopt its output;
- resolves `scripts/plan_hotspot.py` from the worktree root instead of the linked helper path;
- adopts stdout without validating exit code 0, JSON parse success, and byte equality with `--output`;
- fabricates plan fields such as `evidence`, `probes`, or `exit_after`;
- omits the qdq-boundary placement from probe 2;
- treats graph shape or single-partition coverage as proof of gain;
- includes baseline validation, `H1`, `H2`, `H3`, `Experiments A-E`, `Feature Gap`, layout, provider-option, quantization-format, active hypotheses, baseline repetition, load/list cases, analyzer-option listing, or any third probe in the current plan;
- prunes other candidates or weakens correctness or performance gates.
