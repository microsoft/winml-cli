# Per-Family Model Knowledge Base

Each JSON file stores empirical findings for one Hugging Face model family
(`config.json["model_type"]`). The **planner** ([`../agents/planner.md`](../agents/planner.md))
reads the relevant file **before** starting a new model-support contribution (Step 1); the
**learner** ([`../agents/learner.md`](../agents/learner.md)) curates findings **after** (Step 4).

This directory is the self-learning loop for the [`adding-model-support`](../SKILL.md)
skill. It is the model-side analogue of the EP/device and graph knowledge maintained
by the separately installed `auto-optimize` skill and inherits the same epistemic
discipline.

## ⚠️ CRITICAL EPISTEMICS

Findings here are **observational hypotheses, not ground truth**. Each finding was
recorded after a small number of experiments on a small number of checkpoints, on
specific ORT / EP / SDK versions. Before using a finding to skip work:

1. **Is the checkpoint the same family AND similar scale?** (DINOv2-small ≠ DINOv2-giant)
2. **Is the target precision the same?** (W8A16 ≠ W8A8 ≠ FP16)
3. **Is the target EP / device the same?** (QNN NPU ≠ DML GPU ≠ CPU)
4. **Is the ORT / SDK version the same?** (kMaxSupportedOpset shifts across releases)
5. **Is the mechanism confirmed?** (`mechanism_confirmed: false` → still a hypothesis)

**Dialectical rule** — A finding that suggests skipping work must be re-enabled if a
new experiment on a new checkpoint / EP / version contradicts it. Findings degrade
over time as ORT, EP SDKs, and HF model classes change.

## Layout

```
model_knowledge/
├── README.md              # this file
├── _template.json         # blank finding skeleton — copy when starting a new family
├── <family>.json          # one per HF model_type (e.g. dinov2.json, bert.json)
```

Filename = lowercase HF `model_type` from the candidate's `config.json`. One file per
**architecture family**, not per individual checkpoint — checkpoints become entries
inside the family file.

> **Methodology findings live elsewhere.** Findings about the skill itself (path drift,
> missing recipe templates, task-family asymmetries) belong in
> [`../skill_meta/`](../skill_meta/), not here. This directory is per-model only.

## Schema

See [`_template.json`](./_template.json) for the canonical skeleton. Key invariants:

- **`_meta.models_tested`** — every checkpoint a finding has been validated against,
  including the ones that *refuted* an earlier hypothesis.
- **`findings[].scope`** — partitioned into `validated_on`, `falsified_on`,
  `not_yet_tested_on`. The `falsified_on` list is what stops a hypothesis from
  silently overgeneralizing.
- **`findings[].mechanism_confirmed`** — `false` until the cause is traced to source
  (ORT code, EP SDK behavior, calibration math). A speedup or a failure without an
  explained mechanism is still useful data, but mark it honestly.
- **`findings[].feature_gaps_filed`** — issue numbers for gaps you hit and reported.
  This is the audit trail that turns Outcome L1 into a closeable loop.

## Rules of engagement

1. **Preserve evidence; curate instructions.** A genuine counter-example goes into
   `scope.falsified_on` of the old finding and gets a new finding documenting the
   counter-example. Keep refuted findings when they contain unique evidence about a
   previous ORT/SDK era, but mark obsolete commands and recommendations as historical.
   Merge or delete pure duplicates after moving any unique evidence and repairing all
   relationships; stable IDs are not renumbered.
2. **One finding per claim.** Don't pack "needs `nodes_to_exclude` for LayerNorm" and
   "FP16 hits parity on QNN NPU" into one entry. Split them.
3. **Confidence ≠ generality.** A finding can be high-confidence on the one checkpoint
   you tested and still not generalize. Encode reach in `scope`, not in prose.
4. **Cite the artifact.** `observation` must include model id, recipe path, precision,
   EP, and ORT version (where relevant) — enough for another agent to reproduce or
   refute on demand.
