# Docs Expansion v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author 11 new doc pages, rename 2 existing pages with content edits, modify 5 pages, and restructure the MkDocs nav — delivering a Tutorials chapter, a sub-grouped Concepts chapter (Fundamentals + WinML CLI), and polish to Getting Started.

**Architecture:** Six-batch plan executed on `docs/v2`. Foundation first (scaffold + nav + renames), then authoring batches in parallel where pages don't share state, then a cross-link sweep that catches any reference to the renamed files. Verification at every batch is `uv run mkdocs build --strict`.

**Tech Stack:** Python 3.10 + uv, MkDocs Material 9.5+, pymdown-extensions, Bash via Git for Windows for `sed`/`grep` operations.

**Spec:** `docs/superpowers/specs/2026-05-24-docs-expansion-v2-design.md`

**Branch:** `docs/v2` (off `docs/v1`). No remote pushes during execution.

---

## Conventions used in this plan

- **CLI source of truth:** `src/winml/modelkit/commands/<name>.py` and `src/winml/modelkit/commands/_options.py`. Every flag mentioned in a doc must exist in source.
- **Product name in prose:** `winml-cli` (never `wmk` or `ModelKit`).
- **Existing internal docs that must NOT be modified:** `docs/design/`, `docs/naming-convention.md`, `docs/pytest-best-practices.md`, `docs/superpowers/` (other than this plan and its spec).
- **Verification at task end:** `uv run mkdocs build --strict` — must exit 0 with no WARNING lines from MkDocs (the Material upstream advisory banner is not a MkDocs WARNING).
- **Commit style:** Conventional Commits (`docs: ...`). No `Co-Authored-By`. No "Test plan" section.
- **Parallel agent dispatches** within a task = single message with multiple Agent tool calls, agents do NOT commit (orchestrator batch-commits).

---

## Task 1: Scaffold — stubs, renames, nav restructure (Batch A)

**Files (modify):**
- `mkdocs.yml`

**Files (rename):**
- `docs/concepts/onnx-and-eps.md` → `docs/concepts/eps-and-devices.md`
- `docs/concepts/hierarchy.md` → `docs/concepts/hierarchy-and-metadata.md`

**Files (create as stubs — full content authored in later batches):**
- `docs/tutorials/index.md`
- `docs/tutorials/npu-convnext.md`
- `docs/concepts/graphs-and-ir.md`
- `docs/concepts/tensors-and-dtypes.md`
- `docs/concepts/primitives-and-pipeline.md`
- `docs/concepts/config-and-build.md`
- `docs/concepts/load-and-export.md`
- `docs/concepts/analyze-and-optimize.md`
- `docs/concepts/compile-and-epcontext.md`
- `docs/concepts/perf-and-monitoring.md`
- `docs/concepts/eval-and-datasets.md`

- [ ] **Step 1: Rename the 2 existing concept files**

Use `git mv` so history is preserved:

```bash
git mv docs/concepts/onnx-and-eps.md docs/concepts/eps-and-devices.md
git mv docs/concepts/hierarchy.md docs/concepts/hierarchy-and-metadata.md
```

The file contents are unchanged at this step; content edits happen in Batch B.

- [ ] **Step 2: Create 11 stub pages**

Each stub has this exact body shape, with `<Page Title>` filled per the table below:

```markdown
# <Page Title>

!!! note "Coming soon"
    This page is part of the v2 docs expansion and will be authored next.
```

| File path | Page Title |
|---|---|
| `docs/tutorials/index.md` | `Tutorials` |
| `docs/tutorials/npu-convnext.md` | `ConvNeXt on NPU` |
| `docs/concepts/graphs-and-ir.md` | `Models, graphs, and the ONNX IR` |
| `docs/concepts/tensors-and-dtypes.md` | `Tensors and dtypes` |
| `docs/concepts/primitives-and-pipeline.md` | `Primitives and pipeline` |
| `docs/concepts/config-and-build.md` | `Config and build` |
| `docs/concepts/load-and-export.md` | `Load and export` |
| `docs/concepts/analyze-and-optimize.md` | `Analyze and optimize` |
| `docs/concepts/compile-and-epcontext.md` | `Compile and EPContext` |
| `docs/concepts/perf-and-monitoring.md` | `Perf and monitoring` |
| `docs/concepts/eval-and-datasets.md` | `Eval and datasets` |

- [ ] **Step 3: Update `mkdocs.yml` nav**

Replace the existing `nav:` block with this exact block (the rest of the file — `site_name`, `theme`, `plugins`, `markdown_extensions`, `exclude_docs` — stays untouched):

```yaml
nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quickstart: getting-started/quickstart.md
      - End-to-End — HF → NPU: getting-started/end-to-end.md
  - Concepts:
      - Fundamentals:
          - How winml-cli works: concepts/how-it-works.md
          - Models, graphs, and the ONNX IR: concepts/graphs-and-ir.md
          - Tensors and dtypes: concepts/tensors-and-dtypes.md
          - Execution Providers and devices: concepts/eps-and-devices.md
          - Quantization and QDQ: concepts/quantization.md
          - Hierarchy and ONNX metadata: concepts/hierarchy-and-metadata.md
          - BuildConfig and kits: concepts/buildconfig.md
      - WinML CLI:
          - Primitives and pipeline: concepts/primitives-and-pipeline.md
          - Config and build: concepts/config-and-build.md
          - Load and export: concepts/load-and-export.md
          - Analyze and optimize: concepts/analyze-and-optimize.md
          - Compile and EPContext: concepts/compile-and-epcontext.md
          - Perf and monitoring: concepts/perf-and-monitoring.md
          - Eval and datasets: concepts/eval-and-datasets.md
  - Commands:
      - Overview: commands/overview.md
      - Discover:
          - sys: commands/sys.md
          - inspect: commands/inspect.md
          - hub: commands/hub.md
          - analyze: commands/analyze.md
      - Configure:
          - config: commands/config.md
          - optimize: commands/optimize.md
      - Build:
          - export: commands/export.md
          - quantize: commands/quantize.md
          - compile: commands/compile.md
          - build: commands/build.md
      - Measure:
          - perf: commands/perf.md
          - eval: commands/eval.md
  - Samples:
      - ConvNeXt — Primitives Walkthrough: samples/convnext-primitives.md
      - BERT — Config + Build + Perf: samples/bert-config-build.md
      - Qwen3 — Composite Models: samples/qwen3-composite.md
  - Tutorials:
      - Overview: tutorials/index.md
      - ConvNeXt on NPU: tutorials/npu-convnext.md
  - Reference: reference/index.md
  - Troubleshooting: troubleshooting.md
  - Contributing: contributing.md
```

- [ ] **Step 4: Verify strict build**

```bash
uv run mkdocs build --strict
```

Expected: exit 0, message `Documentation built in <N> seconds`, no WARNING lines.

If `--strict` errors with "doc file not found" for any of the 11 new files or the 2 renamed files, fix the path before continuing.

- [ ] **Step 5: Commit**

```bash
git add mkdocs.yml docs/concepts/ docs/tutorials/
git commit -m "docs: scaffold v2 expansion (stubs, renames, nav restructure)

Renames:
- concepts/onnx-and-eps.md -> concepts/eps-and-devices.md
- concepts/hierarchy.md -> concepts/hierarchy-and-metadata.md

Stubs created (content authored in next batches):
- tutorials/index.md, tutorials/npu-convnext.md
- concepts/graphs-and-ir.md, concepts/tensors-and-dtypes.md
- concepts/{primitives-and-pipeline,config-and-build,load-and-export,
  analyze-and-optimize,compile-and-epcontext,perf-and-monitoring,
  eval-and-datasets}.md

Nav restructured: Concepts sub-grouped into Fundamentals + WinML CLI;
Tutorials chapter inserted between Samples and Reference."
```

---

## Task 2: Concepts — Fundamentals authoring (Batch B)

**Files (full content authoring or content editing):**
- Modify: `docs/concepts/how-it-works.md` (rename-in-nav only — content kept; included here so the reviewer notices it)
- Modify: `docs/concepts/eps-and-devices.md` (already renamed in Task 1; small content reframe to match the new pair-topic title)
- Modify: `docs/concepts/hierarchy-and-metadata.md` (already renamed; broaden content to cover other metadata, not just `winml.hierarchy.tag`)
- Modify: `docs/concepts/buildconfig.md` (rename-in-nav only — content kept)
- Modify: `docs/concepts/quantization.md` (tighten — dtype family content moves out to Tensors page)
- Author: `docs/concepts/graphs-and-ir.md`
- Author: `docs/concepts/tensors-and-dtypes.md`

In total: **2 new pages authored, 3 pages content-edited, 2 pages untouched-but-renamed-in-nav**. The 2 untouched-in-nav pages need no editing in this batch.

### Voice anchor (read before dispatching agents)

The 5 existing Fundamentals pages (now renamed/touched) set the voice: clear, direct, 400–700 words, opens with a 1–2 paragraph lead, uses H2 sections, ends with a `## See also` block of 2–4 relative links. Every flag and symbol cited is verified in `src/winml/modelkit/`. No marketing language.

- [ ] **Step 1: Dispatch parallel author agents (wave 1)**

Send all 4 `Agent` tool calls in a single message; `subagent_type: general-purpose`, `model: sonnet`. Agents write only; the orchestrator commits.

#### Agent B1 — Author `concepts/graphs-and-ir.md` (new)

```
You are authoring ONE Concepts page for the winml-cli user-facing docs. Output: overwrite the stub at C:\Users\zhengte\BYOM\ModelKits\mvp\docs\concepts\graphs-and-ir.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Title: # Models, graphs, and the ONNX IR
Length: 400–700 words of prose.

Sources to read first (for accuracy — do not copy verbatim):
- src/winml/modelkit/onnx/ (directory; look at metadata.py and model detection helpers)
- src/winml/modelkit/export/ (directory; opset version selection)
- An external reference: https://github.com/onnx/onnx/blob/main/docs/IR.md (treat as background, do not link)

Body requirements:
1. Lead (1–2 paragraphs): what a model file is at rest; the model is a graph; graphs are described in the ONNX IR; opsets version the operator set.
2. H2 — "What is in a .onnx file": inputs, outputs, nodes (operators), initializers (weights), metadata. Use one short bulleted list.
3. H2 — "Graphs as IR": brief explanation that ONNX is an Intermediate Representation — a static computation graph that's portable across runtimes. Mention nodes have inputs/outputs that wire into the graph; this enables shape inference and EP-targeted compilation.
4. H2 — "Opsets and versioning": opset is a snapshot of the operator catalog at a specific version. winml-cli's `winml export` defaults to opset 17 (verify in src/winml/modelkit/export/ or commands/export.py). New opsets unlock new ops; EPs may not support the latest opset.
5. H2 — "See also": 2–4 relative links. Valid targets (relative to docs/concepts/):
   - eps-and-devices.md
   - tensors-and-dtypes.md
   - hierarchy-and-metadata.md
   - ../commands/inspect.md
   - ../commands/export.md

Rules:
- Use winml-cli (never ModelKit, never wmk).
- Verify opset default by reading the source. If you cannot confirm 17, state the actual default you found.
- No "TBD", no placeholders.
- Code blocks: ```bash for invocations, ```text for output.

Verify the strict build after writing:
  uv run mkdocs build --strict 2>&1 | tail -3

Expected: exit 0, no WARNING lines.

Return: status (DONE/DONE_WITH_CONCERNS/BLOCKED), word count estimate, last 3 lines of mkdocs build output, the opset version you cited and where you confirmed it.
```

#### Agent B2 — Author `concepts/tensors-and-dtypes.md` (new)

```
You are authoring ONE Concepts page for the winml-cli user-facing docs. Output: overwrite the stub at C:\Users\zhengte\BYOM\ModelKits\mvp\docs\concepts\tensors-and-dtypes.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Title: # Tensors and dtypes
Length: 500–800 words of prose (slightly longer than typical Fundamentals page because this absorbs the dtype content from the quantization page).

Sources to read first:
- src/winml/modelkit/commands/_options.py (for _KNOWN_PRECISIONS)
- src/winml/modelkit/onnx/ (for I/O tensor spec and shape inference helpers)
- src/winml/modelkit/commands/quantize.py (for activation_type / weight_type flags)
- src/winml/modelkit/commands/export.py (for --input-specs and --shape-config flags)

Body requirements:
1. Lead (1–2 paragraphs): three roles for tensors in a model — weights (static parameters), activations (intermediate values at inference), I/O tensors (inputs and outputs at the graph boundary). Each role has a dtype that may differ.
2. H2 — "Weights and activations": one paragraph explaining the distinction and why it matters (memory footprint, quantization granularity, EP support tiers).
3. H2 — "Dtype options in winml-cli": markdown table listing the precision strings from _KNOWN_PRECISIONS in _options.py. Columns: Precision | Weight dtype | Activation dtype | Notes. Cover at least auto, fp32, fp16, int8, int16, w8a8, w8a16, w4a16.
4. H2 — "Static vs dynamic shapes": one paragraph. ONNX supports symbolic dimensions ("batch", "sequence") that are resolved at runtime. winml-cli's --input-specs and --shape-config flags let you constrain these at export time. Some EPs (QNN) require fully static shapes; others (DirectML) accept dynamic.
5. H2 — "See also": 2–4 relative links. Valid targets:
   - quantization.md
   - eps-and-devices.md
   - graphs-and-ir.md
   - ../commands/export.md
   - ../commands/quantize.md

Rules:
- Verify every precision string against _KNOWN_PRECISIONS. Do not invent precisions.
- Verify --input-specs and --shape-config exist on winml export (read export.py).
- Use winml-cli (never ModelKit/wmk).
- No "TBD", no placeholders.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines, the precision list you enumerated (verbatim).
```

#### Agent B3 — Edit `concepts/eps-and-devices.md` (rename done; content reframe)

```
You are content-editing ONE Concepts page for the winml-cli user-facing docs. The page is at C:\Users\zhengte\BYOM\ModelKits\mvp\docs\concepts\eps-and-devices.md (just renamed from onnx-and-eps.md; content is the previous version). DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Goal: reframe the page title and lead from "ONNX & Execution Providers" to "Execution Providers and devices". The ONNX intro content should be trimmed (it now lives in graphs-and-ir.md) and the EP × Device matrix should remain front-and-center.

Length: 400–700 words after editing.

Specific edits:
1. Change the H1 to: # Execution Providers and devices
2. Rewrite the lead (1–2 paragraphs): what an EP is, what a device is, how winml-cli's --device and --ep flags map to them. Drop the "what is ONNX" intro paragraph (now covered by graphs-and-ir.md). If you reference ONNX, link to ../concepts/graphs-and-ir.md.
3. Keep the EP × Device table (and update it if you find a missed EP in src/winml/modelkit/sysinfo/).
4. Keep the "Device vs EP on the CLI" section.
5. Update the "## See also" block to include a link to graphs-and-ir.md and tensors-and-dtypes.md if not already present. Keep total at 2–4 links.

Rules:
- Use winml-cli (never ModelKit/wmk). Replace any "ModelKit" string you find inside the page with "winml-cli".
- Do not invent EPs. Verify against src/winml/modelkit/sysinfo/.
- No "TBD", no placeholders.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count after editing, last 3 lines of build, list of EP names referenced in the final table.
```

#### Agent B4 — Edit `concepts/hierarchy-and-metadata.md` (rename done; broaden content)

```
You are content-editing ONE Concepts page for the winml-cli user-facing docs. The page is at C:\Users\zhengte\BYOM\ModelKits\mvp\docs\concepts\hierarchy-and-metadata.md (just renamed from hierarchy.md; current content focuses only on hierarchy.tag). DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Goal: broaden the page from "what is hierarchy_tag" to "what metadata winml-cli writes into the ONNX model, and why each entry exists."

Length target: 500–700 words after editing.

Specific edits:
1. Change the H1 to: # Hierarchy and ONNX metadata
2. Rewrite the lead (1–2 paragraphs): ONNX files carry metadata_props key/value entries beyond the graph itself. winml-cli writes several of these. The most important is winml.hierarchy.tag (the PyTorch module-path tag), but there are others.
3. New H2 — "Metadata winml-cli writes": markdown table. Columns: Key | Set by | Purpose. Inspect src/winml/modelkit/onnx/metadata.py and src/winml/modelkit/export/htp/exporter.py to find the canonical list. Include winml.hierarchy.tag at minimum.
4. Existing H2 — "What hierarchy_tag enables": keep the existing content about per-module benchmarking (winml perf --module) and the --no-hierarchy / --clean-onnx flag on winml export.
5. Existing H2 — "See also": keep but add tensors-and-dtypes.md as a link.

Rules:
- Verify every metadata key by reading the source. State the file:line where you found each key.
- Use winml-cli (never ModelKit/wmk). Replace any "ModelKit" string with "winml-cli".
- No "TBD", no placeholders.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines, the list of metadata keys you documented with file:line evidence.
```

- [ ] **Step 2: Edit `concepts/quantization.md` (tighten — move dtype content to Tensors page)**

Read the current file first; find any paragraph or table that primarily explains the dtype family (fp32/fp16/int8/int16/compound types). Move that content (logically — by trimming here, since the new content already lives in tensors-and-dtypes.md after Step 1).

Make these specific edits to `docs/concepts/quantization.md`:

- If the page has an H2 like "Precision options" that lists the dtype family, replace its body with a short sentence: "See [Tensors and dtypes](tensors-and-dtypes.md) for the full precision family. This page focuses on the quantization algorithm, calibration, and the QDQ pattern."
- Otherwise no changes — the page can keep its calibration and QDQ content.

The orchestrator does this edit directly (not via agent) since it's a one-line surgical change.

- [ ] **Step 3: Edit `concepts/how-it-works.md` and `concepts/buildconfig.md` — verify they don't reference renamed files**

Read each file. If they contain links like `[ONNX & Execution Providers](onnx-and-eps.md)` or `[Hierarchy](hierarchy.md)`, update them to `eps-and-devices.md` and `hierarchy-and-metadata.md` respectively. Otherwise no changes.

- [ ] **Step 4: Verify strict build**

```bash
uv run mkdocs build --strict 2>&1 | tail -3
```

Expected: exit 0, no WARNING lines.

If the build complains about broken links pointing to `onnx-and-eps.md` or `hierarchy.md`, fix those references in whatever file they live in (these are the inbound links flagged in spec §7).

- [ ] **Step 5: Commit (Fundamentals batch)**

```bash
git add docs/concepts/
git commit -m "docs(concepts/fundamentals): author graphs-and-ir + tensors-and-dtypes; reframe eps-and-devices + hierarchy-and-metadata after rename

- New: graphs-and-ir.md (models, graphs, IR, opsets)
- New: tensors-and-dtypes.md (weights/activations/I-O tensors, precision
  family, static-vs-dynamic shapes)
- Reframed: eps-and-devices.md (drops the ONNX intro, now covered by
  graphs-and-ir.md; keeps EP × Device matrix)
- Broadened: hierarchy-and-metadata.md (now covers all metadata
  winml-cli writes, not only winml.hierarchy.tag)
- Tightened: quantization.md (dtype family content moved to
  tensors-and-dtypes.md to remove duplication)"
```

---

## Task 3: Concepts — WinML CLI authoring (Batch C)

**Files (all new, full content authoring):**
- `docs/concepts/primitives-and-pipeline.md`
- `docs/concepts/config-and-build.md`
- `docs/concepts/load-and-export.md`
- `docs/concepts/analyze-and-optimize.md`
- `docs/concepts/compile-and-epcontext.md`
- `docs/concepts/perf-and-monitoring.md`
- `docs/concepts/eval-and-datasets.md`

### Voice anchor

These are **workflow-concept pages**, not command-reference pages. Each explains the **why** and **when**, cross-linking to the per-command reference at `docs/commands/<name>.md` for **what**. No flag tables — that's the command-reference page's job.

- [ ] **Step 1: Dispatch parallel author agents (4 agents, 7 pages)**

Single message, 4 Agent tool calls, `model: sonnet`, agents write only.

| Agent | Pages owned |
|---|---|
| C1 | `primitives-and-pipeline.md`, `config-and-build.md` |
| C2 | `load-and-export.md`, `analyze-and-optimize.md` |
| C3 | `compile-and-epcontext.md`, `perf-and-monitoring.md` |
| C4 | `eval-and-datasets.md` |

#### Reusable agent prompt template

```
You are authoring Concepts pages for the winml-cli user-facing docs. Output: write the markdown files listed below. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Voice and shape per page:
- H1 = page title (given below per page).
- 400–700 words of prose.
- Lead (1–2 paragraphs): what conceptual tension/pair this page covers and why it matters.
- 2–4 H2 sections.
- Closing "## See also" with 2–4 relative links.
- These are workflow-concept pages: explain WHY and WHEN. The /commands/ pages cover WHAT flags.
- No flag tables. If you need to mention a flag, do it inline in prose.
- Use winml-cli throughout (never ModelKit/wmk).

Source verification rule: every flag, file, or symbol you cite must exist in src/winml/modelkit/. Verify by reading or running uv run winml <command> --help.

Pages assigned to you:

[PAGE BLOCKS — see below]

After all your pages are written, run:
  uv run mkdocs build --strict 2>&1 | tail -3

Expected: exit 0, no WARNING lines.

Return: status (DONE/DONE_WITH_CONCERNS/BLOCKED), per-page word count, build output last 3 lines, anything surprising (a flag that doesn't exist where the prompt says it should, a source claim you couldn't verify).
```

#### Page blocks — Agent C1

```
PAGE 1 — concepts/primitives-and-pipeline.md
Title: # Primitives and pipeline
Theme: Two ways to use winml-cli — invoke individual primitive commands (export, optimize, quantize, compile, perf, eval) one at a time, or use `winml build` as the wrapper that runs them all from a config. Teach when to choose which: primitives for learning / debugging / one-off variations; build for production / CI / reproducibility.

Required H2 sections:
- "The primitive commands" — list the staged commands with a one-line role each. Reference the order in docs/concepts/how-it-works.md.
- "The pipeline wrapper" — winml build orchestrates the same stages from a single WinMLBuildConfig.
- "When to choose which" — bullets contrasting the two.
- "See also" — 2–4 links. Valid: how-it-works.md, config-and-build.md, ../commands/build.md, ../samples/convnext-primitives.md, ../samples/bert-config-build.md.

PAGE 2 — concepts/config-and-build.md
Title: # Config and build
Theme: Producer/consumer pair. winml config generates a WinMLBuildConfig JSON; winml build consumes it. Teach the reproducibility angle (version configs, share across CI, replay later), and the override semantics (CLI flags can override config values).

Required H2 sections:
- "Generating a config" — short prose about winml config, --task, --no-quant/--no-compile, --trust-remote-code. No full flag table.
- "Consuming a config" — winml build -c <file>.json --output-dir or --use-cache (exactly one of them). The build runs the stages defined in the config.
- "Overrides at run time" — flags like --no-quant, --no-compile, --no-optimize on winml build override the corresponding config sections without editing the file. Useful for ad-hoc skips.
- "Why version a config" — three concrete reasons: reproducibility, CI, sharing.
- "See also" — 2–4 links. Valid: buildconfig.md, primitives-and-pipeline.md, ../commands/config.md, ../commands/build.md.
```

#### Page blocks — Agent C2

```
PAGE 1 — concepts/load-and-export.md
Title: # Load and export
Theme: The first conceptual stage of the pipeline — bring a model into memory (from Hugging Face Hub or a local checkpoint), then transform it to ONNX. Teach the load step (the loader module in src/winml/modelkit/loader/) and the export step (the winml export command).

NOTE: "load" is not a CLI verb. The loader is internal. Pair this page is "stage 1 load" + "stage 1 export"; both are part of getting a model into ONNX form.

Required H2 sections:
- "Loading a model" — winml-cli loads from HF Hub (with cache at ~/.cache/huggingface) or from a local PyTorch checkpoint. winml inspect is the user-facing way to check the loader picked it up correctly. Trust remote code with --trust-remote-code.
- "Exporting to ONNX" — winml export converts the loaded model to ONNX. Mentions hierarchy preservation (see hierarchy-and-metadata.md), the --no-hierarchy / --clean-onnx flag, and --dynamo for an alternative export backend.
- "Where it goes wrong" — task mismatch (use --task), shape issues (use --shape-config or --input-specs), custom modules (use --torch-module).
- "See also" — 2–4 links. Valid: hierarchy-and-metadata.md, graphs-and-ir.md, ../commands/inspect.md, ../commands/export.md.

PAGE 2 — concepts/analyze-and-optimize.md
Title: # Analyze and optimize
Theme: Two graph-quality commands that work together. winml analyze checks EP compatibility and reports issues; winml optimize applies fusions and rewrites. They share --optim-config and often run together via winml build's analyzer/optimizer loop.

Required H2 sections:
- "What analyze does" — runs operator coverage, shape inference, and runtime checks against a target EP; outputs a report. Reference the --format choices.
- "What optimize does" — applies graph fusions (GELU, LayerNorm, MatMul+Add) and pattern rewrites. References --list-capabilities and the --enable-X / --disable-X dynamic flags. Briefly mention --list-rewrites for the pattern-rewrite family.
- "The analyzer/optimizer loop" — winml build runs analyze → optimize → analyze → optimize up to --max-optim-iterations times to converge. Mention --no-analyze for deterministic single-pass builds.
- "See also" — 2–4 links. Valid: compile-and-epcontext.md, primitives-and-pipeline.md, ../commands/analyze.md, ../commands/optimize.md.
```

#### Page blocks — Agent C3

```
PAGE 1 — concepts/compile-and-epcontext.md
Title: # Compile and EPContext
Theme: What winml compile actually produces. Some EPs (especially QNN) bake a binary blob — the EP context — into the ONNX file at compile time. Compiled models load faster at runtime because the EP-specific setup is pre-computed.

Required H2 sections:
- "What compilation produces" — for ORT-compatible EPs the compile step writes an ONNX file that the runtime can load directly; for QNN the file embeds a binary EPContext blob.
- "Embedded vs external EPContext" — winml compile --embed controls whether the QNN context is inlined into the .onnx or stored as a sidecar binary. Trade-offs: inline = one file but bigger; sidecar = smaller .onnx but two files.
- "Why pre-compile" — runtime cold-start cost. The first inference on a fresh model loads + JIT-compiles; a pre-compiled model loads ready-to-run.
- "Skipping validation" — --no-validate exists for fast iteration; explain when not to use it (production builds).
- "See also" — 2–4 links. Valid: eps-and-devices.md, analyze-and-optimize.md, ../commands/compile.md, ../commands/build.md.

PAGE 2 — concepts/perf-and-monitoring.md
Title: # Perf and monitoring
Theme: winml perf measures latency/throughput. The --monitor flag adds a live hardware utilization chart (NPU primarily); --op-tracing produces per-operator timing breakdowns. Together they let you see both end-to-end numbers and where the time goes.

Required H2 sections:
- "What perf measures" — iterations, warmup, batch size; the output is latency p50/p90/mean and throughput. Mention --device for the EP target.
- "Live monitoring" — --monitor opens a terminal chart of NPU utilization while the benchmark runs. Useful for confirming the workload actually hit the NPU.
- "Per-operator tracing" — --op-tracing basic|detail produces breakdowns. Useful for finding hot ops.
- "Per-module benchmarking" — --module <substring> benchmarks just one HF/PyTorch module from the hierarchy (links to hierarchy-and-metadata.md).
- "See also" — 2–4 links. Valid: hierarchy-and-metadata.md, eval-and-datasets.md, ../commands/perf.md.
```

#### Page blocks — Agent C4

```
PAGE 1 — concepts/eval-and-datasets.md
Title: # Eval and datasets
Theme: winml eval measures accuracy, not speed. It needs a dataset (typically from Hugging Face) and a way to bind dataset columns to model inputs/outputs. Teach when to use eval (always after quantization), how to point it at a dataset, and the column-mapping pattern.

Required H2 sections:
- "What eval reports" — the metric depends on the task (accuracy for classification, mAP for detection, etc.). Output is a JSON with per-metric numbers; --format controls the form.
- "Picking a dataset" — --dataset accepts a Hugging Face dataset path; --dataset-name picks a config; --split selects which split (validation by default); --samples caps the count for quick checks. Note --streaming for large datasets.
- "Column mapping" — --column key=value to bind dataset columns to model inputs; --label-mapping for label index translation.
- "Why eval after quantization" — quantization is lossy; the only way to know you didn't break the model is to check accuracy. Link to quantization.md.
- "See also" — 2–4 links. Valid: quantization.md, perf-and-monitoring.md, ../commands/eval.md.
```

- [ ] **Step 2: Verify strict build**

```bash
uv run mkdocs build --strict 2>&1 | tail -3
```

Expected: exit 0, no WARNING lines.

- [ ] **Step 3: Commit (WinML CLI batch)**

```bash
git add docs/concepts/
git commit -m "docs(concepts/winml-cli): author 7 workflow-concept pages

Each page covers a winml-cli workflow pair, explaining the WHY and
WHEN of using the commands together. Pages: primitives-and-pipeline,
config-and-build, load-and-export, analyze-and-optimize,
compile-and-epcontext, perf-and-monitoring, eval-and-datasets.

No flag tables (those live on the per-command reference pages).
Every flag and symbol verified against src/winml/modelkit/."
```

---

## Task 4: Tutorials authoring (Batch D)

**Files (full content authoring):**
- `docs/tutorials/index.md` — short overview (~150 words)
- `docs/tutorials/npu-convnext.md` — the long-form tutorial (1500–2500 words)

### Why a single agent owns the tutorial

The ConvNeXt-on-NPU tutorial is one long page where prose voice and step transitions matter. A single agent produces more consistent voice than splitting it.

- [ ] **Step 1: Dispatch 1 agent for the tutorial + 1 agent for the index**

Single message, 2 parallel agents (different files, no conflict).

#### Agent D1 — Author `tutorials/npu-convnext.md`

```
You are authoring the flagship tutorial for the winml-cli docs site. Output: overwrite C:\Users\zhengte\BYOM\ModelKits\mvp\docs\tutorials\npu-convnext.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Title: # ConvNeXt on NPU
Model: facebook/convnext-tiny-224
Length: 1500–2500 words of prose (excluding code blocks).
Tone: classroom-style, prescriptive, every step has an explicit "what just happened" callout. Source: adapted from internal WinHECLab lab (saved at temp/winheclab-readme.md as background reference).

Required structure:

# ConvNeXt on NPU

[Lead — 2–3 paragraphs:
- Goal: take facebook/convnext-tiny-224 from Hugging Face to a benchmark-ready compiled model running on NPU.
- Primary hardware: Copilot+PC with Snapdragon X-class NPU (or comparable). Explicit CPU/DirectML fallback documented throughout.
- Two sections: Section A builds the model using primitive commands (so you understand each stage); Section B does the same thing with `winml build` (so you see the wrapper).]

## Prerequisites

- Windows 11 24H2 (required for NPU support)
- Copilot+PC with NPU (40+ TOPS recommended; CPU/DirectML works as fallback)
- Python 3.10, uv installed
- winml-cli installed (see [Installation](../getting-started/installation.md))
- For NPU: QNN SDK (set QNN_SDK_ROOT env var) or OpenVINO

## Section A — Primitive commands

### Step 1: Inspect the model

[bash block: uv run winml inspect -m facebook/convnext-tiny-224]
[text block: short abbreviated expected output]
[!!! note "What we just did" — explains: confirmed task detection, model class, exporter compatibility before transformation.]

### Step 2: Generate a build config

[bash block: uv run winml config -m facebook/convnext-tiny-224 -o convnext_config.json]
[!!! note callout: this is optional for primitives but useful for versioning.]

### Step 3: Export to ONNX

[bash block: uv run winml export -m facebook/convnext-tiny-224 -o convnext.onnx]
[Link to ../concepts/hierarchy-and-metadata.md re: what hierarchy preservation adds.]

### Step 4: Analyze for EP compatibility

[bash block: uv run winml analyze -m convnext.onnx --ep qnn]
(Show that analyze reports operator coverage and any flagged issues.)

### Step 5: Optimize the graph

[bash block: uv run winml optimize -m convnext.onnx -o convnext_optim.onnx]

### Step 6: Quantize

[bash block: uv run winml quantize -m convnext_optim.onnx -o convnext_int8.onnx --precision int8 --samples 32]
[Link to ../concepts/quantization.md.]

### Step 7: Compile for the target EP

Use pymdownx.tabbed for QNN vs OpenVINO:

=== "QNN (Snapdragon NPU)"

    ```bash
    # Requires QNN_SDK_ROOT env var set
    uv run winml compile -m convnext_int8.onnx -o convnext_qnn.onnx --device npu
    ```

=== "OpenVINO (Intel CPU/GPU/NPU)"

    ```bash
    uv run winml compile -m convnext_int8.onnx -o convnext_ov.onnx --device npu --ep openvino
    ```

=== "CPU fallback"

    ```bash
    uv run winml compile -m convnext_int8.onnx -o convnext_cpu.onnx --device cpu
    ```

[Link to ../concepts/compile-and-epcontext.md.]

### Step 8: Benchmark

Tabbed by EP:

=== "QNN NPU"

    ```bash
    uv run winml perf -m convnext_qnn.onnx --device npu --iterations 50 --monitor
    ```

=== "OpenVINO NPU"

    ```bash
    uv run winml perf -m convnext_ov.onnx --device npu --ep openvino --iterations 50 --monitor
    ```

=== "CPU"

    ```bash
    uv run winml perf -m convnext_cpu.onnx --device cpu --iterations 50
    ```

[text block: a short example latency/throughput snippet.]

### Step 9 (optional): Evaluate accuracy

[bash block: uv run winml eval -m convnext_int8.onnx --dataset imagenet-1k --split validation --samples 100 --device npu]
[Link to ../concepts/eval-and-datasets.md.]

## Section B — One-shot with `winml build`

```bash
uv run winml build -c convnext_config.json --output-dir convnext_out/
```

[Brief prose: this single command runs export → optimize → quantize → compile and produces the same final artifact. Use --no-quant / --no-compile / --no-optimize to skip stages.]

[Show a benchmark step at the end using the artifact from convnext_out/.]

## Where to go next

- [Concepts → How winml-cli works](../concepts/how-it-works.md)
- [Concepts → Compile and EPContext](../concepts/compile-and-epcontext.md)
- [Samples → ConvNeXt primitives walkthrough](../samples/convnext-primitives.md) (the CPU/GPU/NPU device comparison version of this material)
- [Commands → Overview](../commands/overview.md)

## See also

(2–4 relative links — pick the most relevant from above.)

Rules:
- Use winml-cli (never ModelKit/wmk).
- Every flag and command must exist in src/winml/modelkit/. Verify by running uv run winml <command> --help.
- For unverifiable claims (e.g. --device value names), DOUBLE-CHECK against source.
- Use pymdownx.tabbed syntax verbatim: `=== "Label"` then blank line then 4-space-indented code block.
- Output snippets use ```text and stay short (5–10 lines).
- No "TBD", no placeholders.
- Adapt the WinHECLab content but rewrite in our voice (drop "Step N" classroom numbering for primary headings; keep step numbering inside Section A only).
- DO NOT reference Visual Studio, Windows App SDK, C#, or any GUI app — Python/CLI only.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, total word count (prose only, exclude code blocks), build output last 3 lines, and confirmation that tabbed blocks rendered (mkdocs --strict accepts them).
```

#### Agent D2 — Author `tutorials/index.md`

```
You are authoring the Tutorials chapter overview page. Output: overwrite C:\Users\zhengte\BYOM\ModelKits\mvp\docs\tutorials\index.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Title: # Tutorials
Length: 100–250 words.

Required structure:

# Tutorials

[One paragraph framing: tutorials are linear, prescriptive, end-to-end walkthroughs. For lookup, use Concepts (the WHY/WHEN) or Commands (the WHAT). Tutorials sit alongside Samples (which are reference-style demos comparing options).]

## Available tutorials

| Tutorial | What you'll build | Hardware |
|---|---|---|
| [ConvNeXt on NPU](npu-convnext.md) | A quantized ConvNeXt image classifier compiled for Snapdragon NPU (with CPU/DirectML fallback) | Copilot+PC NPU primary; CPU works as fallback |

[One short closing paragraph noting more tutorials coming.]

Rules:
- Use winml-cli (never ModelKit/wmk).
- No "TBD", no placeholders.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines.
```

- [ ] **Step 2: Verify strict build**

```bash
uv run mkdocs build --strict 2>&1 | tail -3
```

- [ ] **Step 3: Commit (Tutorials batch)**

```bash
git add docs/tutorials/
git commit -m "docs(tutorials): add Tutorials chapter with ConvNeXt-on-NPU walkthrough

- tutorials/index.md: chapter overview + tutorial table
- tutorials/npu-convnext.md: end-to-end ConvNeXt build on NPU,
  adapted from the internal WinHECLab lab. Primitives walkthrough
  (Section A) covers each stage in turn; one-shot section (Section B)
  shows the same result via winml build. QNN, OpenVINO, and CPU
  paths shown via tabbed code blocks.

Python/winml-cli only — Visual Studio / Windows App SDK / C# app
content from the lab is deliberately out of scope for this iteration."
```

---

## Task 5: Getting Started polish (Batch E)

**Files (content edits to existing pages):**
- `docs/getting-started/installation.md`
- `docs/getting-started/quickstart.md`
- `docs/getting-started/end-to-end.md`

- [ ] **Step 1: Dispatch 3 parallel agents**

Single message, 3 Agent tool calls, `model: sonnet`. Agents write only.

#### Agent E1 — Edit `installation.md`

```
You are editing the winml-cli Installation page. File: C:\Users\zhengte\BYOM\ModelKits\mvp\docs\getting-started\installation.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Goal:
1. Rewrite the prerequisites table to be more specific about NPU requirements.
2. Add a fallback callout for users without NPU hardware.

Specific edits:
- Replace the existing "Prerequisites" section with a table that includes:
  - Windows 11 24H2 or later (required for NPU support)
  - Copilot+PC with NPU (40+ TOPS NPU recommended for NPU acceleration; not required for CPU/DirectML)
  - Python 3.10 (the project pins requires-python = ">=3.10,<3.11"; verify before stating)
  - uv (link https://github.com/astral-sh/uv)
  - git
- After the prereqs table, add a !!! note "No NPU?" callout: explain that --device auto falls back to CPU or DirectML, and the rest of the docs apply with minor flag differences.
- Otherwise keep the page (Install, Verify, Optional extras, Next steps sections all stay).
- Verify the existing extras text matches pyproject.toml lines 79–82.

Rules:
- Use winml-cli (never ModelKit/wmk).
- Keep page under 600 words.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines.
```

#### Agent E2 — Edit `quickstart.md`

```
You are editing the winml-cli Quickstart page. File: C:\Users\zhengte\BYOM\ModelKits\mvp\docs\getting-started\quickstart.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Goal: add winml sys --list-device --list-ep to the verify step. Otherwise leave the page alone.

Specific edit:
- Wherever the page currently shows `uv run winml sys` as the verify command (probably in a "Verify the install" or similar section), replace it with:

  ```bash
  uv run winml sys --list-device --list-ep
  ```

- Update the surrounding prose to mention that this enumerates available devices and execution providers (versus `winml sys` alone, which shows everything).
- No other changes.

Rules:
- Use winml-cli (never ModelKit/wmk).
- Keep page under 600 words.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines.
```

#### Agent E3 — Edit `end-to-end.md`

```
You are editing the winml-cli End-to-End page. File: C:\Users\zhengte\BYOM\ModelKits\mvp\docs\getting-started\end-to-end.md. DO NOT commit.

Working dir: C:\Users\zhengte\BYOM\ModelKits\mvp. Branch: docs/v2.

Goals:
1. Add --monitor to the winml perf step.
2. Add a short CPU-fallback section after the NPU section.
3. Align prereqs callout with the updated installation.md.

Specific edits:
- Wherever the page shows `uv run winml perf ... --device npu`, add --monitor:

  ```bash
  uv run winml perf -m convnext_npu_out/<artifact>.onnx --device npu --iterations 50 --monitor
  ```

  Add a sentence: "The --monitor flag opens a live chart of NPU utilization while the benchmark runs — confirmation that the workload actually hit the NPU."
- After the existing NPU perf step, add a new section:

  ```
  ## (Optional) CPU fallback

  If you don't have NPU hardware, the same artifact runs on CPU via DirectML:

  ```bash
  uv run winml perf -m convnext_npu_out/<artifact>.onnx --device cpu --iterations 50
  ```

  Latency will be higher than NPU but the build pipeline is otherwise identical.
  ```

- In the prerequisites section, reference the updated installation page (link relative path: ../getting-started/installation.md is wrong from within getting-started/ — use installation.md).

Rules:
- Use winml-cli (never ModelKit/wmk).
- Keep page under 1100 words.

Verify: uv run mkdocs build --strict 2>&1 | tail -3

Return: status, word count, build output last 3 lines.
```

- [ ] **Step 2: Verify strict build**

```bash
uv run mkdocs build --strict 2>&1 | tail -3
```

- [ ] **Step 3: Commit (Getting Started polish batch)**

```bash
git add docs/getting-started/
git commit -m "docs(getting-started): polish prereqs, add NPU monitoring, document CPU fallback

- installation.md: rewrite prereqs as a table (Windows 11 24H2,
  Copilot+PC, Python 3.10, uv, git); add 'No NPU?' callout pointing
  at --device auto and CPU/DirectML.
- quickstart.md: verify step now uses 'winml sys --list-device
  --list-ep' for a focused capability check.
- end-to-end.md: add --monitor to the perf step and a short
  CPU-fallback section after the NPU benchmark."
```

---

## Task 6: Cross-link sweep (Batch F)

**Files:** any docs file referencing the renamed `onnx-and-eps.md` or `hierarchy.md`.

- [ ] **Step 1: Find broken references**

```bash
echo "=== References to old onnx-and-eps.md ==="
grep -rn "onnx-and-eps\.md" docs/ 2>/dev/null | grep -v "docs/superpowers/"

echo ""
echo "=== References to old hierarchy.md (not hierarchy-and-metadata.md) ==="
grep -rn "hierarchy\.md" docs/ 2>/dev/null | grep -v "hierarchy-and-metadata\.md" | grep -v "docs/superpowers/"
```

Expected: zero or a small handful of matches. If empty, skip to Step 3.

- [ ] **Step 2: Fix any matches**

For each match, edit the file replacing:
- `onnx-and-eps.md` → `eps-and-devices.md`
- `hierarchy.md` → `hierarchy-and-metadata.md`

If many matches exist (≥3), use sed:

```bash
files_with_old_eps=$(grep -rl "onnx-and-eps\.md" docs/ | grep -v "docs/superpowers/")
files_with_old_hier=$(grep -rl "hierarchy\.md" docs/ | grep -v "hierarchy-and-metadata\.md" | grep -v "docs/superpowers/")
for f in $files_with_old_eps; do sed -i 's|onnx-and-eps\.md|eps-and-devices.md|g' "$f"; done
for f in $files_with_old_hier; do sed -i 's|\bhierarchy\.md|hierarchy-and-metadata.md|g' "$f"; done
```

- [ ] **Step 3: Verify strict build (final)**

```bash
uv run mkdocs build --strict 2>&1 | tail -3
```

Expected: exit 0, no WARNING lines.

- [ ] **Step 4: Commit (if any link fixes happened)**

```bash
git add docs/
git commit -m "docs: fix inbound links to renamed Fundamentals pages

Updates references to onnx-and-eps.md -> eps-and-devices.md and
hierarchy.md -> hierarchy-and-metadata.md across the docs tree.
Internal docs and the design/plan files under docs/superpowers/
are not touched."
```

If Step 1 found no matches, skip the commit — no changes to record.

- [ ] **Step 5: Final smoke check**

```bash
echo "=== Page count by chapter ===" && ls docs/getting-started/*.md docs/concepts/*.md docs/commands/*.md docs/samples/*.md docs/tutorials/*.md 2>&1 | wc -l

echo "=== Final commit log on docs/v2 (vs docs/v1) ===" && git log --oneline docs/v1..HEAD

echo "=== Working tree clean? ===" && git status --short
```

Expected: page count = 32 (3 + 14 + 13 + 3 + 2 - wait, recompute) — actually:
- getting-started: 3
- concepts: 14 (the 5 existing + 2 renamed-and-already-existing + 9 new = 14 — but two of those are renamed (eps-and-devices, hierarchy-and-metadata) so net file count after renames is still 14)
- commands: 13
- samples: 3
- tutorials: 2

Total: **35 markdown files** under those chapters. Plus index.md = 36 user-facing markdown files in the site (excluding stubs in reference/, troubleshooting.md, contributing.md).

If page count is off, investigate; otherwise the v2 expansion is complete.

---

## Self-review notes

- **Spec coverage:** Each section of `docs/superpowers/specs/2026-05-24-docs-expansion-v2-design.md` maps to a task. §4 IA → Task 1; §5.1 Getting Started → Task 5; §5.2 Tutorials → Task 4; §5.3 Concepts/Fundamentals → Task 2; §5.4 Concepts/WinML CLI → Task 3; §6 nav → Task 1; §7 file inventory → tasks 1–6; §8 implementation strategy → directly the 6 batches; §9 acceptance criteria → end of Task 6.
- **Type/name consistency:** `winml-cli` used throughout; file paths use `concepts/`, `tutorials/`, `getting-started/`. Pair-page H1 titles match `mkdocs.yml` nav labels.
- **No placeholders:** every step has actual content. Agent prompts are concrete and self-contained (no "see plan for details").
- **Agent parallelism is explicit** at the start of each authoring task.
- **One known acceptable hack:** in Task 2 Step 2, the dtype-content move is a surgical edit done by the orchestrator (not by an agent) because the edit is one paragraph or fewer.
