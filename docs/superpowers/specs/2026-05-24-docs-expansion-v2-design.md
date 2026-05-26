# Docs Expansion v2 — Design

> **Date:** 2026-05-24
> **Branch:** `docs/v2` (based on `docs/v1`)
> **Status:** Design approved verbally; ready for spec self-review and plan.

## 1. Goal

Expand the user-facing winml-cli docs site with: (a) a new **Tutorials** chapter seeded with a ConvNeXt-on-NPU walkthrough adapted from the internal WinHECLab lab, (b) a restructured **Concepts** chapter with two sub-groups (Fundamentals + WinML CLI) totaling 14 pages of pair-topic content, (c) targeted polish to the three existing Getting Started pages.

## 2. Scope and non-goals

### In scope

- 3 Getting Started pages: targeted edits (prereqs alignment, new flag mentions, CPU/DirectML fallback).
- 2 new Tutorial pages (chapter index + 1 ConvNeXt-on-NPU tutorial).
- 14 Concepts pages: 5 renamed/touched, 9 newly authored. Sub-grouped into Fundamentals and WinML CLI.
- `mkdocs.yml` nav restructure to expose Tutorials and the Concepts sub-groups.

### Out of scope

- The C# Windows App SDK demo app from WinHECLab Steps 9–19 (Python/winml-cli only this iteration).
- Visual Studio / Windows App SDK prerequisites.
- Hardware-specific lab paths (`C:\LabWinML\...`, `Start\`, `Final\`).
- Pinned wheel/SDK versions (we use `>=` semantics).
- Reference, Troubleshooting, Contributing chapters (still P2 stubs).
- A second tutorial or further Concepts pages beyond the 14 listed.

## 3. Source material

- **WinHECLab README** (`we2-microsoft/WinHECLab`, fetched to `temp/winheclab-readme.md` for this design). External publish OK per design discussion.
- **Existing winml-cli sources** at `src/winml/modelkit/` (canonical for any flag or behavior we describe).
- **Existing docs** at `docs/getting-started/`, `docs/concepts/`, `docs/commands/`, `docs/samples/`.

## 4. Information architecture changes

### 4.1 New chapter: Tutorials

A new top-level chapter between **Samples** and **Reference**:

```
- Samples
- Tutorials              ← NEW
    - Overview
    - ConvNeXt on NPU
- Reference
```

The chapter is the home for classroom-style, prescriptive, end-to-end walkthroughs. Distinct from **Samples** (which are reference-style, command-comparison demos) and from **Getting Started** (which is a short onboarding journey).

### 4.2 Concepts restructure

Concepts gets two sub-groups in the nav:

```
- Concepts
    - Fundamentals
        - How winml-cli works
        - Models, graphs, and the ONNX IR
        - Tensors and dtypes
        - Execution Providers and devices
        - Quantization and QDQ
        - Hierarchy and ONNX metadata
        - BuildConfig and kits
    - WinML CLI
        - Primitives and pipeline
        - Config and build
        - Load and export
        - Analyze and optimize
        - Compile and EPContext
        - Perf and monitoring
        - Eval and datasets
```

Every page uses the **pair-topic** framing — the H1 names two related concepts whose contrast or interplay structures the page.

## 5. Per-page detail

### 5.1 Getting Started — 3 pages, targeted edits

#### `installation.md`
- Rewrite prereqs table in lab style: Windows 11 24H2, Copilot+PC 40+ TOPS NPU (recommended for NPU acceleration), Python 3.10, uv, git. Drop the VS / App SDK lines (those were never in our installation anyway — confirming they stay out).
- Add a one-paragraph **"No NPU? Use `--device auto`"** callout that explicitly names CPU and DirectML as the fallback.

#### `quickstart.md`
- Add `winml sys --list-device --list-ep` to the verify step.
- No other changes — quickstart stays a 5-minute zero-to-export.

#### `end-to-end.md`
- Add `--monitor` to the `winml perf` step (live NPU utilization chart).
- Add a short CPU-fallback section after the NPU section showing the same `winml perf` with `--device cpu`.
- Align prereqs callout with the updated `installation.md`.
- Model stays **ConvNeXt** (consistency with existing sample pairing).

### 5.2 Tutorials — 2 new pages

#### `tutorials/index.md` (Overview)
- One paragraph framing: tutorials are linear, end-to-end, prescriptive; for lookup go to Concepts or Commands.
- One-row table linking to the available tutorials.
- ~150 words. Grows as more tutorials are added.

#### `tutorials/npu-convnext.md` (ConvNeXt on NPU)
- **Model:** `facebook/convnext-tiny-224`.
- **Hardware:** Primary path is Copilot+PC NPU; explicit CPU/DirectML fallback documented throughout.
- **Structure:**
  1. **Prerequisites** — adopted from WinHECLab prereqs table.
  2. **Section A — Primitives walkthrough**: `inspect → config → export → analyze → optimize → quantize → compile → perf`. EP-specific steps (`compile` and `perf`) use **`=== "QNN" / === "OpenVINO"` tabbed code blocks** so readers see both NPU backends inline.
  3. **Section B — One-shot with `winml build`**: closing section showing the wrapper command produces the same artifact.
  4. **(Optional) Eval** against an ImageNet sample using `winml eval`.
  5. **Where to go next** — links to Concepts and Samples.
- **Length target:** 1,500–2,500 words. This is the longest single page in the site.

### 5.3 Concepts — Fundamentals (7 pages)

Each page uses pair-topic framing. New = needs full authoring; touched = exists but renamed/expanded.

| File | Status | Pair / focus |
|---|---|---|
| `concepts/how-it-works.md` | **touched** (rename in nav, content kept) | Pipeline overview, mermaid diagram |
| `concepts/graphs-and-ir.md` | **new** | What is a model file, graph nodes/edges, opsets, ONNX as IR |
| `concepts/tensors-and-dtypes.md` | **new** | Weights vs activations vs I/O tensors; fp32/fp16/int8/int16; static vs dynamic shapes |
| `concepts/eps-and-devices.md` | **touched** (renamed from `onnx-and-eps.md`) | EP vs device, the EP matrix, when to use which |
| `concepts/quantization.md` | **touched** (small content tightening; dtype family moves to Tensors page) | Why quantize, calibration, QDQ pattern |
| `concepts/hierarchy-and-metadata.md` | **touched** (renamed from `hierarchy.md`, broadened) | `winml.hierarchy.tag` plus other metadata winml-cli writes |
| `concepts/buildconfig.md` | **touched** (rename in nav, content kept) | WinMLBuildConfig structure, kits, MODEL_BUILD_CONFIGS |

**Rename mapping:**
- `onnx-and-eps.md` → `eps-and-devices.md`
- `hierarchy.md` → `hierarchy-and-metadata.md`
- The other three existing pages keep their file names; only the nav label changes.

Any inbound links from other docs files (Commands, Samples, Getting Started, Tutorials) must be updated to the new file paths.

### 5.4 Concepts — WinML CLI (7 new pages)

All seven are new. Each is a workflow concept page (the **why** and **when**), not a command reference (the **what**). Cross-link to per-command pages under `docs/commands/`.

| File | Pair / focus |
|---|---|
| `concepts/primitives-and-pipeline.md` | Staged commands (`export`, `quantize`, `compile`, …) vs the one-shot `winml build` wrapper. When to choose which. Opens the chapter. |
| `concepts/config-and-build.md` | `winml config` produces a `WinMLBuildConfig`; `winml build` consumes it. The wrapper-flow pair — reproducibility, sharing configs across runs and CI, override flags vs config values. |
| `concepts/load-and-export.md` | The "load model into memory, then transform it to ONNX" arc. Covers HF Hub loading, local PyTorch loading, the `winml inspect` pre-flight check, and `winml export` itself. (Note: `winml load` is not a CLI verb — "load" here is the conceptual stage in the loader module, paired with the `export` command that follows it.) |
| `concepts/analyze-and-optimize.md` | Graph-quality commands. How analyze reports problems and how optimize applies fusions. Shared `--optim-config`. |
| `concepts/compile-and-epcontext.md` | What `winml compile` produces. QNN EPContext binary blobs embedded in ONNX. Why compiled models load faster at runtime. |
| `concepts/perf-and-monitoring.md` | `winml perf` plus `--monitor` (live NPU chart) and `--op-tracing`. When to use each. |
| `concepts/eval-and-datasets.md` | `winml eval` plus dataset semantics (`--dataset`, `--split`, `--column`, `--label-mapping`). When eval matters. |

**Length target per page:** 400–700 words of prose. Same shape as the existing Concepts pages.

**Discipline:** workflow pages explain *why and when*; command pages document *what flags exist*. No flag-table duplication.

## 6. `mkdocs.yml` nav changes

Full updated nav structure:

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
  - Commands: (unchanged)
  - Samples: (unchanged)
  - Tutorials:
      - Overview: tutorials/index.md
      - ConvNeXt on NPU: tutorials/npu-convnext.md
  - Reference: (unchanged P2 stub)
  - Troubleshooting: (unchanged P2 stub)
  - Contributing: (unchanged P2 stub)
```

## 7. File-system changes summary

### New files (11)

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

### Renamed files (2 — rename + content edit)

- `docs/concepts/onnx-and-eps.md` → `docs/concepts/eps-and-devices.md` (small content reframe to match the new pair-topic title)
- `docs/concepts/hierarchy.md` → `docs/concepts/hierarchy-and-metadata.md` (broaden content to cover other metadata winml-cli writes, not just `winml.hierarchy.tag`)

### Modified files (5 — content edits, no rename)

- `docs/getting-started/installation.md`
- `docs/getting-started/quickstart.md`
- `docs/getting-started/end-to-end.md`
- `docs/concepts/quantization.md` (tightening — dtype content moves to the new Tensors page)
- `mkdocs.yml` (full nav restructure to introduce the Concepts sub-groups and Tutorials chapter)

### Inbound links to update

Any reference to `onnx-and-eps.md` or `hierarchy.md` from other pages (Commands, Samples, Tutorials, Getting Started) must be updated to the new paths. Estimated 6–10 inbound links across the site (to be confirmed during implementation).

## 8. Implementation strategy preview

For the plan to formalize:

- **Batch A — Scaffolding (sequential, foundation):** create stubs for all 11 new pages; rename the 2 renamed pages; update `mkdocs.yml` nav. Verify `mkdocs build --strict` passes with stubs. Single commit.
- **Batch B — Concepts (Fundamentals) authoring (parallel, 4–5 agents):** new pages (`graphs-and-ir`, `tensors-and-dtypes`) authored in parallel with content-touch passes on the 5 existing pages.
- **Batch C — Concepts (WinML CLI) authoring (parallel, 3–4 agents):** 7 new workflow pages, agents own 2 pages each (one agent owns 1).
- **Batch D — Tutorials authoring (sequential, 1 agent):** the ConvNeXt-on-NPU tutorial. Single big page — best authored by one agent for consistency. Plus the small overview page.
- **Batch E — Getting Started polish (parallel, 3 agents):** small edits to the 3 existing pages.
- **Batch F — Cross-link fix-up (sequential):** sweep the rest of the docs site for inbound links to the renamed files and update them.

Each batch ends with `uv run mkdocs build --strict` to catch broken links.

## 9. Acceptance criteria

- `uv run mkdocs build --strict` exits 0 with zero warnings on the final commit.
- All 11 new pages exist and contain non-stub content of at least 300 words each (Tutorials index is exempt — it's a short overview).
- All 2 renamed pages have been renamed at the filesystem level (not just nav).
- No remaining inbound links reference the old paths `onnx-and-eps.md` or `hierarchy.md`.
- Tutorial uses `facebook/convnext-tiny-224`, contains tabbed QNN/OpenVINO code blocks for EP-specific steps, contains both a primitives section and a one-shot `winml build` section.
- Every flag mentioned in the new content is verified against `src/winml/modelkit/commands/` source (no invented flags).
- Existing internal docs (`docs/design/`, `docs/naming-convention.md`, `docs/pytest-best-practices.md`, `docs/superpowers/`) are unmodified.
- All commits remain on local `docs/v2` until publish.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 19 doc pages is a lot — author agents may drift from each other in tone | Provide every agent the same template and a short "voice guide" excerpt; require source-grounded claims; consider splitting Batch B into two waves so the first wave's voice anchors the second |
| Inbound-link sweep is easy to miss | Dedicated final batch (F) with `grep` verification before commit |
| `winml.hierarchy.tag` and other metadata details are real source claims | Each agent verifies via source path + line; reported in the agent's return summary |
| Tutorial scope creep (toward classroom-style screenshots etc.) | Length cap (1,500–2,500 words); no screenshots in this iteration |
| ConvNeXt + ConvNeXt overlap between `samples/convnext-primitives.md` and `tutorials/npu-convnext.md` | Sample focuses on **device comparison** (CPU/GPU/NPU); tutorial focuses on **NPU production path** (QNN vs OpenVINO). Different teaching purposes documented in each page's intro paragraph |

## 11. Open items explicitly punted

- A second tutorial (e.g. BERT-config-build on a fresh model). Available content-wise from WinHECLab but deferred to v3.
- Screenshots and embedded outputs. Not in this iteration; can add later under `docs/tutorials/images/`.
- Reference, Troubleshooting, Contributing chapter content. Still P2.
- Versioning (mike plugin). Still P2 from the v1 spec.
- Migration of internal `docs/design/` content into the public docs. Not in scope.
