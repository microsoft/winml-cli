# ModelKit User-Facing Documentation Site — Design

> **Date:** 2026-05-20
> **Branch:** `docs/init` (based on `feat/mvp`)
> **Status:** Design approved; ready for implementation plan.

## 1. Goal

Create a user-facing documentation site for ModelKit (the Python toolkit fronted by the `winml` CLI) targeted at external open-source users discovering the project on GitHub. The site must support markdown authoring, code-block-friendly rendering, mermaid diagrams, and optional Jupyter notebook embedding.

## 2. Audience and scope

- **Primary audience:** External OSS users (developers exporting/quantizing/compiling models for Windows ML deployment). No insider jargon; clear install-to-first-success path required.
- **Out of scope:** Internal-only sections, MS-internal access controls.
- **MVP scope:** Full content for the first four chapters (Getting Started, Concepts, Commands, Samples). Reference / Troubleshooting / Contributing exist as nav stubs only and are tracked as P2.

## 3. Framework decision

**MkDocs Material**, hosted on GitHub Pages, sources in `docs/`.

| Considered | Outcome |
|---|---|
| **MkDocs Material** | Chosen. Python-native, single `uv add --dev mkdocs-material`, first-class mermaid, code-block tabs, instant search, dark mode. Matches existing toolchain. |
| Sphinx + MyST + Furo | Rejected for MVP. Heavier config; autodoc not needed for a CLI tool. Revisit if we add a library API surface. |
| Docusaurus | Rejected. Adds Node ecosystem to a Python repo; MDX features unused. |
| GitHub Wiki | Rejected. No PR review, no code-search integration, weaker mermaid/notebook support. |

**Notebook integration:** `mkdocs-jupyter` plugin, treated as nice-to-have. No notebooks required in MVP; plugin is installed so future samples can drop in `.ipynb` files.

## 4. Hosting and deploy

- Site lives in-repo under `docs/` (alongside existing internal docs, which remain untouched and excluded from the nav).
- Built by GitHub Actions, published to the `gh-pages` branch, served by GitHub Pages.
- **Deploy is held off for now:** the CI workflow is written but configured to require manual `workflow_dispatch`. No automatic pushes to remote during this MVP. All commits stay local on `docs/init` until the user decides to publish.

## 5. Information architecture

```
ModelKit Docs
├── Home (landing)
│
├── 1. Getting Started
│   ├── Installation
│   ├── Quickstart (5-min export)
│   └── End-to-End: HF → NPU (15-min walkthrough)
│
├── 2. Concepts
│   ├── How ModelKit Works (pipeline diagram)
│   ├── ONNX & Execution Providers
│   ├── Quantization & QDQ
│   ├── Hierarchy Preservation
│   └── BuildConfig & Kits
│
├── 3. Commands
│   ├── Overview (12-command map, decision table)
│   ├── Discover  → sys, inspect, hub, analyze
│   ├── Configure → config, optimize
│   ├── Build     → export, quantize, compile, build
│   └── Measure   → perf, eval
│
├── 4. Samples
│   ├── ConvNeXt — primitives walkthrough (all EPs, quantized)
│   ├── BERT — config + build + perf (workflow focus)
│   └── Qwen3 — Composite Models (placeholder, "coming soon")
│
├── 5. Reference          (P2 — nav stub only)
├── 6. Troubleshooting    (P2 — nav stub only)
└── 7. Contributing       (P2 — nav stub only)
```

### 5.1 Grouping rationale

- Commands grouped by **user intent** (discover / configure / build / measure), not alphabetical — matches how a user actually progresses.
- Concepts placed **before** Commands so users have a mental model before reading flag tables.
- Existing `docs/design/`, `docs/naming-convention.md`, `docs/pytest-best-practices.md` stay where they are; they remain contributor-facing and are linked from Contributing (P2).

## 6. Per-page outlines

### 6.1 Getting Started

- **Installation** — Prereqs (Win 10/11, Python 3.10, `uv`), `git clone` + `uv sync`, verify with `winml sys`.
- **Quickstart** — 5-minute path: pick any HF classifier, run `winml export`, view the `.onnx`, run `winml inspect`. No EPs, no quantization — proves the install.
- **End-to-End: HF → NPU** — 15-minute walkthrough: ConvNeXt + `winml build` with QNN, see artifacts, run `winml perf` against NPU. Sets the stage for the Samples chapter.

### 6.2 Concepts

- **How ModelKit Works** — Mermaid pipeline diagram (PyTorch → ONNX → QDQ → EP-compiled). One paragraph per stage with deep-links to its command page.
- **ONNX & Execution Providers** — What ONNX is, what an EP is, EPs ModelKit supports (QNN, OpenVINO, DML, CPU/GPU), hardware mapping table.
- **Quantization & QDQ** — Why quantize, INT8/INT16/FP16, calibration vs. static, QDQ node insertion, lossy trade-offs.
- **Hierarchy Preservation** — Why ONNX needs PyTorch module info, how ModelKit embeds it as metadata, what it enables downstream (per-module benchmarking, targeted optimization).
- **BuildConfig & Kits** — The unified config object, precision policies, per-task templates, where configs live (`MODEL_BUILD_CONFIGS`).

### 6.3 Commands

**Page template** (applied to all 12 command pages — sections kept as headings even if initially sparse; content filled in incrementally):

```
# winml <command>
> one-line tagline

## When to use this
[1–2 sentences: user intent, place in pipeline]

## Synopsis
$ winml <command> [options]

## Flags
[Table: Flag | Short | Type | Default | Description; shared flags collapsed]

## How it works
[2–3 sentences; optional mermaid diagram for non-trivial commands]

## Examples
[3–5 progressively richer examples with expected output snippets]

## Common pitfalls
[Bullet list of gotchas]

## See also
[Links to related commands and concept pages]
```

The **Overview** sub-page contains the 12-command map (grouped) and a "which command for which task" decision table.

The 12 command pages: `sys`, `inspect`, `hub`, `analyze`, `config`, `optimize`, `export`, `quantize`, `compile`, `build`, `perf`, `eval`.

### 6.4 Samples

Each sample has a distinct teaching purpose — together they form an abstraction ladder.

- **ConvNeXt — primitives walkthrough**
  - Style: invoke each command directly (`inspect` → `config` → `export` → `quantize` → `compile` → `perf` → `eval`).
  - EP coverage: CPU, GPU, NPU. For each EP, document the flags that differ, expected outputs, and a "what we just did" callout per step.
  - Goal: reader leaves understanding what each command does and how they compose.

- **BERT — config + build + perf**
  - Style: `winml config` to generate the BuildConfig, `winml build` to run the whole pipeline, `winml perf` on the artifact.
  - EP focus de-emphasized — the page teaches the wrapper workflow, not the EP matrix.
  - Goal: reader leaves understanding the production-style one-shot path and how config files become reusable.

- **Qwen3 — Composite Models** (placeholder)
  - Single page: 1-paragraph teaser, "coming soon" admonition, link to the in-progress feature branch.
  - Goal: reserve the slot in the nav; signal where ModelKit is headed without blocking MVP on unmerged work.

## 7. Reference handling (P2 — nav stubs in MVP)

- **BuildConfig schema, hub catalog, EP/device matrix, precision options:** hand-written when the time comes (decided against autogeneration for MVP — maintenance burden traded against polish).
- **Naming conventions:** existing `docs/naming-convention.md` will be linked from the Reference page when written.

## 8. Repository layout

```
mvp/
├── docs/
│   ├── index.md                          ← landing
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── end-to-end.md
│   ├── concepts/
│   │   ├── how-it-works.md
│   │   ├── onnx-and-eps.md
│   │   ├── quantization.md
│   │   ├── hierarchy.md
│   │   └── buildconfig.md
│   ├── commands/
│   │   ├── overview.md
│   │   ├── sys.md
│   │   ├── inspect.md
│   │   ├── hub.md
│   │   ├── analyze.md
│   │   ├── config.md
│   │   ├── optimize.md
│   │   ├── export.md
│   │   ├── quantize.md
│   │   ├── compile.md
│   │   ├── build.md
│   │   ├── perf.md
│   │   └── eval.md
│   ├── samples/
│   │   ├── convnext-primitives.md
│   │   ├── bert-config-build.md
│   │   └── qwen3-composite.md            ← placeholder
│   ├── reference/                        ← P2 stubs
│   ├── troubleshooting.md                ← P2 stub
│   ├── contributing.md                   ← P2 stub
│   │
│   ├── design/                           ← UNCHANGED (internal)
│   ├── naming-convention.md              ← UNCHANGED (internal)
│   ├── pytest-best-practices.md          ← UNCHANGED (internal)
│   └── superpowers/specs/                ← UNCHANGED (this file lives here)
│
├── mkdocs.yml                            ← new
└── .github/workflows/docs.yml            ← new (manual dispatch only)
```

## 9. MkDocs configuration

- **Theme:** `material` with palette toggle (light/dark), instant navigation, code-copy button, "Edit on GitHub" link per page.
- **Plugins:** `search` (built-in), `mkdocs-jupyter` (notebooks; lazy install).
- **Markdown extensions:** `pymdownx.superfences` (mermaid, tabbed code), `admonition`, `pymdownx.tabbed`, `pymdownx.details`, `pymdownx.tasklist`.
- **Nav:** hand-written, mirroring section 5. Chapters 5-7 appear as stub pages in nav.
- **Strict mode:** `mkdocs build --strict` to fail CI on broken links or missing nav entries.
- **Excluded from nav:** `docs/design/`, `docs/superpowers/`, `docs/naming-convention.md`, `docs/pytest-best-practices.md` (they remain in the repo for contributors).

## 10. CI workflow

- **File:** `.github/workflows/docs.yml`.
- **Triggers:** `workflow_dispatch` only (manual) until the user is ready to publish. No auto-trigger on `push` or `pull_request` for MVP.
- **Steps:** checkout → install `uv` → `uv sync` → `uv run mkdocs build --strict` → `peaceiris/actions-gh-pages` deploy to `gh-pages`.
- **Local equivalent:** `uv run mkdocs serve` for live preview during authoring.

## 11. Implementation strategy (preview for the plan)

The plan will batch work for parallel execution via subagents:

- **Batch A — Site scaffold (sequential, foundation):** create `mkdocs.yml`, repo layout, landing page, nav stubs, CI workflow. Verify `mkdocs build --strict` succeeds with placeholder content.
- **Batch B — Concepts pages (5 pages, parallel):** one subagent per concept page; each reads the relevant source module and drafts the page.
- **Batch C — Command pages (12 command pages + 1 overview page, 4 parallel agents):** one agent per group (Discover / Configure / Build / Measure), each owning 3 commands; agents read source + `--help` output and draft pages using the section 6.3 template. The Commands → Overview page is authored after the 12 command pages settle (sequential), so its decision table reflects the real flag surfaces.
- **Batch D — Sample pages (3 pages, parallel):** ConvNeXt agent runs the primitive command sequence end-to-end and captures real outputs; BERT agent runs `config + build + perf` and captures outputs; Qwen3 page is a static placeholder.
- **Batch E — Getting Started (3 pages, sequential after Concepts and Commands):** authored last so it can cross-link to settled concept and command pages.

Each batch ends with `mkdocs build --strict` to catch broken links before moving on.

## 12. Open items / things explicitly punted

- **Versioning:** Not added in MVP. `mike` plugin available if needed later.
- **Search analytics, Algolia DocSearch:** Not in MVP; Material's built-in search is sufficient.
- **API reference autogeneration:** Not in MVP. Reconsider if/when a stable library API emerges.
- **i18n:** Not in MVP.

## 13. Acceptance criteria

- `uv run mkdocs serve` renders the site locally without errors.
- `uv run mkdocs build --strict` succeeds (no broken links, no missing nav entries).
- All chapters 1-4 have authored content; chapters 5-7 have stub pages.
- Mermaid diagrams render in the "How it works" concept page.
- Existing `docs/design/`, `docs/naming-convention.md`, `docs/pytest-best-practices.md` are unmodified and not in the user-facing nav.
- All commits remain on local `docs/init`; nothing pushed to `origin`.
