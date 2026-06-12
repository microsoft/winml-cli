# Contributing to winml-cli docs

This folder hosts the source for the [winml-cli](https://github.com/microsoft/winml-cli) documentation site, built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Quick reference

| Task | Command |
|---|---|
| Install dev deps | `uv sync --extra dev` |
| Live preview | `uv run mkdocs serve` |
| Build for CI | `uv run mkdocs build --strict` |
| Publish (one-shot from laptop) | `uv run mkdocs gh-deploy --force` |
| Publish (CI workflow) | GitHub Actions → "Build & Publish Docs" → Run workflow |

## What's in here

```
docs/
├── index.md                          ← landing page
├── getting-started/                  ← 3 onboarding pages
├── concepts/                         ← 12 conceptual pages in two sub-groups
│   ├── how-it-works.md, graphs-and-ir.md, weight-and-activation.md,
│   │     eps-and-devices.md, quantization.md         (Fundamentals)
│   └── primitives-and-pipeline.md, config-and-build.md, load-and-export.md, analyze-and-optimize.md,
│         compile-and-epcontext.md, perf-and-monitoring.md, eval-and-datasets.md                         (WinML CLI workflows)
├── commands/                         ← per-command reference (overview + 12 commands)
├── samples/                          ← reference-style walkthroughs
├── tutorials/                        ← classroom-style walkthroughs
├── reference/                        ← P2 stubs
├── troubleshooting.md                ← P2 stub
├── contributing.md                   ← P2 stub
│
├── superpowers/                      ← specs, plans, review notes (excluded from build)
├── design/                           ← internal ADRs and design docs (excluded)
├── naming-convention.md              ← internal style guide (excluded)
└── pytest-best-practices.md          ← internal style guide (excluded)
```

The site config (`mkdocs.yml`) lives at the repo root, not inside `docs/`. The build outputs to `site/` (gitignored).

## Local development

### Prerequisites

Python 3.10+ and [uv](https://github.com/astral-sh/uv).

### Setup and preview

```bash
# from the repo root
uv sync --extra dev
uv run mkdocs serve
```

Open http://127.0.0.1:8000/ in a browser. The server auto-reloads when you edit any `.md` file under `docs/`. Changes to `mkdocs.yml` (nav, theme, plugins) require a manual server restart.

### Validate before pushing

```bash
uv run mkdocs build --strict
```

`--strict` must exit 0 with no `WARNING` lines. Common causes of strict-mode failures:

- A new page added without an entry in `nav:` (gives a "not included in nav" warning)
- A nav entry pointing at a file that doesn't exist
- A relative link like `[text](other-page.md)` whose target file is missing
- A markdown anchor like `[link](#section-heading)` that doesn't match any heading slug

## Publishing

The site publishes to **GitHub Pages** from the `gh-pages` branch. The repo's `Settings → Pages` source is set to "Deploy from a branch" → `gh-pages` → `/ (root)`.

### One-shot publish from your laptop

```bash
uv run mkdocs gh-deploy --force
```

This builds the site locally, commits the static HTML to a local `gh-pages` branch, and force-pushes it to `origin/gh-pages`. GitHub Pages picks up the new commit within ~30–60 seconds.

### Publish via CI

The workflow at `.github/workflows/docs.yml` does the same thing in CI:

1. `Settings → Actions → Build & Publish Docs → Run workflow`
2. Select the branch you want to publish from (typically `main`)

The workflow is `workflow_dispatch` only — there is no automatic publish on push. If you want auto-publish on every push to `main`, change the trigger:

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - 'pyproject.toml'
      - '.github/workflows/docs.yml'
  workflow_dispatch:
```

## Authoring conventions

- **Product name**: `winml-cli` (lowercase, hyphenated) throughout user-facing prose. Use `WinML CLI` (or `Windows ML`) only where the broader Microsoft brand is meant.
- **Command name**: the CLI invocation is always `winml <subcommand>`. Never `wmk`.
- **Flag verification**: every flag mentioned in docs must exist in `src/winml/modelkit/commands/<cmd>.py`. Run `uv run winml <cmd> --help` to confirm.
- **Source citations**: when documenting source-grounded behavior (e.g., "the default opset is 17"), cite the file path and ideally the symbol name. Avoid line numbers — they drift fast.
- **Mermaid diagrams**: use `pymdownx.superfences` syntax (already configured in `mkdocs.yml`).
- **Tabbed code blocks**: use `pymdownx.tabbed` (`=== "Label"` followed by a blank line and 4-space-indented code block).
- **Admonitions**: `!!! note "Title"`, `!!! warning "Title"`, `!!! info "Title"`.
- **No emojis** in pages unless they're part of an external attribution (e.g., a GitHub badge).

## Excluded paths

The following are present in `docs/` but **excluded from the published site** via the `exclude_docs:` block in `mkdocs.yml`. They are kept in-repo for contributors:

- `docs/design/` — internal architecture decision records and design notes
- `docs/superpowers/` — specs, plans, and review notes accumulated during doc development
- `docs/naming-convention.md` — internal naming conventions for code review
- `docs/pytest-best-practices.md` — internal testing style guide

If you add new internal-only content, either place it under one of these excluded paths or add a new entry to `exclude_docs` in `mkdocs.yml`.

## See also

- [MkDocs Material reference](https://squidfunk.github.io/mkdocs-material/reference/)
- [MkDocs Material navigation setup](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/)
- [MkDocs Material color palette](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/)
