# Issues: docs/samples/qwen3-composite.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- (none)

## Important

- **GitHub URL points to `https://github.com/microsoft/winml-cli` (plain repo root).**
  The "Track progress" section says "Follow development and check current status
  at https://github.com/microsoft/winml-cli". There is no issues link, milestone
  link, or branch link. For a placeholder page whose purpose is to track an
  in-progress feature, the URL is minimal but not wrong. However, if the
  feature is tracked on a specific branch or issue, the link should be more
  precise. Acceptable as-is for a placeholder.

- **Forward-looking sketch references `BuildConfig`** (capitalised as proper
  noun) without tying it to `WinMLBuildConfig`. Readers coming from the BERT
  sample know the class name; first-time readers may not. Minor wording issue.

## Minor

- **`!!! info "Coming soon"` admonition** — correctly identifies the page as a
  placeholder. Format is valid MkDocs Material admonition syntax.

- **Cross-link `../samples/bert-config-build.md`** — from inside `docs/samples/`,
  a self-referential path resolves to `docs/samples/bert-config-build.md`.
  The `../samples/` prefix from within `samples/` is redundant (resolves one
  directory up and back in) but should still resolve correctly in MkDocs.
  Could be simplified to `bert-config-build.md`.

## Verified correct

- Page correctly identifies itself as a placeholder and defers all content to
  after the composite-model feature branch lands.
- No commands, flags, or artifact names are asserted (none to verify wrong).
- GitHub URL `https://github.com/microsoft/winml-cli` is the correct upstream
  repository URL.
- No `wmk` or `ModelKit` strings found in user-facing prose.
- "What composite models are" section contains only conceptual prose — no
  verifiable command syntax.
