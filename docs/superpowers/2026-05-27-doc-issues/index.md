# Issues: docs/index.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

(none)

## Important

- **Anchor `#eps-winml-cli-supports` may not resolve.** The link `concepts/eps-and-devices.md#eps-winml-cli-supports` targets a heading "EPs winml-cli supports" (line 7 of that file). MkDocs lowercases and hyphenates heading text, so "EPs winml-cli supports" becomes `#eps-winml-cli-supports`. The "EPs" acronym normalizes correctly here — the anchor is valid as written, but this depends on MkDocs slug behaviour for acronyms (capitals are lowercased). Treat as worth verifying in the rendered site.

## Minor

- **"12 `winml` subcommands"** — the `docs/commands/` directory contains 12 `.md` files (analyze, build, compile, config, eval, export, hub, inspect, optimize, overview, perf, quantize, sys). `overview.md` is a landing page, not a subcommand. The actual executable subcommands registered in the CLI should be counted and verified; if hub or overview are not registered commands the "12" claim would be wrong.

## Verified correct

- No `wmk` or `ModelKit` strings in user-facing prose.
- GitHub URL `https://github.com/microsoft/winml-cli` matches `pyproject.toml` URLs.
- Links to `getting-started/installation.md`, `getting-started/quickstart.md`, `getting-started/end-to-end.md`, `concepts/how-it-works.md`, `commands/overview.md` all resolve to files that exist.
- Link to `samples/convnext-primitives.md` resolves.
- MIT licence link points to `https://github.com/microsoft/winml-cli/blob/main/LICENSE.txt`.
- Tagline and bullets read naturally with no leftover `wmk`/`ModelKit` names.
