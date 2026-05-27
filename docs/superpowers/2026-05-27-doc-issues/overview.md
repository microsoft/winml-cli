# Issues: docs/commands/overview.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- Line 2: States "12 subcommands". Source has 14 command modules (`analyze`, `build`, `catalog`, `compile`, `config`, `eval`, `export`, `inspect`, `optimize`, `perf`, `quantize`, `run`, `serve`, `sys`). `run` and `serve` are disabled at runtime via `_DISABLED_COMMANDS` (cli.py) but the command count is still wrong at 12 — the actual exposed count is 12 only if the two disabled commands are excluded AND `catalog` is counted as `hub`. The command map (line 29) lists `hub` which does not exist; the actual command is `catalog` (catalog.py, `@click.command()` function named `catalog`). There is no `hub` command in the codebase at this commit.
- Line 29 (table row): `hub` command listed as "Browse the curated winml-cli catalog of validated models and benchmarks." The command is named `catalog`, not `hub` (catalog.py). `winml hub` would fail at the CLI.

## Important

- Line 55: References `src/winml/modelkit/commands/_options.py` as the "canonical contract" for global flags. This file does not exist at commit 5e25579 (verified via `git ls-tree`). Global flags are defined in `src/winml/modelkit/cli.py` directly.
- Lines 41–48 ("Choosing a command"): The entry "I want to know if my model is supported → `winml inspect`" is reasonable, but `winml analyze` (Verify EP operator compatibility) is a closer match for pre-deployment compatibility checks. The distinction between `inspect` and `analyze` is not reflected in the choosing-a-command list, making `analyze` effectively undiscoverable from this guide.

## Minor

- Line 63: Shared flags claim "`-p` / `--precision`" is shared. `perf` and `eval` both have `--precision` but `inspect`, `sys`, `hub`/`catalog`, and `analyze` do not. The claim "Defaults and accepted values can differ per command" partially covers this, but listing `-p` as a shared flag implies it exists on most commands, which overstates its reach.
