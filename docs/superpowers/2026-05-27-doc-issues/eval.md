# Issues: docs/commands/eval.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- Line 24: `--device` type column shows `cpu|gpu|npu` with default `cpu`. Source defines `type=click.Choice(["auto", "cpu", "gpu", "npu"])` with `default="auto"` (eval.py). The `auto` choice is missing and the default is wrong.
- Line 25: `-n` is listed as a short alias for `--samples`. Source defines `--samples` with no short flag (eval.py `@click.option("--samples", type=int, default=100, ...)`). The `-n` alias does not exist.

## Important

- Flags table is missing the following options that exist in source (eval.py):
  - `--ep` — execution provider override (`@cli_utils.ep_option`)
  - `--precision` — precision mode (`--precision`, default `auto`)
  - `--dataset-script` — path to a dataset-building script
  - `--trust-remote-code` — required flag when `--dataset-script` is used
  - `--verbose` / `-v` — verbose output flag
- Line 36: "How it works" section says `winml eval` loads the model via `WinMLAutoModel`. Source uses `WinMLEvaluationConfig` and calls `evaluate(cfg)` from the `eval` subpackage (eval.py). The class name `WinMLAutoModel` does not appear in eval.py; the description misrepresents the implementation.

## Minor

- Line 19: `--model` description says "Required (unless `--model-id` is provided directly)." Source actually raises `UsageError` if neither `-m` nor `--model-id` resolves a model, and `--model-id` alone (without `-m`) is accepted only to supply a HuggingFace ID. This nuance is slightly misleading but not a breaking inaccuracy.
- Line 88: Pitfall note "`--streaming` skips the local cache." Source confirms this behaviour. Accurate.
