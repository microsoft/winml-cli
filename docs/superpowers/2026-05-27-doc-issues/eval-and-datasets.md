# Issues: docs/concepts/eval-and-datasets.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- (none)

## Important

- Lines 1–7: The concept doc lists no `--ep`, `--precision`, `--dataset-script`, or `--trust-remote-code` flags, all of which exist in eval.py (lines with `@cli_utils.ep_option`, `--precision`, `--dataset-script`, `--trust-remote-code`). While a concept page need not enumerate every flag, omitting `--precision` is notable because the page is about post-quantization accuracy checks and `--precision` directly affects which model artifact is built.
- Line 25 / `--samples` default: The concept doc does not state a default for `--samples`, but line 34 of docs/commands/eval.md lists the default as `100`. Source confirms `default=100` (eval.py). This is consistent, but the concept page example at line 35 uses `--samples 200` without noting the default, which is fine — no defect here on its own.

## Minor

- Line 22: States `--output` "accepts any `.json` path; if omitted, results are printed but not persisted." Source confirms this (no default for `output_path`). Accurate.
- Line 35: `--streaming` flag description says it "fetches rows on demand instead of materialising the whole dataset locally." Source confirms `is_flag=True, default=False`. Accurate.
- Line 38: `--column key=value` usage is consistent with source (`multiple=True`, key=value parsing in eval.py). Accurate.
