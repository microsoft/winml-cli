# Issues: docs/concepts/perf-and-monitoring.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- Line 11: `--device` is described as accepting `cpu`, `gpu`, or `npu` only, but `perf` calls `cli_utils.device_option(include_auto=True, default="auto")` (perf.py:1113), so `auto` is also a valid choice and is the actual default. The sentence "The `--device` flag selects the target EP — `cpu`, `gpu`, or `npu`" omits `auto` and misstates the default.
- Line 13: Output path default stated as `{model_slug}_perf.json` (implying the current directory). Source writes to `~/.cache/winml/perf/<slug>/<timestamp>.json` (perf.py:871–876). The default location is wrong and the timestamp-per-run filename structure is omitted entirely.

## Important

- Lines 25–31: `--op-tracing` is documented as a user-facing feature with two levels. In source the option is decorated `hidden=True` (perf.py:1183), meaning it is intentionally hidden from `--help`. Documenting a hidden flag as a supported feature is misleading.
- Lines 17–21: `--monitor` is described as streaming "NPU utilisation". Source tracks whichever device is being benchmarked: NPU, GPU, or CPU (`monitor_device = self._model.device or self.config.device or "auto"`, perf.py:409). Calling it NPU-specific is inaccurate.

## Minor

- Line 19: States the chart "updates in place during the iteration loop". The live chart is managed by `LiveMonitorDisplay` (perf.py:943), but this detail is accurate. No issue.
- Line 37: `--module` docstring in source says the argument is a "PyTorch module class name (NOT a dotted module path)" (perf.py:1166–1169). The concept doc example `winml perf -m bert-base-uncased --module BertAttention` is correct, but the doc does not warn users that a dotted path will silently not match, which is the primary pitfall documented in the source help text.
