# Ship Quality Plan

## Functionality (Driver: Zhipeng)

- **CLI Consistency** — Unify parameter naming, output format, and error message style across all commands; produce a CLI convention doc as the alignment standard.
- **E2E per Device** — Independently validate P0 models for each EP x device combination (QNN/NPU, OV/NPU, VitisAI/NPU, CPU), with focus on cross-machine environment and compatibility differences.
- **User Experience Review** — Walk through happy path and unhappy path for every command; book meeting to go through each one.
  - Long-running operations must show progress feedback so users don't think the tool is hanging.
  - Error messages in failure scenarios must be user-friendly and actionable, not raw stack traces.
- **Functional Correctness** — Verify logical correctness on a single-machine basis:
  - Behavior matches spec: each command's core functionality works as designed, output matches intent.
  - Output consistency: all outputs (log, JSON, config, report) accurately reflect user input and actual execution state for device/EP/precision fields.
  - Interruption safety: Ctrl+C leaves no corrupted files, leftover temp files, or inconsistent state.
  - Idempotency: same input + same environment produces identical results on repeated execution.
- **Bug Bash & Triage** — After the above items are substantially complete, organize a team-wide bug bash, categorize all known failures, and drive P0 bugs to zero.

## Performance & Memory (Driver: TBD)

- **Quick Command Responsiveness** — `--help`, `sys`, `inspect` and other commands unrelated to model size must maintain fast cold-start response times.
- **Per-Component Benchmark** — Define latency and peak memory targets per component; measure across small/medium/large models:
  - Static analyzer (`winml analyze`) measured independently.
  - Per-EP x device pipeline stages (export, optimize, quantize, compile) with per-component timing and memory data.
- **Regression Protection** — Key metrics integrated into CI or periodic benchmarks to prevent performance regressions from subsequent PRs.
- **Package / Data Size** — Track total package size, especially static analyzer data growth as more EPs are onboarded.

## EP Data Quality (Driver: TBD)

- **EP Data Accuracy** — Operator support data for each EP must be consistent with actual runtime behavior; no false positives or false negatives.
- **Coverage Targets** — Measured per EP x model: P0 & built-in models >= 90%, Top 200 models >= 80%.
- **Multi-Format Coverage** — Validate data coverage across non-quantized, different opset versions, QDQ, and compiled format combinations.
