# Issues: docs/tutorials/index.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical

- (none)

## Important

- (none)

## Minor

- **Backtick usage inconsistency.** The prose uses `` `winml-cli` `` (with
  backtick) in one sentence but refers to `winml` (the CLI binary name) without
  backtick elsewhere. This is cosmetic only.

## Verified correct

- Table entry `[ConvNeXt on NPU](npu-convnext.md)` — file exists at
  `docs/tutorials/npu-convnext.md`. Link resolves.
- "Hardware" column entry "Copilot+PC NPU primary; CPU works as fallback" —
  consistent with npu-convnext.md content.
- No command invocations to verify.
- No `wmk` or `ModelKit` strings in user-facing prose.
- Page correctly describes tutorials vs samples vs concepts distinctions.
