# Export Device Policy Design

## Goal

Add a generic export-time compatibility policy mechanism that can choose model
configuration overrides per EP/device target. The immediate requirement is to
avoid SDPA-generated ONNX attention guard paths for QNN, because QNN does not
reliably support the resulting `Softmax -> IsNaN -> Where -> MatMul` pattern.

The mechanism must be architecture-agnostic. It must not hardcode model names,
node names, tensor names, or layer naming patterns. It should express target
capabilities at the EP/device level so future support changes can be made by
removing or narrowing policy rules.

## Scope

In scope:

- Add a pure-data export compatibility policy layer.
- Resolve policy from explicit EP/device targets or from the full WinML
  supported EP/device catalog when no target is specified.
- Record the resolved policy in `WinMLExportConfig` so config files and cache
  keys reflect the exported model behavior.
- Apply the resolved policy in the exporter without exposing QNN-specific logic
  in `HTPExporter`.
- Use the policy to force Hugging Face-style attention configs to eager only
  when required by the resolved target policy.

Out of scope:

- Changing runtime EP selection or compile/quant target resolution.
- Rewriting ONNX graphs after export.
- Adding model-specific recipes or model-specific export branches.
- Disabling SDPA globally for all targets.

## Architecture

Introduce a new export compatibility policy module:
`winml.modelkit.export.policy`.

The module owns:

- `ExportCompatibilityConfig`: a serializable dataclass that records resolved
  export compatibility requirements. Initially it contains
  `transformers_attention: Literal["eager"] | None`.
- A small policy registry where each rule maps one EP/device condition to one
  or more export requirements.
- A resolver that takes zero or more EP/device targets and returns a merged
  `ExportCompatibilityConfig`.

`WinMLExportConfig` gains a `compatibility` field of type
`ExportCompatibilityConfig`. The field is included in `to_dict()` and
`from_dict()` and therefore participates in `WinMLBuildConfig.generate_cache_key()`.
This prevents reusing an ONNX artifact exported under a different compatibility
policy.

`HTPExporter` remains target-agnostic. It reads `export_config.compatibility`
and applies the corresponding context managers. For the initial rule, it calls
the existing Hugging Face-style eager attention context manager only when
`transformers_attention == "eager"`.

## Target Resolution Semantics

The policy resolver distinguishes export compatibility target selection from
runtime compile/quant target selection.

When a caller provides an explicit EP/device pair, the resolver applies only
rules for that concrete pair. If the caller provides only one axis, existing
device resolution first produces a concrete pair, then the policy resolver uses
that pair.

When no EP/device target is specified, the resolver uses the full
`EP_DEVICE_SPECS` catalog of WinML-supported EP/device pairs. It merges the
requirements from every catalog target to produce a portable export policy. This
is intentionally more conservative than current runtime auto-resolution and does
not depend on which EPs are installed on the local machine.

Existing compile and quantization policy behavior stays unchanged. The new
portable intersection applies only to export-time compatibility.

## Initial Policy Rule

The initial policy rule is:

- Target: `QNNExecutionProvider` on supported QNN devices.
- Requirement: `transformers_attention = "eager"`.
- Reason: transformers 5 automatically resolves unspecified attention to SDPA
  for models that support SDPA. SDPA export can introduce NaN-guard attention
  paths containing `IsNaN` and `Where`; QNN does not reliably support that
  pattern.

The rule is EP/device-level, not model-level. It does not mention CLIP, BERT,
T5, or any architecture family. If future QNN releases support the SDPA export
pattern, this rule can be removed or narrowed without changing exporter logic.

## Data Flow

For auto-generated HF build configs:

1. Resolve loader and export I/O config as today.
2. Resolve the export compatibility policy from the requested target context.
3. Store the resolved policy on `config.export.compatibility`.
4. Persist and cache using the normal build config serialization path.
5. During export, `HTPExporter` applies compatibility context managers from
   `export_config.compatibility`.

For config-file builds:

- If the config includes `export.compatibility`, use it as serialized.
- If the config omits `export.compatibility`, populate it using the same target
  policy resolution that auto-generated configs use, so older config files gain
  safe defaults.

For pre-exported ONNX builds:

- No export happens, so export compatibility policy is not applied.

## Error Handling

The resolver must be deterministic and fail loudly on incompatible policy
requirements. If future rules assign conflicting values to the same knob for a
merged target set, the resolver raises a clear `ValueError` describing the
conflicting knob and target rules instead of silently choosing one.

Unknown or unset compatibility values are no-ops in the exporter. They must not
trigger broad exception handling or silent fallbacks.

The Hugging Face attention override remains scoped to the export context and
restores every modified config after export, including nested module configs.

## Testing

Unit tests should cover:

1. `ExportCompatibilityConfig` serialization and deserialization through
   `WinMLExportConfig`.
2. `WinMLBuildConfig.generate_cache_key()` changes when export compatibility
   changes.
3. Explicit QNN EP/device targets resolve to `transformers_attention="eager"`.
4. No target resolves against the full supported EP/device catalog and therefore
   includes the QNN eager-attention requirement.
5. Explicit non-QNN targets do not force eager attention unless a future rule
   says they should.
6. `HTPExporter` applies eager attention only when the resolved compatibility
   config requests it, and restores model configs after export.

Local QNN E2E validation for CLIP, multilingual BERT, and T5 should be run
after implementation, but those E2E cases should not become unit-test
dependencies.
