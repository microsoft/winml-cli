# GPT Helper Guide: Adding Patterns to available_ops_all.json

This file is a compact operational guide for GPT-style assistants.

Goal:

- Add or update entries in `patterns` inside `available_ops_all.json` correctly.
- Keep `case_index` encoding/decoding stable.

## Scope and Source of Truth

- `available_ops_all.json`
  - Version sets and 2-char `version_key_map` for ops/patterns.
- `avaliable_providers.json`
  - 1-char key for `(EP, device)`.
- `case_index_key_codec.py`
  - Key validation and file name <-> 4-char prefix codec.
- `../op_utils.py`
  - Case signature and final 36-char `case_index` generation.

## Required Pattern Version Rule

For a pattern with ops `op_1 ... op_n`, let `V(op_i)` be the version set from op maps.

1. `start_version = max(min(V(op_1)), min(V(op_2)), ..., min(V(op_n)))`
2. `pattern_versions = sorted(unique(union(v in V(op_i) where v >= start_version)))`
3. Pattern `version_key_map` keys must exactly match `pattern_versions`.

Example (`MatMulAddPattern`):

- `V(MatMul) = {9, 13}`
- `V(Add) = {7, 13, 14}`
- `start_version = max(9, 7) = 9`
- `pattern_versions = {9, 13, 14}`

## Domain Rule for Ops

When resolving op versions for a pattern:

- Prefer unambiguous lookup from one domain.
- If an op name exists in both domains (`ops_ai_onnx` and `ops_com_microsoft`), select domain intentionally.
- Do not mix domains for the same op without explicit intent.

## Key Allocation Constraints

When adding new `version_key_map` entries:

- Version key must be exactly 2 chars.
- Allowed alphabet: `123456789abcdefghijklmnopqrstuvwxyz`.
- Character `0` is not allowed.
- 2-char version keys must be globally unique across:
  - `ops_ai_onnx`
  - `ops_com_microsoft`
  - `patterns`
  - `ops` (if present)

## case_index Contract

Final format:

- `case_index = <4-char namespace key> + <32-char md5(signature)>`
- Total length is always 36.

4-char namespace key format:

- 1 char: EP/device key (from `avaliable_providers.json`)
- 2 chars: version key (from `available_ops_all.json`)
- 1 char: qdq flag (`0` non-QDQ, `1` QDQ)

## File Name Contract for Codec

The codec expects this stem shape:

`<name>_<ep>_<device>_<domain>_opset<version>[_qdq]`

Supported device tokens: `CPU`, `GPU`, `NPU`.

## Rules-First Behavior in check_patterns

`check_patterns` first queries node-level parquet rules.

- If all nodes are supported and have data, it writes synthetic success and skips real EP run.
- If any node is unsupported/no_data/error, it falls back to real compile/run.

`check_ops` does not use this prefilter path.

## Minimal Update Workflow (for GPT)

1. Identify target pattern and op list.
2. Resolve each op version set from correct domain.
3. Compute `start_version` and `pattern_versions` using the required rule.
4. Allocate new globally unique 2-char keys for those versions.
5. Update `patterns.<PatternName>` in `available_ops_all.json`.
6. Validate JSON parse.
7. Validate no duplicate 2-char keys.
8. Verify codec round-trip for at least one generated file name.

## Quick Validation Checklist

- JSON is valid.
- Pattern version keys follow rule exactly.
- New 2-char keys are unique and alphabet-compliant.
- No accidental edits to unrelated patterns.
- `encode_file_name_to_4char_key` and `decode_4char_key_to_folder_and_file_name` work for the new entry.
