# src/winml/modelkit/analyze/pattern/check_patterns.py

## TL;DR

Two cleanups:

1. **Removed module-level EP registration side-effect.** The `from ... import winml` import and `winml.register_execution_providers(ort=True)` top-level call at module load are deleted. Registration is now driven from inside `EPChecker._get_sess_options()` via the new session-catalog path, so this subprocess tool no longer needs to side-effect-register EPs at import time.
2. **Added carve-out comment on `--ep` allowlist.** A 4-line comment block inside the `argparse` `--ep` definition documents why the allowlist is hand-picked (`QNNExecutionProvider`, `OpenVINOExecutionProvider`) and explicitly tells future maintainers NOT to derive from `eps_for_device("npu")` or `EP_DEVICE_SPECS`.

Net effect: the parser allowlist is unchanged, but the import-time global registration is gone. Behavior at runtime is unchanged because `EPChecker` (now updated) takes care of its own registration via the session catalog.

## Diff metrics

- Lines changed: +4 / -4 (8 total)
- Removed import: `from ... import winml`
- Removed module-level call: `winml.register_execution_providers(ort=True)` (and its blank-line padding)
- Added: 4 comment lines inside the `--ep` argparse definition
- Touched functions: `build_parser` (parser definition only) and the module top
- No code path changed; argparse `choices=` list unchanged

## Role before vs after

Before: this file was a subprocess CLI entrypoint (`check_patterns`) that performed eager EP registration at module import. Any consumer doing `python -m winml.modelkit.analyze.pattern.check_patterns` triggered `winml.register_execution_providers(ort=True)` as a side effect before main ever ran.

After: the file is still the subprocess CLI entrypoint, but EP registration is now lazy and lives inside `EPChecker._get_sess_options()` (see `analyze__runtime_checker__ep_checker.md`). Import this module without intent to use `EPChecker` and you no longer pay the registration cost. The `--ep` allowlist is now self-documenting via the new carve-out comment.

## Symbol-level changes

### Removed module-level statements

```python
from ... import winml
...
winml.register_execution_providers(ort=True)
```

Both deleted. The companion `runtime_checker/check_ops.py` got the same surgery in this commit.

### Unchanged

- `from ..runtime_checker.ep_checker import EPChecker` import is preserved.
- `check_patterns(...)` body is unchanged.
- `build_parser()`'s `--ep` argument: `required=True`, `choices=["QNNExecutionProvider", "OpenVINOExecutionProvider"]`, and `help=` text are unchanged.

### Added in `build_parser()`

```python
# CARVE-OUT: This subprocess tool intentionally supports only a curated subset of
# NPU EPs. VitisAI and future NPU EPs are excluded because this pattern-checking
# tool has not been validated against them. Do NOT derive from eps_for_device("npu")
# or EP_DEVICE_SPECS — this is an explicit opt-in allowlist, not catalog drift.
```

This comment names two session-catalog APIs (`eps_for_device("npu")`, `EP_DEVICE_SPECS`) explicitly so a future contributor doing a catalog-derived refactor sees the deliberate non-migration.

## Behavior / contract changes

1. **No more import-time EP registration.** The module is now safely importable by tools that just want `check_patterns` as a function, without side-effects.
2. **Allowlist behavior is unchanged.** The CLI still accepts exactly `QNNExecutionProvider` and `OpenVINOExecutionProvider`.
3. **Subprocess invocation behavior is unchanged at runtime.** `EPChecker._get_sess_options()` now does the registration lazily on first use, so the CLI works as before.

## Cross-file impact

- **`analyze/runtime_checker/ep_checker.py`** absorbed the registration responsibility. The deletion here is safe **only because** `ep_checker.py` was updated in lockstep — without that change, this file's deletion would break runtime EP availability.
- **Companion file `analyze/runtime_checker/check_ops.py`** got the same module-level cleanup.
- **Carve-out comment** points future maintainers at the session catalog (`eps_for_device`, `EP_DEVICE_SPECS`) — those symbols must exist in `winml.modelkit.session` for the comment to remain meaningful. Confirmed exported in `session/__init__.py`.

## Risks / subtleties

1. **Comment as contract, not enforcement.** As noted in the a509a67 version of this file, a mechanical refactor that swaps the literal list for `sorted(eps_for_device("npu"))` would slip past this comment. The carve-out lives in prose only; no test asserts the allowlist.
2. **Drift vs `check_ops.py`.** The sibling tool still has a 5-EP allowlist (QNN, OpenVINO, VitisAI, MIGraphX, NvTensorRtRtx). The carve-out rationale ("untested against VitisAI / future NPU EPs") is consistent with pattern-checking being stricter than op-checking, but the asymmetry is invisible without reading both files.
3. **Registration timing has shifted.** Anyone who relied on the `winml.register_execution_providers(ort=True)` side-effect to register EPs for **other** code (e.g. a `pyinstaller` bundle that depends on import-time setup) would silently lose that registration. Probability of this footgun is low — the subprocess tool is purpose-built — but worth noting.

## Open questions / TODOs surfaced

- Encode the carve-out structurally? An `EPDeviceSpec.pattern_tested: bool` attribute would let the CLI filter on catalog state instead of a literal list. Until then, the comment is fragile.
- The comment cites `eps_for_device("npu")` and `EP_DEVICE_SPECS` by name — a forward reference. If those symbols are ever renamed, this comment silently goes stale.

## Simplification opportunities

- The carve-out is currently four lines of prose. A single one-line constant — `_PATTERN_TESTED_EPS: Final[tuple[str, ...]] = ("QNNExecutionProvider", "OpenVINOExecutionProvider")` — referenced from both `choices=` and a test fixture would make the rationale executable and prevent the comment from going stale.
- Both `check_patterns.py` and `check_ops.py` got the same module-level registration removal. If the subprocess tools have any other shared boilerplate (imports, parser scaffolding), a single shared helper module would reduce drift.
