# src/winml/modelkit/build/onnx.py

## TL;DR
Single-line modernization: `datetime.timezone.utc` is replaced with the Python-3.11+ alias `datetime.UTC` on the build manifest's `timestamp` field. Pure cosmetic / style; identical runtime semantics. No EP / session refactor work here — the file was swept in only because the squash touched every file with stale `datetime.timezone.utc` callsites.

## Diff metrics
- 1 line changed (1 removed, 1 added), inside the build-manifest `dict` literal at the end of `build_onnx_model`.
- No imports, no symbols, no contracts altered.

## Role before vs after
- **Before:** Wrote the build manifest timestamp as `datetime.datetime.now(datetime.timezone.utc).isoformat()`.
- **After:** Same but using the shorter `datetime.UTC` alias. The manifest `"timestamp"` value is byte-identical (`datetime.UTC is datetime.timezone.utc` per CPython 3.11+).

## Symbol-level changes
- No new / removed / renamed symbols.
- Sole mutation: line 267 inside `build_onnx_model`'s `manifest` dict literal — `datetime.timezone.utc` → `datetime.UTC`.

## Behavior / contract changes
- None. `datetime.UTC` was added in Python 3.11 as an alias for `datetime.timezone.utc`; both are the same singleton instance. Any consumer parsing the manifest JSON sees the same ISO-8601 string with the same `+00:00` suffix.
- **Minimum Python version implicitly bumped to 3.11** (which is already the project floor — `pyproject.toml` ships with this constraint).

## Cross-file impact
- None — the manifest JSON schema (`schema_version: 1`) is unchanged.
- Sibling sweep: only this one site in this file used `datetime.timezone.utc`; no symbol re-exports or callers to update.

## Risks / subtleties
- None observable. If the project ever needed to support Python <3.11 again, this would be a regression — but `pyproject.toml` already pins ≥3.11, and the new style is preferred per CPython docs.

## Open questions / TODOs surfaced
- The same idiom (`datetime.datetime.now(datetime.timezone.utc)`) may still exist elsewhere in the repo; an opportunistic ruff rule (`UP017`) would catch any remaining stragglers in one pass.

## Simplification opportunities
- None for this file; the change *is* the simplification. The bigger structural cleanup would be to factor manifest construction (`schema_version` / `timestamp` / `elapsed_seconds` / `stages` / `final_artifact`) into a helper if other build entry points (`build/hf.py`) replicate the dict shape, but that is orthogonal to this commit's scope.
