# Pending Review Issues — `feat/update-pkg-deps`

Status: 21 verified issues from 4-agent review + Opus-level verifier pass on 2026-05-08.
Format: `ID | Severity | File:Line | Status | Fix sketch`.

Update `Status` to `DONE`/`SKIPPED` as work lands. Do not delete entries — keep history.

---

## Critical (must-fix)

### C-1 | Critical | `src/winml/modelkit/ep_path.py:839-844` | DONE
**Issue**: `list_msix_eps` Returns docstring claims "set to the exact PackageFamilyName plus a trailing `_` for round-trip pinning" but actual impl drops the trailing `_` (line 903 emits `family_name_prefix=str(p.id.family_name)` with no separator).
**Fix**: Rewrite Returns paragraph: "set to the exact PackageFamilyName (no trailing separator) and `version` to the exact installed `Package.Id.Version` string. Round-trip exactness comes from the family-name + version pin together."

### C-2 | Important (verifier-downgraded) | `docs/ep-path-msix-source.md:10, 407, 419, 552` | DONE
**Issue**: Doc inconsistently uses `[active]` (rejected) and `[primary]` (shipped) tags.
**Fix**: Search-replace `[active]` → `[primary]` throughout the doc. Verify via `grep -n '\[active\]' docs/ep-path-msix-source.md` returns nothing after.

### C-3 | Critical | `docs/ep-path-msix-source.md:425-427` | DONE
**Issue**: Doc example uses `discover_eps(extra_sources=discovered_msix, return_shadowed=True)` — but the actual CLI uses `extra_sources_after=msix`. With `extra_sources=`, MSIX entries become `[primary]` and shadow PyPI/Catalog (semantic flip from `[shadowed]`).
**Fix**: Update example to `extra_sources_after=discovered_msix`. Add a sentence noting why: "appended (not prepended) so MSIX entries appear as `[shadowed]` rather than artificially overriding default precedence."

### C-4 | Critical | `tests/unit/ep_path/test_discover_eps_shadowed.py` | DONE
**Issue**: `discover_eps(extra_sources_after=...)` parameter has zero tests. Every existing test uses `extra_sources=`. The actual CLI code path is uncovered.
**Fix**: Add 3 tests:
- `test_extra_sources_after_appears_after_ep_path` — assert order: EP_PATH primary, extra_sources_after shadowed.
- `test_extra_sources_after_does_not_promote_to_primary` — when EP_PATH provides EP, extra_sources_after entries are shadowed.
- `test_extra_sources_after_alone_yields_primary` — when EP_PATH is empty, extra_sources_after's first entry becomes primary.

### I-3 (CLAUDE.md violation upgrade) | Critical | `src/winml/modelkit/commands/sys.py:486-494` | DONE
**Issue**: `_describe_source` swallows `metadata.version()` errors with bare `except Exception:` and no log. PRINCIPLES "Never Suppress Silently" violated.
**Fix**: Narrow to `except metadata.PackageNotFoundError:`, add `logger.debug("metadata.version(%r) failed: %s", source.distribution, e)`, set `distribution_version = None`.

---

## Important (should-fix)

### I-1 | Important | `src/winml/modelkit/commands/sys.py:582-583` | DONE
**Issue**: ORT import failure logged at DEBUG during `--list-ep` invocation. User explicitly requested EP inventory; should be WARN.
**Fix**: Bump `logger.debug("ORT not available: %s", e)` to `logger.warning("onnxruntime import failed during --list-ep; built-in EPs (CPU/Azure/Dml) will not be listed: %s", e)`.

### I-2 + I-4 (compound) | Important | `src/winml/modelkit/commands/sys.py:543, 835-836` | DONE
**Issue**: `_gather_ep_info`'s per-EP `is_compatible()` is unguarded; a single WMI flake breaks the whole `--list-ep`. Combined with the outer DEBUG-swallowed exception in default `winml sys`, the EP panel silently vanishes.
**Fix**:
- Wrap `entries[0].source.is_compatible()` at sys.py:543 in try/except defaulting to True; log WARN with EP name.
- Bump outer `_gather_ep_info()` exception in default sys path to WARN with a banner: "EP detection failed; re-run with `--verbose` for details."

### I-5 | Important | `tests/unit/commands/test_cli.py:167` | DONE
**Issue**: `_gather_ep_info()` is mocked, never exercised end-to-end. The compatible-derivation, is_catalog_default cross-reference, incompatible status override, and built-in fallback paths are all uncovered.
**Fix**: Add a test in `test_cli.py` (or new `test_sys_list_ep_e2e.py`) that mocks ONLY `_get_pkg_manager` and `_get_catalog` (the slow parts), runs the real `_gather_ep_info`, and asserts on the resulting dict shape (compatible bool, entries list, status values, is_catalog_default tag).

### I-6 | Important | `src/winml/modelkit/ep_path.py:803`, `tests/unit/ep_path/test_msix_package_source.py:103` | DONE
**Issue**: `MsixPackageSource.relative_dll` POSIX-style invariant unenforced. `Path / "foo/bar"` tolerates both separators on Windows but the test masks the issue with `.replace("/", "\\")`.
**Fix**:
- Add docstring contract assertion in `MsixPackageSource.__post_init__` (or `resolve()`): if `"\\"` in relative_dll, raise ValueError.
- Remove `.replace("/", "\\")` from test:103; rely on `Path /` joining to handle both POSIX/Windows.

### S-10 (verifier-promoted) | Important | `src/winml/modelkit/ep_path.py:1002-1006` | DONE
**Issue**: Linux comment says "we still list the source — it just yields nothing on x86_64 Linux" but `_default_ep_path_linux()` only contains OpenVINO. QNN on Linux aarch64 is no longer discoverable via default `EP_PATH` — functional regression.
**Fix**: Add the QNN PyPiSource entry to `_default_ep_path_linux()`:
```python
PyPiSource(
    distribution="onnxruntime-qnn",
    relative_dll="onnxruntime_qnn/libs/{arch}/libonnxruntime_providers_qnn.so",
    eps=("QNNExecutionProvider",),
    arch_resolver=_qnn_arch_resolver,
),
```
Verify the .so layout in onnxruntime-qnn 2.1.x for aarch64 — if different, update `relative_dll` accordingly. Or remove the misleading comment if QNN-on-Linux is genuinely unsupported.

---

## Suggestions (nice-to-have, batch in follow-up)

### S-1 | Suggestion | `src/winml/modelkit/commands/sys.py:481-507` | DONE
**Issue**: `_describe_source` uses `hasattr` dispatch (fragile if attrs collide).
**Fix**: Replace with `isinstance` checks against PyPiSource/MsixPackageSource/WinMlCatalogSource/FilesystemSource.

### S-2 | Partial | `src/winml/modelkit/winml.py:13-16` | DONE
**Issue**: `_legacy_ep_plugin_registry` shim docstring says "removed once no internal callers remain" (future-tense — verifier corrected).
**Fix**: Verify with `grep -rn EP_PLUGIN_REGISTRY src/ tests/` whether internal callers exist. If none, delete the shim. If some remain, leave the docstring alone.

### S-3 | Partial | `src/winml/modelkit/ep_path.py:1050` | DONE
**Issue**: `";" if os.name == "nt" else os.pathsep` is functionally redundant; `os.pathsep` already returns `;` on Windows.
**Fix**: Replace with bare `os.pathsep`. Add a comment if the original author had a defensive intent.

### S-4 | Suggestion | `tests/unit/ep_path/test_compat.py` | DONE
**Issue**: `EpSource.iter_eps()` abstract method has no direct test.
**Fix**: Add a test class `TestIterEps` with one case per EpSource subclass asserting `iter_eps()` returns the expected canonical EP names.

### S-5 | Suggestion | `tests/integration/` | DONE
**Issue**: No live-Windows integration test for `list_msix_eps()` / `_get_pkg_manager` / `_get_catalog`.
**Fix**: Add `tests/integration/ep_path/test_live_msix.py` with `@pytest.mark.skipif(os.name != "nt")` calling `list_msix_eps()` and asserting the return is a list (length depends on machine).

### S-6 | Suggestion | `src/winml/modelkit/ep_path.py:227` | DONE
**Issue**: `_qnn_arch_resolver` ARM64 branch is host-dependent (won't exercise on x86_64 CI).
**Fix**: In `tests/unit/ep_path/test_ep_path.py::TestQnnArchResolver`, add a parametrized test that monkeypatches `platform.machine` to force both `arm64` and `x86_64` paths.

### S-7 | Suggestion | `pyproject.toml:142-148` | DONE
**Issue**: "neutralize the upstream transitive dep" wording could be misread.
**Fix**: Reword: "neutralize the upstream **transitive** `onnxruntime` requirement (we explicitly want `onnxruntime-windowsml` only)".

### S-8 | Suggestion | `src/winml/modelkit/commands/sys.py:533-538` | DONE
**Issue**: `type(...).__name__ == "WinMlCatalogSource"` string compare is fragile.
**Fix**: Replace with `isinstance(entry.source, WinMlCatalogSource)`. Import WinMlCatalogSource at module top (no cycle since `commands/sys.py` is downstream).

### S-9 | Suggestion | `docs/ep-path-design.md:3`, `docs/ep-path-msix-source.md:3` | DONE
**Issue**: Status header says "design draft" but impl has shipped.
**Fix**: Bump status: "implemented in commits 17b81c9a..f79e484e, as of 2026-05-08."

---

## Verifier-added findings (already merged into above)

- **S-10 functional bug**: covered above as Important (was Suggestion).
- **C-3 semantic flip**: covered above (stays Critical).
- **I-2 + I-4 compound**: covered above (folded into one fix).
- **I-3 CLAUDE.md violation**: covered above (upgraded to Critical).
- **S-3 defensive rationale**: covered above (acknowledge in fix-sketch).

---

## Remediation order

1. C-1, C-3 — pure docstring/doc fixes (zero risk)
2. C-2 — doc search-replace
3. I-3 — CLAUDE.md violation; narrow exception, add log
4. I-1 — log level bump
5. I-2 + I-4 (compound) — wrap is_compatible, add WARN banner
6. I-6 — runtime invariant check + test cleanup
7. S-10 — Linux QNN entry (verify SO path first)
8. C-4 + I-5 — new tests
9. S-1, S-3, S-7, S-8, S-9 — small mechanical cleanups
10. S-2, S-4, S-5, S-6 — verify-then-act items

Steps 1-7: ~30 min, ~50-line diff. Steps 8: ~30 min, ~80-line test additions. Steps 9-10: ~15 min, ~20-line cleanup.
