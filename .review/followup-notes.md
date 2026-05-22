# Branch `feat/update-pkg-deps` — Followup Notes

**Last updated**: 2026-05-08
**Branch state at handoff**: `ec7a9f16` (18 commits ahead of `main`)
**Test status**: 4533 passed, 0 failed, 73 skipped, 3 xfailed (5:23 runtime)
**Lint status**: ruff clean across `src/` and `tests/`
**Live `winml sys --list-ep`**: working on Snapdragon X Elite dev box with DML EP

---

## What this branch ships

A multi-pronged dependency upgrade and EP discovery refactor:

### Dependency upgrades
- Python `3.10` → `3.11` (`>=3.11,<3.12`)
- `onnxruntime` 1.24.x → **`onnxruntime-windowsml` 1.24.5.202604171637** (Microsoft distro, bundles DML EP)
- `onnx` 1.18 → `1.21` (IR_VERSION 13)
- `transformers` 4.x → 5.x (with local `_transformers_compat.py` shim for the 4.x → 5.x optimum-onnx 0.1 hardcoded internals)
- `onnxruntime-qnn` 1.24 → **2.1**, plus new `onnxruntime-ep-openvino` 1.4 plugin
- New optional `[winml-catalog]` extra → `wasdk-microsoft-windows-ai-machinelearning[all]>=2.0.1` + `winrt-Windows.Management.Deployment>=3.2.1`
- `[tool.uv] override-dependencies` neutralizes wasdk's PEP 440-incompatible exact pin and the upstream `onnxruntime` transitive dep from `onnxruntime-qnn`

### Unified EP discovery layer (`src/winml/modelkit/ep_path.py`, ~1230 lines)
- `EpSource` abstract base class with shared `is_compatible()` body, abstract `resolve()` and `iter_eps()`
- 4 concrete frozen-dataclass subclasses:
  - `PyPiSource` — pip-installed plugin EP wheels (resolved via `importlib.metadata.distribution.locate_file`)
  - `FilesystemSource` — directory drop (installer / unzipped GitHub release / custom build), env-var-gated
  - `WinMlCatalogSource` — WinAppSDK `ExecutionProviderCatalog` MSIX-delivered EPs
  - `MsixPackageSource` — NEW: version-pinned MSIX EP via `Windows.Management.Deployment.PackageManager`, bypasses the catalog's "one version per EP-name" limit
- `discover_eps()` walker with `extra_sources` (prepended), `extra_sources_after` (appended for inventory CLI), and `return_shadowed` modes
- `ResolvedEp` dataclass for the shadowed-aware return shape
- `list_msix_eps()` helper — round-trip-pinning enumeration of installed MSIX EP packages
- `_EP_VENDOR_REQUIREMENT` table + `_ep_is_compatible()` for hardware compatibility (substring-match against detected vendors)
- `MODELKIT_EP_PATH` env var (renamed from not-yet-shipped `WINML_EP_PATH`)
- `EP_NAME_ALIASES` allow-list for canonicalizing PascalCase NVIDIA spellings to camelCase

### CLI inventory: `winml sys --list-ep`
- Comprehensive output across every EP source (PyPI / Catalog / MSIX / filesystem / built-in)
- Two-axis status:
  - Entry-level: `[primary]` / `[shadowed]` / `[incompatible]`
  - Section-level: `[incompatible]` (machine has no compatible vendor hardware)
- Vendor-qualified device-types (`Qualcomm NPU`, `Intel NPU/GPU/CPU`)
- `(catalog default)` annotation cross-referencing `WinMlCatalogSource`'s pick
- JSON output with full structured shape

### Tests
- 113 ep_path-specific tests across 7 files (test_compat, test_discover_eps_shadowed, test_msix_package_source, test_register_execution_providers, test_ep_path, test_winml_catalog_source, test_add_ep_for_device)
- 1 Windows-only integration smoke test (`tests/integration/ep_path/test_live_msix.py`)
- CLI end-to-end test exercising real `_gather_ep_info` (mocking only slow boundaries)
- Existing `WinMLEPRegistry` tests preserved under the new infrastructure
- `xfail` for `musicgen` (transformers 5.7 + huggingface_hub 1.12 strict-dataclass regression)

### Documentation
- `docs/ep-path-design.md` — original unified-EP-discovery design (status: implemented in `17b81c9a..eab52093`)
- `docs/ep-path-msix-source.md` — MsixPackageSource extension (status: implemented in `f79e484e..ec7a9f16`)
- `docs/winml-ep-empirical-findings.md` — live-machine evidence corpus (untracked; consider committing)
- `docs/upgrade-deps-and-winml-refactor.md` — dep upgrade rationale (untracked)

### Review pipeline
- 4-agent specialized review (code, comments, errors, tests) → 16 findings
- Opus-level verifier pass → confirmed all 16, refined 2 severities, added 5 deeper findings
- `f19d0a61` fixed all 16 + the 5 verifier additions
- Step-3 verifier confirmed all 16 fixed cleanly
- Step-5 fresh-context audit (Opus, no prior bias) found **2 NEW Critical bugs**:
  - C1: `MODELKIT_EP_PATH` parser dropped Linux `.so` filenames (silent miss)
  - C2: `register_execution_providers(extra_sources=...)` silently no-op'd on second call (cache bypass missing)
- `ec7a9f16` fixed both Critical issues with new test coverage

---

## Pending — Step-5 Important findings (not yet addressed)

The fresh-context audit surfaced 5 Important issues. None are user-visible bugs; all are maintenance-grade concerns:

### I1: `discover_eps()` return-type Union breaks mypy strict mode
**File**: `src/winml/modelkit/ep_path.py` (`discover_eps` signature)
The return type is `dict[str, tuple[Path, EpSource]] | dict[str, list[ResolvedEp]]`. mypy can't narrow without help. Callers like `commands/sys.py:543` use `assert isinstance(full, dict)` which doesn't actually distinguish the two shapes.

**Fix sketch**: Add `@overload` declarations for `return_shadowed: Literal[False]` and `Literal[True]`. Update the assert sites to type-narrow correctly.

### I2: `atexit.register(_release_winml_handle, ...)` accumulates across `cache_clear()` calls
**File**: `src/winml/modelkit/ep_path.py:494-504`
The catalog singleton uses `@functools.cache`. Tests call `_get_catalog.cache_clear()` heavily. Each fresh `initialize()` after a clear registers ANOTHER atexit handler. After N test runs, N handles release at process exit; the 2nd-Nth `__exit__` calls hit a deactivated runtime (suppressed today). Could surface as flaky shutdown messages or wedge under specific conditions.

**Fix sketch**: Track registration with a module-level boolean OR move the `atexit.register` call into a `_atexit_registered` guard. Alternatively, use the more modern `weakref.finalize` pattern.

### I3: `_winml_catalog_warned_keys` is a module global with inconsistent test-side reset
**File**: `src/winml/modelkit/ep_path.py:421`
Reset in `tests/unit/ep_path/test_winml_catalog_source.py`'s fixture but NOT in `test_ep_path.py`. Order-dependent: a test that emits a WARN populates the set, downstream tests asserting "warning present" see DEBUG instead.

**Fix sketch**: Either fold the warn-once cache into a `_get_catalog`-bound attribute (so `cache_clear()` resets it too), or add a fixture in `tests/unit/ep_path/conftest.py` that resets it for every test.

### I4: `FilesystemSource(root=Path(), env_var=...)` is a CWD footgun if env_var is dropped
**File**: `src/winml/modelkit/ep_path.py:1001-1009` (the `NVIDIA_TRT_RTX_EP` row)
`Path()` resolves to `Path('.')` (CWD). Today the `env_var` gate prevents this from ever firing because if `NVIDIA_TRT_RTX_EP` is unset the source short-circuits. But if a future maintainer drops `env_var`, the source becomes "scan the user's CWD for nv_tensorrt_rtx.dll" — fragile.

**Fix sketch**: Add a `__post_init__` validator: if `env_var is None and not root.is_absolute()`, raise `ValueError`.

### I5: `_FakePackage.id.full_name` doesn't match real WinRT package full-name format
**File**: `tests/unit/ep_path/test_msix_package_source.py:43-51`
The fake's `full_name` is hand-constructed; doesn't match the real `<name>_<ver>_<arch>__<publisher>` format precisely. `list_msix_eps` only uses it in a debug log today, so tests pass. But a future change that parses `full_name` would pass against the fake and break in production.

**Fix sketch**: Either drop `full_name` from the fake, or make the `_FakePackageId` constructor build the correct format from its components.

---

## Pending — Suggestions from Step-5 (lower priority)

- `_get_pkg_manager` and `_get_catalog` use `@functools.cache`; cache_clear has no public wrapper. Document the contract.
- `commands/sys.py` reaches into private `_EP_VENDOR_REQUIREMENT` from `ep_path.py`. Layering smell. Consider exposing a public accessor.
- `"primary"` / `"shadowed"` / `"incompatible"` string literals appear in ~5 places. Centralize as constants on `ResolvedEp` or a `Status` enum.
- `requires-python = ">=3.11,<3.12"` hard ceiling at 3.12 means a 3.13 user sees an install failure rather than a deprecation. Either widen the ceiling or document.

---

## Deferred decisions (intentional, documented)

### S-2: `_legacy_ep_plugin_registry` shim retention
**File**: `src/winml/modelkit/winml.py:49-62`
No internal callers (verified by grep). Kept because the docstring says "will be removed once no internal callers remain" — implying external scripts may import `EP_PLUGIN_REGISTRY`. Per CLAUDE.md "don't refactor beyond what task requires", we left it in place. Delete in a follow-up branch after verifying external usage.

### S-10: Linux QNN PyPiSource entry not added to `_default_ep_path_linux()`
**File**: `src/winml/modelkit/ep_path.py:_default_ep_path_linux`
`onnxruntime-qnn` 2.1.0 publishes `manylinux_2_34_aarch64` wheels for cp311+. Adding a `PyPiSource` would require knowing the internal `.so` layout (likely `onnxruntime_qnn/libs/aarch64/libonnxruntime_providers_qnn.so` but unverified). Documented as TODO in the function docstring. **To resolve**: install the cp311 aarch64 wheel on a Linux aarch64 box, inspect via `python -c "import importlib.metadata as md; [print(p) for p in md.distribution('onnxruntime-qnn').files]"`, then add the entry mirroring the Windows row.

### Untracked docs not committed
The branch's working tree has these untracked design / research docs that were not part of this PR's scope but may be useful:
- `docs/ep-sideloading-research.md`
- `docs/winml-ep-empirical-findings.md`
- `docs/upgrade-deps-and-winml-refactor.md`
- `docs/final-verdict-zhengte-update-pkg-deps.md`
- `pr_334_inline_comments.md`, `pr_334_prd.md`, `pr_334_verdicts.md` (in repo root)

Pickup decision: review and commit any that are intended PR deliverables; delete the rest.

---

## Pickup checklist for the next session

1. **Push to remote**: `git push -u origin feat/update-pkg-deps` (already done at handoff)
2. **Open PR**: `gh pr create --title "feat: upgrade deps + unified EP_PATH discovery + MsixPackageSource"`
   - Body: describe the dep upgrade, the EP_PATH unification, the new MSIX inventory CLI. Include the live `--list-ep` output as evidence.
3. **Address Step-5 Important findings I1-I5** if doing a polish pass before merge.
4. **Review and commit untracked design docs** (or delete them).
5. **Verify Linux QNN entry (S-10)** if Linux aarch64 support is in-scope.
6. **Watch for CI flakes** related to I2 (atexit accumulation) or I3 (warn-once cache pollution).
7. **musicgen xfail**: monitor `transformers` releases — the upstream fix would make this `XPASS`. Remove the marker when that lands.

---

## Tracking files

- `.review/pending-issues.md` — original 16-item review tracking (all DONE) + 5 verifier additions (all merged into commits)
- `.review/followup-notes.md` — this file

Both are committed alongside this handoff.
