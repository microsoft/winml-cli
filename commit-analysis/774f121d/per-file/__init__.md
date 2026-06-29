# src/winml/modelkit/__init__.py

## TL;DR
Three-line top-level package-init change: adds a second side-effecting subpackage import (`from . import _transformers_compat`) immediately after the pre-existing `_warnings` import, and adds an explanatory comment + `# noqa: I001` to lock the import order. The compat module re-injects transformers 4.x symbols that optimum-onnx 0.1.0 still expects. The change is "load-bearing comments + one new import"; everything else in the file is unchanged.

## Diff metrics
- Lines: +4 / -1 (net +3)
- Hunks: 1
- New symbols re-exported: none. New module dependency: `_transformers_compat`.

## Role before vs after
- Before: top-level package init that wired logging, imported `_warnings` for filter configuration before any subpackage import, and exposed `__version__` via `importlib.metadata`. One side-effecting "pre-import" hook (`_warnings`).
- After: same shape, two side-effecting pre-import hooks now. Order matters and is now documented: `_warnings` first (so filters apply to everything else), then `_transformers_compat` (so transformers' `_LazyModule` registry is patched before any `optimum.*` import cascade can see the missing 4.x symbols and ImportError). The `# noqa: I001` suppresses ruff's import-sort lint that would otherwise reorder the two new imports.

## Symbol-level changes
- Removed: bare `from . import _warnings  # Configure warning filters before importing subpackages` (the inline comment).
- Added: a two-line preamble comment explaining the ordering invariant.
- Added: `from . import _warnings  # noqa: I001` (lint suppression added so ruff/isort cannot reorder this against the new import).
- Added: `from . import _transformers_compat`.
- No change to `logging.getLogger(__name__).addHandler(logging.NullHandler())`, the `__version__` resolution block, or any re-export.

## Behavior / contract changes
- Importing `winml.modelkit` (or any `winml.modelkit.*` submodule) now triggers `_transformers_compat` as a side effect — which in turn imports `transformers`, `transformers.modeling_utils`, `transformers.utils`, `transformers.utils.generic`, and conditionally `torch` / `optimum.exporters.onnx.model_patcher`. Import time grows by however long transformers takes to import (typically 1–3 s on cold cache). For very lean callers (e.g. `from winml.modelkit import __version__`) this is now mandatory overhead that wasn't there before.
- No public API addition or removal at this module level.
- Side-effecting state changes documented in `_transformers_compat.md`: `CLIPFeatureExtractor`, `MT5Tokenizer`, `AutoModelForVision2Seq` injected into `transformers._objects`; `is_offline_mode`, `get_parameter_dtype`, `_CAN_RECORD_REGISTRY`, `OutputRecorder` set on `transformers.utils` / `transformers.modeling_utils` / `transformers.utils.generic`; optimum's `sdpa_mask_without_vmap` monkey-patched if optimum is importable.

## Cross-file impact
- `_transformers_compat.py` is a new sibling file. Only consumer is this `__init__.py`.
- Anything that previously did `from winml.modelkit import …` to avoid the heavy transformers stack now pays the import cost regardless.
- Tests that mock `transformers` at module level need to mock *before* `import winml.modelkit`, or the real `transformers` will already be imported by the time the test starts.

## Risks / subtleties
- Ordering is enforceable only by comment + `noqa: I001`. A future autosort tool that ignores the noqa marker could re-shuffle the imports and silently break the compat layer (subsequent optimum import would race and fail). Worth a `tests/unit/test_import_order.py` that asserts `_transformers_compat` is loaded before `optimum` after `import winml.modelkit`.
- Import-time side effects on `transformers` are global to the Python process. Two consumers in the same process (e.g. winml.modelkit + a notebook cell that did `from transformers import CLIPImageProcessor` first) interact: the patch happens at first `import winml.modelkit`, and is durable from then on. Order of `import winml.modelkit` vs `import optimum` matters; the compat layer must come first.
- `_transformers_compat` is silent on the success path — there's no log line marking the patch. Diagnostics for "did the patch apply?" require inspecting `transformers._objects` directly.

## Simplification opportunities
- Replace the `# noqa: I001` + the two-line comment with a `# isort: skip_file` block-comment or move both side-effecting imports into a `_bootstrap.py` and call it via `_bootstrap.run()` — would make the order invariant a function call rather than relying on file order + linter suppression.
- The `_transformers_compat` patch is only needed when `optimum.*` is on the import path. A lazy `_install_compat_if_optimum_present()` helper called from the entry points that actually import optimum would defer the transformers import cost for callers that never touch optimum. Acceptable today because transformers is in pyproject.toml as a hard dep, but worth revisiting if the dep ever becomes optional.

## Open questions / TODOs surfaced
- Should `__version__` resolution be deferred to first access (a `__getattr__` on the package) so even `pip show`-style introspection doesn't trigger the transformers import? Today `importlib.metadata.version("winml-modelkit")` is the slow part, but the transformers import is now the dominant cost.
- No test asserts the side effects of `_transformers_compat` actually applied. Add one that does `import winml.modelkit; from transformers import CLIPFeatureExtractor; assert CLIPFeatureExtractor.__module__.startswith("winml.modelkit")`.
