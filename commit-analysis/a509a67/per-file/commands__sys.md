# src/winml/modelkit/commands/sys.py

## TL;DR
`_gather_ep_info()` is rewritten to be **catalog-driven** rather than **registry-merge-driven**. The previous implementation merged a WinML EP list and an ORT provider list under a flat `ep_device_map: dict[str, str]`, emitting one row per EP. The new implementation walks `EP_DEVICE_SPECS` (the new ordered catalog from `session.ep_device`) and emits one row per `(ep, device)` spec — so an EP that targets multiple devices (e.g. OpenVINO → npu/gpu/cpu) now produces multiple rows in `winml sys`. The flat `get_ep_device_map()` helper from `sysinfo` is dropped from this file's import surface and is no longer consulted.

## Diff metrics
- 41 lines changed (24 insertions / 17 deletions per `--stat`).
- Two hunks: one in the import block, one wholly rewriting the merge tail of `_gather_ep_info`.

## Role before vs after
Role of the **command** unchanged: `winml sys` still surfaces OS / CPU / RAM / accelerator / EP info in either a Rich-printed view or a JSON view.

What changed is the EP-info row schema:
- **Before**: at most one row per EP name, with `device` chosen by `ep_device_map.get(name, "unknown")` — i.e. one canonical device per EP, even for EPs whose catalog has multiple device targets.
- **After**: one row per `(ep, device)` spec entry in `EP_DEVICE_SPECS`, gated on the EP being *installed* (visible via either the WinML registry or ORT's available-providers). Catalog ordering is preserved. EPs that are installed but absent from the catalog appear in a second "UNKNOWN" pass.

## Symbol-level changes
- **Top imports**:
  - removed `get_ep_device_map` from the `..sysinfo` import (the import line becomes just `from ..sysinfo import OS`).
  - added `from ..session import EP_DEVICE_SPECS`.
- **`_gather_ep_info()` (only function touched)**:
  - Existing pre-merge logic (build `winml_eps: dict[str, str]` from `WinMLEPRegistry.get_available_eps()` and `ort_providers: list[str]` from `ort.get_available_providers()`) is unchanged.
  - The merge tail is wholly replaced:
    - Computes `installed_eps = set(winml_eps) | set(ort_providers)` (a flat installed-name set).
    - Computes `seen_pairs: set[tuple[str, str]]` to dedupe `(ep, device)` tuples (the de-dup key is now the *pair*, not the EP name).
    - **Pass 1** iterates `EP_DEVICE_SPECS` in catalog order. For each spec whose `spec.ep` is in `installed_eps`, appends `{"name": spec.ep, "device": spec.device.upper(), "path": winml_eps.get(spec.ep)}`. `path` is `None` when the EP is visible only via ORT (not registered through the WinML EP plugin loader).
    - **Pass 2** sweeps `installed_eps - catalog_eps` (any installed EP whose name is not in the catalog at all — "custom/unknown") and emits one row with `"device": "UNKNOWN"`.
  - Return type unchanged (`list[dict[str, Any]]`).

## Behavior / contract changes
- **Row count**: multi-device catalog entries (OpenVINO is the canonical example called out in the inline comment) now produce multiple rows. A consumer counting EP rows to count *unique EPs* will need to switch to a `set` over `row["name"]`.
- **Device casing**: now always upper-case (`spec.device.upper()`), matching the previous code which also upper-cased. Behavioral parity here.
- **`path` for ORT-only EPs**: previously emitted by a second loop with `path=None` explicitly; now derived from `winml_eps.get(spec.ep)` which yields `None` for the same case. Net: same `path` semantics — `None` when not in the WinML registry.
- **`UNKNOWN` rows**: previously, an unknown-device EP was tagged `"unknown".upper()` ⇒ `"UNKNOWN"`, but only one row per EP. Now the device is `"UNKNOWN"` (string literal, not via `.upper()` round-trip) and the second pass emits one row per *installed-but-uncatalogued* EP. Same render outcome for typical installs.
- **Ordering**: now follows catalog deduction-preference order (which the commit body documents as the canonical ordering for the whole refactor: "Order encodes deduction preference"), with the WinML-first / ORT-second split replaced by catalog-first / unknown-second.

## Cross-file impact
- Removes one consumer of the legacy `get_ep_device_map()`. The commit body says this duplicate was deleted: "_EP_DEVICE_MAP duplicate deleted (catalog is the only source)". `winml sys` was one of the maintenance callers; this is its half of the cleanup.
- Adds `EP_DEVICE_SPECS` to this file's public surface dependency, so `session.__init__` must re-export it. (The directive in the commit body — *"do not import private symbols outside session/ep_device.py — use the session/ facade"* — is honored: this file imports `EP_DEVICE_SPECS` from `..session`, not from `..session.ep_device`.)

## Risks / subtleties
- **Pass-1 short-circuit logic**: `if spec.ep not in installed_eps: continue` will skip a catalog row even if a *different* spec with the same `spec.ep` is later in the catalog. That's fine because the check is per-spec, not per-EP, and only suppresses uninstalled EPs.
- **`seen_pairs` is allocated but only read at the `key in seen_pairs` test, never written to after population by `seen_pairs.add(key)`** — but since the catalog should not contain duplicate `(ep, device)` pairs, the defensive dedupe never fires in practice. Treat as a future-proof guard.
- **Display assumes a UI that handles repeated EP names gracefully**: any per-row formatter that uses the EP name as a key (e.g., a `dict` in `_output_ep_text` or the table-renderer) needs to tolerate duplicates. Looking at `_output_ep_text` (which iterates the list and pads `ep["name"]`), the row-per-pair model is fine — it just renders the same `name` twice with a different `device`. JSON output (`_output_ep_json`) likewise dumps the list as-is.
- **An EP in `ort_providers` but absent from the catalog with no path** falls into Pass 2 ⇒ `device="UNKNOWN", path=None`. Previously it fell into the second loop with the same `path=None` and `device=ep_device_map.get(name, "unknown")` — i.e. potentially also `"unknown"`. So the outcome is the same string but the source of truth has moved.
- **Performance**: O(len(EP_DEVICE_SPECS)) extra walk plus an O(installed - catalog) sweep, both trivial. No concern.

## Open questions / TODOs surfaced
- No TODOs in the diff. The inline doc-comment on the rewrite is explanatory ("one (EP, device) row per `EP_DEVICE_SPECS` entry that is also installed") and self-contained.
- The duplicate-row schema for multi-target EPs might break existing test fixtures that assert "one row per EP" — worth verifying that any `tests/test_sys.py` (or similar) was updated in the same commit (the 41-line stat here suggests test updates landed elsewhere or are already catalog-aware).
- `winml sys` does not yet display the `EPDevice` short string (e.g. `qnn-npu`); only `name` and `device.upper()` separately. If users are expected to consume `winml sys` output to construct CLI args (`--ep qnn --device npu`), a future enhancement could emit a single `qnn-npu`-style identifier per row.
